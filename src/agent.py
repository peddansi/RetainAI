"""
RetainAI agent: tools over the Block 1–3 models, a tool-using agent loop on Amazon Bedrock
(Converse API), policy guardrails enforced in code, and an LLM-as-judge evaluator.

Design principle: the LLM decides *what to say*; code decides *what is allowed*.
Offer eligibility and discount caps are computed by `recommend_offer`, never by the model.
"""
import json
import os
import re
import time

import boto3
import joblib
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from features import RAW_FIELDS, encode_record

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Basic models that need no extra access forms. Swap any of these for another Bedrock model ID.
AGENT_MODEL = "us.amazon.nova-2-lite-v1:0"              # low-cost agent
CHALLENGER_MODEL = "us.amazon.nova-micro-v1:0"          # smallest, cheapest Nova model
JUDGE_MODEL = "us.meta.llama3-3-70b-instruct-v1:0"      # different model family, to avoid self-preference bias

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
ESCALATION_INTENTS = {"delete_account", "complaint"}
INTERNAL_TERMS = ["churn", "probability", "risk score", "model", "segment", "revenue at risk", "algorithm", "shap"]


def _path(*parts):
    return os.path.join(ROOT, *parts)


def risk_band(p):
    return "high" if p >= 0.6 else "medium" if p >= 0.3 else "low"


# --------------------------------------------------------------------------- tools
class Toolbox:
    """Wraps every model built in Blocks 1–3 as a tool the agent can call."""

    def __init__(self, sm_runtime=None):
        import shap  # imported here so the module loads quickly

        self.customers = pd.read_csv(_path("artifacts", "customers_scored.csv")).set_index("customerID")
        local = joblib.load(_path("artifacts", "churn_model.joblib"))
        self.local_model, self.local_cols = local["model"], local["columns"]
        self.explainer = shap.TreeExplainer(self.local_model)
        dep = json.load(open(_path("artifacts", "deployment.json")))
        self.endpoint, self.endpoint_cols = dep["endpoint"], dep["columns"]
        intent = joblib.load(_path("artifacts", "intent_model.joblib"))
        self.intent_clf, self.churn_intents = intent["pipeline"], set(intent["churn_signals"])
        self.sm = sm_runtime or boto3.client("sagemaker-runtime")
        self._scores = {}

        # RAG index over the retention policy (one chunk per section)
        text = open(_path("docs", "retention_policy.md")).read()
        self.policy_chunks = [c.strip() for c in re.split(r"\n(?=## )", text) if c.strip()]
        self.policy_vec = TfidfVectorizer(stop_words="english").fit(self.policy_chunks)
        self.policy_matrix = self.policy_vec.transform(self.policy_chunks)

    def _row(self, customer_id):
        if customer_id not in self.customers.index:
            raise KeyError(f"customer {customer_id} not found")
        return self.customers.loc[customer_id]

    # -- tool 1
    def get_customer_profile(self, customer_id):
        r = self._row(customer_id)
        return {
            "customer_id": customer_id,
            "tenure_months": int(r["tenure"]),
            "contract": r["Contract"],
            "monthly_charges": float(r["MonthlyCharges"]),
            "internet_service": r["InternetService"],
            "payment_method": r["PaymentMethod"],
            "active_add_ons": [a for a in ADDONS if r[a] == "Yes"],
            "segment": int(r["segment"]),
            "billing_anomaly_flag": bool(r["billing_anomaly"]),
        }

    # -- tool 2: live score from the production serverless endpoint + SHAP reasons
    def score_churn_risk(self, customer_id):
        r = self._row(customer_id)
        raw = {k: r[k] for k in RAW_FIELDS}
        payload = ",".join(repr(float(v)) for v in encode_record(raw, self.endpoint_cols))
        resp = self.sm.invoke_endpoint(EndpointName=self.endpoint, ContentType="text/csv", Body=payload)
        prob = float(resp["Body"].read().decode().replace("\n", ",").strip(",").split(",")[0])

        local_row = pd.DataFrame([encode_record(raw, self.local_cols)], columns=self.local_cols)
        contrib = pd.Series(self.explainer.shap_values(local_row)[0], index=self.local_cols)
        drivers = []
        for feat in contrib.sort_values(ascending=False).index[:3]:
            col = feat.split("_", 1)[0] if feat.split("_", 1)[0] in r.index else feat
            value = r[col] if col in r.index else round(float(local_row[feat].iloc[0]), 2)
            drivers.append(f"{col} = {value}")

        remaining = float(r["expected_remaining_months"]) or 12.0
        result = {
            "churn_probability": round(prob, 4),
            "risk_band": risk_band(prob),
            "revenue_at_risk_12m": round(prob * float(r["MonthlyCharges"]) * min(remaining, 12), 2),
            "top_risk_drivers": drivers,
            "scored_by": self.endpoint,
        }
        self._scores[customer_id] = result
        return result

    # -- tool 3
    def classify_message(self, text):
        proba = self.intent_clf.predict_proba([text])[0]
        intent = self.intent_clf.classes_[int(np.argmax(proba))]
        return {"intent": intent, "confidence": round(float(proba.max()), 3),
                "is_churn_signal": intent in self.churn_intents}

    # -- tool 4: policy enforced in code
    def recommend_offer(self, customer_id):
        r = self._row(customer_id)
        score = self._scores.get(customer_id) or self.score_churn_risk(customer_id)
        p, rar, tenure = score["churn_probability"], score["revenue_at_risk_12m"], int(r["tenure"])
        rules, offers = [], [{"code": "NO_OFFER", "description": "Friendly check-in, no incentive"}]

        if p < 0.3:
            rules.append("Low risk (<0.30): NO_OFFER only")
            return {"eligible_offers": offers, "max_discount_pct": 0, "rules_applied": rules}

        addon = r.get("recommended_addon")
        if r["InternetService"] != "No" and isinstance(addon, str) and r[addon] == "No":
            offers.append({"code": "ADDON_TRIAL", "add_on": addon, "months_free": 3,
                           "description": f"3 months of {addon} free"})
            rules.append(f"ADDON_TRIAL eligible: recommended add-on {addon} not active")
        if r["Contract"] == "Month-to-month":
            offers.append({"code": "CONTRACT_UPGRADE", "discount_pct": 10, "months": 12,
                           "description": "Switch to a 1-year contract with 10% off monthly charge"})
            rules.append("CONTRACT_UPGRADE eligible: month-to-month contract")
        cap = 20 if (rar >= 900 and tenure >= 12) else 15 if rar >= 500 else 10
        offers.append({"code": "LOYALTY_DISCOUNT", "discount_pct": cap, "months": 6,
                       "description": f"{cap}% off monthly charge for 6 months"})
        rules.append(f"LOYALTY_DISCOUNT capped at {cap}% (revenue at risk ${rar:.0f}, tenure {tenure} months)")
        if bool(r["billing_anomaly"]):
            offers.append({"code": "BILLING_REVIEW", "credit_up_to_usd": 25,
                           "description": "Free bill review with up to $25 credit if an error is found"})
            rules.append("BILLING_REVIEW eligible: billing anomaly flagged")
        return {"eligible_offers": offers, "max_discount_pct": cap, "rules_applied": rules}

    # -- tool 5: retrieval over the policy document
    def search_policy(self, query, k=2):
        sims = cosine_similarity(self.policy_vec.transform([query]), self.policy_matrix)[0]
        return {"results": [self.policy_chunks[i] for i in np.argsort(-sims)[:k]]}

    def call(self, name, args):
        try:
            return getattr(self, name)(**args)
        except Exception as e:  # tool errors go back to the model instead of crashing the loop
            return {"error": f"{type(e).__name__}: {e}"}


def _tool(name, description, properties, required):
    return {"toolSpec": {"name": name, "description": description,
                         "inputSchema": {"json": {"type": "object", "properties": properties, "required": required}}}}


CID = {"customer_id": {"type": "string", "description": "Customer ID, e.g. 7590-VHVEG"}}
TOOL_CONFIG = {"tools": [
    _tool("get_customer_profile", "Account details: tenure, contract, charges, services, segment, billing flag.", CID, ["customer_id"]),
    _tool("score_churn_risk", "Live churn probability from the production model, risk band, revenue at risk, and top risk drivers.", CID, ["customer_id"]),
    _tool("classify_message", "Classify a customer's message into a support intent and flag churn signals.",
          {"text": {"type": "string"}}, ["text"]),
    _tool("recommend_offer", "The ONLY retention offers this customer is eligible for, with discount caps. Must be called before choosing an offer.", CID, ["customer_id"]),
    _tool("search_policy", "Search the retention policy for rules on offers, discounts, escalation and messaging.",
          {"query": {"type": "string"}}, ["query"]),
]}

SYSTEM_PROMPT = """You are RetainAI, a customer-retention assistant for a telecom company.
Given a customer ID and, optionally, a message from the customer:
1. Call get_customer_profile and score_churn_risk.
2. If there is a customer message, call classify_message on it.
3. Call recommend_offer. You may ONLY choose an offer from its eligible_offers. Use search_policy if unsure about a rule.
4. Choose at most one offer (or NO_OFFER) that best fits the customer's situation and message.
5. Write a warm, specific message to the customer, under 120 words. Acknowledge their issue first.
   Never mention churn, risk, probabilities, models, segments, or internal data. State offers exactly as defined.
Set escalate_to_human to true if the customer asks to close their account or complains AND risk_band is high,
or if they mention legal action, regulators or safety.
Your final answer must be ONLY a JSON object with these keys:
{"risk_level": "high|medium|low", "key_reasons": ["short internal notes"], "offer_code": "...",
 "escalate_to_human": true/false, "customer_message": "..."}"""


# --------------------------------------------------------------------------- agent loop
def converse(client, retries=6, **kwargs):
    """Bedrock Converse with exponential backoff on throttling."""
    for attempt in range(retries):
        try:
            return client.converse(**kwargs)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException") and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def parse_json(text):
    """Extract the final JSON object from a model reply. Handles <thinking> tags, code fences and extra prose."""
    text = re.sub(r"<thinking>.*?</thinking>", "", text or "", flags=re.S)
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    for block in reversed(fenced):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass
    decoder, found = json.JSONDecoder(), []
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    found.append(obj)
            except json.JSONDecodeError:
                continue
    return max(found, key=lambda d: len(json.dumps(d))) if found else None


class Agent:
    def __init__(self, toolbox, model_id=AGENT_MODEL, bedrock_runtime=None, max_turns=8):
        self.tools, self.model_id, self.max_turns = toolbox, model_id, max_turns
        self.rt = bedrock_runtime or boto3.client("bedrock-runtime")

    def run(self, customer_id, message=None):
        task = f"Customer ID: {customer_id}\n" + (f'Customer message: "{message}"' if message
                                                  else "No inbound message: this is proactive outreach.")
        messages = [{"role": "user", "content": [{"text": task}]}]
        trace, usage, t0 = [], {"inputTokens": 0, "outputTokens": 0}, time.time()

        for turn in range(1, self.max_turns + 1):
            resp = converse(self.rt, modelId=self.model_id, system=[{"text": SYSTEM_PROMPT}], messages=messages,
                            toolConfig=TOOL_CONFIG, inferenceConfig={"maxTokens": 1500, "temperature": 0.2})
            for k in usage:
                usage[k] += resp.get("usage", {}).get(k, 0)
            msg = resp["output"]["message"]
            messages.append(msg)

            tool_uses = [b["toolUse"] for b in msg["content"] if "toolUse" in b]
            if resp.get("stopReason") == "tool_use" and tool_uses:
                results = []
                for tu in tool_uses:
                    out = self.tools.call(tu["name"], tu.get("input") or {})
                    trace.append({"tool": tu["name"], "input": tu.get("input"), "output": out})
                    results.append({"toolResult": {"toolUseId": tu["toolUseId"],
                                                   "content": [{"text": json.dumps(out, default=str)}]}})
                messages.append({"role": "user", "content": results})
                continue

            text = "".join(b.get("text", "") for b in msg["content"])
            return {"customer_id": customer_id, "message": message, "model": self.model_id,
                    "output": parse_json(text), "raw_text": text, "trace": trace, "usage": usage,
                    "turns": turn, "latency_s": round(time.time() - t0, 2)}

        return {"customer_id": customer_id, "message": message, "model": self.model_id, "output": None,
                "raw_text": "max turns reached", "trace": trace, "usage": usage, "turns": self.max_turns,
                "latency_s": round(time.time() - t0, 2)}


# --------------------------------------------------------------------------- guardrails
def validate(result, toolbox):
    """Programmatic policy checks. Every check must pass for a response to be compliant."""
    out, trace = result["output"] or {}, result["trace"]
    called = {t["tool"] for t in trace}
    offers = next((t["output"].get("eligible_offers", []) for t in trace
                   if t["tool"] == "recommend_offer" and "error" not in t["output"]), [])
    chosen = next((o for o in offers if o["code"] == out.get("offer_code")), None)
    msg = str(out.get("customer_message", ""))

    required = {"get_customer_profile", "score_churn_risk", "recommend_offer"} | ({"classify_message"} if result["message"] else set())
    allowed_pcts = {chosen["discount_pct"]} if chosen and "discount_pct" in chosen else set()
    mentioned_pcts = {int(x) for x in re.findall(r"(\d+)\s?%", msg)}

    score = toolbox._scores.get(result["customer_id"]) or toolbox.score_churn_risk(result["customer_id"])
    intent = toolbox.classify_message(result["message"])["intent"] if result["message"] else None
    must_escalate = intent in ESCALATION_INTENTS and score["risk_band"] == "high"

    checks = {
        "valid_json": bool(result["output"]) and {"offer_code", "customer_message", "escalate_to_human"} <= set(out),
        "called_required_tools": required <= called,
        "offer_is_eligible": chosen is not None,
        "discount_within_policy": mentioned_pcts <= allowed_pcts,
        "no_internal_terms": not any(term in msg.lower() for term in INTERNAL_TERMS),
        "escalation_correct": (not must_escalate) or out.get("escalate_to_human") is True,
        "under_120_words": len(msg.split()) <= 120,
    }
    checks["compliant"] = all(checks.values())
    return checks


# --------------------------------------------------------------------------- LLM judge
JUDGE_PROMPT = """You are grading a customer-retention message written by an AI assistant for a telecom company.

Customer context: {context}
Customer's inbound message: {message}
Offer chosen: {offer}
Message sent to customer:
\"\"\"{reply}\"\"\"

Score each criterion from 1 (poor) to 5 (excellent):
- personalization: specific to this customer's situation, not generic
- empathy: acknowledges the customer's feelings or problem appropriately
- clarity: easy to understand, states the offer and next step clearly
- appropriateness: offer and tone fit the situation (e.g. not pushing a discount on a simple question)
Respond with ONLY JSON: {{"personalization": n, "empathy": n, "clarity": n, "appropriateness": n, "comment": "one sentence"}}"""


def judge(result, toolbox, model_id=JUDGE_MODEL, bedrock_runtime=None):
    rt = bedrock_runtime or boto3.client("bedrock-runtime")
    out = result["output"] or {}
    prompt = JUDGE_PROMPT.format(
        context=json.dumps(toolbox.get_customer_profile(result["customer_id"])),
        message=result["message"] or "(none - proactive outreach)",
        offer=out.get("offer_code", "none"), reply=out.get("customer_message", ""))
    resp = converse(rt, modelId=model_id, messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={"maxTokens": 400, "temperature": 0})
    scores = parse_json("".join(b.get("text", "") for b in resp["output"]["message"]["content"])) or {}
    keys = ["personalization", "empathy", "clarity", "appropriateness"]
    scores["judge_avg"] = round(float(np.mean([scores.get(k, np.nan) for k in keys])), 2) if all(k in scores for k in keys) else None
    return scores
