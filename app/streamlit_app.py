"""
RetainAI dashboard.

Run from the repo root:
    streamlit run app/streamlit_app.py --server.port 8501
"""
import json
import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from agent import AGENT_MODEL, CHALLENGER_MODEL, STRONG_MODEL, Agent, Toolbox, judge, validate  # noqa: E402


def art(name):
    return os.path.join(ROOT, "artifacts", name)


def load_json(name):
    path = art(name)
    return json.load(open(path)) if os.path.exists(path) else None


st.set_page_config(page_title="RetainAI", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def load_tools():
    return Toolbox()


tools = load_tools()
customers = tools.customers
active = customers[customers["Churn"] == 0]
segment_stats = customers.groupby("segment").agg(seg_churn=("Churn", "mean"), seg_monthly=("MonthlyCharges", "mean"))


@st.cache_data(ttl=900, show_spinner="Scoring on the SageMaker endpoint...")
def live_score(cid):
    return tools.score_churn_risk(cid)


PLAN_LABELS = {
    "BILLING_REVIEW": "🧾 Billing review", "ESCALATE_TO_SPECIALIST": "🧑‍💼 Escalate to a specialist",
    "ADDRESS_CONCERN": "💬 Address the concern", "ANSWER_QUESTION": "❓ Answer the question",
    "PROACTIVE_CHECK_IN": "👋 Proactive check-in",
}

st.title("🛡️ RetainAI")
st.caption("Enter a customer → see their history → paste their message → get the right action and offer → AI writes the reply")

tab_console, tab_portfolio, tab_results = st.tabs(["Customer console", "Portfolio overview", "Model & evaluation results"])

# ============================================================================ customer console
with tab_console:
    top_ids = active.sort_values("revenue_at_risk_12m", ascending=False).index[:50].tolist()
    c_in, c_msg = st.columns([1, 2])
    with c_in:
        cid = st.text_input("Customer ID", value=top_ids[0], key="cid").strip()
        with st.expander("Need an ID? Top revenue-at-risk customers"):
            st.dataframe(active.loc[top_ids, ["Contract", "tenure", "MonthlyCharges", "churn_prob", "revenue_at_risk_12m"]]
                         .round(2), width="stretch", height=250)
    with c_msg:
        message = st.text_area("Customer message (leave empty for proactive outreach)",
                               "My bill looks wrong this month.", height=110, key="message").strip()

    if cid not in customers.index:
        st.error(f"Customer {cid} not found. Pick one from the list above.")
        st.stop()

    r = customers.loc[cid]
    score = live_score(cid)
    seg = segment_stats.loc[r["segment"]]
    rec = tools.recommend_offer(cid, message)
    plan = rec["recommended_plan"]

    # ---------------------------------------------------------------- history & metrics
    st.subheader("Customer history & metrics")
    m = st.columns(6)
    m[0].metric("Customer for", f"{int(r['tenure'])} months")
    m[1].metric("Lifetime revenue", f"${float(r['TotalCharges']):,.0f}")
    bill_change = (r["MonthlyCharges"] - r["avg_monthly_spend"]) / max(r["avg_monthly_spend"], 1)
    m[2].metric("Current bill", f"${r['MonthlyCharges']:.2f}", f"{bill_change:+.0%} vs their average",
                delta_color="inverse")
    m[3].metric("Churn probability", f"{score['churn_probability']:.0%}",
                f"{score['churn_probability'] - seg['seg_churn']:+.0%} vs segment", delta_color="inverse")
    m[4].metric("Revenue at risk (12m)", f"${score['revenue_at_risk_12m']:,.0f}")
    savings = next((o for o in rec["eligible_offers"] if o["code"] == plan["offer_code"] and "discount_pct" in o), None)
    m[5].metric("Savings with offer", f"${r['MonthlyCharges'] * savings['discount_pct'] / 100:.2f}/mo" if savings else "—",
                plan["offer_code"].replace("_", " ").title() if plan["offer_code"] != "NO_OFFER" else None, delta_color="off")

    d1, d2 = st.columns([3, 2])
    with d1:
        active_addons = [a for a in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                                     "StreamingTV", "StreamingMovies"] if r[a] == "Yes"]
        st.markdown(
            f"**Plan:** {r['Contract']} · {r['InternetService']} internet · {r['PaymentMethod']}"
            f"{' · paperless' if r['PaperlessBilling'] == 'Yes' else ''}  \n"
            f"**Add-ons:** {', '.join(active_addons) or 'none'}  \n"
            f"**Segment {int(r['segment'])}:** {seg['seg_churn']:.0%} churn rate in this segment  \n"
            f"**Why at risk (SHAP):** {' · '.join(score['top_risk_drivers'])}"
            + ("  \n⚠️ **Billing anomaly flagged** on this account" if r["billing_anomaly"] else ""))
        st.caption(f"Scored live by `{score['scored_by']}`")
    with d2:
        st.bar_chart(pd.Series({"Their average bill": r["avg_monthly_spend"], "Current bill": r["MonthlyCharges"],
                                "Segment average": seg["seg_monthly"]}, name="USD per month"), height=180, horizontal=True)

    # ---------------------------------------------------------------- understanding + plan
    st.divider()
    p1, p2 = st.columns([2, 3])
    with p1:
        st.subheader("What the customer wants")
        if rec["message_intent"]:
            mi = rec["message_intent"]
            flags = [k.replace("_", " ") for k, v in mi["signals"].items() if v]
            st.markdown(f"**Situation:** {mi['intent_group']}  \n**Signals:** {', '.join(flags) or 'none'}  \n"
                        f"**Intent model:** {mi['intent']} ({mi['confidence']:.0%} confidence)")
        else:
            st.markdown("No message: **proactive outreach**")
    with p2:
        st.subheader("Recommended plan")
        a, b, c = st.columns(3)
        a.metric("1 · Resolve", PLAN_LABELS[plan["resolution"]].split(" ", 1)[1])
        b.metric("2 · Offer", plan["offer_code"].replace("_", " ").title())
        c.metric("Escalate", "Yes" if plan["escalate_to_human"] else "No")
        st.caption(f"Why: {plan['why']}. Offer: {plan['offer_description'] or 'none'}"
                   + (f" — {plan['offer_benefit']}" if plan["offer_benefit"] else ""))

    with st.expander("Policy details: allowed offers, rules, learned offer values"):
        e1, e2 = st.columns(2)
        e1.dataframe(pd.DataFrame(rec["eligible_offers"]).fillna("").astype(str), hide_index=True, width="stretch")
        for rule in rec["rules_applied"]:
            e1.caption(f"• {rule}")
        if rec["bandit_values_usd"]:
            e2.bar_chart(pd.Series(rec["bandit_values_usd"], name="learned value (USD)"))
            e2.caption("Learned by the Block 5 bandit: estimated 12-month net revenue per offer")

    # ---------------------------------------------------------------- AI reply
    st.divider()
    st.subheader("AI-written reply")
    model_id = st.selectbox("Agent model (Amazon Bedrock)", [AGENT_MODEL, STRONG_MODEL, CHALLENGER_MODEL], key="model")
    if st.button("Write reply", type="primary", key="run_agent"):
        with st.spinner("Agent is calling tools and writing..."):
            result = Agent(tools, model_id=model_id).run(cid, message or None)
            checks = validate(result, tools)
        out = result["output"] or {}
        if checks["compliant"]:
            st.success(out.get("customer_message", ""))
        else:
            st.error("Blocked by guardrails, not sent. Failed: "
                     + ", ".join(k for k, v in checks.items() if not v and k != "compliant"))
            st.text(out.get("customer_message", result["raw_text"]))

        with st.spinner("Scoring reply quality with the judge model..."):
            scores = judge(result, tools) if result["output"] else {}
        q = st.columns(5)
        q[0].metric("Guardrails", "✅ Pass" if checks["compliant"] else "❌ Fail")
        for col, k in zip(q[1:], ["personalization", "empathy", "clarity", "appropriateness"]):
            col.metric(k.title(), f"{scores.get(k, '–')}/5")
        if scores.get("comment"):
            st.caption(f"Judge: {scores['comment']}")

        with st.expander("Agent trace and checks"):
            for step in result["trace"]:
                st.markdown(f"🔧 **{step['tool']}** `{json.dumps(step['input'])}`")
                st.json(step["output"], expanded=False)
            st.caption(f"{result['turns']} turns · {result['latency_s']}s · tokens {result['usage']}")
            st.write(checks)

# ============================================================================ portfolio
with tab_portfolio:
    at_risk = active[active["churn_prob"] >= 0.3]
    k = st.columns(4)
    k[0].metric("Active customers", f"{len(active):,}")
    k[1].metric("At risk (≥30%)", f"{len(at_risk):,}")
    k[2].metric("12-month revenue at risk", f"${active['revenue_at_risk_12m'].sum():,.0f}")
    k[3].metric("Billing anomalies flagged", f"{int(active['billing_anomaly'].sum()):,}")
    st.subheader("Segments")
    st.dataframe(customers.groupby("segment").agg(
        customers=("Churn", "size"), churn_rate=("Churn", "mean"), avg_tenure=("tenure", "mean"),
        avg_monthly=("MonthlyCharges", "mean"), revenue_at_risk=("revenue_at_risk_12m", "sum")).round(2), width="stretch")
    st.subheader("Revenue at risk by contract type")
    st.bar_chart(active.groupby("Contract")["revenue_at_risk_12m"].sum())

# ============================================================================ results
with tab_results:
    for title, name in [("SHAP: what drives churn", "shap_summary.png"),
                        ("Semi-supervised learning with scarce labels", "semi_supervised_curve.png"),
                        ("Bandit learning curves", "bandit_learning_curves.png")]:
        if os.path.exists(art(name)):
            st.subheader(title)
            st.image(art(name), width="stretch")
    if comparison := load_json("agent_comparison.json"):
        st.subheader("Agent evaluation: model comparison")
        st.dataframe(pd.DataFrame(comparison), hide_index=True, width="stretch")
    if summary := load_json("bandit_summary.json"):
        st.subheader("Offer optimization (simulation)")
        st.json({k: v for k, v in summary.items() if k != "assumptions"})
