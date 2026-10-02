"""Guardrail and plan-engine tests: the validator must block every kind of policy violation."""

import pytest

from agent import message_signals, parse_json, risk_band, validate

OFFERS = [{"code": "NO_OFFER", "description": "No incentive"},
          {"code": "BILLING_REVIEW", "credit_up_to_usd": 25, "description": "Bill review"},
          {"code": "CONTRACT_UPGRADE", "discount_pct": 10, "description": "1-year contract, 10% off"}]


class StubToolbox:
    """Returns a fixed policy decision so validator logic is tested in isolation."""
    def __init__(self, offer="CONTRACT_UPGRADE", escalate=False, resolution="BILLING_REVIEW"):
        self.plan = {"resolution": resolution, "offer_code": offer, "escalate_to_human": escalate}

    def recommend_offer(self, customer_id, customer_message=""):
        return {"eligible_offers": OFFERS, "recommended_plan": self.plan}


def good_result():
    return {
        "customer_id": "C1", "message": "My bill looks wrong this month.",
        "output": {"offer_code": "CONTRACT_UPGRADE", "escalate_to_human": False, "resolution": "BILLING_REVIEW",
                   "customer_message": "Sorry about that. Our billing team will review this month's charges. "
                                       "If you like, you could also switch to a 1-year contract with 10% off."},
        "trace": [{"tool": "get_customer_profile", "input": {}, "output": {}},
                  {"tool": "score_churn_risk", "input": {}, "output": {}},
                  {"tool": "recommend_offer", "input": {"customer_id": "C1", "customer_message": "My bill looks wrong"},
                   "output": {"eligible_offers": OFFERS}}],
    }


def failed(result, toolbox=None):
    return {k for k, v in validate(result, toolbox or StubToolbox()).items() if not v}


def test_compliant_response_passes():
    assert failed(good_result()) == set()


@pytest.mark.parametrize("change, expected", [
    ({"offer_code": "FREE_PHONE"}, "offer_is_eligible"),
    ({"offer_code": "NO_OFFER"}, "follows_plan"),
    ({"customer_message": "Sorry! Our billing team will review it. Enjoy 30% off for 6 months!"}, "discount_within_policy"),
    ({"customer_message": "Our model says your churn probability is high; billing will review it."}, "no_internal_terms"),
    ({"customer_message": "Billing will review it. Switch to 1-year at 10% off and lock in your rate!"}, "no_invented_perks"),
    ({"customer_message": "Thanks! You could switch to a 1-year contract with 10% off."}, "resolution_mentioned"),
    ({"customer_message": "Our billing team will review this. " + "word " * 120}, "under_120_words"),
])
def test_violations_are_blocked(change, expected):
    r = good_result()
    r["output"].update(change)
    assert expected in failed(r) and "compliant" in failed(r)


def test_missed_escalation_is_blocked():
    r = good_result()
    assert {"escalation_correct", "follows_plan"} <= failed(r, StubToolbox(escalate=True))


def test_skipping_policy_engine_is_blocked():
    r = good_result()
    r["trace"] = [t for t in r["trace"] if t["tool"] != "recommend_offer"]
    assert "called_required_tools" in failed(r)


def test_not_passing_the_message_to_policy_engine_is_blocked():
    r = good_result()
    r["trace"][2]["input"]["customer_message"] = ""
    assert "called_required_tools" in failed(r)


@pytest.mark.parametrize("text, group", [
    ("My bill looks wrong this month.", "billing"),
    ("You charged me twice this month!", "billing"),
    ("Third outage this month. I want to close my account.", "leaving"),
    ("Thinking about switching providers.", "leaving"),
    ("This service is terrible, nobody has helped me.", "complaint"),
    ("Where can I find my latest invoice?", "general"),
    ("How do I update my payment method?", "general"),
])
def test_message_signals(text, group):
    assert message_signals(text, intent="unknown")[0] == group


def test_low_confidence_intent_is_ignored():
    assert message_signals("How do I update my payment method?", "payment_issue", confidence=0.18)[0] == "general"
    assert message_signals("How do I update my payment method?", "payment_issue", confidence=0.95)[0] == "billing"


@pytest.mark.parametrize("text, offer", [
    ('{"offer_code": "NO_OFFER"}', "NO_OFFER"),
    ('<thinking>use {braces}</thinking> Answer: {"offer_code": "ADDON_TRIAL"}', "ADDON_TRIAL"),
    ('```json\n{"offer_code": "LOYALTY_DISCOUNT"}\n```', "LOYALTY_DISCOUNT"),
])
def test_parse_json_handles_model_output_styles(text, offer):
    assert parse_json(text)["offer_code"] == offer


def test_parse_json_returns_none_for_garbage():
    assert parse_json("no json here") is None


def test_risk_band_thresholds():
    assert [risk_band(p) for p in (0.1, 0.3, 0.59, 0.6)] == ["low", "medium", "medium", "high"]
