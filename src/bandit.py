"""
RetainAI offer optimization with contextual bandits.

A simulator estimates how each customer would respond to each retention offer, using what-if predictions
from the Block 1 churn model plus explicit, documented business assumptions. Bandit policies then learn,
by trial and error, which offer maximizes net revenue for which customer, under the same policy engine
(eligibility + discount caps) the agent uses.
"""
import numpy as np
import pandas as pd

from features import build_frame

ARMS = ["NO_OFFER", "ADDON_TRIAL", "CONTRACT_UPGRADE", "LOYALTY_DISCOUNT"]

ASSUMPTIONS = {
    "horizon_months": 12,            # reward = net revenue over the next 12 months
    "causal_shrinkage": 0.5,         # model what-ifs are correlational; assume only half the effect is causal
    "accept_addon_trial": 0.6,       # share of customers who activate a free add-on trial
    "accept_contract_new": 0.25,     # share accepting a 1-year contract (tenure < 12 months)
    "accept_contract_tenured": 0.45, # share accepting a 1-year contract (tenure >= 12 months)
    "accept_discount": 1.0,          # discounts are applied automatically
    "addon_cost_per_month": 8.0,     # company's cost of providing a free add-on month (USD)
    "contract_discount_pct": 10,
    "discount_months": 6,
}


def _whatif_prob(raw, model, cols):
    X, _ = build_frame(raw)
    return model.predict_proba(X.reindex(columns=cols, fill_value=0.0))[:, 1]


def build_environment(customers, model, cols, eligible_fn, a=ASSUMPTIONS):
    """Precompute, for every customer x offer: eligibility, acceptance probability,
    churn probability if accepted, costs, and the true expected reward (used only by the oracle)."""
    c = customers.copy()
    n, k = len(c), len(ARMS)
    raw = c.reset_index().rename(columns={"index": "customerID"})
    raw["Churn"] = raw["Churn"].astype(int)

    p0 = c["churn_prob"].values                                  # out-of-fold baseline churn probability
    base_local = _whatif_prob(raw, model, cols)

    # Eligibility and discount caps from the SAME policy engine the agent uses
    eligible = np.zeros((n, k), dtype=bool)
    loyalty_pct = np.zeros(n)
    for i, cid in enumerate(c.index):
        for offer in eligible_fn(cid):
            if offer["code"] in ARMS:
                eligible[i, ARMS.index(offer["code"])] = True
                if offer["code"] == "LOYALTY_DISCOUNT":
                    loyalty_pct[i] = offer["discount_pct"]

    # What-if scenarios scored by the churn model
    addon_raw = raw.copy()
    for i, addon in enumerate(c["recommended_addon"].values):
        if isinstance(addon, str):
            addon_raw.loc[i, addon] = "Yes"
    contract_raw = raw.copy(); contract_raw["Contract"] = "One year"
    discount_raw = raw.copy(); discount_raw["MonthlyCharges"] = raw["MonthlyCharges"] * (1 - loyalty_pct / 100)

    def after(whatif):  # offers never increase churn; shrink the model's effect to a conservative causal estimate
        return np.clip(p0 - a["causal_shrinkage"] * np.maximum(0, base_local - whatif), 0, 1)

    p_after = np.column_stack([p0, after(_whatif_prob(addon_raw, model, cols)),
                               after(_whatif_prob(contract_raw, model, cols)), after(_whatif_prob(discount_raw, model, cols))])

    monthly = c["MonthlyCharges"].values
    revenue = monthly * a["horizon_months"]
    accept = np.column_stack([np.ones(n), np.full(n, a["accept_addon_trial"]),
                              np.where(c["tenure"].values >= 12, a["accept_contract_tenured"], a["accept_contract_new"]),
                              np.full(n, a["accept_discount"])])
    discount_cost = np.column_stack([np.zeros(n), np.zeros(n), revenue * a["contract_discount_pct"] / 100,
                                     monthly * loyalty_pct / 100 * a["discount_months"]])   # paid only if customer stays
    fixed_cost = np.column_stack([np.zeros(n), np.full(n, a["addon_cost_per_month"] * 3), np.zeros(n), np.zeros(n)])

    expected = accept * ((1 - p_after) * (revenue[:, None] - discount_cost) - fixed_cost) \
        + (1 - accept) * ((1 - p0) * revenue)[:, None]
    expected = np.where(eligible, expected, -np.inf)

    X, names = context_features(c)
    return {"ids": c.index.values, "X": X, "feature_names": names, "eligible": eligible,
            "p0": p0, "p_after": p_after, "accept": accept, "revenue": revenue, "discount_cost": discount_cost,
            "fixed_cost": fixed_cost, "expected": expected, "loyalty_pct": loyalty_pct, "segment": c["segment"].values}


BASE_FEATURES = ["bias", "churn_prob", "tenure", "monthly_charges", "month_to_month", "fiber", "electronic_check",
                 "senior", "num_services", "has_security", "has_tech_support", "billing_anomaly"]


def context_features(c):
    X = np.column_stack([
        np.ones(len(c)), c["churn_prob"], c["tenure"] / 72, c["MonthlyCharges"] / 120,
        c["Contract"].eq("Month-to-month"), c["InternetService"].eq("Fiber optic"),
        c["PaymentMethod"].eq("Electronic check"), c["SeniorCitizen"], c["num_services"] / 9,
        c["OnlineSecurity"].eq("Yes"), c["TechSupport"].eq("Yes"), c["billing_anomaly"],
    ]).astype(float)
    segs = pd.get_dummies(c["segment"], prefix="segment").astype(float)
    return np.hstack([X, segs.values]), BASE_FEATURES + list(segs.columns)


# --------------------------------------------------------------------------- policies
class RandomPolicy:
    name = "Random eligible offer"
    def __init__(self, rng): self.rng = rng
    def select(self, i, x, eligible): return self.rng.choice(np.flatnonzero(eligible))
    def update(self, x, arm, r): pass


class FixedPolicy:
    def __init__(self, arm):
        self.arm, self.name = ARMS.index(arm), f"Always {arm}"
    def select(self, i, x, eligible): return self.arm if eligible[self.arm] else 0
    def update(self, x, arm, r): pass


class RulePolicy:
    """Typical hand-written business rule."""
    name = "Business rule"
    def select(self, i, x, eligible):
        churn, month_to_month = x[1], x[4]
        for arm, cond in [(3, churn >= 0.6), (2, month_to_month == 1)]:
            if cond and eligible[arm]:
                return arm
        return 0
    def update(self, x, arm, r): pass


class OraclePolicy:
    name = "Oracle (knows true effects)"
    def __init__(self, expected): self.expected = expected
    def select(self, i, x, eligible): return int(np.argmax(self.expected[i]))
    def update(self, x, arm, r): pass


class LinUCB:
    """Disjoint LinUCB: one ridge regression per arm, pick the highest upper confidence bound."""
    def __init__(self, d, alpha=1.0, lam=1.0):
        self.name, self.alpha = f"LinUCB (alpha={alpha})", alpha
        self.A_inv = np.stack([np.eye(d) / lam for _ in ARMS]); self.b = np.zeros((len(ARMS), d))
        self.n, self.mean_reward = 0, 0.0
    def select(self, i, x, eligible):
        theta = np.einsum("kij,kj->ki", self.A_inv, self.b)
        ucb = theta @ x + self.alpha * np.sqrt(np.einsum("i,kij,j->k", x, self.A_inv, x))
        return int(np.argmax(np.where(eligible, ucb, -np.inf)))
    def update(self, x, arm, r):
        # Center rewards with a running mean so the zero prior is neutral, not pessimistic.
        # Without this, all-positive rewards make untried offers look worse than the first lucky one,
        # and exploration can lock in early.
        self.n += 1
        self.mean_reward += (r - self.mean_reward) / self.n
        r = r - self.mean_reward
        Ax = self.A_inv[arm] @ x
        self.A_inv[arm] -= np.outer(Ax, Ax) / (1 + x @ Ax)   # Sherman-Morrison rank-1 update
        self.b[arm] += r * x
    def greedy(self, X):
        return np.einsum("kij,kj->ki", self.A_inv, self.b) @ X.T


class LinThompson(LinUCB):
    """Linear Thompson sampling: sample a weight vector per arm from its posterior, act greedily on the sample."""
    def __init__(self, d, v=0.5, lam=1.0, rng=None):
        super().__init__(d, lam=lam)
        self.name, self.v, self.rng = f"Thompson sampling (v={v})", v, rng or np.random.default_rng()
    def select(self, i, x, eligible):
        theta = np.einsum("kij,kj->ki", self.A_inv, self.b)
        mean = theta @ x
        std = self.v * np.sqrt(np.einsum("i,kij,j->k", x, self.A_inv, x))  # posterior of x·theta is Gaussian
        return int(np.argmax(np.where(eligible, self.rng.normal(mean, std), -np.inf)))


# --------------------------------------------------------------------------- simulation
def simulate(env, policy, T, rng, scale=100.0):
    """Customers arrive one at a time; the policy picks an eligible offer and observes realized net revenue."""
    n = len(env["ids"])
    best = env["expected"].max(axis=1)
    arms, rewards, regret = np.zeros(T, int), np.zeros(T), np.zeros(T)
    for t, i in enumerate(rng.integers(0, n, T)):
        x, elig = env["X"][i], env["eligible"][i]
        a = policy.select(i, x, elig)
        accepted = rng.random() < env["accept"][i, a]
        p_churn = env["p_after"][i, a] if accepted else env["p0"][i]
        stays = rng.random() >= p_churn
        r = stays * (env["revenue"][i] - (env["discount_cost"][i, a] if accepted else 0)) \
            - (env["fixed_cost"][i, a] if accepted else 0)
        policy.update(x, a, r / scale)
        arms[t], rewards[t], regret[t] = a, r, best[i] - env["expected"][i, a]
    return {"arms": arms, "rewards": rewards, "regret": regret}
