"""Bandit tests on a tiny synthetic problem where the best offer depends on the customer."""
import numpy as np

from bandit import ARMS, LinUCB, OraclePolicy, simulate


def toy_env(n=200, seed=0):
    rng = np.random.default_rng(seed)
    kind = rng.integers(0, 2, n)                         # two customer types
    X = np.column_stack([np.ones(n), kind, 1 - kind]).astype(float)
    k = len(ARMS)
    p_after = np.full((n, k), 0.5)
    p_after[kind == 1, 2] = 0.1                           # type 1 responds to CONTRACT_UPGRADE
    p_after[kind == 0, 1] = 0.1                           # type 0 responds to ADDON_TRIAL
    revenue = np.full(n, 1000.0)
    zeros = np.zeros((n, k))
    expected = (1 - p_after) * revenue[:, None]
    return {"ids": np.arange(n), "X": X, "eligible": np.ones((n, k), bool), "accept": np.ones((n, k)),
            "p_after": p_after, "p0": p_after[:, 0], "revenue": revenue, "discount_cost": zeros,
            "fixed_cost": zeros, "expected": expected}, kind


def test_oracle_has_zero_regret():
    env, _ = toy_env()
    res = simulate(env, OraclePolicy(env["expected"]), 500, np.random.default_rng(1))
    assert res["regret"].sum() == 0


def test_linucb_learns_customer_specific_offers():
    env, kind = toy_env()
    policy = LinUCB(env["X"].shape[1], alpha=1.0)
    simulate(env, policy, 4000, np.random.default_rng(2))
    greedy = policy.greedy(env["X"]).argmax(0)
    best = np.where(kind == 1, ARMS.index("CONTRACT_UPGRADE"), ARMS.index("ADDON_TRIAL"))
    assert (greedy == best).mean() > 0.95


def test_policy_respects_eligibility():
    env, _ = toy_env()
    env["eligible"][:, 2] = False
    policy = LinUCB(env["X"].shape[1])
    res = simulate(env, policy, 1000, np.random.default_rng(3))
    assert not (res["arms"] == 2).any()
