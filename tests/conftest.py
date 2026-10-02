import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


@pytest.fixture(scope="session")
def telco_raw():
    """Public Telco churn dataset (downloaded once, cached locally, never committed)."""
    local = os.path.join(ROOT, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    cache = os.path.join(ROOT, "tests", ".data", "telco.csv")
    if os.path.exists(local):
        return pd.read_csv(local)
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        pd.read_csv(DATA_URL).to_csv(cache, index=False)
    return pd.read_csv(cache)
