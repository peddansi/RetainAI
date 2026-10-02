"""
RetainAI feature engineering — single source of truth for training AND serving.

build_frame()   -> pandas, used for training (batch)
encode_record() -> pure Python, used at inference time (Lambda has no pandas)
Both must produce identical feature vectors; the parity test in 02_production checks this.
"""

NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
SERVICES = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
RAW_FIELDS = NUMERIC + CATEGORICAL
ENGINEERED = ["num_services", "avg_monthly_spend", "charge_shock", "is_month_to_month"]


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _engineered(r):
    tenure, monthly, total = _num(r["tenure"]), _num(r["MonthlyCharges"]), _num(r["TotalCharges"])
    avg = total / tenure if tenure > 0 else monthly
    return {
        "num_services": float(sum(r[s] in ("Yes", "DSL", "Fiber optic") for s in SERVICES)),
        "avg_monthly_spend": avg,
        "charge_shock": monthly - avg,
        "is_month_to_month": float(r["Contract"] == "Month-to-month"),
    }


def encode_record(record, columns):
    """Turn one raw customer dict into the model's feature vector (ordered by `columns`)."""
    values = {k: _num(record[k]) for k in NUMERIC}
    values.update(_engineered(record))
    row = []
    for c in columns:
        if c in values:
            row.append(float(values[c]))
        else:
            col, val = c.split("_", 1)          # one-hot column, e.g. "Contract_Two year"
            row.append(1.0 if str(record.get(col)) == val else 0.0)
    return row


def build_frame(df):
    """Pandas version for training. Returns (X, y)."""
    import numpy as np
    import pandas as pd

    d = df.copy()
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0)
    d["num_services"] = d[SERVICES].isin(["Yes", "DSL", "Fiber optic"]).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d["avg_monthly_spend"] = np.where(d["tenure"] > 0, d["TotalCharges"] / d["tenure"], d["MonthlyCharges"])
    d["charge_shock"] = d["MonthlyCharges"] - d["avg_monthly_spend"]
    d["is_month_to_month"] = (d["Contract"] == "Month-to-month").astype(int)
    y = (d["Churn"] == "Yes").astype(int) if d["Churn"].dtype == object else d["Churn"].astype(int)
    X = pd.get_dummies(d[RAW_FIELDS + ENGINEERED], drop_first=True).astype(float)
    return X, y
