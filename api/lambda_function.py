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


COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "num_services", "avg_monthly_spend", "charge_shock", "is_month_to_month", "gender_Male", "Partner_Yes", "Dependents_Yes", "PhoneService_Yes", "MultipleLines_No phone service", "MultipleLines_Yes", "InternetService_Fiber optic", "InternetService_No", "OnlineSecurity_No internet service", "OnlineSecurity_Yes", "OnlineBackup_No internet service", "OnlineBackup_Yes", "DeviceProtection_No internet service", "DeviceProtection_Yes", "TechSupport_No internet service", "TechSupport_Yes", "StreamingTV_No internet service", "StreamingTV_Yes", "StreamingMovies_No internet service", "StreamingMovies_Yes", "Contract_One year", "Contract_Two year", "PaperlessBilling_Yes", "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"]

# --- RetainAI Lambda handler ---
# The 02_production notebook assembles lambda_function.py = src/features.py + COLUMNS + this file,
# so serving uses exactly the same feature code as training.
import json
import os

import boto3

runtime = boto3.client("sagemaker-runtime")
ENDPOINT = os.environ.get("ENDPOINT_NAME", "retainai-churn-serverless")


def _response(code, body):
    return {"statusCode": code, "headers": {"Content-Type": "application/json"}, "body": json.dumps(body)}


def _risk_band(p):
    return "high" if p >= 0.6 else "medium" if p >= 0.3 else "low"


def lambda_handler(event, context):
    try:
        record = json.loads(event["body"]) if isinstance(event, dict) and "body" in event else event
        missing = [f for f in RAW_FIELDS if f not in record]
        if missing:
            return _response(400, {"error": f"missing fields: {missing}"})

        row = encode_record(record, COLUMNS)
        payload = ",".join(repr(v) for v in row)
        result = runtime.invoke_endpoint(EndpointName=ENDPOINT, ContentType="text/csv", Body=payload)
        raw = result["Body"].read().decode("utf-8")
        prob = float(raw.replace("\n", ",").strip(",").split(",")[0])

        monthly = float(record["MonthlyCharges"])
        return _response(200, {
            "customerID": record.get("customerID"),
            "churn_probability": round(prob, 4),
            "risk_band": _risk_band(prob),
            "monthly_revenue": monthly,
            "revenue_at_risk_12m": round(prob * monthly * 12, 2),
            "model_endpoint": ENDPOINT,
        })
    except Exception as e:  # return errors as JSON instead of crashing
        return _response(500, {"error": str(e)})
