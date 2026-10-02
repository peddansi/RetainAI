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
