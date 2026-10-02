"""The Lambda handler, assembled exactly like 02_production does, with a mocked SageMaker endpoint."""
import json
import os

import boto3

from features import RAW_FIELDS, build_frame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeRuntime:
    def __init__(self, n_features):
        self.n = n_features

    def invoke_endpoint(self, EndpointName, ContentType, Body):
        assert ContentType == "text/csv" and len(Body.split(",")) == self.n

        class Body_:
            def read(self):
                return b"0.75\n"
        return {"Body": Body_()}


def load_handler(telco_raw, monkeypatch):
    X, _ = build_frame(telco_raw)
    cols = list(X.columns)
    monkeypatch.setattr(boto3, "client", lambda *a, **k: FakeRuntime(len(cols)))
    src = (open(os.path.join(ROOT, "src", "features.py")).read() + f"\n\nCOLUMNS = {json.dumps(cols)}\n\n"
           + open(os.path.join(ROOT, "api", "handler.py")).read())
    ns = {}
    exec(compile(src, "lambda_function.py", "exec"), ns)
    return ns["lambda_handler"]


def event(telco_raw):
    row = telco_raw.iloc[0]
    return {"customerID": row["customerID"], **{k: (v.item() if hasattr(v, "item") else v)
                                                 for k, v in row[RAW_FIELDS].items()}}


def test_scores_customer(telco_raw, monkeypatch):
    handler = load_handler(telco_raw, monkeypatch)
    resp = handler(event(telco_raw), None)
    body = json.loads(resp["body"])
    assert resp["statusCode"] == 200 and body["churn_probability"] == 0.75 and body["risk_band"] == "high"


def test_accepts_function_url_style_body(telco_raw, monkeypatch):
    handler = load_handler(telco_raw, monkeypatch)
    assert handler({"body": json.dumps(event(telco_raw))}, None)["statusCode"] == 200


def test_rejects_missing_fields(telco_raw, monkeypatch):
    handler = load_handler(telco_raw, monkeypatch)
    ev = event(telco_raw)
    ev.pop("tenure")
    resp = handler(ev, None)
    assert resp["statusCode"] == 400 and "tenure" in resp["body"]
