import numpy as np

from features import RAW_FIELDS, build_frame, encode_record


def test_build_frame_shapes(telco_raw):
    X, y = build_frame(telco_raw)
    assert len(X) == len(y) == len(telco_raw)
    assert set(y.unique()) == {0, 1}
    assert not X.isna().any().any()


def test_training_serving_parity(telco_raw):
    """The pure-Python encoder used in Lambda must match the pandas training pipeline exactly."""
    X, _ = build_frame(telco_raw)
    cols = list(X.columns)
    sample = telco_raw.sample(500, random_state=0)
    for idx, row in sample.iterrows():
        assert np.allclose(encode_record(row.to_dict(), cols), X.loc[idx].values), f"skew at row {idx}"


def test_new_customer_with_blank_total_charges(telco_raw):
    X, _ = build_frame(telco_raw)
    record = telco_raw.iloc[0][RAW_FIELDS].to_dict()
    record.update(tenure=0, TotalCharges=" ")
    vec = dict(zip(X.columns, encode_record(record, list(X.columns))))
    assert vec["avg_monthly_spend"] == float(record["MonthlyCharges"])
    assert vec["charge_shock"] == 0.0
