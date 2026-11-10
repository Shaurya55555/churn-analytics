import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pipeline as p  # noqa: E402


def sample():
    return pd.DataFrame({
        "tenure": [0, 10], "MonthlyCharges": [50.0, 70.0], "TotalCharges": [" ", "700"],
        "SeniorCitizen": [0, 1], "Churn": ["No", "Yes"], "PaymentMethod": ["Bank transfer (automatic)", "Mailed check"],
        **{c: ["Yes", "No"] for c in p.SERVICES},
    })


def test_clean_handles_blank_total_charges(tmp_path):
    f = tmp_path / "s.csv"
    sample().to_csv(f, index=False)
    df = p.load_clean(f)
    assert df["TotalCharges"].tolist() == [0.0, 700.0] and df["y"].tolist() == [0, 1]


def test_features(tmp_path):
    f = tmp_path / "s.csv"
    sample().to_csv(f, index=False)
    d = p.add_features(p.load_clean(f))
    assert d["n_services"].tolist() == [6, 0]
    assert d["autopay"].tolist() == [1, 0]
    assert d["avg_monthly_spend"].tolist() == [50.0, 70.0]
