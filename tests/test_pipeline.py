from pathlib import Path
import sys
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BUNDLE_PATH = ROOT / "models" / "car_origin_pipeline.joblib"


def test_saved_pipeline_predicts_three_classes():
    bundle = joblib.load(BUNDLE_PATH)
    raw = pd.DataFrame([{
        "Name": "toyota corolla",
        "Miles_per_Gallon": 30.0,
        "Cylinders": 4,
        "Displacement": 120.0,
        "Horsepower": 85.0,
        "Weight_in_lbs": 2200,
        "Acceleration": 16.0,
        "Year": "1980-01-01",
    }])[bundle["raw_feature_columns"]]
    pred = bundle["pipeline"].predict(raw)
    proba = bundle["pipeline"].predict_proba(raw)
    assert pred[0] in bundle["class_order"]
    assert proba.shape == (1, 3)
    assert abs(float(proba.sum()) - 1.0) < 1e-8
    assert dict(zip(bundle["class_order"], proba[0]))[pred[0]] == max(proba[0])
