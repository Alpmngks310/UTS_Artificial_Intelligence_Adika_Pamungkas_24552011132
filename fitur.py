"""Feature engineering deterministik untuk UTS AI Kasus 6.

Transformasi ini sengaja tidak mempelajari statistik dari seluruh dataset.
Statistik imputasi, encoding, dan scaling dipelajari oleh Pipeline sklearn
pada training fold saja.
"""
from __future__ import annotations

import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

RAW_FEATURES = [
    "Name", "Miles_per_Gallon", "Cylinders", "Displacement",
    "Horsepower", "Weight_in_lbs", "Acceleration", "Year",
]
NUMERIC_FEATURES = [
    "Miles_per_Gallon", "Cylinders", "Displacement", "Horsepower",
    "Weight_in_lbs", "Acceleration", "Model_Year",
]
BRAND_ALIASES = {
    # Alias/ejaan yang jelas dari pemeriksaan nilai unik dataset.
    "chevy": "chevrolet",
    "chevroelt": "chevrolet",
    "maxda": "mazda",
    "toyouta": "toyota",
    "vokswagen": "volkswagen",
    "vw": "volkswagen",
    "mercedes": "mercedes-benz",
}


def extract_brand(name: object) -> str:
    """Ambil token pertama dari nama mobil dan normalisasi alias eksplisit."""
    if pd.isna(name):
        return "Unknown"
    cleaned = re.sub(r"\s+", " ", str(name).strip().lower())
    if not cleaned:
        return "Unknown"
    token = cleaned.split(" ", 1)[0]
    return BRAND_ALIASES.get(token, token)


class CarFeatureEngineer(BaseEstimator, TransformerMixin):
    """Ubah input mentah menjadi fitur numerik dan opsional Brand."""

    def __init__(self, include_brand: bool = True):
        self.include_brand = include_brand

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CarFeatureEngineer membutuhkan pandas DataFrame.")
        missing = sorted(set(RAW_FEATURES) - set(X.columns))
        if missing:
            raise ValueError(f"Kolom input tidak lengkap: {missing}")
        self.feature_names_out_ = NUMERIC_FEATURES + (["Brand"] if self.include_brand else [])
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_out_")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=RAW_FEATURES)
        d = X.copy()

        # Operasi deterministik: tidak menggunakan statistik dari data test.
        year = pd.to_datetime(d["Year"], errors="coerce")
        out = d[[
            "Miles_per_Gallon", "Cylinders", "Displacement", "Horsepower",
            "Weight_in_lbs", "Acceleration",
        ]].copy()
        out["Model_Year"] = year.dt.year.astype("float64")

        if self.include_brand:
            out["Brand"] = d["Name"].map(extract_brand).astype("object")
        return out[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_
