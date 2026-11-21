"""Training, pemilihan model via CV, evaluasi test satu kali, dan Joblib.
Jalankan dari folder proyek: python latih.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fitur import CarFeatureEngineer, NUMERIC_FEATURES, RAW_FEATURES, extract_brand

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Origin"

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "kasus06_cars.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"
FIG_DIR = OUTPUT_DIR / "figures"

CLASSES = ["USA", "Europe", "Japan"]


def build_preprocessor(include_brand: bool) -> ColumnTransformer:
    """Membangun preprocessing numerik dan opsional kategorikal Brand."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])

    transformers = [
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
    ]

    if include_brand:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("brand", categorical_pipe, ["Brand"]))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def build_pipeline(model, include_brand: bool) -> Pipeline:
    """Menggabungkan feature engineering, preprocessing, dan classifier."""
    return Pipeline([
        ("feature_engineering", CarFeatureEngineer(include_brand=include_brand)),
        ("preprocessing", build_preprocessor(include_brand)),
        ("classifier", model),
    ])


def audit_data(
    df: pd.DataFrame,
    duplicate_rows_before_removal: int,
    shape_before_removal: tuple[int, int],
) -> dict:
    """Membuat ringkasan audit data setelah pemeriksaan duplikasi."""
    numeric_columns = [
        "Miles_per_Gallon",
        "Cylinders",
        "Displacement",
        "Horsepower",
        "Weight_in_lbs",
        "Acceleration",
    ]

    domain_checks = {
        f"{col}_le_zero": int((df[col].dropna() <= 0).sum())
        for col in numeric_columns
    }

    iqr_candidates = {}
    for col in numeric_columns:
        values = df[col].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        iqr_candidates[col] = int(
            ((values < lower_bound) | (values > upper_bound)).sum()
        )

    audit = {
        "shape_before_duplicate_removal": [
            int(shape_before_removal[0]),
            int(shape_before_removal[1]),
        ],
        "shape_after_duplicate_removal": [
            int(df.shape[0]),
            int(df.shape[1]),
        ],
        "duplicate_rows_before_removal": int(duplicate_rows_before_removal),
        "duplicate_rows_after_removal": int(df.duplicated().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_by_column": {
            col: int(value) for col, value in df.isna().sum().items()
        },
        "target_counts": {
            str(key): int(value) for key, value in df[TARGET].value_counts().items()
        },
        "target_percent": {
            str(key): float(value * 100)
            for key, value in df[TARGET].value_counts(normalize=True).items()
        },
        "domain_checks_nonpositive_numeric": domain_checks,
        "iqr_outlier_candidates_not_automatically_removed": iqr_candidates,
        "empty_name_rows": int(
            (df["Name"].fillna("").astype(str).str.strip() == "").sum()
        ),
        "unparseable_year_rows": int(
            pd.to_datetime(df["Year"], errors="coerce").isna().sum()
        ),
        "numeric_summary": df.select_dtypes(include=np.number).describe().to_dict(),
    }

    return audit


def run_training() -> dict:
    """Menjalankan audit, CV, evaluasi final, dan penyimpanan model."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_PATH}\n"
            "Pastikan kasus06_cars.csv berada di folder dataset/."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    expected = set(RAW_FEATURES + [TARGET])
    missing_cols = sorted(expected - set(df.columns))
    if missing_cols:
        raise ValueError(f"Kolom dataset tidak sesuai: {missing_cols}")

    if df[TARGET].isna().any():
        raise ValueError(
            "Target Origin memiliki nilai kosong; periksa dataset sebelum training."
        )

    # Catat kondisi duplikasi sebelum penghapusan.
    shape_before_removal = df.shape
    duplicate_count = int(df.duplicated().sum())

    # Penghapusan dilakukan secara eksplisit dan dicatat di audit.
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    X = df[RAW_FEATURES].copy()
    y = df[TARGET].astype(str).copy()

    if set(y.unique()) != set(CLASSES):
        raise ValueError(
            f"Kelas target tidak sesuai. Ditemukan: {sorted(y.unique())}"
        )

    audit = audit_data(
        df,
        duplicate_rows_before_removal=duplicate_count,
        shape_before_removal=shape_before_removal,
    )

    # Split data mentah sebelum imputasi, encoding, maupun scaling.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Analisis hubungan Brand-Origin hanya menggunakan data training.
    train_brand = X_train["Name"].map(extract_brand)
    brand_origin_table = pd.crosstab(train_brand, y_train)
    brand_origin_table.to_csv(OUTPUT_DIR / "brand_origin_cross_tab.csv")

    brands_per_origin_count = int(
        (brand_origin_table.gt(0).sum(axis=1) > 1).sum()
    )

    brand_summary = {
        "normalized_unique_brands_in_train": int(train_brand.nunique()),
        "train_brands_associated_with_multiple_origins": brands_per_origin_count,
        "each_observed_train_brand_has_one_origin_label": (
            brands_per_origin_count == 0
        ),
        "alias_rules": {
            "chevy/chevroelt": "chevrolet",
            "maxda": "mazda",
            "toyouta": "toyota",
            "vokswagen/vw": "volkswagen",
            "mercedes": "mercedes-benz",
        },
    }

    audit["brand_analysis_train_only"] = brand_summary

    with open(OUTPUT_DIR / "audit_data.json", "w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2, ensure_ascii=False)

    # Pemilihan model menggunakan 5-fold CV pada training set.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    experiments = []
    best_search = None
    best_key = None
    best_score = -np.inf

    # Ablation study: membandingkan model dengan dan tanpa fitur Brand.
    for include_brand in (False, True):
        for model_name in ("Logistic Regression", "Random Forest"):
            if model_name == "Logistic Regression":
                estimator = LogisticRegression(
                    max_iter=4000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                )
                param_grid = {
                    "classifier__C": [0.1, 1.0, 10.0],
                    "classifier__class_weight": [None, "balanced"],
                }
            else:
                estimator = RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                )
                param_grid = {
                    "classifier__n_estimators": [120],
                    "classifier__max_features": ["sqrt"],
                    "classifier__min_samples_leaf": [1, 2],
                    "classifier__class_weight": [None, "balanced_subsample"],
                }

            pipeline = build_pipeline(estimator, include_brand)

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="f1_macro",
                cv=cv,
                n_jobs=1,
                refit=True,
                return_train_score=True,
                error_score="raise",
            )

            search.fit(X_train, y_train)

            row = {
                "model": model_name,
                "brand_feature": include_brand,
                "cv_f1_macro_mean": float(search.best_score_),
                "cv_f1_macro_std": float(
                    search.cv_results_["std_test_score"][search.best_index_]
                ),
                "cv_train_f1_macro_mean": float(
                    search.cv_results_["mean_train_score"][search.best_index_]
                ),
                "best_params": json.dumps(
                    search.best_params_,
                    ensure_ascii=False,
                    default=str,
                ),
            }
            experiments.append(row)

            key = (model_name, include_brand)
            if search.best_score_ > best_score:
                best_score = float(search.best_score_)
                best_search = search
                best_key = key

    if best_search is None:
        raise RuntimeError("Tidak ada model yang berhasil dilatih.")

    # Diagnostik tambahan untuk merek yang tidak muncul pada fold training.
    # Ini merupakan diagnostik generalisasi, bukan pengganti CV utama.
    brand_groups = X_train["Name"].map(extract_brand)

    group_cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    group_macro_f1 = make_scorer(
        f1_score,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    group_scores = cross_val_score(
        best_search.best_estimator_,
        X_train,
        y_train,
        cv=group_cv,
        groups=brand_groups,
        scoring=group_macro_f1,
        n_jobs=1,
    )

    group_cv_diagnostic = {
        "description": (
            "StratifiedGroupKFold grouped by normalized car brand; "
            "train split only"
        ),
        "macro_f1_mean": float(group_scores.mean()),
        "macro_f1_std": float(group_scores.std()),
        "fold_scores": [float(value) for value in group_scores],
    }

    # Baseline Dummy juga dinilai melalui CV pada training set.
    dummy_pipe = build_pipeline(
        DummyClassifier(strategy="prior"),
        include_brand=False,
    )

    dummy_scores = cross_val_score(
        dummy_pipe,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
    )

    baseline_cv = {
        "model": "DummyClassifier (prior)",
        "cv_f1_macro_mean": float(dummy_scores.mean()),
        "cv_f1_macro_std": float(dummy_scores.std()),
    }

    comparison = pd.DataFrame(experiments).sort_values(
        "cv_f1_macro_mean",
        ascending=False,
    )
    comparison.to_csv(OUTPUT_DIR / "perbandingan_cv.csv", index=False)

    # GridSearchCV dengan refit=True sudah melatih model terbaik pada X_train.
    final_model = best_search.best_estimator_

    # Evaluasi held-out test dilakukan satu kali setelah pemilihan model.
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)

    model_classes = list(final_model.named_steps["classifier"].classes_)

    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision_macro": float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "roc_auc_ovr_macro": float(
            roc_auc_score(
                y_test,
                y_proba,
                labels=model_classes,
                multi_class="ovr",
                average="macro",
            )
        ),
    }

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{class_name}" for class_name in CLASSES],
        columns=[f"Predicted_{class_name}" for class_name in CLASSES],
    )

    cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")
    pd.DataFrame(report_dict).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    # Visualisasi confusion matrix.
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix - Car Origin Classification")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    # Visualisasi perbandingan eksperimen CV.
    cv_plot = comparison.copy()
    cv_plot["experiment"] = (
        cv_plot["model"]
        + cv_plot["brand_feature"].map({
            True: " + Brand",
            False: " tanpa Brand",
        })
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=cv_plot,
        x="cv_f1_macro_mean",
        y="experiment",
        color="#4472C4",
    )
    plt.xlim(0, 1)
    plt.xlabel("Mean Macro F1 (5-fold CV, training set)")
    plt.ylabel("Experiment")
    plt.title("Perbandingan Model dan Ablation Brand")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "perbandingan_cv.png", dpi=160)
    plt.close()

    metadata = {
        "project": (
            "UTS Intelegensi Buatan - Kasus 6: "
            "Car Origin Classification"
        ),
        "nim": "24552011132",
        "case": 6,
        "target": TARGET,
        "classes": CLASSES,
        "probability_class_order": model_classes,
        "selected_model": best_key[0],
        "brand_feature_used": bool(best_key[1]),
        "brand_analysis": brand_summary,
        "best_cv_macro_f1": float(best_score),
        "best_params": best_search.best_params_,
        "baseline_cv_macro_f1": baseline_cv,
        "unseen_brand_group_cv_diagnostic": group_cv_diagnostic,
        "test_metrics": metrics,
        "test_confusion_matrix": cm.tolist(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "feature_columns_raw": RAW_FEATURES,
        "available_model_years": sorted(
            int(value)
            for value in pd.to_datetime(
                df["Year"], errors="coerce"
            ).dt.year.dropna().unique()
        ),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "notes": [
            "Preprocessing statistics are learned inside Pipeline and CV training folds only.",
            "Brand alias normalization is deterministic and based on inspected dataset variants.",
            "The held-out test set is evaluated once after model selection.",
            "Group CV is a supplementary diagnostic for unseen-brand generalization.",
        ],
    }

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False, default=str)

    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False, default=str)

    bundle = {
        "pipeline": final_model,
        "metadata": metadata,
        "class_order": model_classes,
        "raw_feature_columns": RAW_FEATURES,
    }

    joblib.dump(
        bundle,
        MODEL_DIR / "car_origin_pipeline.joblib",
        compress=3,
    )

    print("\n=== AUDIT DATA ===")
    print("Shape before duplicate removal:", shape_before_removal)
    print("Shape after duplicate removal:", df.shape)
    print("Missing:\n", df.isna().sum().to_string())
    print("Duplicate rows before removal:", duplicate_count)
    print("Duplicate rows after removal:", int(df.duplicated().sum()))
    print("Target:\n", y.value_counts().to_string())

    print("\n=== CV COMPARISON (TRAIN ONLY) ===")
    print(
        comparison[
            [
                "model",
                "brand_feature",
                "cv_f1_macro_mean",
                "cv_f1_macro_std",
            ]
        ].to_string(index=False)
    )

    print(
        "\nBaseline Dummy CV Macro F1:",
        round(baseline_cv["cv_f1_macro_mean"], 4),
    )
    print(
        "Unseen-brand Group CV Macro F1 (diagnostic):",
        round(group_cv_diagnostic["macro_f1_mean"], 4),
    )
    print(
        "\nSelected:",
        best_key,
        "| CV Macro F1:",
        round(best_score, 4),
    )

    print("\n=== FINAL TEST METRICS (EVALUATED ONCE) ===")
    for metric_name, value in metrics.items():
        print(f"{metric_name:22s}: {value:.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=CLASSES,
            digits=4,
            zero_division=0,
        )
    )

    print("=== CONFUSION MATRIX ===")
    print(cm_df)

    print("\nModel saved:", MODEL_DIR / "car_origin_pipeline.joblib")
    print("Artifacts saved:", OUTPUT_DIR)

    return {
        "metadata": metadata,
        "comparison": comparison,
        "confusion_matrix": cm_df,
    }


if __name__ == "__main__":
    run_training()
