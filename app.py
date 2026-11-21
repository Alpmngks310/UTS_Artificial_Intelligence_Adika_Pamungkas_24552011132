"""Streamlit app: prediksi negara asal mobil dari spesifikasi mentah."""

from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "car_origin_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

st.set_page_config(
    page_title="Car Origin Predictor",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Origin Predictor")
st.caption(
    "Klasifikasi negara asal mobil berdasarkan spesifikasi "
    "dan informasi model kendaraan."
)


@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        return None

    loaded_bundle = joblib.load(MODEL_PATH)

    required_keys = {
        "pipeline",
        "metadata",
        "class_order",
        "raw_feature_columns",
    }

    if not isinstance(loaded_bundle, dict):
        raise ValueError("Format model bundle tidak valid.")

    missing_keys = required_keys - set(loaded_bundle.keys())

    if missing_keys:
        raise ValueError(
            f"Komponen model bundle tidak lengkap: {sorted(missing_keys)}"
        )

    if not isinstance(loaded_bundle["metadata"], dict):
        raise ValueError("Metadata model tidak valid.")

    # Gunakan metadata JSON terbaru jika tersedia.
    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            latest_metadata = json.load(f)

        loaded_bundle["metadata"] = latest_metadata

    return loaded_bundle


bundle = load_bundle()

if bundle is None:
    st.error("Model belum tersedia. Periksa folder models.")
    st.stop()

meta = bundle["metadata"]
pipeline = bundle["pipeline"]

with st.sidebar:
    st.header("Informasi Model")
    st.write(f"**Algoritma:** {meta['selected_model']}")
    st.write(
        f"**Brand feature:** "
        f"{'Digunakan' if meta['brand_feature_used'] else 'Tidak digunakan'}"
    )
    st.write(f"**Kelas:** {', '.join(meta['classes'])}")
    st.caption(
        "Hasil prediksi merupakan estimasi model berdasarkan pola "
        "pada dataset UTS, bukan bukti asal manufaktur yang pasti."
    )


pred_tab, eval_tab, about_tab = st.tabs(
    ["Prediksi", "Evaluasi Model", "Tentang"]
)


# ==========================================================
# TAB 1: PREDIKSI
# ==========================================================

with pred_tab:
    st.subheader("Masukkan spesifikasi mobil")

    with st.form("car_prediction_form"):
        left, right = st.columns(2)

        with left:
            name = st.text_input(
                "Nama / merek dan model mobil",
                value="toyota corolla",
                help="Contoh: toyota corolla atau ford maverick"
            )

            year_options = meta.get(
                "available_model_years",
                list(range(1970, 1983))
            )

            year = st.selectbox(
                "Tahun model",
                year_options,
                index=0
            )

            mpg_unknown = st.checkbox(
                "Miles per gallon tidak diketahui",
                value=False
            )

            mpg = st.number_input(
                "Miles per gallon",
                min_value=0.0,
                max_value=100.0,
                value=23.0,
                step=0.1,
                disabled=mpg_unknown
            )

            cylinders = st.selectbox(
                "Jumlah silinder",
                [3, 4, 5, 6, 8],
                index=1
            )

        with right:
            displacement = st.number_input(
                "Displacement",
                min_value=1.0,
                max_value=600.0,
                value=120.0,
                step=1.0
            )

            hp_unknown = st.checkbox(
                "Horsepower tidak diketahui",
                value=False
            )

            horsepower = st.number_input(
                "Horsepower",
                min_value=1.0,
                max_value=500.0,
                value=90.0,
                step=1.0,
                disabled=hp_unknown
            )

            weight = st.number_input(
                "Weight (lbs)",
                min_value=500,
                max_value=8000,
                value=2500,
                step=25
            )

            acceleration = st.number_input(
                "Acceleration",
                min_value=1.0,
                max_value=40.0,
                value=15.0,
                step=0.1
            )

        submit = st.form_submit_button(
            "Prediksi negara asal",
            type="primary",
            width="stretch"
        )

    if submit:
        if not name.strip():
            st.warning(
                "Nama mobil perlu diisi agar fitur Brand bisa diekstraksi."
            )
        else:
            row = pd.DataFrame([{
                "Name": name,
                "Miles_per_Gallon": None if mpg_unknown else mpg,
                "Cylinders": cylinders,
                "Displacement": displacement,
                "Horsepower": None if hp_unknown else horsepower,
                "Weight_in_lbs": weight,
                "Acceleration": acceleration,
                "Year": f"{year}-01-01",
            }])[bundle["raw_feature_columns"]]

            pred = pipeline.predict(row)[0]
            proba = pipeline.predict_proba(row)[0]

            probs = pd.DataFrame({
                "Origin": bundle["class_order"],
                "Probability": proba
            }).sort_values(
                "Probability",
                ascending=False
            )

            st.divider()
            st.success(f"Prediksi negara asal: **{pred}**")

            c1, c2 = st.columns([1, 2])

            with c1:
                st.metric("Predicted origin", pred)
                st.metric(
                    "Probabilitas tertinggi",
                    f"{probs.iloc[0]['Probability']:.1%}"
                )

            with c2:
                st.write("**Probabilitas setiap kelas**")
                st.bar_chart(probs.set_index("Origin"))
                st.dataframe(
                    probs.style.format({"Probability": "{:.1%}"}),
                    hide_index=True,
                    width="stretch"
                )


# ==========================================================
# TAB 2: EVALUASI MODEL
# ==========================================================

with eval_tab:
    st.subheader("Evaluasi pada held-out test set")

    st.caption(
        "Metrik berikut berasal dari evaluasi final satu kali pada "
        "data test setelah pemilihan model melalui cross-validation "
        "data train."
    )

    metrics = meta["test_metrics"]
    metric_items = list(metrics.items())

    for start in range(0, len(metric_items), 3):
        cols = st.columns(3)

        for col, (key, val) in zip(
            cols,
            metric_items[start:start + 3]
        ):
            col.metric(
                key.replace("_", " ").title(),
                f"{val:.3f}"
            )

    st.divider()

    st.write("**Confusion matrix**")

    cm = pd.DataFrame(
        meta["test_confusion_matrix"],
        index=meta["classes"],
        columns=meta["classes"]
    )

    st.dataframe(cm, width="stretch")

    st.divider()

    st.write("**Perbandingan model pada cross-validation**")

    cv_path = ROOT / "output" / "perbandingan_cv.csv"

    if cv_path.exists():
        cv_results = pd.read_csv(cv_path)
        st.dataframe(cv_results, width="stretch")
    else:
        st.warning("File perbandingan_cv.csv tidak ditemukan.")

    # ------------------------------------------------------
    # GROUP CROSS-VALIDATION
    # ------------------------------------------------------

    st.divider()

    st.subheader("Evaluasi generalisasi terhadap Brand baru")

    st.caption(
        "Evaluasi tambahan menggunakan StratifiedGroupKFold. "
        "Data dikelompokkan berdasarkan Brand yang telah dinormalisasi "
        "dan hanya menggunakan data train."
    )

    group_result = meta.get(
        "unseen_brand_group_cv_diagnostic",
        {}
    )

    group_mean = group_result.get("macro_f1_mean")
    group_std = group_result.get("macro_f1_std")
    fold_scores = group_result.get("fold_scores", [])

    if group_mean is not None and group_std is not None:
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Group CV Macro F1",
            f"{group_mean:.4f}"
        )

        col2.metric(
            "Standar deviasi",
            f"{group_std:.4f}"
        )

        col3.metric(
            "Jumlah fold",
            len(fold_scores)
        )

        if fold_scores:
            fold_table = pd.DataFrame({
                "Fold": [
                    f"Fold {i + 1}"
                    for i in range(len(fold_scores))
                ],
                "Macro F1": fold_scores
            })

            st.write("**Hasil setiap fold**")

            st.dataframe(
                fold_table.style.format({
                    "Macro F1": "{:.4f}"
                }),
                hide_index=True,
                width="stretch"
            )

        st.info(
            "Interpretasi: Skor Group CV lebih rendah dibandingkan "
            "cross-validation biasa. Hal ini menunjukkan bahwa model "
            "sangat terbantu oleh fitur Brand dan kemampuan prediksinya "
            "menurun ketika menghadapi Brand yang tidak ditemukan "
            "pada data train fold terkait. Hasil ini merupakan "
            "evaluasi diagnostik, bukan pengganti evaluasi held-out test set."
        )

    else:
        st.warning(
            "Hasil Group CV belum tersedia pada metadata."
        )


# ==========================================================
# TAB 3: TENTANG
# ==========================================================

with about_tab:
    st.subheader("Tentang proyek")

    st.write(
        "Proyek UTS Intelegensi Buatan - Kasus 6. "
        "Target klasifikasi adalah Origin dengan tiga kelas: "
        "USA, Europe, dan Japan."
    )

    st.write(f"Model terpilih: **{meta['selected_model']}**")
    st.write(f"Brand digunakan: **{meta['brand_feature_used']}**")

    st.write(
        f"Data train: **{meta['train_rows']}** | "
        f"Data test: **{meta['test_rows']}**"
    )

    st.write(f"Cross-validation: **{meta['cv']}**")

    st.info(
        "Probabilitas menunjukkan tingkat keyakinan relatif model "
        "pada tiga kelas yang dipelajari. Probabilitas bukan jaminan "
        "kebenaran faktual."
    )

    st.caption(
        "Group CV merupakan evaluasi tambahan untuk mengamati "
        "generalisasi model terhadap Brand yang tidak muncul "
        "pada data pelatihan fold terkait."
    )
