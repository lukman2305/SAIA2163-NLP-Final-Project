"""
Multilingual Streamlit app for SAIA2163 NLP Final Project.

Use this after training with train_multilingual_models.py.
The saved pipeline handles Unicode-safe multilingual preprocessing internally.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from preprocessing_utils import SUPPORTED_LANGUAGES, add_language_token, language_name, normalize_language_code

st.set_page_config(
    page_title="Multilingual App Review Sentiment AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_sentiment_pipeline.pkl"
META_PATH = BASE_DIR / "backend_metadata.pkl"
DATA_PATH = BASE_DIR / "preprocessed_reviews.csv"
RAW_DATA_PATH = BASE_DIR / "Training_Data_Google_Play_reviews_6000.csv"
PRED_PATH = BASE_DIR / "final_model_predictions.csv"
COMPARISON_PATH = BASE_DIR / "model_evaluation_comparison.csv"
LANGUAGE_METRICS_PATH = BASE_DIR / "multilingual_language_metrics.csv"

LABEL_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_EMOJI = {"negative": "😞", "neutral": "😐", "positive": "😊"}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if not META_PATH.exists():
        return {}
    return joblib.load(META_PATH)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def model_input_for_prediction(text: str, lang_code: str, metadata: dict) -> str:
    uses_language_token = bool(metadata.get("uses_language_token", True))
    if uses_language_token:
        return add_language_token(text, lang_code)
    return text


def label_from_prediction(prediction) -> str:
    if isinstance(prediction, str):
        return prediction
    inverse = {0: "negative", 1: "neutral", 2: "positive"}
    try:
        return inverse[int(prediction)]
    except Exception:
        return str(prediction)


model = load_model()
metadata = load_metadata()
df = load_csv(DATA_PATH)
comparison_df = load_csv(COMPARISON_PATH)
language_metrics_df = load_csv(LANGUAGE_METRICS_PATH)
predictions_df = load_csv(PRED_PATH)

st.markdown("# 🌍 Multilingual App Review Sentiment AI")
st.caption("Sentiment analysis for English, French, German, Japanese, and Italian Google Play reviews.")

if model is None:
    st.error("Model file not found. Run `python train_multilingual_models.py --data Training_Data_Google_Play_reviews_6000.csv` first.")
    st.stop()

with st.sidebar:
    st.header("Project Files")
    st.write("Model:", "✅" if MODEL_PATH.exists() else "❌")
    st.write("Preprocessed data:", "✅" if DATA_PATH.exists() else "❌")
    st.write("Comparison CSV:", "✅" if COMPARISON_PATH.exists() else "❌")
    st.write("Language metrics:", "✅" if LANGUAGE_METRICS_PATH.exists() else "❌")

    st.header("Model Info")
    st.write("Best model:", metadata.get("best_model_name", "Unknown"))
    best_metrics = metadata.get("best_metrics", {})
    if best_metrics:
        st.write("Macro-F1:", f"{best_metrics.get('macro_f1', 0):.4f}")
        st.write("Neutral recall:", f"{best_metrics.get('neutral_recall', 0):.4f}")

st.subheader("Try a Review")
col1, col2 = st.columns([2, 1])
with col1:
    review_text = st.text_area(
        "Enter review text",
        height=130,
        placeholder="Example: This app keeps crashing after the update.",
    )
with col2:
    language_options = {code: f"{code} - {name}" for code, name in SUPPORTED_LANGUAGES.items()}
    selected_lang = st.selectbox(
        "Review language",
        options=list(language_options.keys()),
        format_func=lambda code: language_options[code],
        index=0,
        help="Choose the review language because the trained model uses a language token.",
    )

if st.button("Predict Sentiment", type="primary"):
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        prepared_text = model_input_for_prediction(review_text, selected_lang, metadata)
        pred = model.predict([prepared_text])[0]
        pred_label = label_from_prediction(pred)
        st.success(f"Prediction: {SENTIMENT_EMOJI.get(pred_label, '')} **{pred_label.title()}**")

        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba([prepared_text])[0]
                prob_df = pd.DataFrame({"Sentiment": LABEL_ORDER, "Probability": probs})
                st.bar_chart(prob_df.set_index("Sentiment"))
            except Exception:
                pass

        with st.expander("Prepared model input"):
            st.code(prepared_text)

st.divider()

if not comparison_df.empty:
    st.subheader("Model Comparison")
    show_cols = [c for c in ["model_name", "accuracy", "f1_score", "macro_f1", "neutral_recall"] if c in comparison_df.columns]
    st.dataframe(comparison_df[show_cols], use_container_width=True)

if not language_metrics_df.empty:
    st.subheader("Best Model Performance by Language")
    show_cols = [c for c in ["language", "language_name", "rows", "accuracy", "f1_score", "macro_f1", "neutral_recall"] if c in language_metrics_df.columns]
    st.dataframe(language_metrics_df[show_cols], use_container_width=True)

if not df.empty:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Languages", df["userLang"].nunique() if "userLang" in df.columns else "-")
    c3.metric("Sentiment Classes", df["label"].nunique() if "label" in df.columns else "-")

    col_a, col_b = st.columns(2)
    with col_a:
        if "label" in df.columns:
            st.write("Sentiment distribution")
            st.bar_chart(df["label"].value_counts().reindex(LABEL_ORDER).fillna(0))
    with col_b:
        if "userLang" in df.columns:
            st.write("Language distribution")
            st.bar_chart(df["userLang"].value_counts().sort_index())

    with st.expander("Preview preprocessed data"):
        preview_cols = [c for c in ["userLang", "language_name", "raw_text", "label", "app_name"] if c in df.columns]
        st.dataframe(df[preview_cols].head(30), use_container_width=True)

if not predictions_df.empty:
    with st.expander("Final model predictions sample"):
        preview_cols = [c for c in ["userLang", "raw_text", "label", "predicted_label", "app_name"] if c in predictions_df.columns]
        st.dataframe(predictions_df[preview_cols].head(50), use_container_width=True)
