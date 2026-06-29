"""
Multilingual App Review Sentiment AI
Rubric-ready Streamlit application with a polished dashboard.

Required sections covered:
1. Home/About
2. Text Analyzer
3. Data Explorer
4. Visualizations
5. Model Info

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    from preprocessing_utils import (
        SUPPORTED_LANGUAGES,
        add_language_token,
        language_name,
        normalize_language_code,
    )
except Exception:
    # Fallback so the UI can still open even if VS Code/Pylance or the runtime cannot find the helper file.
    SUPPORTED_LANGUAGES = {"EN": "English", "FR": "French", "DE": "German", "IT": "Italian", "JP": "Japanese"}

    def normalize_language_code(code: str) -> str:
        code = str(code).strip().upper()
        return code if code in SUPPORTED_LANGUAGES else "EN"

    def language_name(code: str) -> str:
        return SUPPORTED_LANGUAGES.get(normalize_language_code(code), "English")

    def add_language_token(text: str, lang: str) -> str:
        return f"lang_{normalize_language_code(lang).lower()} {text}"


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"

MODEL_PATH = BASE_DIR / "best_sentiment_pipeline.pkl"
META_PKL_PATH = BASE_DIR / "backend_metadata.pkl"
META_JSON_PATH = BASE_DIR / "backend_metadata.json"
DATA_PATH = BASE_DIR / "preprocessed_reviews.csv"
RAW_DATA_PATH = BASE_DIR / "Training_Data_Google_Play_reviews_6000.csv"
PRED_PATH = BASE_DIR / "final_model_predictions.csv"
COMPARISON_PATH = BASE_DIR / "model_evaluation_comparison.csv"
LANGUAGE_METRICS_PATH = BASE_DIR / "multilingual_language_metrics.csv"

# The trained model may return numeric classes (0, 1, 2).
# This mapping converts them into readable sentiment labels for the app UI.
# Change this mapping here only if your training notebook used a different encoding.
LABEL_ID_TO_NAME = {
    "0": "negative",
    "1": "neutral",
    "2": "positive",
    0: "negative",
    1: "neutral",
    2: "positive",
}

LABEL_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_EMOJI = {"negative": "😞", "neutral": "😐", "positive": "😊"}
SENTIMENT_COLOR = {"negative": "#ef4444", "neutral": "#f59e0b", "positive": "#22c55e"}


def sentiment_label(value: Any) -> str:
    """Convert model/dataset labels like 0, 1, 2 into negative/neutral/positive."""
    if pd.isna(value):
        return "unknown"
    raw = value
    text_value = str(value).strip().lower()

    if raw in LABEL_ID_TO_NAME:
        return LABEL_ID_TO_NAME[raw]
    if text_value in LABEL_ID_TO_NAME:
        return LABEL_ID_TO_NAME[text_value]
    if text_value in LABEL_ORDER:
        return text_value
    return text_value


def sentiment_label_title(value: Any) -> str:
    label = sentiment_label(value)
    return label.capitalize() if label else "Unknown"


def display_sentiment_series(series: pd.Series) -> pd.Series:
    return series.apply(sentiment_label)

TEAM_MEMBERS = [
    "Muhammad Lukman Bin Nasrum (A24AI0061)",
    "Muhammad Zahin Bin Mohd Zamri (A24AI0065)",
    "Raqib Hazim Bin Abdul Hamid (A24AI0118)",
    "Hasnawi Imran Bin Mohd Saidi (A24AI0032)",
]


# -----------------------------------------------------------------------------
# Page config and style
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multilingual App Review Sentiment AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 2.5rem;
    }

    .hero {
        padding: 2rem 2rem;
        border-radius: 26px;
        background: linear-gradient(135deg, #111827 0%, #312e81 45%, #0f766e 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.25);
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .hero p {
        margin-top: 0.6rem;
        margin-bottom: 0;
        font-size: 1.02rem;
        opacity: 0.92;
        max-width: 950px;
    }

    .pill-row {
        margin-top: 1.1rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }

    .pill {
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.22);
        color: white;
        padding: 0.42rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 1rem 1.05rem;
        min-height: 150px;
        box-sizing: border-box;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
        margin-bottom: 1.4rem;
        overflow: visible;
    }

    .metric-card .label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .metric-card .value {
        color: #0f172a;
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 0.25rem;
        line-height: 1.18;
        word-wrap: break-word;
        overflow-wrap: anywhere;
    }

    .metric-card .sub {
        color: #64748b;
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }

    .section-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        padding: 1.25rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }

    .soft-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1rem;
        margin-bottom: 0.85rem;
    }

    .result-box {
        border-radius: 24px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 16px 35px rgba(15, 23, 42, 0.2);
    }

    .result-box h2 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
    }

    .result-box p {
        margin: 0.4rem 0 0 0;
        opacity: 0.92;
    }

    .small-muted {
        color: #64748b;
        font-size: 0.88rem;
    }

    .callout {
        border-left: 5px solid #2563eb;
        background: #eff6ff;
        padding: 1rem;
        border-radius: 14px;
        color: #1e3a8a;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
    }

    div[data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    div[data-testid="stSidebar"] .stRadio label {
        font-weight: 600;
    }

    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 1.2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model() -> Any:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data(show_spinner=False)
def load_metadata() -> Dict[str, Any]:
    if META_JSON_PATH.exists():
        try:
            with open(META_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    if META_PKL_PATH.exists():
        try:
            meta = joblib.load(META_PKL_PATH)
            return meta if isinstance(meta, dict) else {}
        except Exception:
            pass
    return {}


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    if RAW_DATA_PATH.exists():
        return pd.read_csv(RAW_DATA_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_comparison() -> pd.DataFrame:
    if COMPARISON_PATH.exists():
        return pd.read_csv(COMPARISON_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_language_metrics() -> pd.DataFrame:
    if LANGUAGE_METRICS_PATH.exists():
        return pd.read_csv(LANGUAGE_METRICS_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    if PRED_PATH.exists():
        return pd.read_csv(PRED_PATH)
    return pd.DataFrame()


model = load_model()
metadata = load_metadata()
df = load_dataset()
comparison_df = load_comparison()
language_metrics_df = load_language_metrics()
pred_df = load_predictions()


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def fmt_pct(value: Any, digits: int = 2) -> str:
    try:
        value = float(value)
        if value <= 1:
            value *= 100
        return f"{value:.{digits}f}%"
    except Exception:
        return "N/A"


def first_existing_column(frame: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def get_label_column(frame: pd.DataFrame) -> Optional[str]:
    return first_existing_column(frame, ["label", "sentiment", "sentiment_label", "target"])


def get_text_column(frame: pd.DataFrame) -> Optional[str]:
    return first_existing_column(frame, ["text", "clean_text", "processed_text", "content", "review", "raw_text"])


def get_language_column(frame: pd.DataFrame) -> Optional[str]:
    return first_existing_column(frame, ["language", "lang", "userLang", "language_code"])


def infer_best_model_name() -> str:
    for key in ["best_model", "model_name", "selected_model", "deployed_model"]:
        if key in metadata and metadata[key]:
            return str(metadata[key])
    if not comparison_df.empty:
        metric = "macro_f1" if "macro_f1" in comparison_df.columns else "accuracy"
        if metric in comparison_df.columns and "model_name" in comparison_df.columns:
            row = comparison_df.sort_values(metric, ascending=False).iloc[0]
            return str(row["model_name"])
    return "Multilingual Sentiment Pipeline"


def infer_metric(metric_name: str) -> Optional[float]:
    # Metadata first
    for key in [metric_name, metric_name.lower(), metric_name.replace("-", "_"), metric_name.replace(" ", "_")]:
        if key in metadata:
            try:
                return float(metadata[key])
            except Exception:
                pass

    # Comparison CSV fallback
    if not comparison_df.empty:
        metric = metric_name.lower().replace("-", "_").replace(" ", "_")
        if metric in comparison_df.columns:
            row = comparison_df.sort_values(metric, ascending=False).iloc[0]
            try:
                return float(row[metric])
            except Exception:
                pass
    return None


def prepare_input_text(text: str, lang_code: str) -> str:
    lang_code = normalize_language_code(lang_code)
    return add_language_token(text, lang_code)


def predict_review(text: str, lang_code: str) -> Tuple[Optional[str], Optional[float], Dict[str, float], str]:
    if model is None:
        return None, None, {}, ""

    prepared_text = prepare_input_text(text, lang_code)

    # Raw prediction can be a number such as 0/1/2 depending on how the model was trained.
    # Convert it immediately so the UI shows negative/neutral/positive instead of raw IDs.
    raw_predicted = model.predict([prepared_text])[0]
    predicted = sentiment_label(raw_predicted)

    probabilities: Dict[str, float] = {}
    confidence: Optional[float] = None

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba([prepared_text])[0]
            raw_classes = list(getattr(model, "classes_", LABEL_ORDER))

            # Map model classes to readable labels, for example 0 -> negative.
            for i, class_value in enumerate(raw_classes):
                readable_class = sentiment_label(class_value)
                probabilities[readable_class] = float(proba[i])

            confidence = probabilities.get(predicted, float(np.max(proba)))
        except Exception:
            pass

    if not probabilities and hasattr(model, "decision_function"):
        try:
            scores = np.array(model.decision_function([prepared_text])[0], dtype=float)
            exp_scores = np.exp(scores - np.max(scores))
            proba = exp_scores / exp_scores.sum()
            raw_classes = list(getattr(model, "classes_", LABEL_ORDER))

            for i, class_value in enumerate(raw_classes):
                readable_class = sentiment_label(class_value)
                probabilities[readable_class] = float(proba[i])

            confidence = probabilities.get(predicted, float(np.max(proba)))
        except Exception:
            pass

    # Keep the table/chart in a clean order when all three classes are available.
    if probabilities:
        probabilities = {label: probabilities[label] for label in LABEL_ORDER if label in probabilities}

    return predicted, confidence, probabilities, prepared_text


def get_pipeline_parts(pipeline: Any) -> Tuple[Optional[Any], Optional[Any]]:
    if not hasattr(pipeline, "steps"):
        return None, None
    try:
        from sklearn.pipeline import Pipeline

        estimator = pipeline.steps[-1][1]
        if len(pipeline.steps) > 1:
            feature_pipe = Pipeline(pipeline.steps[:-1])
        else:
            feature_pipe = None
        return feature_pipe, estimator
    except Exception:
        return None, None


def get_influential_features(pipeline: Any, prepared_text: str, predicted_label: str, top_n: int = 10) -> pd.DataFrame:
    feature_pipe, estimator = get_pipeline_parts(pipeline)
    if feature_pipe is None or estimator is None:
        return pd.DataFrame()
    if not hasattr(estimator, "coef_"):
        return pd.DataFrame()

    try:
        X = feature_pipe.transform([prepared_text])
        feature_names = feature_pipe.get_feature_names_out()
        raw_classes = list(getattr(estimator, "classes_", LABEL_ORDER))
        readable_classes = [sentiment_label(c) for c in raw_classes]
        class_idx = readable_classes.index(sentiment_label(predicted_label)) if sentiment_label(predicted_label) in readable_classes else int(np.argmax(estimator.coef_.sum(axis=1)))

        coef = estimator.coef_
        if coef.ndim == 1:
            class_coef = coef
        elif coef.shape[0] == 1:
            class_coef = coef[0]
        else:
            class_coef = coef[class_idx]

        if hasattr(X, "multiply"):
            contributions = X.multiply(class_coef).toarray().ravel()
            values = X.toarray().ravel()
        else:
            X_arr = np.asarray(X).ravel()
            contributions = X_arr * class_coef
            values = X_arr

        non_zero_idx = np.where(values != 0)[0]
        if len(non_zero_idx) == 0:
            return pd.DataFrame()

        top_idx = non_zero_idx[np.argsort(contributions[non_zero_idx])[-top_n:]][::-1]
        rows = []
        for idx in top_idx:
            name = str(feature_names[idx])
            # Clean common sklearn prefixes from FeatureUnion/ColumnTransformer.
            name = name.replace("word__", "").replace("char__", "").replace("tfidf__", "")
            name = name.replace("features__", "")
            rows.append({"feature": name, "influence_score": float(contributions[idx])})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def image_card(title: str, image_name: str, caption: str = "") -> None:
    path = IMAGES_DIR / image_name
    st.markdown(f"#### {title}")
    if path.exists():
        st.image(str(path), caption=caption or title, use_container_width=True)
    else:
        st.info(f"Image not found: images/{image_name}")


def render_metric_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dataset_summary() -> Dict[str, Any]:
    label_col = get_label_column(df)
    lang_col = get_language_column(df)
    text_col = get_text_column(df)
    total = len(df)
    classes = df[label_col].nunique() if label_col else 0
    languages = df[lang_col].nunique() if lang_col else len(SUPPORTED_LANGUAGES)
    avg_len = 0
    if text_col and total:
        avg_len = df[text_col].dropna().astype(str).str.len().mean()
    return {"total": total, "classes": classes, "languages": languages, "avg_len": avg_len}


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.markdown("# 🌍 Sentiment AI")
st.sidebar.markdown("Multilingual Google Play Review Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📌 Home/About",
        "🔎 Text Analyzer",
        "📊 Data Explorer",
        "🖼️ Visualizations",
        "🤖 Model Info",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Supported Languages")
for code, name in SUPPORTED_LANGUAGES.items() if isinstance(SUPPORTED_LANGUAGES, dict) else []:
    st.sidebar.markdown(f"• **{code}** — {name}")

if model is None:
    st.sidebar.error("Model file not found. Train the model first.")
else:
    st.sidebar.success("Model loaded successfully")


# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------
def dashboard_page() -> None:
    summary = dataset_summary()
    best_model_name = infer_best_model_name()
    accuracy = infer_metric("accuracy")
    macro_f1 = infer_metric("macro_f1")
    neutral_recall = infer_metric("neutral_recall")

    st.markdown(
        """
        <div class="hero">
            <h1>Multilingual App Review Sentiment AI</h1>
            <p>Explore multilingual Google Play reviews, analyzing user sentiment, and presenting model performance.</p>
            <div class="pill-row">
                <span class="pill">Negative / Neutral / Positive</span>
                <span class="pill">EN · FR · DE · IT · JP</span>
                <span class="pill">TF-IDF + N-grams</span>
                <span class="pill">Streamlit Deployment</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Total Reviews", f"{summary['total']:,}", "Full multilingual dataset")
    with c2:
        render_metric_card("Languages", str(summary["languages"]), "English, French, German, Italian, Japanese")
    with c3:
        render_metric_card("Classes", str(summary["classes"] or 3), "Negative, neutral, positive")
    with c4:
        render_metric_card("Best Model", best_model_name, "Selected deployment pipeline")

    st.markdown("<div style='height: 1.6rem;'></div>", unsafe_allow_html=True)

    c5, c6, c7 = st.columns(3)
    with c5:
        render_metric_card("Accuracy", fmt_pct(accuracy), "Overall prediction correctness")
    with c6:
        render_metric_card("Macro-F1", fmt_pct(macro_f1), "Fairer class-balanced metric")
    with c7:
        render_metric_card("Neutral Recall", fmt_pct(neutral_recall), "Minority-class detection")

    st.markdown("### Dashboard Overview")
    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Sentiment and Language Snapshot")
        label_col = get_label_column(df)
        lang_col = get_language_column(df)
        if not df.empty and label_col:
            sentiment_counts = display_sentiment_series(df[label_col]).value_counts().reindex(LABEL_ORDER).dropna()
            st.bar_chart(sentiment_counts)
        else:
            st.info("Dataset not available yet.")

        if not df.empty and lang_col:
            st.markdown("#### Language Counts")
            lang_counts = df[lang_col].value_counts().sort_index()
            st.bar_chart(lang_counts)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Quick Project Summary")
        st.markdown(
            """
            - Solves multilingual app review sentiment classification.
            - Uses Unicode-safe preprocessing instead of English-only lemmatization.
            - Compares word TF-IDF, character n-grams, combined TF-IDF, and optional Transformer models.
            - Selects the final model using accuracy, Macro-F1, and neutral recall.
            - Deploys the model in a Streamlit app for real-time prediction.
            """
        )
        st.markdown('<div class="callout"><b>Key insight:</b> Neutral sentiment is the hardest class because it has fewer examples and is more ambiguous than positive or negative reviews.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Main Visual Summary")
    v1, v2 = st.columns(2)
    with v1:
        image_card("Sentiment Class Distribution", "class_distribution_multilingual.png")
        image_card("Model Accuracy Comparison", "model_accuracy_comparison.png")
    with v2:
        image_card("Language Distribution", "language_distribution.png")
        image_card("Model Macro-F1 Comparison", "model_macro_f1_comparison.png")


# -----------------------------------------------------------------------------
# Home/About
# -----------------------------------------------------------------------------
def home_page() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>About the Project</h1>
            <p>This Streamlit application supports multilingual sentiment analysis for Google Play app reviews.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Project Title")
    st.markdown("**Multilingual App Review Sentiment Analysis**")

    st.markdown("### Problem Being Solved")
    st.markdown(
        """
        App reviews are written in different languages and often contain short text, spelling mistakes, emojis, slang, and mixed opinions. Manual analysis is slow and inconsistent. This app uses NLP to classify reviews into **negative**, **neutral**, or **positive** sentiment so that user feedback can be understood faster.
        """
    )

    st.markdown("### How to Use the App")
    st.markdown(
        """
        1. Open **Text Analyzer** from the sidebar.
        2. Enter or paste an app review.
        3. Choose the review language.
        4. Click **Analyze Review**.
        5. View the predicted sentiment, confidence score, probability distribution, and influential words/features.
        """
    )

    st.markdown("### Team Members")
    for member in TEAM_MEMBERS:
        st.markdown(f"- {member}")

    st.markdown("### Project Workflow")
    st.markdown(
        """
        Raw dataset → Multilingual preprocessing → Language token injection → TF-IDF/n-gram features → Model training → Evaluation → Streamlit deployment
        """
    )


# -----------------------------------------------------------------------------
# Text Analyzer
# -----------------------------------------------------------------------------
def analyzer_page() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Text Analyzer</h1>
            <p>Type a review, select the language, and analyze the predicted sentiment in real time.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is None:
        st.error("Model file `best_sentiment_pipeline.pkl` was not found. Train the model before using the analyzer.")
        return

    examples = {
        "Positive English": ("This app is very useful and easy to use.", "EN"),
        "Negative English": ("The app keeps crashing and I cannot login.", "EN"),
        "Neutral English": ("Now how to mute stories?", "EN"),
        "Positive French": ("Cette application est très pratique.", "FR"),
        "Negative French": ("Cette application est très frustrante.", "FR"),
        "Neutral French": ("Cette application permet de discuter.", "FR"),
        "Positive German": ("Die App ist sehr nützlich und benutzerfreundlich.", "DE"),
        "Negative German": ("Die App ist sehr frustrierend und voller Bugs.", "DE"),
        "Neutral German": ("Zeigt neue nachrichten an obwohl keine gekommen sind. roter kreis.", "DE")
        
    }

    left, right = st.columns([1.2, 0.8])
    with left:
        selected_example = st.selectbox("Try an example", ["Custom input"] + list(examples.keys()))
        default_text = ""
        default_lang = "EN"
        if selected_example != "Custom input":
            default_text, default_lang = examples[selected_example]

        review_text = st.text_area(
            "Enter app review text",
            value=default_text,
            height=170,
            placeholder="Example: This app is useful but sometimes crashes after update...",
        )

        language_options = list(SUPPORTED_LANGUAGES.keys()) if isinstance(SUPPORTED_LANGUAGES, dict) else ["EN", "FR", "DE", "IT", "JP"]
        selected_lang = st.selectbox(
            "Review language",
            language_options,
            index=language_options.index(default_lang) if default_lang in language_options else 0,
            format_func=lambda code: f"{code} — {language_name(code)}",
        )

        analyze = st.button("Analyze Review", type="primary", use_container_width=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### What the analyzer shows")
        st.markdown(
            """
            - Predicted sentiment label
            - Confidence/probability score
            - Probability distribution by class
            - Influential words/features when available
            - Prepared model input with language token
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze:
        if not review_text.strip():
            st.warning("Please enter a review first.")
            return

        predicted, confidence, probabilities, prepared_text = predict_review(review_text, selected_lang)
        if predicted is None:
            st.error("Prediction failed because the model is not loaded.")
            return

        color = SENTIMENT_COLOR.get(predicted, "#2563eb")
        emoji = SENTIMENT_EMOJI.get(predicted, "💬")
        conf_text = fmt_pct(confidence) if confidence is not None else "N/A"

        st.markdown(
            f"""
            <div class="result-box" style="background: linear-gradient(135deg, {color}, #0f172a);">
                <h2>{emoji} {predicted.capitalize()}</h2>
                <p>Confidence score: <b>{conf_text}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Prediction Probabilities")
        if probabilities:
            prob_df = pd.DataFrame(
                [{"sentiment": key, "probability": value} for key, value in probabilities.items()]
            ).sort_values("probability", ascending=False)
            st.dataframe(prob_df.assign(probability_percent=prob_df["probability"].apply(fmt_pct)), use_container_width=True)
            st.bar_chart(prob_df.set_index("sentiment")["probability"])
        else:
            st.info("This model does not expose probability scores. Showing predicted label only.")

        st.markdown("### Words / Features Influencing the Prediction")
        influence_df = get_influential_features(model, prepared_text, predicted, top_n=12)
        if not influence_df.empty:
            st.dataframe(influence_df, use_container_width=True)
            st.bar_chart(influence_df.set_index("feature")["influence_score"])
        else:
            st.info("Feature influence is not available for this model type, or no non-zero features were found.")

        with st.expander("Show prepared model input"):
            st.code(prepared_text)


# -----------------------------------------------------------------------------
# Data Explorer
# -----------------------------------------------------------------------------
def data_explorer_page() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Data Explorer</h1>
            <p>Inspect sample reviews, dataset statistics, sentiment distribution, and language distribution.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        st.error("No dataset found. Expected `preprocessed_reviews.csv` or raw CSV in the project folder.")
        return

    summary = dataset_summary()
    label_col = get_label_column(df)
    lang_col = get_language_column(df)
    text_col = get_text_column(df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Rows", f"{summary['total']:,}", "Dataset records")
    with c2:
        render_metric_card("Languages", str(summary["languages"]), "Detected language groups")
    with c3:
        render_metric_card("Classes", str(summary["classes"] or 3), "Sentiment labels")
    with c4:
        render_metric_card("Avg Text Length", f"{summary['avg_len']:.1f}", "Characters")

    st.markdown("### Filters")
    f1, f2 = st.columns(2)
    filtered = df.copy()
    with f1:
        if lang_col:
            lang_values = sorted(filtered[lang_col].dropna().astype(str).unique())
            selected_langs = st.multiselect("Filter languages", lang_values, default=lang_values)
            filtered = filtered[filtered[lang_col].astype(str).isin(selected_langs)]
    with f2:
        if label_col:
            label_values = [label for label in LABEL_ORDER if label in display_sentiment_series(filtered[label_col]).unique()]
            selected_labels = st.multiselect("Filter sentiments", label_values, default=label_values)
            filtered = filtered[display_sentiment_series(filtered[label_col]).isin(selected_labels)]

    st.markdown("### Sample Data")
    display_frame = filtered.copy()
    if label_col:
        display_frame["sentiment_name"] = display_sentiment_series(display_frame[label_col])
    display_cols = [col for col in [lang_col, "sentiment_name", label_col, text_col, "raw_text", "score", "app_name", "reviewCreatedVersion"] if col and col in display_frame.columns]
    st.dataframe(display_frame[display_cols].head(100) if display_cols else display_frame.head(100), use_container_width=True)

    st.markdown("### Dataset Distributions")
    c5, c6 = st.columns(2)
    with c5:
        if label_col:
            st.markdown("#### Sentiment Distribution")
            st.bar_chart(display_sentiment_series(filtered[label_col]).value_counts().reindex(LABEL_ORDER).dropna())
    with c6:
        if lang_col:
            st.markdown("#### Language Distribution")
            st.bar_chart(filtered[lang_col].value_counts())


# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
def visualizations_page() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Visualizations</h1>
            <p>Required rubric charts plus additional multilingual model analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Required Charts")
    c1, c2 = st.columns(2)
    with c1:
        image_card("1. Word Cloud", "wordcloud_multilingual.png", "Most frequent words in multilingual reviews")
        image_card("2. Class Distribution", "class_distribution_multilingual.png", "Negative, neutral, and positive review counts")
        image_card("3. Confusion Matrix", "confusion_matrix_best_model.png", "Best selected model confusion matrix")
    with c2:
        image_card("4. Model Comparison by Accuracy", "model_accuracy_comparison.png", "Accuracy comparison across trained models")
        image_card("5. Top 20 Words", "top_20_words_multilingual.png", "Most frequent processed words")
        if not (IMAGES_DIR / "top_20_words_multilingual.png").exists():
            st.warning("Top 20 words image is missing. Generate it from the training script/notebook if required by the rubric.")

    st.markdown("### Optional Extra Visualizations")
    c3, c4 = st.columns(2)
    with c3:
        image_card("Language Distribution", "language_distribution.png")
        image_card("Macro-F1 Model Comparison", "model_macro_f1_comparison.png")
        image_card("N-gram Analysis: Top Bigrams", "top_15_bigrams.png")
    with c4:
        image_card("N-gram Analysis: Top Trigrams", "top_15_trigrams.png")
        image_card("GridSearch Logistic Regression Confusion Matrix", "confusion_matrix_gridsearch_logistic_regression_tf_idf.png")
        image_card("Naive Bayes Confusion Matrix", "confusion_matrix_naive_bayes_tf_idf_bigram.png")

    with st.expander("Show all available images"):
        if IMAGES_DIR.exists():
            for image_path in sorted(IMAGES_DIR.glob("*.png")):
                st.write(f"images/{image_path.name}")
        else:
            st.info("Images folder not found.")


# -----------------------------------------------------------------------------
# Model Info
# -----------------------------------------------------------------------------
def model_info_page() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Model Info</h1>
            <p>Understand the selected model, performance metrics, training details, and limitations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    best_model_name = infer_best_model_name()
    st.markdown("### Selected Model")
    st.markdown(f"**{best_model_name}**")
    st.markdown(
        """
        The deployed model is selected based on practical performance for a Streamlit app. The final decision prioritizes balanced performance, fast prediction, and ability to handle multilingual/noisy review text.
        """
    )

    st.markdown("### Performance Metrics")
    if not comparison_df.empty:
        st.dataframe(comparison_df, use_container_width=True)
    else:
        st.info("Model comparison CSV not found.")

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Accuracy", fmt_pct(infer_metric("accuracy")), "Overall correctness")
    with c2:
        render_metric_card("Macro-F1", fmt_pct(infer_metric("macro_f1")), "Balanced across classes")
    with c3:
        render_metric_card("Neutral Recall", fmt_pct(infer_metric("neutral_recall")), "Neutral-class detection")

    st.markdown("### Per-Language Metrics")
    if not language_metrics_df.empty:
        st.dataframe(language_metrics_df, use_container_width=True)
    else:
        st.info("Language metrics CSV not found.")

    st.markdown("### Training Details")
    st.markdown(
        """
        - Dataset: `Training_Data_Google_Play_reviews_6000.csv`
        - Languages: English, French, German, Italian, Japanese
        - Labels: negative, neutral, positive
        - Features: word TF-IDF, character n-grams, and combined word-character TF-IDF
        - Models compared: Naive Bayes, Logistic Regression, SGD, and optional XLM-RoBERTa Transformer
        - Evaluation metrics: Accuracy, Precision, Recall, F1-score, Macro-F1, and Neutral Recall
        """
    )

    st.markdown("### Label Mapping")
    st.table(
        pd.DataFrame(
            {
                "Review Score": ["1–2", "3", "4–5"],
                "Sentiment": ["negative", "neutral", "positive"],
            }
        )
    )

    st.markdown("### Important Limitation")
    st.markdown(
        '<div class="callout"><b>Neutral reviews remain the most difficult class.</b> This happens because neutral reviews are fewer and often ambiguous, so Macro-F1 and neutral recall are more meaningful than accuracy alone.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Show raw metadata"):
        st.json(metadata if metadata else {"message": "No metadata loaded."})


# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
if page == "🏠 Dashboard":
    dashboard_page()
elif page == "📌 Home/About":
    home_page()
elif page == "🔎 Text Analyzer":
    analyzer_page()
elif page == "📊 Data Explorer":
    data_explorer_page()
elif page == "🖼️ Visualizations":
    visualizations_page()
elif page == "🤖 Model Info":
    model_info_page()
