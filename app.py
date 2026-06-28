"""
Rubric-ready Streamlit application for SAIA2163 NLP Final Project.

Pages included according to rubric:
1. Home/About
2. Text Analyzer
3. Data Explorer
4. Visualizations
5. Model Info

Use after training with:
    python train_multilingual_models.py --data Training_Data_Google_Play_reviews_6000.csv
Then run:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
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
    # Fallback only keeps the app page readable if preprocessing_utils.py is missing.
    # The trained model still requires preprocessing_utils.py to load correctly.
    SUPPORTED_LANGUAGES = {
        "EN": "English",
        "FR": "French",
        "DE": "German",
        "IT": "Italian",
        "JP": "Japanese",
    }

    def normalize_language_code(code: str) -> str:
        code = str(code).strip().upper()
        return code if code in SUPPORTED_LANGUAGES else "EN"

    def language_name(code: str) -> str:
        return SUPPORTED_LANGUAGES.get(normalize_language_code(code), "English")

    def add_language_token(text: str, lang_code: str) -> str:
        return f"lang_{normalize_language_code(lang_code).lower()} {text}"


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multilingual App Review Sentiment AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"

MODEL_PATH = BASE_DIR / "best_sentiment_pipeline.pkl"
META_PATH = BASE_DIR / "backend_metadata.pkl"
META_JSON_PATH = BASE_DIR / "backend_metadata.json"
DATA_PATH = BASE_DIR / "preprocessed_reviews.csv"
RAW_DATA_PATH = BASE_DIR / "Training_Data_Google_Play_reviews_6000.csv"
PRED_PATH = BASE_DIR / "final_model_predictions.csv"
COMPARISON_PATH = BASE_DIR / "model_evaluation_comparison.csv"
LANGUAGE_METRICS_PATH = BASE_DIR / "multilingual_language_metrics.csv"

LABEL_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_EMOJI = {"negative": "😞", "neutral": "😐", "positive": "😊"}
SENTIMENT_COLOR = {"negative": "#ff6b6b", "neutral": "#f4d35e", "positive": "#51cf66"}

# Edit this list if your group member names should be shown differently.
TEAM_MEMBERS = [
    "Student 4 / SAIA2163 NLP Project Team",
]


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        opacity: 0.78;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        border: 1px solid rgba(120, 120, 120, 0.25);
        border-radius: 14px;
        padding: 1rem;
        background: rgba(120, 120, 120, 0.06);
    }
    .small-note {
        font-size: 0.9rem;
        opacity: 0.75;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Data/model loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model() -> Any | None:
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.error(f"Unable to load model: {exc}")
        return None


@st.cache_data
def load_metadata() -> dict[str, Any]:
    if META_PATH.exists():
        try:
            return joblib.load(META_PATH)
        except Exception:
            pass
    if META_JSON_PATH.exists():
        try:
            import json

            return json.loads(META_JSON_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()


model = load_model()
metadata = load_metadata()
df = load_csv(DATA_PATH)
raw_df = load_csv(RAW_DATA_PATH)
comparison_df = load_csv(COMPARISON_PATH)
language_metrics_df = load_csv(LANGUAGE_METRICS_PATH)
predictions_df = load_csv(PRED_PATH)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def model_input_for_prediction(text: str, lang_code: str, meta: dict[str, Any]) -> str:
    uses_language_token = bool(meta.get("uses_language_token", True))
    if uses_language_token:
        return add_language_token(text, lang_code)
    return text


def label_from_prediction(prediction: Any) -> str:
    if isinstance(prediction, str):
        return prediction
    inverse = {0: "negative", 1: "neutral", 2: "positive"}
    try:
        return inverse[int(prediction)]
    except Exception:
        return str(prediction)


def probability_dataframe(model_obj: Any, prepared_text: str) -> pd.DataFrame:
    """Return probability/confidence table if the trained model supports it."""
    if model_obj is None:
        return pd.DataFrame()

    if hasattr(model_obj, "predict_proba"):
        try:
            probs = model_obj.predict_proba([prepared_text])[0]
            classes = getattr(model_obj, "classes_", LABEL_ORDER)
            rows = []
            for cls, prob in zip(classes, probs):
                rows.append({"Sentiment": label_from_prediction(cls), "Probability": float(prob)})
            out = pd.DataFrame(rows)
            out["Sentiment"] = pd.Categorical(out["Sentiment"], categories=LABEL_ORDER, ordered=True)
            return out.sort_values("Sentiment")
        except Exception:
            return pd.DataFrame()

    # Fallback for models with decision_function but no probability.
    if hasattr(model_obj, "decision_function"):
        try:
            scores = np.array(model_obj.decision_function([prepared_text])).reshape(-1)
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            classes = getattr(model_obj, "classes_", LABEL_ORDER)
            rows = []
            for cls, prob in zip(classes, probs):
                rows.append({"Sentiment": label_from_prediction(cls), "Probability": float(prob)})
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def get_feature_names_from_pipeline(model_obj: Any) -> np.ndarray | None:
    """Try to obtain TF-IDF feature names from the saved sklearn pipeline."""
    try:
        if hasattr(model_obj, "named_steps"):
            for step_name in ["features", "tfidf", "vectorizer"]:
                if step_name in model_obj.named_steps:
                    step = model_obj.named_steps[step_name]
                    if hasattr(step, "get_feature_names_out"):
                        return step.get_feature_names_out()
    except Exception:
        return None
    return None


def get_feature_transformer_from_pipeline(model_obj: Any) -> Any | None:
    try:
        if hasattr(model_obj, "named_steps"):
            for step_name in ["features", "tfidf", "vectorizer"]:
                if step_name in model_obj.named_steps:
                    return model_obj.named_steps[step_name]
    except Exception:
        return None
    return None


def get_classifier_from_pipeline(model_obj: Any) -> Any | None:
    try:
        if hasattr(model_obj, "named_steps") and "classifier" in model_obj.named_steps:
            return model_obj.named_steps["classifier"]
    except Exception:
        return None
    return None


def influential_features(model_obj: Any, prepared_text: str, predicted_label: str, top_n: int = 12) -> pd.DataFrame:
    """
    Estimate which TF-IDF features contributed most to the selected class.

    This works best for linear models such as Logistic Regression and SGDClassifier.
    For non-linear models, the app returns a friendly fallback message.
    """
    transformer = get_feature_transformer_from_pipeline(model_obj)
    classifier = get_classifier_from_pipeline(model_obj)
    feature_names = get_feature_names_from_pipeline(model_obj)

    if transformer is None or classifier is None or feature_names is None:
        return pd.DataFrame()

    try:
        X = transformer.transform([prepared_text])
        classes = [label_from_prediction(c) for c in getattr(classifier, "classes_", LABEL_ORDER)]

        # Standard linear classifier: SGDClassifier / LogisticRegression.
        if hasattr(classifier, "coef_"):
            if predicted_label in classes:
                class_index = classes.index(predicted_label)
            else:
                class_index = int(np.argmax(model_obj.predict_proba([prepared_text])[0])) if hasattr(model_obj, "predict_proba") else 0

            coef = classifier.coef_[class_index]
            contributions = X.multiply(coef).toarray().ravel()

        # OneVsRestClassifier wrapper.
        elif hasattr(classifier, "estimators_") and hasattr(classifier.estimators_[0], "coef_"):
            if predicted_label in classes:
                class_index = classes.index(predicted_label)
            else:
                class_index = 0
            estimator = classifier.estimators_[class_index]
            coef = estimator.coef_[0]
            contributions = X.multiply(coef).toarray().ravel()
        else:
            return pd.DataFrame()

        nonzero = np.flatnonzero(contributions)
        if len(nonzero) == 0:
            return pd.DataFrame()

        top_idx = nonzero[np.argsort(contributions[nonzero])[-top_n:]][::-1]
        rows = []
        for idx in top_idx:
            value = float(contributions[idx])
            if value <= 0:
                continue
            feature = str(feature_names[idx]).replace("word__", "").replace("char__", "")
            rows.append({"Influential word/feature": feature, "Contribution": value})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def show_image_if_exists(filename: str, caption: str | None = None, use_container_width: bool = True) -> bool:
    path = IMAGES_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption or filename, use_container_width=use_container_width)
        return True
    st.info(f"Image not found: images/{filename}")
    return False


def plot_pie_from_counts(counts: pd.Series, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="pie", autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    ax.set_title(title)
    st.pyplot(fig)


def display_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="main-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🌍 NLP App Menu")
    page = st.radio(
        "Go to section",
        [
            "1. Home/About",
            "2. Text Analyzer",
            "3. Data Explorer",
            "4. Visualizations",
            "5. Model Info",
        ],
    )

    st.divider()
    st.subheader("System Status")
    st.write("Model file:", "✅" if MODEL_PATH.exists() else "❌")
    st.write("Preprocessed data:", "✅" if DATA_PATH.exists() else "❌")
    st.write("Evaluation CSV:", "✅" if COMPARISON_PATH.exists() else "❌")
    st.write("Language metrics:", "✅" if LANGUAGE_METRICS_PATH.exists() else "❌")

    if metadata:
        st.divider()
        st.subheader("Selected Model")
        st.write(metadata.get("best_model_name", "Unknown model"))
        best_metrics = metadata.get("best_metrics", {})
        if isinstance(best_metrics, dict) and best_metrics:
            if "accuracy" in best_metrics:
                st.metric("Accuracy", f"{best_metrics['accuracy']:.4f}")
            if "macro_f1" in best_metrics:
                st.metric("Macro-F1", f"{best_metrics['macro_f1']:.4f}")


# -----------------------------------------------------------------------------
# 1. Home/About
# -----------------------------------------------------------------------------
if page == "1. Home/About":
    display_header(
        "🌍 Multilingual App Review Sentiment AI",
        "A Streamlit NLP application for multilingual Google Play review sentiment analysis.",
    )

    st.subheader("Project Title and Description")
    st.write(
        "This project classifies Google Play app reviews into **negative**, **neutral**, "
        "and **positive** sentiment. The improved version supports multilingual reviews "
        "from English, French, German, Italian, and Japanese datasets."
    )

    st.subheader("Problem Being Solved")
    st.write(
        "App developers receive many reviews in different languages. Manually reading "
        "and interpreting all reviews is slow and difficult. This app helps users quickly "
        "understand whether a review expresses negative, neutral, or positive sentiment."
    )

    st.subheader("How to Use the App")
    st.markdown(
        """
        1. Go to **Text Analyzer**.
        2. Enter or paste an app review.
        3. Select the review language.
        4. Click **Analyze Review**.
        5. View the predicted sentiment, confidence score, and influential words/features.
        6. Use **Data Explorer**, **Visualizations**, and **Model Info** to understand the dataset and model performance.
        """
    )

    st.subheader("Team Members")
    for member in TEAM_MEMBERS:
        st.write(f"- {member}")
    st.caption("Edit the TEAM_MEMBERS list near the top of app.py if you need to add exact group member names.")

    st.subheader("Supported Languages")
    lang_cols = st.columns(len(SUPPORTED_LANGUAGES))
    for col, (code, name) in zip(lang_cols, SUPPORTED_LANGUAGES.items()):
        col.metric(code, name)


# -----------------------------------------------------------------------------
# 2. Text Analyzer
# -----------------------------------------------------------------------------
elif page == "2. Text Analyzer":
    display_header(
        "🧪 Text Analyzer",
        "Enter a multilingual app review and analyze its sentiment using the trained model.",
    )

    if model is None:
        st.error(
            "Model file not found or could not be loaded. Run training first: "
            "`python train_multilingual_models.py --data Training_Data_Google_Play_reviews_6000.csv`"
        )
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        review_text = st.text_area(
            "Review text",
            height=170,
            placeholder="Example: This app is useful, but it crashes sometimes.",
        )
    with col2:
        language_options = {code: f"{code} - {name}" for code, name in SUPPORTED_LANGUAGES.items()}
        selected_lang = st.selectbox(
            "Review language",
            options=list(language_options.keys()),
            format_func=lambda code: language_options[code],
            index=0,
        )
        st.info(
            "The model uses a language token, so selecting the correct language helps prediction."
        )

    if st.button("Analyze Review", type="primary", use_container_width=True):
        if not review_text.strip():
            st.warning("Please enter a review first.")
        else:
            lang_code = normalize_language_code(selected_lang)
            prepared_text = model_input_for_prediction(review_text, lang_code, metadata)
            prediction = model.predict([prepared_text])[0]
            pred_label = label_from_prediction(prediction)

            prob_df = probability_dataframe(model, prepared_text)
            confidence = None
            if not prob_df.empty:
                match = prob_df[prob_df["Sentiment"].astype(str) == pred_label]
                if not match.empty:
                    confidence = float(match["Probability"].iloc[0])

            result_col, conf_col = st.columns(2)
            result_col.metric(
                "Prediction Result",
                f"{SENTIMENT_EMOJI.get(pred_label, '')} {pred_label.title()}",
            )
            if confidence is not None:
                conf_col.metric("Confidence Score", f"{confidence * 100:.2f}%")
            else:
                conf_col.metric("Confidence Score", "Not available")

            if not prob_df.empty:
                st.subheader("Confidence / Probability Score")
                chart_df = prob_df.copy()
                chart_df["Probability"] = chart_df["Probability"].astype(float)
                st.bar_chart(chart_df.set_index("Sentiment"))
                st.dataframe(chart_df, use_container_width=True, hide_index=True)

            st.subheader("Words/Features That Influenced the Prediction")
            influence_df = influential_features(model, prepared_text, pred_label)
            if not influence_df.empty:
                st.dataframe(influence_df, use_container_width=True, hide_index=True)
                st.caption(
                    "These are the strongest positive TF-IDF feature contributions toward the predicted class. "
                    "Character n-grams may appear because the multilingual model uses character-level features."
                )
            else:
                st.info(
                    "Feature contribution explanation is not available for this model structure. "
                    "The prediction result and confidence score are still valid."
                )

            with st.expander("Prepared model input"):
                st.code(prepared_text)

    st.divider()
    st.subheader("Try Sample Reviews")
    samples = pd.DataFrame(
        [
            {"Language": "EN", "Review": "This app is very useful and easy to use."},
            {"Language": "EN", "Review": "The app keeps crashing and I cannot login."},
            {"Language": "FR", "Review": "Cette application est très utile."},
            {"Language": "DE", "Review": "Die App funktioniert nicht richtig."},
            {"Language": "IT", "Review": "Applicazione buona ma lenta a volte."},
            {"Language": "JP", "Review": "このアプリは便利ですが、時々遅いです。"},
        ]
    )
    st.dataframe(samples, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# 3. Data Explorer
# -----------------------------------------------------------------------------
elif page == "3. Data Explorer":
    display_header(
        "📊 Data Explorer",
        "Explore sample data, dataset statistics, and distribution charts.",
    )

    data_to_show = df if not df.empty else raw_df
    data_name = "preprocessed_reviews.csv" if not df.empty else "raw dataset"

    if data_to_show.empty:
        st.warning("No dataset file was found. Please check the CSV files in your project folder.")
        st.stop()

    st.subheader("Dataset Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(data_to_show):,}")
    c2.metric("Columns", f"{len(data_to_show.columns):,}")
    if "userLang" in data_to_show.columns:
        c3.metric("Languages", f"{data_to_show['userLang'].nunique():,}")
    else:
        c3.metric("Languages", "N/A")
    if "label" in data_to_show.columns:
        c4.metric("Sentiment Classes", f"{data_to_show['label'].nunique():,}")
    else:
        c4.metric("Sentiment Classes", "N/A")

    st.subheader("Sample Data")
    st.caption(f"Showing data from {data_name}")
    preferred_cols = [
        "userLang",
        "language_name",
        "score",
        "label",
        "raw_text",
        "text",
        "app_name",
        "predicted_label",
    ]
    show_cols = [c for c in preferred_cols if c in data_to_show.columns]
    if not show_cols:
        show_cols = list(data_to_show.columns[:8])
    st.dataframe(data_to_show[show_cols].head(100), use_container_width=True)

    st.subheader("Data Distribution")
    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        if "label" in data_to_show.columns:
            st.write("Sentiment Distribution - Bar Chart")
            sentiment_counts = data_to_show["label"].value_counts().reindex(LABEL_ORDER).fillna(0)
            st.bar_chart(sentiment_counts)
        else:
            st.info("No `label` column found for sentiment distribution.")

    with dist_col2:
        if "label" in data_to_show.columns:
            st.write("Sentiment Distribution - Pie Chart")
            plot_pie_from_counts(data_to_show["label"].value_counts(), "Sentiment Distribution")
        else:
            st.info("No `label` column found for pie chart.")

    lang_col1, lang_col2 = st.columns(2)
    with lang_col1:
        if "userLang" in data_to_show.columns:
            st.write("Language Distribution")
            st.bar_chart(data_to_show["userLang"].value_counts().sort_index())
    with lang_col2:
        if "score" in data_to_show.columns:
            st.write("Review Score Distribution")
            st.bar_chart(data_to_show["score"].value_counts().sort_index())


# -----------------------------------------------------------------------------
# 4. Visualizations
# -----------------------------------------------------------------------------
elif page == "4. Visualizations":
    display_header(
        "📈 Visualizations",
        "View word clouds, class distributions, confusion matrices, and model comparison charts.",
    )

    st.subheader("Word Cloud of Most Common Words")
    show_image_if_exists("wordcloud_multilingual.png", "Multilingual word cloud")

    st.subheader("Bar Chart of Label Distribution")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        show_image_if_exists("class_distribution_multilingual.png", "Sentiment class distribution")
    with img_col2:
        show_image_if_exists("language_distribution.png", "Language distribution")

    st.subheader("Confusion Matrix Heatmap")
    show_image_if_exists("confusion_matrix_best_model.png", "Best model confusion matrix")

    st.subheader("Model Comparison Charts")
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        show_image_if_exists("model_accuracy_comparison.png", "Model comparison by accuracy")
    with comp_col2:
        show_image_if_exists("model_macro_f1_comparison.png", "Model comparison by Macro-F1")

    st.subheader("Other Relevant Charts")
    extra_images = [
        "confusion_matrix_logistic_regression_tf_idf_unigram.png",
        "confusion_matrix_logistic_regression_tf_idf_bigram_balanced.png",
        "confusion_matrix_logistic_regression_character_n_grams.png",
        "confusion_matrix_gridsearch_logistic_regression_tf_idf.png",
        "confusion_matrix_naive_bayes_tf_idf_bigram.png",
    ]
    existing_extra = [img for img in extra_images if (IMAGES_DIR / img).exists()]
    if existing_extra:
        selected_img = st.selectbox("Select additional chart", existing_extra)
        show_image_if_exists(selected_img, selected_img)
    else:
        st.info("No additional model-specific charts found.")
    st.subheader("N-gram Analysis")
    if Path("images/top_15_bigrams.png").exists():
        st.image("images/top_15_bigrams.png", caption="Top 15 Most Frequent Bigrams")
    else:
        st.info("Bigram visualization not found.")
    if Path("images/top_15_trigrams.png").exists():
        st.image("images/top_15_trigrams.png", caption="Top 15 Most Frequent Trigrams")


# -----------------------------------------------------------------------------
# 5. Model Info
# -----------------------------------------------------------------------------
elif page == "5. Model Info":
    display_header(
        "🤖 Model Info",
        "Understand the models, performance metrics, and training details.",
    )

    st.subheader("Model Explanation")
    st.write(
        "The project compares multiple multilingual sentiment classification models. "
        "The deployed Streamlit model uses a saved Scikit-learn pipeline that combines "
        "multilingual-safe preprocessing with TF-IDF features and a supervised classifier."
    )

    st.markdown(
        """
        **Models compared:**
        - Multilingual Naive Bayes Word TF-IDF
        - Multilingual Logistic Regression Word TF-IDF
        - Multilingual SGD Character N-grams
        - Multilingual SGD Word + Character TF-IDF
        - Optional XLM-RoBERTa Transformer experiment

        **Why Macro-F1 matters:**  
        Accuracy can look high even when the model fails on the minority neutral class. Macro-F1 treats negative, neutral, and positive classes more equally, making it a fairer metric for this dataset.
        """
    )

    st.subheader("Performance Metrics")
    if not comparison_df.empty:
        metric_cols = [
            c
            for c in [
                "model_name",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "macro_f1",
                "neutral_recall",
            ]
            if c in comparison_df.columns
        ]
        st.dataframe(comparison_df[metric_cols], use_container_width=True)
    else:
        st.info("Model comparison CSV not found. Train the models to generate performance metrics.")

    st.subheader("Best Model Summary")
    best_model_name = metadata.get("best_model_name", "Unknown")
    st.write(f"**Selected model:** {best_model_name}")

    best_metrics = metadata.get("best_metrics", {})
    if isinstance(best_metrics, dict) and best_metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{best_metrics.get('accuracy', 0):.4f}")
        m2.metric("F1-score", f"{best_metrics.get('f1_score', 0):.4f}")
        m3.metric("Macro-F1", f"{best_metrics.get('macro_f1', 0):.4f}")
        m4.metric("Neutral Recall", f"{best_metrics.get('neutral_recall', 0):.4f}")

    st.subheader("Training Details")
    st.markdown(
        """
        - Dataset: `Training_Data_Google_Play_reviews_6000.csv`
        - Languages used: English, French, German, Italian, and Japanese
        - Labels: negative, neutral, positive
        - Feature types: word TF-IDF and character n-gram TF-IDF
        - Class imbalance handling: balanced classifier settings where applicable
        - Evaluation outputs: model comparison CSV, prediction CSV, language-level metrics, and visualization images
        """
    )

    if not language_metrics_df.empty:
        st.subheader("Performance by Language")
        show_cols = [
            c
            for c in [
                "language",
                "language_name",
                "rows",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "macro_f1",
                "neutral_recall",
            ]
            if c in language_metrics_df.columns
        ]
        st.dataframe(language_metrics_df[show_cols], use_container_width=True)

    with st.expander("Saved model and data files"):
        files = [
            MODEL_PATH,
            META_PATH,
            META_JSON_PATH,
            DATA_PATH,
            PRED_PATH,
            COMPARISON_PATH,
            LANGUAGE_METRICS_PATH,
        ]
        file_df = pd.DataFrame(
            [
                {"File": f.name, "Exists": f.exists(), "Path": str(f.relative_to(BASE_DIR)) if f.exists() else f.name}
                for f in files
            ]
        )
        st.dataframe(file_df, use_container_width=True, hide_index=True)
