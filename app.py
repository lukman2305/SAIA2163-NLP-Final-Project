import os
import re
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud


# =============================
# Page configuration
# =============================
st.set_page_config(
    page_title="App Review Sentiment Analyzer",
    page_icon="📱",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_sentiment_pipeline.pkl")
META_PATH = os.path.join(BASE_DIR, "backend_metadata.pkl")
DATA_PATH = os.path.join(BASE_DIR, "preprocessed_reviews.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "Training_Data_Google_Play_reviews_6000.csv")
PRED_PATH = os.path.join(BASE_DIR, "final_model_predictions.csv")


# =============================
# Helper functions
# =============================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    return joblib.load(META_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    return df


@st.cache_data
def load_raw_data():
    if os.path.exists(RAW_DATA_PATH):
        return pd.read_csv(RAW_DATA_PATH)
    return pd.DataFrame()


@st.cache_data
def load_predictions():
    if os.path.exists(PRED_PATH):
        return pd.read_csv(PRED_PATH)
    return pd.DataFrame()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_sentiment(text, model, label_mapping):
    cleaned = clean_text(text)
    pred = model.predict([cleaned])[0]

    # Convert numeric prediction to label if needed
    inverse_mapping = {v: k for k, v in label_mapping.items()}
    label = inverse_mapping.get(pred, pred)

    confidence = None
    probabilities = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([cleaned])[0]
        classes = model.classes_
        probabilities = {}
        for cls, p in zip(classes, proba):
            probabilities[inverse_mapping.get(cls, cls)] = float(p)
        confidence = float(np.max(proba))

    return cleaned, label, confidence, probabilities


def get_influential_words(text, model, predicted_label, top_n=10):
    try:
        vectorizer = model.named_steps.get("tfidf")
        classifier = model.named_steps.get("classifier")
        if vectorizer is None or classifier is None or not hasattr(classifier, "coef_"):
            return []

        cleaned = clean_text(text)
        feature_names = np.array(vectorizer.get_feature_names_out())
        transformed = vectorizer.transform([cleaned])
        active_indices = transformed.nonzero()[1]

        if len(active_indices) == 0:
            return []

        classes = list(classifier.classes_)
        if predicted_label in classes:
            class_index = classes.index(predicted_label)
        else:
            class_index = int(np.argmax(classifier.predict_proba([cleaned])[0]))

        coef = classifier.coef_[class_index]
        word_scores = []
        for idx in active_indices:
            word_scores.append((feature_names[idx], coef[idx]))

        word_scores = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)
        return word_scores[:top_n]
    except Exception:
        return []


def plot_label_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    order = df["label"].value_counts().index
    sns.countplot(data=df, x="label", order=order, ax=ax)
    ax.set_title("Sentiment Class Distribution")
    ax.set_xlabel("Sentiment Label")
    ax.set_ylabel("Number of Reviews")
    return fig


def plot_wordcloud(df, label_filter="All"):
    if label_filter != "All":
        text_data = " ".join(df[df["label"] == label_filter]["text"].dropna().astype(str))
    else:
        text_data = " ".join(df["text"].dropna().astype(str))

    if not text_data.strip():
        text_data = "no text available"

    wc = WordCloud(width=1000, height=450, background_color="white", max_words=120).generate(text_data)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud: {label_filter}")
    return fig


def plot_confusion_matrix(metadata):
    cm = np.array(metadata.get("confusion_matrix", []))
    label_mapping = metadata.get("label_mapping", {})
    labels = [label for label, _ in sorted(label_mapping.items(), key=lambda x: x[1])]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix - Best Model")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    return fig


def plot_model_comparison(pred_df):
    model_cols = [c for c in pred_df.columns if c.startswith("sentiment_")]
    rows = []
    for col in model_cols:
        accuracy = (pred_df[col] == pred_df["true_sentiment"]).mean()
        model_name = col.replace("sentiment_", "").replace("_", " ").title()
        rows.append({"Model": model_name, "Accuracy": accuracy})

    comparison_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=comparison_df, x="Model", y="Accuracy", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f")
    return fig, comparison_df


def plot_top_words(df, top_n=20):
    all_words = " ".join(df["text"].dropna().astype(str)).split()
    counts = Counter(all_words).most_common(top_n)
    top_df = pd.DataFrame(counts, columns=["Word", "Frequency"])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=top_df, y="Word", x="Frequency", ax=ax)
    ax.set_title(f"Top {top_n} Most Common Words")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Word")
    return fig, top_df


def plot_text_length(df):
    temp = df.copy()
    temp["text_length"] = temp["text"].fillna("").astype(str).apply(lambda x: len(x.split()))
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(temp["text_length"], bins=30, kde=True, ax=ax)
    ax.set_title("Review Text Length Distribution")
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Number of Reviews")
    return fig


# =============================
# Load files
# =============================
try:
    model = load_model()
    metadata = load_metadata()
    df = load_data()
    raw_df = load_raw_data()
    pred_df = load_predictions()
except Exception as e:
    st.error("Required project files could not be loaded. Make sure app.py is in the same folder as the model, metadata and CSV files.")
    st.exception(e)
    st.stop()

label_mapping = metadata.get("label_mapping", {})


# =============================
# Sidebar
# =============================
st.sidebar.title("📱 NLP App")
st.sidebar.caption("Google Play Review Sentiment Analyzer")
page = st.sidebar.radio(
    "Navigation",
    ["Home / About", "Text Analyzer", "Data Explorer", "Visualizations", "Model Info"],
)


# =============================
# Home Page
# =============================
if page == "Home / About":
    st.title("📱 App Review Sentiment Analyzer")
    st.write(
        "This Streamlit application analyzes Google Play Store app reviews and predicts whether a review is **positive**, **neutral**, or **negative**."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Used", f"{len(df):,} reviews")
    col2.metric("Best Model", metadata.get("best_model_name", "N/A"))
    col3.metric("Accuracy", f"{metadata.get('metrics', {}).get('accuracy', 0):.2%}")

    st.subheader("Problem Statement")
    st.write(
        "Mobile app developers receive many user reviews every day. Manually reading all reviews is time-consuming, so this system helps classify review sentiment automatically and gives quick insights from the dataset."
    )

    st.subheader("How to Use")
    st.write("Go to **Text Analyzer**, paste an app review, then click **Analyze Sentiment** to get the prediction and confidence score.")

    st.subheader("Team AKATSUKI")
    st.write("1. Muhammad Lukman Bin Nasrum  \n2. Hasnawi Imran Bin Mohd Saidi  \n3. Muhammad Zahin Bin Mohd Zamri  \n4. Raqib Hazim Bin Abdul Hamid")


# =============================
# Text Analyzer Page
# =============================
elif page == "Text Analyzer":
    st.title("🔍 Text Analyzer")
    st.write("Enter a Google Play Store review below and the model will predict its sentiment.")

    sample_reviews = {
        "Positive sample": "This app is very useful and easy to use. The interface is clean and fast.",
        "Negative sample": "The app keeps crashing after the update and the login does not work.",
        "Neutral sample": "The app is okay. Some features work well but others still need improvement.",
    }

    selected_sample = st.selectbox("Try a sample review", ["Write my own"] + list(sample_reviews.keys()))
    default_text = "" if selected_sample == "Write my own" else sample_reviews[selected_sample]

    user_text = st.text_area("Review text", value=default_text, height=170)

    if st.button("Analyze Sentiment", type="primary"):
        if not user_text.strip():
            st.warning("Please enter a review first.")
        else:
            cleaned, label, confidence, probabilities = predict_sentiment(user_text, model, label_mapping)

            st.subheader("Prediction Result")
            st.success(f"Predicted sentiment: **{str(label).upper()}**")

            if confidence is not None:
                st.metric("Confidence Score", f"{confidence:.2%}")

            if probabilities:
                prob_df = pd.DataFrame({"Sentiment": list(probabilities.keys()), "Probability": list(probabilities.values())})
                st.bar_chart(prob_df.set_index("Sentiment"))
                st.dataframe(prob_df, use_container_width=True)

            st.subheader("Cleaned Text")
            st.code(cleaned)

            st.subheader("Influential Words")
            important_words = get_influential_words(user_text, model, label)
            if important_words:
                imp_df = pd.DataFrame(important_words, columns=["Word / Phrase", "Model Weight"])
                st.dataframe(imp_df, use_container_width=True)
            else:
                st.info("Influential words are not available for this input/model.")


# =============================
# Data Explorer Page
# =============================
elif page == "Data Explorer":
    st.title("📊 Data Explorer")

    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Labels", df["label"].nunique())
    col4.metric("Apps", df["app_name"].nunique() if "app_name" in df.columns else "N/A")

    st.subheader("Sample Preprocessed Data")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Dataset Statistics")
    label_counts = df["label"].value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]
    st.dataframe(label_counts, use_container_width=True)

    if "app_name" in df.columns:
        st.subheader("Reviews by App")
        app_counts = df["app_name"].value_counts().reset_index()
        app_counts.columns = ["App Name", "Count"]
        st.dataframe(app_counts, use_container_width=True)

    if not raw_df.empty:
        with st.expander("Show raw dataset sample"):
            st.dataframe(raw_df.head(30), use_container_width=True)

    st.pyplot(plot_label_distribution(df))


# =============================
# Visualizations Page
# =============================
elif page == "Visualizations":
    st.title("📈 Visualizations")
    st.write("This section shows dataset insights and model performance charts.")

    st.subheader("1. Word Cloud")
    label_options = ["All"] + sorted(df["label"].dropna().unique().tolist())
    selected_label = st.selectbox("Choose sentiment for word cloud", label_options)
    st.pyplot(plot_wordcloud(df, selected_label))

    st.subheader("2. Class Distribution")
    st.pyplot(plot_label_distribution(df))

    st.subheader("3. Confusion Matrix")
    st.pyplot(plot_confusion_matrix(metadata))

    st.subheader("4. Model Comparison")
    if not pred_df.empty:
        fig, comparison_df = plot_model_comparison(pred_df)
        st.pyplot(fig)
        st.dataframe(comparison_df, use_container_width=True)
    else:
        st.info("Prediction comparison file not found.")

    st.subheader("5. Top 20 Words")
    fig, top_df = plot_top_words(df)
    st.pyplot(fig)
    st.dataframe(top_df, use_container_width=True)

    st.subheader("Extra: Text Length Distribution")
    st.pyplot(plot_text_length(df))


# =============================
# Model Info Page
# =============================
elif page == "Model Info":
    st.title("🤖 Model Info")

    st.subheader("Best Model")
    st.write(f"**{metadata.get('best_model_name', 'N/A')}**")

    st.subheader("Pipeline Structure")
    st.code(str(model))

    st.subheader("Performance Metrics")
    metrics = metadata.get("metrics", {})
    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    metric_cols[1].metric("Precision", f"{metrics.get('precision', 0):.2%}")
    metric_cols[2].metric("Recall", f"{metrics.get('recall', 0):.2%}")
    metric_cols[3].metric("F1-score", f"{metrics.get('f1_score', 0):.2%}")

    st.subheader("Label Mapping")
    st.json(label_mapping)

    st.subheader("Training Details")
    st.write(
        "The final model uses TF-IDF feature extraction with unigram and bigram features, followed by Logistic Regression for sentiment classification. The output labels are negative, neutral, and positive."
    )

    st.subheader("Confusion Matrix")
    st.pyplot(plot_confusion_matrix(metadata))
