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
    page_title="App Review Sentiment AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_sentiment_pipeline.pkl")
META_PATH = os.path.join(BASE_DIR, "backend_metadata.pkl")
DATA_PATH = os.path.join(BASE_DIR, "preprocessed_reviews.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "Training_Data_Google_Play_reviews_6000.csv")
PRED_PATH = os.path.join(BASE_DIR, "final_model_predictions.csv")


# =============================
# Bold rose luxury minimalist theme
# =============================
LUXURY_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #FFE0E9;
    --card: #FFFFFF;
    --text: #434343;
    --muted: #6F5D64;
    --line: #E8C8D1;
    --blue: #B9375E;
    --blue-soft: #FFE0E9;
    --green: #5F7A44;
    --green-soft: #CEDDBB;
    --amber: #BE9A60;
    --amber-soft: #F6E6C7;
    --red: #B9375E;
    --red-soft: #FFE0E9;
    --shadow: 0 14px 34px rgba(67, 67, 67, 0.12);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #FFE0E9;
}

.block-container {
    padding-top: 2.1rem;
    padding-bottom: 3rem;
    max-width: 1320px;
}

[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.84);
    border-right: 1px solid rgba(229, 231, 235, 0.9);
    backdrop-filter: blur(18px);
}

[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif;
}

.sidebar-brand {
    padding: 12px 8px 18px 8px;
}

.brand-mark {
    width: 42px;
    height: 42px;
    border-radius: 14px;
    background: #B9375E;
    color: white;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    margin-right: 10px;
    box-shadow: 0 12px 24px rgba(185, 55, 94, 0.22);
}

.brand-title {
    font-size: 1.05rem;
    color: var(--text);
    font-weight: 800;
}

.brand-subtitle {
    color: var(--muted);
    font-size: 0.82rem;
    margin-top: 4px;
}

h1, h2, h3 {
    color: var(--text);
    letter-spacing: -0.04em;
}

p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
    color: var(--muted);
}

.hero-card {
    background: #FFFFFF;
    border: 1px solid rgba(229, 231, 235, 0.95);
    box-shadow: var(--shadow);
    border-radius: 30px;
    padding: 34px 38px;
    margin-bottom: 24px;
}

.hero-kicker {
    color: var(--blue);
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.hero-title {
    font-size: 3.05rem;
    line-height: 1.02;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 1.08rem;
    color: var(--muted);
    max-width: 760px;
    line-height: 1.7;
}

.lux-card {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(229, 231, 235, 0.98);
    border-radius: 24px;
    box-shadow: 0 12px 35px rgba(15, 23, 42, 0.055);
    padding: 24px;
    margin-bottom: 18px;
}

.card-title {
    color: var(--text);
    font-size: 1.05rem;
    font-weight: 750;
    margin-bottom: 8px;
}

.card-muted {
    color: var(--muted);
    font-size: 0.93rem;
    line-height: 1.6;
}

.kpi-card {
    background: rgba(255,255,255,0.96);
    border: 1px solid rgba(229,231,235,0.95);
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.055);
    min-height: 128px;
}

.kpi-label {
    color: var(--muted);
    font-size: 0.82rem;
    font-weight: 650;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.kpi-value {
    color: var(--text);
    font-size: 2.05rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    margin-top: 12px;
}

.kpi-note {
    color: var(--muted);
    font-size: 0.86rem;
    margin-top: 4px;
}

.result-card {
    background: rgba(255,255,255,0.96);
    border: 1px solid rgba(229,231,235,0.95);
    border-radius: 26px;
    padding: 26px;
    box-shadow: var(--shadow);
    margin-top: 16px;
}

.result-label {
    font-size: 2.3rem;
    font-weight: 850;
    letter-spacing: -0.04em;
    margin: 4px 0 8px 0;
}

.pill {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.82rem;
}

.pos { color: var(--green); background: var(--green-soft); }
.neu { color: var(--amber); background: var(--amber-soft); }
.neg { color: var(--red); background: var(--red-soft); }
.blue { color: var(--blue); background: var(--blue-soft); }

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.95);
    border: 1px solid rgba(229, 231, 235, 0.95);
    border-radius: 22px;
    padding: 18px 20px;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.055);
}

[data-testid="stMetricLabel"] p {
    color: var(--muted) !important;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-size: 0.78rem;
}

[data-testid="stMetricValue"] {
    color: var(--text);
    font-weight: 800;
}

.stButton > button {
    width: 100%;
    border-radius: 16px;
    min-height: 50px;
    border: 0;
    background: #B9375E;
    color: white;
    font-weight: 750;
    box-shadow: 0 12px 24px rgba(185, 55, 94, 0.22);
}

.stButton > button:hover {
    background: #434343;
    color: white;
    border: 0;
}

.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div,
.stTextInput input {
    border-radius: 16px !important;
    border-color: var(--line) !important;
    background-color: rgba(255,255,255,0.92) !important;
}

[data-testid="stDataFrame"] {
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid var(--line);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    background: white;
    border: 1px solid var(--line);
    padding: 10px 18px;
}

hr {
    border: none;
    height: 1px;
    background: var(--line);
    margin: 1.5rem 0;
}
</style>
"""
st.markdown(LUXURY_CSS, unsafe_allow_html=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "#E8C8D1",
    "axes.labelcolor": "#6F5D64",
    "xtick.color": "#6F5D64",
    "ytick.color": "#6F5D64",
    "text.color": "#434343",
    "axes.titleweight": "bold",
})


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


def sentiment_style(label):
    label_text = str(label).lower()
    if "pos" in label_text:
        return "#5F7A44", "#CEDDBB", "Positive", "pos"
    if "neu" in label_text:
        return "#BE9A60", "#F6E6C7", "Neutral", "neu"
    if "neg" in label_text:
        return "#B9375E", "#FFE0E9", "Negative", "neg"
    return "#5F7A44", "#CEDDBB", str(label).title(), "blue"


def predict_sentiment(text, model, label_mapping):
    cleaned = clean_text(text)
    pred = model.predict([cleaned])[0]

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
        class_index = 0
        if predicted_label in classes:
            class_index = classes.index(predicted_label)
        elif hasattr(classifier, "predict_proba"):
            class_index = int(np.argmax(classifier.predict_proba([cleaned])[0]))

        coef = classifier.coef_[class_index]
        word_scores = [(feature_names[idx], coef[idx]) for idx in active_indices]
        word_scores = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)
        return word_scores[:top_n]
    except Exception:
        return []


def build_kpi(label, value, note=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_card(title, description=""):
    desc_html = f'<div class="card-muted">{description}</div>' if description else ""
    st.markdown(
        f"""
        <div class="lux-card">
            <div class="card-title">{title}</div>
            {desc_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_fig_clean(fig, ax):
    ax.grid(axis="y", color="#E8C8D1", linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E8C8D1")
    ax.spines["bottom"].set_color("#E8C8D1")
    fig.tight_layout()
    return fig


def plot_label_distribution(df):
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    order = df["label"].value_counts().index
    palette = [sentiment_style(x)[0] for x in order]
    sns.countplot(data=df, x="label", order=order, ax=ax, palette=palette)
    ax.set_title("Sentiment Class Distribution", fontsize=14, pad=14)
    ax.set_xlabel("Sentiment Label")
    ax.set_ylabel("Number of Reviews")
    return make_fig_clean(fig, ax)


def plot_wordcloud(df, label_filter="All"):
    if label_filter != "All":
        text_data = " ".join(df[df["label"] == label_filter]["text"].dropna().astype(str))
    else:
        text_data = " ".join(df["text"].dropna().astype(str))

    if not text_data.strip():
        text_data = "no text available"

    wc = WordCloud(
        width=1200,
        height=520,
        background_color="#FFE0E9",
        max_words=120,
        colormap="viridis",
        contour_width=0,
    ).generate(text_data)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud: {label_filter}", fontsize=14, pad=12)
    return fig


def plot_confusion_matrix(metadata):
    cm = np.array(metadata.get("confusion_matrix", []))
    label_mapping = metadata.get("label_mapping", {})
    labels = [label for label, _ in sorted(label_mapping.items(), key=lambda x: x[1])]

    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="BuPu",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False,
        linewidths=1,
        linecolor="#E8C8D1",
        square=True,
    )
    ax.set_title("Confusion Matrix - Best Model", fontsize=14, pad=14)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    return fig


def plot_model_comparison(pred_df):
    model_cols = [c for c in pred_df.columns if c.startswith("sentiment_")]
    rows = []
    for col in model_cols:
        accuracy = (pred_df[col] == pred_df["true_sentiment"]).mean()
        model_name = col.replace("sentiment_", "").replace("_", " ").title()
        rows.append({"Model": model_name, "Accuracy": accuracy})

    comparison_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sns.barplot(data=comparison_df, x="Model", y="Accuracy", ax=ax, color="#5F7A44")
    ax.set_ylim(0, 1)
    ax.set_title("Model Accuracy Comparison", fontsize=14, pad=14)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=4)
    return make_fig_clean(fig, ax), comparison_df


def plot_top_words(df, top_n=20):
    all_words = " ".join(df["text"].dropna().astype(str)).split()
    counts = Counter(all_words).most_common(top_n)
    top_df = pd.DataFrame(counts, columns=["Word", "Frequency"])

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    sns.barplot(data=top_df, y="Word", x="Frequency", ax=ax, color="#434343")
    ax.set_title(f"Top {top_n} Most Common Words", fontsize=14, pad=14)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Word")
    return make_fig_clean(fig, ax), top_df


def plot_text_length(df):
    temp = df.copy()
    temp["text_length"] = temp["text"].fillna("").astype(str).apply(lambda x: len(x.split()))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    sns.histplot(temp["text_length"], bins=30, kde=True, ax=ax, color="#5F7A44")
    ax.set_title("Review Text Length Distribution", fontsize=14, pad=14)
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Number of Reviews")
    return make_fig_clean(fig, ax)


def probability_table(probabilities):
    prob_df = pd.DataFrame({"Sentiment": list(probabilities.keys()), "Probability": list(probabilities.values())})
    prob_df["Probability (%)"] = (prob_df["Probability"] * 100).round(2)
    return prob_df


# =============================
# Load project files
# =============================
try:
    model = load_model()
    metadata = load_metadata()
    df = load_data()
    raw_df = load_raw_data()
    pred_df = load_predictions()
except Exception as e:
    st.error("Required project files could not be loaded. Make sure this app file is in the same folder as the model, metadata and CSV files.")
    st.exception(e)
    st.stop()

label_mapping = metadata.get("label_mapping", {})
metrics = metadata.get("metrics", {})
accuracy = metrics.get("accuracy", 0)
label_counts = df["label"].value_counts() if "label" in df.columns else pd.Series(dtype=int)
total_reviews = len(df)


# =============================
# Sidebar navigation
# =============================
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div style="display:flex;align-items:center;">
            <div class="brand-mark">AI</div>
            <div>
                <div class="brand-title">Sentiment AI</div>
                <div class="brand-subtitle">Google Play Review Analytics</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Sentiment Analyzer", "Data Explorer", "Visualizations", "Model Performance"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div class="lux-card" style="padding:18px;box-shadow:none;">
        <div class="card-title">Model Status</div>
        <div class="card-muted">Best model</div>
        <div style="color:#434343;font-weight:800;margin-top:6px;">{metadata.get('best_model_name', 'N/A')}</div>
        <div style="margin-top:12px;"><span class="pill blue">Accuracy {accuracy:.2%}</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================
# Dashboard Page
# =============================
if page == "Dashboard":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">NLP Sentiment Intelligence</div>
            <div class="hero-title">App Review Sentiment AI</div>
            <div class="hero-subtitle">
                A luxury minimalist dashboard for analyzing Google Play Store reviews,
                predicting user sentiment, and presenting model insights clearly.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        build_kpi("Total Reviews", f"{total_reviews:,}", "Preprocessed app reviews")
    with col2:
        build_kpi("Best Model", metadata.get("best_model_name", "N/A"), "Selected final pipeline")
    with col3:
        build_kpi("Accuracy", f"{accuracy:.2%}", "Final evaluation score")
    with col4:
        build_kpi("Classes", f"{df['label'].nunique() if 'label' in df.columns else 'N/A'}", "Positive, neutral, negative")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<div class="lux-card"><div class="card-title">Project Overview</div>', unsafe_allow_html=True)
        st.write(
            "Mobile app developers receive large volumes of user reviews every day. "
            "This system automatically classifies each review into positive, neutral, or negative sentiment, "
            "helping teams understand user feedback faster and more consistently."
        )
        st.write("**Team AKATSUKI**")
        st.write("1. Muhammad Lukman Bin Nasrum  \n2. Hasnawi Imran Bin Mohd Saidi  \n3. Muhammad Zahin Bin Mohd Zamri  \n4. Raqib Hazim Bin Abdul Hamid")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="lux-card"><div class="card-title">Sentiment Snapshot</div>', unsafe_allow_html=True)
        if not label_counts.empty:
            snapshot = label_counts.reset_index()
            snapshot.columns = ["Sentiment", "Count"]
            snapshot["Percentage"] = (snapshot["Count"] / snapshot["Count"].sum() * 100).round(2)
            st.dataframe(snapshot, use_container_width=True, hide_index=True)
        else:
            st.info("No label column found in the dataset.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="lux-card"><div class="card-title">Class Distribution</div>', unsafe_allow_html=True)
    st.pyplot(plot_label_distribution(df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================
# Sentiment Analyzer Page
# =============================
elif page == "Sentiment Analyzer":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Live Prediction</div>
            <div class="hero-title">Analyze a Review</div>
            <div class="hero-subtitle">
                Paste a Google Play Store review and let the NLP model predict its sentiment with confidence scoring.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sample_reviews = {
        "Positive sample": "This app is very useful and easy to use. The interface is clean and fast.",
        "Negative sample": "The app keeps crashing after the update and the login does not work.",
        "Neutral sample": "The app is okay. Some features work well but others still need improvement.",
    }

    left, right = st.columns([1.05, 0.95])

    with left:
        st.markdown('<div class="lux-card"><div class="card-title">Review Input</div>', unsafe_allow_html=True)
        selected_sample = st.selectbox("Try a sample review", ["Write my own"] + list(sample_reviews.keys()))
        default_text = "" if selected_sample == "Write my own" else sample_reviews[selected_sample]
        user_text = st.text_area("Review text", value=default_text, height=190, placeholder="Type or paste an app review here...")
        analyze = st.button("Analyze Sentiment", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="lux-card"><div class="card-title">Prediction Guide</div>', unsafe_allow_html=True)
        st.write("**Positive** — user expresses satisfaction or recommendation.")
        st.write("**Neutral** — mixed, factual, or moderate opinion.")
        st.write("**Negative** — user reports bugs, frustration, crash, or poor experience.")
        st.markdown("</div>", unsafe_allow_html=True)

    if analyze:
        if not user_text.strip():
            st.warning("Please enter a review first.")
        else:
            cleaned, label, confidence, probabilities = predict_sentiment(user_text, model, label_mapping)
            main_color, soft_color, display_label, pill_class = sentiment_style(label)
            confidence_text = f"{confidence:.2%}" if confidence is not None else "Not available"

            st.markdown(
                f"""
                <div class="result-card" style="border-left:8px solid {main_color};">
                    <div class="card-muted">Prediction Result</div>
                    <div class="result-label" style="color:{main_color};">{display_label}</div>
                    <span class="pill {pill_class}">Confidence {confidence_text}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if probabilities:
                prob_df = probability_table(probabilities)
                st.markdown('<div class="lux-card"><div class="card-title">Class Probability</div>', unsafe_allow_html=True)
                chart_df = prob_df.set_index("Sentiment")[["Probability"]]
                st.bar_chart(chart_df, use_container_width=True)
                st.dataframe(prob_df[["Sentiment", "Probability (%)"]], use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)

            col_clean, col_words = st.columns(2)
            with col_clean:
                st.markdown('<div class="lux-card"><div class="card-title">Cleaned Text</div>', unsafe_allow_html=True)
                st.code(cleaned)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_words:
                st.markdown('<div class="lux-card"><div class="card-title">Influential Words</div>', unsafe_allow_html=True)
                important_words = get_influential_words(user_text, model, label)
                if important_words:
                    imp_df = pd.DataFrame(important_words, columns=["Word / Phrase", "Model Weight"])
                    st.dataframe(imp_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Influential words are not available for this input/model.")
                st.markdown('</div>', unsafe_allow_html=True)


# =============================
# Data Explorer Page
# =============================
elif page == "Data Explorer":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Dataset Explorer</div>
            <div class="hero-title">Review Data Overview</div>
            <div class="hero-subtitle">
                Inspect the preprocessed dataset, label counts, app coverage, and raw review samples.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        build_kpi("Rows", f"{len(df):,}", "Total records")
    with col2:
        build_kpi("Columns", f"{len(df.columns)}", "Available fields")
    with col3:
        build_kpi("Labels", f"{df['label'].nunique() if 'label' in df.columns else 'N/A'}", "Sentiment classes")
    with col4:
        build_kpi("Apps", f"{df['app_name'].nunique() if 'app_name' in df.columns else 'N/A'}", "Unique apps")

    st.markdown('<div class="lux-card"><div class="card-title">Sample Preprocessed Data</div>', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="lux-card"><div class="card-title">Label Statistics</div>', unsafe_allow_html=True)
        label_counts_df = df["label"].value_counts().reset_index()
        label_counts_df.columns = ["Label", "Count"]
        label_counts_df["Percentage"] = (label_counts_df["Count"] / label_counts_df["Count"].sum() * 100).round(2)
        st.dataframe(label_counts_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="lux-card"><div class="card-title">Reviews by App</div>', unsafe_allow_html=True)
        if "app_name" in df.columns:
            app_counts = df["app_name"].value_counts().reset_index()
            app_counts.columns = ["App Name", "Count"]
            st.dataframe(app_counts, use_container_width=True, hide_index=True)
        else:
            st.info("No app_name column found.")
        st.markdown('</div>', unsafe_allow_html=True)

    if not raw_df.empty:
        with st.expander("Show raw dataset sample"):
            st.dataframe(raw_df.head(30), use_container_width=True)

    st.markdown('<div class="lux-card"><div class="card-title">Sentiment Class Distribution</div>', unsafe_allow_html=True)
    st.pyplot(plot_label_distribution(df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================
# Visualizations Page
# =============================
elif page == "Visualizations":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Visual Analytics</div>
            <div class="hero-title">Sentiment Insights</div>
            <div class="hero-subtitle">
                Explore word patterns, class balance, common tokens, and review length distribution.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Word Cloud", "Class Distribution", "Top Words", "Text Length"])

    with tab1:
        st.markdown('<div class="lux-card"><div class="card-title">Word Cloud</div>', unsafe_allow_html=True)
        label_options = ["All"] + sorted(df["label"].dropna().unique().tolist())
        selected_label = st.selectbox("Choose sentiment", label_options)
        st.pyplot(plot_wordcloud(df, selected_label), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="lux-card"><div class="card-title">Class Distribution</div>', unsafe_allow_html=True)
        st.pyplot(plot_label_distribution(df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="lux-card"><div class="card-title">Top 20 Most Common Words</div>', unsafe_allow_html=True)
        fig, top_df = plot_top_words(df)
        st.pyplot(fig, use_container_width=True)
        st.dataframe(top_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="lux-card"><div class="card-title">Review Text Length Distribution</div>', unsafe_allow_html=True)
        st.pyplot(plot_text_length(df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =============================
# Model Performance Page
# =============================
elif page == "Model Performance":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Machine Learning Model</div>
            <div class="hero-title">Model Performance</div>
            <div class="hero-subtitle">
                Review the selected model, evaluation metrics, label mapping, confusion matrix, and comparison results.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    metric_cols[1].metric("Precision", f"{metrics.get('precision', 0):.2%}")
    metric_cols[2].metric("Recall", f"{metrics.get('recall', 0):.2%}")
    metric_cols[3].metric("F1-score", f"{metrics.get('f1_score', 0):.2%}")

    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="lux-card"><div class="card-title">Best Model</div>', unsafe_allow_html=True)
        st.write(f"**{metadata.get('best_model_name', 'N/A')}**")
        st.write(
            "The final model uses TF-IDF feature extraction with unigram and bigram features, "
            "followed by Logistic Regression for sentiment classification."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="lux-card"><div class="card-title">Label Mapping</div>', unsafe_allow_html=True)
        st.json(label_mapping)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="lux-card"><div class="card-title">Confusion Matrix</div>', unsafe_allow_html=True)
        st.pyplot(plot_confusion_matrix(metadata), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="lux-card"><div class="card-title">Model Comparison</div>', unsafe_allow_html=True)
    if not pred_df.empty:
        fig, comparison_df = plot_model_comparison(pred_df)
        st.pyplot(fig, use_container_width=True)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    else:
        st.info("Prediction comparison file not found.")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Show pipeline structure"):
        st.code(str(model))
