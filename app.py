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
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["Label", "Count"]
    counts["Percentage"] = counts["Count"] / counts["Count"].sum() * 100
    counts = counts.sort_values("Count", ascending=True)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    colors = [sentiment_style(x)[0] for x in counts["Label"]]
    bars = ax.barh(counts["Label"], counts["Count"], color=colors, edgecolor="white", linewidth=2)

    for bar, pct in zip(bars, counts["Percentage"]):
        width = bar.get_width()
        ax.text(width + max(counts["Count"]) * 0.015, bar.get_y() + bar.get_height()/2,
                f"{int(width):,} reviews  •  {pct:.1f}%", va="center", fontsize=10, fontweight="bold", color="#434343")

    ax.set_title("Sentiment Class Distribution", fontsize=17, pad=16, fontweight="bold")
    ax.set_xlabel("Number of reviews", fontsize=11)
    ax.set_ylabel("")
    ax.set_xlim(0, max(counts["Count"]) * 1.22)
    ax.grid(axis="x", color="#E8C8D1", linewidth=1, alpha=0.85)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#E8C8D1")
    fig.tight_layout()
    return fig


def plot_wordcloud(df, label_filter="All"):
    text_col = "text" if "text" in df.columns else "clean_text" if "clean_text" in df.columns else "content"
    if label_filter != "All":
        text_data = " ".join(df[df["label"] == label_filter][text_col].dropna().astype(str))
    else:
        text_data = " ".join(df[text_col].dropna().astype(str))

    if not text_data.strip():
        text_data = "no text available"

    wc = WordCloud(
        width=1400,
        height=620,
        background_color="white",
        max_words=150,
        colormap="viridis",
        collocations=False,
        contour_width=2,
        contour_color="#E8C8D1",
    ).generate(text_data)
    fig, ax = plt.subplots(figsize=(12, 5.3))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud: {label_filter}", fontsize=17, pad=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(metadata):
    cm = np.array(metadata.get("confusion_matrix", []))
    label_mapping = metadata.get("label_mapping", {})
    labels = [label for label, _ in sorted(label_mapping.items(), key=lambda x: x[1])]

    if cm.size == 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Confusion matrix not available", ha="center", va="center", fontsize=13)
        ax.axis("off")
        return fig

    row_sum = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0) * 100
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n{pct[i, j]:.1f}%"

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="RdPu",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=True,
        linewidths=1.8,
        linecolor="white",
        square=True,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )
    ax.set_title("Confusion Matrix — Best Model", fontsize=17, pad=16, fontweight="bold")
    ax.set_xlabel("Predicted sentiment", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual sentiment", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig


def build_model_comparison_df(pred_df):
    if pred_df.empty or "true_sentiment" not in pred_df.columns:
        return pd.DataFrame()

    model_cols = [c for c in pred_df.columns if c.startswith("sentiment_")]
    rows = []
    for col in model_cols:
        accuracy = (pred_df[col] == pred_df["true_sentiment"]).mean()
        model_name = col.replace("sentiment_", "").replace("_", " ").title()
        rows.append({"Model": model_name, "Accuracy": accuracy})

    comparison_df = pd.DataFrame(rows)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
        comparison_df["Rank"] = comparison_df.index + 1
        comparison_df["Accuracy (%)"] = (comparison_df["Accuracy"] * 100).round(2)
    return comparison_df


def plot_model_comparison(pred_df):
    comparison_df = build_model_comparison_df(pred_df)
    fig, ax = plt.subplots(figsize=(9.4, 5.2))

    if comparison_df.empty:
        ax.text(0.5, 0.5, "Prediction comparison file not found", ha="center", va="center", fontsize=13)
        ax.axis("off")
        return fig, comparison_df

    colors = ["#B9375E" if i == 0 else "#5F7A44" if i == 1 else "#BE9A60" for i in range(len(comparison_df))]
    bars = ax.bar(comparison_df["Model"], comparison_df["Accuracy (%)"], color=colors, edgecolor="white", linewidth=2)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        badge = "BEST" if i == 0 else f"#{i+1}"
        ax.text(bar.get_x() + bar.get_width()/2, height + 1.2, f"{height:.2f}%\n{badge}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#434343")

    ax.set_ylim(0, 105)
    ax.set_title("Model Accuracy Comparison", fontsize=17, pad=18, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xlabel("")
    ax.grid(axis="y", color="#E8C8D1", linewidth=1, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E8C8D1")
    ax.spines["bottom"].set_color("#E8C8D1")
    fig.tight_layout()
    return fig, comparison_df


def plot_top_words(df, top_n=20):
    text_col = "text" if "text" in df.columns else "clean_text" if "clean_text" in df.columns else "content"
    all_words = " ".join(df[text_col].dropna().astype(str)).split()
    counts = Counter(all_words).most_common(top_n)
    top_df = pd.DataFrame(counts, columns=["Word", "Frequency"])
    top_df = top_df.sort_values("Frequency", ascending=True)

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    bars = ax.barh(top_df["Word"], top_df["Frequency"], color="#B9375E", edgecolor="white", linewidth=1.5)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + max(top_df["Frequency"]) * 0.012, bar.get_y() + bar.get_height()/2,
                f"{int(width):,}", va="center", fontsize=9.5, fontweight="bold", color="#434343")

    ax.set_title(f"Top {top_n} Most Frequent Words", fontsize=17, pad=16, fontweight="bold")
    ax.set_xlabel("Frequency", fontsize=11)
    ax.set_ylabel("")
    ax.set_xlim(0, max(top_df["Frequency"]) * 1.16)
    ax.grid(axis="x", color="#E8C8D1", linewidth=1, alpha=0.85)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#E8C8D1")
    fig.tight_layout()
    return fig, top_df.sort_values("Frequency", ascending=False).reset_index(drop=True)


def plot_text_length(df):
    text_col = "text" if "text" in df.columns else "clean_text" if "clean_text" in df.columns else "content"
    temp = df.copy()
    temp["text_length"] = temp[text_col].fillna("").astype(str).apply(lambda x: len(x.split()))
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    sns.histplot(temp["text_length"], bins=30, kde=True, ax=ax, color="#B9375E", edgecolor="white", linewidth=1)
    median_len = temp["text_length"].median()
    ax.axvline(median_len, color="#434343", linestyle="--", linewidth=2)
    ax.text(median_len, ax.get_ylim()[1] * 0.9, f"Median: {median_len:.0f} words", rotation=90,
            va="top", ha="right", fontsize=10, fontweight="bold", color="#434343")
    ax.set_title("Review Text Length Distribution", fontsize=17, pad=16, fontweight="bold")
    ax.set_xlabel("Number of words", fontsize=11)
    ax.set_ylabel("Number of reviews", fontsize=11)
    return make_fig_clean(fig, ax)


def plot_metric_summary(metrics):
    metric_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Score": [
            metrics.get("accuracy", 0),
            metrics.get("precision", 0),
            metrics.get("recall", 0),
            metrics.get("f1_score", 0),
        ],
    })
    metric_df["Score (%)"] = metric_df["Score"] * 100
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    bars = ax.bar(metric_df["Metric"], metric_df["Score (%)"], color=["#B9375E", "#5F7A44", "#BE9A60", "#434343"], edgecolor="white", linewidth=2)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 1, f"{height:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_title("Best Model Evaluation Metrics", fontsize=17, pad=16, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("")
    return make_fig_clean(fig, ax), metric_df


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

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Sentiment Analyzer",
        "Data Explorer",
        "Visualizations",
        "Model Performance"
    ]
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
                Explore richer visual summaries of review sentiment, common words, review length, and class balance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        build_kpi("Reviews", f"{len(df):,}", "Dataset records")
    with c2:
        build_kpi("Classes", f"{df['label'].nunique() if 'label' in df.columns else 'N/A'}", "Sentiment labels")
    with c3:
        top_label = df["label"].value_counts().idxmax() if "label" in df.columns and not df.empty else "N/A"
        build_kpi("Majority Class", str(top_label).title(), "Most common label")
    with c4:
        text_col = "text" if "text" in df.columns else "clean_text" if "clean_text" in df.columns else "content"
        avg_len = df[text_col].fillna("").astype(str).apply(lambda x: len(x.split())).mean()
        build_kpi("Avg. Length", f"{avg_len:.0f}", "Words per review")

    tab1, tab2, tab3, tab4 = st.tabs(["☁️ Word Cloud", "📊 Class Balance", "🔤 Top Words", "📝 Text Length"])

    with tab1:
        st.markdown('<div class="lux-card"><div class="card-title">Word Cloud</div><div class="card-muted">The larger the word, the more frequently it appears in the review dataset.</div>', unsafe_allow_html=True)
        label_options = ["All"] + sorted(df["label"].dropna().unique().tolist())
        selected_label = st.selectbox("Choose sentiment", label_options)
        st.pyplot(plot_wordcloud(df, selected_label), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="lux-card"><div class="card-title">Sentiment Class Distribution</div><div class="card-muted">This chart shows whether the dataset is balanced or dominated by one sentiment class.</div>', unsafe_allow_html=True)
        st.pyplot(plot_label_distribution(df), use_container_width=True)
        label_summary = df["label"].value_counts().reset_index()
        label_summary.columns = ["Sentiment", "Count"]
        label_summary["Percentage"] = (label_summary["Count"] / label_summary["Count"].sum() * 100).round(2)
        st.dataframe(label_summary, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="lux-card"><div class="card-title">Top 20 Most Frequent Words</div><div class="card-muted">These words help explain the dominant themes found in user reviews.</div>', unsafe_allow_html=True)
        fig, top_df = plot_top_words(df)
        st.pyplot(fig, use_container_width=True)
        st.dataframe(top_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="lux-card"><div class="card-title">Review Text Length Distribution</div><div class="card-muted">This chart shows whether users usually leave short comments or longer detailed reviews.</div>', unsafe_allow_html=True)
        st.pyplot(plot_text_length(df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =============================
# Model Performance Page
# =============================
elif page == "Model Performance":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Machine Learning Evaluation</div>
            <div class="hero-title">Model Performance</div>
            <div class="hero-subtitle">
                Compare model performance using accuracy, precision, recall, F1-score, confusion matrix, and final model details.
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

    st.markdown('<div class="lux-card"><div class="card-title">Evaluation Metrics Summary</div><div class="card-muted">The final selected model is evaluated using four standard classification metrics.</div>', unsafe_allow_html=True)
    metric_fig, metric_df = plot_metric_summary(metrics)
    st.pyplot(metric_fig, use_container_width=True)
    metric_table = metric_df.copy()
    metric_table["Score (%)"] = metric_table["Score (%)"].round(2)
    st.dataframe(metric_table[["Metric", "Score (%)"]], use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns([0.95, 1.05])
    with left:
        st.markdown('<div class="lux-card"><div class="card-title">Best Model Selected</div>', unsafe_allow_html=True)
        best_name = metadata.get("best_model_name", "N/A")
        st.markdown(f"### 🏆 {best_name}")
        st.write(
            "The best model was selected based on the overall evaluation results. "
            "It is saved as `best_sentiment_pipeline.pkl` and loaded by the Streamlit app for prediction."
        )
        st.write("**Training setup:** 80:20 train-test split")
        st.write("**Feature extraction:** TF-IDF text representation")
        st.write("**Task:** Multi-class sentiment classification")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="lux-card"><div class="card-title">Label Mapping</div><div class="card-muted">How the model encodes sentiment labels internally.</div>', unsafe_allow_html=True)
        if label_mapping:
            mapping_df = pd.DataFrame(list(label_mapping.items()), columns=["Sentiment", "Encoded Label"])
            st.dataframe(mapping_df, use_container_width=True, hide_index=True)
        else:
            st.info("Label mapping not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="lux-card"><div class="card-title">Confusion Matrix</div><div class="card-muted">Correct predictions appear on the diagonal. Off-diagonal values show where the model confused one class with another.</div>', unsafe_allow_html=True)
        st.pyplot(plot_confusion_matrix(metadata), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="lux-card"><div class="card-title">Model Comparison</div><div class="card-muted">Accuracy comparison across the trained models. The highest bar indicates the best-performing model.</div>', unsafe_allow_html=True)
    if not pred_df.empty:
        fig, comparison_df = plot_model_comparison(pred_df)
        st.pyplot(fig, use_container_width=True)
        display_df = comparison_df.copy()
        if not display_df.empty:
            display_df["Accuracy"] = (display_df["Accuracy"] * 100).round(2).astype(str) + "%"
            st.dataframe(display_df[["Rank", "Model", "Accuracy"]], use_container_width=True, hide_index=True)
    else:
        st.info("Prediction comparison file not found.")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Show pipeline structure"):
        st.code(str(model))
