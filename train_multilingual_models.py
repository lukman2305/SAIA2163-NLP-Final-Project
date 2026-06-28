"""
Multilingual training script for the SAIA2163 NLP Final Project.

This version uses ALL dataset languages instead of filtering English only:
EN, FR, DE, JP, and IT.

Main changes from the English-only version:
1. No English-only lemmatization.
2. No English-only stopword removal.
3. Unicode-safe multilingual cleaning.
4. Language token added to text: lang_en, lang_fr, lang_de, lang_jp, lang_it.
5. Character n-gram and combined word+character TF-IDF models for multilingual/noisy text.
6. Extra language-level evaluation is saved to multilingual_language_metrics.csv.

Run:
    python train_multilingual_models.py --data Training_Data_Google_Play_reviews_6000.csv

Optional multilingual transformer training:
    python train_multilingual_models.py --data Training_Data_Google_Play_reviews_6000.csv --run-transformer
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from wordcloud import WordCloud

from preprocessing_utils import (
    INVERSE_LABEL_MAPPING,
    LABEL_MAPPING,
    LABEL_NAMES,
    SUPPORTED_LANGUAGES,
    add_language_token,
    language_name,
    map_score_to_label,
    multilingual_light_clean_text,
    normalize_language_code,
    transformer_clean_text,
)

RANDOM_STATE = 42
APP_MAPPING = {
    "org.telegram.messenger": "Telegram",
    "com.facebook.orca": "Facebook Messenger",
    "com.whatsapp": "WhatsApp",
    "com.viber.voip": "Viber",
    "com.snapchat.android": "Snapchat",
    "com.tencent.mm": "WeChat",
}


def load_and_prepare_dataset(csv_path: Path, output_dir: Path) -> pd.DataFrame:
    """Load full multilingual CSV, map labels, and create model-ready columns."""
    df = pd.read_csv(csv_path)

    required_columns = {"content", "score"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    if "userLang" in df.columns:
        df["userLang"] = df["userLang"].apply(normalize_language_code)
    else:
        df["userLang"] = "UNKNOWN"

    # Keep only the five expected languages when they exist in the dataset.
    expected = set(SUPPORTED_LANGUAGES.keys())
    if df["userLang"].isin(expected).any():
        df = df[df["userLang"].isin(expected)].copy()

    df["language_name"] = df["userLang"].apply(language_name)

    if "app_id" in df.columns:
        df["app_name"] = df["app_id"].map(APP_MAPPING).fillna(df["app_id"].astype(str))
    else:
        df["app_name"] = "Unknown"

    if "at" in df.columns:
        df["review_date"] = pd.to_datetime(df["at"], errors="coerce")
    else:
        df["review_date"] = pd.NaT

    df["label"] = df["score"].apply(map_score_to_label)
    df["target"] = df["label"].map(LABEL_MAPPING).astype(int)
    df["raw_text"] = df["content"].fillna("").astype(str)

    # Language token improves traditional ML models because words and scripts differ by language.
    df["model_input"] = [add_language_token(text, lang) for text, lang in zip(df["raw_text"], df["userLang"])]
    df["text"] = df["model_input"].apply(multilingual_light_clean_text)
    df["transformer_text"] = df["raw_text"].apply(transformer_clean_text)

    # Drop truly empty rows after cleaning.
    df["text"] = df["text"].replace("", np.nan)
    df = df.dropna(subset=["text", "label", "target"]).copy()

    columns_to_save = [
        "raw_text",
        "model_input",
        "text",
        "transformer_text",
        "label",
        "target",
        "userLang",
        "language_name",
        "app_name",
        "review_date",
    ]
    df[columns_to_save].to_csv(output_dir / "preprocessed_reviews.csv", index=False)
    return df


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(weighted_precision),
        "recall": float(weighted_recall),
        "f1_score": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "neutral_recall": float(per_class_recall[1]),
    }


def evaluate_pipeline(name: str, pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    preds = pipeline.predict(X_test)
    metrics = metric_dict(y_test.to_numpy(), preds)
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
    report = classification_report(
        y_test,
        preds,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        zero_division=0,
        output_dict=True,
    )

    print(f"\n==== {name} ====")
    for key, value in metrics.items():
        print(f"{key:>16}: {value:.4f}")
    print(classification_report(y_test, preds, labels=[0, 1, 2], target_names=LABEL_NAMES, zero_division=0))

    return {
        "model_name": name,
        "pipeline": pipeline,
        "predictions": preds,
        "confusion_matrix": cm,
        "classification_report": report,
        **metrics,
    }


def build_multilingual_models() -> dict[str, Pipeline]:
    """Traditional ML models suitable for multilingual text."""
    word_tfidf = TfidfVectorizer(
        preprocessor=multilingual_light_clean_text,
        analyzer="word",
        ngram_range=(1, 2),
        max_features=15000,
        min_df=2,
        max_df=0.95,
        token_pattern=r"(?u)\b\w+\b",
        sublinear_tf=True,
    )
    char_tfidf = TfidfVectorizer(
        preprocessor=multilingual_light_clean_text,
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=12000,
        min_df=2,
        sublinear_tf=True,
    )

    return {
        "Multilingual Logistic Regression Word TF-IDF": Pipeline([
            ("tfidf", word_tfidf),
            ("classifier", OneVsRestClassifier(LogisticRegression(max_iter=1500, solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE))),
        ]),
        "Multilingual Naive Bayes Word TF-IDF": Pipeline([
            ("tfidf", word_tfidf),
            ("classifier", MultinomialNB(alpha=0.5)),
        ]),
        "Multilingual SGD Character N-grams": Pipeline([
            ("tfidf", char_tfidf),
            ("classifier", SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=1000, tol=1e-3, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Multilingual SGD Word + Character TF-IDF": Pipeline([
            ("features", FeatureUnion([
                ("word", word_tfidf),
                ("char", char_tfidf),
            ])),
            ("classifier", SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=1000, tol=1e-3, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }


def run_grid_search(X_train: pd.Series, y_train: pd.Series) -> Pipeline:
    """Small grid search for the multilingual combined model."""
    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(
                preprocessor=multilingual_light_clean_text,
                analyzer="word",
                token_pattern=r"(?u)\b\w+\b",
                sublinear_tf=True,
            )),
            ("char", TfidfVectorizer(
                preprocessor=multilingual_light_clean_text,
                analyzer="char_wb",
                sublinear_tf=True,
            )),
        ])),
        ("classifier", OneVsRestClassifier(LogisticRegression(max_iter=1500, solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE))),
    ])

    param_grid = {
        "features__word__ngram_range": [(1, 1), (1, 2)],
        "features__word__max_features": [10000, 15000],
        "features__char__ngram_range": [(2, 4), (2, 5)],
        "features__char__max_features": [10000, 12000],
        "classifier__estimator__C": [0.5, 1.0, 2.0],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    print("\nBest grid parameters:")
    print(json.dumps(grid.best_params_, indent=2))
    print(f"Best CV macro-F1: {grid.best_score_:.4f}")
    return grid.best_estimator_


def plot_confusion_matrix(cm: np.ndarray, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xticks(range(len(LABEL_NAMES)))
    ax.set_yticks(range(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_visuals(df: pd.DataFrame, results_df: pd.DataFrame, output_dir: Path) -> None:
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Class distribution
    label_counts = df["label"].value_counts().reindex(LABEL_NAMES).fillna(0)
    ax = label_counts.plot(kind="bar", figsize=(7, 4), title="Multilingual Sentiment Class Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(images_dir / "class_distribution_multilingual.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Language distribution
    lang_counts = df["userLang"].value_counts().sort_index()
    ax = lang_counts.plot(kind="bar", figsize=(7, 4), title="Review Count by Language")
    ax.set_xlabel("Language")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(images_dir / "language_distribution.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Model comparison
    metric_col = "macro_f1" if "macro_f1" in results_df.columns else "accuracy"
    plot_df = results_df.sort_values(metric_col, ascending=True)
    ax = plot_df.plot(kind="barh", x="model_name", y=metric_col, figsize=(10, 5), legend=False, title=f"Model Comparison by {metric_col}")
    ax.set_xlabel(metric_col)
    ax.set_ylabel("Model")
    plt.tight_layout()
    plt.savefig(images_dir / "model_macro_f1_comparison.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Word cloud. It may not render Japanese perfectly without a system Japanese font, but it will not break training.
    joined_text = " ".join(df["text"].dropna().astype(str).sample(min(3000, len(df)), random_state=RANDOM_STATE))
    if joined_text.strip():
        try:
            wc = WordCloud(width=1200, height=600, background_color="white", collocations=False).generate(joined_text)
            plt.figure(figsize=(12, 6))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(images_dir / "wordcloud_multilingual.png", dpi=160, bbox_inches="tight")
            plt.close()
        except Exception as exc:
            print(f"Skipping word cloud because WordCloud failed: {exc}")

    # Top 20 words bar chart
    from sklearn.feature_extraction.text import CountVectorizer

    top_text = df["text"].dropna().astype(str)
    if len(top_text) > 0:
        vectorizer = CountVectorizer(
            max_features=20,
            token_pattern=r"(?u)\b\w+\b"
        )
        word_counts = vectorizer.fit_transform(top_text)
        word_sum = word_counts.sum(axis=0).A1

        top_words_df = pd.DataFrame({
            "word": vectorizer.get_feature_names_out(),
            "count": word_sum
        }).sort_values("count", ascending=True)

        ax = top_words_df.plot(
            kind="barh",
            x="word",
            y="count",
            figsize=(10, 6),
            legend=False,
            title="Top 20 Most Frequent Words"
        )
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Word")
        plt.tight_layout()
        plt.savefig(images_dir / "top_20_words_multilingual.png", dpi=160, bbox_inches="tight")
        plt.close()


def language_level_metrics(test_df: pd.DataFrame, predictions: np.ndarray, output_dir: Path) -> pd.DataFrame:
    temp = test_df.copy()
    temp["prediction"] = predictions
    rows: list[dict[str, Any]] = []

    for lang, group in temp.groupby("userLang"):
        y_true = group["target"].to_numpy()
        y_pred = group["prediction"].to_numpy()
        metrics = metric_dict(y_true, y_pred)
        rows.append({
            "language": lang,
            "language_name": language_name(lang),
            "rows": len(group),
            **metrics,
        })

    out = pd.DataFrame(rows).sort_values("language")
    out.to_csv(output_dir / "multilingual_language_metrics.csv", index=False)
    return out


def train_optional_transformer(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> dict[str, Any] | None:
    """Optional XLM-RoBERTa training. This is slower and requires transformers + torch."""
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        print(f"Skipping transformer training because dependencies are missing: {exc}")
        return None

    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_data = Dataset.from_pandas(train_df[["transformer_text", "target"]].rename(columns={"target": "labels"}))
    test_data = Dataset.from_pandas(test_df[["transformer_text", "target"]].rename(columns={"target": "labels"}))

    def tokenize(batch):
        return tokenizer(batch["transformer_text"], truncation=True, max_length=128)

    train_data = train_data.map(tokenize, batched=True)
    test_data = test_data.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={i: label for i, label in enumerate(LABEL_NAMES)},
        label2id=LABEL_MAPPING,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        metrics = metric_dict(np.array(labels), np.array(preds))
        return {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "neutral_recall": metrics["neutral_recall"]}

    args = TrainingArguments(
        output_dir=str(output_dir / "transformer_results_multilingual"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    pred_output = trainer.predict(test_data)
    preds = np.argmax(pred_output.predictions, axis=-1)
    metrics = metric_dict(test_df["target"].to_numpy(), preds)
    model_dir = output_dir / "multilingual_transformer_model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    return {
        "model_name": "XLM-RoBERTa Multilingual Transformer",
        "pipeline": None,
        "predictions": preds,
        "confusion_matrix": confusion_matrix(test_df["target"], preds, labels=[0, 1, 2]),
        "classification_report": classification_report(test_df["target"], preds, labels=[0, 1, 2], target_names=LABEL_NAMES, zero_division=0, output_dict=True),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="Training_Data_Google_Play_reviews_6000.csv")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--run-grid", action="store_true", help="Run slower GridSearchCV model.")
    parser.add_argument("--run-transformer", action="store_true", help="Run optional multilingual transformer training.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = output_dir / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path.resolve()}")

    df = load_and_prepare_dataset(data_path, output_dir)

    print("\nRows after preprocessing:", len(df))
    print("\nLanguage distribution:")
    print(df["userLang"].value_counts().sort_index())
    print("\nSentiment distribution:")
    print(df["label"].value_counts().reindex(LABEL_NAMES).fillna(0).astype(int))
    print("\nSentiment by language:")
    print(pd.crosstab(df["userLang"], df["label"]).reindex(columns=LABEL_NAMES).fillna(0).astype(int))

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df["target"],
    )

    X_train = train_df["model_input"]
    X_test = test_df["model_input"]
    y_train = train_df["target"]
    y_test = test_df["target"]

    results: list[dict[str, Any]] = []
    for name, model in build_multilingual_models().items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        results.append(evaluate_pipeline(name, model, X_test, y_test))

    if args.run_grid:
        print("\nTraining GridSearchCV multilingual model...")
        grid_model = run_grid_search(X_train, y_train)
        results.append(evaluate_pipeline("GridSearch Multilingual Word + Character TF-IDF", grid_model, X_test, y_test))

    if args.run_transformer:
        transformer_result = train_optional_transformer(train_df, test_df, output_dir)
        if transformer_result is not None:
            results.append(transformer_result)

    results_df = pd.DataFrame([
        {k: v for k, v in result.items() if k not in {"pipeline", "predictions", "confusion_matrix", "classification_report"}}
        for result in results
    ])
    results_df = results_df.sort_values(["macro_f1", "f1_score", "accuracy"], ascending=False).reset_index(drop=True)
    results_df.to_csv(output_dir / "model_evaluation_comparison.csv", index=False)

    best_name = str(results_df.loc[0, "model_name"])
    best_result = next(result for result in results if result["model_name"] == best_name)
    best_pipeline = best_result["pipeline"]
    best_predictions = best_result["predictions"]

    print("\nBest model:", best_name)
    print(results_df.head())

    if best_pipeline is not None:
        joblib.dump(best_pipeline, output_dir / "best_sentiment_pipeline.pkl")

    predictions_df = test_df[["raw_text", "model_input", "text", "label", "target", "userLang", "language_name", "app_name", "review_date"]].copy()
    predictions_df["predicted_target"] = best_predictions
    predictions_df["predicted_label"] = predictions_df["predicted_target"].map(INVERSE_LABEL_MAPPING)
    predictions_df.to_csv(output_dir / "final_model_predictions.csv", index=False)

    language_metrics_df = language_level_metrics(test_df, best_predictions, output_dir)
    print("\nBest model language-level metrics:")
    print(language_metrics_df)

    plot_confusion_matrix(best_result["confusion_matrix"], f"Confusion Matrix - {best_name}", output_dir / "images" / "confusion_matrix_best_model.png")
    save_visuals(df, results_df, output_dir)

    metadata = {
        "project_type": "multilingual_sentiment_analysis",
        "best_model_name": best_name,
        "label_mapping": LABEL_MAPPING,
        "inverse_label_mapping": INVERSE_LABEL_MAPPING,
        "label_names": LABEL_NAMES,
        "supported_languages": SUPPORTED_LANGUAGES,
        "uses_language_token": True,
        "preprocessing": "Unicode-safe multilingual cleaning; no English-only lemmatization or stopword removal.",
        "dataset_rows_after_preprocessing": int(len(df)),
        "language_distribution": df["userLang"].value_counts().sort_index().to_dict(),
        "sentiment_distribution": df["label"].value_counts().to_dict(),
        "best_metrics": {k: float(best_result[k]) for k in ["accuracy", "precision", "recall", "f1_score", "macro_f1", "neutral_recall"]},
    }
    joblib.dump(metadata, output_dir / "backend_metadata.pkl")

    with open(output_dir / "backend_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nSaved files:")
    for filename in [
        "best_sentiment_pipeline.pkl",
        "backend_metadata.pkl",
        "backend_metadata.json",
        "preprocessed_reviews.csv",
        "final_model_predictions.csv",
        "model_evaluation_comparison.csv",
        "multilingual_language_metrics.csv",
    ]:
        path = output_dir / filename
        if path.exists():
            print("-", path.name)


if __name__ == "__main__":
    main()
