# Multilingual Improvement Patch

This patch changes the project from **English-only sentiment analysis** to **multilingual sentiment analysis** using all five dataset languages:

- EN: English
- FR: French
- DE: German
- JP: Japanese
- IT: Italian

## Why this change was needed

The English-only dataset had only 70 neutral reviews, which caused neutral recall to become 0. Using all languages increases the number of neutral examples and gives the model more training data.

## Important method change

The previous improved version used English lemmatization. That is good for English-only data, but not for multilingual data.

For multilingual training, this patch uses:

- Unicode-safe text cleaning
- Emoji conversion using `emoji.demojize`
- Language tokens such as `lang_en`, `lang_fr`, `lang_de`, `lang_jp`, `lang_it`
- Word TF-IDF n-grams
- Character n-grams
- Combined word + character TF-IDF
- Class balancing
- Macro-F1 and neutral recall evaluation
- Per-language metrics

It does **not** use English-only stopword removal or English-only lemmatization.

## Files

Copy these files into the project root folder:

```text
preprocessing_utils.py
train_multilingual_models.py
app.py
requirement.txt
```

You can rename:

```text
preprocessing_utils_multilingual.py -> preprocessing_utils.py
app_multilingual.py -> app.py
requirement_multilingual.txt -> requirement.txt
```

## Run training

```bash
pip install -r requirement.txt
python train_multilingual_models.py --data Training_Data_Google_Play_reviews_6000.csv
```

For slower GridSearchCV tuning:

```bash
python train_multilingual_models.py --data Training_Data_Google_Play_reviews_6000.csv --run-grid
```

For optional multilingual transformer training:

```bash
python train_multilingual_models.py --data Training_Data_Google_Play_reviews_6000.csv --run-transformer
```

The transformer uses `xlm-roberta-base`, which is more suitable for multilingual text than English DistilBERT.

## Run the app

```bash
streamlit run app.py
```

## Main output files

Training creates:

```text
best_sentiment_pipeline.pkl
backend_metadata.pkl
backend_metadata.json
preprocessed_reviews.csv
final_model_predictions.csv
model_evaluation_comparison.csv
multilingual_language_metrics.csv
images/
```

## Report explanation

Use this in your report:

> The improved multilingual model uses the full dataset across English, French, German, Japanese, and Italian reviews. Because English-only lemmatization and stopword removal are not suitable for multilingual text, the preprocessing was changed to Unicode-safe light cleaning. Language tokens were added to help the model learn language-specific patterns. Word-level and character-level TF-IDF n-gram models were evaluated, and performance was compared using accuracy, weighted F1-score, macro-F1, neutral recall, and per-language metrics.
