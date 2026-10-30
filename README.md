# Sentiment Analyzer

Classifies text (tweets, reviews, comments) as **positive, negative, neutral, or mixed** using NLTK's VADER.

    pip install -r requirements.txt
    python sentiment_tool.py "Great screen, but the battery is awful"   # single text
    python sentiment_tool.py --demo                                     # sample data: metrics + charts
    python sentiment_tool.py --csv tweets.csv --text-col text [--label-col sentiment] [--encoding latin-1]

Outputs go to `output/`: `predictions.csv`, `distribution.png`, `compound_hist.png`, and `confusion_matrix.png` (when labels are given).

## How it works
1. **Cleaning** – removes URLs, @mentions, HTML entities, `#`, and `RT`. Case, punctuation, and emoticons are kept because VADER scores them.
2. **Scoring** – VADER returns pos/neg/neu proportions and a compound score in [-1, 1].
3. **Labelling** – compound >= 0.05 positive, <= -0.05 negative, else neutral.
   **Mixed** is when separate clauses (split on sentence ends and words like *but/however/yet*) have opposite polarity, each with |compound| >= 0.3.

## Using the Kaggle Twitter dataset
Download it, then pass `--csv`. Its labels must be positive/negative/neutral/mixed (map other values first, e.g. "irrelevant" -> neutral) to get accuracy and the confusion matrix. Thresholds are constants at the top of `sentiment_tool.py`.

## Limits
VADER is lexicon-based: sarcasm and slang are weak spots, and the bundled 32-row sample is tiny, so its 97% accuracy is illustrative only (thresholds were tuned on it). Re-check on your own data.
