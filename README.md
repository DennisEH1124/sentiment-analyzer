# Sentiment Analyzer

## The problem
Project output: A sentiment analysis tool that classifies user comments, tweets, or reviews. 
Tools used: Python, NLTK, VADER, Pandas, Matplotlib 
Data source: Public Twitter sentiment dataset from Kaggle or sample social media text data 
Core concept: Natural language processing, sentiment scoring, text cleaning, and polarity analysis 

Idea from: https://skillifysolutions.com/blogs/artificial-intelligence/best-ai-project-ideas-for-students/

## Product hypothesis
Claude code-generated solution. 

## Demo
Experimented with manually entered examples. 

## My role
Claude code-generated solution. Tested self-generated examples. 

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

## Evaluation and results
Mixed sentiment (combination of positive/negative) seemed to be harder to detect. Slang was also hit or miss.
VADER is lexicon-based: sarcasm and slang are weak spots, and the bundled 32-row sample is tiny, so its 97% accuracy is illustrative only (thresholds were tuned on it). Re-check on your own data.

## Product decisions and tradeoffs
Would like to explore retraining with a larger sample and to incorporate different linguistic styles (e.g. slang)

## What I would do next
1. Expand solution to understand slang
1. Improve accuracy of "mixed" to better represent responses with both positive and negative sentiment.
1. Could also explore assigning an emotion/intensity score to the sentiment analysis (i.e. anger, joy, sadness, etc.)
