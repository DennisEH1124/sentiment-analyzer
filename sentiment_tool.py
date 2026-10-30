"""Sentiment analysis tool: classifies text as positive, negative, neutral, or mixed.

Uses NLTK's VADER lexicon for polarity scoring. VADER gives one compound score
per text, which cannot express "mixed" on its own, so mixed is detected from
the pos/neg proportions and from sentence-level polarity.

Usage:
    python sentiment_tool.py "I love the screen but the battery is awful"
    python sentiment_tool.py --demo                      # bundled sample data + charts
    python sentiment_tool.py --csv tweets.csv --text-col text [--label-col label]
"""
import argparse
import html
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNGs; no display needed
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

LABELS = ["positive", "negative", "neutral", "mixed"]
COLORS = {"positive": "#2a9d8f", "negative": "#e76f51", "neutral": "#8d99ae", "mixed": "#e9c46a"}

# Thresholds (tuned on the sample data; adjust for your own corpus).
POLAR = 0.05        # |compound| >= this -> positive/negative (VADER's convention)
MIX_SEGMENT = 0.3   # segments at least this strong in opposite directions -> mixed

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
SPACE_RE = re.compile(r"\s+")
# Contrast words split a sentence into clauses so "great screen, but awful battery" is seen as two opinions.
CONTRAST_RE = re.compile(r"(?<!\w)(?:but|however|yet|although|though|while|whereas)(?!\w)|;", re.I)


def clean_text(text: str) -> str:
    """Strip noise (URLs, @mentions, HTML entities, '#' marks, RT prefix).

    Deliberately keeps case, punctuation, and emoticons: VADER uses ALL CAPS,
    '!' and ':)' as sentiment signals, and removing stopwords would drop
    negations like "not".
    """
    text = html.unescape(str(text))
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = re.sub(r"^\s*RT\b", " ", text)
    return SPACE_RE.sub(" ", text).strip()


def _load_vader() -> SentimentIntensityAnalyzer:
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        return SentimentIntensityAnalyzer()


_VADER = None


def _segments(text: str) -> list:
    """Split into sentences, then into clauses at contrast words."""
    try:
        sents = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        sents = nltk.sent_tokenize(text)
    return [c for s in sents for c in CONTRAST_RE.split(s) if c.strip(" ,.!?")]


def analyze(text: str) -> dict:
    """Return label plus VADER scores (compound, pos, neg, neu) for one text."""
    global _VADER
    if _VADER is None:
        _VADER = _load_vader()
    cleaned = clean_text(text)
    scores = _VADER.polarity_scores(cleaned)
    compound = scores["compound"]

    seg_scores = [_VADER.polarity_scores(c)["compound"] for c in _segments(cleaned)] or [0.0]
    mixed = max(seg_scores) >= MIX_SEGMENT and min(seg_scores) <= -MIX_SEGMENT
    if mixed:
        label = "mixed"
    elif compound >= POLAR:
        label = "positive"
    elif compound <= -POLAR:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "compound": compound, "pos": scores["pos"],
            "neg": scores["neg"], "neu": scores["neu"], "cleaned": cleaned}


def analyze_dataframe(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    results = pd.DataFrame([analyze(t) for t in df[text_col].fillna("")], index=df.index)
    return df.join(results.rename(columns={"label": "predicted"}))


def evaluate(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Print accuracy and per-class precision/recall; return the confusion matrix."""
    truth = df[label_col].str.lower().str.strip()
    pred = df["predicted"]
    print(f"\nAccuracy: {(truth == pred).mean():.1%} on {len(df)} labelled rows")
    rows = []
    for lab in LABELS:
        tp = ((truth == lab) & (pred == lab)).sum()
        p = tp / max((pred == lab).sum(), 1)
        r = tp / max((truth == lab).sum(), 1)
        rows.append({"class": lab, "precision": round(p, 2), "recall": round(r, 2),
                     "support": int((truth == lab).sum())})
    print(pd.DataFrame(rows).to_string(index=False))
    return pd.crosstab(truth, pred).reindex(index=LABELS, columns=LABELS, fill_value=0)


def make_charts(df: pd.DataFrame, outdir: Path, confusion: pd.DataFrame = None) -> list:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []

    counts = df["predicted"].value_counts().reindex(LABELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=[COLORS[l] for l in counts.index])
    ax.bar_label(bars)
    ax.set(title="Sentiment distribution", ylabel="Comments")
    ax.spines[["top", "right"]].set_visible(False)
    paths.append(outdir / "distribution.png")
    fig.tight_layout(); fig.savefig(paths[-1], dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["compound"], bins=20, range=(-1, 1), color="#577590", edgecolor="white")
    ax.axvline(POLAR, ls="--", c="grey", lw=1); ax.axvline(-POLAR, ls="--", c="grey", lw=1)
    ax.set(title="VADER compound score", xlabel="Compound (-1 negative ... +1 positive)",
           ylabel="Comments")
    ax.spines[["top", "right"]].set_visible(False)
    paths.append(outdir / "compound_hist.png")
    fig.tight_layout(); fig.savefig(paths[-1], dpi=150); plt.close(fig)

    if confusion is not None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.imshow(confusion.values, cmap="Blues")
        ax.set_xticks(range(4), LABELS, rotation=30); ax.set_yticks(range(4), LABELS)
        for i in range(4):
            for j in range(4):
                v = confusion.values[i, j]
                ax.text(j, i, v, ha="center", va="center",
                        color="white" if v > confusion.values.max() / 2 else "black")
        ax.set(title="Confusion matrix", xlabel="Predicted", ylabel="Actual")
        paths.append(outdir / "confusion_matrix.png")
        fig.tight_layout(); fig.savefig(paths[-1], dpi=150); plt.close(fig)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("text", nargs="?", help="a single text to classify")
    ap.add_argument("--demo", action="store_true", help="run on the bundled sample_data.csv")
    ap.add_argument("--csv", help="CSV file of texts to classify")
    ap.add_argument("--text-col", default="text", help="text column name (default: text)")
    ap.add_argument("--label-col", help="optional ground-truth column (positive/negative/neutral/mixed)")
    ap.add_argument("--encoding", default="utf-8", help="CSV encoding (Kaggle Twitter sets are often latin-1)")
    ap.add_argument("--out", default="output", help="output directory (default: output)")
    args = ap.parse_args(argv)

    if args.text:
        r = analyze(args.text)
        print(f"{r['label'].upper()}  (compound={r['compound']:+.3f}, pos={r['pos']:.2f}, "
              f"neg={r['neg']:.2f}, neu={r['neu']:.2f})")
        return 0

    if args.demo:
        args.csv = str(Path(__file__).with_name("sample_data.csv"))
        args.label_col = "label"
    if not args.csv:
        ap.print_help()
        return 1

    df = pd.read_csv(args.csv, encoding=args.encoding)
    if args.text_col not in df.columns:
        sys.exit(f"Column '{args.text_col}' not found. Columns: {list(df.columns)}")
    df = analyze_dataframe(df, args.text_col)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["cleaned"]).to_csv(out / "predictions.csv", index=False)

    print("Predicted sentiment counts:")
    print(df["predicted"].value_counts().reindex(LABELS, fill_value=0).to_string())

    confusion = None
    if args.label_col and args.label_col in df.columns:
        confusion = evaluate(df, args.label_col)
    for p in make_charts(df, out, confusion):
        print(f"Saved {p}")
    print(f"Saved {out / 'predictions.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
