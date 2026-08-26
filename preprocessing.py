"""
Shared text cleaning for the Product Review Sentiment Analyzer.
Used by BOTH train_model.py and app.py so training and inference
always see text cleaned the exact same way.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Keep negation words — dropping them is one of the biggest silent
# accuracy killers in sentiment analysis ("not good" becomes "good").
_NEGATIONS = {"not", "no", "nor", "never", "n't", "cannot"}
STOPWORDS = set(stopwords.words("english")) - _NEGATIONS

lemmatizer = WordNetLemmatizer()

CONTRACTIONS = {
    "don't": "do not", "didn't": "did not", "doesn't": "does not",
    "isn't": "is not", "wasn't": "was not", "weren't": "were not",
    "aren't": "are not", "can't": "cannot", "couldn't": "could not",
    "wouldn't": "would not", "shouldn't": "should not", "won't": "will not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "i'm": "i am", "it's": "it is", "that's": "that is",
    "there's": "there is", "what's": "what is", "let's": "let us",
    "i've": "i have", "we've": "we have", "you've": "you have",
    "i'll": "i will", "you'll": "you will", "we'll": "we will",
}


def expand_contractions(text: str) -> str:
    for k, v in CONTRACTIONS.items():
        text = re.sub(r"\b" + re.escape(k) + r"\b", v, text)
    return text


def clean_text(text: str) -> str:
    """Lowercase, expand contractions, strip noise, remove stopwords
    (except negations), lemmatize. Returns a space-joined string ready
    for TfidfVectorizer."""
    text = str(text).lower()
    text = expand_contractions(text)
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"<.*?>", " ", text)                    # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                 # keep letters only
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in STOPWORDS and len(t) > 1
    ]
    return " ".join(tokens)
