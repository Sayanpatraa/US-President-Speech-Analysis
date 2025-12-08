#%% [markdown]
# # Presidential Rhetoric & Polarization – Full NLP Pipeline (PyTorch)
#
# Steps:
# 1. Load + merge APP & Presidency Project data
# 2. Clean transcripts
# 3. Add president - party mapping
# 4. EDA
# 5. Sentiment (VADER + FinBERT-style BERT sentiment)
# 6. Topic Modeling (LDA + coherence)
# 7. ML Models (LogReg, Naive Bayes)
# 8. LSTM Text Classifier (PyTorch)
# 9. Explainability (SHAP) & Plots
#
# NOTE:
# - Paths are hard-coded for now (Ubuntu EC2 layout).
# - Requires: pandas, numpy, nltk, gensim, pyLDAvis, transformers, torch,
#             shap, seaborn, matplotlib, wordcloud.

#%% [markdown]
# ## 0. Imports & Global Config

#%%
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence HF tokenizers fork warning

import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# NLP / preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Sentiment
from nltk.sentiment import SentimentIntensityAnalyzer

# Topic modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Classical ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# SHAP
import shap

# Deep learning (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Transformers (PyTorch)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from IPython.display import display  # for pyLDAvis


try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    print("wordcloud not installed; skipping wordcloud plots.")

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

RANDOM_STATE = 42
TEST_SIZE = 0.2

MAX_VOCAB = 20000
SEQ_LEN = 256           # shorter seq length for LSTM speed
BATCH_SIZE = 32
EPOCHS_LSTM = 3
EMBED_DIM = 128
LSTM_UNITS = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

#%% [markdown]
# ## 1. NLTK Setup & Basic Utilities

#%%
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("vader_lexicon")

STOPWORDS = set(stopwords.words("english"))
LEMM = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Lowercase, remove non-letters, tokenize, remove stopwords, lemmatize.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [LEMM.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


# President → Party mapping
PRESIDENT_TO_PARTY = {
    "George Washington": "Independent",
    "John Adams": "Federalist",
    "Thomas Jefferson": "Democratic-Republican",
    "James Madison": "Democratic-Republican",
    "James Monroe": "Democratic-Republican",
    "John Quincy Adams": "Democratic-Republican",
    "Andrew Jackson": "Democratic",
    "Martin Van Buren": "Democratic",
    "William Henry Harrison": "Whig",
    "John Tyler": "Whig",
    "James K. Polk": "Democratic",
    "Zachary Taylor": "Whig",
    "Millard Fillmore": "Whig",
    "Franklin Pierce": "Democratic",
    "James Buchanan": "Democratic",
    "Abraham Lincoln": "Republican",
    "Andrew Johnson": "Democratic",
    "Ulysses S. Grant": "Republican",
    "Rutherford B. Hayes": "Republican",
    "James A. Garfield": "Republican",
    "Chester A. Arthur": "Republican",
    "Grover Cleveland": "Democratic",
    "Benjamin Harrison": "Republican",
    "William McKinley": "Republican",
    "Theodore Roosevelt": "Republican",
    "William Howard Taft": "Republican",
    "Woodrow Wilson": "Democratic",
    "Warren G. Harding": "Republican",
    "Calvin Coolidge": "Republican",
    "Herbert Hoover": "Republican",
    "Franklin D. Roosevelt": "Democratic",
    "Harry S. Truman": "Democratic",
    "Dwight D. Eisenhower": "Republican",
    "John F. Kennedy": "Democratic",
    "Lyndon B. Johnson": "Democratic",
    "Richard Nixon": "Republican",
    "Gerald Ford": "Republican",
    "Jimmy Carter": "Democratic",
    "Ronald Reagan": "Republican",
    "George H. W. Bush": "Republican",
    "Bill Clinton": "Democratic",
    "George W. Bush": "Republican",
    "Barack Obama": "Democratic",
    "Donald Trump": "Republican",
    "Donald J. Trump": "Republican",
    "Joe Biden": "Democratic",
}

# Optional: term start/end metadata
PRES_TERM = {
    "George Washington": (1789, 1797),
    "John Adams": (1797, 1801),
    "Thomas Jefferson": (1801, 1809),
    "James Madison": (1809, 1817),
    "James Monroe": (1817, 1825),
    "John Quincy Adams": (1825, 1829),
    "Andrew Jackson": (1829, 1837),
    "Martin Van Buren": (1837, 1841),
    "William Henry Harrison": (1841, 1841),
    "John Tyler": (1841, 1845),
    "James K. Polk": (1845, 1849),
    "Zachary Taylor": (1849, 1850),
    "Millard Fillmore": (1850, 1853),
    "Franklin Pierce": (1853, 1857),
    "James Buchanan": (1857, 1861),
    "Abraham Lincoln": (1861, 1865),
    "Andrew Johnson": (1865, 1869),
    "Ulysses S. Grant": (1869, 1877),
    "Rutherford B. Hayes": (1877, 1881),
    "James A. Garfield": (1881, 1881),
    "Chester A. Arthur": (1881, 1885),
    "Grover Cleveland": (1885, 1889),
    "Benjamin Harrison": (1889, 1893),
    "Grover Cleveland": (1893, 1897),
    "William McKinley": (1897, 1901),
    "Theodore Roosevelt": (1901, 1909),
    "William Howard Taft": (1909, 1913),
    "Woodrow Wilson": (1913, 1921),
    "Warren G. Harding": (1921, 1923),
    "Calvin Coolidge": (1923, 1929),
    "Herbert Hoover": (1929, 1933),
    "Franklin D. Roosevelt": (1933, 1945),
    "Harry S. Truman": (1945, 1953),
    "Dwight D. Eisenhower": (1953, 1961),
    "John F. Kennedy": (1961, 1963),
    "Lyndon B. Johnson": (1963, 1969),
    "Richard Nixon": (1969, 1974),
    "Gerald Ford": (1974, 1977),
    "Jimmy Carter": (1977, 1981),
    "Ronald Reagan": (1981, 1989),
    "George H. W. Bush": (1989, 1993),
    "Bill Clinton": (1993, 2001),
    "George W. Bush": (2001, 2009),
    "Barack Obama": (2009, 2017),
    "Donald Trump": (2017, 2021),
    "Joe Biden": (2021, None),
}

#%% [markdown]
# ## 2. Load & Prepare Data

#%%
app_csv_path = "/home/ubuntu/NLP/APP_spoken_addresses.csv"
pres_csv_path = "/home/ubuntu/NLP/presidential_statements_scraped.csv"

TEXT_COL = "transcript"
PRES_COL = "president"
DATE_COL = "date"


def load_and_prepare(app_path: str, pres_path: str) -> pd.DataFrame:
    print("\n=== LOADING DATASETS ===")

    # Load APP (metadata only – no transcript in your CSV)
    df_app = pd.read_csv(app_path)
    df_app.columns = df_app.columns.str.lower()
    print(f"APP loaded: {df_app.shape}")

    # Load scraped speeches (full text)
    df_pres = pd.read_csv(pres_path)
    df_pres.columns = df_pres.columns.str.lower()
    print(f"Presidency scraped loaded: {df_pres.shape}")

    # Validate required columns
    if "content" not in df_pres.columns:
        raise KeyError(f"'content' column not found in scraped dataset: {df_pres.columns}")
    if "president" not in df_pres.columns:
        raise KeyError(f"'president' column not found in scraped dataset: {df_pres.columns}")

    # Normalize schema
    df_pres = df_pres.rename(columns={"content": "transcript", "url": "link"})
    kept_cols = ["title", "link", "date", "president", "transcript"]
    df_pres = df_pres[kept_cols]

    print("\nAfter selecting relevant columns from scraped dataset:")
    print(df_pres.head())

    # Clean missing / bad transcripts
    df_pres = df_pres.dropna(subset=["transcript", "president"])
    df_pres = df_pres[df_pres["transcript"].str.len() > 20]
    df_pres = df_pres.drop_duplicates(subset=["transcript"]).reset_index(drop=True)

    print("\nAfter removing missing / duplicate transcripts:", df_pres.shape)

    # Map party
    df_pres["party"] = df_pres["president"].map(PRESIDENT_TO_PARTY)
    df_pres = df_pres.dropna(subset=["party"]).reset_index(drop=True)

    # Add year
    df_pres["year"] = pd.to_datetime(df_pres["date"], errors="coerce").dt.year

    # Add term start/end (optional enhancement)
    df_pres["term_start"] = df_pres["president"].apply(
        lambda x: PRES_TERM[x][0] if x in PRES_TERM else None
    )
    df_pres["term_end"] = df_pres["president"].apply(
        lambda x: PRES_TERM[x][1] if x in PRES_TERM else None
    )

    # Clean transcripts
    print("\nCleaning transcripts... this may take a few minutes.")
    df_pres["clean_text"] = df_pres["transcript"].astype(str).apply(clean_text)

    # Remove too-short cleaned texts
    df_pres = df_pres[df_pres["clean_text"].str.split().str.len() > 10]
    df_pres = df_pres.reset_index(drop=True)

    print("\n=== FINAL CLEANED DATA ===")
    print(df_pres.shape)
    print(df_pres.head())

    return df_pres


df = load_and_prepare(app_csv_path, pres_csv_path)
df.head()

#%% [markdown]
# ## 3. EDA – Distributions & Basic Plots

#%%
def run_eda(df: pd.DataFrame):
    df["length"] = df["clean_text"].str.split().str.len()

    # Length distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(df["length"], bins=50)
    plt.title("Speech Length Distribution (tokens)")
    plt.xlabel("Length (tokens)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Speeches per party
    plt.figure(figsize=(6, 4))
    sns.countplot(x="party", data=df)
    plt.title("Speeches per Party")
    plt.tight_layout()
    plt.show()

    # Speeches per decade by party
    if "year" in df.columns:
        df["decade"] = (df["year"] // 10) * 10
        plt.figure(figsize=(10, 4))
        sns.countplot(x="decade", hue="party", data=df.dropna(subset=["decade"]))
        plt.title("Speeches per Decade by Party")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Top 30 words
    all_words = " ".join(df["clean_text"]).split()
    counts = Counter(all_words).most_common(30)
    words, freqs = zip(*counts)

    plt.figure(figsize=(10, 4))
    sns.barplot(x=list(words), y=list(freqs))
    plt.xticks(rotation=90)
    plt.title("Top 30 Words in Cleaned Speeches")
    plt.tight_layout()
    plt.show()


run_eda(df)

#%% [markdown]
# ## 4. Sentiment Analysis (VADER + FinBERT-Style BERT Sentiment)

#%%
# 4a. VADER sentiment
sia = SentimentIntensityAnalyzer()
df["vader_compound"] = df["clean_text"].apply(
    lambda x: sia.polarity_scores(x)["compound"]
)

# Sentiment over time (VADER)
if "year" in df.columns:
    plt.figure(figsize=(10, 4))
    df.groupby("year")["vader_compound"].mean().plot(kind="line")
    plt.title("Average VADER Sentiment Over Time")
    plt.ylabel("Compound Score")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    sns.boxplot(x="party", y="vader_compound", data=df)
    plt.title("VADER Sentiment by Party")
    plt.tight_layout()
    plt.show()

# 4b. BERT-based sentiment (ProsusAI/finbert)

print("Loading PyTorch sentiment model (ProsusAI/finbert)...")
bert_model_name = "ProsusAI/finbert"

bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name)
bert_pipeline = pipeline(
    "sentiment-analysis",
    model=bert_model,
    tokenizer=bert_tokenizer,
    truncation=True,
    max_length=512,
    device=0 if torch.cuda.is_available() else -1,
)


def bert_sentiment(text: str):
    """Run BERT sentiment on one speech (truncate for safety)."""
    out = bert_pipeline(text[:4000])[0]
    return out["label"], float(out["score"])


# Apply on subset for efficiency
sample_idx = df.sample(min(200, len(df)), random_state=RANDOM_STATE).index
df.loc[sample_idx, ["bert_label", "bert_score"]] = df.loc[
    sample_idx, "clean_text"
].apply(lambda t: pd.Series(bert_sentiment(t)))

df[["bert_label", "bert_score"]].dropna().head()

#%% [markdown]
# ## 5. Topic Modeling with LDA + Coherence

#%%
def run_lda(df: pd.DataFrame, num_topics: int = 10):
    texts = [t.split() for t in df["clean_text"]]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(t) for t in texts]

    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=RANDOM_STATE,
        passes=10,
        alpha="auto",
    )

    for i, topic in lda_model.print_topics(num_topics=num_topics, num_words=10):
        print(f"Topic {i}: {topic}")

    # Coherence (C_v)
    cm = CoherenceModel(
        model=lda_model,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
    )
    coherence = cm.get_coherence()
    print(f"\nTopic Coherence (C_v): {coherence:.4f}")

    # pyLDAvis visualization (in notebook)
    lda_vis = gensimvis.prepare(lda_model, corpus, dictionary)
    display(pyLDAvis.display(lda_vis))

    return lda_model, dictionary, corpus


lda_model, lda_dictionary, lda_corpus = run_lda(df, num_topics=10)

#%% [markdown]
# ## 6. N-grams & Word Clouds (Optional Enhancements)

#%%
def get_ngrams(texts, n=2, top_k=30):
    vec = CountVectorizer(ngram_range=(n, n), min_df=5).fit(texts)
    bag = vec.transform(texts)
    sums = bag.sum(axis=0)
    freqs = [(word, sums[0, idx]) for word, idx in vec.vocabulary_.items()]
    top = sorted(freqs, key=lambda x: x[1], reverse=True)[:top_k]
    return top

bigrams = get_ngrams(df["clean_text"], n=2, top_k=30)
trigrams = get_ngrams(df["clean_text"], n=3, top_k=30)

print("Top Bigrams:")
for b in bigrams:
    print(b)

print("\nTop Trigrams:")
for t in trigrams:
    print(t)

# Plot bigrams
words_bi, freqs_bi = zip(*bigrams)
plt.figure(figsize=(12, 4))
sns.barplot(x=list(words_bi), y=list(freqs_bi))
plt.xticks(rotation=90)
plt.title("Top 30 Bigrams")
plt.tight_layout()
plt.show()

# Word clouds (if wordcloud installed)
if HAS_WORDCLOUD:
    text_all = " ".join(df["clean_text"])
    wc_all = WordCloud(width=800, height=400, background_color="white").generate(text_all)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc_all, interpolation="bilinear")
    plt.axis("off")
    plt.title("Overall Word Cloud")
    plt.show()

    positive_text = " ".join(df[df["vader_compound"] > 0.4]["clean_text"])
    negative_text = " ".join(df[df["vader_compound"] < -0.4]["clean_text"])

    if positive_text:
        wc_pos = WordCloud(width=800, height=400, background_color="white").generate(positive_text)
        plt.figure(figsize=(12, 6))
        plt.imshow(wc_pos, interpolation="bilinear")
        plt.axis("off")
        plt.title("Positive Speeches Word Cloud (VADER > 0.4)")
        plt.show()

    if negative_text:
        wc_neg = WordCloud(width=800, height=400, background_color="white").generate(negative_text)
        plt.figure(figsize=(12, 6))
        plt.imshow(wc_neg, interpolation="bilinear")
        plt.axis("off")
        plt.title("Negative Speeches Word Cloud (VADER < -0.4)")
        plt.show()
else:
    print("Skipping wordcloud plots (wordcloud not installed).")

#%% [markdown]
# ## 7. Classical ML Models: TF-IDF + Logistic Regression & Naive Bayes

#%%
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=5,
)

X = tfidf.fit_transform(df["clean_text"])
y_str = df["party"].values

label_enc = LabelEncoder()
y = label_enc.fit_transform(y_str)
class_names = label_enc.classes_
print("Classes:", class_names)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Logistic Regression
logreg = LogisticRegression(max_iter=2000, n_jobs=-1)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("F1 (macro):", f1_score(y_test, y_pred_lr, average="macro"))
print(classification_report(y_test, y_pred_lr, target_names=class_names))

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

print("=== Naive Bayes ===")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("F1 (macro):", f1_score(y_test, y_pred_nb, average="macro"))
print(classification_report(y_test, y_pred_nb, target_names=class_names))

#%% [markdown]
# ## 8. PyTorch LSTM Text Classifier (Party Prediction)

#%%
#%% [markdown]
# ## 8. PyTorch LSTM Text Classifier (Party Prediction)

#%%
# ---- 8.1 Build simple vocab over cleaned text ----

tokenizer_simple = lambda s: s.split()

def build_vocab(texts, max_vocab=MAX_VOCAB, min_freq=2):
    counter = Counter()
    for t in texts:
        counter.update(tokenizer_simple(t))
    most_common = counter.most_common(max_vocab)
    stoi = {"<pad>": 0, "<unk>": 1}
    for word, freq in most_common:
        if freq >= min_freq and word not in stoi:
            stoi[word] = len(stoi)
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos

stoi, itos = build_vocab(df["clean_text"].values, max_vocab=MAX_VOCAB, min_freq=2)
vocab_size = len(stoi)
print("Vocab size:", vocab_size)

# ---- 8.2 Encoding utility + Dataset ----

def encode_text(text, stoi, max_len=SEQ_LEN):
    tokens = tokenizer_simple(text)
    ids = [stoi.get(t, stoi["<unk>"]) for t in tokens][:max_len]
    if len(ids) < max_len:
        ids = ids + [stoi["<pad>"]] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)

class SpeechDataset(Dataset):
    def __init__(self, texts, labels, stoi, max_len=SEQ_LEN):
        self.texts = texts
        self.labels = labels
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x_enc = encode_text(self.texts[idx], self.stoi, self.max_len)
        y = self.labels[idx]
        return torch.tensor(x_enc, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# ---- 8.3 Train/val split for LSTM ----

texts_all = df["clean_text"].values
y_all = y  # from earlier label encoding of party

Xtr_txt, Xte_txt, ytr_txt, yte_txt = train_test_split(
    texts_all,
    y_all,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_all,
)

train_dataset = SpeechDataset(Xtr_txt, ytr_txt, stoi, max_len=SEQ_LEN)
val_dataset   = SpeechDataset(Xte_txt, yte_txt, stoi, max_len=SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# ---- 8.4 LSTM model definition ----

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)          # (B, T, E)
        out, _ = self.lstm(emb)          # (B, T, 2H)
        out, _ = torch.max(out, dim=1)   # global max pooling → (B, 2H)
        out = self.dropout(out)
        logits = self.fc(out)            # (B, C)
        return logits

num_classes = len(class_names)
lstm_model = LSTMClassifier(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    hidden_dim=LSTM_UNITS,
    num_classes=num_classes,
    dropout=0.5,
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=1e-3)

# ---- 8.5 Training loop ----

def train_lstm(model, train_loader, val_loader, epochs):
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer_lstm.zero_grad()
            logits = model(xb)                 # <<-- pass integer IDs
            loss = criterion(logits, yb)
            loss.backward()
            optimizer_lstm.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total_examples += xb.size(0)

        train_loss = total_loss / total_examples
        train_acc = total_correct / total_examples

        # Validation
        model.eval()
        val_correct = 0
        val_examples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_examples += xb.size(0)
        val_acc = val_correct / val_examples

        print(
            f"Epoch {epoch}/{epochs} "
            f"- Train Loss: {train_loss:.4f} "
            f"- Train Acc: {train_acc:.4f} "
            f"- Val Acc: {val_acc:.4f}"
        )

train_lstm(lstm_model, train_loader, val_loader, EPOCHS_LSTM)

# ---- 8.6 Final evaluation on val set ----

lstm_model.eval()
all_preds = []
all_true = []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(DEVICE)
        logits = lstm_model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(yb.numpy())

print("=== LSTM Performance (PyTorch) ===")
print("Accuracy:", accuracy_score(all_true, all_preds))
print("F1 (macro):", f1_score(all_true, all_preds, average="macro"))
print(classification_report(all_true, all_preds, target_names=class_names))
#%% [markdown]
# ## 9. Explainability with SHAP (Logistic Regression)

#%%
# Use small subset for speed
sample_size = min(500, X_train.shape[0])
X_train_sample = X_train[:sample_size].toarray()
X_test_sample = X_test[:200].toarray()

explainer = shap.LinearExplainer(logreg, X_train_sample)
shap_values = explainer.shap_values(X_test_sample)

# Handle multiclass / binary robustly
if isinstance(shap_values, list):
    sv_to_plot = shap_values[0]  # first class
else:
    sv_to_plot = shap_values     # already 2D

shap.summary_plot(
    sv_to_plot,
    X_test_sample,
    feature_names=tfidf.get_feature_names_out(),
)

#%% [markdown]
# ## 10. Extra Plots (Confusion Matrix, Sentiment Over Time by Party)

#%%
# Confusion matrix (LogReg)
cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – Logistic Regression")
plt.tight_layout()
plt.show()

# Sentiment over time by party (VADER)
if "year" in df.columns:
    plt.figure(figsize=(10, 4))
    sns.lineplot(
        data=df.dropna(subset=["year"]),
        x="year",
        y="vader_compound",
        hue="party",
        estimator="mean",
    )
    plt.title("Average VADER Sentiment Over Time by Party")
    plt.tight_layout()
    plt.show()

# Sentiment distribution
plt.figure(figsize=(8, 4))
sns.histplot(df["vader_compound"], bins=50, kde=True)
plt.title("Distribution of VADER Sentiment")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.kdeplot(data=df, x="vader_compound", hue="party", fill=True)
plt.title("Sentiment Distribution by Party (VADER)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="length",
    y="vader_compound",
    hue="party",
    alpha=0.5,
)
plt.title("Speech Length vs VADER Sentiment")
plt.tight_layout()
plt.show()

#%% [markdown]
# ## 11. Save Fully Processed Dataset (Optional)

#%%
save_path = input(
    "Enter filename to save processed dataset (e.g. processed_speeches.csv), or leave blank to skip: "
).strip()

if save_path:
    df.to_csv(save_path, index=False)
    print(f"Saved to: {save_path}")
else:
    print("Skipping save.")
