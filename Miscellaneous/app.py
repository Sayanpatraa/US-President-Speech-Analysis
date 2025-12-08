import os
import ast
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from transformers import AutoTokenizer, AutoModelForSequenceClassification

sns.set(style="whitegrid")

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Presidential Rhetoric",
    layout="wide"
)

# ============================================================
# SUPPORT: SimpleTokenizer (for pickled LSTM tokenizer)
# ============================================================

class SimpleTokenizer:
    """
    Minimal word-level tokenizer similar to Keras' Tokenizer.
    0 = PAD, 1 = OOV.
    """
    def __init__(self, num_words: int = 50000, oov_token: str = "<OOV>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {}
        self.oov_index = 1

    def _tokenize(self, text: str):
        import re
        text = text.lower()
        return re.findall(r"\b\w+\b", text)

    def fit_on_texts(self, texts):
        counter = Counter()
        for t in texts:
            tokens = self._tokenize(str(t))
            counter.update(tokens)
        most_common = counter.most_common(max(self.num_words - 2, 0))
        self.word_index = {w: i + 2 for i, (w, _) in enumerate(most_common)}
        self.word_index[self.oov_token] = self.oov_index

    def texts_to_sequences(self, texts):
        sequences = []
        for t in texts:
            tokens = self._tokenize(str(t))
            seq = [self.word_index.get(tok, self.oov_index) for tok in tokens]
            sequences.append(seq)
        return sequences


# ============================================================
# CACHED LOADERS (HYBRID STRATEGY)
# ============================================================

@st.cache_data
def load_dataset():
    """
    Load the enhanced dataset produced by Transformers_topic_Logreg_code.py
    (with tone, strategies, emotions, transformer_sentiment, etc).
    """
    df = pd.read_csv("presidential_statements_enhanced.csv")

    # Parse list-like columns that may be stored as strings
    for col in ["strategies", "emotions"]:
        if col in df.columns and df[col].dtype == object:
            def _to_list(x):
                if isinstance(x, list):
                    return x
                if isinstance(x, str):
                    x = x.strip()
                    if x.startswith("[") and x.endswith("]"):
                        try:
                            return ast.literal_eval(x)
                        except Exception:
                            return [x]
                    if x == "":
                        return []
                return [x]
            df[col] = df[col].apply(_to_list)

    return df


@st.cache_resource
def load_label_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_baseline_cls():
    with open("baseline_model.pkl", "rb") as f:
        baseline = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return baseline, tfidf


@st.cache_resource
def load_lstm_model():
    """
    Lazy-load LSTM classifier & tokenizer.
    """
    with open("lstm_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    class FineTunedLSTM(nn.Module):
        def __init__(self, vocab_size, embedding_matrix, embedding_dim, lstm_units, num_classes):
            super().__init__()
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix,
                freeze=False,
                padding_idx=0,
            )
            self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=lstm_units,
                batch_first=True,
                bidirectional=True,
                num_layers=2,
                dropout=0.3,
            )
            self.attn = nn.Linear(lstm_units * 2, 1)
            self.layernorm = nn.LayerNorm(lstm_units * 2)
            self.dropout = nn.Dropout(0.4)
            self.fc = nn.Linear(lstm_units * 2, num_classes)

        def forward(self, x):
            x = self.embedding(x)
            lstm_out, _ = self.lstm(x)
            weights = torch.softmax(self.attn(lstm_out), dim=1)
            context = (weights * lstm_out).sum(dim=1)
            context = self.layernorm(context)
            context = self.dropout(context)
            return self.fc(context)

    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100
    dummy_embedding_matrix = torch.randn(vocab_size, embedding_dim)  # overwritten by state_dict

    le = load_label_encoder()
    num_classes = len(le.classes_)

    model = FineTunedLSTM(
        vocab_size=vocab_size,
        embedding_matrix=dummy_embedding_matrix,
        embedding_dim=embedding_dim,
        lstm_units=256,
        num_classes=num_classes,
    )

    state_dict = torch.load("best_lstm.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer


@st.cache_resource
def load_bert_model():
    """
    Lazy-load DistilBERT classifier (fine-tuned).
    """
    model = AutoModelForSequenceClassification.from_pretrained("bert_finetuned")
    tokenizer = AutoTokenizer.from_pretrained("bert_finetuned")
    model.eval()
    return model, tokenizer


@st.cache_resource
def load_sentiment_baseline():
    """
    Logistic regression sentiment baseline trained on transformer_sentiment labels.
    """
    with open("sentiment_tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("sentiment_logreg.pkl", "rb") as f:
        clf = pickle.load(f)
    return clf, tfidf


@st.cache_resource
def load_topic_models():
    """
    Load sklearn LDA/NMF topic models + their vectorizers.
    (Gensim model is optional – we just use sklearn here.)
    """
    with open("lda_sklearn.pkl", "rb") as f:
        lda_sklearn = pickle.load(f)
    with open("lda_vectorizer.pkl", "rb") as f:
        lda_vectorizer = pickle.load(f)

    with open("nmf_model.pkl", "rb") as f:
        nmf_model = pickle.load(f)
    with open("nmf_tfidf_vectorizer.pkl", "rb") as f:
        nmf_vectorizer = pickle.load(f)

    return lda_sklearn, lda_vectorizer, nmf_model, nmf_vectorizer


# ============================================================
# PREDICTION HELPERS
# ============================================================

def baseline_predict(text, model, tfidf, le):
    X = tfidf.transform([text])
    pred_id = model.predict(X)[0]
    return le.inverse_transform([pred_id])[0]


def lstm_predict(text, model, tokenizer, le, max_len=400):
    tokens = tokenizer.texts_to_sequences([text])[0]
    if len(tokens) < max_len:
        padded = [0] * (max_len - len(tokens)) + tokens
    else:
        padded = tokens[-max_len:]
    x = torch.tensor([padded], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).numpy()[0]
    pred_id = int(np.argmax(probs))
    return le.inverse_transform([pred_id])[0], probs


def bert_predict(text, model, tokenizer, le, max_length=256):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).numpy()[0]
    pred_id = int(np.argmax(probs))
    return le.inverse_transform([pred_id])[0], probs


def sentiment_predict(text, clf, tfidf):
    X = tfidf.transform([text])
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    classes = clf.classes_
    return pred, classes, proba


# ============================================================
# TOPIC MODEL HELPER
# ============================================================

def get_top_words(model, feature_names, n_top_words=10):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_ids = topic.argsort()[:-n_top_words - 1:-1]
        terms = [feature_names[i] for i in top_ids]
        topics.append((topic_idx, terms))
    return topics


# ============================================================
# LAYOUT
# ============================================================

st.title("🇺🇸 Presidential Rhetoric ")

tab_pred, tab_dash, tab_topics, tab_sent, tab_data = st.tabs(
    [
        "🔮 President Predictor",
        "📊 Rhetoric Dashboard",
        "🗂 Topic Modeling",
        "💬 Sentiment Baseline",
        "📁 Dataset Viewer",
    ]
)

# ============================================================
# TAB 1: PRESIDENT PREDICTOR (A)
# ============================================================
with tab_pred:
    st.subheader("🔮 Predict Which President a Speech Resembles")

    st.markdown("""
This tab uses  **trained models**:
- Baseline **TF-IDF + Logistic Regression**
- **BiLSTM + Attention (GloVe)** with smart sampling & Focal Loss
- **DistilBERT** 
""")

    text_input = st.text_area("Paste a speech or statement:", height=200)

    if st.button("Run Prediction"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            le = load_label_encoder()
            baseline, tfidf = load_baseline_cls()
            lstm_model, lstm_tokenizer = load_lstm_model()
            bert_model, bert_tokenizer = load_bert_model()

            st.markdown("### 📌 Model Outputs")

            # Baseline
            base_pred = baseline_predict(text_input, baseline, tfidf, le)
            st.write(f"**Baseline TF-IDF + LR** → `{base_pred}`")

            # LSTM
            lstm_pred, lstm_probs = lstm_predict(
                text_input, lstm_model, lstm_tokenizer, le
            )
            st.write(f"**BiLSTM + Attention** → `{lstm_pred}`")

            # BERT
            bert_pred, bert_probs = bert_predict(
                text_input, bert_model, bert_tokenizer, le
            )
            st.write(f"**DistilBERT Fine-Tuned** → `{bert_pred}`")

            # Prob table
            st.markdown("### 🔍 LSTM & BERT Probability Table")
            prob_df = pd.DataFrame({
                "Class": le.classes_,
                "LSTM Probability": lstm_probs,
                "BERT Probability": bert_probs,
            })
            st.dataframe(prob_df, use_container_width=True)

    st.markdown("---")
    st.subheader(" Confusion Matrices (from training)")

    cols = st.columns(3)
    if os.path.exists("cm_baseline.png"):
        cols[0].image("cm_baseline.png", caption="Baseline Confusion Matrix", use_container_width=True)
    else:
        cols[0].info("cm_baseline.png not found.")

    if os.path.exists("cm_lstm.png"):
        cols[1].image("cm_lstm.png", caption="LSTM Confusion Matrix", use_container_width=True)
    else:
        cols[1].info("cm_lstm.png not found.")

    if os.path.exists("cm_bert.png"):
        cols[2].image("cm_bert.png", caption="BERT Confusion Matrix", use_container_width=True)
    else:
        cols[2].info("cm_bert.png not found.")

    st.markdown("---")
    st.subheader(" Combined Error Map & Metrics")

    if os.path.exists("cm_combined.png"):
        st.image("cm_combined.png", caption="Combined Error Map", use_container_width=True)
    else:
        st.info("cm_combined.png not found.")

    if os.path.exists("model_metrics.csv"):
        metrics_df = pd.read_csv("model_metrics.csv")
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.info("model_metrics.csv not found.")


# ============================================================
# TAB 2: RHETORIC & SENTIMENT DASHBOARD (B – zero-shot & cardiff outputs)
# ============================================================
with tab_dash:
    st.subheader(" Rhetoric & Sentiment Dashboard")

    df = load_dataset()

    st.markdown("""
This dashboard visualizes:
- **Tone** (combative / conciliatory / neutral-ceremonial / mixed)
- **Rhetorical strategies** (patriotic-appeal, policy-detail, etc.)
- **Emotions** (anger, fear, hope, pride, sadness, trust)
- **Party vs tone**, **President vs tone**, **Emotion trends over time**
""")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        parties = ["All"] + sorted(df["party"].dropna().unique().tolist()) if "party" in df.columns else ["All"]
        party_sel = st.selectbox("Filter by Party", parties, index=0)
    with col2:
        presidents = ["All"] + sorted(df["president"].dropna().unique().tolist())
        pres_sel = st.selectbox("Filter by President", presidents, index=0)

    df_f = df.copy()
    if "party" in df_f.columns and party_sel != "All":
        df_f = df_f[df_f["party"] == party_sel]
    if pres_sel != "All":
        df_f = df_f[df_f["president"] == pres_sel]

    st.caption(f"Rows after filter: {len(df_f)}")

    # Tone
    st.markdown("### 🎭 Tone Distribution")
    if "tone" in df_f.columns:
        tone_counts = df_f["tone"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=tone_counts.index, y=tone_counts.values, ax=ax)
        ax.set_xlabel("Tone")
        ax.set_ylabel("Count")
        ax.set_title("Rhetorical Tone")
        plt.xticks(rotation=30)
        st.pyplot(fig)
    else:
        st.warning("Column 'tone' missing.")

    # Strategies
    st.markdown("###  Strategies (Top 10)")
    if "strategies" in df_f.columns:
        strat_expl = df_f["strategies"].explode()
        strat_counts = strat_expl.value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(y=strat_counts.index, x=strat_counts.values, ax=ax)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Strategy")
        ax.set_title("Top Rhetorical Strategies")
        st.pyplot(fig)
    else:
        st.warning("Column 'strategies' missing.")

    # Emotions
    st.markdown("###  Emotions")
    if "emotions" in df_f.columns:
        emo_expl = df_f["emotions"].explode()
        emo_counts = emo_expl.value_counts()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(x=emo_counts.index, y=emo_counts.values, ax=ax)
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Count")
        ax.set_title("Emotion Distribution")
        plt.xticks(rotation=30)
        st.pyplot(fig)
    else:
        st.warning("Column 'emotions' missing.")

    st.markdown("---")

    # President vs tone heatmap
    st.markdown("###  President vs Tone (Top 15 by speech count)")
    if "tone" in df.columns:
        top_pres = df["president"].value_counts().head(15).index
        subset = df[df["president"].isin(top_pres)]
        ctab = pd.crosstab(subset["president"], subset["tone"])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(ctab, annot=False, cmap="Blues", ax=ax)
        ax.set_xlabel("Tone")
        ax.set_ylabel("President")
        ax.set_title("Tone by President (Top 15)")
        st.pyplot(fig)
    else:
        st.info("Cannot build heatmap – 'tone' missing.")

    # Party vs tone
    st.markdown("### 🏛️ Tone by Party")
    if "party" in df.columns and "tone" in df.columns:
        party_tone = pd.crosstab(df["party"], df["tone"])
        fig, ax = plt.subplots(figsize=(8, 5))
        party_tone.plot(kind="bar", stacked=True, ax=ax)
        ax.set_xlabel("Party")
        ax.set_ylabel("Count")
        ax.set_title("Tone Distribution Across Parties")
        plt.xticks(rotation=30)
        st.pyplot(fig)
    else:
        st.info("Need 'party' and 'tone' to show this chart.")

    st.markdown("---")

    # Emotion trends over time
    st.markdown("### 📈 Emotion Trends Over Time")
    if "date" in df.columns and "emotions" in df.columns:
        df_t = df.copy()
        df_t["year"] = pd.to_datetime(df_t["date"], errors="coerce").dt.year
        emo_t = df_t.explode("emotions").dropna(subset=["year", "emotions"])
        if not emo_t.empty:
            trend = emo_t.groupby(["year", "emotions"]).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(10, 5))
            trend.plot(ax=ax)
            ax.set_xlabel("Year")
            ax.set_ylabel("Count")
            ax.set_title("Emotion Frequency Over Time")
            st.pyplot(fig)
        else:
            st.info("No valid year/emotion data after parsing.")
    else:
        st.info("Need 'date' and 'emotions' columns.")


# ============================================================
# TAB 3: TOPIC MODELING (from Transformers_topic_Logreg_code.py)
# ============================================================
with tab_topics:
    st.subheader("🗂 Topic Modeling – LDA & NMF")

    df = load_dataset()
    lda_sklearn, lda_vectorizer, nmf_model, nmf_vectorizer = load_topic_models()

    mode = st.radio("Choose Topic Model", ["Sklearn LDA", "NMF"], horizontal=True)
    n_top = st.slider("Top words per topic", 5, 25, 10)

    if mode == "Sklearn LDA":
        feature_names = lda_vectorizer.get_feature_names_out()
        topics = get_top_words(lda_sklearn, feature_names, n_top_words=n_top)
    else:
        feature_names = nmf_vectorizer.get_feature_names_out()
        topics = get_top_words(nmf_model, feature_names, n_top_words=n_top)

    st.markdown("### Topic Summaries")
    for idx, terms in topics:
        st.write(f"**Topic {idx}**: " + ", ".join(terms))

    st.markdown("---")
    st.markdown("### Explore Topic Distribution for a Random Speech")

    if "cleaned_content" in df.columns:
        if st.button("Show topics for a random speech"):
            rand_row = df.sample(1).iloc[0]
            text = rand_row["cleaned_content"]
            st.write("**President:**", rand_row["president"])
            st.write("**Excerpt:**")
            st.write(text[:800] + ("..." if len(text) > 800 else ""))

            if mode == "Sklearn LDA":
                X = lda_vectorizer.transform([text])
                topic_dist = lda_sklearn.transform(X)[0]
            else:
                X = nmf_vectorizer.transform([text])
                topic_dist = nmf_model.transform(X)[0]

            topic_df = pd.DataFrame({
                "Topic": list(range(len(topic_dist))),
                "Weight": topic_dist
            }).sort_values("Weight", ascending=False)

            st.dataframe(topic_df, use_container_width=True)
    else:
        st.warning("Column 'cleaned_content' not found in dataset.")


# ============================================================
# TAB 4: SENTIMENT BASELINE (LogReg on transformer_sentiment)
# ============================================================
with tab_sent:
    st.subheader("💬 Sentiment Baseline (Logistic Regression)")

    st.markdown("""
This is the **bag-of-words TF-IDF + Logistic Regression** model trained on
CardiffNLP's `transformer_sentiment` labels (Positive / Negative / Neutral).
""")

    text_input = st.text_area("Enter text to classify sentiment:", height=150, key="sent_text")

    if st.button("Predict Sentiment"):
        if not text_input.strip():
            st.warning("Please type something.")
        else:
            clf, tfidf = load_sentiment_baseline()
            pred, classes, proba = sentiment_predict(text_input, clf, tfidf)
            st.write(f"**Predicted sentiment:** `{pred}`")

            prob_df = pd.DataFrame({
                "Sentiment": classes,
                "Probability": proba
            }).sort_values("Probability", ascending=False)
            st.dataframe(prob_df, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Label Distribution in Training Data (from enhanced dataset)")
    df = load_dataset()
    if "transformer_sentiment" in df.columns:
        counts = df["transformer_sentiment"].value_counts(normalize=False)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        ax.set_title("CardiffNLP Sentiment Labels in Dataset")
        st.pyplot(fig)
    else:
        st.info("Column 'transformer_sentiment' not found in dataset.")


# ============================================================
# TAB 5: DATASET VIEWER
# ============================================================
with tab_data:
    st.subheader("📁 Enhanced Dataset Viewer")

    df = load_dataset()

    st.markdown("Columns available:")
    st.write(list(df.columns))

    st.markdown("#### Sample Rows")
    n_show = st.slider("Number of rows to display", 5, 50, 10)
    st.dataframe(df.head(n_show), use_container_width=True)

    st.markdown("#### Download Enhanced CSV")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download enhanced dataset as CSV",
        data=csv_bytes,
        file_name="presidential_statements_enhanced.csv",
        mime="text/csv",
    )
