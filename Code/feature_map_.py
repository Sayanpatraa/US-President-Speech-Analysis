#%%
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # ensure transformers doesn't try to use TensorFlow

import re
import pickle
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import WeightedRandomSampler

class IndexedLSTMDataset(Dataset):
    """
    Same as LSTMDataset, but also returns the original index.
    Used so the smart sampler can track per-sample difficulty.
    """
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
            idx  # important: return index for loss tracking
        )


def build_smart_train_loader(X_train, y_train, batch_size, sample_losses=None):
    """
    Build a DataLoader with a weighted sampler.
    If sample_losses is None → uniform sampling.
    Otherwise → weights proportional to loss (harder samples sampled more).
    """
    num_samples = len(y_train)

    if sample_losses is None:
        # start uniform
        weights = np.ones(num_samples, dtype=np.float32)
    else:
        losses = np.array(sample_losses, dtype=np.float32)
        # avoid zeros, normalize
        losses = losses - losses.min() + 1e-6
        weights = losses / losses.sum()

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=num_samples,
        replacement=True
    )

    loader = DataLoader(
        IndexedLSTMDataset(X_train, y_train),
        batch_size=batch_size,
        sampler=sampler
    )
    return loader
# ============================================================
# 0. UTILS: TEXT CLEANING & LABEL NORMALIZATION
# ============================================================

DEFAULT_DATA_PATH = "presidential_statements_scraped.csv"


def clean_text(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"[^a-zA-Z ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_labels(df: pd.DataFrame, label_col: str = "president", min_count: int = 5):
    """
    Replace all presidents with < min_count speeches into label 'Other'.
    Ensures no unseen labels appear in test set.
    """
    vc = df[label_col].value_counts()
    rare = vc[vc < min_count].index

    df = df.copy()
    norm_col = label_col + "_normalized"
    df[norm_col] = df[label_col].replace(rare, "Other")
    return df, norm_col


def load_default_dataset(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}. Make sure it exists.")
    df = pd.read_csv(path)
    print(f"Loaded dataset from {path} with shape {df.shape}")
    return df


def train_test_split_fixed(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    strat_labels = df[label_col] if stratify else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_labels,
    )
    print(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")
    return train_df, test_df


# ============================================================
# 1. PLOTTING HELPERS
# ============================================================

def plot_error_map(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
):
    """Single-model confusion matrix / error map."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Reds",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    return cm


def plot_combined_error_maps(
    baseline: Tuple[np.ndarray, np.ndarray],
    lstm: Tuple[np.ndarray, np.ndarray],
    bert: Tuple[np.ndarray, np.ndarray],
    class_names: List[str],
    save_path: Optional[str] = None,
):
    """
    baseline, lstm, bert = (y_true, y_pred) in encoded label space.
    """
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))

    titles = ["Baseline TF-IDF", "LSTM + Attention", "DistilBERT Fine-Tuned"]
    datasets = [baseline, lstm, bert]

    for ax, (y_true, y_pred), title in zip(axes, datasets, titles):
        if y_true is None or y_pred is None:
            ax.set_title(f"{title}\n(No model)")
            ax.axis("off")
            continue

        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
        sns.heatmap(
            cm,
            ax=ax,
            cmap="Reds",
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def show_misclassified_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    texts: pd.Series,
    label_encoder: LabelEncoder,
    model_name: str,
    max_samples: int = 10,
):
    """
    Print a list of wrongly classified samples with true/pred labels & raw text.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    wrong_idx = np.where(y_true != y_pred)[0]

    print("\n============================")
    print(f" Misclassified Samples ({model_name})")
    print("============================")

    if len(wrong_idx) == 0:
        print("No misclassifications!")
        return

    for i in wrong_idx[:max_samples]:
        true_label = label_encoder.inverse_transform([y_true[i]])[0]
        pred_label = label_encoder.inverse_transform([y_pred[i]])[0]
        print("\n---------------------------")
        print(f"TRUE LABEL : {true_label}")
        print(f"PRED LABEL : {pred_label}")
        print("TEXT SAMPLE:")
        print(texts.iloc[i][:400], "...")


# ============================================================
# 2. ERROR ANALYSIS: OVERLAP & HARDEST CLASSES
# ============================================================

def summarize_error_overlap(
    baseline_pred: np.ndarray,
    lstm_pred: np.ndarray,
    bert_pred: np.ndarray,
    y_true: np.ndarray,
):
    """
    Print counts of overlap of misclassifications across models.
    Assumes all arrays aligned with the same test_df order.
    """
    n = len(y_true)
    idx = np.arange(n)

    wrong_baseline = idx[baseline_pred != y_true]
    wrong_lstm = idx[lstm_pred != y_true]
    wrong_bert = idx[bert_pred != y_true]

    set_b = set(wrong_baseline)
    set_l = set(wrong_lstm)
    set_r = set(wrong_bert)

    all_wrong = set_b | set_l | set_r
    all_correct = set(idx) - all_wrong

    bl = set_b & set_l
    br = set_b & set_r
    lr = set_l & set_r
    blr = set_b & set_l & set_r

    print("\n============================")
    print(" ERROR OVERLAP SUMMARY")
    print("============================")
    print(f"Total test samples            : {n}")
    print(f"Correct by ALL models         : {len(all_correct)}")
    print(f"Wrong by Baseline only        : {len(set_b - (set_l | set_r))}")
    print(f"Wrong by LSTM only            : {len(set_l - (set_b | set_r))}")
    print(f"Wrong by BERT only            : {len(set_r - (set_b | set_l))}")
    print(f"Wrong by Baseline & LSTM only : {len(bl - set_r)}")
    print(f"Wrong by Baseline & BERT only : {len(br - set_l)}")
    print(f"Wrong by LSTM & BERT only     : {len(lr - set_b)}")
    print(f"Wrong by ALL THREE            : {len(blr)}")


def rank_hardest_classes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    top_k: int = 10,
):
    """
    Rank classes by their difficulty for a given model.
    Difficulty = 1 - (correct / total for that class).
    """
    n_classes = len(label_encoder.classes_)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    per_class_counts = cm.sum(axis=1)
    correct = np.diag(cm)

    difficulty = []
    for i in range(n_classes):
        total = per_class_counts[i]
        if total == 0:
            continue
        err_rate = 1.0 - (correct[i] / total)
        difficulty.append((label_encoder.classes_[i], err_rate, total))

    difficulty.sort(key=lambda x: x[1], reverse=True)

    print("\n============================")
    print(" HARDEST-TO-CLASSIFY CLASSES")
    print("============================")
    for name, err_rate, total in difficulty[:top_k]:
        print(f"{name:25s} | error rate={err_rate:.3f} | count={total}")


# ============================================================
# 3. SIMPLE TOKENIZER + PADDING (LSTM)
# ============================================================

class SimpleTokenizer:
    """
    Minimal word-level tokenizer similar to Keras' Tokenizer.
    0 = PAD, 1 = OOV.
    """

    def __init__(self, num_words: int = 50000, oov_token: str = "<OOV>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index: Dict[str, int] = {}
        self.oov_index: int = 1

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        return re.findall(r"\b\w+\b", text)

    def fit_on_texts(self, texts: List[str]) -> None:
        from collections import Counter

        counter = Counter()
        for t in texts:
            tokens = self._tokenize(str(t))
            counter.update(tokens)

        most_common = counter.most_common(max(self.num_words - 2, 0))
        self.word_index = {w: i + 2 for i, (w, _) in enumerate(most_common)}
        self.word_index[self.oov_token] = self.oov_index

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        sequences = []
        for t in texts:
            tokens = self._tokenize(str(t))
            seq = [self.word_index.get(tok, self.oov_index) for tok in tokens]
            sequences.append(seq)
        return sequences


def pad_sequences(
    sequences: List[List[int]],
    maxlen: int = 200,
    padding: str = "pre",
    truncating: str = "pre",
    value: int = 0,
) -> np.ndarray:
    padded = np.full((len(sequences), maxlen), value, dtype=np.int64)

    for idx, seq in enumerate(sequences):
        if len(seq) == 0:
            continue

        if truncating == "pre":
            trunc = seq[-maxlen:]
        else:
            trunc = seq[:maxlen]

        if padding == "pre":
            padded[idx, -len(trunc):] = trunc
        else:
            padded[idx, :len(trunc)] = trunc

    return padded
def save_model(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

def build_tokenizer(
    texts: pd.Series,
    num_words: int = 50000,
    oov_token: str = "<OOV>",
) -> SimpleTokenizer:
    tok = SimpleTokenizer(num_words=num_words, oov_token=oov_token)
    tok.fit_on_texts(texts.astype(str).tolist())
    return tok


def texts_to_padded_sequences(
    tokenizer: SimpleTokenizer,
    texts: pd.Series,
    max_len: int = 200,
) -> np.ndarray:
    seqs = tokenizer.texts_to_sequences(texts.astype(str).tolist())
    return pad_sequences(seqs, maxlen=max_len)


# ============================================================
# 4. LSTM TEXT CLASSIFIER (PyTorch)
# ============================================================

def load_glove_embeddings(
    tokenizer: SimpleTokenizer,
    embedding_dim: int = 100,
    glove_path: str = "glove.6B.100d.txt",
) -> torch.Tensor:
    """
    Build embedding matrix from GloVe file for words in tokenizer.word_index.
    """
    print(f"Loading GloVe from {glove_path} ...")
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector

    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.random.normal(0, 1, (vocab_size, embedding_dim)).astype("float32")

    hit = 0
    for word, i in tokenizer.word_index.items():
        vec = embeddings_index.get(word)
        if vec is not None:
            embedding_matrix[i] = vec
            hit += 1
    print(f"GloVe hits: {hit}/{len(tokenizer.word_index)}")
    return torch.tensor(embedding_matrix, dtype=torch.float32)


class LSTMDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.CrossEntropyLoss(reduction="none")(logits, targets)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def train_lstm_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    label_encoder: LabelEncoder,
    max_words: int = 50000,
    max_len: int = 400,
    embedding_dim: int = 100,
    lstm_units: int = 256,
    batch_size: int = 64,
    epochs: int = 12,
    learning_rate: float = 1e-3,
    random_state: int = 42,
):
    """
    Fine-tuned BiLSTM + Attention + GloVe + OneCycleLR + Early Stopping
    NOW WITH SMART SAMPLER:
      - tracks per-sample loss
      - reweights sampling each epoch towards harder examples
    """



    torch.manual_seed(random_state)
    np.random.seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------------------------------------------------------
    # Label encoding
    y_train = label_encoder.transform(train_df[label_col])
    y_test = label_encoder.transform(test_df[label_col])

    # Tokenize
    tokenizer = build_tokenizer(train_df[text_col], num_words=max_words)
    X_train = texts_to_padded_sequences(tokenizer, train_df[text_col], max_len)
    X_test = texts_to_padded_sequences(tokenizer, test_df[text_col], max_len)

    num_classes = len(np.unique(y_train))
    vocab_size = min(max_words, len(tokenizer.word_index) + 1)

    # --------------------------------------------------------
    # Data loaders (SMART SAMPLER for train, normal for test)
    # --------------------------------------------------------
    num_train = len(y_train)
    # start with uniform losses
    sample_losses = np.ones(num_train, dtype=np.float32)

    train_loader = build_smart_train_loader(
        X_train, y_train, batch_size=batch_size, sample_losses=None
    )

    test_loader = DataLoader(
        LSTMDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )

    # --------------------------------------------------------
    # Load pretrained GloVe
    # --------------------------------------------------------
    embedding_matrix = load_glove_embeddings(tokenizer, embedding_dim=embedding_dim)

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    model = FineTunedLSTM(
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        num_classes=num_classes,
    ).to(device)

    class FocalLoss(nn.Module):
        def __init__(self, gamma=2, alpha=None):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
            self.ce = nn.CrossEntropyLoss(reduction="none")

        def forward(self, logits, targets):
            ce = self.ce(logits, targets)  # per-sample CE
            pt = torch.exp(-ce)
            focal = ((1 - pt) ** self.gamma) * ce

            if self.alpha is not None:
                focal = self.alpha[targets] * focal

            return focal.mean(), ce  # return mean for backprop + per-sample CE for sampler

    criterion = FocalLoss(gamma=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )

    # --------------------------------------------------------
    # Training Loop with Early Stopping + SMART SAMPLER
    # --------------------------------------------------------
    best_val_loss = float("inf")
    patience = 3
    wait = 0

    training_log = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0

        # reset epoch losses and counts
        epoch_sample_losses = np.zeros(num_train, dtype=np.float32)
        epoch_counts = np.zeros(num_train, dtype=np.int32)

        for X_batch, y_batch, idx_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            out = model(X_batch)
            loss_mean, ce_per_sample = criterion(out, y_batch)

            loss_mean.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss_mean.item()

            # update per-sample losses for smart sampler (on CPU)
            ce_np = ce_per_sample.detach().cpu().numpy()
            idx_np = idx_batch.numpy()
            for i, idx in enumerate(idx_np):
                epoch_sample_losses[idx] += ce_np[i]
                epoch_counts[idx] += 1

        # average loss per sample over this epoch
        mask = epoch_counts > 0
        epoch_sample_losses[mask] /= epoch_counts[mask]
        # for samples not seen (just in case), keep previous
        sample_losses[mask] = epoch_sample_losses[mask]

        avg_train = train_loss / len(train_loader)
        training_log["train_loss"].append(avg_train)

        # ----------------- VALIDATION -----------------
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                out = model(X_batch)
                loss_mean, _ = criterion(out, y_batch)
                val_loss += loss_mean.item()

        avg_val = val_loss / len(test_loader)
        training_log["val_loss"].append(avg_val)

        print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        # ----------------- EARLY STOPPING -----------------
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            wait = 0
            torch.save(model.state_dict(), "best_lstm.pt")
        else:
            wait += 1
            if wait >= patience:
                print("\nEarly stopping triggered.")
                break

        # ----------------- REBUILD SMART SAMPLER FOR NEXT EPOCH -----------------
        train_loader = build_smart_train_loader(
            X_train, y_train, batch_size=batch_size, sample_losses=sample_losses
        )

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------
    print("\nLoading best LSTM checkpoint...")
    model.load_state_dict(torch.load("best_lstm.pt"))

    # ============================================================
    # FINAL LSTM EVALUATION: F1, ACCURACY, KAPPA, CONF MATRIX
    # ============================================================
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        cohen_kappa_score,
        confusion_matrix
    )

    def evaluate_lstm_full(model, data_loader, label_encoder, device):
        model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 3:
                    X_batch, y_batch, _ = batch
                else:
                    X_batch, y_batch = batch
                X_batch = X_batch.to(device)
                out = model(X_batch)
                pred = torch.argmax(out, dim=1).cpu().numpy()
                preds.extend(pred.tolist())
                trues.extend(y_batch.numpy().tolist())

        preds = np.array(preds)
        trues = np.array(trues)

        accuracy = accuracy_score(trues, preds)
        f1_macro = f1_score(trues, preds, average="macro", zero_division=0)
        f1_micro = f1_score(trues, preds, average="micro", zero_division=0)
        f1_weighted = f1_score(trues, preds, average="weighted", zero_division=0)
        kappa = cohen_kappa_score(trues, preds)

        print("\n============================")
        print(" OVERALL METRICS")
        print("============================")
        print(f"Accuracy       : {accuracy:.4f}")
        print(f"F1 Macro       : {f1_macro:.4f}")
        print(f"F1 Micro       : {f1_micro:.4f}")
        print(f"F1 Weighted    : {f1_weighted:.4f}")
        print(f"Cohen Kappa    : {kappa:.4f}")

        print("\n============================")
        print(" CLASSIFICATION REPORT")
        print("============================")
        print(classification_report(
            trues,
            preds,
            target_names=label_encoder.classes_,
            zero_division=0
        ))

        cm = confusion_matrix(trues, preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "kappa": kappa,
        }

    print("\n==============================")
    print(" FULL LSTM METRICS REPORT")
    print("==============================")

    train_loader_eval = DataLoader(LSTMDataset(X_train, y_train), batch_size=64)
    test_loader_eval = DataLoader(LSTMDataset(X_test, y_test), batch_size=64)

    print("\n--- TRAIN METRICS ---")
    train_metrics = evaluate_lstm_full(model, train_loader_eval, label_encoder, device)
    print("\n--- TEST METRICS ---")
    test_metrics = evaluate_lstm_full(model, test_loader_eval, label_encoder, device)

    def get_preds(model, data_loader, device):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(device)
                out = model(X_batch)
                pred = torch.argmax(out, dim=1).cpu().numpy()
                preds.append(pred)
                trues.append(y_batch.numpy())
        return np.concatenate(trues), np.concatenate(preds)

    y_true_lstm, y_pred_lstm = get_preds(model, test_loader_eval, device)

    lstm_metrics = test_metrics
    # These are the outputs needed later
    lstm_metrics = test_metrics
    with open("lstm_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("Saved lstm_tokenizer.pkl")

    # --- Save best model weights (already saved earlier but repeat for safety) ---
    torch.save(model.state_dict(), "best_lstm.pt")
    print("Saved best_lstm.pt")

    return model, tokenizer, lstm_metrics, (y_true_lstm, y_pred_lstm)
def predict_text_baseline(
    text: str,
    tfidf: TfidfVectorizer,
    model: LogisticRegression,
    label_encoder: LabelEncoder,
) -> str:
    text = "" if text is None else str(text)
    X_vec = tfidf.transform([text])
    pred_id = model.predict(X_vec)[0]
    return label_encoder.inverse_transform([pred_id])[0]


# ============================================================
# 6. BERT FINE-TUNING (DistilBERT)
# ============================================================

class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def prepare_bert_encodings(
    texts: pd.Series,
    tokenizer,
    max_length: int = 256,
):
    encodings = tokenizer(
        texts.astype(str).tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    return encodings


def train_bert_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    label_encoder: LabelEncoder,
    model_name: str = "distilbert-base-uncased",
    max_length: int = 256,
    num_epochs: int = 4,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    output_dir: str = "bert_finetuned",
    random_state: int = 42,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, Dict, Tuple[np.ndarray, np.ndarray]]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    y_train = label_encoder.transform(train_df[label_col])
    y_test = label_encoder.transform(test_df[label_col])
    num_labels = len(label_encoder.classes_)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    train_encodings = prepare_bert_encodings(train_df[text_col], tokenizer, max_length=max_length)
    test_encodings = prepare_bert_encodings(test_df[text_col], tokenizer, max_length=max_length)

    train_dataset = TextClassificationDataset(train_encodings, y_train)
    test_dataset = TextClassificationDataset(test_encodings, y_test)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=50,
        save_steps=200,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate(test_dataset)
    print("\n[BERT] Eval metrics:", eval_metrics)

    # predictions for confusion matrix, etc.
    pred_output = trainer.predict(test_dataset)
    logits = pred_output.predictions
    y_pred = np.argmax(logits, axis=-1)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_test, y_pred, average="micro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(y_test, y_pred),
    }

    print("\n[BERT] Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved BERT model + tokenizer in {output_dir}/")

    return model, tokenizer, metrics, (y_test, y_pred)


def predict_text_bert(
    text: str,
    model,
    tokenizer,
    label_encoder: LabelEncoder,
    max_length: int = 256,
) -> str:
    text = "" if text is None else str(text)
    inputs = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1).detach().cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    return label_encoder.inverse_transform([pred_id])[0]


# ============================================================
# 7. METRICS TABLE & CSV
# ============================================================

def build_metrics_table(metrics_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    metrics_dict:
        {
          "Baseline": {"accuracy": ..., "f1_macro": ..., ...},
          "LSTM": {...},
          "BERT": {...}
        }
    """
    rows = []
    for model_name, md in metrics_dict.items():
        row = {"model": model_name}
        row.update(md)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def train_baseline_tfidf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    label_encoder: LabelEncoder,
    max_features: int = 5000,
):
    """
    Baseline TF-IDF + Logistic Regression classifier.
    Returns:
      - model
      - tfidf
      - metrics dict
      - (y_true, y_pred)
    """

    # Encode labels once
    y_train = label_encoder.transform(train_df[label_col])
    y_test = label_encoder.transform(test_df[label_col])

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=max_features)
    X_train = tfidf.fit_transform(train_df[text_col])
    X_test = tfidf.transform(test_df[text_col])

    # Train model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_test, y_pred, average="micro", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(y_test, y_pred),
    }

    print("\n[BASELINE] Classification report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))
    with open("baseline_model.pkl", "wb") as f:
        pickle.dump(clf, f)

    with open("tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    print("Saved baseline_model.pkl and tfidf.pkl")

    return clf, tfidf, metrics, (y_test, y_pred)
# ============================================================
# 8. MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    text_col = "content"
    raw_label_col = "president"

    print("\n==============================")
    print(" LOADING & PREPROCESSING DATA ")
    print("==============================")
    df = load_default_dataset()
    df[text_col] = df[text_col].apply(clean_text)

    df, normalized_label_col = normalize_labels(df, label_col=raw_label_col, min_count=5)
    print(f"Using normalized label column: {normalized_label_col}")

    # Global label encoder shared by all models
    le = LabelEncoder()
    le.fit(df[normalized_label_col])
    print("Classes:", list(le.classes_))
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print("Saved label_encoder.pkl")

    # Single fixed train/test split for all models
    train_df, test_df = train_test_split_fixed(
        df,
        text_col=text_col,
        label_col=normalized_label_col,
        test_size=0.2,
        random_state=42,
        stratify=True,
    )

    # --------------------------------------------------------
    # BASELINE MODEL
    # --------------------------------------------------------
    print("\n==============================")
    print(" TRAINING BASELINE (TF-IDF)  ")
    print("==============================")
    baseline_clf, tfidf, baseline_metrics, (y_true_base, y_pred_base) = train_baseline_tfidf(
        train_df,
        test_df,
        text_col=text_col,
        label_col=normalized_label_col,
        label_encoder=le,
        max_features=5000,
    )

    plot_error_map(
        y_true=y_true_base,
        y_pred=y_pred_base,
        class_names=list(le.classes_),
        title="Baseline Logistic Regression – Error Map",
        save_path="cm_baseline.png",
    )

    # --------------------------------------------------------
    # LSTM MODEL
    # --------------------------------------------------------
    print("\n==============================")
    print(" TRAINING LSTM MODEL          ")
    print("==============================")
    lstm_model, lstm_tokenizer, lstm_log, (y_true_lstm, y_pred_lstm) = train_lstm_classifier(
        train_df,
        test_df,
        text_col=text_col,
        label_col=normalized_label_col,
        label_encoder=le,
        max_words=50000,
        max_len=400,
        embedding_dim=100,
        lstm_units=256,
        batch_size=64,
        epochs=12,
        learning_rate=1e-3,
        random_state=42,
    )

    plot_error_map(
        y_true=y_true_lstm,
        y_pred=y_pred_lstm,
        class_names=list(le.classes_),
        title="LSTM + Attention – Error Map",
        save_path="cm_lstm.png",
    )

    # --------------------------------------------------------
    # BERT MODEL
    # --------------------------------------------------------
    print("\n==============================")
    print(" TRAINING DISTILBERT MODEL    ")
    print("==============================")
    bert_model, bert_tokenizer, bert_metrics, (y_true_bert, y_pred_bert) = train_bert_classifier(
        train_df,
        test_df,
        text_col=text_col,
        label_col=normalized_label_col,
        label_encoder=le,
        model_name="distilbert-base-uncased",
        max_length=256,
        num_epochs=4,
        batch_size=8,
        learning_rate=2e-5,
        output_dir="bert_finetuned",
        random_state=42,
    )

    plot_error_map(
        y_true=y_true_bert,
        y_pred=y_pred_bert,
        class_names=list(le.classes_),
        title="DistilBERT – Error Map",
        save_path="cm_bert.png",
    )

    # --------------------------------------------------------
    # COMBINED ERROR MAP
    # --------------------------------------------------------
    print("\n==============================")
    print(" COMBINED ERROR MAP           ")
    print("==============================")
    plot_combined_error_maps(
        baseline=(y_true_base, y_pred_base),
        lstm=(y_true_lstm, y_pred_lstm),
        bert=(y_true_bert, y_pred_bert),
        class_names=list(le.classes_),
        save_path="cm_combined.png",
    )

    # --------------------------------------------------------
    # MISCLASSIFIED SAMPLES
    # --------------------------------------------------------
    print("\n==============================")
    print(" MISCLASSIFIED SAMPLES        ")
    print("==============================")
    show_misclassified_samples(
        y_true_base,
        y_pred_base,
        test_df[text_col],
        le,
        "Baseline TF-IDF",
        max_samples=10,
    )
    show_misclassified_samples(
        y_true_lstm,
        y_pred_lstm,
        test_df[text_col],
        le,
        "LSTM + Attention",
        max_samples=10,
    )
    show_misclassified_samples(
        y_true_bert,
        y_pred_bert,
        test_df[text_col],
        le,
        "DistilBERT",
        max_samples=10,
    )

    # --------------------------------------------------------
    # ERROR OVERLAP & HARDEST CLASSES
    # --------------------------------------------------------
    print("\n==============================")
    print(" ERROR OVERLAP & HARDEST CLS  ")
    print("==============================")

    # all y_true arrays should be identical encoded labels
    # but we can pick one as reference (they all align with test_df)
    summarize_error_overlap(
        baseline_pred=y_pred_base,
        lstm_pred=y_pred_lstm,
        bert_pred=y_pred_bert,
        y_true=y_true_base,
    )

    print("\n[Baseline] Hardest classes:")
    rank_hardest_classes(y_true_base, y_pred_base, le, top_k=10)

    print("\n[LSTM] Hardest classes:")
    rank_hardest_classes(y_true_lstm, y_pred_lstm, le, top_k=10)

    print("\n[BERT] Hardest classes:")
    rank_hardest_classes(y_true_bert, y_pred_bert, le, top_k=10)

    # --------------------------------------------------------
    # METRICS TABLE + CSV
    # --------------------------------------------------------
    print("\n==============================")
    print(" METRICS TABLE                ")
    print("==============================")
    all_metrics = {
        "Baseline": baseline_metrics,
        "LSTM": lstm_log,
        "BERT": bert_metrics,
    }
    metrics_df = build_metrics_table(all_metrics)
    print(metrics_df)
    metrics_df.to_csv("model_metrics.csv", index=False)
    print("Saved model_metrics.csv")
