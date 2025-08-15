# ─────────────────────────────────────────────────────────────────────────────
# Flair NER (CoNLL-2003) • Cleaner UI/UX • Imbalance handling
# ─────────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Tuple, Optional
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

# Flair
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings

# Optional NLTK
try:
    import nltk
    from nltk.tokenize import word_tokenize
    _HAS_NLTK = True
except Exception:
    _HAS_NTLK = False
    _HAS_NLTK = False  # keep flag consistent

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit config & CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Flair NER (CoNLL-2003)", layout="wide")

st.markdown("""
<style>
:root{
  --bg: #0f1115;
  --card: #171a21;
  --text: #e8eaf0;
  --muted: #9aa3b2;
  --line: #2a2f3a;
  --per:#fde0e0; --per-b:#f5b5b5;
  --org:#e0f0ff; --org-b:#b5d4f5;
  --geo:#e8f6e8; --geo-b:#b9e3b9;
  --gpe:#f4e8ff; --gpe-b:#d9c6f5;
  --nat:#fff2df; --nat-b:#ffd9a8;
  --tim:#eaf7ff; --tim-b:#bde4ff;
  --art:#fff0f6; --art-b:#ffc3db;
  --eve:#eef9ee; --eve-b:#c5ecc5;
}
html, body { color: var(--text); }
.block-container{ padding-top: 1.2rem; }
hr{ border:0; border-top:1px solid var(--line); margin:1rem 0; }

.badge{padding:2px 8px;border-radius:999px;font-size:.8rem;border:1px solid var(--line);color:var(--muted);display:inline-block;}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;}
.grid{display:grid;gap:12px;}
.grid.cols-3{grid-template-columns:repeat(3,1fr);}
.grid.cols-2{grid-template-columns:repeat(2,1fr);}

.entity{padding:2px 6px;border-radius:6px;margin:0 2px;display:inline-block;border:1px solid;}
.entity.per{background:var(--per);border-color:var(--per-b);color:#5a2d2d;}
.entity.org{background:var(--org);border-color:var(--org-b);color:#243a59;}
.entity.geo{background:var(--geo);border-color:var(--geo-b);color:#264b2a;}
.entity.gpe{background:var(--gpe);border-color:var(--gpe-b);color:#3d2d59;}
.entity.nat{background:var(--nat);border-color:var(--nat-b);color:#5e3f20;}
.entity.tim{background:var(--tim);border-color:var(--tim-b);color:#1c3e56;}
.entity.art{background:var(--art);border-color:var(--art-b);color:#5a2741;}
.entity.eve{background:var(--eve);border-color:var(--eve-b);color:#2f4b2f;}

.small-note{color:var(--muted);font-size:.92rem;}
.kpi{font-size:1.6rem;font-weight:700;}
.kpi-sub{color:var(--muted);font-size:.85rem;margin-top:-6px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "ner_dataset.csv")
RANDOM_STATE = 42

PREFERRED_LABEL_ORDER = [
    "O",
    "B-per", "I-per",
    "B-org", "I-org",
    "B-nat", "I-nat",
    "B-tim", "I-tim",
    "B-geo", "I-geo",
    "B-gpe", "I-gpe",
    "B-art", "I-art",
    "B-eve", "I-eve",
]

ENTITY_CLASS_MAP = {"per":"per","org":"org","geo":"geo","gpe":"gpe","nat":"nat","tim":"tim","art":"art","eve":"eve"}

# ─────────────────────────────────────────────────────────────────────────────
# Caching
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_flair_embedder(model_name: str = "bert-base-cased"):
    return TransformerWordEmbeddings(
        model=model_name,
        layers="-1",
        subtoken_pooling="first",
        fine_tune=False,
    )

@st.cache_data(show_spinner=False)
def load_conll_csv(path_or_buffer) -> pd.DataFrame:
    encodings = ["utf-8", "cp1252", "latin1"]
    last_err, df = None, None

    def _rewind(f):
        if hasattr(f, "seek"):
            try: f.seek(0)
            except Exception: pass

    for enc in encodings:
        try:
            _rewind(path_or_buffer)
            wanted = {"sentence #", "word", "pos", "tag"}
            df = pd.read_csv(
                path_or_buffer,
                encoding=enc,
                engine="python",
                on_bad_lines="skip",
                usecols=lambda c: c.strip().lower() in wanted,
            )
            break
        except Exception as e:
            last_err, df = e, None

    if df is None:
        raise last_err

    cols_lower = {c.lower().strip(): c for c in df.columns}
    required = ["sentence #", "word", "pos", "tag"]
    for r in required:
        if r not in cols_lower:
            matches = [c for c in df.columns if c.lower().strip() == r]
            if matches: cols_lower[r] = matches[0]
            else: raise ValueError(f"Missing required column: {r}")

    df = df.rename(columns={
        cols_lower["sentence #"]: "Sentence #",
        cols_lower["word"]: "Word",
        cols_lower["pos"]: "POS",
        cols_lower["tag"]: "Tag",
    })
    df["Word"] = df["Word"].astype(str)
    df["Tag"]  = df["Tag"].astype(str)
    df["POS"]  = df["POS"].astype(str)

    if df["Sentence #"].isna().any():
        df["Sentence #"] = df["Sentence #"].ffill()

    return df

@st.cache_data(show_spinner=False)
def group_sentences(df: pd.DataFrame, max_sentences: int = 2000, max_tokens_per_sent: int = 50) -> List[Tuple[List[str], List[str]]]:
    grouped = []
    for _, s_df in df.groupby("Sentence #", sort=True):
        words = s_df["Word"].tolist()[:max_tokens_per_sent]
        tags = s_df["Tag"].tolist()[:max_tokens_per_sent]
        grouped.append((words, tags))
        if len(grouped) >= max_sentences: break
    return grouped

@st.cache_data(show_spinner=False)
def embed_tokens(sentences: List[Tuple[List[str], List[str]]], model_name: str):
    embedder = get_flair_embedder(model_name)
    X_list, y_list, labels = [], [], set()

    for words, tags in sentences:
        sent = Sentence(" ".join(words), use_tokenizer=False)
        embedder.embed(sent)
        n = min(len(sent), len(tags))
        for i in range(n):
            X_list.append(sent[i].embedding.cpu().numpy())
            y_list.append(tags[i]); labels.add(tags[i])

    X = np.vstack(X_list) if X_list else np.zeros((0, embedder.embedding_length))
    y = np.array(y_list)

    label_list = [l for l in PREFERRED_LABEL_ORDER if l in labels]
    for l in sorted(labels):
        if l not in label_list: label_list.append(l)

    return X, y, label_list

# ─────────────────────────────────────────────────────────────────────────────
# Imbalance helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_weights_dict(y_train: np.ndarray) -> dict:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return dict(zip(classes, weights))

def downsample_O(X: np.ndarray, y: np.ndarray, max_ratio: int = 3, random_state: int = RANDOM_STATE):
    """Keep at most `max_ratio` times as many 'O' tokens as total non-'O' tokens."""
    idx_O = np.where(y == "O")[0]
    idx_non = np.where(y != "O")[0]
    if len(idx_non) == 0 or len(idx_O) == 0:
        return X, y
    keep_O = min(len(idx_O), max_ratio * len(idx_non))
    rng = np.random.default_rng(random_state)
    sel_O = rng.choice(idx_O, size=keep_O, replace=False)
    keep_idx = np.concatenate([idx_non, sel_O])
    rng.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx]

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: Optional[dict] = None,
    sample_weight: Optional[np.ndarray] = None,
):
    # Logistic Regression: use class_weight only (avoid double-weighting)
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight=class_weight)
    lr.fit(X_train, y_train)

    # Random Forest: use per-sample weights (robust across sklearn versions)
    rf = RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train, sample_weight=sample_weight)
    return lr, rf

# ─────────────────────────────────────────────────────────────────────────────
# Viz + inference
# ─────────────────────────────────────────────────────────────────────────────
def plot_confmat(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    st.pyplot(fig)

def multi_class_roc_auc(y_true, proba, classes):
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_true)
    class_to_col = {c: i for i, c in enumerate(classes)}
    proba_aligned = np.zeros((proba.shape[0], len(lb.classes_)))
    for i, c in enumerate(lb.classes_):
        if c in class_to_col: proba_aligned[:, i] = proba[:, class_to_col[c]]
    fpr, tpr, _ = roc_curve(y_bin.ravel(), proba_aligned.ravel())
    micro_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"micro-average AUC = {micro_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle=":"); ax.legend(loc="lower right")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (OvR, micro-average)")
    st.pyplot(fig)
    return micro_auc

def tokenize_text(txt: str) -> List[str]:
    if _HAS_NLTK:
        try: return word_tokenize(txt)
        except Exception: pass
    return txt.split()

def bio_spans(tokens: List[str], tags: List[str]):
    spans, i = [], 0
    while i < len(tokens):
        tag = tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]; start = i; i += 1
            while i < len(tokens) and tags[i] == f"I-{etype}": i += 1
            spans.append((etype, start, i))
        else: i += 1
    return spans

def render_entities(tokens: List[str], tags: List[str]) -> str:
    spans = bio_spans(tokens, tags)
    idx_to_span = {}
    for etype, s, e in spans:
        for i in range(s, e): idx_to_span[i] = etype
    parts = []
    for i, tok in enumerate(tokens):
        etype = idx_to_span.get(i)
        if etype:
            cls = ENTITY_CLASS_MAP.get(etype, "")
            parts.append(f'<span class="entity {cls}"><b>{tok}</b> <small>({etype})</small></span>')
        else:
            parts.append(tok)
    return " ".join(parts)

# ─────────────────────────────────────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────────────────────────────────────
def page_dataset(df: pd.DataFrame, grouped: List[Tuple[List[str], List[str]]]):
    st.markdown("### NER with Flair Embeddings")
    st.caption("Load a CoNLL-style CSV, view tags, and prepare sentences.")

    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="card"><div class="kpi">{:,}</div><div class="kpi-sub">Rows</div></div>'.format(len(df)), unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card"><div class="kpi">{:,}</div><div class="kpi-sub">Sentences</div></div>'.format(len(grouped)), unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="card"><div class="kpi">{:,}</div><div class="kpi-sub">Unique Tags</div></div>'.format(df["Tag"].nunique()), unsafe_allow_html=True)

    st.markdown("#### Preview")
    st.dataframe(df.head(1000))

    st.markdown("#### Tag distribution")
    tag_counts = df["Tag"].value_counts().reset_index()
    tag_counts.columns = ["Tag", "Count"]
    fig, ax = plt.subplots()
    sns.barplot(data=tag_counts, x="Tag", y="Count", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Counts per Tag")
    st.pyplot(fig)
    st.markdown("<span class='small-note'>Tip: heavy 'O' is normal; use the imbalance options in the sidebar.</span>", unsafe_allow_html=True)

def page_preprocessing(model_name: str, grouped: List[Tuple[List[str], List[str]]]):
    st.markdown("### Embeddings preview")
    st.caption("Flair TransformerWordEmbeddings are applied token-by-token.")
    with st.spinner("Embedding a small sample..."):
        sample = grouped[:3]
        X_sample, y_sample, classes = embed_tokens(sample, model_name)

    left, right = st.columns([1,2])
    with left:
        st.markdown("**Feature matrix**")
        st.write("Shape (tokens × dims):", X_sample.shape)
        st.write("Classes:", classes)
        if X_sample.size:
            st.write("First token (first 10 dims):", np.round(X_sample[0][:10], 4))
    with right:
        if sample:
            s_tokens, s_tags = sample[0]
            st.markdown("**Sample tokens & tags**")
            st.dataframe(pd.DataFrame({"Token": s_tokens, "Tag": s_tags}))

def page_train_eval(model_name: str, grouped: List[Tuple[List[str], List[str]]], test_size: float,
                    use_class_weight: bool, use_downsample_O: bool, max_o_ratio: int,
                    show_excluding_O: bool):
    st.markdown("### Train & evaluate")
    idx = np.arange(len(grouped))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=RANDOM_STATE, shuffle=True)
    train_sents = [grouped[i] for i in train_idx]
    test_sents  = [grouped[i] for i in test_idx]

    with st.spinner("Embedding training set..."):
        X_train, y_train, classes = embed_tokens(train_sents, model_name)
    with st.spinner("Embedding test set..."):
        X_test, y_test, _ = embed_tokens(test_sents, model_name)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.write("Train:", X_train.shape, " Test:", X_test.shape)
    st.write("Classes:", classes)

    # Imbalance handling
    class_weight = None
    sample_weight = None
    if use_class_weight:
        class_weight = compute_class_weights_dict(y_train)
        sample_weight = np.array([class_weight.get(lbl, 1.0) for lbl in y_train])

    if use_downsample_O:
        X_train, y_train = downsample_O(X_train, y_train, max_ratio=max_o_ratio, random_state=RANDOM_STATE)
        if use_class_weight:
            sample_weight = np.array([class_weight.get(lbl, 1.0) for lbl in y_train])

    with st.expander("Class weights (after any downsampling)", expanded=False):
        if use_class_weight:
            st.write(class_weight)
        else:
            st.caption("Class weighting disabled.")

    with st.spinner("Training models..."):
        lr, rf = train_models(X_train, y_train, class_weight=class_weight, sample_weight=sample_weight)

    # Logistic Regression
    st.markdown("#### Logistic Regression")
    y_pred_lr = lr.predict(X_test)
    st.text("Classification report (ALL classes)")
    st.text(classification_report(y_test, y_pred_lr, labels=classes, zero_division=0))
    if show_excluding_O and "O" in classes:
        labels_no_o = [l for l in classes if l != "O"]
        st.text("Classification report (EXCLUDING 'O')")
        st.text(classification_report(y_test, y_pred_lr, labels=labels_no_o, zero_division=0))
    plot_confmat(y_test, y_pred_lr, classes, "Confusion Matrix (LR)")
    if hasattr(lr, "predict_proba"):
        auc_lr = multi_class_roc_auc(y_test, lr.predict_proba(X_test), list(lr.classes_))
        st.markdown(f"**Micro-average ROC-AUC:** {auc_lr:.3f}")

    # Random Forest
    st.markdown("#### Random Forest")
    y_pred_rf = rf.predict(X_test)
    st.text("Classification report (ALL classes)")
    st.text(classification_report(y_test, y_pred_rf, labels=classes, zero_division=0))
    if show_excluding_O and "O" in classes:
        labels_no_o = [l for l in classes if l != "O"]
        st.text("Classification report (EXCLUDING 'O')")
        st.text(classification_report(y_test, y_pred_rf, labels=labels_no_o, zero_division=0))
    plot_confmat(y_test, y_pred_rf, classes, "Confusion Matrix (RF)")
    if hasattr(rf, "predict_proba"):
        auc_rf = multi_class_roc_auc(y_test, rf.predict_proba(X_test), list(rf.classes_))
        st.markdown(f"**Micro-average ROC-AUC:** {auc_rf:.3f}")

    # Comparison
    st.markdown("#### Model comparison (macro averages)")
    rep_lr = classification_report(y_test, y_pred_lr, output_dict=True, zero_division=0)
    rep_rf = classification_report(y_test, y_pred_rf, output_dict=True, zero_division=0)
    comp = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Precision (macro)": [rep_lr.get("macro avg", {}).get("precision", np.nan),
                              rep_rf.get("macro avg", {}).get("precision", np.nan)],
        "Recall (macro)":    [rep_lr.get("macro avg", {}).get("recall", np.nan),
                              rep_rf.get("macro avg", {}).get("recall", np.nan)],
        "F1 (macro)":        [rep_lr.get("macro avg", {}).get("f1-score", np.nan),
                              rep_rf.get("macro avg", {}).get("f1-score", np.nan)],
    })
    st.dataframe(comp.style.format({c: "{:.3f}" for c in comp.columns if c != "Model"}))

def page_predict(model_name: str, model_obj):
    st.markdown("### Try your own text")
    st.caption("Tokens are embedded with Flair and tagged with the selected model.")
    txt = st.text_area("Enter text")

    if st.button("Tag entities"):
        if not txt.strip():
            st.warning("Please enter some text."); return

        tokens = tokenize_text(txt)
        sent = Sentence(" ".join(tokens), use_tokenizer=False)
        embedder = get_flair_embedder(model_name)
        with st.spinner("Embedding..."):
            embedder.embed(sent)
        X = np.vstack([t.embedding.cpu().numpy() for t in sent]) if len(sent) else np.zeros((0, embedder.embedding_length))
        if X.shape[0] == 0:
            st.info("No tokens found."); return

        y_pred = model_obj.predict(X)
        html = render_entities(tokens, y_pred.tolist())
        st.markdown(html, unsafe_allow_html=True)

        st.markdown("#### Token-level output")
        st.dataframe(pd.DataFrame({"Token": tokens, "Pred": y_pred}))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Dataset", "Preprocessing", "Train & Evaluate", "Predict"])

    st.sidebar.title("Settings")
    use_uploaded = st.sidebar.checkbox("Use uploaded CSV", value=False)
    model_name = st.sidebar.selectbox(
        "Flair embedding model",
        ["bert-base-cased", "xlm-roberta-base", "roberta-base"],
        index=0,
        help="Loaded via Flair's TransformerWordEmbeddings."
    )
    max_sentences = st.sidebar.slider("Max sentences", 500, 40000, 1500, step=100)
    max_tokens = st.sidebar.slider("Max tokens per sentence", 10, 128, 40, step=5)
    test_size = st.sidebar.slider("Test size (fraction)", 0.1, 0.4, 0.2, step=0.05)

    st.sidebar.title("Imbalance")
    use_class_weight = st.sidebar.checkbox("Use class weights (balanced)", value=True)
    use_downsample_O = st.sidebar.checkbox("Downsample 'O' in train", value=False)
    max_o_ratio = st.sidebar.slider("Max O : non-O ratio", 1, 10, 3, step=1)
    show_excluding_O = st.sidebar.checkbox("Show reports excluding 'O'", value=True)

    # Load data
    df = None
    if use_uploaded:
        up = st.sidebar.file_uploader("Upload CSV (Sentence #, Word, POS, Tag)", type=["csv"])
        if up is not None:
            try:
                df = load_conll_csv(up)
            except Exception as e:
                st.sidebar.error(f"Upload error: {e}")
    else:
        if os.path.exists(DEFAULT_DATA_PATH):
            try:
                df = load_conll_csv(DEFAULT_DATA_PATH)
                st.sidebar.caption(f"Loaded: {DEFAULT_DATA_PATH}")
            except Exception as e:
                st.sidebar.error(f"Read error: {e}")
        else:
            st.sidebar.warning("Default dataset not found. Upload a CSV in the sidebar.")

    if df is None:
        st.info("Waiting for dataset…"); return

    grouped = group_sentences(df, max_sentences=max_sentences, max_tokens_per_sent=max_tokens)

    if page == "Dataset":
        page_dataset(df, grouped)

    elif page == "Preprocessing":
        page_preprocessing(model_name, grouped)

    elif page == "Train & Evaluate":
        page_train_eval(
            model_name, grouped, test_size,
            use_class_weight=use_class_weight,
            use_downsample_O=use_downsample_O,
            max_o_ratio=max_o_ratio,
            show_excluding_O=show_excluding_O
        )

    elif page == "Predict":
        # quick-fit a small model for demo predictions, respecting imbalance settings
        idx = np.arange(len(grouped))
        if len(idx) < 5:
            st.warning("Need at least 5 sentences to train a quick model for predictions."); return
        train_idx, _ = train_test_split(idx, test_size=0.8, random_state=RANDOM_STATE)
        train_sents = [grouped[i] for i in train_idx]
        X_train, y_train, _ = embed_tokens(train_sents, model_name)

        class_weight = compute_class_weights_dict(y_train) if use_class_weight else None
        sample_weight = np.array([class_weight.get(lbl, 1.0) for lbl in y_train]) if use_class_weight else None
        if use_downsample_O:
            X_train, y_train = downsample_O(X_train, y_train, max_ratio=max_o_ratio, random_state=RANDOM_STATE)
            if use_class_weight:
                sample_weight = np.array([class_weight.get(lbl, 1.0) for lbl in y_train])

        lr, rf = train_models(X_train, y_train, class_weight=class_weight, sample_weight=sample_weight)
        model_choice = st.selectbox("Choose model", ["Logistic Regression", "Random Forest"])
        model_obj = lr if model_choice == "Logistic Regression" else rf
        page_predict(model_name, model_obj)

    elif page == "About":
        page_about()

if __name__ == "__main__":
    main()
