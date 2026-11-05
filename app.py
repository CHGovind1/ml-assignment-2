"""
Cart or Close? - Streamlit app for ML Assignment 2
Loads 5 pre-trained classifiers (no retraining) and evaluates them on an uploaded test CSV.
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix,
                             classification_report, roc_curve)

from preprocessing import TARGET, FEATURES, clean

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
GREEN, CLAY = "#1F5C4A", "#C8553D"

st.set_page_config(page_title="Cart or Close?", page_icon="🛒", layout="wide")


@st.cache_resource
def load_models():
    return {n: joblib.load(os.path.join(MODEL_DIR, f)) for n, f in MODEL_FILES.items()}


def evaluate(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    scores = {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, prob) if y.nunique() > 1 else np.nan,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
    return scores, pred, prob


# ---------------- Header ----------------
st.title("🛒 Cart or Close?")
st.write("Will an online shopping session end in a purchase? Upload test sessions, "
         "pick a model, and see how well it predicts the **Revenue** outcome. "
         "Models were trained once offline; nothing is retrained here.")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload test CSV (features + Revenue)", type="csv")
    if os.path.exists("test_data.csv"):
        with open("test_data.csv", "rb") as f:
            st.download_button("Download sample test_data.csv", f, "test_data.csv", "text/csv")
    use_sample = st.checkbox("Use the bundled sample test set", value=uploaded is None)

    st.header("2 · Model")
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES))
    st.caption("Dataset: UCI Online Shoppers Purchasing Intention (12,330 sessions, 17 features).")

# ---------------- Load data ----------------
if uploaded is not None:
    df = pd.read_csv(uploaded)
    source = uploaded.name
elif use_sample and os.path.exists("test_data.csv"):
    df = pd.read_csv("test_data.csv")
    source = "test_data.csv (bundled)"
else:
    st.info("⬅️ Upload a test CSV or tick *Use the bundled sample test set* to begin.")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"The CSV is missing required columns: {', '.join(missing)}")
    st.stop()

try:
    X, y = clean(df)
except Exception as e:
    st.error(f"Could not read the CSV values: {e}")
    st.stop()

models = load_models()
st.caption(f"Evaluating on **{source}** · {len(df):,} sessions · "
           f"{int(y.sum()):,} purchases ({y.mean():.1%})")

tab_model, tab_compare, tab_data = st.tabs(["Selected model", "Compare all models", "Data preview"])

# ---------------- Tab 1: selected model ----------------
with tab_model:
    scores, pred, prob = evaluate(models[model_name], X, y)
    st.subheader(model_name)
    cols = st.columns(6)
    for col, (k, v) in zip(cols, scores.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax,
                    xticklabels=["No purchase", "Purchase"],
                    yticklabels=["No purchase", "Purchase"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.markdown("**ROC curve**")
        fpr, tpr, _ = roc_curve(y, prob)
        fig2, ax2 = plt.subplots(figsize=(4.5, 3.8))
        ax2.plot(fpr, tpr, color=GREEN, lw=2, label=f"AUC = {scores['AUC']:.3f}")
        ax2.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
        ax2.set_xlabel("False positive rate")
        ax2.set_ylabel("True positive rate")
        ax2.legend(loc="lower right")
        st.pyplot(fig2)

    st.markdown("**Classification report**")
    rep = classification_report(y, pred, target_names=["No purchase", "Purchase"],
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(4), width="stretch")

# ---------------- Tab 2: compare ----------------
with tab_compare:
    table = pd.DataFrame({n: evaluate(m, X, y)[0] for n, m in models.items()}).T
    table.index.name = "ML Model Name"
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#CDE6DA"),
                 width="stretch")
    best = table["MCC"].idxmax()
    st.success(f"Best balanced performer on this data (highest MCC): **{best}**")

    metric = st.radio("Metric to plot", list(table.columns), horizontal=True, index=5)
    fig3, ax3 = plt.subplots(figsize=(8, 3.2))
    colors = [CLAY if n == best else GREEN for n in table.index]
    ax3.barh(table.index, table[metric], color=colors)
    ax3.set_xlim(0, 1)
    ax3.set_xlabel(metric)
    ax3.invert_yaxis()
    st.pyplot(fig3)

# ---------------- Tab 3: data ----------------
with tab_data:
    st.dataframe(df.head(50), width="stretch")
    st.download_button("Download predictions for selected model",
                       df.assign(Predicted=pred, Purchase_Probability=prob.round(4))
                         .to_csv(index=False),
                       f"predictions_{model_name.replace(' ', '_').lower()}.csv", "text/csv")
