"""
Train 5 classifiers on the UCI Online Shoppers Purchasing Intention dataset.
Outputs:
  model/<name>.joblib   - full pipelines (preprocessing + classifier)
  model/metrics.csv     - evaluation metrics on the held-out test set
  test_data.csv         - raw held-out test set (features + Revenue) for the Streamlit app
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

from preprocessing import TARGET, NUMERIC_COLS, CATEGORICAL_COLS, clean

SEED = 42
DATA_FILE = "online_shoppers_intention.csv"
MODEL_DIR = "model"

# ---------- 1. Load & split ----------
raw = pd.read_csv(DATA_FILE)
print(f"Dataset shape: {raw.shape}")
train_raw, test_raw = train_test_split(raw, test_size=0.2, stratify=raw[TARGET],
                                       random_state=SEED)
test_raw.to_csv("test_data.csv", index=False)
print(f"Train: {len(train_raw)} rows | Test: {len(test_raw)} rows -> test_data.csv")

X_train, y_train = clean(train_raw)
X_test, y_test = clean(test_raw)

# ---------- 2. Preprocessing (fit only on training data, inside each pipeline) ----------
def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
    ])

def make_nb_preprocessor():
    # GaussianNB assumes each feature is roughly normal. One-hot 0/1 columns and the
    # heavily skewed duration/page counts break that assumption, so for NB we use
    # only the numeric columns, made more Gaussian with a Yeo-Johnson power transform.
    return ColumnTransformer([("num", PowerTransformer(), NUMERIC_COLS)])

# ---------- 3. Models ----------
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                            class_weight="balanced", random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
    "Naive Bayes": GaussianNB(),
    # Calibrated wrapper gives predict_proba (needed for AUC) on all recent sklearn versions
    "Support Vector Machine": CalibratedClassifierCV(
        SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=SEED),
        ensemble=False),
}

os.makedirs(MODEL_DIR, exist_ok=True)
rows = []
for name, clf in classifiers.items():
    prep = make_nb_preprocessor() if name == "Naive Bayes" else make_preprocessor()
    pipe = Pipeline([("prep", prep), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    })
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, os.path.join(MODEL_DIR, fname))
    print(f"Saved {fname}")

metrics = pd.DataFrame(rows).round(4)
metrics.to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index=False)
print("\n=== Test-set metrics ===")
print(metrics.to_string(index=False))
