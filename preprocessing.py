"""Shared column definitions and cleaning, used by both training and the Streamlit app."""
import pandas as pd

TARGET = "Revenue"
NUMERIC_COLS = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Weekend",
]
# Integer codes (OS, browser, ...) are IDs, not quantities -> treat as categorical
CATEGORICAL_COLS = ["Month", "VisitorType", "OperatingSystems", "Browser",
                    "Region", "TrafficType"]
FEATURES = NUMERIC_COLS + CATEGORICAL_COLS


def _to_int_bool(s: pd.Series) -> pd.Series:
    """Map True/False, 'TRUE'/'false', 1/0 to 1/0."""
    return s.astype(str).str.strip().str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0}).astype(int)


def clean(df: pd.DataFrame):
    """Return (X, y). y is None if the target column is missing."""
    df = df.copy()
    df["Weekend"] = _to_int_bool(df["Weekend"])
    for c in ["Month", "VisitorType"]:
        df[c] = df[c].astype(str)
    for c in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        df[c] = df[c].astype(int).astype(str)
    y = _to_int_bool(df[TARGET]) if TARGET in df.columns else None
    return df[FEATURES], y
