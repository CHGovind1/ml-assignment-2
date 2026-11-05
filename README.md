# Cart or Close? — Predicting Online Purchase Intent

**Live app:** https://ml-assignment-2-govind.streamlit.app/
**Repository:** https://github.com/CHGovind1/ml-assignment-2

## a. Problem statement
An e-commerce site wants to know, from the behaviour recorded during a browsing session, whether that session will end in a purchase. This is framed as a **binary classification** problem: predict `Revenue` (True = purchase, False = no purchase) from 17 session features. Five classifiers are trained, evaluated on a held-out test set with six metrics, and served through an interactive Streamlit app.

## b. Dataset description
- **Source:** UCI Machine Learning Repository — [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (Sakar & Kastro, 2018), CC BY 4.0
- **Instances:** 12,330 sessions (one per user, over one year) · **Features:** 17 · **Target:** `Revenue`
- **Class balance:** 10,422 no-purchase (84.5%) vs 1,908 purchase (15.5%) — imbalanced
- **Missing values:** none

| Group | Features |
|---|---|
| Page visits & time | Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration |
| Google Analytics metrics | BounceRates, ExitRates, PageValues |
| Calendar | SpecialDay, Month, Weekend |
| Visitor / technical | VisitorType, OperatingSystems, Browser, Region, TrafficType |

**Preprocessing:** 80/20 stratified train/test split (`random_state=42`). Numeric features are standardised; Month, VisitorType and the integer-coded OperatingSystems / Browser / Region / TrafficType are one-hot encoded (they are IDs, not quantities). For Gaussian Naive Bayes only the numeric features are used, after a Yeo-Johnson power transform, because one-hot 0/1 columns and heavily skewed durations violate its Gaussian assumption. Preprocessing is stored inside each saved pipeline, so the app applies exactly the training-time transformations. Logistic Regression, Decision Tree and SVM use `class_weight="balanced"` to handle the imbalance.

## c. GitHub repository
```
├── app.py                 # Streamlit app (loads saved models, no retraining)
├── train_models.py        # trains & evaluates the 5 models, saves them + test_data.csv
├── preprocessing.py       # shared column lists and cleaning
├── online_shoppers_intention.csv   # full dataset
├── test_data.csv          # held-out 20% test set (2,466 rows) — upload this in the app
├── model/                 # 5 saved pipelines (.joblib) + metrics.csv
├── requirements.txt
└── .streamlit/config.toml # app theme
```
Run locally: `pip install -r requirements.txt` → `python train_models.py` → `streamlit run app.py`

## d. Models used

### Comparison table (test set, 2,466 sessions)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8410 | 0.8933 | 0.4913 | 0.7435 | 0.5917 | 0.5145 |
| Decision Tree | 0.8508 | 0.9137 | 0.5113 | 0.8298 | 0.6327 | 0.5708 |
| KNN | 0.8759 | 0.8267 | 0.7262 | 0.3194 | 0.4436 | 0.4269 |
| Naive Bayes | 0.8175 | 0.8253 | 0.4450 | 0.7199 | 0.5500 | 0.4636 |
| Support Vector Machine | 0.8824 | 0.8954 | 0.6667 | 0.4817 | 0.5593 | 0.5021 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | With balanced class weights it catches about 74% of buyers, but roughly half of its predicted purchases are false alarms (precision 0.49). Its AUC of 0.89 shows a linear boundary separates the classes reasonably well, largely driven by PageValues. |
| Decision Tree | Best overall: highest AUC (0.914), recall (0.83), F1 (0.633) and MCC (0.571). PageValues alone accounts for about 82% of its feature importance, followed by November sessions and product-page duration. Its shallow threshold splits capture the non-linear thresholds in shopper behaviour; depth and leaf-size limits keep it from overfitting. |
| KNN | High accuracy (0.876) and the best precision (0.73), but recall is only 0.32: in the 85%-negative neighbourhoods most buyers are outvoted. This shows why accuracy alone is misleading on imbalanced data; its MCC (0.43) is the lowest. |
| Naive Bayes | Lowest AUC of the five (0.825), with MCC 0.464. Even after power-transforming the features, the independence assumption fails because many features are correlated (e.g. page counts and their durations). It still has good recall (0.72) at the cost of precision. |
| Support Vector Machine | Highest accuracy (0.882) with a strong AUC (0.895). It is more conservative than Logistic Regression (precision 0.67, recall 0.48), giving fewer false alarms but missing more buyers. It is also the slowest model to train. |
| **Overall winner** | **Decision Tree** — it leads on AUC, Recall, F1 and MCC, the metrics that matter most for an imbalanced target. SVM is the choice if false alarms are costlier than missed buyers. |

## Streamlit app features
- CSV upload for test data (plus a downloadable sample `test_data.csv`)
- Model selection dropdown (5 classifiers)
- All six evaluation metrics for the selected model
- Confusion matrix, ROC curve and classification report
- Side-by-side comparison of all models, and downloadable predictions
