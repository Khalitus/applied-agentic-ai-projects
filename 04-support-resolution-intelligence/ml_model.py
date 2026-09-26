import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from config import MODEL_PATH
from database import load_modeling_frame


TARGET = "escalated"

NUMERIC_FEATURES = [
    "sentiment_score",
    "previous_tickets_90d",
    "days_since_purchase",
    "account_age_months",
    "total_orders",
    "lifetime_value",
    "warranty_months",
    "unit_price",
]

CATEGORICAL_FEATURES = [
    "channel",
    "priority",
    "issue_type",
    "region",
    "customer_tier",
    "category",
]

LEAKAGE_COLUMNS = [
    "first_response_minutes",
    "resolution_hours",
    "resolution_text",
]


def build_preprocessor():
    # TODO Task 4:
    # 1. median-impute numeric columns
    # 2. most-frequent-impute categorical columns
    # 3. one-hot encode categorical columns with handle_unknown="ignore"
    # 4. combine both branches with ColumnTransformer
    raise NotImplementedError


def build_pipeline():
    preprocessor = build_preprocessor()

    # TODO Task 5: configure an XGBClassifier.
    # Start simple; tune only after you have a trustworthy validation baseline.
    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


def train_and_evaluate():
    df = load_modeling_frame()

    forbidden = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if forbidden:
        print("Leakage audit: these columns exist but must not be model features:", forbidden)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    pipeline = build_pipeline()

    # TODO Task 5: add 5-fold stratified cross-validation on the training split.
    # Report mean ROC-AUC before fitting the final training model.

    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    print("\nTest ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
    print("\nClassification report:\n", classification_report(y_test, predictions))
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved model: {MODEL_PATH}")


def predict_escalation(ticket_features):
    pipeline = joblib.load(MODEL_PATH)
    row = pd.DataFrame([ticket_features])
    probability = float(pipeline.predict_proba(row)[0, 1])
    return {
        "probability": probability,
        "prediction": int(probability >= 0.50),
    }


if __name__ == "__main__":
    train_and_evaluate()
