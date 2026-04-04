from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_customer_churn.csv"
METRICS_PATH = BASE_DIR / "outputs" / "metrics" / "model_metrics.csv"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run src/data_prep.py first."
        )
    return pd.read_csv(path)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }


def main() -> None:
    df = load_dataset(DATA_PATH)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    results = []
    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        results.append(evaluate_model(name, pipeline, X_test, y_test))

    results_df = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(METRICS_PATH, index=False)

    print(results_df.to_string(index=False))
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
