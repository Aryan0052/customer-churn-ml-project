from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.core.config import DATASET_PATH, METADATA_PATH, MODEL_PATH
from app.ml.dataset import build_sample_dataset


FEATURE_COLUMNS = ["height", "skin_tone", "body_type", "gender", "style_preference"]
TARGET_COLUMNS = [
    "fit_recommendation",
    "top_recommendation",
    "color_recommendation",
    "extra_recommendation",
]


@dataclass
class TrainingArtifacts:
    pipeline: Pipeline
    target_columns: List[str]
    training_accuracy: float


def load_training_data(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    if not dataset_path.exists():
        return build_sample_dataset(dataset_path)
    return pd.read_csv(dataset_path)


def train_model(dataset_path: Path = DATASET_PATH) -> TrainingArtifacts:
    df = load_training_data(dataset_path)
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, shuffle=True
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLUMNS)
        ]
    )

    model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            random_state=42,
        )
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    per_target_scores = []
    for index, target_name in enumerate(TARGET_COLUMNS):
        per_target_scores.append(
            accuracy_score(y_test[target_name], predictions[:, index])
        )

    training_accuracy = round(sum(per_target_scores) / len(per_target_scores), 3)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(
        {
            "target_columns": TARGET_COLUMNS,
            "training_accuracy": training_accuracy,
            "model_version": "random-forest-v1",
        },
        METADATA_PATH,
    )

    return TrainingArtifacts(
        pipeline=pipeline,
        target_columns=TARGET_COLUMNS,
        training_accuracy=training_accuracy,
    )


def load_or_train_model() -> tuple[Pipeline, dict]:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        train_model()
    pipeline = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return pipeline, metadata
