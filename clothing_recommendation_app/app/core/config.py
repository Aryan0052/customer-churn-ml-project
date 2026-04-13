from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
TEMPLATE_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"

DATASET_PATH = DATA_DIR / "clothing_recommendations.csv"
MODEL_PATH = MODEL_DIR / "recommendation_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.joblib"
FEEDBACK_PATH = DATA_DIR / "feedback_log.jsonl"
