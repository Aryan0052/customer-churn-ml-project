from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "telco_customer_churn.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_customer_churn.csv"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it and place it in data/raw/."
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned.columns = [col.strip() for col in cleaned.columns]

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    cleaned = cleaned.dropna()

    return cleaned


def main() -> None:
    df = load_data(RAW_DATA_PATH)
    cleaned_df = clean_data(df)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("Raw shape:", df.shape)
    print("Cleaned shape:", cleaned_df.shape)
    print(f"Saved cleaned data to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
