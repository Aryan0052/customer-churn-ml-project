from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_customer_churn.csv"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run src/data_prep.py first."
        )
    return pd.read_csv(path)


def save_plot(filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid")
    df = load_dataset(DATA_PATH)

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churn", palette="Set2")
    plt.title("Churn Distribution")
    save_plot("churn_distribution.png")

    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="Contract", hue="Churn", palette="Set2")
    plt.title("Contract Type vs Churn")
    plt.xticks(rotation=15)
    save_plot("contract_vs_churn.png")

    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette="Set3")
    plt.title("Monthly Charges vs Churn")
    save_plot("monthly_charges_vs_churn.png")

    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="Churn", y="tenure", palette="Pastel1")
    plt.title("Tenure vs Churn")
    save_plot("tenure_vs_churn.png")

    plt.figure(figsize=(10, 4))
    sns.countplot(data=df, x="PaymentMethod", hue="Churn", palette="Set2")
    plt.title("Payment Method vs Churn")
    plt.xticks(rotation=20, ha="right")
    save_plot("payment_method_vs_churn.png")

    print(f"Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
