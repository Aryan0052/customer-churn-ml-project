from app.ml.pipeline import train_model


if __name__ == "__main__":
    artifacts = train_model()
    print(
        f"Model trained successfully. Average validation accuracy: {artifacts.training_accuracy}"
    )
