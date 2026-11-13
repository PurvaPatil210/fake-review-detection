
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_PATH = PROJECT_ROOT / "data" / "amazon_reviews.txt"

# Load the recreated metadata + text model
MODEL_PATH = PROJECT_ROOT / "model" / "best_model_metadata.pkl"

LABEL_MAPPING = {
    "__label1__": 1,
    "__label2__": 0
}


def main():

    print("\nMODEL DATASET VERIFICATION")
    print("=" * 55)

    # Load dataset
    df = pd.read_csv(
        DATASET_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8"
    )

    # Prepare labels
    df["LABEL"] = df["LABEL"].map(LABEL_MAPPING)

    # Prepare rating
    df["RATING"] = pd.to_numeric(
        df["RATING"],
        errors="coerce"
    )

    # Prepare categorical features
    df["VERIFIED_PURCHASE"] = (
        df["VERIFIED_PURCHASE"].str.strip()
    )

    df["PRODUCT_CATEGORY"] = (
        df["PRODUCT_CATEGORY"].str.strip()
    )

    # Combine review title and review text
    df["COMBINED_TEXT"] = (
        df["REVIEW_TITLE"].str.strip()
        + " "
        + df["REVIEW_TEXT"].str.strip()
    ).str.strip()

    # Remove invalid records
    df = df.dropna(
        subset=["LABEL"]
    )

    df = df[
        (df["RATING"].between(1, 5))
        & (df["COMBINED_TEXT"].str.len() > 0)
    ].copy()

    df["LABEL"] = df["LABEL"].astype(int)

    # Select 10 random records from each class
    genuine = df[
        df["LABEL"] == 0
    ].sample(
        n=10,
        random_state=42
    )

    fake = df[
        df["LABEL"] == 1
    ].sample(
        n=10,
        random_state=42
    )

    test_data = pd.concat(
        [genuine, fake]
    ).sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # Prepare features exactly as during training
    feature_columns = [
        "COMBINED_TEXT",
        "VERIFIED_PURCHASE",
        "PRODUCT_CATEGORY",
        "RATING"
    ]

    X_test = test_data[feature_columns]

    actual = test_data["LABEL"]

    # Load saved metadata model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    model = bundle["pipeline"]

    print("\nLoaded model:", bundle["model_name"])
    print("Model version:", bundle["version"])
    print("Records tested:", len(test_data))

    # Generate predictions
    predictions = model.predict(X_test)

    # Display individual results
    for index, row in test_data.iterrows():

        actual_label = int(actual.iloc[index])
        predicted_label = int(predictions[index])

        actual_name = (
            "Fake / Suspicious"
            if actual_label == 1
            else "Genuine"
        )

        predicted_name = (
            "Fake / Suspicious"
            if predicted_label == 1
            else "Genuine"
        )

        is_correct = actual_label == predicted_label

        print("\n-------------------------------------------")
        print("Review:", row["REVIEW_TEXT"][:180])
        print("Actual:", actual_name)
        print("Predicted:", predicted_name)
        print("Correct:", "Yes" if is_correct else "No")

    # Calculate evaluation metrics
    accuracy = accuracy_score(
        actual,
        predictions
    )

    print("\n===========================================")
    print("VERIFICATION SUMMARY")
    print("===========================================")

    print("Total tested:", len(test_data))
    print(
        "Correct predictions:",
        (actual == predictions).sum()
    )

    print(
        "Incorrect predictions:",
        (actual != predictions).sum()
    )

    print(
        "Accuracy on selected records:",
        round(accuracy * 100, 2),
        "%"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            actual,
            predictions,
            target_names=[
                "Genuine",
                "Fake / Suspicious"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            actual,
            predictions
        )
    )


if __name__ == "__main__":
    main()
