
import os
import json
import pickle
import pandas as pd

from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "amazon_reviews.txt"
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "best_model_metadata.pkl"
METRICS_PATH = MODEL_DIR / "model_metrics_metadata.json"


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_dataset():

    print("\nLoading dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        encoding="utf-8",
        on_bad_lines="skip"
    )

    required_columns = [
        "LABEL",
        "RATING",
        "VERIFIED_PURCHASE",
        "PRODUCT_CATEGORY",
        "REVIEW_TITLE",
        "REVIEW_TEXT"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df[required_columns].copy()

    # Remove records without a label or review text
    df.dropna(
        subset=["LABEL", "REVIEW_TEXT"],
        inplace=True
    )

    # Convert labels into numerical values
    # __label1__ = Fake/Suspicious (1)
    # __label2__ = Genuine (0)

    label_mapping = {
        "__label1__": 1,
        "__label2__": 0
    }

    df["LABEL"] = df["LABEL"].map(label_mapping)

    df.dropna(subset=["LABEL"], inplace=True)

    df["LABEL"] = df["LABEL"].astype(int)

    # Combine review title and review body
    df["COMBINED_TEXT"] = (
        df["REVIEW_TITLE"].fillna("").astype(str)
        + " "
        + df["REVIEW_TEXT"].fillna("").astype(str)
    )

    # Clean categorical columns
    df["VERIFIED_PURCHASE"] = (
        df["VERIFIED_PURCHASE"]
        .fillna("Unknown")
        .astype(str)
    )

    df["PRODUCT_CATEGORY"] = (
        df["PRODUCT_CATEGORY"]
        .fillna("Unknown")
        .astype(str)
    )

    # Convert rating into numeric values
    df["RATING"] = pd.to_numeric(
        df["RATING"],
        errors="coerce"
    )

    # Remove empty reviews
    df["COMBINED_TEXT"] = (
        df["COMBINED_TEXT"].str.strip()
    )

    df = df[
        df["COMBINED_TEXT"].str.len() > 0
    ].copy()

    print("\nDataset loaded successfully.")
    print("Total records:", len(df))

    print("\nClass distribution:")
    print(df["LABEL"].value_counts())

    return df


# --------------------------------------------------
# FEATURE PREPARATION
# --------------------------------------------------

def prepare_features(df):

    X = df[
        [
            "COMBINED_TEXT",
            "VERIFIED_PURCHASE",
            "PRODUCT_CATEGORY",
            "RATING"
        ]
    ]

    y = df["LABEL"]

    return X, y


# --------------------------------------------------
# BUILD PREPROCESSING PIPELINE
# --------------------------------------------------

def create_preprocessor():

    text_features = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=100000,
        sublinear_tf=True
    )

    categorical_features = [
        "VERIFIED_PURCHASE",
        "PRODUCT_CATEGORY"
    ]

    numeric_features = [
        "RATING"
    ]

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "text",
                text_features,
                "COMBINED_TEXT"
            ),

            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            )
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            )
                        )
                    ]
                ),
                categorical_features
            ),

            (
                "numeric",
                SimpleImputer(
                    strategy="median"
                ),
                numeric_features
            )

        ]
    )

    return preprocessor


# --------------------------------------------------
# MODEL DEFINITIONS
# --------------------------------------------------

def get_models():

    models = {

        "Naive Bayes": MultinomialNB(
            alpha=0.5
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

        "Linear SVM": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42
        )

    }

    return models


# --------------------------------------------------
# TRAIN AND EVALUATE MODELS
# --------------------------------------------------

def train_models(X_train, X_test, y_train, y_test):

    models = get_models()

    results = {}
    trained_pipelines = {}

    best_model = None
    best_model_name = None
    best_f1 = -1

    print("\nStarting model training...")

    for model_name, classifier in models.items():

        print("\n--------------------------------")
        print("Training:", model_name)
        print("--------------------------------")

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor()
                ),
                (
                    "classifier",
                    classifier
                )
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0
        )

        results[model_name] = {
            "accuracy": float(accuracy),
            "fake_precision": float(precision),
            "fake_recall": float(recall),
            "fake_f1_score": float(f1)
        }

        trained_pipelines[model_name] = pipeline

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"Fake F1:   {f1:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                target_names=[
                    "Genuine",
                    "Fake/Suspicious"
                ],
                zero_division=0
            )
        )

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

        # Select the model using fake-class F1
        if f1 > best_f1:

            best_f1 = f1
            best_model = pipeline
            best_model_name = model_name

    return (
        best_model,
        best_model_name,
        results
    )


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

def save_model(best_model, best_model_name, results):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_bundle = {

        "pipeline": best_model,

        "model_name": best_model_name,

        "label_mapping": {
            0: "Genuine",
            1: "Fake/Suspicious"
        },

        "feature_columns": [
            "COMBINED_TEXT",
            "VERIFIED_PURCHASE",
            "PRODUCT_CATEGORY",
            "RATING"
        ],

        "dataset_name": "amazon_reviews.txt",

        "version": 1,

        "created_at": datetime.now().isoformat()

    }

    with open(MODEL_PATH, "wb") as file:

        pickle.dump(
            model_bundle,
            file
        )

    metrics = {

        "selected_model": best_model_name,

        "selection_metric": "Fake-class F1-score",

        "model_results": results,

        "dataset": "amazon_reviews.txt",

        "training_date": datetime.now().isoformat()

    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print("\nModel saved successfully!")
    print("Model path:", MODEL_PATH)
    print("Metrics path:", METRICS_PATH)


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    print("====================================")
    print(" FAKE REVIEW DETECTION - MODEL TRAINING")
    print("====================================")

    df = load_dataset()

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    best_model, best_model_name, results = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\n====================================")
    print("BEST MODEL:", best_model_name)
    print("====================================")

    save_model(
        best_model,
        best_model_name,
        results
    )

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
