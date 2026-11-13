
"""Prediction utilities for the Fake Review Detection System.

Uses the metadata + text TF-IDF classification pipeline.
"""

from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load the recreated metadata + text model.
MODEL_PATH = PROJECT_ROOT / "model" / "best_model_metadata.pkl"


@lru_cache(maxsize=1)
def _load_model():
    """Load the trained model once per application run."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run 'python -m src.train_model' first."
        )

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError(
            "Invalid model file. Expected a bundle containing "
            "a 'pipeline' entry."
        )

    if bundle.get("version") != 1:
        raise ValueError(
            "Incompatible model version. "
            "Please check the metadata model."
        )

    return bundle


def _prepare_review(review):
    """Convert a review into the features expected by the model.

    Accepts:
        A string containing review text.
        A dictionary containing review text and optional metadata.

    Missing metadata is represented as Unknown or None.
    It is not guessed or fabricated.
    """

    if isinstance(review, str):

        review_text = review.strip()
        review_title = ""

        rating = None
        verified_purchase = "Unknown"
        product_category = "Unknown"

    elif isinstance(review, dict):

        review_text = str(
            review.get("review_text")
            or review.get("review")
            or ""
        ).strip()

        review_title = str(
            review.get("review_title")
            or review.get("title")
            or ""
        ).strip()

        rating = review.get("rating")

        verified_purchase = (
            review.get("verified_purchase") or "Unknown"
        )

        product_category = (
            review.get("product_category") or "Unknown"
        )

    else:
        return None

    if not review_text and not review_title:
        return None

    combined_text = (
        review_title + " " + review_text
    ).strip()

    try:
        rating = (
            float(rating)
            if rating is not None and str(rating).strip()
            else None
        )
    except (ValueError, TypeError):
        rating = None

    return {
        "COMBINED_TEXT": combined_text,
        "VERIFIED_PURCHASE": str(verified_purchase),
        "PRODUCT_CATEGORY": str(product_category),
        "RATING": rating,
    }


def predict_reviews(reviews):
    """Predict whether supplied reviews are genuine or suspicious.

    Accepts:
        A single review string.
        A list of review strings.
        A list of review dictionaries containing optional metadata.

    Returns:
        total
        fake_count
        percentage_fake
        results
    """

    if reviews is None:
        raise ValueError("Reviews cannot be None.")

    if isinstance(reviews, (str, dict)):
        reviews = [reviews]
    else:
        reviews = list(reviews)

    prepared_reviews = []
    original_reviews = []

    for review in reviews:

        prepared = _prepare_review(review)

        if prepared is not None:

            prepared_reviews.append(prepared)
            original_reviews.append(review)

    if not prepared_reviews:
        return {
            "total": 0,
            "fake_count": 0,
            "percentage_fake": 0.0,
            "results": [],
        }

    bundle = _load_model()

    pipeline = bundle["pipeline"]

    # Create a DataFrame with the exact features
    # used during model training.
    prediction_data = pd.DataFrame(
        prepared_reviews,
        columns=[
            "COMBINED_TEXT",
            "VERIFIED_PURCHASE",
            "PRODUCT_CATEGORY",
            "RATING",
        ],
    )

    predictions = pipeline.predict(prediction_data)

    results = []

    for original, prediction in zip(
        original_reviews,
        predictions,
    ):

        if isinstance(original, str):
            review_text = original.strip()
        else:
            review_text = (
                original.get("review_text")
                or original.get("review")
                or original.get("review_title")
                or original.get("title")
                or ""
            )

        results.append({
            "review": review_text,
            "prediction": int(prediction),
        })

    total = len(results)

    fake_count = sum(
        result["prediction"] == 1
        for result in results
    )

    percentage_fake = (
        fake_count / total
    ) * 100

    return {
        "total": total,
        "fake_count": fake_count,
        "percentage_fake": round(percentage_fake, 2),
        "results": results,
    }
