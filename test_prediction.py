
from src.predict import predict_reviews

sample_reviews = [
    "The product quality is excellent and delivery was very fast.",
    "Amazing product best product ever buy now highly recommended.",
    "The product stopped working after two days. Very disappointed."
]

try:
    result = predict_reviews(sample_reviews)

    print("\nPREDICTION RESULTS")
    print("==================")

    for item in result["results"]:
        label = (
            "Fake / Suspicious"
            if item["prediction"] == 1
            else "Genuine"
        )

        print(f"\nReview: {item['review']}")
        print(f"Prediction: {label}")

    print("\nSUMMARY")
    print("=======")
    print("Total reviews:", result["total"])
    print("Fake reviews:", result["fake_count"])
    print("Fake percentage:", round(result["percentage_fake"], 2), "%")

except Exception as error:
    print("Prediction test failed:", error)
