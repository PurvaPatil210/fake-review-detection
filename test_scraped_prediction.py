from src.scraper import get_reviews, ScrapingError
from src.predict import predict_reviews

url = input("Enter product URL: ").strip()

try:
    # Step 1: Scrape reviews
    reviews = get_reviews(url)

    print("\nTotal reviews extracted:", len(reviews))

    # Step 2: Predict reviews
    predictions = predict_reviews(reviews)

    print("\nFAKE REVIEW DETECTION RESULTS")
    print("=" * 50)

    print("Total Reviews:", predictions["total"])
    print("Fake Reviews:", predictions["fake_count"])
    print("Percentage Fake:", predictions["percentage_fake"], "%")

    for index, result in enumerate(predictions["results"], start=1):
        print(f"\nReview {index}")
        print("-" * 40)
        print("Review:", result["review"])
        print("Prediction:", result["prediction"])

except ScrapingError as error:
    print("\nScraping failed:")
    print(error)

except Exception as error:
    print("\nError:", error)