
from src.scraper import get_reviews, ScrapingError


url = input("Enter product URL: ").strip()

try:

    reviews = get_reviews(url)

    print("\nSCRAPING RESULTS")
    print("=" * 50)

    print("Total reviews extracted:", len(reviews))

    for index, review in enumerate(reviews, start=1):

        print(f"\nReview {index}")
        print("-" * 40)

        for key, value in review.items():
            print(f"{key}: {value}")

except ScrapingError as error:

    print("\nScraping failed:")
    print(error)
