
"""
Platform-aware review scraper for:
Amazon, Flipkart, Myntra and Meesho.

Uses Requests and BeautifulSoup.
Extracts review text and available metadata.

Does not bypass CAPTCHA, login, anti-bot protections,
or other website access restrictions.
"""

import json
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SUPPORTED_PLATFORMS = {
    "amazon.in": "Amazon",
    "amazon.com": "Amazon",
    "flipkart.com": "Flipkart",
    "myntra.com": "Myntra",
    "meesho.com": "Meesho",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

CONNECT_TIMEOUT = 5
READ_TIMEOUT = 15
MAX_REVIEWS = 50
MIN_REVIEW_LENGTH = 15


# --------------------------------------------------
# CUSTOM EXCEPTION
# --------------------------------------------------

class ScrapingError(Exception):
    """Raised when product reviews cannot be extracted."""


# --------------------------------------------------
# PLATFORM DETECTION
# --------------------------------------------------

def detect_platform(url):
    """Identify the shopping platform from its URL."""

    try:
        parsed = urlparse(url.strip())

        if parsed.scheme not in ("http", "https"):
            raise ScrapingError(
                "Only HTTP and HTTPS URLs are supported."
            )

        hostname = (parsed.hostname or "").lower()

        if not hostname:
            raise ScrapingError("Invalid website URL.")

        for domain, platform in SUPPORTED_PLATFORMS.items():

            if hostname == domain or hostname.endswith("." + domain):
                return platform

        raise ScrapingError(
            "Unsupported website. Supported platforms: "
            "Amazon, Flipkart, Myntra and Meesho."
        )

    except ValueError as error:
        raise ScrapingError("Invalid URL format.") from error


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):
    """Clean extracted text without applying review-length rules."""

    if not isinstance(text, str):
        return ""

    text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_review(text):
    """Clean and validate review text."""

    text = clean_text(text)

    if len(text) < MIN_REVIEW_LENGTH:
        return ""

    unwanted = [
        "read more",
        "helpful",
        "report abuse",
        "write a review",
        "see all reviews",
        "customer questions",
        "view more",
    ]

    if text.lower() in unwanted:
        return ""

    return text


# --------------------------------------------------
# METADATA EXTRACTION HELPERS
# --------------------------------------------------

def extract_rating(data):
    """Extract a numeric rating when available."""

    if isinstance(data, dict):
        data = (
            data.get("ratingValue")
            or data.get("value")
        )

    if data is None:
        return None

    match = re.search(r"\d+(?:\.\d+)?", str(data))

    if not match:
        return None

    try:
        rating = float(match.group())

        if 0 < rating <= 5:
            return rating

    except ValueError:
        pass

    return None


def extract_category(soup):
    """Try to extract the product category from page metadata."""

    # Try product structured data.
    for script in soup.find_all(
        "script",
        type="application/ld+json"
    ):

        try:
            data = json.loads(
                script.string or script.get_text()
            )

            categories = find_product_category(data)

            if categories:
                return categories

        except (json.JSONDecodeError, TypeError):
            continue

    # Try breadcrumb navigation.
    breadcrumb_selectors = [
        '[aria-label="Breadcrumb"]',
        '[class*="breadcrumb"]',
        '[class*="Breadcrumb"]',
    ]

    for selector in breadcrumb_selectors:

        element = soup.select_one(selector)

        if element:
            category = clean_text(
                element.get_text(" ", strip=True)
            )

            if category:
                return category[:150]

    return None


def find_product_category(data):
    """Search JSON-LD data for a product category."""

    if isinstance(data, dict):

        category = data.get("category")

        if isinstance(category, str) and category.strip():
            return category.strip()

        for value in data.values():

            result = find_product_category(value)

            if result:
                return result

    elif isinstance(data, list):

        for item in data:

            result = find_product_category(item)

            if result:
                return result

    return None


def extract_verified_purchase(element):
    """Detect visible verified-purchase information."""

    text = element.get_text(" ", strip=True).lower()

    verified_indicators = [
        "verified purchase",
        "verified purchaser",
        "verified buyer",
    ]

    for indicator in verified_indicators:

        if indicator in text:
            return "Y"

    # Do not assume that absence of a badge means N.
    return "Unknown"


def extract_element_rating(element):
    """Extract rating from common HTML rating attributes."""

    selectors = [
        '[data-hook="review-star-rating"]',
        '[data-hook="cmps-review-star-rating"]',
        '[class*="rating"]',
        '[class*="Rating"]',
        '[aria-label*="out of 5"]',
        '[aria-label*="stars"]',
    ]

    for selector in selectors:

        rating_element = element.select_one(selector)

        if not rating_element:
            continue

        possible_values = [
            rating_element.get("aria-label"),
            rating_element.get("title"),
            rating_element.get("data-rating"),
            rating_element.get_text(" ", strip=True),
        ]

        for value in possible_values:

            rating = extract_rating(value)

            if rating is not None:
                return rating

    return None


def extract_review_title(element):
    """Extract a review heading when available."""

    selectors = [
        '[data-hook="review-title"]',
        '[class*="review-title"]',
        '[class*="reviewTitle"]',
        '[class*="ReviewTitle"]',
        '[itemprop="name"]',
    ]

    for selector in selectors:

        title_element = element.select_one(selector)

        if title_element:

            title = clean_text(
                title_element.get_text(" ", strip=True)
            )

            if title:
                return title

    return ""


def create_review(
    review_text,
    review_title=None,
    rating=None,
    verified_purchase="Unknown",
    product_category=None
):
    """Create a structured review record."""

    review_text = clean_review(review_text)

    if not review_text:
        return None

    if rating is not None:
        rating = extract_rating(rating)

    return {
        "review_text": review_text,
        "review_title": clean_text(review_title or ""),
        "rating": rating,
        "verified_purchase": verified_purchase or "Unknown",
        "product_category": product_category,
    }


# --------------------------------------------------
# JSON-LD REVIEW EXTRACTION
# --------------------------------------------------

def extract_jsonld_reviews(soup):
    """Extract structured Review objects from JSON-LD."""

    reviews = []

    def inspect_data(data):

        if isinstance(data, dict):

            review_type = data.get("@type", [])

            if isinstance(review_type, str):
                review_type = [review_type]

            if "Review" in review_type:

                body = data.get("reviewBody", "")

                title = data.get("name", "")

                rating = extract_rating(
                    data.get("reviewRating")
                )

                author = data.get("author")

                # Extract review content only.
                # Author information is not used as a feature.
                if isinstance(author, dict):
                    author = author.get("name", "")

                review = create_review(
                    review_text=body,
                    review_title=title,
                    rating=rating,
                )

                if review:
                    reviews.append(review)

            for value in data.values():
                inspect_data(value)

        elif isinstance(data, list):

            for item in data:
                inspect_data(item)

    for script in soup.find_all(
        "script",
        type="application/ld+json"
    ):

        try:

            data = json.loads(
                script.string or script.get_text()
            )

            inspect_data(data)

        except (json.JSONDecodeError, TypeError):
            continue

    return reviews


# --------------------------------------------------
# PLATFORM-SPECIFIC REVIEW CONTAINERS
# --------------------------------------------------

PLATFORM_REVIEW_SELECTORS = {

    "Amazon": [
        '[data-hook="review"]',
    ],

    "Flipkart": [
        '[class*="_27M-vq"]',
        '[class*="col EPCmJX"]',
        '[class*="review"]',
        '[data-testid*="review"]',
    ],

    "Myntra": [
        '[class*="userReview"]',
        '[class*="user-review"]',
        '[class*="review"]',
        '[data-testid*="review"]',
    ],

    "Meesho": [
        '[class*="review"]',
        '[class*="Review"]',
        '[data-testid*="review"]',
    ],
}


def extract_review_body(element, platform):
    """Extract review body from a review container."""

    selectors = [
        '[data-hook="review-body"]',
        '[itemprop="reviewBody"]',
        '[class*="reviewText"]',
        '[class*="review-text"]',
        '[class*="reviewContent"]',
        '[class*="review-description"]',
        '[class*="userReview"]',
        '[class*="user-review"]',
    ]

    for selector in selectors:

        body_element = element.select_one(selector)

        if body_element:

            text = clean_review(
                body_element.get_text(" ", strip=True)
            )

            if text:
                return text

    # Amazon has a specific review body selector.
    if platform == "Amazon":
        return clean_review(
            element.get_text(" ", strip=True)
        )

    # Other platforms: use the container text only
    # when a specific review body can be identified.
    return ""


def extract_platform_reviews(soup, platform, category):
    """Extract structured review records using platform selectors."""

    reviews = []

    selectors = PLATFORM_REVIEW_SELECTORS.get(
        platform,
        []
    )

    for selector in selectors:

        elements = soup.select(selector)

        for element in elements:

            body = extract_review_body(
                element,
                platform
            )

            if not body:
                continue

            title = extract_review_title(element)

            rating = extract_element_rating(element)

            verified = extract_verified_purchase(element)

            review = create_review(
                review_text=body,
                review_title=title,
                rating=rating,
                verified_purchase=verified,
                product_category=category,
            )

            if review:
                reviews.append(review)

        if reviews:
            break

    return reviews


# --------------------------------------------------
# MAIN REVIEW EXTRACTION
# --------------------------------------------------

def get_reviews(url):
    """
    Fetch and extract structured product reviews.

    Returns:
        list of dictionaries containing:
            review_text
            review_title
            rating
            verified_purchase
            product_category

    Raises:
        ScrapingError if reviews cannot be extracted.
    """

    platform = detect_platform(url)

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            allow_redirects=True
        )

        response.raise_for_status()

        # Check redirects.
        redirected_platform = detect_platform(response.url)

        if redirected_platform != platform:

            raise ScrapingError(
                "The webpage redirected to a different platform."
            )

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        if "html" not in content_type:

            raise ScrapingError(
                "The website did not return an HTML webpage."
            )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        page_text = soup.get_text(
            " ",
            strip=True
        ).lower()

        # Detect access restrictions.
        blocked_indicators = [
            "captcha",
            "robot check",
            "access denied",
            "verify you are human",
            "unusual traffic",
        ]

        if any(
            indicator in page_text
            for indicator in blocked_indicators
        ):

            raise ScrapingError(
                f"{platform} restricted automated access. "
                "Please use an authorized review source."
            )

        # Extract product category where available.
        category = extract_category(soup)

        # First try JSON-LD structured data.
        reviews = extract_jsonld_reviews(soup)

        # Add page category if structured reviews lack it.
        for review in reviews:

            if not review["product_category"]:
                review["product_category"] = category

        # Try platform-specific HTML selectors.
        if not reviews:

            reviews = extract_platform_reviews(
                soup,
                platform,
                category
            )

        # Remove duplicates.
        unique_reviews = []
        seen = set()

        for review in reviews:

            normalized = (
                review["review_text"].casefold()
            )

            if normalized in seen:
                continue

            seen.add(normalized)
            unique_reviews.append(review)

            if len(unique_reviews) >= MAX_REVIEWS:
                break

        if not unique_reviews:

            raise ScrapingError(
                f"No customer reviews could be extracted from "
                f"{platform}. The page may load reviews dynamically "
                "or use a different HTML structure. "
                "Sample reviews will not be substituted."
            )

        return unique_reviews

    except ScrapingError:
        raise

    except requests.exceptions.Timeout as error:

        raise ScrapingError(
            f"{platform} took too long to respond. Try again later."
        ) from error

    except requests.exceptions.HTTPError as error:

        status = (
            error.response.status_code
            if error.response is not None
            else "Unknown"
        )

        raise ScrapingError(
            f"{platform} returned HTTP error {status}."
        ) from error

    except requests.exceptions.ConnectionError as error:

        raise ScrapingError(
            f"Could not connect to {platform}. "
            "Check your internet connection."
        ) from error

    except requests.exceptions.RequestException as error:

        raise ScrapingError(
            f"Unable to retrieve the webpage: {error}"
        ) from error
