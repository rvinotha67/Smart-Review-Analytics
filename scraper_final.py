import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


def scrape_reviews(product_url, limit=100):

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9"
    }

    try:
        response = requests.get(
            product_url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

    except requests.RequestException:
        return []

    # Amazon blocking / CAPTCHA detection
    page_text = response.text.lower()

    if "captcha" in page_text:
        return []

    if "robot check" in page_text:
        return []

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    reviews = []

    elements = soup.select(
        '[data-hook="review-body"]'
    )

    for element in elements:

        text = element.get_text(
            " ",
            strip=True
        )

        if text and text not in reviews:
            reviews.append(text)

        if len(reviews) >= limit:
            break

    return reviews[:limit]