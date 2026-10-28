from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time


def scrape_reviews(url, limit=100):

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
        time.sleep(3)

        reviews = []

        elements = driver.find_elements(
            By.CSS_SELECTOR,
            ".review"
        )

        for element in elements:

            text = element.text.strip()

            if text and text not in reviews:
                reviews.append(text)

            if len(reviews) >= limit:
                break

        return reviews[:limit]

    finally:
        driver.quit()


if __name__ == "__main__":

    url = "https://web-scraping.dev/reviews"

    reviews = scrape_reviews(url, 100)

    print("Reviews found:", len(reviews))

    for review in reviews[:5]:
        print("-", review)