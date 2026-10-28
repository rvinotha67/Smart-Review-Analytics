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

    reviews = []

    try:
        driver.get(url)
        time.sleep(3)

        for click in range(6):

            elements = driver.find_elements(
                By.CSS_SELECTOR,
                ".review"
            )

            for element in elements:
                text = element.text.strip()

                if text and text not in reviews:
                    reviews.append(text)

                if len(reviews) >= limit:
                    return reviews[:limit]

            print(
                f"After click {click}: "
                f"{len(reviews)} reviews"
            )

            buttons = driver.find_elements(
                By.XPATH,
                "//*[contains("
                "translate(text(),"
                "'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
                "'abcdefghijklmnopqrstuvwxyz'),"
                "'load more')]"
            )

            if not buttons:
                break

            driver.execute_script(
                "arguments[0].click();",
                buttons[0]
            )

            time.sleep(2)

        return reviews[:limit]

    finally:
        driver.quit()


if __name__ == "__main__":

    url = "https://web-scraping.dev/reviews"

    reviews = scrape_reviews(url, 100)

    print("Reviews found:", len(reviews))

    for i, review in enumerate(reviews[:5], 1):
        print(f"{i}. {review}")
