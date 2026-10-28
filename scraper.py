import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def scrape_reviews(product_url, limit=100):

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=options)

    reviews = []

    try:
        print(f"Opening: {product_url}")

        driver.get(product_url)

        wait = WebDriverWait(driver, 10)

        time.sleep(2)

        click_count = 0

        while len(reviews) < limit:

            # Collect currently visible reviews
            elements = driver.find_elements(
                By.CSS_SELECTOR,
                "article, .review, [class*='review']"
            )

            for element in elements:

                text = element.text.strip()

                if (
                    text
                    and len(text) > 15
                    and text not in reviews
                ):
                    reviews.append(text)

                    if len(reviews) >= limit:
                        break

            print(
                f"After click {click_count}: "
                f"{len(reviews)} reviews"
            )

            if len(reviews) >= limit:
                break

            # Find Load More button
            buttons = driver.find_elements(
                By.XPATH,
                "//button[contains(translate(normalize-space(.), "
                "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
                "'abcdefghijklmnopqrstuvwxyz'), 'load more')]"
            )

            if not buttons:
                print("Load More button not found.")
                break

            button = buttons[0]

            # Scroll to button
            driver.execute_script(
                "arguments[0].scrollIntoView({block:'center'});",
                button
            )

            time.sleep(0.5)

            # Click Load More
            try:
                driver.execute_script(
                    "arguments[0].click();",
                    button
                )
            except Exception:
                try:
                    button.click()
                except Exception:
                    print("Could not click Load More.")
                    break

            click_count += 1

            # Wait for new reviews
            time.sleep(1.5)

        # Final extraction
        elements = driver.find_elements(
            By.CSS_SELECTOR,
            "article, .review, [class*='review']"
        )

        for element in elements:

            text = element.text.strip()

            if (
                text
                and len(text) > 15
                and text not in reviews
            ):
                reviews.append(text)

            if len(reviews) >= limit:
                break

        reviews = reviews[:limit]

        print("=" * 50)
        print(f"FINAL REVIEWS: {len(reviews)}")
        print("=" * 50)

        return reviews

    except Exception as e:

        print("Scraping error:", e)

        return reviews[:limit]

    finally:
        driver.quit()