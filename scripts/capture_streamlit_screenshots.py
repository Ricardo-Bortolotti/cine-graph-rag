"""Capture Streamlit pages for README documentation."""

from __future__ import annotations

import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
BASE = "http://127.0.0.1:8502"
PAGES = [
    ("Home", "screenshot-home.png", "CineGraphRAG"),
    ("Personalized Recommendations", "screenshot-recommendations.png", "Personalized Recommendations"),
    ("Explain Recommendation", "screenshot-explain.png", "Explain Recommendation"),
    ("Graph Explorer", "screenshot-explorer.png", "Graph Explorer"),
    ("Graph Analytics Dashboard", "screenshot-analytics.png", "Graph Analytics Dashboard"),
]


def dismiss_toasts(driver: webdriver.Chrome) -> None:
    for selector in (
        "[data-testid='stNotificationCloseButton']",
        "button[aria-label='Close']",
        "[data-testid='stBaseButton-headerNoPadding']",
    ):
        for el in driver.find_elements(By.CSS_SELECTOR, selector):
            try:
                el.click()
            except Exception:
                pass


def click_nav(driver: webdriver.Chrome, label: str) -> None:
    wait = WebDriverWait(driver, 20)
    # Streamlit multipage nav renders page titles as sidebar links/buttons.
    xpath = (
        "//*[@data-testid='stSidebarNav']//*[normalize-space()=$label]"
        .replace("$label", f"'{label}'")
    )
    fallback = f"//*[self::a or self::p or self::span][normalize-space()='{label}']"
    for expr in (xpath, fallback):
        try:
            el = wait.until(EC.element_to_be_clickable((By.XPATH, expr)))
            driver.execute_script("arguments[0].click();", el)
            return
        except Exception:
            continue
    raise RuntimeError(f"Could not click sidebar page: {label}")


def _click_button(driver: webdriver.Chrome, label: str) -> None:
    xpath = f"//button[.//p[normalize-space()='{label}'] or normalize-space()='{label}']"
    el = WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH, xpath)))
    driver.execute_script("arguments[0].click();", el)


def _wait_text(driver: webdriver.Chrome, text: str, timeout: int = 30) -> None:
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located(
            (By.XPATH, f"//*[contains(normalize-space(), '{text}')]")
        )
    )


def wait_heading(driver: webdriver.Chrome, text: str, timeout: int = 25) -> None:
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located(
            (By.XPATH, f"//*[contains(normalize-space(), '{text}')]")
        )
    )
    time.sleep(1.5)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--hide-scrollbars")
    options.add_argument("--window-size=1600,1100")
    options.add_argument("--force-device-scale-factor=1")
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(1600, 1100)
    try:
        driver.get(BASE + "/")
        WebDriverWait(driver, 25).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='stAppViewContainer']"))
        )
        time.sleep(2)
        dismiss_toasts(driver)

        for label, filename, marker in PAGES:
            print(f"Capturing {label}")
            click_nav(driver, label)
            wait_heading(driver, marker)
            dismiss_toasts(driver)
            if label == "Personalized Recommendations":
                _click_button(driver, "Generate recommendations")
                _wait_text(driver, "Seed:", timeout=40)
            elif label == "Explain Recommendation":
                _click_button(driver, "Explain connection")
                _wait_text(driver, "Cypher", timeout=40)
            elif label == "Graph Explorer":
                _click_button(driver, "Build interactive graph")
                _wait_text(driver, "Nodes", timeout=50)
                time.sleep(3)
            elif label == "Graph Analytics Dashboard":
                _wait_text(driver, "Highest-rated movies", timeout=30)
                time.sleep(1.5)
            driver.execute_script("window.scrollTo(0, 0)")
            time.sleep(1)
            out = ASSETS / filename
            driver.save_screenshot(str(out))
            print(f"  wrote {out} ({out.stat().st_size} bytes)")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
