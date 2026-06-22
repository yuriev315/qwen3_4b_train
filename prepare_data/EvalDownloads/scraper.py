"""
Albedo eval downloader.

Scans the Downloads folder for existing eval-{uuid}.zip files, then iterates
the Albedo history table and downloads any evals not yet present locally.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

INDEX_URL = "https://us-east-1.hippius.com/albedo/index.html"
EVAL_ID_PATTERN = re.compile(
    r"^eval-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.zip$",
    re.IGNORECASE,
)
TABLE_HEADER_MARKERS = ("when", "uid", "model", "vs king", "result")
HISTORY_SECTION_SELECTOR = "#history"
HISTORY_ROW_SELECTOR = "#history tbody tr.clickable"
DOWNLOAD_TIMEOUT_SEC = 300
PAGE_LOAD_TIMEOUT_SEC = 120
POLL_INTERVAL_SEC = 1


def get_downloads_folder() -> Path:
    return Path("./Downloads")


def load_existing_eval_ids(downloads_dir: Path) -> set[str]:
    ids: set[str] = set()
    if not downloads_dir.is_dir():
        return ids
    for path in downloads_dir.iterdir():
        match = EVAL_ID_PATTERN.match(path.name)
        if match:
            ids.add(match.group(1).lower())
    return ids


def build_driver(downloads_dir: Path) -> webdriver.Chrome:
    options = Options()
    prefs = {
        "download.default_directory": str(downloads_dir.resolve()),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    options.add_experimental_option("prefs", prefs)
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1400,900")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(0)
    return driver


def wait_for_page_load(driver: webdriver.Chrome, timeout: int = 30) -> None:
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )


def parse_index_html(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def find_history_table(soup: BeautifulSoup):
    history = soup.find("section", id="history")
    if history is None:
        return None
    for table in history.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if not headers:
            continue
        if all(marker in headers for marker in TABLE_HEADER_MARKERS):
            return table
    return None


def wait_for_history_table(driver: webdriver.Chrome, timeout: int = PAGE_LOAD_TIMEOUT_SEC):
    def table_ready(d: webdriver.Chrome):
        meta = d.find_elements(By.CSS_SELECTOR, "#history-meta")
        if not meta:
            return False
        meta_text = meta[0].text.strip().lower()
        if not meta_text or meta_text == "loading…" or meta_text == "loading...":
            return False
        rows = d.find_elements(By.CSS_SELECTOR, HISTORY_ROW_SELECTOR)
        return len(rows) > 0

    WebDriverWait(driver, timeout).until(table_ready)
    soup = parse_index_html(driver.page_source)
    return find_history_table(soup)


def get_table_rows(driver: webdriver.Chrome):
    soup = parse_index_html(driver.page_source)
    table = find_history_table(soup)
    if table is None:
        raise RuntimeError("History table not found on index page.")
    tbody = table.find("tbody")
    if tbody is None:
        return []
    return tbody.find_all("tr")


def click_table_row(driver: webdriver.Chrome, row_index: int) -> None:
    rows = driver.find_elements(By.CSS_SELECTOR, HISTORY_ROW_SELECTOR)
    if row_index >= len(rows):
        raise IndexError(f"Row index {row_index} out of range ({len(rows)} rows).")
    row = rows[row_index]
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", row)
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable(row))
    driver.execute_script("arguments[0].click();", row)


def extract_eval_run_id(url: str) -> str | None:
    match = re.search(r"[?&]eval_run_id=([0-9a-f-]{36})", url, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower()


def wait_for_detail_page(driver: webdriver.Chrome, timeout: int = 30) -> str:
    WebDriverWait(driver, timeout).until(lambda d: "detail.html" in d.current_url)
    wait_for_page_load(driver, timeout)
    eval_id = extract_eval_run_id(driver.current_url)
    if eval_id is None:
        raise RuntimeError(f"Could not parse eval_run_id from URL: {driver.current_url}")
    return eval_id


def click_download_button(driver: webdriver.Chrome) -> None:
    button = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn-zip"))
    )
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
    button.click()


def wait_for_download(
    downloads_dir: Path,
    eval_id: str,
    known_files: set[str],
    timeout: int = DOWNLOAD_TIMEOUT_SEC,
) -> Path:
    target_name = f"eval-{eval_id}.zip"
    target_lower = target_name.lower()
    deadline = time.time() + timeout

    while time.time() < deadline:
        for path in downloads_dir.iterdir():
            if path.name.lower() == target_lower and path.is_file():
                if path.stat().st_size > 0:
                    # Chrome may still be writing; wait until size stabilizes.
                    size = path.stat().st_size
                    time.sleep(POLL_INTERVAL_SEC)
                    if path.exists() and path.stat().st_size == size:
                        return path

        # Also accept a newly appeared eval zip even if id casing differs.
        for path in downloads_dir.iterdir():
            if path.name in known_files:
                continue
            match = EVAL_ID_PATTERN.match(path.name)
            if match and match.group(1).lower() == eval_id.lower() and path.stat().st_size > 0:
                size = path.stat().st_size
                time.sleep(POLL_INTERVAL_SEC)
                if path.exists() and path.stat().st_size == size:
                    return path

        time.sleep(POLL_INTERVAL_SEC)

    raise TimeoutError(f"Timed out waiting for download: {target_name}")


def run() -> None:
    downloads_dir = get_downloads_folder()
    downloads_dir.mkdir(parents=True, exist_ok=True)

    known_ids = load_existing_eval_ids(downloads_dir)
    known_files = {p.name for p in downloads_dir.iterdir() if p.is_file()}

    print(f"Downloads folder: {downloads_dir}")
    print(f"Found {len(known_ids)} existing eval archive(s).")

    driver = build_driver(downloads_dir)
    row_index = 0

    try:
        while True:
            driver.get(INDEX_URL)
            wait_for_page_load(driver)
            wait_for_history_table(driver)

            rows = get_table_rows(driver)
            if row_index >= len(rows):
                print("Update Finished")
                return

            click_table_row(driver, row_index)
            row_index += 1

            eval_id = wait_for_detail_page(driver)

            if eval_id in known_ids:
                print("Update Finished")
                return

            print(f"Downloading eval {eval_id} ...")
            click_download_button(driver)
            downloaded = wait_for_download(downloads_dir, eval_id, known_files)
            known_ids.add(eval_id)
            known_files.add(downloaded.name)
            print(f"Saved {downloaded.name}")

    except TimeoutException as exc:
        print(f"Timed out: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        driver.quit()


if __name__ == "__main__":
    run()
