import os
import time
import re
import hashlib
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- Configuration ---
URL = "https://us-east-1.hippius.com/albedo/index.html"
DOWNLOAD_FOLDER = os.path.abspath("data/albedo_zip")

if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)


def get_file_hash(filepath):
    """Calculate MD5 hash of a file for deduplication"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# def deduplicate_by_hash(folder_path):
#     """Remove duplicate files based on hash (content)"""
#     if not os.path.exists(folder_path):
#         return 0
#
#     print(f"\n🧹 Checking for duplicate files by content...")
#
#     # Get all ZIP files
#     files = [f for f in os.listdir(folder_path) if f.endswith('.zip')]
#     if len(files) <= 1:
#         print("📁 No duplicates to clean")
#         return 0
#
#     # Group by hash
#     hash_groups = {}
#     for file in files:
#         filepath = os.path.join(folder_path, file)
#         try:
#             file_hash = get_file_hash(filepath)
#             if file_hash not in hash_groups:
#                 hash_groups[file_hash] = []
#             hash_groups[file_hash].append(filepath)
#         except Exception as e:
#             print(f"⚠️ Error hashing {file}: {e}")
#
#     # Remove duplicates (keep one per hash group)
#     duplicates_removed = 0
#     for file_hash, filepaths in hash_groups.items():
#         if len(filepaths) > 1:
#             # Keep the file with most descriptive name (longest)
#             filepaths.sort(key=lambda x: len(x), reverse=True)
#             keep_file = filepaths[0]
#             for i in range(1, len(filepaths)):
#                 try:
#                     os.remove(filepaths[i])
#                     duplicates_removed += 1
#                     print(f"🗑️ Removed duplicate: {os.path.basename(filepaths[i])}")
#                 except Exception as e:
#                     print(f"⚠️ Could not remove {filepaths[i]}: {e}")
#
#     print(f"🧹 Removed {duplicates_removed} duplicate files by content")
#     return duplicates_removed


def keep_latest_only(folder_path=DOWNLOAD_FOLDER):
    """Keep only the latest version of each evaluation and rename it to remove version numbers"""
    if not os.path.exists(folder_path):
        return 0, 0

    print(f"\n🧹 Cleaning up older versions and renaming latest...")

    # Group files by evaluation ID
    eval_groups = {}
    for file in os.listdir(folder_path):
        if file.endswith('.zip'):
            # Extract evaluation ID (e.g., 168 from eval-2026-06-09-168.zip)
            match = re.search(r'eval-(\d{4}-\d{2}-\d{2})-(\d+)', file)
            if match:
                date = match.group(1)
                eval_id = match.group(2)
                key = f"{date}-{eval_id}"

                # Extract version number from filename
                version = 0
                if '(' in file and ')' in file:
                    version_match = re.search(r'\((\d+)\)', file)
                    if version_match:
                        version = int(version_match.group(1))

                if key not in eval_groups:
                    eval_groups[key] = []
                eval_groups[key].append((version, file))

    # Process each evaluation group
    removed = 0
    renamed = 0

    for key, files in eval_groups.items():
        if len(files) > 1:
            # Sort by version number (descending)
            files.sort(key=lambda x: x[0], reverse=True)

            # The latest version
            latest_version, latest_file = files[0]

            # Remove all older versions
            for version, file in files[1:]:
                filepath = os.path.join(folder_path, file)
                try:
                    os.remove(filepath)
                    removed += 1
                    print(f"🗑️ Removed older version: {file}")
                except Exception as e:
                    print(f"⚠️ Could not remove {file}: {e}")

            # Rename the latest version to remove version number
            base_name = f"eval-{key}.zip"
            latest_path = os.path.join(folder_path, latest_file)
            new_path = os.path.join(folder_path, base_name)

            # If the base name already exists, remove it first
            if os.path.exists(new_path) and new_path != latest_path:
                try:
                    os.remove(new_path)
                    print(f"🗑️ Removed existing base file: {base_name}")
                except:
                    pass

            # Rename the latest file
            if latest_path != new_path:
                try:
                    os.rename(latest_path, new_path)
                    renamed += 1
                    print(f"✏️ Renamed: {latest_file} -> {base_name}")
                except Exception as e:
                    print(f"⚠️ Could not rename {latest_file}: {e}")

        elif len(files) == 1:
            # Single file - check if it needs renaming (has version number)
            version, file = files[0]
            if version > 0:
                base_name = f"eval-{key}.zip"
                file_path = os.path.join(folder_path, file)
                new_path = os.path.join(folder_path, base_name)

                # If base name exists, remove it
                if os.path.exists(new_path) and new_path != file_path:
                    try:
                        os.remove(new_path)
                        print(f"🗑️ Removed existing base file: {base_name}")
                    except:
                        pass

                try:
                    os.rename(file_path, new_path)
                    renamed += 1
                    print(f"✏️ Renamed: {file} -> {base_name}")
                except Exception as e:
                    print(f"⚠️ Could not rename {file}: {e}")

    print(f"🧹 Removed {removed} older files")
    print(f"✏️ Renamed {renamed} files to clean version")
    return removed, renamed


def setup_firefox_driver():
    """Setup Firefox with automatic download settings"""
    options = webdriver.FirefoxOptions()

    # Set download directory for Firefox
    options.set_preference("browser.download.folderList", 2)
    options.set_preference("browser.download.dir", DOWNLOAD_FOLDER)
    options.set_preference("browser.download.manager.showWhenStarting", False)
    options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/zip,application/octet-stream")

    # Improve stability
    options.set_preference("browser.privatebrowsing.autostart", False)
    options.set_preference("browser.startup.page", 0)
    options.set_preference("browser.startup.homepage_override.mstone", "ignore")

    # Headless mode
    # options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")

    return webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=options)


def download_all_eval_zips():
    driver = setup_firefox_driver()

    try:
        print("🌐 Opening Albedo page (headless mode)...")
        driver.get(URL)

        # Wait for page to load
        wait = WebDriverWait(driver, 30)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#history")))

        # Scroll to load all entries
        print("📜 Scrolling to load all entries...")
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)

        # Collect all URLs first
        all_buttons = driver.find_elements(By.CSS_SELECTOR, "#history button.data-dl")
        zip_urls = []
        url_to_filename = {}

        for button in all_buttons:
            zip_url = button.get_attribute("data-zip-dir")
            if zip_url:
                filename = f"eval_{len(zip_urls)}.zip"
                if "eval" in zip_url:
                    parts = zip_url.split('/')
                    date_part = parts[-2] if len(parts) > 1 else "unknown"
                    filename = f"eval_{date_part}.zip"
                zip_urls.append(zip_url)
                url_to_filename[zip_url] = filename

        print(f"🔍 Found {len(zip_urls)} evaluation URLs")

        if not zip_urls:
            print("No download URLs found!")
            return

        # Download each URL by re-finding the button and clicking
        successful_clicks = 0
        already_existed = 0

        for index, zip_url in enumerate(zip_urls):
            filename = url_to_filename[zip_url]

            # Check if file already exists
            if os.path.exists(os.path.join(DOWNLOAD_FOLDER, filename)):
                print(f"⏭️ File {filename} already exists. Skipping...")
                already_existed += 1
                continue

            print(f"⬇️ Downloading {index + 1}/{len(zip_urls)}: {filename}")

            try:
                # Re-find the specific button for this URL
                buttons = driver.find_elements(By.CSS_SELECTOR, "#history button.data-dl")
                target_button = None
                for btn in buttons:
                    if btn.get_attribute("data-zip-dir") == zip_url:
                        target_button = btn
                        break

                if target_button:
                    # Scroll to button
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", target_button)
                    time.sleep(2)

                    # Click using JavaScript
                    driver.execute_script("arguments[0].click();", target_button)
                    successful_clicks += 1
                    print(f"✅ Click successful: {filename}")
                else:
                    print(f"⚠️ Could not find button for {filename}")

                time.sleep(3)

            except Exception as e:
                print(f"⚠️ Error downloading {filename}: {e}")

        print(f"\n✅ Successfully clicked {successful_clicks}/{len(zip_urls)} buttons")
        print(f"⏭️ Skipped {already_existed} files (already exist)")

        # Wait for downloads to complete
        print("⏳ Waiting for downloads to complete...")
        time.sleep(10)

        # Post-download cleanup
        print("\n--- Post-Download Cleanup ---")
        # deduplicate_by_hash(DOWNLOAD_FOLDER)
        keep_latest_only(DOWNLOAD_FOLDER)

        # Final listing
        files = [f for f in os.listdir(DOWNLOAD_FOLDER) if f.endswith('.zip')]
        total_size = sum(os.path.getsize(os.path.join(DOWNLOAD_FOLDER, f)) for f in files)
        print(f"\n📂 Final count: {len(files)} ZIP files")
        print(f"📊 Total size: {total_size / 1024 / 1024:.2f} MB")

    finally:
        driver.quit()
        print("\n🎉 Process completed!")


if __name__ == "__main__":
    download_all_eval_zips()