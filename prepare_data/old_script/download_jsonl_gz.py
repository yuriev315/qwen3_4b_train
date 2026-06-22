import requests
import xml.etree.ElementTree as ET
from pathlib import Path

def download_jsonl_gz():
    # Replace with your bucket URL
    BUCKET_URL = "https://us-east-1.hippius.com/albedo"

    DOWNLOAD_DIR = "data/albedo_jsonl_gz"

    Path(DOWNLOAD_DIR).mkdir(exist_ok=True)

    print("Fetching bucket listing...")

    resp = requests.get(BUCKET_URL)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)

    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    files = []

    for obj in root.findall("s3:Contents", ns):
        key = obj.find("s3:Key", ns).text

        if key.endswith(".jsonl.gz"):
            files.append(key)

    print(f"Found {len(files)} jsonl.gz files")

    for i, key in enumerate(files, start=1):

        local_path = Path(DOWNLOAD_DIR) / key
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists():
            print(f"[{i}/{len(files)}] Skip {key}")
            continue

        url = f"{BUCKET_URL.rstrip('/')}/{key}"

        print(f"[{i}/{len(files)}] Downloading {key}")

        with requests.get(url, stream=True) as r:
            r.raise_for_status()

            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    print("Downloading jsonl_gz files finished.")

if __name__ == '__main__':
    download_jsonl_gz()