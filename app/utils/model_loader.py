import os
import requests

def download_file(url, save_path):
    # create folder if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # skip if already exists
    if os.path.exists(save_path):
        print(f"{save_path} already exists. Skipping download.")
        return

    print(f"Downloading {save_path}...")

    response = requests.get(url)

    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Saved to {save_path}")
    else:
        print(f"Failed to download {url}")