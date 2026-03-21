import os
import requests

def download_file(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # delete old file
    if os.path.exists(save_path):
        os.remove(save_path)

    print(f"Downloading {save_path}...")

    response = requests.get(url)

    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Saved to {save_path}")
    else:
        raise Exception(f"Failed to download {url}")