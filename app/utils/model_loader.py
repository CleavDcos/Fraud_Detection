import os
import gdown

def download_file(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # extract file ID
    file_id = url.split("id=")[-1]
    download_url = f"https://drive.google.com/uc?id={file_id}"

    # always re-download (for now)
    if os.path.exists(save_path):
        print(f"Deleting old file: {save_path}")
        os.remove(save_path)

    print(f"Downloading {save_path}...")

    gdown.download(download_url, save_path, quiet=False)

    print(f"Saved to {save_path}")