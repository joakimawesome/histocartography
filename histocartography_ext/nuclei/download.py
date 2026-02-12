import os
import gdown
from pathlib import Path

# Provide GDrive IDs for checkpoints
DATASET_TO_GDRIVE_ID = {
    "pannuke": "1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR",
    "monusac": "13qkxDqv7CUqxN-l5CpeFVmc24mDw6CeV",
    "consep": "1FtoTDDnuZShZmQujjaFSLVJLD5sAh2_P"
}

def download_checkpoints(download_dir="checkpoints"):
    """
    Download HoVer-Net checkpoints (pannuke, monusac, consep).
    """
    os.makedirs(download_dir, exist_ok=True)
    
    for name, file_id in DATASET_TO_GDRIVE_ID.items():
        fname = os.path.join(download_dir, f"hovernet_{name}.pth")
        print(f"Checking {fname}...")
        
        if not os.path.exists(fname):
            print(f"Downloading {name} model to {fname}...")
            try:
                # gdown handles GDrive authentication/cookie issues usually well
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, fname, quiet=False)
                print("Done.")
            except Exception as e:
                print(f"Failed to download {name}: {e}")
        else:
            print(f"Found {fname}, skipping.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()
    
    download_checkpoints(args.dir)
