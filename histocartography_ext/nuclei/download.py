import os
from pathlib import Path
from histocartography_ext.utils.io import download_box_link

DATASET_TO_BOX_URL = {
    "pannuke": "https://ibm.box.com/shared/static/hrt04i3dcv1ph1veoz8x6g8a72u0uw58.pt",
    "monusac": "https://ibm.box.com/shared/static/u563aoydow9w2kpgw0l8esuklegdtdij.pt",
}

def download_checkpoints(download_dir="checkpoints"):
    """
    Download HoVer-Net checkpoints (pannuke, monusac).
    """
    os.makedirs(download_dir, exist_ok=True)
    
    for name, url in DATASET_TO_BOX_URL.items():
        fname = os.path.join(download_dir, f"hovernet_{name}.pth")
        print(f"Checking {fname}...")
        if not os.path.exists(fname):
            print(f"Downloading {name} model to {fname}...")
            # Note: The original code used .pt extension in URL, but saved as target name?
            # Original code: model_path = ... pretrained_data + ".pt"
            # User error says: 'checkpoints/hovernet_pannuke.pth'
            # Let's save as .pth to match user expectation, or .pt and rename.
            # download_box_link(url, fname)
            
            # Use requests directly or use valid utility? 
            # histocartography_ext.utils.io.download_box_link is available.
            try:
                download_box_link(url, fname)
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
