import os
import urllib.request
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

URLS = {
    "train": "https://huggingface.co/datasets/ura-hcmut/UIT-VSFC/raw/main/UIT-VSFC_train.csv",
    "valid": "https://huggingface.co/datasets/ura-hcmut/UIT-VSFC/raw/main/UIT-VSFC_valid.csv",
    "test": "https://huggingface.co/datasets/ura-hcmut/UIT-VSFC/raw/main/UIT-VSFC_test.csv"
}

def download_uit_vsfc():
    print(f"Downloading UIT-VSFC dataset to {DATA_DIR}...")
    for split, url in URLS.items():
        file_path = os.path.join(DATA_DIR, f"UIT-VSFC_{split}.csv")
        if not os.path.exists(file_path):
            print(f"Downloading {split} from {url}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp, open(file_path, 'wb') as f:
                f.write(resp.read())
            print(f"Saved: {file_path}")
        else:
            print(f"Already exists: {file_path}")

        df = pd.read_csv(file_path)
        print(f"  {split}: {df.shape[0]} rows, columns: {df.columns.tolist()}")
        print(f"  Labels distribution:\n{df['label'].value_counts().to_string()}\n")

if __name__ == "__main__":
    download_uit_vsfc()
