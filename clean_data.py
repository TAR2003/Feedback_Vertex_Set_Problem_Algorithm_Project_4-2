import os
import torch
from glob import glob


def clean_dataset(base_dir="gnn_model/datasets/pt"):
    print(f"Scanning {base_dir} for corrupted files...")
    # Find all .pt files recursively
    files = glob(os.path.join(base_dir, "**", "*.pt"), recursive=True)

    removed_count = 0
    for file_path in files:
        try:
            data = torch.load(file_path)
            # Check if any features contain NaN or Infinity
            if torch.isnan(data.x).any() or torch.isinf(data.x).any():
                print(f"Deleting corrupted file: {file_path}")
                os.remove(file_path)
                removed_count += 1
        except Exception as e:
            print(f"Could not read {file_path}. Deleting it. Error: {e}")
            os.remove(file_path)
            removed_count += 1

    print(f"Done! Scanned {len(files)} files. Removed {removed_count} bad files.")


if __name__ == "__main__":
    clean_dataset()
