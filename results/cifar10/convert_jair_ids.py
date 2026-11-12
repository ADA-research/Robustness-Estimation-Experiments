"""Convert the JAIR alpha,beta-CROWN results on CIFAR-10 to CIFAR-10 test indices.

The JAIR experiment (NNV_JAIR_robustness_distributions, experiments/CIFAR-10/main.py) ran alpha,beta-CROWN on a
class-balanced sample of 100 test images and passed `Subset(CIFAR10(train=False), test_idx)` to VERONA, which stores
the position in that subset as `image_id`. The attack results in this repository use the test-set index instead.
This script recomputes the sample exactly as in the JAIR experiment and writes

* abcrown_3600s.csv: test-split results, `image_id` = CIFAR-10 test index, `jair_sample_position` = original id
* balanced_sample_test_indices.csv: the 100 test indices of the sample

Usage:
    python convert_jair_ids.py --cifar-dir <path to cifar-10-batches-py> [--jair-results <path or URL>]
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

JAIR_RESULTS = ("https://raw.githubusercontent.com/ADA-research/NNV_JAIR_robustness_distributions/"
                "HEAD/data/CIFAR-10/results_CIFAR_JAIR.csv")
OUTPUT_DIR = Path(__file__).resolve().parent


def load_test_labels(cifar_dir):
    with open(Path(cifar_dir) / "test_batch", "rb") as f:
        return np.array(pickle.load(f, encoding="bytes")[b"labels"])


def balanced_sample(labels, seed=42):
    """Same as get_balanced_sample(train_bool=False) in the JAIR main.py."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=100, random_state=seed)
    _, test_idx = next(splitter.split(np.zeros(len(labels)), labels))
    return test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar-dir", required=True, help="folder with the CIFAR-10 python batches (test_batch)")
    parser.add_argument("--jair-results", default=JAIR_RESULTS, help="results_CIFAR_JAIR.csv (path or URL)")
    args = parser.parse_args()

    labels = load_test_labels(args.cifar_dir)
    test_idx = balanced_sample(labels)

    sample = pd.DataFrame({"jair_sample_position": range(len(test_idx)), "image_id": test_idx, "label": labels[test_idx]})
    assert sample.label.value_counts().eq(10).all(), "the sample must contain 10 images per class"

    df = pd.read_csv(args.jair_results)
    df = df[df.split == "test"].copy()
    df["network"] = df.network_path.str.strip().str.replace(".onnx", "", regex=False)
    df = df.rename(columns={"image_id": "jair_sample_position"})
    df["image_id"] = test_idx[df.jair_sample_position]
    # Check: after the conversion, every stored label must equal the CIFAR-10 test label of that index
    assert (labels[df.image_id] == df.original_label).all(), "labels do not match after the conversion"
    df["verifier"] = "ABcrown"

    columns = ["network", "image_id", "jair_sample_position", "original_label", "epsilon_value", "smallest_sat_value",
               "total_time", "training", "verifier"]
    df = df[columns].sort_values(["network", "image_id"]).reset_index(drop=True)

    df.to_csv(OUTPUT_DIR / "abcrown_3600s.csv", index=False)
    sample.to_csv(OUTPUT_DIR / "balanced_sample_test_indices.csv", index=False)
    print(f"{len(df)} results, labels match for all of them; written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
