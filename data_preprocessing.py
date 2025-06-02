
"""data_preprocessing.py
Utility script for:
  1) Loading the cotton‑impurity CSV dataset.
  2) Visualising a few spectra per class.
  3) Generating a reproducible 5‑fold split (saved to .npz files).

Usage (example):
    python data_preprocessing.py --csv cotton_impurity_dataset.csv --out_dir folds

The script expects the CSV last column to be the label (class name); all preceding
columns are treated as features (band‑1 … band‑288).
"""

import argparse, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(csv_path):
    """
    读取 CSV，返回:
        X : (N, D) float32  — 数值特征矩阵
        y : (N,) int64      — 标签索引
        label_to_idx : dict — 标签→索引映射
    """
    import pandas as pd, numpy as np

    df = pd.read_csv(csv_path)

    # -------- 1) 找到标签列 --------
    possible_label_cols = [c for c in df.columns if c.lower() == "label"]
    if possible_label_cols:
        label_col = possible_label_cols[0]
    else:
        label_col = df.columns[-1]          # 回退到最后一列

    y_text = df[label_col].astype(str).values

    # -------- 2) 选择纯数值特征列 --------
    cols_to_drop = {label_col, "id", "ID", "Id", "x", "X", "y", "Y"}
    numeric_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns]) \
                   .select_dtypes(include=[np.number])

    X = numeric_df.values.astype(np.float32)

    # -------- 3) 标签编码 --------
    labels_sorted = sorted(set(y_text))
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels_sorted)}
    y = np.array([label_to_idx[t] for t in y_text], dtype=np.int64)

    return X, y, label_to_idx


def visualise_samples(X, y, label_to_idx, samples_per_class=3, fig_path=None):
    """Plot a few random spectra from each class for quick inspection."""
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    plt.figure(figsize=(10, 6))
    n_classes = len(label_to_idx)
    rng = np.random.default_rng(0)
    for c in range(n_classes):
        ax = plt.subplot(2, (n_classes + 1) // 2, c + 1)
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for s in idx[:samples_per_class]:
            ax.plot(X[s], alpha=0.8)
        ax.set_title(idx_to_label[c])
        ax.set_xlabel("Band index")
        ax.set_ylabel("Reflectance")
        ax.grid(True, ls='--', lw=0.4)
    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=300)
    else:
        plt.show()
    plt.close()


def kfold_indices(n_samples, n_splits=5, shuffle=True, seed=42):
    """Return list of (train_idx, test_idx) tuples for K‑Fold."""
    idx = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    fold_sizes = np.full(n_splits, n_samples // n_splits, int)
    fold_sizes[: n_samples % n_splits] += 1
    folds = []
    start = 0
    for fold_size in fold_sizes:
        stop = start + fold_size
        test_idx = idx[start:stop]
        train_idx = np.concatenate((idx[:start], idx[stop:]))
        folds.append((train_idx, test_idx))
        start = stop
    return folds


def save_folds(X, y, folds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, (tr_idx, te_idx) in enumerate(folds, 1):
        np.savez_compressed(
            os.path.join(out_dir, f"fold_{i}_train.npz"),
            X=X[tr_idx],
            y=y[tr_idx],
        )
        np.savez_compressed(
            os.path.join(out_dir, f"fold_{i}_test.npz"),
            X=X[te_idx],
            y=y[te_idx],
        )
    print(f"Saved {len(folds)} folds to: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV dataset.")
    ap.add_argument("--out_dir", default="folds", help="Directory to save folds.")
    ap.add_argument("--vis", action="store_true", help="Visualise sample spectra.")
    args = ap.parse_args()

    X, y, label_to_idx = load_data(args.csv)
    print("Dataset loaded → samples:", len(X), "features:", X.shape[1])
    print("Class distribution:", {k: int((y == v).sum()) for k, v in label_to_idx.items()})

    if args.vis:
        visualise_samples(X, y, label_to_idx, fig_path=os.path.join(args.out_dir, "spectra_examples.png"))

    folds = kfold_indices(len(X), n_splits=5)
    # persist folds and label map
    save_folds(X, y, folds, args.out_dir)
    with open(os.path.join(args.out_dir, "label_map.json"), "w") as f:
        json.dump(label_to_idx, f, indent=2)
    print("Pre‑processing complete.")


if __name__ == "__main__":
    main()
