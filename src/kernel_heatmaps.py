"""Heatmaps and simple statistics for saved quantum kernel matrices.

Each saved matrix is the (n_train x n_train) kernel of one CV fold, with rows in
`train_idx` order. To make class structure visible, rows/columns are re-ordered
by class label (malignant first), recovered with the same StratifiedKFold split
the notebooks use.

Usage (from the repository root):
    python src/kernel_heatmaps.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "Notebooks" /"kernel-matrices"
FIG_DIR = ROOT / "results" / "figures"

# label -> file name template ({i} = fold number, 1-based)
KERNELS = {
    "EfficientSU2": "efficient-su2-{i}.npy",
    "Angle-Phase": "Angle-phase-{i}.npy",
}


def fold_labels():
    """y_train for every fold, identical to the notebooks' split."""
    y = load_breast_cancer().target
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    return [y[tr] for tr, _ in skf.split(np.zeros((len(y), 1)), y)]


def sort_by_class(K, y):
    order = np.argsort(y, kind="stable")  # 0 = malignant, 1 = benign
    return K[np.ix_(order, order)], y[order]


def kernel_stats(K, y):
    """Mean within/between-class similarity and kernel-target alignment."""
    n = len(y)
    off = ~np.eye(n, dtype=bool)
    same = (y[:, None] == y[None, :]) & off
    diff = y[:, None] != y[None, :]
    t = np.where(y == 1, 1.0, -1.0)
    yy = np.outer(t, t)
    kta = (K * yy).sum() / (np.linalg.norm(K) * np.linalg.norm(yy))
    return {
        "within": K[same].mean(),
        "between": K[diff].mean(),
        "offdiag_mean": K[off].mean(),
        "kta": kta,
    }


def plot_fold(ax, K, y, title):
    Ks, ys = sort_by_class(K, y)
    im = ax.imshow(Ks, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
    b = int((ys == 0).sum()) - 0.5  # malignant | benign boundary
    ax.axhline(b, color="white", lw=0.8)
    ax.axvline(b, color="white", lw=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ys = fold_labels()
    for label, template in KERNELS.items():
        files = [KERNEL_DIR / template.format(i=i) for i in range(1, 6)]
        if not all(f.exists() for f in files):
            print(f"[skip] {label}: missing files in {KERNEL_DIR}")
            continue
        mats = [np.load(f) for f in files]

        fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), constrained_layout=True)
        rows = []
        for i, (K, y, ax) in enumerate(zip(mats, ys, axes), start=1):
            assert K.shape == (len(y), len(y)), (label, i, K.shape, len(y))
            im = plot_fold(ax, K, y, f"fold {i}")
            rows.append(kernel_stats(K, y))
        fig.colorbar(im, ax=axes, shrink=0.8, label="k(x, x')")
        fig.suptitle(f"{label} kernel, train folds (sorted: malignant | benign)")
        out = FIG_DIR / f"heatmap_{label.lower().replace(' ', '_')}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)

        print(f"\n{label} -> {out.relative_to(ROOT)}")
        print("fold  within  between  offdiag  KTA")
        for i, r in enumerate(rows, start=1):
            print(f"{i:>4}  {r['within']:.4f}  {r['between']:.4f}   "
                  f"{r['offdiag_mean']:.4f}  {r['kta']:.4f}")
        print("mean  " + "  ".join(
            f"{np.mean([r[k] for r in rows]):.4f}" for k in ("within", "between", "offdiag_mean", "kta")))


if __name__ == "__main__":
    main()
