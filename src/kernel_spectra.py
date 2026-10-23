"""Eigenvalue spectra of the saved quantum kernel matrices (5 CV folds).

For every kernel in kernel_heatmaps.KERNELS it prints per-fold summary statistics,
and saves results/figures/spectrum_<label>.png. A classical RBF kernel
(gamma="scale", MinMax-scaled data, same folds) is drawn as a dashed reference.

Usage (repository root):  python src/kernel_spectra.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import rbf_kernel

from kernel_heatmaps import KERNELS, KERNEL_DIR, FIG_DIR, ROOT

FLOOR = 1e-12  # only for the log-scale plot


def spectrum(K):
    K = (K + K.T) / 2                      # remove float asymmetry
    return np.sort(np.linalg.eigvalsh(K))[::-1]


def centered_spectrum(K):
    n = len(K)
    H = np.eye(n) - np.ones((n, n)) / n
    return spectrum(H @ K @ H)


def stats(lam):
    n = len(lam)
    p = np.clip(lam, 0, None) / np.clip(lam, 0, None).sum()
    cum = np.cumsum(p)
    return {
        "min_eig": lam[-1],
        "trace/n": lam.sum() / n,                              # = 1 if diag(K)=1
        "lam1/n": lam[0] / n,
        "num_rank": int((lam > 1e-10 * lam[0]).sum()),
        "PR": float(lam.sum() ** 2 / (lam ** 2).sum()),       # participation ratio
        "eff_rank": float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())),
        "k90": int(np.searchsorted(cum, 0.90) + 1),
        "k99": int(np.searchsorted(cum, 0.99) + 1),
    }


def rbf_reference():
    data = load_breast_cancer()
    X, y = data.data, data.target
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    out = []
    for tr, _ in skf.split(X, y):
        A = MinMaxScaler().fit_transform(X[tr])
        out.append(spectrum(rbf_kernel(A, gamma=1.0 / (A.shape[1] * A.var()))))
    return out


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ref = rbf_reference()
    ref_stats = [stats(l) for l in ref]
    cols = list(ref_stats[0])
    print("RBF reference (mean over folds): " +
          "  ".join(f"{c}={np.mean([s[c] for s in ref_stats]):.4g}" for c in cols))

    for label, template in KERNELS.items():
        files = [KERNEL_DIR / template.format(i=i) for i in range(1, 6)]
        if not all(f.exists() for f in files):
            print(f"[skip] {label}: missing files")
            continue
        specs = [spectrum(np.load(f)) for f in files]
        cspecs = [centered_spectrum(np.load(f)) for f in files]
        st = [stats(l) for l in specs]

        print(f"\n== {label} ==")
        print("fold " + " ".join(f"{c:>9}" for c in cols))
        for i, s in enumerate(st, 1):
            print(f"{i:>4} " + " ".join(f"{s[c]:>9.3g}" for c in cols))
        print("mean " + " ".join(f"{np.mean([s[c] for s in st]):>9.3g}" for c in cols))
        print("centered lam1/trace per fold:",
              np.round([c[0] / c.sum() for c in cspecs], 3))

        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
        for i, (l, c) in enumerate(zip(specs, cspecs), 1):
            ax[0].semilogy(np.arange(1, len(l) + 1), np.clip(l, FLOOR, None), lw=1, label=f"fold {i}")
            ax[1].semilogy(np.arange(1, len(c) + 1), np.clip(c, FLOOR, None), lw=1)
            ax[2].plot(np.arange(1, len(l) + 1), np.cumsum(np.clip(l, 0, None)) / l.sum(), lw=1)
        for l in ref:
            ax[0].semilogy(np.arange(1, len(l) + 1), np.clip(l, FLOOR, None), "k--", lw=0.6, alpha=0.5)
        ax[0].plot([], [], "k--", lw=0.6, label="RBF (ref.)")
        ax[0].set(title="eigenvalues of K", xlabel="index", ylabel="λ (log)")
        ax[1].set(title="eigenvalues of centered K (HKH)", xlabel="index")
        ax[2].set(title="cumulative share of trace", xlabel="index", ylim=(0, 1.02))
        ax[0].legend(fontsize=7)
        fig.suptitle(f"{label} kernel spectrum, 5 folds")
        out = FIG_DIR / f"spectrum_{label.lower().replace(' ', '_')}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
