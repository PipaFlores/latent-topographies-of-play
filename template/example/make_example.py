"""A toy view store and its explorer page, from synthetic data: a tutorial and a smoke test.

Three groups of points in 12 dimensions, with a numeric label ("spread", how far a point lies
from its group's centre) and a categorical one ("group"). Three views map the same points: a
PCA of the clean data, a PCA of the data with noise added, and a random projection. Each view
gets k-means clusters and two measures per label (neighbourhood hit on the map and in the full
space), so the page's colour menu, measures panel and ranking all have something to show.
Points have no hover media (the domain file sets "media": "none").

    python template/example/make_example.py            # store/ and build/ beside this file
    python template/example/make_example.py --open     # and open the page

Needs only numpy. Runs in a few seconds.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from views import write_view

N_PER_GROUP, DIMS, K = 200, 12, 5


def pca(X: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The first n principal components of X and their explained variance ratios."""
    Xc = X - X.mean(axis=0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:n].T, (s**2 / (s**2).sum())[:n]


def kmeans(
    X: np.ndarray, k: int, rng: np.random.Generator, steps: int = 50
) -> np.ndarray:
    centres = X[rng.choice(len(X), k, replace=False)]
    for _ in range(steps):
        assign = ((X[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
        centres = np.array(
            [
                X[assign == c].mean(0) if (assign == c).any() else centres[c]
                for c in range(k)
            ]
        )
    return assign


def neighbourhood_hit(X: np.ndarray, bins: np.ndarray, k: int = K) -> float:
    """Share of each point's k nearest neighbours in its own bin, minus that share by chance."""
    d = ((X[:, None] - X[None]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    neighbours = np.argsort(d, axis=1)[:, :k]
    hit = (bins[neighbours] == bins[:, None]).mean()
    chance = sum((np.mean(bins == b)) ** 2 for b in np.unique(bins))
    return float(hit - chance)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--store", default=str(HERE / "store"), help="View store to write."
    )
    parser.add_argument("--out", default=str(HERE / "build"), help="Page folder.")
    parser.add_argument(
        "--open", action="store_true", help="Open the page when it is built."
    )
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    # The data: three groups, each spread around its own centre.
    centres = rng.normal(0, 4, (3, DIMS))
    group = np.repeat(np.arange(3), N_PER_GROUP)
    offsets = rng.normal(0, 1, (len(group), DIMS)) * rng.uniform(
        0.3, 2.0, (len(group), 1)
    )
    X = centres[group] + offsets
    spread = np.linalg.norm(offsets, axis=1)
    keys = [f"point_{i:04d}" for i in range(len(X))]
    participant = rng.integers(1, 21, len(X))

    # A numeric label is "<name>_value": its value colours as a number, its bins as classes.
    spread_bins = np.digitize(spread, np.quantile(spread, [0.25, 0.5, 0.75]))
    labels = {"spread_value": spread_bins, "group": group}
    values = {"spread_value": spread}
    label_names = {"group": ["group A", "group B", "group C"]}

    spaces = {
        "pca_clean": ("PCA", "none", X),
        "pca_noisy": ("PCA", "high", X + rng.normal(0, 3, X.shape)),
        "random_projection": (
            "Random projection",
            "none",
            X @ rng.normal(0, 1, (DIMS, DIMS)),
        ),
    }
    folder = Path(args.store) / "toy"
    for view_id, (method, noise, space) in spaces.items():
        map3d, explained = pca(space, 3) if method == "PCA" else (space[:, :3], None)
        cluster = kmeans(map3d[:, :2], 4, rng)
        measures = [
            {
                "validation_set": name,
                "neigh_hit_adj": neighbourhood_hit(map3d[:, :2], bins),
                "full_neigh_hit_adj": neighbourhood_hit(space, bins),
            }
            for name, bins in labels.items()
        ]
        write_view(
            folder,
            view_id,
            keys,
            map2d=map3d[:, :2],
            cluster=cluster,
            meta={"participant": participant},
            labels=labels,
            values=values,
            map3d=map3d,
            pca=None if explained is None else map3d,
            pca_explained=explained,
            measures=measures,
            full_space={"space": "input", "metric": "euclidean", "dims": DIMS},
            label_names=label_names,
            facets={"method": method, "noise": noise},
            summary=[
                {"name": "method", "value": method, "note": "of the 12-D points"},
                {"name": "noise", "value": noise},
            ],
        )
    print(f"{len(spaces)} views -> {folder}")

    subprocess.run(
        [
            sys.executable,
            str(HERE.parent / "build_explorer.py"),
            "--views",
            args.store,
            "--out",
            args.out,
            "--domain",
            str(HERE / "example_domain.json"),
        ],
        check=True,
    )
    page = Path(args.out) / "index.html"
    if args.open:
        webbrowser.open(page.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
