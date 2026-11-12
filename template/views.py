"""Write one view of the embedding explorer's view store: `<folder>/<view_id>.npz` with the
per-point arrays and `<folder>/<view_id>.json` describing it. build_explorer.py reads a store of
these, one folder per data set (README.md has the format). Needs only numpy.

    from views import write_view

    write_view("store/my_data", "umap_seed0", keys, map2d, cluster,
               meta={"participant": ids}, values={"speed_value": speed},
               labels={"speed_value": speed_bins, "route": route_codes},
               label_names={"route": ["left", "right"]},
               facets={"model": "UMAP", "seed": "0"})

Both files are written under temporary names and moved into place, so a reader (or a second
writer, as in an array job) never sees half a view.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

STORE_VERSION = 1


def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.number, np.bool_))


def _is_gap(v) -> bool:
    return v is None or (isinstance(v, (float, np.floating)) and np.isnan(v))


def jsonable(value, where: str = "view"):
    """value as plain JSON: numpy scalars and arrays as Python numbers and lists, NaN and
    infinities as None (JSON has neither), mapping keys as text. Anything else raises."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v, f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [jsonable(v, f"{where}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(
        f"{where}: {type(value).__name__} can't be stored in a view's json "
        "(use numbers, text, lists and dicts)"
    )


def _column(values, name: str, n: int, kind: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != n:
        raise ValueError(f"{kind} {name!r}: shape {array.shape}, expected ({n},)")
    if array.dtype.kind in "biuf":
        return array if kind == "meta" else array.astype(float)
    # The elements as given: np.asarray turns a list mixing numbers and text into all text.
    items = values.tolist() if isinstance(values, np.ndarray) else list(values)
    if kind == "meta":
        # Numbers with gaps (None) stay numbers, with NaN for the gaps; anything else is text.
        present = [v for v in items if not _is_gap(v)]
        if present and all(_is_number(v) for v in present):
            return np.array([np.nan if _is_gap(v) else v for v in items], dtype=float)
        # Text: a gap (None or NaN) is stored as "" and shown grey, not as a "None" category.
        return np.array(["" if _is_gap(v) else str(v) for v in items], dtype=str)
    bad = next((v for v in items if v is not None and not _is_number(v)), None)
    if bad is not None:
        what = (
            "bins must be numeric codes (integers; NaN or -1 where undefined), with text "
            "names for the codes in label_names"
            if kind == "label"
            else "values must be numbers (NaN or None where undefined)"
        )
        raise ValueError(f"{kind}s {name!r}: {what}; got {bad!r}") from None
    return np.array([np.nan if v is None else v for v in items], dtype=float)


def _code_names(names):
    """A label's code names as a list, or as {code: name} with integer codes written "2" (not
    "2.0", which the page would not find)."""
    if not isinstance(names, Mapping):
        return names
    out = {}
    for code, name in names.items():
        if isinstance(code, (float, np.floating)) and float(code).is_integer():
            code = int(code)
        out[str(code.item() if isinstance(code, np.generic) else code)] = name
    return out


def _map(values, n: int, dims: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (n, dims):
        raise ValueError(f"{name}: {array.shape}, expected ({n}, {dims})")
    if not np.isfinite(array).all():
        raise ValueError(
            f"{name}: {int((~np.isfinite(array)).any(axis=1).sum())} points have NaN or inf coordinates"
        )
    return array


def write_view(
    folder: str | os.PathLike,
    view_id: str,
    keys: Sequence,
    map2d,
    cluster,
    *,
    meta: Mapping | None = None,
    labels: Mapping | None = None,
    values: Mapping | None = None,
    map3d=None,
    pca=None,
    pca_explained: Sequence[float] | None = None,
    measures: list[dict] | None = None,
    full_space: dict | None = None,
    label_names: dict | None = None,
    facets: Mapping[str, str] | None = None,
    label: str | None = None,
    summary: list[dict] | None = None,
    **fields,
) -> Path:
    """Write one view and return the path of its npz.

    Per point (one row each, all the same length):
        keys       the point's identity, the same in every view of the data set; the hover
                   media is <gifs>/<key>.gif
        map2d      (n, 2) the map. May be None when map3d is given: a 3-D-only view, which
                   the page opens in 3-D with 2-D disabled
        cluster    integer cluster labels, -1 for noise
        meta       {name: array} metadata to colour by (numbers, None/NaN for gaps, or text)
        labels     {name: array} a label's bins as numeric codes (NaN, None or -1: undefined);
                   name the codes of a categorical label in label_names. A numeric label is
                   "<name>_value", in both labels (its bins) and values
        values     {name: array} a label's numeric value, coloured as a number
        map3d      (n, 3) optional 3-D map
        pca        (n, 2) or (n, 3) optional PCA map, with pca_explained its variance ratios

    About the view (json):
        measures       [{"validation_set": name, <measure>: value, ...}], one row per label
        full_space     {"space", "metric", "dims"} the space the full_<measure> columns were
                       computed in
        label_names    {label: [name of code 0, of code 1, ...]} what each code of a
                       categorical label stands for (an object {code: name} works too)
        facets         {facet: value} the fields the page filters views by, as text
        label          the view's name in the page (default: its facet values, joined)
        summary        [{"name", "value", "note"}] rows shown under the view's note
        **fields       anything else to record (provenance, a model path, ...)
    """
    keys = np.asarray(keys)
    if keys.ndim != 1:
        raise ValueError(f"keys: shape {keys.shape}, expected one key per point")
    keys = keys.astype(str)
    n = len(keys)
    arrays: dict[str, np.ndarray] = {"key": keys}
    cluster = np.asarray(cluster)
    if cluster.shape != (n,):
        raise ValueError(f"cluster: shape {cluster.shape}, expected ({n},)")
    if cluster.dtype.kind not in "biu" and not (
        cluster.dtype.kind == "f"
        and np.isfinite(cluster).all()
        and (cluster == np.round(cluster)).all()
    ):
        raise ValueError("cluster: integer labels expected (-1 for noise)")
    arrays["cluster"] = cluster.astype(np.int32)
    if map2d is None and map3d is None:
        raise ValueError("A view needs map2d (or at least map3d)")
    if map2d is not None:
        arrays["map2d"] = _map(map2d, n, 2, "map2d")
    if map3d is not None:
        arrays["map3d"] = _map(map3d, n, 3, "map3d")
    if pca is not None:
        pca = np.asarray(pca, dtype=np.float32)
        if pca.ndim != 2 or pca.shape[0] != n or pca.shape[1] not in (2, 3):
            raise ValueError(f"pca: {pca.shape}, expected ({n}, 2) or ({n}, 3)")
        arrays[f"pca{pca.shape[1]}d"] = _map(pca, n, pca.shape[1], "pca")
    for prefix, columns in (("meta", meta), ("label", labels), ("value", values)):
        for name, column in (columns or {}).items():
            arrays[f"{prefix}__{name}"] = _column(column, name, n, prefix)

    description = {
        "view_id": view_id,
        "store_version": STORE_VERSION,
        **fields,
        "n": int(n),
        "n_clusters": len(set(arrays["cluster"].tolist()) - {-1}),
        "has_map3d": "map3d" in arrays,
        "pca_explained": None if pca_explained is None else list(pca_explained),
        "label_names": {
            str(column): _code_names(names)
            for column, names in (label_names or {}).items()
        },
        "measures": measures,
        "full_space": full_space,
    }
    if facets is not None:
        description["facets"] = {str(k): str(v) for k, v in facets.items()}
        description["label"] = label or " · ".join(description["facets"].values())
    elif label is not None:
        description["label"] = label
    if summary is not None:
        description["summary"] = summary
    description.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S"))

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = folder / f"{view_id}.npz", folder / f"{view_id}.json"
    tmp_npz = folder / f".{view_id}.tmp.npz"
    tmp_json = json_path.with_name(json_path.name + ".tmp")
    text = json.dumps(jsonable(description, view_id), indent=2, allow_nan=False)
    np.savez_compressed(tmp_npz, **arrays)
    os.replace(tmp_npz, npz_path)
    tmp_json.write_text(text, encoding="utf-8")
    os.replace(tmp_json, json_path)
    return npz_path
