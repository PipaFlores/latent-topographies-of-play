"""Build the embedding explorer: one static Plotly page over a view store.

Every view in <views>/<data set>/<view_id>.{json,npz} (written by views.write_view) becomes one
selectable map. The page switches between the views of a data set by their facets, colours the
points by cluster, label (value or bin) or metadata, shows each view's measures, ranks the views
by one measure, and shows each point's hover media. Views with a 3-D map can be shown in 3-D.
README.md describes the view format, the domain file and the hover media.

--domain <file.json> sets the page's wording, facets, focus labels, label descriptions, measures
and hover media (gif | board | none); a domain file may name another that it "extends" and
override its keys. Without --domain the page uses the template's generic defaults.

Output, all under --out (default: build/ beside this script):

    index.html                 the page, with the manifest of every view inlined
    data/table_<id>.js         per data set: keys, GIF presence, metadata, labels
    data/view_<id>.js          per view: cluster labels, 2-D map, 3-D map, PCA
    data/boards_<id>.js        per data set with a boards.jsonl: the positions of each key's plies

The data files are <script>s rather than JSON, so the page also works from file:// (no server).
GIFs are linked, not copied: <img src> is the path from --out to --gifs (default: a gifs/ folder
inside --out), so keep the two folders where they were when the page was built, or copy the
GIFs into <out>/gifs/ and build with that default. A point whose GIF is missing shows "not
rendered". Needs only numpy.

    python build_explorer.py --views my_store --out my_page --domain my_domain.json
    python build_explorer.py --views my_store --data-sets runs_a,runs_b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import urllib.parse
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "explorer_template.html"
PLOTLY_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"


# The keys a domain file may set (README.md, "The domain file"); others are reported as typos.
DOMAIN_KEYS = {
    "title", "subtitle", "privacy", "unit_noun", "label_noun", "default_type", "media",
    "find_placeholder", "participant", "participant_label", "participants_noun",
    "hidden_colours", "facets", "value_names", "focus", "first", "label_names",
    "label_subset", "other_group", "measures", "measure_columns", "full_space_names",
    "map_name", "clusterer", "notes",
}  # fmt: skip
# The data files the builder writes (a rebuild deletes only stale ones of these).
DATA_PREFIXES = ("table_", "view_", "boards_")


def parse_args(argv=None, defaults: dict | None = None) -> argparse.Namespace:
    """The command line; `defaults` ({dest: value}) lets a wrapper script set its own store,
    GIFs and domain (main(defaults=...))."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--views",
        default=None,
        help="View store folder: one folder per data set (required).",
    )
    parser.add_argument(
        "--gifs", default=None, help="Folder holding <key>.gif (default: <out>/gifs)."
    )
    parser.add_argument(
        "--out",
        default=str(HERE / "build"),
        help="Output folder (default: build/ beside this script).",
    )
    parser.add_argument(
        "--data-sets",
        "--sequence-types",
        dest="data_sets",
        default=None,
        help="Comma-separated data-set folders to include (default: all).",
    )
    parser.add_argument(
        "--plotly",
        default=PLOTLY_CDN,
        help="plotly.js URL, or a path relative to the page, for offline use.",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Domain JSON: wording, facets, focus labels, measures and media "
        "(default: the template's generic wording).",
    )
    parser.set_defaults(**(defaults or {}))
    args = parser.parse_args(argv)
    if args.views is None:
        parser.error(
            "pass --views <store folder> (one folder per data set of view files)"
        )
    if args.gifs is None:
        args.gifs = str(Path(args.out) / "gifs")
    return args


def load_domain(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    """A domain file's keys over those of the file it "extends" (a path relative to it)."""
    path = path.resolve()
    if path in seen:
        raise SystemExit(f"Domain files extend each other in a loop: {path}")
    try:
        domain = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Domain file {path}: {exc}") from None
    if unknown := sorted(set(domain) - DOMAIN_KEYS - {"extends"}):
        print(
            f"warning: {path.name} sets keys the page does not read: {', '.join(unknown)}"
        )
    domain = _finite(
        domain
    )  # json.loads reads NaN and Infinity, which the page's JSON cannot hold
    parent = domain.pop("extends", None)
    if parent is None:
        return domain
    return {**load_domain(path.parent / parent, (*seen, path)), **domain}


# --- number formatting: the data files are the page's size, so keep them short -------------


def _num(value) -> str:
    if value is None:
        return "null"
    value = float(value)
    if not math.isfinite(value):
        return "null"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.4g}"


def _array(values: np.ndarray, digits: int | None = None) -> str:
    if values.dtype.kind in "USO":
        # views.write_view stores a gap in a text column as "": null, so the page greys it.
        return json.dumps(
            [None if v is None or v == "" else str(v) for v in values.tolist()]
        )
    if values.dtype.kind in "iub":
        return "[" + ",".join(str(int(v)) for v in values) + "]"
    if digits is not None:
        return (
            "["
            + ",".join(
                "null" if not math.isfinite(v) else f"{v:.{digits}f}"
                for v in values.tolist()
            )
            + "]"
        )
    return "[" + ",".join(_num(v) for v in values.tolist()) + "]"


def _write_js(path: Path, data_id: str, parts: dict[str, str]) -> int:
    body = "{" + ",".join(f"{json.dumps(k)}:{v}" for k, v in parts.items()) + "}"
    text = f"EXPLORER_LOAD({json.dumps(data_id)},{body});\n"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return len(text)


def _object(columns: dict[str, np.ndarray]) -> str:
    return (
        "{"
        + ",".join(
            f"{json.dumps(name)}:{_array(values)}" for name, values in columns.items()
        )
        + "}"
    )


# --- views -------------------------------------------------------------------------------


def _finite(value):
    """value with NaN and infinities as None, for JSON (a json written by something other than
    views.write_view may hold them)."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _table_columns(npz) -> dict[str, dict[str, np.ndarray]]:
    groups = {"meta": {}, "label": {}, "value": {}}
    for name in npz.files:
        prefix, _, column = name.partition("__")
        if prefix in groups:
            groups[prefix][column] = npz[name]
    return groups


def _table_fingerprint(
    keys: np.ndarray, columns: dict[str, dict[str, np.ndarray]]
) -> str:
    digest = hashlib.md5(keys.tobytes())
    for group in sorted(columns):
        for name in sorted(columns[group]):
            digest.update(f"{group}/{name}".encode())
            digest.update(np.ascontiguousarray(columns[group][name]).tobytes())
    return digest.hexdigest()[:10]


def _write_boards(
    path: Path, table_id: str, keys: np.ndarray, boards_path: Path
) -> tuple[int, int]:
    """data/boards_<table>.js: per key of the table (in its order) the start FEN, the placement after
    every ply, uci and san, or null for a key boards.jsonl lacks. Returns (bytes, keys with a board)."""
    by_key = {}
    with open(boards_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                b = json.loads(line)
                by_key[b["key"]] = b
    rows = [by_key.get(key) for key in keys.tolist()]
    fields = ("start", "boards", "uci", "san", "focal")
    parts = {
        name: json.dumps(
            [None if r is None else r.get(name) for r in rows], separators=(",", ":")
        )
        for name in fields
    }
    n_boards = sum(r is not None for r in rows)
    return _write_js(path, f"boards_{table_id}", parts), n_boards


def _read_store(views_dir: Path) -> list[tuple[Path, dict]]:
    """Every view json under <views>/<data set>/ with its npz beside it; its folder is its data set."""
    store = []
    for json_path in sorted(views_dir.glob("*/*.json")):
        if json_path.name.startswith(".") or not json_path.with_suffix(".npz").exists():
            print(
                f"skipped {json_path}: not a view (no {json_path.stem}.npz beside it)"
            )
            continue
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"{json_path}: {exc}") from None
        if not isinstance(meta, dict) or "n" not in meta or "n_clusters" not in meta:
            print(f"skipped {json_path}: not a view (no n or n_clusters)")
            continue
        # A view's data set is its folder (a sequence_type field in the json does not move it).
        meta["sequence_type"] = json_path.parent.name
        store.append((json_path, _finite(meta)))
    return store


def _gif_base(gifs_dir: Path, out_dir: Path) -> str:
    """How the page links the GIFs: the path from the page to them, or a file:// URL when they
    are on another drive."""
    try:
        relative = Path(os.path.relpath(gifs_dir.resolve(), out_dir.resolve()))
    except ValueError:
        return gifs_dir.resolve().as_uri() + "/"
    # Percent-encoded, so a "#", "?" or "%" in a folder name stays part of the path.
    return urllib.parse.quote(relative.as_posix()) + "/"


def _inline(manifest: dict) -> str:
    """The manifest as a script literal, with "</" and "<!--" escaped so that no text from the
    data (a label, key or summary) can end the page's <script>. Both escapes are valid JSON, so
    the inlined manifest still reads as JSON."""
    text = json.dumps(manifest, separators=(",", ":"), allow_nan=False)
    return text.replace("</", "<\\/").replace("<!--", "\\u003c!--")


def _page_ids(views: list[tuple[Path, dict]]) -> dict[Path, str]:
    """Each view's id in the page (its data file, the manifest and the URL hash): its view_id
    (else its file name), prefixed with "<data set>_" when another data set has the same one."""
    bare = {path: str(meta.get("view_id") or path.stem) for path, meta in views}
    sets_of = defaultdict(set)
    for path, meta in views:
        sets_of[bare[path]].add(meta["sequence_type"])
    ids = {
        path: bare[path]
        if len(sets_of[bare[path]]) == 1
        else f"{meta['sequence_type']}_{bare[path]}"
        for path, meta in views
    }
    seen = {}
    for path, page_id in ids.items():
        if page_id in seen:
            raise SystemExit(
                f"Two views have the page id {page_id!r}: {seen[page_id]} and {path}"
            )
        seen[page_id] = path
    return ids


def main(
    argv=None,
    defaults: dict | None = None,
    view_facets: Callable[[dict], tuple[dict, str] | None] | None = None,
) -> int:
    """Build the page. A wrapper script may pass `defaults` for the flags, and `view_facets`: a
    function from a view's json to its (facets, label), or to None to use the json's own."""
    args = parse_args(argv, defaults)
    started = time.perf_counter()
    views_dir, out_dir, gifs_dir = Path(args.views), Path(args.out), Path(args.gifs)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.data_sets.split(",")) if args.data_sets else None
    domain = load_domain(Path(args.domain)) if args.domain else None

    present_gifs = (
        {p.stem for p in gifs_dir.glob("*.gif")} if gifs_dir.is_dir() else set()
    )
    gif_base = _gif_base(gifs_dir, out_dir)

    store = _read_store(views_dir)
    # Over the whole store, so a view keeps its id (and its links) in a build of fewer data sets.
    page_ids = _page_ids(store)
    views = [(p, m) for p, m in store if wanted is None or m["sequence_type"] in wanted]

    manifest_views, tables, written = [], {}, 0
    for json_path, meta in views:
        view_id = page_ids[json_path]
        with np.load(json_path.with_suffix(".npz"), allow_pickle=False) as npz:
            keys = npz["key"]
            columns = _table_columns(npz)
            table_id = f"{meta['sequence_type']}_{_table_fingerprint(keys, columns)}"
            if table_id not in tables:
                has_gif = np.array(
                    [key in present_gifs for key in keys.tolist()], dtype=np.int8
                )
                written += _write_js(
                    data_dir / f"table_{table_id}.js",
                    table_id,
                    {
                        "keys": json.dumps(keys.tolist()),
                        "gif": _array(has_gif),
                        "meta": _object(columns["meta"]),
                        "label": _object(columns["label"]),
                        "value": _object(columns["value"]),
                    },
                )
                tables[table_id] = {
                    "sequence_type": meta["sequence_type"],
                    "n": len(keys),
                    "n_gifs": int(has_gif.sum()),
                    "meta": list(columns["meta"]),
                    "label": list(columns["label"]),
                    "value": list(columns["value"]),
                    "label_names": {},
                }
                boards_path = json_path.parent / "boards.jsonl"
                if boards_path.exists():
                    size, n_boards = _write_boards(
                        data_dir / f"boards_{table_id}.js", table_id, keys, boards_path
                    )
                    written += size
                    tables[table_id]["has_boards"] = True
                    tables[table_id]["n_boards"] = n_boards
            # Names of the categorical labels' codes (routes, quadrants, attack ends).
            for column, names in (meta.get("label_names") or {}).items():
                tables[table_id]["label_names"].setdefault(column, names)
            parts = {"cluster": _array(npz["cluster"])}
            # A view without map2d (3-D only) opens in 3-D; the page disables 2-D for it.
            if "map2d" in npz.files:
                map2d = npz["map2d"]
                parts["x"], parts["y"] = _array(map2d[:, 0], 3), _array(map2d[:, 1], 3)
            elif "map3d" not in npz.files:
                raise SystemExit(f"{json_path.with_suffix('.npz')}: no map2d or map3d")
            if "map3d" in npz.files:
                map3d = npz["map3d"]
                parts["x3"], parts["y3"], parts["z3"] = (
                    _array(map3d[:, i], 3) for i in range(3)
                )
            pca_name = next((n for n in ("pca3d", "pca2d") if n in npz.files), None)
            if pca_name:
                pca = npz[pca_name]
                for i in range(pca.shape[1]):
                    parts[f"p{i + 1}"] = _array(pca[:, i], 4)
            has_pca = pca_name is not None
            written += _write_js(data_dir / f"view_{view_id}.js", view_id, parts)

        found = view_facets(meta) if view_facets else None
        if found:
            facets, label = found
        else:
            facets = {str(k): str(v) for k, v in (meta.get("facets") or {}).items()}
            label = meta.get("label") or " · ".join(facets.values()) or view_id
        manifest_views.append(
            {
                "id": view_id,
                "sequence_type": meta["sequence_type"],
                "table": table_id,
                "label": label,
                "facets": facets,
                "n": meta["n"],
                "n_clusters": meta["n_clusters"],
                "has_map3d": bool(meta.get("has_map3d")),
                "has_pca": has_pca,
                "pca_explained": meta.get("pca_explained"),
                "model": meta.get("model_basename"),
                "reducer_params": meta.get("reducer_params"),
                "created_at": meta.get("created_at"),
                "measures": meta.get("measures") or [],
                "full_space": meta.get("full_space"),
            }
        )
        if meta.get("summary"):
            manifest_views[-1]["summary"] = meta["summary"]

    if not manifest_views:
        raise SystemExit(
            f"No views under {views_dir}{' for ' + args.data_sets if wanted else ''}."
        )

    # Data files no view refers to any more (a view rebuilt with new content, a dropped type).
    live = {f"table_{t}.js" for t in tables} | {
        f"view_{v['id']}.js" for v in manifest_views
    }
    live |= {f"boards_{t}.js" for t, table in tables.items() if table.get("has_boards")}
    if wanted is None:
        for stale in data_dir.glob("*.js"):
            if stale.name.startswith(DATA_PREFIXES) and stale.name not in live:
                stale.unlink()

    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M"),
        "gif_base": gif_base,
        "views": manifest_views,
        "tables": tables,
    }
    if domain is not None:
        manifest["domain"] = domain
    page = (
        TEMPLATE.read_text(encoding="utf-8")
        .replace("__PLOTLY_SRC__", args.plotly)
        .replace("/*__MANIFEST__*/null", _inline(manifest))
    )
    (out_dir / "index.html").write_text(page, encoding="utf-8")

    per_type = defaultdict(int)
    for view in manifest_views:
        per_type[view["sequence_type"]] += 1
    print(
        f"{len(manifest_views)} views, {len(tables)} tables, {written / 1e6:.1f} MB of data "
        f"in {time.perf_counter() - started:.0f}s -> {out_dir / 'index.html'}"
    )
    for sequence_type, count in sorted(per_type.items()):
        table = next(t for t in tables.values() if t["sequence_type"] == sequence_type)
        if table.get("has_boards"):
            media = f", {table['n_boards']}/{table['n']} boards"
        elif (domain or {}).get("media") == "gif":
            media = f", {table['n_gifs']}/{table['n']} GIFs rendered"
        else:
            media = ""
        print(f"  {sequence_type}: {count} views{media}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
