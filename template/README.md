# Embedding explorer

One static web page over a store of 2-D maps ("views") of the same points: embeddings of one
data set by different models, inputs or seeds. The page switches between views, colours the
points by cluster, label or metadata, shows each view's measures, ranks the views by one
measure, and shows each point's hover media (a GIF, a chess board, or nothing). It needs no
server: it opens from `file://` or any static host.

Building a page needs Python 3.9+ and numpy. Viewing it needs a browser; plotly.js loads from
a CDN unless you pass a local copy (`--plotly`).

## Try it

```bash
python template/example/make_example.py --open
```

This writes three views of 600 synthetic points to `template/example/store/toy/` and builds
`template/example/build/index.html` in a few seconds. Read
[`example/make_example.py`](example/make_example.py) as the tutorial: it builds the data,
writes each view with `write_view`, and calls the builder with
[`example/example_domain.json`](example/example_domain.json).

## Files

| File | Role |
|---|---|
| [`views.py`](views.py) | `write_view()`: writes one view (an `.npz` and a `.json`). The format lives here. |
| [`build_explorer.py`](build_explorer.py) | Reads a store and writes the page and its data files. |
| [`explorer_template.html`](explorer_template.html) | The page; the builder inlines the manifest into it. |
| [`example/`](example/) | The toy store, its domain file and the script that makes both. |
| [`example/pacman_domain.json`](example/pacman_domain.json) | The Pac-Man settings of the explorer in this repository: a full domain file to copy from. |

## Build a page

```bash
python template/build_explorer.py --views <store> --out <page folder> --domain <domain.json>
```

| Flag | Default | |
|---|---|---|
| `--views` | required | The store: one folder per data set, holding `<view_id>.json` + `<view_id>.npz` pairs. Other files there are skipped. |
| `--out` | `build/` beside the builder | The page folder: `index.html` and `data/*.js`. |
| `--domain` | none (generic wording) | The page's wording and settings (below). |
| `--gifs` | `<out>/gifs` | Folder of `<key>.gif`, for `"media": "gif"`. Linked by relative path, not copied. |
| `--data-sets` | all | Comma-separated data-set folders to include (alias `--sequence-types`). |
| `--plotly` | jsDelivr URL | plotly.js URL, or a path relative to the page, for offline use. |

Rerun it whenever views change; it takes seconds. The data files are `<script>`s rather than
JSON, so the page also works from `file://`. A full rebuild deletes the `data/` files no view
uses any more (only the builder's own `table_`, `view_` and `boards_` files). The builder warns
about domain keys the page does not read, which are usually typos.

## The view format

A view is two files in `<store>/<data set>/`, written by `write_view(folder, view_id, ...)`.
The folder name is the data set; every view in a folder must describe the same points.

**`<view_id>.npz`**, one row per point:

| Array | Required | |
|---|---|---|
| `key` | yes | The point's identity (text). Views of one data set share their keys; a GIF is `<key>.gif`. |
| `map2d` | yes, or `map3d` | `(n, 2)` float, the map. A view with only `map3d` opens in 3-D, with 2-D disabled. |
| `cluster` | yes | `(n,)` int, cluster labels; `-1` is noise (grey). |
| `map3d` | no | `(n, 3)`, enables the 3-D toggle. |
| `pca3d` / `pca2d` | no | A PCA map of the same space, enables the PCA toggle. |
| `meta__<name>` | no | Metadata to colour by (numbers or text; a gap is NaN, or `""` in text, and shows grey). Numbers with 12 or fewer distinct values are coloured as categories. |
| `label__<name>` | no | A label's bins as numeric codes (integers); NaN or `-1` where it is undefined. Text categories go in as codes, named by the json's `label_names`. |
| `value__<name>` | no | A label's numeric value (NaN where undefined). The page shows it only beside a `label__<name>` of the same name. |

Points with the same metadata and labels form one table, which the views of a data set share.

A label named `<name>_value` with both `label__` and `value__` arrays is a numeric label: the
colour menu offers its value (as a number) and its bins (as classes), and the colour menu and
the ranking list it before the categorical labels. A label without the suffix is categorical,
under "Other labels".

**`<view_id>.json`**:

| Field | Required | |
|---|---|---|
| `view_id`, `n`, `n_clusters` | yes | Written by `write_view`. The view's id in the page and its URL is `view_id`, or `<data set>_<view_id>` when another data set has a view of the same id. |
| `facets` | no | `{facet: value}` as text: what the page filters views by (model, input, seed...). |
| `label` | no | The view's name in the page (default: its facet values joined with " · ", else its id). |
| `measures` | no | `[{"validation_set": <label>, <measure>: value, ...}]`, one row per label, by its full name (`speed_value`, not `speed`); `null` (or NaN, which `write_view` stores as `null`) where a measure is undefined. |
| `full_space` | no | `{"space", "metric", "dims"}`: where the `full_<measure>` columns were computed. |
| `label_names` | no | `{label: [name of code 0, name of code 1, ...]}`: what each code of a categorical label stands for. (The domain file's `label_names` is something else: a description per label.) |
| `pca_explained` | no | The PCA map's explained variance ratios. |
| `summary` | no | `[{"name", "value", "note"}]` rows shown under the view's note. |
| `reducer_params` | no | `{"metric", "n_neighbors"}`, shown in the view's subtitle. |
| `model_basename` | no | Shown in the measures note as the view's checkpoint. |
| anything else | no | Kept in the json (provenance, a model path...); the page ignores it. The folder, not a `sequence_type` field, decides the data set. |

`write_view` stores numpy numbers and arrays as plain JSON numbers and lists, NaN and infinities
as `null`, and refuses anything else (and maps with NaN coordinates). Keys and ids are best kept
to letters, digits, `_`, `-` and `.`: they become file names and URLs.

The measures are yours to compute: the page shows whatever columns the domain's
`measure_columns` and `measures` name. The example computes neighbourhood hit on the map and in
the full space.

## The domain file

A JSON object whose keys override the page's defaults. Every key is optional. `notes`,
`subtitle`, `privacy`, the titles in `facets`, `measure_columns`, `full_space_names` and
`label_subset`'s note go into the page as HTML (so `R<sup>2</sup>` works there); every
other text is shown as plain text, as is all text from the views (keys, metadata, label and view
names, summaries, measure values).
`"extends": "<file>"` (a path relative to this file) starts from another domain file and
overrides its keys. Keys replace whole values: an `extends`
child that sets `notes` replaces all of its parent's notes.

| Key | Default | |
|---|---|---|
| `title`, `subtitle` | "Embedding explorer", "" | Page title and the line under it. |
| `privacy` | "" | A note under the title (for example, "local only"). |
| `unit_noun` | "Data set" | What the data-set select is called. |
| `label_noun` | "Label" | What a label is called ("Behavlet", ...). |
| `default_type` | first | The data set the page opens on. |
| `media` | "none" | Hover media: `"gif"` (`<gifs>/<key>.gif`), `"board"` (chess, from `boards.jsonl`) or `"none"`. |
| `find_placeholder` | "part of a key" | Hint in the key search box. |
| `participant` | null | A meta column naming who produced each point: shown on the hover card, counted in the gallery. |
| `participant_label`, `participants_noun` | the column name, "participants" | How that column reads. |
| `hidden_colours` | [] | Meta columns kept off the colour menu (still on the pinned card). |
| `facets` | every facet key the views have | `[[key, title], ...]`: the view filters, in order. A facet with one value is hidden. A facet named `experiment` is also appended to each view's name, unless the name already holds it. |
| `value_names` | {} | `{facet: {value: display name}}` for the filters. In a view's label, each part between " · " that equals one of the view's facet values is swapped for its display name. The key `sequence_type` names the data sets in their select and the view's subtitle. |
| `focus` | [] | Labels marked ★ and listed first in their group of the colour menu: a list, or `{data set: [labels]}`, by full name (`speed_value`). |
| `first` | `focus` | Labels listed first, without the star, in the same shapes. |
| `label_names` | none | `{label: description}`: the "Description" toggle shows these instead of the label names (keys without the `_value` suffix). Not the view json's `label_names`, which names a label's codes. |
| `label_subset` | null | `{"name", "note", "sets": {data set: [labels]}}`: an "All labels \| name" toggle. |
| `other_group` | "Other labels" | Heading of the categorical labels in the colour menu. |
| `measures` | NH and kNN R² (adj., on the map and in full), AMI, ARI, NMI, NH and kNN R² (raw) | `[[key, title], ...]`: the ranking's measure menu. |
| `measure_columns` | NH and kNN R², map and full; AMI | `[[key, header, tooltip], ...]`: the measures table. A `full_` column shows only when the view has values for it. |
| `full_space_names` | `{"latent": "the latent", "default": "the input"}` | How the measures note names `full_space.space`. |
| `map_name`, `clusterer` | "Map", "" | Method names: the map toggle's label ("UMAP"), the clusters' ("HDBSCAN"). |
| `notes` | | `{"measures", "caption", "undefined", "no_label", "focus", "point_labels", "summary", "framing"}`: texts under the measures table and appended to it, for an undefined value and an unlabelled point (grey), for the ★, on the pinned card, under a view's summary and under the title. The default `measures` text describes neighbourhood hit and kNN R² at k = 5 in the map and a cosine full space: replace it when your measures differ. |

A complete example: [`example/pacman_domain.json`](example/pacman_domain.json) (GIF hover, Behavlet labels).

## Hover media

The domain's `media` decides what a point shows on hover, when pinned and in the gallery.

- `"none"`: nothing; the gallery lists the selected keys.
- `"gif"`: `<gifs>/<key>.gif`, linked from the page by a relative path. The table records which
  keys had a GIF at build time; the others show "not rendered".
- `"board"`: a chess position that replays a point's moves, from a `boards.jsonl` in the data
  set's folder, one JSON object per line: `key`; `start`, the FEN before the first move;
  `boards`, the piece placement (FEN's first field) after each move; `uci` and `san`, the moves;
  and `focal`, `"white"` or `"black"`, the side shown at the bottom. A key without a line shows
  "no board".

## Publish

The page folder (`--out`) is self-contained except for two things it links to:

- **GIFs.** Copy them into `<out>/gifs/` and build with `--gifs <out>/gifs`, so the page links
  to `gifs/<key>.gif` and the folder can move as one. Otherwise keep the page and the GIF folder
  where they were when you built it.
- **plotly.js.** The page loads it from jsDelivr. For a copy that works offline, download
  `plotly.min.js` (the version `PLOTLY_CDN` in `build_explorer.py` names) into the page folder
  and build with `--plotly plotly.min.js`.

Then serve the folder from any static host (GitHub Pages works) or open `index.html` directly.

## In the page

The URL hash keeps the view and the display (`view`, `color`, `d=3`, `r=pca`, `m` for the
ranking measure, `n=1` for label descriptions, `k=density` with `ref=rel` and `bw` for the
density's reference and bandwidth, `cm=plasma`, `ls=1`, `s` for the measures sort, `rw=<px>` to
fix the right panel's width), so a link reopens what you saw.
