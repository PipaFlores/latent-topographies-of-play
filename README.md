# Latent Topographies of Play

An interactive explorer of latent spaces learned from human Pac-Man play.
It accompanies a demo paper under review at ACM IUI 2027; the citation will follow on acceptance.

**Open the explorer:** https://pipaflores.github.io/latent-topographies-of-play/

The explorer holds 70 views: two sequence types, each in five input representations, each embedded by one of seven methods.
Four are autoencoders and three are baselines: the raw input, a generic summary and an engineered summary.
Every view projects its embedding to 2-D and 3-D maps.

**Sequence type:** The explorer covers two sequence types from 263 players, logged at 20 Hz:

- **Openings:** the first five seconds of a level (2,317 sequences).
- **Hunts:** power-pill episodes (7,029 sequences), described by Behavlets, model-based gameplay
  features derived from psychologically interpretable traits such as aggression and caution.

**Input representation:** Sequences can be represented in 5 ways:

- **Pac-Man + ghost positions:** Pac-Man and ghost x and y positions (10 features).
- **Ghost distances:** Maze distances from Pac-Man to the ghosts, sorted by proximity (4 features).
- **Distances + Score:** Same as above plus score (5 features).
- **Compressed game state:** Pac-Man and ghost positions, ghost distances, score, pellet and power-pill counts, and the states of the four power pills and two fruits (23 features).
- **Full game state:** All the above plus a 244-column binary map of which pellets remain (267 features).

**Embedding and baselines:**

- **Autoencoders:** Four deep networks (LSTM, MLP, VAE, and TSDAE), each with a 256-dimensional latent space.
- **Raw input:** The input sequence itself, flattened. Hunts vary in length, so each is padded to the longest (217 steps) by repeating its last step.
- **Generic summary:** The mean, SD, min and max of each variable in the input representation, plus the sequence length.
- **Engineered summary:** The generic summary plus each column's last value and, for inputs with the score, the number of score jumps.

Next to each view there are chance-corrected scores of how strongly each descriptor organises
the map's neighbourhoods (neighbourhood hit, kNN R²) and its clusters (AMI).
Clicking a score recolours the map by that descriptor.
A ranking compares all embeddings of one sequence type on any score, and hovering a point replays its sequence.

## Contents

| Path | What |
|---|---|
| `index.html` | The page, with the manifest of every view inlined |
| `data/` | One script per sequence type (keys, metadata, descriptors) and per view (clusters, 2-D and 3-D maps) |
| `gifs/` | Replays, downscaled to 320 px at 10 fps |
| `plotly.min.js` | plotly.js 2.35.2 (MIT, see `plotly.LICENSE.txt`) |
| `template/` | The explorer as a general-purpose tool, with a guide and a toy example |

The page is static and needs no server: open `index.html` from a local clone, or serve the
folder with `python -m http.server`.

## Use the explorer with your own data

`template/` holds the explorer as a general-purpose tool: the page template, the builder, a
writer for the view format, a toy example and a guide (`template/README.md`).
Building a page needs only Python and numpy:

```bash
python template/example/make_example.py --open
```

This builds a toy explorer from synthetic points in a few seconds.
`template/example/pacman_domain.json` holds the settings of the Pac-Man explorer above, as a full example.

## Data and consent

This repository holds only the gameplay sequences shown in the explorer, as replays and derived
descriptors.
The umbrella study (work in progress) that collected the gameplay data followed the Declaration of Helsinki.
Participants consented to the publication of their anonymised gameplay data.
Player, game and level identifiers are replaced by pseudonyms; the key to the study's IDs is not published.

## Licence

The page and the code in `template/` are under the MIT licence (`LICENSE`).
The data and replays in `data/` and `gifs/` are under CC BY 4.0 (`LICENSE-DATA.md`): free to use and adapt with credit.

## Citation

If you use the explorer or its data, please cite this repository:

```bibtex
@misc{floresromero2026latenttopographies,
  author       = {Flores Romero, Pablo and Rudakov, Evgenii and Cowley, Benjamin U.},
  title        = {Latent Topographies of Play: An Explorer of Latent Gameplay},
  year         = {2026},
  note         = {Data set and software},
  howpublished = {\url{https://github.com/PipaFlores/latent-topographies-of-play}}
}
```
