# NF-MCD: Neuro-Fuzzy Multimodal Community Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22959565.svg)](https://doi.org/10.5281/zenodo.22959565)

Reference implementation accompanying the draft paper *"Neuro-Fuzzy
Multimodal Community Detection: Fusing Visual-Textual Semantics with
Network Topology for Explainable Social Network Analysis."*

## What this is

A working, five-stage NF-MCD pipeline, evaluated end-to-end on five real
social-network datasets (CrisisMMD, PHEME, Fakeddit, SNAP DBLP, SNAP Amazon),
a real relational (non-label-sampled) Fakeddit graph, and a controllable
synthetic generator, backed by close to two dozen targeted stress tests
(missing modalities, structural noise, cross-modal mismatch, scale up to
20,000 nodes, statistical significance re-analysis, and more). Every number
reported in the accompanying manuscript is traceable to a run recorded under
`experiments/` in this repository.

The `data/` directory (raw downloaded datasets) and `experiments/cache/`
(regeneratable fitted-model pickles) are not tracked in git — see "Quick
start" below to reproduce them.

## Package layout

Each module corresponds directly to a subsection of the paper's Section 4,
so you can cite specific files/functions in your Methods section:

| Module | Paper section | What it does |
|---|---|---|
| `nf_mcd/encoders.py` | 4.1 | Text (sentence-transformers) + image (CLIP) encoders, with a deterministic fallback when pretrained models aren't reachable |
| `nf_mcd/fuzzy_fusion.py` | 4.2 | PCA-whitened CCA cross-modal alignment, three-term ANFIS fuzzy confidence, fused content embedding |
| `nf_mcd/topology.py` | 4.3 | Spectral structural embedding + fuzzy content/structure integration (per-node trust weight alpha) |
| `nf_mcd/community_detection.py` | 4.4 | Fuzzy c-means soft/overlapping community detection, from scratch |
| `nf_mcd/explain.py` | 4.5 / 5.4 | Per-node explanations, global IF-THEN fuzzy rules, and the neighborhood-based explanation alternative developed in the paper |
| `nf_mcd/metrics.py` | 5.3 | Modularity, LFK overlapping NMI (via `cdlib`), membership F1 |
| `nf_mcd/datasets.py` | 5.1 | Synthetic multimodal graph generator + real dataset loaders (CrisisMMD, PHEME, Fakeddit; SNAP DBLP/Amazon) |
| `nf_mcd/baselines.py` | 5.2 | Every baseline and NF-MCD ablation used in the evaluation, behind one registry |
| `nf_mcd/clusterers.py` | — | Alternative clustering back-ends (`fcm_adaptive_m`, `gmm`, `kmeans_softmax`, `spectral_soft`) used by `NFMCD.robust()` and other configurations |
| `nf_mcd/pipeline.py` | 4 (all) | `NFMCD` class: the scikit-learn-style `fit`/`predict` interface tying everything together, plus the `NFMCD.robust()` preset |

`run_*.py` at the repo root are the individual evaluation experiments (baselines,
missing-modality robustness, structural-noise sweep, CCA rank-cap sweep, scale,
significance testing, overlap-threshold sensitivity, explainability, and more);
each writes its results, a progress log, and a summary log under `experiments/`.

## Quick start

```bash
py -m pip install -r requirements.txt
py demo.py
```

`demo.py` runs the pipeline on synthetic data (no model downloads needed) —
a fast, dependency-light sanity check of the fusion → topology → clustering
→ explanation flow. For real experiments with real CLIP/sentence-transformer
encoders and real datasets:

```bash
py run_real_experiments.py            # all five datasets
py run_real_experiments.py crisismmd  # just one
```

Raw downloaded data lands in `data/`; per-dataset logs and community plots in
`experiments/`. Without `sentence-transformers`/`transformers`/`torch`
installed *and* network access to download pretrained weights,
`nf_mcd.encoders` silently falls back to deterministic pseudo-random
embeddings (logged via `UserWarning`) — fine for pipeline development, not
valid for real results.

## Minimal usage example

```python
from nf_mcd import NFMCD
from nf_mcd.datasets import generate_synthetic_multimodal_graph

data = generate_synthetic_multimodal_graph(n_nodes=120, n_communities=4, seed=0)

model = NFMCD(n_communities=4, seed=0)
model.fit(data.G, text_embeddings=data.text_embeddings, image_embeddings=data.image_embeddings)

hard_labels = model.predict_hard()
print(model.explain_node(node_id=0).text)
for rule in model.global_rules():
    print(rule.text)

scores = model.evaluate(true_communities_per_node=data.true_communities)
print(scores)  # {'modularity': ..., 'overlapping_nmi': ..., 'membership_f1': ..., 'rule_fidelity': ...}
```

To use real text/images instead of precomputed embeddings:

```python
model.fit(G, texts=list_of_captions, images=list_of_PIL_images)
```

For the more robust preset (adaptive fuzziness, unsupervised content
features, a tighter CCA rank cap — see the manuscript, Section 3):

```python
model = NFMCD.robust(n_communities=4, seed=0)
```

## Key calibration choices (validated, not defaults you should casually change)

Three defaults deliberately depart from textbook choices, each backed by an
ablation reported in the manuscript:

1. **Cross-modal fusion uses PCA-whitened CCA fit on paired nodes**, not a
   fixed random projection (`fuzzy_fusion.py`) — a random projection cannot
   recover cross-modal alignment that was never linearly present.
2. **`structural_dim` defaults to `n_communities`**, not a larger "safe"
   eigenspace (`topology.py`) — requesting far more spectral eigenvectors
   than the true community count dilutes signal once row-normalized.
3. **Fuzzy c-means uses `m=1.5`**, not the textbook `m=2.0`
   (`community_detection.py`) — at `m=2.0`, memberships on this pipeline's
   feature scale collapse to ~1/k for every node.

The ANFIS confidence layer's own parameters (three Gaussian membership
functions, centers 0.35/0.65/0.88) are likewise fixed by hand rather than
learned by gradient descent — see `fuzzy_fusion.py`'s `ANFISAgreement` and
the manuscript's Section 3 for why, and what happens if you swap it for a
simpler linear mapping instead (Section 6.4).

If you change `common_dim`, `structural_dim`, or the feature scale
substantially, re-check `m`: inspect `U.max(axis=1)` after fitting — pinned
near `1/k` for nearly every node means `m` is too high for the current
feature scale.

## Reproducing the manuscript's results

Every table and figure in the manuscript traces to a specific script and a
specific file under `experiments/`; the most load-bearing ones:

| Result | Script | Output |
|---|---|---|
| Table 2 (overall accuracy) | `run_baselines.py` | `experiments/baselines_summary.log` |
| Robust-preset numbers | `run_robust_refresh.py` | `experiments/robust_refresh_summary.log` |
| CCA rank-cap sweep | `run_rank.py` | `experiments/rank_summary.log` |
| Missing-modality robustness | `run_missing.py` | `experiments/missing_summary.log` |
| Structural-noise / collapse | `run_collapse_p08_diagnosis.py` | `experiments/collapsep08_summary.log` |
| Scale (up to 20,000 nodes) | `run_scale.py` | `experiments/scale_summary.log` |
| Statistical validation (n=20 seeds) | `run_sigtest.py` | `experiments/sigtest_summary.log` |
| Overlap-threshold sensitivity | `run_threshold_sweep.py` | `experiments/threshold_sweep_summary.log` |
| Explainability (alpha-based and neighborhood-based) | `run_explain.py`, `run_nbr_explain.py` | `experiments/explain_summary.log`, `experiments/nbr_explain_summary.log` |

All are resumable and crash-safe (results appended to CSV, fsynced per job).

## Citation

If you use this code, please cite the accompanying manuscript (full
reference list, including this software, is in the manuscript itself) and
this repository's archived release, DOI: 10.5281/zenodo.22959565.
