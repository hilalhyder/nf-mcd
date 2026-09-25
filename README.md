# NF-MCD: Neuro-Fuzzy Multimodal Community Detection

Reference implementation accompanying the draft paper *"Neuro-Fuzzy
Multimodal Community Detection: Fusing Visual-Textual Semantics with
Network Topology for Explainable Social Network Analysis."*

## What this is (and isn't)

This is a **working, tested prototype** of the four-stage NF-MCD architecture
described in Section 4 of the paper, runnable end-to-end on synthetic data.
It is **not** yet plugged into real datasets or real pretrained encoders —
see "Next steps" below for exactly what's left to do before this produces
publishable results.

## Package layout

Each module corresponds directly to a subsection of Section 4, so you can
cite specific files/functions in your Methods section:

| Module | Paper section | What it does |
|---|---|---|
| `nf_mcd/encoders.py` | 4.1 | Text + image (CLIP) encoders, with a deterministic fallback when pretrained models aren't available |
| `nf_mcd/fuzzy_fusion.py` | 4.2 | ANFIS-style neuro-fuzzy layer: cross-modal agreement score, fuzzy confidence, fused content embedding |
| `nf_mcd/topology.py` | 4.3 | Spectral structural embedding + fuzzy content/structure integration (per-node alpha) |
| `nf_mcd/community_detection.py` | 4.4 | Fuzzy c-means soft/overlapping community detection |
| `nf_mcd/explain.py` | 4.5 / 5.4 | Per-node explanations + global IF-THEN fuzzy rule extraction, rule fidelity |
| `nf_mcd/metrics.py` | 5.3 | Modularity, overlapping NMI (approximate), membership F1 |
| `nf_mcd/datasets.py` | 5.1 | Synthetic multimodal graph generator + stubs for MMCas/CrisisMMD/Fakeddit/PHEME loaders |
| `nf_mcd/pipeline.py` | 4 (all) | `NFMCD` class: the scikit-learn-style `fit`/`predict` interface tying everything together |

## Quick start

```bash
pip install -r requirements.txt
python demo.py
```

`demo.py` generates a synthetic 120-node graph with 4 (mildly overlapping)
communities, injects missing modalities and cross-modal misalignment,
fits NF-MCD, evaluates it against the known ground truth, prints a couple
of per-node explanations and the extracted global fuzzy rules, and saves a
plot of the graph colored by predicted community to `demo_output.png`.

## Minimal usage example

```python
from nf_mcd import NFMCD
from nf_mcd.datasets import generate_synthetic_multimodal_graph, node_sets_to_community_view

data = generate_synthetic_multimodal_graph(n_nodes=120, n_communities=4, seed=0)

model = NFMCD(n_communities=4, seed=0)
model.fit(
    data.G,
    text_embeddings=data.text_embeddings,
    image_embeddings=data.image_embeddings,
)

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

This routes through `nf_mcd.encoders.MultimodalEncoder`, which uses
`sentence-transformers` + CLIP if installed and reachable, or otherwise
falls back to deterministic pseudo-embeddings (clearly logged via a
`UserWarning` — do not use the fallback for real experiments).

## Validation performed while building this

Three non-obvious bugs surfaced during development and were fixed and
empirically verified before this code was delivered — worth knowing about
since they'll matter again if you change the defaults substantially:

1. **Cross-modal projection.** An early version projected text/image
   embeddings into a shared space with independent fixed random matrices.
   This is mathematically incapable of recovering cross-modal alignment —
   verified directly: even a perfectly shared underlying signal produced
   ~0 cosine similarity after two unrelated random projections. Fixed by
   fitting a PCA-whitened CCA on the paired (both-modalities-present)
   nodes instead (`nf_mcd/fuzzy_fusion.py`), which correctly separates
   aligned from misaligned pairs (validated: ~0.7–0.9 vs ~0.5–0.6 cosine
   agreement on synthetic data with known ground truth).
2. **Structural embedding dimensionality.** Requesting far more spectral
   eigenvectors than the true number of communities (e.g. 32 for a
   4-community graph) dilutes the real signal across mostly-noise
   dimensions once row-normalized — standalone k-means accuracy dropped
   from 92% (at dim≈4–8) to 40% (chance level, at dim=32) on identical
   data. Fixed by defaulting `structural_dim` to `n_communities`
   (`nf_mcd/pipeline.py`, `nf_mcd/topology.py`).
3. **FCM fuzziness exponent.** The generic-FCM textbook default `m=2.0`
   produced mathematically-valid but practically useless memberships
   pinned at ~1/k for every node (correct *ranking*, zero usable
   confidence signal) on this pipeline's unit-norm-ish feature scale.
   `m=1.5` was found to preserve both a correct hard partition and
   genuinely graded soft memberships (`nf_mcd/community_detection.py`).

With all three fixes, `demo.py` recovers the synthetic ground truth at
modularity ≈0.47, overlapping NMI ≈0.69, and membership F1 ≈0.91 — treat
these as a sanity-check baseline, not a claim about real-dataset
performance, which still needs to be established per "Next steps" below.

## Next steps to get from this prototype to paper results

1. **Real encoders.** Install `sentence-transformers`, `transformers`, and
   `torch` (already in `requirements.txt`), and make sure the machine you
   run this on has internet access to download pretrained weights
   (`all-MiniLM-L6-v2`, `openai/clip-vit-base-patch32` by default —
   configurable via `TextEncoder(model_name=...)` / `ImageEncoder(model_name=...)`).
2. **Real datasets.** Fill in the loader stubs in `nf_mcd/datasets.py`
   (`load_mmcas_twitter`, `load_crisismmd`, `load_fakeddit`, `load_pheme`)
   once you've downloaded each dataset — see the docstrings for access
   notes and expected return shapes.
3. **Baselines.** Section 5.2's ablation grid needs: (i) a text-only fuzzy
   community detector (set `image_embeddings=None` for every node — NF-MCD
   already degrades gracefully to this case), (ii) a non-fuzzy multimodal
   baseline (e.g. CLIP embeddings + Louvain/spectral clustering — not
   included here, since it's a different method, not a variant of NF-MCD),
   and (iii) FCGL or another fuzzy topology-only baseline.
4. **Hyperparameter tuning.** `alpha_min`/`alpha_max`, the ANFIS
   centers/widths, `common_dim`, `structural_dim`, and the FCM fuzziness
   `m` are all currently fixed/defaulted; tune these (or make the ANFIS
   parameters trainable via gradient descent against a validation
   objective) once real data is available.
5. **k selection.** `NFMCD(n_communities=k)` currently requires k up
   front. Use `nf_mcd.community_detection.select_k_by_fpc` as a quick
   unsupervised starting point, or a modularity scan, before committing to
   a final k for each dataset.
6. **Overlapping-NMI validation.** `nf_mcd.metrics.overlapping_nmi` is a
   documented approximation — cross-check final reported numbers against
   a reference ONMI implementation before submission.

## A note on the demo's synthetic data

`generate_synthetic_multimodal_graph` synthesizes embeddings directly
(rather than routing realistic-looking text/images through the actual
encoders), so the demo can meaningfully exercise the fusion → topology →
clustering → explanation pipeline without needing pretrained model
downloads. This is a testing/demonstration convenience only — real
experiments should use `nf_mcd.encoders` on genuine text/image content, or
precomputed embeddings from a real dataset.
