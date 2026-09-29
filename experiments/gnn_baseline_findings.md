# GNN / attributed-graph baseline (reviewer request)

Reviewer's ask: *"The related-work section should explicitly engage with...
graph neural networks, graph autoencoders... At least one modern GNN-based or
attributed-community-detection baseline should be considered if
computationally feasible."*

Script: `run_gnn_baseline.py`. Results: `experiments/gnn_baseline_results.csv`
(15 rows, 0 errors). Summary: `experiments/gnn_baseline_summary.log`.

## What was actually implemented (be precise about this in the manuscript)

- **torch is not installed in this environment.** `py -c "import torch"`
  fails with `ModuleNotFoundError`, despite the manuscript's reproducibility
  section claiming "torch 2.14.0 (CPU build)". `torch_geometric` is also not
  installed. Per the task brief, no large new dependency (torch,
  torch_geometric, DGL, etc.) was `pip install`-ed to work around this.
- Instead, a **hand-rolled, pure-numpy 2-layer Graph Convolutional Network**
  (Kipf & Welling 2017 propagation rule) was implemented from scratch in
  `run_gnn_baseline.py`:
  - `A_hat = D^-1/2 (A + I) D^-1/2` (symmetric-normalized adjacency with
    self-loops), computed densely (fine at this project's dataset sizes,
    n <= 2000).
  - `H1 = ReLU(A_hat @ X @ W0)`, `H2 = A_hat @ H1 @ W1`, hidden width 64,
    output width 32.
- **The weights W0/W1 are randomly initialized (Glorot scale) and NEVER
  TRAINED.** There is no gradient descent, no autoencoder reconstruction
  loss, no backprop anywhere in this script. This is deliberately the
  "untrained GCN as a smoothing/embedding operator" baseline the task brief
  explicitly sanctions as legitimate, and which the paper should cite as
  closely related to **SGC** (Simplifying Graph Convolutional Networks, Wu
  et al., ICML 2019) — SGC's central finding is that a linear, untrained
  propagation operator over the normalized adjacency already captures most
  of what a trained GCN captures for downstream clustering/classification.
  **Do not describe this baseline in the manuscript as "trained" or as a
  "graph autoencoder" — it is neither.** If reviewers want a trained variant,
  that is future work requiring torch (or a hand-written backprop pass,
  which was judged out of scope for this revision pass given the time
  budget).
- **Node features fed into the GCN:**
  - CrisisMMD, PHEME, Fakeddit: `nf_mcd.baselines.content_matrix(d)` — the
    same *raw* (not NF-MCD's CCA-fused) concatenated, L2-normalized
    text+image content vectors the project's own `kmeans_content`/
    `fcm_content` baselines already consume. A missing modality contributes
    a zero block (all of PHEME's image block is zero — PHEME has no images).
  - DBLP, Amazon: these SNAP datasets ship **no per-node content at all**
    (`nf_mcd/datasets.py` sets `texts=images=[None]*n` by construction), so
    `content_matrix()` returns `None`. For these two only, node features are
    `nf_mcd.topology.compute_structural_embedding(G, dim=n_communities)` —
    the exact same spectral fallback the project's own `nfmcd_structure_only`
    / `fcm_structure_only` ablations already use for these datasets, so the
    GCN baseline isn't handicapped by an ad hoc feature choice unused
    elsewhere in the codebase.
- **Clustering on top of the GCN embeddings:** this project's own
  `FuzzyCMeans(m=1.5)` (the same calibrated fuzziness exponent used
  throughout the pipeline), at k = ground-truth community count — identical
  protocol to every other method in Table 2. Row-L2-normalized GCN output
  embeddings feed the clusterer. Overlapping-community view uses the same
  threshold (0.2) as every other fuzzy baseline.
- **Scoring:** identical to `run_baselines.py` — modularity of the hard
  (argmax) partition, LFK overlapping NMI via cdlib (`nf_mcd/metrics.py`),
  best-match membership F1. 3 seeds (0, 1, 2) per dataset, all 5 datasets,
  mean±sd reported. All 15 (dataset, seed) jobs completed with zero errors.

**Environment note:** this run also required installing `scikit-learn` and
`cdlib` via pip — both are already-declared *core* dependencies in
`requirements.txt` (not new/heavy ones like torch) but were missing from
this machine's environment; they were installed to match the project's own
documented requirements, not added as new dependencies for this experiment.

## Results

New method name in the CSV: `gcn_embed_fcm`.

### ONMI (LFK overlapping NMI, cdlib) — vs. `experiments/baselines_summary.log`

| method | crisismmd | pheme | fakeddit | dblp | amazon |
|---|---|---|---|---|---|
| nfmcd_full | 0.609±0.016 | 0.005±0.007 | 0.308±0.053 | 0.670±0.000 | 0.554±0.000 |
| spectral_graph+content | 0.879±0.000 | 0.373±0.001 | 0.554±0.004 | 0.630±0.019 | 0.619±0.000 |
| louvain | 0.556±0.018 | 0.000±0.000 | 0.665±0.026 | 0.355±0.005 | 0.373±0.000 |
| demon | 0.141±0.008 | 0.000±0.000 | 0.271±0.004 | 0.159±0.002 | 0.370±0.006 |
| slpa | 0.189±0.024 | 0.000±0.000 | 0.151±0.019 | 0.094±0.020 | 0.329±0.018 |
| **gcn_embed_fcm (new)** | **0.672±0.032** | **0.024±0.009** | **0.504±0.085** | **0.539±0.051** | **0.480±0.045** |

### Modularity (hard partition)

| method | crisismmd | pheme | fakeddit | dblp | amazon |
|---|---|---|---|---|---|
| nfmcd_full | 0.520±0.006 | 0.859±0.005 | 0.297±0.014 | 0.848±0.000 | 0.717±0.000 |
| spectral_graph+content | 0.517±0.000 | 0.375±0.002 | 0.365±0.002 | 0.826±0.002 | 0.714±0.000 |
| louvain | 0.536±0.001 | 0.984±0.000 | 0.450±0.001 | 0.920±0.000 | 0.757±0.000 |
| **gcn_embed_fcm (new)** | **0.444±0.030** | **0.416±0.032** | **0.380±0.025** | **0.839±0.004** | **0.703±0.020** |

### Membership F1

| method | crisismmd | pheme | fakeddit | dblp | amazon |
|---|---|---|---|---|---|
| nfmcd_full | 0.667±0.114 | 0.267±0.013 | 0.253±0.043 | 0.733±0.000 | 0.797±0.000 |
| spectral_graph+content | 0.956±0.000 | 0.540±0.001 | 0.710±0.003 | 0.736±0.010 | 0.834±0.000 |
| **gcn_embed_fcm (new)** | **0.687±0.040** | **0.127±0.022** | **0.609±0.035** | **0.679±0.064** | **0.733±0.014** |

Mean communities found (fixed k = ground truth by construction; `n_pred` in
the CSV equals `k_used` every time since FuzzyCMeans always produces exactly
k non-empty clusters here): crisismmd=7.0, pheme=9.0, fakeddit=10.0, dblp=8.0,
amazon=4.0.

## Interpretation, in plain numeric terms

Ranking `gcn_embed_fcm` against `nfmcd_full` on ONMI, per dataset:

- **crisismmd**: GCN 0.672 > NF-MCD 0.609 (GCN wins by 0.063)
- **pheme**: GCN 0.024 > NF-MCD 0.005 (both near-zero; GCN wins narrowly, well
  within noise of either)
- **fakeddit**: GCN 0.504 > NF-MCD 0.308 (GCN wins by 0.196 — the largest gap)
- **dblp**: GCN 0.539 < NF-MCD 0.670 (NF-MCD wins by 0.131)
- **amazon**: GCN 0.480 < NF-MCD 0.554 (NF-MCD wins by 0.074)

So the untrained-GCN-embedding baseline **beats full NF-MCD on ONMI on the
three content-bearing datasets (crisismmd, pheme, fakeddit) and loses on the
two pure-topology SNAP datasets (dblp, amazon)** — consistent with the rest
of Table 2's pattern that NF-MCD's advantage is strongest where structure
alone is highly informative (dblp/amazon modularity of the hard partition is
already 0.82-0.85 for nearly every method in the table, including this GCN
baseline at 0.839/0.703).

Against the *strongest* baseline per dataset already in Table 2
(`spectral_graph+content` wins crisismmd/pheme/amazon, `nfmcd_structure_only`
wins fakeddit/dblp at 0.823/0.670 respectively — see
`baselines_summary.log`), `gcn_embed_fcm` is never the best method on any of
the five datasets and trails the column-best ONMI by 0.021 (fakeddit, vs.
0.823) up to 0.207 (crisismmd, vs. 0.879). It does, however, consistently
beat `louvain`, `demon`, and `slpa` on ONMI except on fakeddit, where louvain
(0.665) beats it (0.504).

**Seed variance is non-trivial for this baseline** (sd 0.032-0.085 on ONMI,
vs. sd typically <0.05 for the other seeded methods in the table) — expected,
since the GCN's own random weight initialization is an additional source of
stochasticity on top of FuzzyCMeans's k-means++ seeding that no other method
in Table 2 has.

## Honest caveats for the manuscript

1. This is a **lightweight, untrained** GNN baseline (SGC-style propagation
   operator), not a trained GCN and not a graph autoencoder. It should be
   labeled as such wherever it's cited — e.g. "an untrained 2-layer GCN
   embedding (SGC-style, Wu et al. 2019) followed by fuzzy c-means", not
   "a GNN baseline" unqualified.
2. No hyperparameter search was performed (hidden=64, out=32 chosen once,
   not tuned per dataset); a trained GCN or a tuned architecture could score
   materially differently in either direction.
3. `torch`/`torch_geometric` being absent from this environment despite the
   manuscript's claimed reproducibility environment is worth flagging to
   whoever maintains that section — either the environment drifted, or the
   claim needs correcting.
4. content_matrix's raw content features (not NF-MCD's CCA-aligned fusion)
   were deliberately used as GCN input so the comparison is to an
   *independent* attributed-graph pipeline, not one that reuses NF-MCD's own
   cross-modal alignment machinery.

## Re-running

```
cd nfmcd_impl
py run_gnn_baseline.py run       # resumable; 15 jobs, ~13s total on this machine
py run_gnn_baseline.py summary   # rebuilds experiments/gnn_baseline_summary.log
```
