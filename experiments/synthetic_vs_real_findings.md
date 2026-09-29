# Synthetic-vs-real structural comparison (reviewer request)

Reviewer's ask: *"The scale experiments use the synthetic generator. This
demonstrates computational scalability, but not necessarily real-world
scalability... compare synthetic and real graphs using: degree distribution,
clustering coefficient, community-size distribution, modularity, overlap
distribution, content-similarity distribution. Otherwise the 20,000-node
result should be described specifically as a computational stress test
rather than as evidence of real social-network scalability."*

Script: `run_synth_vs_real.py`. Results: `experiments/synth_vs_real_results.csv`
(7 rows, 0 errors). Summary: `experiments/synth_vs_real_summary.log`.

## What was compared

Five real datasets, using the exact cached objects `run_baselines.py` scores
(`experiments/cache/{crisismmd,pheme,fakeddit,dblp,amazon}.pkl`), against two
synthetic configurations:

- **`synthetic_typical`**: `generate_synthetic_multimodal_graph`'s own
  defaults — n_nodes=120, n_communities=4, p_in=0.18, p_out=0.02,
  missing_modality_rate=0.15, misalignment_rate=0.15, overlap_rate=0.10 —
  i.e. the exact configuration `demo.py`, the README's usage example, and
  `validate_onmi.py` all use. This is "the" synthetic setting most readers
  of this codebase would encounter first.
- **`synthetic_stress20k`**: the *literal* configuration `run_scale.py`'s
  Part A used to produce its headline 20,000-node scaling result —
  n_communities=8, p_in/p_out from `run_scale.p_in_out_for(20000)` (target
  avg degree ~18, same p_out/p_in ratio as the generator's default),
  missing_modality_rate=0, misalignment_rate=0, overlap_rate=0 (`run_scale.py`
  deliberately zeroes these for a "clean" scaling measurement). `run_scale.py`
  was only imported from (`p_in_out_for`, `N_COMMUNITIES`), never modified.

Six statistics, computed identically across all seven graphs (methodology
detail: content-similarity is cosine similarity of
`nf_mcd.baselines.content_matrix()` vectors — the same raw concatenated,
L2-normalized text+image content the project's own `kmeans_content` baseline
uses — on a random sample of up to 1500 nodes per dataset, split into
within-primary-community vs. between-primary-community pairs; chosen because
it is directly comparable to how content actually feeds the pipeline, cheap
to compute, and needs no extra dependencies).

## Results table

| Statistic | crisismmd | pheme | fakeddit | dblp | amazon | synthetic_typical | synthetic_stress20k |
|---|---|---|---|---|---|---|---|
| n_nodes | 800 | 2000 | 1200 | 1072 | 146 | 120 | 20000 |
| n_edges | 4889 | 1841 | 12529 | 2766 | 390 | 416 | 179460 |
| degree mean | 12.22 | 1.84 | 20.88 | 5.16 | 5.34 | 6.93 | 17.95 |
| degree median | 12.0 | 1.0 | 17.0 | 4.0 | 5.0 | 7.0 | 18.0 |
| degree max | 26 | 28 | 58 | 38 | 15 | 14 | 37 |
| degree p99 | 23.0 | 15.0 | 47.0 | 22.0 | 12.0 | 12.81 | 28.0 |
| degree tail ratio (max/p99) | 1.13 | 1.87 | 1.23 | 1.73 | 1.25 | 1.09 | 1.32 |
| avg clustering coeff. | **0.037** | **0.000** | **0.049** | **0.687** | **0.386** | 0.115 | **0.002** |
| n ground-truth communities | 7 | 9 | 10 | 8 | 4 | 4 | 8 |
| community size mean | 114.3 | 222.2 | 120.0 | 135.4 | 53.5 | 32.5 | 2500.0 |
| community size median | 87 | 221 | 100 | 140 | 48 | 32 | 2500 |
| community size min/max | 25 / 211 | 21 / 462 | 35 / 401 | 34 / 229 | 22 / 96 | 31 / 34 | 2500 / 2500 |
| modularity of ground truth | 0.534 | 0.830 | 0.450 | 0.821 | 0.712 | 0.418 | 0.434 |
| overlap fraction (>1 community) | 0.000 | 0.000 | 0.000 | **0.010** | **0.466** | 0.083 | 0.000 |
| content: within-comm cosine mean | 0.391 | 0.277 | 0.287 | n/a | n/a | 0.596 | 0.900 |
| content: between-comm cosine mean | 0.322 | 0.219 | 0.231 | n/a | n/a | 0.047 | -0.002 |
| content: within-between delta | **0.069** | **0.058** | **0.055** | n/a | n/a | **0.548** | **0.901** |

(DBLP/Amazon: `content_matrix()` returns `None` — these SNAP datasets ship no
per-node text/image content at all, by construction; see
`nf_mcd/datasets.py`'s `load_snap_community` docstring. Marked "n/a", not 0.)

## Where the synthetic generator is close to real data, and where it isn't

**Degree distribution**: the synthetic_stress20k graph (mean degree 17.95,
median 18) sits inside the real range (1.84-20.88 mean degree across the five
real datasets) essentially by construction — `run_scale.py` explicitly
targets ~18 avg degree to match this project's own SBM default ratio.
synthetic_typical (mean 6.93) is on the low end but still within the observed
real range. Tail ratio (max/p99) for both synthetic configs (1.09-1.32) is
comparable to or *thinner*-tailed than every real dataset (1.13-1.87) — real
social graphs here have a heavier degree tail than either synthetic setting,
most visibly PHEME (1.87) and DBLP (1.73).

**Clustering coefficient — the largest, most unambiguous gap.**
Real datasets range from 0.000 (PHEME, a forest of reply trees — no
triangles by construction) to 0.687 (DBLP, a collaboration network with
strong real triadic closure) and 0.386 (Amazon). The synthetic_stress20k
graph's clustering coefficient is **0.002** — two to three orders of
magnitude below DBLP and Amazon, and below every real dataset except PHEME.
This is expected and mechanical: an SBM with p_in ≈ 0.004 at n=20,000
produces almost no triangles by construction (SBM triangle probability
scales with p_in², which is tiny at this size/density), whereas real
collaboration/co-purchase graphs have genuine local clustering from
transitive social/product relationships that an SBM does not model. Even
synthetic_typical (0.115, smaller/denser p_in=0.18) undershoots DBLP and
Amazon and only lands near CrisisMMD/Fakeddit's (themselves synthetically
reconstructed, SBM-sampled) topology.

**Community-size distribution**: synthetic_stress20k's 8 communities are
*exactly* 2500 nodes each (min=max=2500) — a perfectly uniform partition by
construction (`n_nodes // n_communities` with remainder absorbed into the
last block, at n divisible by n_communities here). Every real dataset has a
visibly skewed size distribution (e.g. Fakeddit 35-401, PHEME 21-462, a
~20x range). This uniformity is itself a structural mismatch: real
communities are not equal-sized, and the stress-test graph does not exercise
NF-MCD (or any baseline) against that imbalance at 20k-node scale.

**Modularity of the ground-truth partition**: synthetic_stress20k (0.434) and
synthetic_typical (0.418) both sit at the low end of the real range
(0.450-0.830), closest to Fakeddit (0.450) and below every other real
dataset. This is consistent with the clustering-coefficient finding — an SBM
partition with the thin p_in used to hit avg-degree-18 at 20,000 nodes has
less internal cohesion, relative to its cut, than most of the real ground
truths.

**Overlap distribution**: synthetic_stress20k has **zero** overlap by
construction (`overlap_rate=0.0`, deliberately set by `run_scale.py` for a
clean timing measurement) — it cannot speak to overlapping-community
behavior at all. synthetic_typical has 8.3% overlap (its own default
`overlap_rate=0.10`, realized rate 0.083). Real overlap in this project's
datasets is concentrated entirely in the two SNAP graphs: **Amazon is 46.6%
overlapping** (co-purchase products routinely belong to several product
categories/communities) and DBLP is a modest 1.0%; CrisisMMD, PHEME, and
Fakeddit have **exactly 0% overlap** because their "ground truth" (disaster
event / news event / subreddit id) is a single categorical label per node by
construction, not a multi-membership structure — this should be stated
plainly in the manuscript rather than implied. Neither synthetic
configuration comes close to reproducing Amazon's real overlap magnitude.

**Content-similarity distribution**: this is the second unambiguous gap,
and it runs the *opposite* direction from clustering coefficient — the
synthetic generator's content signal is far *more* separable than any real
dataset's. synthetic_stress20k's within-community cosine similarity (0.900)
vs. between-community (-0.002) gives a within/between delta of **0.901**;
synthetic_typical's delta is 0.548. The three real content-bearing datasets
show deltas of only **0.055-0.069** — content embeddings are only weakly
more similar within a "community" (disaster event / news event / subreddit)
than across one. This makes sense given how those "communities" were
defined (event/subreddit id, not a semantic-content cluster) and given that
CrisisMMD/Fakeddit/PHEME use real, noisy encoder embeddings while the
synthetic generator draws directly from well-separated Gaussian centroids
with per-node noise — the generator's content signal-to-noise ratio is
roughly an order of magnitude higher than what NF-MCD actually faces on real
text/images.

## Bottom line for the manuscript

On two of the six dimensions the reviewer asked for — **clustering
coefficient** and **content-similarity separability** — the synthetic
generator (at both the "typical" and the literal 20,000-node stress-test
configuration) is **structurally very different from the real datasets**:
near-zero triangle density (0.002 vs. up to 0.687 real) and far more
separable content signal (0.90 within/between delta vs. 0.055-0.069 real).
Community-size uniformity (all communities exactly 2500 nodes at 20k) and
zero overlap (by explicit choice in `run_scale.py`) are also not
representative of any real dataset studied here, most starkly Amazon's
46.6% overlap rate. Degree distribution and modularity are the two
dimensions where the stress-test configuration is reasonably close to real
data (degree mean/median match by explicit design; modularity is at the low
end of, but within, the real range).

**Recommendation**: per the reviewer's own framing, the 20,000-node result
should be described in the manuscript specifically as **a computational
stress test of the pipeline's runtime/memory scaling**, not as evidence that
NF-MCD scales to real 20,000-node social networks with realistic clustering,
content-separability, community-size skew, or overlap structure — on 4 of
the 6 structural dimensions examined here, the stress-test graph is
measurably closer to an idealized SBM than to any of this project's own five
real datasets.

## Honest caveats

- Modularity of ground truth uses each node's **primary/first-listed**
  community as a hard label when ground truth allows overlap (DBLP, Amazon,
  synthetic with overlap_rate>0) — a small fraction of nodes' secondary
  memberships are therefore not reflected in this particular statistic
  (overlap is captured separately, dimension 5).
- Content-similarity is computed on a bounded random sample (cap 1500 nodes)
  for memory/runtime reasons, not full pairwise, for every dataset above
  that cap (PHEME, synthetic_stress20k); CrisisMMD/Fakeddit/DBLP/Amazon/
  synthetic_typical are under the cap and use full pairwise.
  This is a sampling estimate, not an exhaustive computation, though at
  n=1500 samples (>1M pairs) the standard error on the reported means is
  small relative to the effect sizes discussed above.
- CrisisMMD/Fakeddit/PHEME's graphs are themselves SBM-reconstructed (no
  native social graph in the source releases — see `nf_mcd/datasets.py`
  docstrings), which is *also* a synthetic-topology choice, just fit to
  real per-node content and real event/subreddit group sizes; only PHEME's
  edges (reply trees) and DBLP/Amazon's edges (SNAP's real graphs) are
  genuinely observed real-world topology. This nuance matters if the
  manuscript wants to draw a hard synthetic/real line for topology alone.

## Re-running

```
cd nfmcd_impl
py run_synth_vs_real.py run       # resumable; 7 rows, ~2-3s total on this machine
py run_synth_vs_real.py summary   # rebuilds experiments/synth_vs_real_summary.log
```
