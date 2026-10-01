# CESNA baseline — findings

Added after reviewer feedback that CESNA (Yang, McAuley & Leskovec, ICDM 2013) is "a
natural baseline" for this paper's setting — overlapping community detection with
node attributes — and was not cited or compared against.

## Method

`run_cesna_baseline.py` implements CESNA from scratch (no existing implementation was
available in `cdlib` or elsewhere in this project's dependencies): the BigCLAM edge
model (P(edge u,v) = 1 - exp(-F_u . F_v)) jointly fit with a logistic attribute model
(P(attr=1) = sigmoid(F_u . W)) via full-batch gradient ascent, 300 iterations, no
labels used. CESNA requires binary attributes, unlike this paper's continuous
multimodal embeddings; node features are PCA-reduced to 32 dimensions (8/4 for
DBLP/Amazon, capped by their structural-embedding dimensionality) and median-binarized
per dimension — a real methodological choice, reported as such. Communities are read
off the fitted F via BigCLAM's own standard threshold (delta_c = sqrt(-ln(1-eps)),
eps=1/(n*k)). Scored identically to every other Table 2 baseline.

**A bug was caught before results were trusted**: an initial implementation had `F @
W.T` where `F @ W` was needed (and the corresponding gradient term similarly
transposed). This silently ran without error on Amazon only because its k=4 happens to
equal its attribute dimensionality, making the transpose numerically valid but
semantically wrong; it crashed with a shape-mismatch error on CrisisMMD (k=7), which is
what caught it. Fixed and re-verified on both datasets before running the full suite.

## Results (mean ONMI over seeds 0,1,2)

| Dataset | CESNA | NF-MCD default | Spectral+content | Louvain |
|---|---|---|---|---|
| CrisisMMD | 0.254 | 0.609 | 0.879 | 0.556 |
| PHEME | 0.000 | 0.005 | 0.373 | 0.000 |
| Fakeddit | 0.260 | 0.308 | 0.554 | 0.665 |
| DBLP | 0.022 | 0.670 | 0.630 | 0.355 |
| Amazon | 0.325 | 0.554 | 0.619 | 0.373 |

## Interpretation

CESNA does not beat NF-MCD or spectral+content on any dataset, and is in fact one of
the weaker methods tested overall — comparable to or below DEMON/SLPA on most
datasets, notably collapsing on DBLP (0.022, well below even Louvain's 0.355). This
doesn't change the paper's central finding (spectral+content remains the strongest or
tied-strongest baseline), but it does close the specific "this natural baseline was
never tried" gap honestly: CESNA is not a stronger alternative, under the integration
choices (binarized, PCA-reduced attributes) required to run it on this paper's data at
all.
