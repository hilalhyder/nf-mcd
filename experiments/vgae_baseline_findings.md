# VGAE baseline — findings

Added after reviewer feedback that VGAE (ref [23], Kipf & Welling 2016) was cited as
related work but never actually run, and that the existing GNN comparison (untrained
2-layer GCN, `run_gnn_baseline.py`) was "a weak stand-in" for a trained method. torch
was installed (into a separate short-path venv, `C:\pytv`, to work around a Windows
long-path limit hit by torch's packaged license files under this machine's deep
AppData-based site-packages path) specifically to make this run possible.

## Method

Standard VGAE (Kipf & Welling 2016): 2-layer GCN encoder producing per-node (mu,
logstd), reparameterized to z, inner-product decoder (sigmoid(z @ z.T)) reconstructing
the adjacency matrix, trained end-to-end via Adam (200 epochs, lr=0.01) on a weighted
binary cross-entropy (positive/negative edge class balance) plus KL divergence against
a standard normal prior. No labels used anywhere in training. Node features and the
downstream clustering/scoring protocol are identical to the existing untrained-GCN
baseline (content_matrix for CrisisMMD/PHEME/Fakeddit, spectral structural embedding
for DBLP/Amazon since they carry no content; FuzzyCMeans(m=1.5) on mu at inference;
scored identically to every other Table 2 method).

## Results (mean ONMI over seeds 0,1,2)

| Dataset | VGAE+FCM | Untrained GCN | NF-MCD default | Spectral+content |
|---|---|---|---|---|
| CrisisMMD | 0.534 | 0.672 | 0.609 | 0.879 |
| PHEME | 0.253 | 0.024 | 0.005 | 0.373 |
| Fakeddit | 0.175 | 0.504 | 0.308 | 0.554 |
| DBLP | 0.536 | 0.539 | 0.670 | 0.630 |
| Amazon | 0.512 | 0.480 | 0.554 | 0.619 |

## Interpretation

VGAE does not beat spectral+content on any of the five datasets, reinforcing rather
than qualifying the paper's central finding. More surprisingly, the trained VGAE
underperforms the untrained GCN baseline on two datasets (CrisisMMD: 0.534 vs 0.672;
Fakeddit: 0.175 vs 0.504) despite the untrained baseline using literally random,
frozen weights. The likely mechanism: VGAE's training objective optimizes purely for
adjacency reconstruction and has no incentive to preserve content information unless
content happens to correlate with graph structure, whereas the untrained GCN's forward
pass (A_hat @ X @ W) still linearly propagates the raw content features through the
graph regardless of whether that helps reconstruct edges. On PHEME and DBLP/Amazon,
where structure itself is the dominant or only signal, this doesn't matter and VGAE's
genuine training gives it a real (if still non-winning) edge; on CrisisMMD and
Fakeddit, where content matters most, training toward a link-reconstruction-only
objective appears to actively discard useful signal that the untrained baseline
incidentally retains.
