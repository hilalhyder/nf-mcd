"""
NF-MCD: Neuro-Fuzzy Multimodal Community Detection
====================================================

Reference implementation accompanying the paper:

    "Neuro-Fuzzy Multimodal Community Detection: Fusing Visual-Textual
    Semantics with Network Topology for Explainable Social Network Analysis"

This package implements the four stages described in Section 4 of the paper:

    1. Multimodal node representation      -> nf_mcd.encoders
    2. Neuro-fuzzy fusion layer             -> nf_mcd.fuzzy_fusion
    3. Fuzzy integration with topology      -> nf_mcd.topology
    4. Soft/overlapping community detection -> nf_mcd.community_detection
    5. Explainability (fuzzy rule layer)    -> nf_mcd.explain

`nf_mcd.pipeline.NFMCD` wires all five stages together into a single
`fit` / `predict` interface.

This is a research prototype, not a production system. In particular:
  - The image/text encoders fall back to deterministic pseudo-random
    embeddings when `sentence-transformers` / `transformers` (and network
    access to download pretrained weights) are unavailable. Swap in real
    encoders by passing your own callables (see encoders.py).
  - The overlapping-NMI implementation in metrics.py is a simplified
    approximation, documented as such; for publication-grade evaluation,
    cross-check against a reference implementation (e.g. Lancichinetti
    et al.'s ONMI code).
"""

from .pipeline import NFMCD

__all__ = ["NFMCD"]
__version__ = "0.1.0"
