"""
Cross-check nf_mcd.metrics.overlapping_nmi (a documented Hungarian-matched
binary-NMI approximation - see its module docstring and README.md "Next
steps" item 6) against a genuine, independently-implemented reference
overlapping-NMI metric, across every dataset this project has fitted so far:
the synthetic demo.py scenario plus all five real datasets (CrisisMMD,
PHEME, Fakeddit, SNAP com-DBLP, SNAP com-Amazon).

Reference implementation: cdlib's `evaluation.overlapping_normalized_mutual_
information_LFK` and `_MGH`, which wrap `cdlib.evaluation.internal.onmi` - a
ported implementation of the actual Lancichinetti-Fortunato-Kertesz (2009)
conditional-entropy-based overlapping NMI (LFK), plus the McDaid-Greene-
Hurley (2011) variant (MGH) with a different normalization. This is a
genuinely different algorithm from this project's own approximation (which
does Hungarian-matched pairwise binary-NMI); it is NOT another wrapper
around the same idea - confirmed by reading cdlib's internal onmi.py source,
which implements the conditional-entropy formulation from the LFK paper
directly (comPairConditionalEntropy / coverConditionalEntropy), not a
Hungarian-assignment shortcut.

Run with:  py validate_onmi.py
Full output is duplicated to experiments/onmi_validation.log.
"""

from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np

from cdlib import NodeClustering, evaluation as cdlib_eval

from nf_mcd import NFMCD
from nf_mcd.encoders import MultimodalEncoder
from nf_mcd import datasets as ds_mod
from nf_mcd import metrics as mx

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "urllib3", "filelock"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class _Tee:
    def __init__(self, stream, file_path):
        self.stream = stream
        self.file = open(file_path, "w", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def build_views(model: NFMCD, true_communities_per_node, overlap_threshold: float = 0.2):
    """Exactly reproduces the pred_view/true_view construction inside
    NFMCD.evaluate() (nf_mcd/pipeline.py), so we feed the reference metric
    the identical inputs our own overlapping_nmi() was scored on."""
    true_view = [set() for _ in range(model.n_communities)]
    for node_idx, comms in enumerate(true_communities_per_node):
        for c in comms:
            if 0 <= c < model.n_communities:
                true_view[c].add(node_idx)
    pred_view = model.overlapping_communities_view(threshold=overlap_threshold)
    return pred_view, true_view


def compare(name: str, G, pred_view, true_view, n_nodes: int) -> dict:
    own = mx.overlapping_nmi_approx(pred_view, true_view, n_nodes=n_nodes)

    pred_nonempty = [sorted(c) for c in pred_view if c]
    true_nonempty = [sorted(c) for c in true_view if c]
    n_pred_empty = sum(1 for c in pred_view if not c)
    n_true_empty = sum(1 for c in true_view if not c)

    pred_nc = NodeClustering(pred_nonempty, graph=G, method_name="NFMCD")
    true_nc = NodeClustering(true_nonempty, graph=G, method_name="ground_truth")

    lfk = cdlib_eval.overlapping_normalized_mutual_information_LFK(pred_nc, true_nc).score
    mgh = cdlib_eval.overlapping_normalized_mutual_information_MGH(pred_nc, true_nc).score

    row = {
        "dataset": name,
        "own_onmi": own,
        "lfk_onmi": lfk,
        "mgh_onmi": mgh,
        "diff_lfk": own - lfk,
        "diff_mgh": own - mgh,
        "n_pred_comms": len(pred_view),
        "n_true_comms": len(true_view),
        "n_pred_empty": n_pred_empty,
        "n_true_empty": n_true_empty,
    }
    print(
        f"\n[{name}] own={own:.4f}  LFK={lfk:.4f} (diff {row['diff_lfk']:+.4f})  "
        f"MGH={mgh:.4f} (diff {row['diff_mgh']:+.4f})  "
        f"pred_comms={len(pred_view)} (empty={n_pred_empty})  "
        f"true_comms={len(true_view)} (empty={n_true_empty})"
    )
    return row


def run_synthetic() -> dict:
    print("=" * 78)
    print("Scenario: synthetic demo.py setup")
    print("=" * 78)
    data = ds_mod.generate_synthetic_multimodal_graph(n_nodes=120, n_communities=4, seed=42)
    model = NFMCD(n_communities=4, seed=0)
    model.fit(data.G, text_embeddings=data.text_embeddings, image_embeddings=data.image_embeddings)
    mod = model.evaluate()["modularity"]
    print(f"Refit modularity={mod:.4f} (expected ~0.47 from README)")
    pred_view, true_view = build_views(model, data.true_communities)
    return compare("demo_synthetic", data.G, pred_view, true_view, n_nodes=len(model.nodes_))


def run_real(name: str, loader, k_candidates=None) -> dict:
    print("\n" + "=" * 78)
    print(f"Scenario: {name}")
    print("=" * 78)
    t0 = time.monotonic()
    data = loader()
    print(f"Loaded {name} in {time.monotonic() - t0:.1f}s "
          f"({data.G.number_of_nodes()} nodes, {data.G.number_of_edges()} edges)")

    encoder = MultimodalEncoder()
    e_t, e_v = encoder.encode(data.texts, data.images)

    if k_candidates is None:
        base_k = max(2, data.n_communities)
        k_candidates = sorted(set(k for k in (base_k - 1, base_k, base_k + 1) if k >= 2))

    best = None
    for k in k_candidates:
        model = NFMCD(n_communities=k, seed=0)
        model.fit(data.G, text_embeddings=e_t, image_embeddings=e_v)
        mod = model.evaluate()["modularity"]
        print(f"  k={k}: modularity={mod:.4f}")
        if best is None or mod > best[1]:
            best = (k, mod, model)
    k, mod, model = best
    print(f"Selected k={k} (modularity={mod:.4f})")

    pred_view, true_view = build_views(model, data.true_communities)
    return compare(name, data.G, pred_view, true_view, n_nodes=len(model.nodes_))


def main():
    rows = []
    rows.append(run_synthetic())
    rows.append(run_real("crisismmd", lambda: ds_mod.load_crisismmd(max_nodes=800)))
    rows.append(run_real("pheme", lambda: ds_mod.load_pheme(max_nodes=2000)))
    rows.append(run_real("fakeddit", lambda: ds_mod.load_fakeddit(max_nodes=1200)))
    rows.append(run_real("dblp", lambda: ds_mod.load_snap_community("dblp", n_communities=8)))
    rows.append(run_real("amazon", lambda: ds_mod.load_snap_community("amazon", n_communities=8, min_community_size=15)))

    print("\n" + "=" * 100)
    print("COMPARISON: own overlapping_nmi() vs. reference implementations (cdlib)")
    print("=" * 100)
    header = f"{'dataset':<14}{'own':>8}{'LFK':>8}{'diff_LFK':>10}{'MGH':>8}{'diff_MGH':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['dataset']:<14}{r['own_onmi']:>8.4f}{r['lfk_onmi']:>8.4f}{r['diff_lfk']:>+10.4f}"
            f"{r['mgh_onmi']:>8.4f}{r['diff_mgh']:>+10.4f}"
        )

    diffs_lfk = np.array([r["diff_lfk"] for r in rows])
    diffs_mgh = np.array([r["diff_mgh"] for r in rows])
    print(f"\nMean signed diff vs LFK: {diffs_lfk.mean():+.4f}   Mean abs diff vs LFK: {np.abs(diffs_lfk).mean():.4f}")
    print(f"Mean signed diff vs MGH: {diffs_mgh.mean():+.4f}   Mean abs diff vs MGH: {np.abs(diffs_mgh).mean():.4f}")
    print(f"Max abs diff vs LFK: {np.abs(diffs_lfk).max():.4f} ({rows[int(np.argmax(np.abs(diffs_lfk)))]['dataset']})")
    print(f"Max abs diff vs MGH: {np.abs(diffs_mgh).max():.4f} ({rows[int(np.argmax(np.abs(diffs_mgh)))]['dataset']})")


if __name__ == "__main__":
    log_path = os.path.join(EXPERIMENTS_DIR, "onmi_validation.log")
    orig_stdout = sys.stdout
    tee = _Tee(orig_stdout, log_path)
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = orig_stdout
        tee.file.close()
    print(f"Full log saved to {log_path}")
