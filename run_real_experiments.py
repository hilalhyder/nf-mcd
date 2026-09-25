"""
Real-data, real-encoder NF-MCD experiments (README "Next steps" items 1-2).

Unlike demo.py (synthetic data, synthetic embeddings), this script routes
actual dataset text/images through nf_mcd.encoders.MultimodalEncoder (real
sentence-transformers + CLIP), and fits/evaluates NF-MCD on each real
dataset that nf_mcd.datasets can currently load: CrisisMMD, PHEME, and a
bounded multimodal-only subsample of Fakeddit. MMCas Twitter is skipped -
see nf_mcd.datasets.load_mmcas_twitter's docstring for why (its Google
Drive release exists but is currently permission-restricted).

No baselines are implemented here - this is NF-MCD only, per project scope
for this round. See README.md "Next steps" item 3 for baselines.

Run with:  py run_real_experiments.py [crisismmd pheme fakeddit dblp amazon]
With no arguments all five run; with names, only those run (and only their
logs/plots are overwritten).
Each dataset's console output is duplicated to experiments/<name>_run.log,
and a community-assignment plot is saved to experiments/<name>_output.png.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from nf_mcd import NFMCD
from nf_mcd.encoders import MultimodalEncoder
from nf_mcd import datasets as ds_mod

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class _Tee:
    """Duplicates writes to both the original stream and a log file."""

    def __init__(self, stream, file_path):
        self.stream = stream
        self.file = open(file_path, "w", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def run_dataset(name: str, data: "ds_mod.RealMultimodalData", k_candidates=None):
    print("=" * 78)
    print(f"NF-MCD real-data experiment: {name}")
    print("=" * 78)
    n_nodes = data.G.number_of_nodes()
    n_edges = data.G.number_of_edges()
    n_with_text = sum(1 for t in data.texts if t is not None)
    n_with_image = sum(1 for v in data.images if v is not None)
    print(f"\nGraph: {n_nodes} nodes, {n_edges} edges, "
          f"{nx.number_connected_components(data.G)} connected components")
    print(f"Communities (proxy groups: {data.dataset_name}): {data.n_communities} -> {data.community_names}")
    print(f"Nodes with text: {n_with_text}/{n_nodes}  |  Nodes with image: {n_with_image}/{n_nodes}")

    if n_nodes < 20:
        print("\nToo few nodes to run a meaningful experiment - skipping fit.")
        return

    print("\nEncoding real text/image content via nf_mcd.encoders.MultimodalEncoder "
          "(sentence-transformers + CLIP)...")
    t0 = time.monotonic()
    encoder = MultimodalEncoder()
    e_t, e_v = encoder.encode(data.texts, data.images)
    print(f"Encoding done in {time.monotonic() - t0:.1f}s "
          f"(text backend={encoder.text_encoder._backend}, image backend={encoder.image_encoder._backend})")

    if k_candidates is None:
        base_k = max(2, data.n_communities)
        k_candidates = sorted(set(k for k in (base_k - 1, base_k, base_k + 1) if k >= 2))

    print(f"\nScanning k in {k_candidates} (selecting by modularity of the hard partition)...")
    best = None
    for k in k_candidates:
        try:
            model = NFMCD(n_communities=k, seed=0)
            model.fit(data.G, text_embeddings=e_t, image_embeddings=e_v)
            mod = model.evaluate()["modularity"]
        except Exception as exc:  # noqa: BLE001
            print(f"  k={k}: FAILED ({exc})")
            continue
        print(f"  k={k}: modularity={mod:.4f}, "
              f"fcm_iters={model.fcm_result_.n_iter}, "
              f"final_obj={model.fcm_result_.objective_history[-1]:.3f}")
        if best is None or mod > best[1]:
            best = (k, mod, model)

    if best is None:
        print("\nAll k values failed to fit - aborting this dataset.")
        return
    k, mod, model = best
    print(f"\nSelected k={k} (modularity={mod:.4f})")

    true_view = None
    if data.true_communities is not None:
        eval_scores = model.evaluate(true_communities_per_node=data.true_communities)
    else:
        eval_scores = model.evaluate()
    print("\nEvaluation:")
    for key, val in eval_scores.items():
        print(f"  {key:>18s}: {val:.4f}")

    print("\n" + "-" * 78)
    print("Example per-node explanations")
    print("-" * 78)
    flags = model.modality_flags_
    nodes_list = list(data.G.nodes())
    shown = set()
    for flag_wanted in ("both", "both_unaligned", "text_only", "image_only", "none"):
        for i, f in enumerate(flags):
            if f == flag_wanted and flag_wanted not in shown:
                node_id = nodes_list[i]
                expl = model.explain_node(node_id=node_id)
                print(f"\n[{flag_wanted}]")
                print(" ", expl.text)
                shown.add(flag_wanted)
                break

    print("\n" + "-" * 78)
    print("Extracted global fuzzy rules")
    print("-" * 78)
    rules = model.global_rules(min_support=3)
    if not rules:
        print("  (no bin reached the minimum support threshold at this graph size)")
    for rule in rules:
        print(" ", rule.text)

    hard_labels = model.predict_hard()
    pos = nx.spring_layout(data.G, seed=0)
    cmap = plt.get_cmap("tab10")

    if data.true_communities is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        true_labels = np.array([min(c) for c in data.true_communities])
        nx.draw_networkx(
            data.G, pos, ax=axes[0], node_color=[cmap(c % 10) for c in true_labels],
            node_size=40, with_labels=False, edge_color="lightgray",
        )
        axes[0].set_title(f"{name}: proxy ground-truth ({data.dataset_name} group)")
        axes[0].axis("off")
        nx.draw_networkx(
            data.G, pos, ax=axes[1], node_color=[cmap(c % 10) for c in hard_labels],
            node_size=40, with_labels=False, edge_color="lightgray",
        )
        axes[1].set_title(f"{name}: NF-MCD predicted communities (k={k})")
        axes[1].axis("off")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw_networkx(
            data.G, pos, ax=ax, node_color=[cmap(c % 10) for c in hard_labels],
            node_size=40, with_labels=False, edge_color="lightgray",
        )
        ax.set_title(f"{name}: NF-MCD predicted communities (k={k})")
        ax.axis("off")

    plt.tight_layout()
    out_png = os.path.join(EXPERIMENTS_DIR, f"{name}_output.png")
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved visualization to {out_png}")


def main():
    jobs = [
        ("crisismmd", lambda: ds_mod.load_crisismmd(max_nodes=800)),
        ("pheme", lambda: ds_mod.load_pheme(max_nodes=2000)),
        ("fakeddit", lambda: ds_mod.load_fakeddit(max_nodes=1200)),
        ("dblp", lambda: ds_mod.load_snap_community("dblp", n_communities=8)),
        ("amazon", lambda: ds_mod.load_snap_community("amazon", n_communities=8, min_community_size=15)),
    ]

    wanted = set(sys.argv[1:])
    unknown = wanted - {name for name, _ in jobs}
    if unknown:
        sys.exit(f"Unknown dataset(s): {sorted(unknown)}. Choose from: {[n for n, _ in jobs]}")

    for name, loader in jobs:
        if wanted and name not in wanted:
            continue
        log_path = os.path.join(EXPERIMENTS_DIR, f"{name}_run.log")
        orig_stdout = sys.stdout
        tee = _Tee(orig_stdout, log_path)
        sys.stdout = tee
        try:
            print(f"Loading {name}...")
            t0 = time.monotonic()
            data = loader()
            print(f"Loaded {name} in {time.monotonic() - t0:.1f}s")
            run_dataset(name, data)
        except Exception:
            print(f"\n*** {name} FAILED ***")
            traceback.print_exc(file=tee)
        finally:
            sys.stdout = orig_stdout
            tee.file.close()
        print(f"[{name}] done, log at {log_path}")

    print("\nMMCas Twitter: skipped - see nf_mcd.datasets.load_mmcas_twitter docstring "
          "(dataset link exists but is permission-restricted as of this run).")


if __name__ == "__main__":
    main()
