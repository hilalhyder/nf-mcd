"""
End-to-end demonstration of NF-MCD on synthetic multimodal social-graph data.

Run with:  python demo.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from nf_mcd import NFMCD
from nf_mcd.datasets import generate_synthetic_multimodal_graph


def main():
    print("=" * 78)
    print("NF-MCD demo: synthetic multimodal social graph")
    print("=" * 78)

    data = generate_synthetic_multimodal_graph(
        n_nodes=120,
        n_communities=4,
        p_in=0.18,
        p_out=0.02,
        missing_modality_rate=0.15,
        misalignment_rate=0.15,
        overlap_rate=0.10,
        seed=42,
    )
    print(f"\nGenerated graph: {data.G.number_of_nodes()} nodes, {data.G.number_of_edges()} edges, "
          f"{data.n_communities} ground-truth communities")
    print(f"  Nodes with missing text:  {len(data.missing_text_nodes)}")
    print(f"  Nodes with missing image: {len(data.missing_image_nodes)}")
    print(f"  Nodes with misaligned image/text: {len(data.misaligned_nodes)}")

    model = NFMCD(n_communities=data.n_communities, seed=0)
    model.fit(
        data.G,
        text_embeddings=data.text_embeddings,
        image_embeddings=data.image_embeddings,
    )
    print("\nFuzzy c-means converged in", model.fcm_result_.n_iter, "iterations "
          f"(final objective {model.fcm_result_.objective_history[-1]:.3f})")

    scores = model.evaluate(true_communities_per_node=data.true_communities)
    print("\nEvaluation against ground truth:")
    for k, v in scores.items():
        print(f"  {k:>18s}: {v:.3f}")

    print("\n" + "-" * 78)
    print("Example per-node explanations")
    print("-" * 78)
    # One example from each interesting case: aligned, misaligned, missing modality.
    example_nodes = []
    if data.misaligned_nodes:
        example_nodes.append(("misaligned image/text", next(iter(data.misaligned_nodes))))
    if data.missing_image_nodes:
        example_nodes.append(("missing image", next(iter(data.missing_image_nodes))))
    if data.missing_text_nodes:
        example_nodes.append(("missing text", next(iter(data.missing_text_nodes))))
    clean_nodes = (
        set(range(data.G.number_of_nodes()))
        - data.misaligned_nodes
        - data.missing_image_nodes
        - data.missing_text_nodes
    )
    if clean_nodes:
        example_nodes.append(("clean / aligned", next(iter(clean_nodes))))

    for label, node_id in example_nodes:
        expl = model.explain_node(node_id)
        print(f"\n[{label}]")
        print(" ", expl.text)

    print("\n" + "-" * 78)
    print("Extracted global fuzzy rules (Section 4.5 / 5.4)")
    print("-" * 78)
    rules = model.global_rules(min_support=3)
    if not rules:
        print("  (no bin reached the minimum support threshold at this graph size)")
    for rule in rules:
        print(" ", rule.text)

    # --- Visualization -----------------------------------------------------
    hard_labels = model.predict_hard()
    pos = nx.spring_layout(data.G, seed=0)
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    true_labels = np.array([min(c) for c in data.true_communities])  # primary community for coloring
    nx.draw_networkx(
        data.G, pos, ax=axes[0], node_color=[cmap(c) for c in true_labels],
        node_size=80, with_labels=False, edge_color="lightgray",
    )
    axes[0].set_title("Ground-truth communities")
    axes[0].axis("off")

    nx.draw_networkx(
        data.G, pos, ax=axes[1], node_color=[cmap(c) for c in hard_labels],
        node_size=80, with_labels=False, edge_color="lightgray",
    )
    axes[1].set_title("NF-MCD predicted communities (defuzzified)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("demo_output.png", dpi=150)
    print("\nSaved visualization to demo_output.png")


if __name__ == "__main__":
    main()
