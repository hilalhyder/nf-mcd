"""
Stage 5 (Section 4.5 / 5.4): Explainability via fuzzy rule extraction.

Because the fusion and assignment stages are fuzzy/neuro-fuzzy by
construction, every community assignment can be traced back to:

  (a) a per-node explanation: the cross-modal agreement rule that fired for
      that node (from the ANFIS in nf_mcd.fuzzy_fusion), how much the model
      trusted content vs. structure (alpha_i), and its resulting community
      membership degrees; and
  (b) a small set of global IF-THEN rules summarizing how confidence and
      alpha, taken together, relate to community assignment across the
      whole graph - the rule-set compactness/fidelity evaluation described
      in Section 5.4.

This is interpretability *by construction*: no post-hoc explainer (e.g.
LIME/SHAP/GNNExplainer) is required, since the fuzzy membership degrees
already carry the explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .fuzzy_fusion import ANFISAgreement


def _fuzzy_bin(x: float, low: float = 1.0 / 3.0, high: float = 2.0 / 3.0) -> str:
    if x < low:
        return "Low"
    if x < high:
        return "Medium"
    return "High"


@dataclass
class NodeExplanation:
    node_id: int
    modality_flag: str
    agreement: Optional[float]
    confidence: float
    alpha: float
    top_community: int
    top_membership: float
    other_memberships: Dict[int, float]
    text: str


def explain_node(
    node_id: int,
    node_index: int,
    U: np.ndarray,
    confidence: np.ndarray,
    alpha: np.ndarray,
    agreement: np.ndarray,
    modality_flags: List[str],
    anfis: Optional[ANFISAgreement] = None,
    top_k_other: int = 2,
) -> NodeExplanation:
    """Build a human-readable explanation for a single node's community assignment."""
    anfis = anfis or ANFISAgreement()
    row = U[node_index]
    order = np.argsort(-row)
    top_c = int(order[0])
    top_m = float(row[top_c])
    others = {int(c): float(row[c]) for c in order[1: 1 + top_k_other]}

    s_i = agreement[node_index]
    flag = modality_flags[node_index]
    trust_term = _fuzzy_bin(alpha[node_index])

    if flag == "both" and not np.isnan(s_i):
        agree_rule = anfis.dominant_rule(float(s_i))
        basis = (
            f"{agree_rule}. Because agreement confidence is "
            f"{confidence[node_index]:.2f}, the fusion trusts content "
            f"{trust_term.lower()}ly (alpha={alpha[node_index]:.2f}) relative to structure."
        )
    elif flag == "both_unaligned":
        basis = (
            f"Both text and image were present for this node, but too few nodes "
            f"in the dataset had both modalities to reliably learn a cross-modal "
            f"alignment, so agreement could not be estimated; assignment falls back "
            f"to unweighted content and structure at low content trust "
            f"(alpha={alpha[node_index]:.2f})."
        )
    elif flag == "text_only":
        basis = (
            f"Image was unavailable for this node; assignment relies mainly on "
            f"text content and structure (alpha={alpha[node_index]:.2f}, "
            f"trust in content is {trust_term.lower()})."
        )
    elif flag == "image_only":
        basis = (
            f"Text was unavailable for this node; assignment relies mainly on "
            f"image content and structure (alpha={alpha[node_index]:.2f}, "
            f"trust in content is {trust_term.lower()})."
        )
    else:
        basis = (
            "No content (text or image) was available for this node; the "
            "assignment is driven entirely by network structure."
        )

    text = (
        f"Node {node_id}: assigned to community {top_c} with membership "
        f"{top_m:.2f} (secondary memberships: "
        f"{', '.join(f'community {c} ({v:.2f})' for c, v in others.items()) or 'none'}). "
        f"{basis}"
    )

    return NodeExplanation(
        node_id=node_id,
        modality_flag=flag,
        agreement=None if np.isnan(s_i) else float(s_i),
        confidence=float(confidence[node_index]),
        alpha=float(alpha[node_index]),
        top_community=top_c,
        top_membership=top_m,
        other_memberships=others,
        text=text,
    )


@dataclass
class GlobalRule:
    confidence_bin: str
    alpha_bin: str
    support: int
    dominant_community: int
    purity: float
    text: str


def extract_global_rules(
    U: np.ndarray,
    confidence: np.ndarray,
    alpha: np.ndarray,
    min_support: int = 3,
) -> List[GlobalRule]:
    """Bin nodes by fuzzy sets of (confidence, alpha) and report, for each
    non-trivial bin, the dominant community and how "pure" that bin is.

    This is the automatic global rule-extraction procedure referenced in
    Section 5.4: rule-set compactness is controlled by using only 3x3 = 9
    coarse (Low/Medium/High) bins, and fidelity can be computed by checking
    how often a node's actual dominant community matches its bin's
    dominant community (see `rule_fidelity` below).
    """
    hard = np.argmax(U, axis=1)
    conf_bins = np.array([_fuzzy_bin(c) for c in confidence])
    alpha_bins = np.array([_fuzzy_bin(a) for a in alpha])

    rules: List[GlobalRule] = []
    for cb in ["Low", "Medium", "High"]:
        for ab in ["Low", "Medium", "High"]:
            mask = (conf_bins == cb) & (alpha_bins == ab)
            support = int(mask.sum())
            if support < min_support:
                continue
            communities, counts = np.unique(hard[mask], return_counts=True)
            dom_idx = int(np.argmax(counts))
            dominant_community = int(communities[dom_idx])
            purity = float(counts[dom_idx] / support)
            text = (
                f"IF cross-modal confidence is {cb} AND content-trust (alpha) is {ab} "
                f"THEN dominant community is {dominant_community} "
                f"(support={support} nodes, purity={purity:.0%})"
            )
            rules.append(GlobalRule(cb, ab, support, dominant_community, purity, text))

    rules.sort(key=lambda r: (-r.support, -r.purity))
    return rules


def rule_fidelity(rules: List[GlobalRule], U: np.ndarray, confidence: np.ndarray, alpha: np.ndarray) -> float:
    """Fraction of nodes whose actual dominant community matches the dominant
    community predicted by their (confidence-bin, alpha-bin) global rule.
    Higher fidelity means the compact rule set is a faithful summary of the
    full fuzzy model, not just a plausible-looking approximation.
    """
    hard = np.argmax(U, axis=1)
    conf_bins = np.array([_fuzzy_bin(c) for c in confidence])
    alpha_bins = np.array([_fuzzy_bin(a) for a in alpha])
    lookup = {(r.confidence_bin, r.alpha_bin): r.dominant_community for r in rules}

    matches, total = 0, 0
    for i in range(len(hard)):
        key = (conf_bins[i], alpha_bins[i])
        if key in lookup:
            total += 1
            if lookup[key] == hard[i]:
                matches += 1
    return matches / total if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# Neighbourhood-based explanations and measured block sensitivity (additions).
#
# The (confidence, alpha) global rules above do not explain community
# identity (see experiments/explain_summary.log). What predicts a node's
# assignment is its graph neighbourhood, so these explanations state that, and
# report content/structure influence by intervention (zeroing a feature block
# at the fitted cluster centres) instead of reading it off alpha. alpha is an
# input weight in the feature fusion, not a causal measure of influence.
# ---------------------------------------------------------------------------

def fcm_memberships(F: np.ndarray, centers: np.ndarray, m: float) -> np.ndarray:
    """Fuzzy c-means memberships of features F at fixed centres (exact FCM rule)."""
    d2 = (F ** 2).sum(1)[:, None] - 2.0 * F @ centers.T + (centers ** 2).sum(1)[None, :]
    d2 = np.maximum(d2, 1e-12)
    inv = d2 ** (-1.0 / (m - 1.0))
    return inv / inv.sum(axis=1, keepdims=True)


def block_sensitivity(U: np.ndarray, F: np.ndarray, centers: np.ndarray, m: float, n_content_cols: int) -> Dict[str, np.ndarray]:
    """Recompute memberships with the content block zeroed (structure only) and
    with the structure block zeroed (content only), at the fitted centres.

    tv_content[i]   = total variation between U[i] and memberships without content
    tv_structure[i] = total variation between U[i] and memberships without structure
    flip_*          = whether the strongest community changes."""
    nc = n_content_cols
    F_nc = F.copy()
    F_nc[:, :nc] = 0.0
    F_ns = F.copy()
    F_ns[:, nc:] = 0.0
    U_no_content = fcm_memberships(F_nc, centers, m)
    U_no_structure = fcm_memberships(F_ns, centers, m)
    top = U.argmax(1)
    idx = np.arange(len(top))
    is_top = np.eye(U.shape[1], dtype=bool)[top]

    def lead_of_rival(Ux):
        # best rival membership minus the node's own community's membership after the
        # intervention: > 0 means the assignment flips to `rival`
        other = np.where(is_top, -np.inf, Ux)
        return other.max(1) - Ux[idx, top], other.argmax(1)

    margin_c, rival_c = lead_of_rival(U_no_content)
    margin_s, rival_s = lead_of_rival(U_no_structure)
    return dict(
        U_no_content=U_no_content,
        U_no_structure=U_no_structure,
        tv_content=0.5 * np.abs(U - U_no_content).sum(1),
        tv_structure=0.5 * np.abs(U - U_no_structure).sum(1),
        flip_content=(U_no_content.argmax(1) != top),
        flip_structure=(U_no_structure.argmax(1) != top),
        content_only_top=U_no_structure.argmax(1),
        margin_no_content=margin_c,
        rival_no_content=rival_c,
        margin_no_structure=margin_s,
        rival_no_structure=rival_s,
    )


def neighbour_shares(G, nodes, U: np.ndarray):
    """Per-node neighbour counts by community: hard (argmax) and membership-weighted.
    Returns (degree, hard_counts (n,k), weighted_counts (n,k)). Self-loops ignored."""
    n, k = U.shape
    hard = U.argmax(1)
    pos = {v: i for i, v in enumerate(nodes)}
    deg = np.zeros(n)
    counts = np.zeros((n, k))
    wcounts = np.zeros((n, k))
    for v in nodes:
        i = pos[v]
        nb = [pos[w] for w in G.neighbors(v) if w != v]
        deg[i] = len(nb)
        if nb:
            counts[i] = np.bincount(hard[nb], minlength=k)[:k]
            wcounts[i] = U[nb].sum(axis=0)
    return deg, counts, wcounts


@dataclass
class NeighbourhoodExplanation:
    node_id: object
    node_index: int
    modality_flag: str
    top_community: int
    top_membership: float
    n_neighbours: int
    neighbours_in_top: int
    share_top: float               # hard share of neighbours in the node's community
    weighted_share_top: float      # membership-weighted share
    chance_share: float            # fraction of all nodes in that community
    lift: float                    # share_top / chance_share
    second_community: Optional[int]
    second_share: float
    content_only_community: Optional[int]
    content_agrees: Optional[bool]
    content_sensitivity: float     # TV shift in memberships if content is removed
    structure_sensitivity: float   # TV shift if structure is removed
    flips_without_content: bool
    flips_without_structure: bool
    text: str
    content_flip_margin: float = 0.0     # > 0: assignment flips if content is removed
    structure_flip_margin: float = 0.0


_MODALITY_SENTENCE = {
    "text_only": "The image was unavailable for this node.",
    "image_only": "The text was unavailable for this node.",
    "both_unaligned": "Too few nodes had both modalities to learn a cross-modal alignment.",
}


def explain_node_neighbourhood(
    node_id,
    node_index: int,
    U: np.ndarray,
    deg: np.ndarray,
    counts: np.ndarray,
    wcounts: np.ndarray,
    sens: Dict[str, np.ndarray],
    modality_flags: List[str],
) -> NeighbourhoodExplanation:
    """Neighbourhood explanation for one node (see module note above)."""
    i = node_index
    n, k = U.shape
    hard = U.argmax(1)
    c = int(hard[i])
    mem = float(U[i, c])
    d = int(deg[i])
    chance = float((hard == c).mean())
    flag = modality_flags[i]
    has_content = flag != "none"
    tvc = float(sens["tv_content"][i])
    tvs = float(sens["tv_structure"][i])
    flip_c = bool(sens["flip_content"][i])
    flip_s = bool(sens["flip_structure"][i])

    share = share_w = lift = second_share = 0.0
    n_in = 0
    second: Optional[int] = None
    if d > 0:
        n_in = int(counts[i, c])
        share = n_in / d
        share_w = float(wcounts[i, c] / max(wcounts[i].sum(), 1e-12))
        lift = share / chance if chance > 0 else float("nan")
        order = np.argsort(-counts[i])
        for cc in order:
            if int(cc) != c and counts[i, cc] > 0:
                second, second_share = int(cc), float(counts[i, cc] / d)
                break

    content_only = int(sens["content_only_top"][i]) if has_content else None
    agrees = None if content_only is None else (content_only == c)

    if d == 0:
        head = f"Node {node_id} is in community {c} (membership {mem:.2f}): it has no neighbours in the graph, so the assignment comes from content only."
    else:
        head = (
            f"Node {node_id} is in community {c} (membership {mem:.2f}): {n_in} of its {d} neighbours "
            f"({share:.0%}) are also in community {c}, versus {chance:.0%} expected by chance."
        )
        if second is not None and second_share > share:
            head += f" Most of its neighbours are in community {second} ({second_share:.0%}) instead."
        elif second is not None:
            head += f" Next strongest neighbourhood: community {second} ({second_share:.0%})."

    if not has_content:
        body = " No content (text or image) was available, so the assignment rests on network structure alone."
        if d == 0:
            body = " It also has no content, so this assignment is essentially arbitrary."
    else:
        parts = []
        if flag in _MODALITY_SENTENCE:
            parts.append(_MODALITY_SENTENCE[flag])
        if agrees:
            parts.append(f"Content supports this (content alone also gives community {c}).")
        else:
            parts.append(f"Content contradicts the neighbourhood: content alone would place it in community {content_only}.")
        mc = float(sens["margin_no_content"][i])
        ms = float(sens["margin_no_structure"][i])
        rc = int(sens["rival_no_content"][i])
        rs = int(sens["rival_no_structure"][i])
        if abs(mc) < 1e-3:
            content_txt = "Removing content would leave its assignment undetermined (memberships tie)"
        elif flip_c:
            content_txt = f"Removing content would flip its assignment to community {rc}"
        else:
            content_txt = f"Removing content would keep it in community {c} (lead {-mc:.2f})"
        if abs(ms) < 1e-3:
            struct_txt = "removing structure would leave it undetermined (memberships tie)"
        elif flip_s:
            struct_txt = f"removing structure would flip it to community {rs}"
        else:
            struct_txt = f"removing structure would keep it in community {c} (lead {-ms:.2f})"
        parts.append(f"{content_txt}; {struct_txt}.")
        body = " " + " ".join(parts)

    return NeighbourhoodExplanation(
        node_id=node_id, node_index=i, modality_flag=flag, top_community=c, top_membership=mem,
        n_neighbours=d, neighbours_in_top=n_in, share_top=share, weighted_share_top=share_w,
        chance_share=chance, lift=lift, second_community=second, second_share=second_share,
        content_only_community=content_only, content_agrees=agrees,
        content_sensitivity=tvc, structure_sensitivity=tvs,
        flips_without_content=flip_c, flips_without_structure=flip_s, text=head + body,
        content_flip_margin=float(sens["margin_no_content"][i]),
        structure_flip_margin=float(sens["margin_no_structure"][i]),
    )


@dataclass
class NeighbourhoodRule:
    community: int
    threshold: float
    support: int
    precision: float
    text: str


NBR_THRESHOLD_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def extract_neighbourhood_rules(
    deg: np.ndarray,
    counts: np.ndarray,
    hard: np.ndarray,
    min_support: int = 5,
    precision_target: float = 0.8,
    grid=NBR_THRESHOLD_GRID,
) -> List[NeighbourhoodRule]:
    """One rule per community: IF at least t of a node's neighbours are in community c
    THEN the node is in community c. t is the smallest value on a fixed grid whose
    precision reaches `precision_target` with at least `min_support` nodes; if none
    does, the grid value with the best F1 among those with enough support is used;
    communities with no adequately supported threshold get no rule."""
    n, k = counts.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = np.where(deg[:, None] > 0, counts / np.maximum(deg[:, None], 1), 0.0)
    rules: List[NeighbourhoodRule] = []
    for c in range(k):
        size = int((hard == c).sum())
        if size == 0:
            continue
        cand = []
        for t in grid:
            fire = (shares[:, c] >= t) & (deg > 0)
            sup = int(fire.sum())
            if sup < min_support:
                continue
            tp = int((fire & (hard == c)).sum())
            prec = tp / sup
            rec = tp / size
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            cand.append((t, sup, prec, f1))
        if not cand:
            continue
        good = [x for x in cand if x[2] >= precision_target]
        t, sup, prec, _ = good[0] if good else max(cand, key=lambda x: x[3])
        rules.append(NeighbourhoodRule(
            community=c, threshold=float(t), support=sup, precision=float(prec),
            text=(f"IF at least {t:.0%} of a node's neighbours are in community {c} "
                  f"THEN the node is in community {c} (support={sup} nodes, precision={prec:.0%})"),
        ))
    rules.sort(key=lambda r: (-r.support, -r.precision))
    return rules


def apply_neighbourhood_rules(rules: List[NeighbourhoodRule], deg: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Predicted community per node from the rule set; -1 if no rule fires. When several
    fire, the largest neighbour share wins; exact ties count as no prediction (-1)."""
    n, k = counts.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = np.where(deg[:, None] > 0, counts / np.maximum(deg[:, None], 1), 0.0)
    pred = np.full(n, -1)
    best = np.full(n, -1.0)
    tie = np.zeros(n, dtype=bool)
    for r in rules:
        fire = (shares[:, r.community] >= r.threshold) & (deg > 0)
        s = shares[:, r.community]
        better = fire & (s > best + 1e-12)
        equal = fire & (np.abs(s - best) <= 1e-12) & (pred != r.community)
        pred[better] = r.community
        best[better] = s[better]
        tie[better] = False
        tie[equal] = True
    pred[tie] = -1
    return pred


def neighbourhood_rule_fidelity(
    rules: List[NeighbourhoodRule],
    deg: np.ndarray,
    counts: np.ndarray,
    hard: np.ndarray,
    fallback: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Fidelity of a neighbourhood rule set to the model's hard assignments.
    `fidelity` counts nodes where no rule fires (or a tie) as misses; `coverage` is the
    share of nodes where a rule fires; `precision_covered` is fidelity among covered
    nodes; `fidelity_with_fallback` predicts `fallback` (e.g. content-only community)
    for uncovered nodes; `majority` is the always-largest-community baseline."""
    pred = apply_neighbourhood_rules(rules, deg, counts)
    covered = pred >= 0
    hit = pred == hard
    out = {
        "n_rules": float(len(rules)),
        "coverage": float(covered.mean()),
        "fidelity": float(hit.mean()),
        "precision_covered": float(hit[covered].mean()) if covered.any() else float("nan"),
        "majority": float(np.bincount(hard).max() / len(hard)),
    }
    if fallback is not None:
        p2 = np.where(covered, pred, fallback)
        out["fidelity_with_fallback"] = float((p2 == hard).mean())
    return out
