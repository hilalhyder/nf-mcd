"""
Label-independent graphs for the Fakeddit and CrisisMMD sensitivity studies.

1. `fakeddit_relation_sample`: a graph built only from real Reddit relations
   (shared author, shared non-generic link domain, shared linked_submission_id).
   Nodes are chosen by snowball sampling from label-blind random seed posts, so
   neither node selection nor edges use the subreddit label. The subreddit is
   only used afterwards as the ground-truth community (and to drop subreddits
   that ended up with too few sampled posts). Author behaviour can still
   correlate with subreddit, so homophily is reported, not assumed.

2. `sbm_graph`: a controlled stochastic block model over given groups, with
   p_out/p_in as the homophily knob. Synthetic sensitivity study, not a real graph.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

GENERIC_DOMAINS = {
    "i.redd.it", "i.imgur.com", "imgur.com", "i.reddituploads.com",
    "upload.wikimedia.org", "reddit.com", "v.redd.it", "youtube.com", "youtu.be",
    "twitter.com", "i.redditmedia.com", "gfycat.com", "flickr.com", "i.sli.mg",
    "instagram.com", "facebook.com", "en.wikipedia.org", "giphy.com",
    "pbs.twimg.com", "i.postimg.cc", "media.giphy.com", "preview.redd.it",
    "external-preview.redd.it",
}
SKIP_AUTHORS = {"[deleted]", "AutoModerator"}


def _groups(series, min_size: int, max_size: int, exclude=None) -> Dict[object, np.ndarray]:
    out = {}
    for key, ix in series.groupby(series).indices.items():
        if exclude is not None and key in exclude:
            continue
        out[key] = ix
    return out


def fakeddit_relation_sample(
    pool,
    n_target: int = 1400,
    n_seeds: int = 12,
    neighbours_per_group: int = 6,
    max_posts_per_author: int = 40,
    domain_group_range: Tuple[int, int] = (2, 25),
    link_group_range: Tuple[int, int] = (2, 25),
    min_sub_nodes: int = 30,
    use_linked: bool = True,
    seed: int = 42,
):
    """Snowball-sample a connected-ish node set from `pool` (a DataFrame with
    author, domain, linked_submission_id, subreddit columns; row index = node id)
    and return (sub_df, G, stats). `sub_df` rows are the kept nodes in node order.
    """
    rng = np.random.default_rng(seed)
    pool = pool.reset_index(drop=True)
    pool = pool[~pool["author"].isin(SKIP_AUTHORS)].reset_index(drop=True)

    by_author = pool.groupby("author").indices
    dom_ok = ~pool["domain"].isin(GENERIC_DOMAINS) & pool["domain"].notna()
    by_domain = pool[dom_ok].groupby("domain").indices
    link_ok = pool["linked_submission_id"].notna() if use_linked else np.zeros(len(pool), bool)
    by_link = pool[link_ok].groupby("linked_submission_id").indices if use_linked else {}

    author_of = pool["author"].to_numpy()
    domain_of = pool["domain"].to_numpy()
    link_of = pool["linked_submission_id"].to_numpy()

    def neighbours(i: int) -> List[int]:
        cand: List[int] = []
        a = author_of[i]
        cand += list(by_author.get(a, []))
        d = domain_of[i]
        if d in by_domain and domain_group_range[0] <= len(by_domain[d]) <= domain_group_range[1] * 40:
            cand += list(by_domain[d])
        if use_linked:
            lk = link_of[i]
            if lk in by_link and link_group_range[0] <= len(by_link[lk]) <= link_group_range[1] * 40:
                cand += list(by_link[lk])
        cand = [c for c in set(cand) if c != i]
        if len(cand) > neighbours_per_group:
            cand = list(rng.choice(cand, size=neighbours_per_group, replace=False))
        return cand

    seeds = rng.choice(len(pool), size=n_seeds, replace=False)
    selected: Dict[int, None] = {int(s): None for s in seeds}
    frontiers = [[int(s)] for s in seeds]
    per_author: Dict[object, int] = {}
    for i in selected:
        per_author[author_of[i]] = per_author.get(author_of[i], 0) + 1
    while len(selected) < n_target and any(frontiers):
        for fi in range(len(frontiers)):
            if not frontiers[fi] or len(selected) >= n_target:
                continue
            cur = frontiers[fi].pop(0)
            for nb in neighbours(cur):
                if nb in selected or len(selected) >= n_target:
                    continue
                if per_author.get(author_of[nb], 0) >= max_posts_per_author:
                    continue
                selected[nb] = None
                per_author[author_of[nb]] = per_author.get(author_of[nb], 0) + 1
                frontiers[fi].append(nb)

    sub = pool.iloc[list(selected)].reset_index(drop=True)
    # Keep only subreddits with enough sampled posts to be meaningful communities.
    counts = sub["subreddit"].value_counts()
    keep_subs = counts[counts >= min_sub_nodes].index
    sub = sub[sub["subreddit"].isin(keep_subs)].reset_index(drop=True)

    edges: Dict[Tuple[int, int], set] = {}

    def add_clique(ix, kind):
        for u, v in itertools.combinations(sorted(int(x) for x in ix), 2):
            edges.setdefault((u, v), set()).add(kind)

    for a, ix in sub.groupby("author").indices.items():
        add_clique(ix, "author")
    sub_dom_ok = ~sub["domain"].isin(GENERIC_DOMAINS) & sub["domain"].notna()
    for d, ix in sub[sub_dom_ok].groupby("domain").indices.items():
        if domain_group_range[0] <= len(ix) <= domain_group_range[1]:
            add_clique(ix, "domain")
    if use_linked:
        sl = sub["linked_submission_id"].notna()
        for lk, ix in sub[sl].groupby("linked_submission_id").indices.items():
            if link_group_range[0] <= len(ix) <= link_group_range[1]:
                add_clique(ix, "linked")

    G = nx.Graph()
    G.add_nodes_from(range(len(sub)))
    G.add_edges_from(edges)
    lcc = max(nx.connected_components(G), key=len)
    stats = {"nodes_sampled": len(sub), "edges": G.number_of_edges(),
             "components": nx.number_connected_components(G), "lcc_nodes": len(lcc)}
    lab = sub["subreddit"].to_numpy()
    for kind in ("author", "domain", "linked"):
        es = [e for e, kinds in edges.items() if kind in kinds]
        stats[f"edges_{kind}"] = len(es)
        stats[f"homophily_{kind}"] = float(np.mean([lab[u] == lab[v] for u, v in es])) if es else float("nan")
    stats["homophily_all"] = float(np.mean([lab[u] == lab[v] for u, v in edges])) if edges else float("nan")
    p = sub["subreddit"].value_counts(normalize=True).to_numpy()
    stats["homophily_random_baseline"] = float((p ** 2).sum())
    return sub, G, stats


def sbm_graph(groups: np.ndarray, p_in: float, ratio: float, seed: int) -> nx.Graph:
    """SBM over `groups` with p_out = ratio * p_in; ratio=1.0 carries no label information."""
    rng = np.random.default_rng(seed)
    n = len(groups)
    same = groups[:, None] == groups[None, :]
    P = np.where(same, p_in, ratio * p_in)
    R = rng.random((n, n))
    A = np.triu(R < P, k=1)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(*np.nonzero(A)))
    return G
