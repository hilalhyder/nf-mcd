"""
Datasets: a synthetic multimodal-graph generator for development and
demonstration, plus stubs for the real datasets identified in the paper's
experimental design (Section 5.1 / Table 3).

The synthetic generator directly synthesizes *embeddings* with known,
controllable community-correlated signal (rather than raw text/images run
through nf_mcd.encoders), so that the rest of the pipeline can be tested
and demonstrated end-to-end without needing real pretrained-model
downloads. It injects, by design, the two failure modes NF-MCD is meant to
be robust to:
  - missing modalities (some nodes have no text and/or no image), and
  - cross-modal misalignment (some nodes' image embedding is drawn from a
    *different* community's centroid than their text embedding, simulating
    a mismatched image-caption pair).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))


@dataclass
class RealMultimodalData:
    """Common return shape for the real-dataset loaders below (CrisisMMD,
    PHEME, Fakeddit). `true_communities` uses a reconstructed/proxy notion
    of "community" in every case (see each loader's docstring for exactly
    what was used and why) since none of these datasets ships a native
    ground-truth community partition the way the synthetic generator does.
    """
    G: nx.Graph
    texts: List[Optional[str]]
    images: List[Optional[object]]         # PIL.Image.Image or None
    true_communities: List[Set[int]]
    n_communities: int
    community_names: List[str]
    dataset_name: str


def _sbm_like_edges(group_ids: np.ndarray, p_in: float, p_out: float, rng: np.random.Generator) -> np.ndarray:
    """Vectorized stochastic-block-model-style edge sampling: denser within
    the same group (e.g. disaster event / subreddit) than across groups.
    Used by the real-dataset loaders below to reconstruct a plausible graph
    topology when no native social graph is available in the source data
    (see each loader's docstring)."""
    n = len(group_ids)
    same = group_ids[:, None] == group_ids[None, :]
    probs = np.where(same, p_in, p_out)
    rand = rng.random((n, n))
    adj = (rand < probs) & ~np.eye(n, dtype=bool)
    adj = np.triu(adj, 1)
    return np.argwhere(adj)


@dataclass
class SyntheticData:
    G: nx.Graph
    text_embeddings: List[Optional[np.ndarray]]
    image_embeddings: List[Optional[np.ndarray]]
    true_communities: List[Set[int]]   # one set of community ids per node (supports overlap)
    n_communities: int
    misaligned_nodes: Set[int]         # nodes whose image embedding was drawn off-community
    missing_text_nodes: Set[int]
    missing_image_nodes: Set[int]


def generate_synthetic_multimodal_graph(
    n_nodes: int = 120,
    n_communities: int = 4,
    p_in: float = 0.18,
    p_out: float = 0.02,
    text_dim: int = 384,
    image_dim: int = 512,
    missing_modality_rate: float = 0.15,
    misalignment_rate: float = 0.15,
    overlap_rate: float = 0.10,
    centroid_scale: float = 3.0,
    noise_scale: float = 1.0,
    seed: int = 42,
    blend_strength: float = 0.0,
) -> SyntheticData:
    """Generate a synthetic social graph with known overlapping community
    structure and community-correlated multimodal content.

    Structure: a stochastic block model with `n_communities` equal-sized
    blocks (p_in within-block edge probability, p_out between-block).
    Content: each community has a random centroid direction in text and
    image embedding space; each node's embeddings are its community's
    centroid plus Gaussian noise, EXCEPT for `misalignment_rate` of nodes
    (image drawn from a different, randomly chosen community's centroid)
    and `missing_modality_rate` of nodes (text and/or image set to None).

    blend_strength : float, default 0.0 (fully backward compatible)
        At the default, a node given a second ("overlapping") community by
        `overlap_rate` is a label with NO footprint anywhere else: its edges
        and embeddings are drawn purely from its primary community, so the
        "overlap" is undetectable by any method (confirmed empirically --
        see experiments/overlap_summary.log). `blend_strength` in (0, 1]
        gives such nodes real, detectable signal instead: their edges are
        additionally connected into their secondary community's block at a
        rate interpolated from p_out (0.0) to p_in (1.0), and their text/image
        embeddings become a blend_strength/2-weighted mix of both communities'
        centroids (so 1.0 = a symmetric 50/50 mix). Only affects nodes with
        a genuine second community; has no effect when overlap_rate == 0.
        At the default 0.0, this parameter changes nothing: the extra edge
        randomness is drawn from an RNG independent of `rng` below and only
        touched when blend_strength > 0, and the content-mixing weight is
        exactly 0 (so the centroid used is exactly `text_centroids[c]`, byte-
        identical to before this parameter existed, for any seed/overlap_rate).
    """
    rng = np.random.default_rng(seed)
    sizes = [n_nodes // n_communities] * n_communities
    sizes[-1] += n_nodes - sum(sizes)  # absorb remainder into last block

    probs = np.full((n_communities, n_communities), p_out)
    np.fill_diagonal(probs, p_in)
    G = nx.stochastic_block_model(sizes, probs.tolist(), seed=seed)

    # Primary community per node, in node order.
    primary = []
    for c, size in enumerate(sizes):
        primary.extend([c] * size)
    primary = np.array(primary[:n_nodes])
    nodes_by_block = {c: np.where(primary == c)[0].tolist() for c in range(n_communities)}

    true_communities: List[Set[int]] = [{int(c)} for c in primary]
    if overlap_rate > 0:
        for i in range(n_nodes):
            if rng.random() < overlap_rate:
                other = int(rng.integers(0, n_communities))
                true_communities[i].add(other)

    if blend_strength > 0:
        # Independent RNG stream: never consumes from `rng`, so it cannot
        # perturb the draw sequence used below when blend_strength == 0.
        edge_rng = np.random.default_rng((seed, 7919, int(round(blend_strength, 6) * 1_000_000)))
        extra_p = blend_strength * (p_in - p_out)
        for i in range(n_nodes):
            comms = true_communities[i]
            if len(comms) <= 1:
                continue
            c2 = next(c for c in comms if c != int(primary[i]))
            for j in nodes_by_block[c2]:
                if j == i or G.has_edge(i, j):
                    continue
                if edge_rng.random() < extra_p:
                    G.add_edge(i, j)

    text_centroids = rng.normal(scale=centroid_scale, size=(n_communities, text_dim))
    image_centroids = rng.normal(scale=centroid_scale, size=(n_communities, image_dim))

    text_embeddings: List[Optional[np.ndarray]] = [None] * n_nodes
    image_embeddings: List[Optional[np.ndarray]] = [None] * n_nodes
    misaligned_nodes: Set[int] = set()
    missing_text_nodes: Set[int] = set()
    missing_image_nodes: Set[int] = set()

    for i in range(n_nodes):
        c = int(primary[i])
        comms = true_communities[i]
        w, c2 = 0.0, None
        if blend_strength > 0 and len(comms) > 1:
            c2 = next(cc for cc in comms if cc != c)
            w = 0.5 * blend_strength

        text_centroid_i = text_centroids[c] if w == 0.0 else (1 - w) * text_centroids[c] + w * text_centroids[c2]
        text_embeddings[i] = text_centroid_i + rng.normal(scale=noise_scale, size=text_dim)

        if rng.random() < misalignment_rate:
            wrong_c = int(rng.integers(0, n_communities))
            while wrong_c == c and n_communities > 1:
                wrong_c = int(rng.integers(0, n_communities))
            image_embeddings[i] = image_centroids[wrong_c] + rng.normal(scale=noise_scale, size=image_dim)
            misaligned_nodes.add(i)
        else:
            image_centroid_i = image_centroids[c] if w == 0.0 else (1 - w) * image_centroids[c] + w * image_centroids[c2]
            image_embeddings[i] = image_centroid_i + rng.normal(scale=noise_scale, size=image_dim)

        if rng.random() < missing_modality_rate:
            if rng.random() < 0.5:
                text_embeddings[i] = None
                missing_text_nodes.add(i)
            else:
                image_embeddings[i] = None
                missing_image_nodes.add(i)

    return SyntheticData(
        G=G,
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        true_communities=true_communities,
        n_communities=n_communities,
        misaligned_nodes=misaligned_nodes,
        missing_text_nodes=missing_text_nodes,
        missing_image_nodes=missing_image_nodes,
    )


def node_sets_to_community_view(node_communities: List[Set[int]], n_communities: int) -> List[Set[int]]:
    """Convert a per-node representation (node -> set of community ids) into
    a per-community representation (community -> set of node indices), as
    expected by nf_mcd.metrics.overlapping_nmi / membership_f1.
    """
    view: List[Set[int]] = [set() for _ in range(n_communities)]
    for node_idx, communities in enumerate(node_communities):
        for c in communities:
            if 0 <= c < n_communities:
                view[c].add(node_idx)
    return view


# ---------------------------------------------------------------------------
# Real-dataset loader stubs (Section 5.1 / Table 3 of the paper).
#
# These are intentionally left unimplemented: each dataset has its own
# access process, license, and file format, so filling these in is a
# per-dataset engineering task best done against the actual downloaded
# files. Each stub documents where to get the data and what NFMCD.fit()
# expects back.
# ---------------------------------------------------------------------------

def load_mmcas_twitter(root_dir: str):
    """MMCas Twitter benchmark (UM-Data-Intelligence-Lab, WWW 2026).

    Status as of this writing (checked directly): the project's GitHub repo
    (github.com/UM-Data-Intelligence-Lab/MMCas) now lists a Google Drive
    link for the "Twitter"/"weibo" datasets (Twitter-hashtag is still
    marked "coming soon" in their README). However the linked Google Drive
    file is access-restricted - both `gdown` and a direct fetch fail with
    "Cannot retrieve the public link of the file... may need to change the
    permission to 'Anyone with the link'" - so it is NOT actually
    obtainable non-interactively despite the link existing. This may just
    be a temporary/quota issue on the authors' end; re-check the repo link
    above before concluding it's permanently unavailable.

    Should return (G, text_embeddings_or_texts, image_embeddings_or_images,
    true_communities) in the same shapes as `generate_synthetic_multimodal_graph`.
    """
    raise NotImplementedError(
        "The MMCas Google Drive link (see docstring) is currently permission-"
        "restricted and not downloadable via gdown/requests. Fill in once "
        "the authors make it publicly accessible, or once you have "
        "credentials to access it interactively."
    )


def load_crisismmd(
    root_dir: Optional[str] = None,
    config: str = "humanitarian",
    split: str = "train",
    max_nodes: int = 800,
    download_images: bool = True,
    p_in: float = 0.06,
    p_out: float = 0.004,
    seed: int = 42,
) -> RealMultimodalData:
    """CrisisMMD (Alam, Ofli & Imran, ICWSM 2018), loaded from HuggingFace
    `QCRI/CrisisMMD` (config: humanitarian/informative/damage; default
    "humanitarian"). ~16k labeled tweets with images from seven 2017
    disasters; documented image-text misalignment makes it well suited to
    robustness testing.

    Graph reconstruction: CrisisMMD has no native social graph (tweets are
    independently sampled, not linked by reply/retweet in the release), so
    topology is reconstructed via `_sbm_like_edges`: nodes from the same
    disaster event are connected with probability `p_in`, nodes from
    different events with probability `p_out` - i.e. "same disaster event"
    is treated as the co-occurrence signal the docstring originally called
    for, sampled rather than a dense clique so the graph has realistic
    social-network sparsity instead of being a trivial union of complete
    components. `true_communities` = disaster event id, which is therefore
    partially circular by construction (the graph itself is built from the
    event labels) - treat modularity/NMI on this dataset as validating
    "does NF-MCD recover the event-correlated structure we injected",
    analogous to the synthetic generator, not as an independent real-world
    ground truth the way a native social graph would be.

    One row is kept per unique `tweet_id` (a tweet may have multiple
    associated images; only the first is used). Images are downloaded
    individually via `huggingface_hub.hf_hub_download` (the HF dataset's
    `image` column is not populated with bytes, only `image_path`) into
    `root_dir` (default: `<package>/../data/crisismmd`) and opened as PIL
    images; text is the raw (unfiltered) `tweet_text` field, on purpose,
    since noisy social-media text is exactly what `nf_mcd.encoders`
    should be exercised against.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from PIL import Image

    root_dir = root_dir or os.path.join(DEFAULT_DATA_ROOT, "crisismmd")
    os.makedirs(root_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    ds = load_dataset("QCRI/CrisisMMD", config, split=split)

    seen = {}
    for i in range(len(ds)):
        row = ds[i]
        tid = row["tweet_id"]
        if tid not in seen:
            seen[tid] = row
    rows = list(seen.values())
    order = rng.permutation(len(rows))
    rows = [rows[i] for i in order][:max_nodes]

    events = sorted(set(r["event_name"] for r in rows))
    event_to_id = {e: i for i, e in enumerate(events)}
    node_events = np.array([event_to_id[r["event_name"]] for r in rows])
    texts: List[Optional[str]] = [r["tweet_text"] for r in rows]

    images: List[Optional[object]] = [None] * len(rows)
    if download_images:
        n_ok, n_fail = 0, 0
        for i, r in enumerate(rows):
            try:
                path = hf_hub_download(
                    repo_id="QCRI/CrisisMMD",
                    filename=r["image_path"],
                    repo_type="dataset",
                    cache_dir=root_dir,
                )
                images[i] = Image.open(path).convert("RGB")
                n_ok += 1
            except Exception as exc:  # noqa: BLE001 - individual download failures are expected/non-fatal
                n_fail += 1
                logger.warning("CrisisMMD: failed to download image for tweet %s: %s", r["tweet_id"], exc)
        logger.info("CrisisMMD: downloaded %d/%d images (%d failed)", n_ok, len(rows), n_fail)

    n = len(rows)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(_sbm_like_edges(node_events, p_in, p_out, rng).tolist())

    true_communities: List[Set[int]] = [{int(c)} for c in node_events]

    return RealMultimodalData(
        G=G,
        texts=texts,
        images=images,
        true_communities=true_communities,
        n_communities=len(events),
        community_names=events,
        dataset_name="crisismmd",
    )


def load_fakeddit(
    root_dir: Optional[str] = None,
    tsv_path: Optional[str] = None,
    max_nodes: int = 1200,
    max_download_attempts: int = 3000,
    max_download_seconds: float = 900.0,
    request_timeout: float = 5.0,
    top_n_subreddits: int = 10,
    min_success_rate_for_images: float = 0.15,
    p_in: float = 0.08,
    p_out: float = 0.004,
    seed: int = 42,
) -> RealMultimodalData:
    """Fakeddit (Nakamura, Levy & Wang, LREC 2020). >1M Reddit posts; this
    loader uses the `multimodal_only_samples/multimodal_train.tsv` split
    (the paper's own recommendation: "results in the paper are based on
    multimodal samples only"), NOT the full >1M-row `all_samples` release -
    fully downloading/embedding >1M images is explicitly out of scope for
    a single session (see module-level notes). `tsv_path` defaults to
    `<root_dir>/multimodal_only_samples/multimodal_train.tsv`; download the
    "text and metadata" Google Drive folder linked from the Fakeddit GitHub
    README (entitize/Fakeddit) into `root_dir` (default:
    `<package>/../data/fakeddit`) first.

    Sampling: restricts to the `top_n_subreddits` most frequent subreddits
    in the TSV (Fakeddit spans hundreds of subreddits; using all of them
    would make "community" nearly one-node-per-community), then randomly
    samples up to `max_nodes` rows from those, using `clean_title` as text.

    Images: Fakeddit ships only `image_url` (no bundled image files), so
    each row's image must be fetched individually over HTTP - this is the
    single biggest real-world risk in this loader (dead links, host
    rate-limiting, redirects to placeholder/removed images). Downloading is
    bounded by BOTH `max_download_attempts` and `max_download_seconds`
    (whichever hits first stops further attempts), each request has its
    own `request_timeout`. If the resulting success rate falls below
    `min_success_rate_for_images`, this loader gives up on images entirely
    for the whole run and returns `images=[None]*n` (degrading gracefully
    to a text-only experiment) rather than returning a mostly-empty,
    unrepresentative image set - this condition is logged clearly, check
    the log if you expected images and didn't get them.

    Graph reconstruction: no native social graph in the release, so (same
    approach as `load_crisismmd`, for the same reason) `_sbm_like_edges` is
    used with subreddit as the group signal: denser within-subreddit
    (`p_in`) than across (`p_out`). `true_communities` = subreddit id, with
    the same "partially circular by construction" caveat noted in
    `load_crisismmd`'s docstring.
    """
    import time

    import pandas as pd
    import requests
    from PIL import Image
    from io import BytesIO

    root_dir = root_dir or os.path.join(DEFAULT_DATA_ROOT, "fakeddit")
    tsv_path = tsv_path or os.path.join(root_dir, "multimodal_only_samples", "multimodal_train.tsv")
    if not os.path.isfile(tsv_path):
        raise FileNotFoundError(
            f"Expected Fakeddit multimodal TSV at {tsv_path!r}. Download the "
            f"'text and metadata' folder from the Fakeddit GitHub README "
            f"(entitize/Fakeddit) into {root_dir!r} first."
        )

    rng = np.random.default_rng(seed)
    df = pd.read_csv(tsv_path, sep="\t")
    df = df.dropna(subset=["clean_title", "image_url", "subreddit"])
    df = df[df["image_url"].astype(str).str.startswith("http")]

    top_subs = df["subreddit"].value_counts().nlargest(top_n_subreddits).index.tolist()
    df = df[df["subreddit"].isin(top_subs)]

    idx = rng.permutation(len(df))
    df = df.iloc[idx]

    subs = sorted(top_subs)
    sub_to_id = {s: i for i, s in enumerate(subs)}

    texts: List[Optional[str]] = []
    node_subs: List[int] = []
    urls: List[str] = []
    for _, row in df.iterrows():
        if len(texts) >= max_nodes:
            break
        texts.append(str(row["clean_title"]))
        node_subs.append(sub_to_id[row["subreddit"]])
        urls.append(str(row["image_url"]))
    node_subs_arr = np.array(node_subs)
    n = len(texts)

    # Disk-cache downloaded images by a hash of their URL, so a crashed/
    # interrupted run doesn't have to re-fetch from the network on retry
    # (image downloading is by far the slowest, most failure-prone part of
    # this loader - verified directly: a prior run lost ~750s of downloads
    # to an out-of-memory kill in a later pipeline stage, unrelated to the
    # download itself).
    import hashlib
    cache_dir = os.path.join(root_dir, "image_cache")
    os.makedirs(cache_dir, exist_ok=True)

    images: List[Optional[object]] = [None] * n
    n_ok = n_fail = n_attempts = n_cached = 0
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 (research data collection; nf_mcd dataset loader)"}
    t0 = time.monotonic()
    for i, url in enumerate(urls):
        cache_path = os.path.join(cache_dir, hashlib.sha256(url.encode("utf-8")).hexdigest() + ".jpg")
        if os.path.isfile(cache_path):
            try:
                images[i] = Image.open(cache_path).convert("RGB")
                n_ok += 1
                n_cached += 1
                continue
            except Exception:
                pass  # corrupt cache entry, fall through to re-download

        if n_attempts >= max_download_attempts or (time.monotonic() - t0) > max_download_seconds:
            logger.info("Fakeddit: stopping image downloads early (attempt/time budget exhausted)")
            break
        n_attempts += 1
        try:
            resp = session.get(url, headers=headers, timeout=request_timeout)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img.save(cache_path, format="JPEG")
            images[i] = img
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 - dead links/timeouts are expected at scale, not fatal
            n_fail += 1
            logger.debug("Fakeddit: failed to download image for row %d (%s): %s", i, url, exc)

    fresh_ok = n_ok - n_cached
    success_rate = n_ok / (n_ok + n_fail) if (n_ok + n_fail) else 0.0
    logger.info(
        "Fakeddit: %d/%d images available (%d from disk cache, %d freshly downloaded of %d "
        "fresh attempts = %.1f%% success, %d rows never attempted)",
        n_ok, n, n_cached, fresh_ok, n_attempts, 100 * success_rate, n - n_attempts - n_cached,
    )
    if success_rate < min_success_rate_for_images:
        logger.warning(
            "Fakeddit: image download success rate %.1f%% is below min_success_rate_for_images=%.0f%%; "
            "discarding all downloaded images and falling back to text-only for this run.",
            100 * success_rate, 100 * min_success_rate_for_images,
        )
        images = [None] * n

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(_sbm_like_edges(node_subs_arr, p_in, p_out, rng).tolist())

    true_communities: List[Set[int]] = [{int(c)} for c in node_subs_arr]

    return RealMultimodalData(
        G=G,
        texts=texts,
        images=images,
        true_communities=true_communities,
        n_communities=len(subs),
        community_names=subs,
        dataset_name="fakeddit",
    )


_SNAP_URLS = {
    "dblp": {
        "ungraph": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
        "cmty": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.top5000.cmty.txt.gz",
    },
    "amazon": {
        "ungraph": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz",
        "cmty": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.top5000.cmty.txt.gz",
    },
}


def _download_if_missing(url: str, dest_path: str) -> None:
    import requests

    if os.path.isfile(dest_path):
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest_path)
    resp = requests.get(url, timeout=60, stream=True)
    resp.raise_for_status()
    tmp_path = dest_path + ".part"
    with open(tmp_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
    os.replace(tmp_path, dest_path)


def load_snap_community(
    dataset: str,
    root_dir: Optional[str] = None,
    n_communities: int = 8,
    min_community_size: int = 30,
    max_community_size: int = 300,
    seed: int = 42,
) -> RealMultimodalData:
    """SNAP ground-truth community benchmarks (Yang & Leskovec, ICDM 2012 /
    MDS 2013) - `dataset` is "dblp" (com-DBLP, collaboration network) or
    "amazon" (com-Amazon, product co-purchasing network). Downloaded from
    https://snap.stanford.edu/data/com-DBLP.html /
    https://snap.stanford.edu/data/com-Amazon.html into `root_dir` (default:
    `<package>/../data/<dataset>`).

    Unlike CrisisMMD/PHEME/Fakeddit, this is a **pure-topology benchmark**:
    there is no per-node text or image content anywhere in the release.
    `texts`/`images` are therefore `[None] * n` for every node by
    construction - this is the intended graph-only / total-missing-content
    case, exercising `nf_mcd.fuzzy_fusion.NeuroFuzzyFusion`'s dataset-wide
    "none" fallback (see the warning it raises) and NF-MCD's structure-only
    degradation path (`nf_mcd.topology.compute_alpha` pins every node's
    alpha near `alpha_min`), not a partial-missingness case.

    The full graphs (~300k-2M nodes) are far too large for this pipeline's
    dense fuzzy c-means / spectral-eigendecomposition machinery, so this
    loader subsamples: from the official "top 5000 communities" ground-truth
    file, it selects `n_communities` communities whose size falls in
    [`min_community_size`, `max_community_size`] (a band chosen to avoid
    both trivial 3-node communities and communities so large they'd dominate
    the induced subgraph), preferring a spread across the size-sorted list
    rather than the `n_communities` largest (which tend to cluster near the
    same size and heavily overlap in SNAP's community files) - it walks the
    size-sorted candidate list at an even stride and keeps the first
    `n_communities` picks whose *new* (not-yet-selected) node contribution is
    non-trivial, to reduce redundant overlap. The induced subgraph on the
    union of selected communities' member nodes is returned (isolated,
    degree-0 nodes dropped after induction); `true_communities` is each
    kept node's set of selected-community indices (supports the overlap
    that SNAP's ground truth actually has - a node can belong to more than
    one selected community).
    """
    if dataset not in _SNAP_URLS:
        raise ValueError(f"dataset must be one of {sorted(_SNAP_URLS)}, got {dataset!r}")

    root_dir = root_dir or os.path.join(DEFAULT_DATA_ROOT, dataset)
    os.makedirs(root_dir, exist_ok=True)

    ungraph_gz = os.path.join(root_dir, f"com-{dataset}.ungraph.txt.gz")
    cmty_gz = os.path.join(root_dir, f"com-{dataset}.top5000.cmty.txt.gz")
    _download_if_missing(_SNAP_URLS[dataset]["ungraph"], ungraph_gz)
    _download_if_missing(_SNAP_URLS[dataset]["cmty"], cmty_gz)

    import gzip

    logger.info("SNAP %s: parsing edge list...", dataset)
    edges = []
    with gzip.open(ungraph_gz, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            u, v = line.split()
            edges.append((int(u), int(v)))

    logger.info("SNAP %s: parsing top-5000 ground-truth communities...", dataset)
    communities = []
    with gzip.open(cmty_gz, "rt") as fh:
        for line in fh:
            members = [int(x) for x in line.split()]
            if members:
                communities.append(members)

    sizes = np.array([len(c) for c in communities])
    candidate_idx = [i for i in range(len(communities)) if min_community_size <= sizes[i] <= max_community_size]
    candidate_idx.sort(key=lambda i: sizes[i])
    if len(candidate_idx) < n_communities:
        raise ValueError(
            f"SNAP {dataset}: only {len(candidate_idx)} communities fall in "
            f"[{min_community_size}, {max_community_size}] nodes; need at least "
            f"{n_communities}. Widen the size band."
        )

    # Even stride across the size-sorted candidates, skipping any pick whose
    # member set is >=90% already covered by previously-selected communities
    # (SNAP's ground-truth communities overlap heavily; a near-duplicate
    # pick would barely grow the induced subgraph while still eating one of
    # the n_communities slots).
    # Connected selection: seed with the median-size candidate, then repeatedly
    # add the candidate with the most edges into the already-selected nodes
    # (requiring >=30% new nodes), so the induced subgraph is one connected
    # piece with genuinely adjacent/overlapping communities instead of
    # `n_communities` isolated components.
    adj: Dict[int, Set[int]] = {}
    cand_nodes = {v for i in candidate_idx for v in communities[i]}
    for u, v in edges:
        if u in cand_nodes and v in cand_nodes:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

    def _internal_connected_frac(members: Set[int]) -> float:
        sub = nx.Graph()
        sub.add_nodes_from(members)
        for v in members:
            for w in adj.get(v, ()):
                if w in members:
                    sub.add_edge(v, w)
        return len(max(nx.connected_components(sub), key=len)) / len(members)

    connected_idx = [i for i in candidate_idx if _internal_connected_frac(set(communities[i])) >= 0.8]
    if len(connected_idx) >= n_communities:
        candidate_idx = connected_idx

    node_comms: Dict[int, List[int]] = {}
    for i in candidate_idx:
        for v in communities[i]:
            node_comms.setdefault(v, []).append(i)

    def _touching(i: int) -> int:
        members = set(communities[i])
        touched = {j for v in members for j in node_comms.get(v, ()) if j != i}
        touched |= {j for v in members for w in adj.get(v, ()) for j in node_comms.get(w, ()) if j != i}
        return len(touched)

    seed_i = max(candidate_idx, key=_touching)
    selected: List[int] = [seed_i]
    covered: Set[int] = set(communities[seed_i])
    while len(selected) < n_communities:
        best_i, best_score = None, 0
        for i in candidate_idx:
            if i in selected:
                continue
            members = set(communities[i])
            if len(members - covered) / len(members) < 0.15:
                continue
            score = sum(len(adj.get(v, set()) & covered) for v in members - covered)
            score += 5 * len(members & covered)
            if score > best_score:
                best_i, best_score = i, score
        if best_i is None:
            break
        selected.append(best_i)
        covered |= set(communities[best_i])
    # Fall back to filling any remaining slots from unused candidates if the
    # stride walk (with its overlap guard) came up short.
    if len(selected) < n_communities:
        logger.warning(
            "SNAP %s: only %d mutually-connected communities found (wanted %d); "
            "widen the size band for more.", dataset, len(selected), n_communities,
        )

    kept_nodes = sorted(covered)
    node_to_idx = {node: i for i, node in enumerate(kept_nodes)}

    G = nx.Graph()
    G.add_nodes_from(range(len(kept_nodes)))
    for u, v in edges:
        if u in node_to_idx and v in node_to_idx:
            G.add_edge(node_to_idx[u], node_to_idx[v])
    largest_cc = max(nx.connected_components(G), key=len)
    isolated = [n for n in G.nodes() if n not in largest_cc]
    G.remove_nodes_from(isolated)
    # Relabel again to a dense 0..n-1 range after dropping isolated nodes.
    remaining = sorted(G.nodes())
    relabel = {old: new for new, old in enumerate(remaining)}
    G = nx.relabel_nodes(G, relabel)

    true_communities: List[Set[int]] = [set() for _ in remaining]
    for rank, comm_idx in enumerate(selected):
        for raw_node in communities[comm_idx]:
            idx0 = node_to_idx.get(raw_node)
            if idx0 is not None and idx0 in relabel:
                true_communities[relabel[idx0]].add(rank)

    n = len(remaining)
    logger.info(
        "SNAP %s: selected %d/%d candidate communities (sizes %d-%d nodes), "
        "induced subgraph has %d nodes / %d edges after dropping %d isolated nodes",
        dataset, len(selected), len(candidate_idx),
        min(sizes[i] for i in selected), max(sizes[i] for i in selected),
        n, G.number_of_edges(), len(isolated),
    )

    return RealMultimodalData(
        G=G,
        texts=[None] * n,
        images=[None] * n,
        true_communities=true_communities,
        n_communities=len(selected),
        community_names=[f"community_{i}" for i in range(len(selected))],
        dataset_name=f"snap_{dataset}",
    )


def load_pheme(root_dir: Optional[str] = None, max_nodes: int = 2500, seed: int = 42) -> RealMultimodalData:
    """PHEME (Zubiaga et al., 2016), "PHEME dataset for Rumour Detection and
    Veracity Classification" (figshare 6392078). 9 breaking-news events,
    each with rumour/non-rumour Twitter conversation threads. Unlike
    CrisisMMD, this dataset ships a REAL native graph: `structure.json` per
    thread gives the actual reply tree (source tweet -> reactions), which
    this loader uses directly as edges - no synthetic/reconstructed
    topology here. No images are included in the release, so this is used
    as the text-only / vision-ablation setting the docstring originally
    called for: `images` is `[None] * n` for every node, letting NF-MCD's
    documented graceful-degradation path (confidence=0, alpha pinned
    toward structure) do the work.

    Expects the extracted PHEME_veracity.tar.bz2 archive (i.e. a
    `root_dir/all-rnr-annotated-threads/<event>-all-rnr-threads/{rumours,
    non-rumours}/<tweet_id>/{source-tweets,reactions,structure.json,...}`
    layout) at `root_dir` (default: `<package>/../data/pheme`).

    `true_communities` = the 9 breaking-news events (event id), since
    that's the only dataset-provided grouping with enough per-group size to
    be a meaningful "community" (rumour-vs-non-rumour is only 2 classes and
    highly imbalanced per event). Threads are sampled round-robin across
    events (up to `max_nodes` total tweets, counting every node in a
    thread's reply tree, not just the source tweet) so smaller events
    aren't crowded out by large ones (e.g. charliehebdo/ferguson are much
    bigger than gurlitt/putinmissing). The resulting graph is a forest of
    many small reply trees (one per sampled thread) rather than one
    densely-connected component - this is the real propagation structure,
    not an artifact of the loader.
    """
    import json

    root_dir = root_dir or os.path.join(DEFAULT_DATA_ROOT, "pheme")
    events_root = os.path.join(root_dir, "all-rnr-annotated-threads")
    if not os.path.isdir(events_root):
        raise FileNotFoundError(
            f"Expected extracted PHEME data at {events_root!r}. Download "
            f"PHEME_veracity.tar.bz2 from figshare (article 6392078) and "
            f"extract it under {root_dir!r} first."
        )

    def _read_json_utf8(path):
        with open(path, "rb") as fh:
            raw = fh.read()
        try:
            return json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError:
            return json.loads(raw.decode("utf-8", errors="replace"))

    event_dirs = sorted(d for d in os.listdir(events_root) if os.path.isdir(os.path.join(events_root, d)))
    event_to_id = {d: i for i, d in enumerate(event_dirs)}

    rng = np.random.default_rng(seed)

    G = nx.Graph()
    texts: dict = {}
    node_event: dict = {}

    def load_thread(thread_dir: str, event_idx: int, budget: int) -> int:
        struct_path = os.path.join(thread_dir, "structure.json")
        if not os.path.isfile(struct_path) or budget <= 0:
            return 0
        try:
            struct = _read_json_utf8(struct_path)
        except Exception:
            return 0

        tweet_texts = {}
        for sub in ("source-tweets", "reactions"):
            d = os.path.join(thread_dir, sub)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if not fn.endswith(".json") or fn.startswith("."):
                    continue
                try:
                    tw = _read_json_utf8(os.path.join(d, fn))
                    tweet_texts[str(tw["id"])] = tw.get("text", "")
                except Exception:
                    continue

        added = 0

        def walk(node, parent_key):
            nonlocal added
            for tid, sub in node.items():
                next_parent = parent_key
                if tid in tweet_texts and tid not in texts and added < budget:
                    texts[tid] = tweet_texts[tid]
                    node_event[tid] = event_idx
                    G.add_node(tid)
                    if parent_key is not None and parent_key in texts:
                        G.add_edge(parent_key, tid)
                    next_parent = tid
                    added += 1
                elif tid in texts:
                    if parent_key is not None and parent_key in texts:
                        G.add_edge(parent_key, tid)
                    next_parent = tid
                if isinstance(sub, dict) and sub:
                    walk(sub, next_parent)

        walk(struct, None)
        return added

    per_event_threads = {ed: [] for ed in event_dirs}
    for ed in event_dirs:
        for label in ("rumours", "non-rumours"):
            ldir = os.path.join(events_root, ed, label)
            if not os.path.isdir(ldir):
                continue
            for tid_dir in os.listdir(ldir):
                p = os.path.join(ldir, tid_dir)
                if os.path.isdir(p):
                    per_event_threads[ed].append(p)
    for ed in event_dirs:
        rng.shuffle(per_event_threads[ed])

    iters = {ed: iter(per_event_threads[ed]) for ed in event_dirs}
    active = True
    while len(texts) < max_nodes and active:
        active = False
        for ed in event_dirs:
            if len(texts) >= max_nodes:
                break
            try:
                thread_dir = next(iters[ed])
            except StopIteration:
                continue
            active = True
            load_thread(thread_dir, event_to_id[ed], budget=max_nodes - len(texts))

    node_ids = list(texts.keys())
    mapping = {tid: i for i, tid in enumerate(node_ids)}
    G = nx.relabel_nodes(G.subgraph(node_ids).copy(), mapping)

    texts_list: List[Optional[str]] = [texts[tid] for tid in node_ids]
    true_communities: List[Set[int]] = [{node_event[tid]} for tid in node_ids]

    return RealMultimodalData(
        G=G,
        texts=texts_list,
        images=[None] * len(node_ids),
        true_communities=true_communities,
        n_communities=len(event_dirs),
        community_names=event_dirs,
        dataset_name="pheme",
    )
