"""
nf_mcd.pipeline.NFMCD: the end-to-end Neuro-Fuzzy Multimodal Community
Detection method, wiring together Sections 4.1-4.5 of the paper behind a
single fit/predict interface, in the spirit of a scikit-learn estimator.

Typical usage
-------------
>>> from nf_mcd import NFMCD
>>> model = NFMCD(n_communities=4)
>>> model.fit(G, text_embeddings=text_embeds, image_embeddings=image_embeds)
>>> hard_labels = model.predict_hard()
>>> print(model.explain_node(node_id=7).text)
>>> for rule in model.global_rules():
...     print(rule.text)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import networkx as nx
import numpy as np

from . import community_detection as cd
from . import explain as expl
from . import metrics as mx
from . import topology as topo
from .encoders import MultimodalEncoder
from .fuzzy_fusion import ANFISAgreement, NeuroFuzzyFusion


class NFMCD:
    """Neuro-Fuzzy Multimodal Community Detection.

    Parameters
    ----------
    n_communities : int
        Number of communities k (see `nf_mcd.community_detection.select_k_by_fpc`
        for an unsupervised heuristic to help choose this).
    common_dim : int
        Maximum number of CCA components for the learned shared text/image
        space (Section 4.2); the effective dimensionality is capped by the
        smaller modality's raw dimension and by the number of nodes with
        both modalities present (see nf_mcd.fuzzy_fusion.NeuroFuzzyFusion).
    structural_dim : int, optional
        Dimensionality of the spectral structural embedding (Section 4.3).
        Defaults to `n_communities` if not given, following classical
        spectral clustering practice (use k eigenvectors for k target
        clusters) - see the warning in nf_mcd.topology.compute_structural_embedding
        about why a much larger value silently degrades separability.
    fcm_m : float
        Fuzzy c-means fuzziness exponent (Section 4.4). Defaults to 1.5,
        not the generic-FCM textbook default of 2.0 - see the extended
        rationale in nf_mcd.community_detection.FuzzyCMeans.
    alpha_min, alpha_max : float
        Bounds on the per-node content-vs-structure trust weight (Section 4.3).
    anfis : ANFISAgreement, optional
        Custom fuzzy-agreement antecedent/consequent parameters; defaults
        to the three-term (Low/Medium/High) ANFIS in nf_mcd.fuzzy_fusion.
    encoder : MultimodalEncoder, optional
        Custom text/image encoder pair; defaults to sentence-transformers +
        CLIP with a deterministic fallback (see nf_mcd.encoders).
    seed : int
        Random seed for the fixed projections, structural embedding
        fallback, and fuzzy c-means initialization.
    """

    def __init__(
        self,
        n_communities: int,
        common_dim: int = 8,
        structural_dim: Optional[int] = None,
        fcm_m: float = 1.5,
        alpha_min: float = 0.15,
        alpha_max: float = 0.85,
        anfis: Optional[ANFISAgreement] = None,
        encoder: Optional[MultimodalEncoder] = None,
        seed: int = 0,
        confidence_mode: str = "agreement",
        pca_rank_div: int = 4,
        single_modality_fill: str = "raw",
        content_features: str = "cca",
        raw_pca_dim: int = 16,
        standardize_blocks: bool = False,
        clusterer: str = "fcm",
        vectorized_fusion: bool = False,
    ):
        if clusterer not in ("fcm", "kmeans_softmax", "gmm", "fcm_adaptive_m", "spectral_soft"):
            raise ValueError("clusterer must be 'fcm', 'kmeans_softmax', 'gmm', 'fcm_adaptive_m' or 'spectral_soft'")
        self.clusterer = clusterer
        if confidence_mode not in ("agreement", "consistency", "blend"):
            raise ValueError("confidence_mode must be 'agreement', 'consistency' or 'blend'")
        if content_features not in ("cca", "raw_pca"):
            raise ValueError("content_features must be 'cca' or 'raw_pca'")
        self.content_features = content_features
        self.raw_pca_dim = raw_pca_dim
        self.standardize_blocks = standardize_blocks
        self.confidence_mode = confidence_mode
        self.n_communities = n_communities
        self.common_dim = common_dim
        self.structural_dim = structural_dim if structural_dim is not None else n_communities
        self.fcm_m = fcm_m
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.seed = seed

        self.anfis = anfis or ANFISAgreement()
        self.encoder = encoder  # lazily constructed only if raw text/images are passed
        self.vectorized_fusion = vectorized_fusion
        self.fusion = NeuroFuzzyFusion(
            common_dim=common_dim, anfis=self.anfis, seed=seed,
            pca_rank_div=pca_rank_div, single_modality_fill=single_modality_fill,
            vectorized=vectorized_fusion,
        )

        # Populated by fit():
        self.G_: Optional[nx.Graph] = None
        self.nodes_: List = []
        self._node_to_idx: Dict = {}
        self.U_: Optional[np.ndarray] = None
        self.confidence_: Optional[np.ndarray] = None
        self.alpha_: Optional[np.ndarray] = None
        self.agreement_: Optional[np.ndarray] = None
        self.modality_flags_: List[str] = []
        self.Z_s_: Optional[np.ndarray] = None
        self.fused_content_: Optional[np.ndarray] = None
        self.fcm_result_: Optional[cd.FCMResult] = None

    @classmethod
    def robust(cls, n_communities: int, **kwargs) -> "NFMCD":
        """Opt-in preset that avoids the fuzzy c-means collapse seen with noisy
        structure (adaptive-m fuzzy c-means on raw-PCA content) and the CCA
        overfit that inflates cross-modal agreement (PCA rank cap n_paired // 16
        instead of // 4), plus the vectorized fusion path (verified bit-for-bit
        equivalent to the default loop, ~20x faster at n=20,000; see
        experiments/vectorize_summary.log). Defaults of the plain constructor
        are unchanged. See experiments/clusterer_summary.log and
        experiments/rank_summary.log."""
        kwargs.setdefault("clusterer", "fcm_adaptive_m")
        kwargs.setdefault("content_features", "raw_pca")
        kwargs.setdefault("pca_rank_div", 16)
        kwargs.setdefault("vectorized_fusion", True)
        return cls(n_communities=n_communities, **kwargs)

    def fit(
        self,
        G: nx.Graph,
        texts: Optional[Sequence[Optional[str]]] = None,
        images: Optional[Sequence[Optional[object]]] = None,
        text_embeddings: Optional[Sequence[Optional[np.ndarray]]] = None,
        image_embeddings: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> "NFMCD":
        """Fit NF-MCD on graph `G`.

        Content can be supplied either as raw `texts`/`images` (encoded
        internally via nf_mcd.encoders.MultimodalEncoder) or as precomputed
        `text_embeddings`/`image_embeddings` (e.g. cached CLIP/text-encoder
        outputs, or the synthetic embeddings from nf_mcd.datasets). Exactly
        one of the two input styles should be provided per modality; `None`
        entries within a sequence denote a missing modality for that node.

        All sequences must be aligned with `list(G.nodes())` order.
        """
        self.G_ = G
        self.nodes_ = list(G.nodes())
        self._node_to_idx = {node: i for i, node in enumerate(self.nodes_)}
        n = len(self.nodes_)

        use_raw = texts is not None or images is not None
        use_precomputed = text_embeddings is not None or image_embeddings is not None
        if use_raw and use_precomputed:
            raise ValueError(
                "Pass either raw texts/images OR precomputed text_embeddings/"
                "image_embeddings, not both."
            )

        if use_raw:
            texts = texts if texts is not None else [None] * n
            images = images if images is not None else [None] * n
            if self.encoder is None:
                self.encoder = MultimodalEncoder()
            e_t, e_v = self.encoder.encode(texts, images)
        elif use_precomputed:
            e_t = list(text_embeddings) if text_embeddings is not None else [None] * n
            e_v = list(image_embeddings) if image_embeddings is not None else [None] * n
        else:
            raise ValueError("Provide at least one of texts/images or text_embeddings/image_embeddings.")

        assert len(e_t) == n and len(e_v) == n, "Content sequences must align with G.nodes()."

        # Stage 2: neuro-fuzzy fusion.
        fusion_result = self.fusion.fuse(e_t, e_v)
        self.fused_content_ = fusion_result.fused
        self.confidence_ = fusion_result.confidence
        self.agreement_ = fusion_result.agreement
        self.modality_flags_ = fusion_result.modality_flags

        if self.confidence_mode != "agreement":
            cons = topo.consistency_confidence(G, self.nodes_, self.fused_content_, self.modality_flags_)
            self.confidence_ = cons if self.confidence_mode == "consistency" else 0.5 * (self.confidence_ + cons)

        # Stage 3: structural embedding + fuzzy content/structure integration.
        self.Z_s_ = topo.compute_structural_embedding(G, dim=self.structural_dim, seed=self.seed)
        self.alpha_ = topo.compute_alpha(self.confidence_, self.alpha_min, self.alpha_max)
        Z_c = self.fused_content_
        if self.content_features == "raw_pca":
            Z_c = topo.raw_pca_content(e_t, e_v, dim=self.raw_pca_dim, seed=self.seed)
        fused_features = topo.fuse_features(Z_c, self.Z_s_, self.alpha_, standardize_blocks=self.standardize_blocks)
        self.fused_features_ = fused_features
        self.n_content_cols_ = Z_c.shape[1]
        self._sens_cache = None

        # Stage 4: fuzzy c-means soft/overlapping community detection.
        if self.clusterer == "fcm":
            fcm = cd.FuzzyCMeans(n_clusters=self.n_communities, m=self.fcm_m, seed=self.seed)
            self.fcm_result_ = fcm.fit(fused_features)
        else:
            from . import clusterers
            self.fcm_result_ = clusterers.cluster(fused_features, self.n_communities, self.clusterer, self.seed)
        self.U_ = self.fcm_result_.U

        return self

    # -- Stage 4 accessors -------------------------------------------------

    def predict_hard(self) -> np.ndarray:
        self._check_fitted()
        return cd.defuzzify(self.U_)

    def overlapping_communities(self, threshold: float = 0.2) -> List[Set[int]]:
        """Per-node view: for each node, the set of community indices it
        belongs to with membership >= threshold."""
        self._check_fitted()
        return cd.overlapping_communities(self.U_, threshold)

    def overlapping_communities_view(self, threshold: float = 0.2) -> List[Set[int]]:
        """Per-community view: for each community, the set of node indices
        belonging to it with membership >= threshold. Matches the input
        format expected by nf_mcd.metrics.overlapping_nmi / membership_f1."""
        node_view = self.overlapping_communities(threshold)
        comm_view: List[Set[int]] = [set() for _ in range(self.n_communities)]
        for node_idx, comms in enumerate(node_view):
            for c in comms:
                comm_view[c].add(node_idx)
        return comm_view

    # -- Stage 5 accessors (explainability) ---------------------------------

    def explain_node(self, node_id, top_k_other: int = 2) -> expl.NodeExplanation:
        self._check_fitted()
        idx = self._node_to_idx[node_id]
        return expl.explain_node(
            node_id=node_id,
            node_index=idx,
            U=self.U_,
            confidence=self.confidence_,
            alpha=self.alpha_,
            agreement=self.agreement_,
            modality_flags=self.modality_flags_,
            anfis=self.anfis,
            top_k_other=top_k_other,
        )

    def global_rules(self, min_support: int = 3) -> List[expl.GlobalRule]:
        self._check_fitted()
        return expl.extract_global_rules(self.U_, self.confidence_, self.alpha_, min_support=min_support)

    def rule_fidelity(self, min_support: int = 3) -> float:
        rules = self.global_rules(min_support=min_support)
        return expl.rule_fidelity(rules, self.U_, self.confidence_, self.alpha_)

    # -- Neighbourhood explanations and measured block sensitivity -----------
    # (additions; the alpha-based methods above are unchanged). alpha is an input
    # weight in the feature fusion, not a causal measure of how much content or
    # structure influenced an assignment: use content_sensitivity() for that.

    def _block_sensitivity(self) -> Dict[str, np.ndarray]:
        self._check_fitted()
        if getattr(self, "_sens_cache", None) is None:
            if self.clusterer not in ("fcm", "fcm_adaptive_m"):
                raise NotImplementedError(
                    "block sensitivity recomputes fuzzy c-means memberships at fixed centres; "
                    f"clusterer={self.clusterer!r} is not supported"
                )
            m = getattr(self.fcm_result_, "m_used", self.fcm_m)
            self._sens_cache = expl.block_sensitivity(
                self.U_, self.fused_features_, self.fcm_result_.centers, m, self.n_content_cols_
            )
        return self._sens_cache

    def content_flip_margin(self) -> np.ndarray:
        """Per node: best rival's membership minus the node's own community's membership
        after zeroing the content block at the fitted centres. > 0 means the assignment
        flips without content. Predicts which nodes flip in an actual content-less refit
        (AUC 0.68-0.93 on the content datasets); used in the explanation text."""
        return self._block_sensitivity()["margin_no_content"]

    def structure_flip_margin(self) -> np.ndarray:
        """As content_flip_margin, for zeroing the structure block."""
        return self._block_sensitivity()["margin_no_structure"]

    def content_sensitivity(self) -> np.ndarray:
        """Per node in [0, 1]: total-variation shift of the memberships when the content
        block is zeroed at the fitted centres. 0 for nodes without content. This is a
        magnitude of change, NOT a flip predictor (it ranks confident nodes highest, and
        was worse than chance at predicting content-less refit flips): use
        content_flip_margin() for that."""
        return self._block_sensitivity()["tv_content"]

    def structure_sensitivity(self) -> np.ndarray:
        """Per node in [0, 1]: total-variation shift when the structure block is zeroed."""
        return self._block_sensitivity()["tv_structure"]

    def _neighbour_shares(self):
        return expl.neighbour_shares(self.G_, self.nodes_, self.U_)

    def explain_node_neighbourhood(self, node_id) -> expl.NeighbourhoodExplanation:
        self._check_fitted()
        deg, counts, wcounts = self._neighbour_shares()
        return expl.explain_node_neighbourhood(
            node_id, self._node_to_idx[node_id], self.U_, deg, counts, wcounts,
            self._block_sensitivity(), self.modality_flags_,
        )

    def global_neighbourhood_rules(self, min_support: int = 5) -> List[expl.NeighbourhoodRule]:
        """One rule per community: IF >= t of a node's neighbours are in community c THEN
        the node is in community c (see nf_mcd.explain.extract_neighbourhood_rules)."""
        self._check_fitted()
        deg, counts, _ = self._neighbour_shares()
        return expl.extract_neighbourhood_rules(deg, counts, self.predict_hard(), min_support=min_support)

    def neighbourhood_rule_fidelity(self, min_support: int = 5) -> Dict[str, float]:
        """Fidelity, coverage, precision on covered nodes and majority baseline of the
        neighbourhood rules; fidelity_with_fallback falls back to the content-only community."""
        self._check_fitted()
        deg, counts, _ = self._neighbour_shares()
        hard = self.predict_hard()
        rules = expl.extract_neighbourhood_rules(deg, counts, hard, min_support=min_support)
        sens = self._block_sensitivity()
        has_content = np.array([f != "none" for f in self.modality_flags_])
        fb = np.where(has_content, sens["content_only_top"], np.bincount(hard).argmax())
        return expl.neighbourhood_rule_fidelity(rules, deg, counts, hard, fallback=fb)

    # -- Evaluation ----------------------------------------------------------

    def evaluate(
        self,
        true_communities_per_node: Optional[List[Set[int]]] = None,
        overlap_threshold: float = 0.2,
    ) -> Dict[str, float]:
        """Compute modularity (always) and, if ground truth is supplied,
        overlapping NMI and membership F1 (Section 5.3)."""
        self._check_fitted()
        results = {"modularity": mx.modularity_score(self.G_, self.predict_hard())}

        if true_communities_per_node is not None:
            true_view: List[Set[int]] = [set() for _ in range(self.n_communities)]
            for node_idx, comms in enumerate(true_communities_per_node):
                for c in comms:
                    if 0 <= c < self.n_communities:
                        true_view[c].add(node_idx)
            pred_view = self.overlapping_communities_view(threshold=overlap_threshold)

            results["overlapping_nmi"] = mx.overlapping_nmi(pred_view, true_view, n_nodes=len(self.nodes_))
            results["membership_f1"] = mx.membership_f1(pred_view, true_view)

        results["rule_fidelity"] = self.rule_fidelity()
        return results

    def _check_fitted(self):
        if self.U_ is None:
            raise RuntimeError("NFMCD.fit(...) must be called before this method.")
