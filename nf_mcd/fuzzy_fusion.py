"""
Stage 2 (Section 4.2): Neuro-fuzzy fusion layer.

Rather than naively concatenating or averaging modality embeddings (which
cannot express *agreement* or *conflict* between modalities), this module
computes:

  1. A cross-modal agreement score s_i in [-1, 1]: cosine similarity between
     the text and image embeddings after they are projected into a shared
     space.
  2. A fuzzy confidence c_i in [0, 1] for that agreement, produced by a small
     Takagi-Sugeno (order-0) ANFIS with three linguistic terms - Low,
     Medium, High agreement - so that c_i is a smooth, interpretable
     function of s_i rather than a black-box score. c_i is low both when
     s_i is genuinely low (image/caption mismatch) *and* when a modality is
     missing entirely.
  3. A fused content embedding per node, combining whatever modalities are
     actually present.

The antecedent (membership-function) parameters are the "rules" referenced
in Section 4.5 / nf_mcd.explain: e.g. "IF agreement is High THEN
confidence is High", each with a legible center/width in similarity space.

A note on how the shared space is obtained
-------------------------------------------
Computing a meaningful cosine similarity between a text embedding and an
image embedding requires the two to already live in - or be mapped into -
a genuinely *shared* space. Two independently-trained encoders (e.g. a
general-purpose sentence encoder plus a separately-trained image encoder)
do NOT produce comparable vectors, and critically, an arbitrary *fixed
random* projection cannot manufacture alignment that was never there: a
random linear map applied independently to each modality destroys any
shared signal rather than recovering it (verified empirically during
development of this module - see the project notes). This module
therefore fits a Canonical Correlation Analysis (CCA) on the paired nodes
(those with both modalities present) to LEARN a shared space in which
cross-modal correlation is maximized, and reuses that fitted mapping for
every node. This still assumes some real cross-modal correlation exists in
the data to learn from (true for embeddings from a jointly-trained model
such as CLIP, and for our synthetic demo data); if the two encoders truly
share no signal, CCA will correctly find none; the graceful degradation in
that case is low estimated confidence and reliance on network structure.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA


def _gaussian_membership(x: np.ndarray, center: float, width: float) -> np.ndarray:
    width = max(width, 1e-6)
    return np.exp(-0.5 * ((x - center) / width) ** 2)


@dataclass
class ANFISAgreement:
    """A minimal order-0 Takagi-Sugeno ANFIS over a single input (cosine similarity).

    Three linguistic terms are used for the antecedent (Low / Medium / High
    agreement), each a Gaussian membership function, with fixed zero-order
    Sugeno consequents. Centers/widths are exposed as public attributes so
    they can be (a) tuned by hand, (b) fit to labeled data by an external
    optimizer, or (c) read directly by nf_mcd.explain to print rules.
    """

    # Calibrated for cosine similarity in the *learned* (CCA) shared space,
    # which - because CCA explicitly maximizes correlation - tends to sit
    # well above 0 even for weakly-related pairs; genuinely mismatched
    # pairs still separate clearly below well-aligned ones (validated
    # empirically on synthetic data with known ground truth; re-calibrate
    # centers/widths for your own dataset by inspecting the empirical
    # distribution of `NeuroFuzzyFusion.fuse(...).agreement`).
    centers: np.ndarray = field(default_factory=lambda: np.array([0.35, 0.65, 0.88]))
    widths: np.ndarray = field(default_factory=lambda: np.array([0.30, 0.18, 0.15]))
    consequents: np.ndarray = field(default_factory=lambda: np.array([0.10, 0.50, 0.90]))
    term_names: tuple = ("Low", "Medium", "High")

    def membership(self, s: np.ndarray) -> np.ndarray:
        """Return firing strengths, shape (n, 3), one column per linguistic term."""
        s = np.atleast_1d(s)
        mus = np.stack(
            [_gaussian_membership(s, c, w) for c, w in zip(self.centers, self.widths)],
            axis=-1,
        )
        return mus

    def infer(self, s: np.ndarray) -> np.ndarray:
        """Defuzzified confidence c in [0, 1] for each similarity value in s."""
        mus = self.membership(s)                      # (n, 3)
        weights = mus / (mus.sum(axis=-1, keepdims=True) + 1e-12)
        c = weights @ self.consequents                # (n,)
        return np.clip(c, 0.0, 1.0)

    def dominant_rule(self, s: float) -> str:
        """Human-readable dominant rule for a single similarity value (for explain.py)."""
        mus = self.membership(np.array([s]))[0]
        k = int(np.argmax(mus))
        return (
            f"IF cross-modal agreement is {self.term_names[k]} "
            f"(similarity {s:.2f}, membership {mus[k]:.2f}) "
            f"THEN fusion confidence is ~{self.consequents[k]:.2f}"
        )


@dataclass
class FusionResult:
    fused: np.ndarray                 # (n, d) fused content embedding, zero row if both missing
    confidence: np.ndarray            # (n,) fuzzy confidence c_i in [0, 1]
    agreement: np.ndarray             # (n,) cosine similarity s_i, NaN if not computable
    modality_flags: List[str]         # "both" | "text_only" | "image_only" | "none"


class NeuroFuzzyFusion:
    """Fuses per-node text and image embeddings into a single content vector
    with an associated fuzzy confidence, per Section 4.2 of the paper.

    Parameters
    ----------
    common_dim : int
        Maximum number of CCA components for the shared space. The
        effective number used is min(common_dim, dim_t, dim_v, n_paired - 1),
        since CCA cannot extract more components than the smaller
        modality's dimensionality or than the paired-sample count allows.
    min_paired_samples : int
        Minimum number of both-modalities-present nodes required to fit
        the CCA alignment. Below this, cross-modal agreement cannot be
        estimated reliably; confidence falls back to 0 for every node
        (pure structural reliance downstream) and a warning is raised.
    pca_rank_div : int
        The PCA rank cap before CCA is n_paired // pca_rank_div (default 4,
        the original behaviour). Larger values regularise CCA harder.
    single_modality_fill : {"raw", "zero"}
        Only matters when the CCA alignment was fitted: a node with exactly
        one modality gets its first coordinates of the raw embedding ("raw",
        original behaviour, a different space from the CCA space of paired
        nodes) or a zero content vector ("zero").
    vectorized : bool, default False
        Opt-in: use a batched (matrix-multiply-over-all-nodes) implementation
        of `fuse()`'s final assembly step instead of the original per-node
        Python loop. Verified numerically equivalent (see
        experiments/vectorize_summary.log) and much faster at large n (this
        loop was found to be ~94% of total NFMCD.fit() time at n=20,000 -
        see experiments/scale_summary.log). Default False: nothing about
        existing behaviour changes unless this is explicitly turned on.
    """

    def __init__(
        self,
        common_dim: int = 8,
        anfis: Optional[ANFISAgreement] = None,
        seed: int = 0,
        min_paired_samples: int = 10,
        pca_rank_div: int = 4,
        single_modality_fill: str = "raw",
        vectorized: bool = False,
    ):
        if single_modality_fill not in ("raw", "zero"):
            raise ValueError("single_modality_fill must be 'raw' or 'zero'")
        if pca_rank_div < 1:
            raise ValueError("pca_rank_div must be >= 1")
        self.pca_rank_div = pca_rank_div
        self.single_modality_fill = single_modality_fill
        self.vectorized = vectorized
        self.common_dim = common_dim
        self.anfis = anfis or ANFISAgreement()
        self._seed = seed
        self.min_paired_samples = min_paired_samples
        self._cca = None
        self._pca_t: Optional[PCA] = None
        self._pca_v: Optional[PCA] = None
        self._xt_mean: Optional[np.ndarray] = None
        self._xt_std: Optional[np.ndarray] = None
        self._xv_mean: Optional[np.ndarray] = None
        self._xv_std: Optional[np.ndarray] = None
        self._fitted = False

    def _fit_alignment(self, Xt_paired: np.ndarray, Xv_paired: np.ndarray) -> bool:
        """Fit a PCA-whitened CCA mapping from the nodes where both modalities
        are present.

        Raw embeddings are typically high-dimensional (e.g. 384/512) relative
        to the number of paired nodes available, especially in small or
        heavily-missing-modality datasets. Fitting CCA directly in that
        "n << p" regime produces severely overfit, spuriously high canonical
        correlations for essentially any pair of matrices, aligned or not
        (a well-known failure mode of CCA at high dimensionality with few
        samples). We therefore first reduce each modality to a safe rank via
        PCA (capped well below the paired-sample count) before fitting CCA,
        a standard regularization strategy for this setting.

        Standardization/PCA are computed and applied manually/explicitly
        (rather than relying on library-internal centering) so the fitted
        transform is self-contained. Returns False (leaving the fusion
        layer in "no alignment" fallback mode) if there are too few paired
        samples to fit reliably at all.
        """
        n_paired = Xt_paired.shape[0]
        if n_paired < self.min_paired_samples:
            warnings.warn(
                f"NeuroFuzzyFusion: only {n_paired} nodes have both modalities present "
                f"(< min_paired_samples={self.min_paired_samples}); cannot reliably fit a "
                f"cross-modal alignment. Falling back to confidence=0 for all nodes "
                f"(structure-only reliance downstream). Provide more paired nodes, or "
                f"lower min_paired_samples if you understand the risk of overfitting.",
                stacklevel=3,
            )
            return False

        self._xt_mean = Xt_paired.mean(axis=0)
        self._xt_std = Xt_paired.std(axis=0) + 1e-8
        self._xv_mean = Xv_paired.mean(axis=0)
        self._xv_std = Xv_paired.std(axis=0) + 1e-8

        Xt_std = (Xt_paired - self._xt_mean) / self._xt_std
        Xv_std = (Xv_paired - self._xv_mean) / self._xv_std

        # Cap PCA rank well below the paired-sample count to avoid overfitting.
        safe_rank = max(2, n_paired // self.pca_rank_div)
        pca_dim_t = max(1, min(safe_rank, Xt_std.shape[1], n_paired - 1))
        pca_dim_v = max(1, min(safe_rank, Xv_std.shape[1], n_paired - 1))
        self._pca_t = PCA(n_components=pca_dim_t, random_state=self._seed).fit(Xt_std)
        self._pca_v = PCA(n_components=pca_dim_v, random_state=self._seed).fit(Xv_std)
        Zt = self._pca_t.transform(Xt_std)
        Zv = self._pca_v.transform(Xv_std)

        # Standardize the PCA scores ourselves (rather than relying on
        # library-internal centering) so we can transform each side
        # independently later without touching any private CCA attributes.
        self._zt_mean, self._zt_std = Zt.mean(axis=0), Zt.std(axis=0) + 1e-8
        self._zv_mean, self._zv_std = Zv.mean(axis=0), Zv.std(axis=0) + 1e-8
        Zt_std = (Zt - self._zt_mean) / self._zt_std
        Zv_std = (Zv - self._zv_mean) / self._zv_std

        n_components = max(1, min(self.common_dim, Zt_std.shape[1], Zv_std.shape[1], n_paired - 1))
        self._cca = CCA(n_components=n_components, scale=False)
        self._cca.fit(Zt_std, Zv_std)
        self._fitted = True
        return True

    def _project_text(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self._xt_mean) / self._xt_std
        Z = self._pca_t.transform(Xs)
        Zs = (Z - self._zt_mean) / self._zt_std
        return Zs @ self._cca.x_rotations_

    def _project_image(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self._xv_mean) / self._xv_std
        Z = self._pca_v.transform(Xs)
        Zs = (Z - self._zv_mean) / self._zv_std
        return Zs @ self._cca.y_rotations_

    def fuse(
        self,
        e_t: Sequence[Optional[np.ndarray]],
        e_v: Sequence[Optional[np.ndarray]],
    ) -> FusionResult:
        n = len(e_t)
        assert len(e_v) == n

        dim_t = next((v.shape[0] for v in e_t if v is not None), None)
        dim_v = next((v.shape[0] for v in e_v if v is not None), None)
        if dim_t is None and dim_v is None:
            # Every node lacks both modalities dataset-wide (e.g. a
            # graph-only benchmark like SNAP's com-DBLP/com-Amazon, which
            # has no per-node text/image content at all). This is a
            # legitimate input, not a misconfiguration: the rest of the
            # pipeline already has a well-defined "none" path per node
            # (see nf_mcd.topology.compute_alpha, nf_mcd.explain's "none"
            # case) - apply it uniformly instead of failing the whole fit.
            warnings.warn(
                "NeuroFuzzyFusion: no node has any text or image content "
                "(both modalities entirely absent dataset-wide). Falling "
                "back to a degenerate zero content vector for every node; "
                "NFMCD will rely entirely on network structure "
                "(confidence=0, alpha pinned toward structure for all "
                "nodes). This is expected for graph-only datasets.",
                stacklevel=2,
            )
            return FusionResult(
                fused=np.zeros((n, 1)),
                confidence=np.zeros(n),
                agreement=np.full(n, np.nan),
                modality_flags=["none"] * n,
            )

        paired_idx = [i for i in range(n) if e_t[i] is not None and e_v[i] is not None]
        alignment_ok = False
        if dim_t is not None and dim_v is not None and paired_idx:
            Xt_paired = np.stack([e_t[i] for i in paired_idx])
            Xv_paired = np.stack([e_v[i] for i in paired_idx])
            alignment_ok = self._fit_alignment(Xt_paired, Xv_paired)

        out_dim = self._cca.x_rotations_.shape[1] if alignment_ok else max(dim_t or 0, dim_v or 0, 1)

        if self.vectorized:
            return self._fuse_vectorized(e_t, e_v, n, out_dim, alignment_ok)

        fused = np.zeros((n, out_dim))
        agreement = np.full(n, np.nan)
        confidence = np.zeros(n)
        flags: List[str] = []

        for i in range(n):
            t_i, v_i = e_t[i], e_v[i]
            has_t, has_v = t_i is not None, v_i is not None

            if has_t and has_v and alignment_ok:
                p_t = self._project_text(t_i[None, :])[0]
                p_v = self._project_image(v_i[None, :])[0]
                p_t = p_t / (np.linalg.norm(p_t) + 1e-12)
                p_v = p_v / (np.linalg.norm(p_v) + 1e-12)
                s_i = float(np.dot(p_t, p_v))
                agreement[i] = s_i
                confidence[i] = self.anfis.infer(np.array([s_i]))[0]
                fused[i, : p_t.shape[0]] = (p_t + p_v) / 2.0
                flags.append("both")
            elif has_t:
                if not (alignment_ok and not has_v and self.single_modality_fill == "zero"):
                    v = t_i / (np.linalg.norm(t_i) + 1e-12)
                    fused[i, : min(out_dim, v.shape[0])] = v[: out_dim]
                confidence[i] = 0.0  # no cross-modal evidence to corroborate
                flags.append("text_only" if not has_v else "both_unaligned")
            elif has_v:
                if not (alignment_ok and self.single_modality_fill == "zero"):
                    v = v_i / (np.linalg.norm(v_i) + 1e-12)
                    fused[i, : min(out_dim, v.shape[0])] = v[: out_dim]
                confidence[i] = 0.0
                flags.append("image_only")
            else:
                confidence[i] = 0.0
                flags.append("none")

        return FusionResult(fused=fused, confidence=confidence, agreement=agreement, modality_flags=flags)

    def _fill_raw_batch(self, fused: np.ndarray, idx: np.ndarray, embeddings: Sequence, out_dim: int) -> None:
        """Batched equivalent of the loop's per-node
        `v = e_i / (norm(e_i)+1e-12); fused[i, :min(out_dim, v.shape[0])] = v[:out_dim]`."""
        if len(idx) == 0:
            return
        X = np.stack([embeddings[i] for i in idx])
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        w = min(out_dim, Xn.shape[1])
        fused[np.asarray(idx)[:, None], np.arange(w)] = Xn[:, :w]

    def _fuse_vectorized(
        self,
        e_t: Sequence[Optional[np.ndarray]],
        e_v: Sequence[Optional[np.ndarray]],
        n: int,
        out_dim: int,
        alignment_ok: bool,
    ) -> "FusionResult":
        """Batched equivalent of the per-node loop in `fuse()`. Groups nodes
        by (has_t, has_v) once via boolean masks, then does one matrix
        operation per group instead of one Python-level call per node.
        Replicates the original loop's branching EXACTLY, including its
        specific choice to fill a "both present but CCA not fitted" node
        from text alone (never blending in the image) - see `fuse()`'s
        loop and experiments/vectorize_summary.log for the equivalence
        check against that loop, case by case."""
        has_t = np.array([t is not None for t in e_t])
        has_v = np.array([v is not None for v in e_v])
        idx_both = np.where(has_t & has_v)[0]
        idx_text_only = np.where(has_t & ~has_v)[0]
        idx_image_only = np.where(~has_t & has_v)[0]
        idx_none = np.where(~has_t & ~has_v)[0]

        fused = np.zeros((n, out_dim))
        agreement = np.full(n, np.nan)
        confidence = np.zeros(n)
        flags = np.empty(n, dtype=object)
        flags[idx_none] = "none"

        if alignment_ok:
            if len(idx_both) > 0:
                Xt = np.stack([e_t[i] for i in idx_both])
                Xv = np.stack([e_v[i] for i in idx_both])
                Pt = self._project_text(Xt)
                Pv = self._project_image(Xv)
                Pt_n = Pt / (np.linalg.norm(Pt, axis=1, keepdims=True) + 1e-12)
                Pv_n = Pv / (np.linalg.norm(Pv, axis=1, keepdims=True) + 1e-12)
                s = np.sum(Pt_n * Pv_n, axis=1)
                agreement[idx_both] = s
                confidence[idx_both] = self.anfis.infer(s)
                fused[idx_both, :] = (Pt_n + Pv_n) / 2.0
            flags[idx_both] = "both"
            flags[idx_text_only] = "text_only"
            flags[idx_image_only] = "image_only"
            if self.single_modality_fill == "raw":
                self._fill_raw_batch(fused, idx_text_only, e_t, out_dim)
                self._fill_raw_batch(fused, idx_image_only, e_v, out_dim)
            # single_modality_fill == "zero": leave the zero-initialized rows as-is.
        else:
            # No fitted alignment: every has_t node (whether or not it also
            # has an image) is filled from its text embedding alone, exactly
            # as the original loop's `elif has_t:` branch does; single_modality_fill
            # never applies here (that flag only matters when alignment_ok).
            flags[idx_both] = "both_unaligned"
            flags[idx_text_only] = "text_only"
            flags[idx_image_only] = "image_only"
            self._fill_raw_batch(fused, idx_both, e_t, out_dim)
            self._fill_raw_batch(fused, idx_text_only, e_t, out_dim)
            self._fill_raw_batch(fused, idx_image_only, e_v, out_dim)

        return FusionResult(fused=fused, confidence=confidence, agreement=agreement, modality_flags=flags.tolist())
