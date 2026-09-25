"""
Stage 1 (Section 4.1): Multimodal node representation.

Each node i carries raw text t_i and, optionally, an image v_i. This module
turns that raw content into dense embeddings e_i^t, e_i^v. Modality
availability is treated as partial by design: a missing modality produces
`None`, not an imputed zero vector, so that downstream fuzzy-fusion code
(nf_mcd.fuzzy_fusion) can reason explicitly about missingness rather than
silently treating "missing" as "neutral".

Real encoders
-------------
By default this module tries to use:
  - `sentence-transformers` (e.g. "all-MiniLM-L6-v2") for text
  - `transformers` CLIP (e.g. "openai/clip-vit-base-patch32") for images

Both require network access to download pretrained weights the first time
they are used. If either library or the model weights are unavailable, the
encoder transparently falls back to a deterministic pseudo-random embedding
(seeded by a hash of the input) so that the rest of the pipeline can still
be developed, tested, and demonstrated end-to-end without those downloads.
This fallback is clearly logged and is NOT suitable for real experiments —
swap in real encoders (or your own callables) before running on actual data.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

TEXT_EMBED_DIM_FALLBACK = 384   # matches all-MiniLM-L6-v2 output dim
IMAGE_EMBED_DIM_FALLBACK = 512  # matches CLIP ViT-B/32 output dim


def _deterministic_pseudo_embedding(key: str, dim: int) -> np.ndarray:
    """Deterministic, reproducible pseudo-embedding derived from a string.

    Used only as a fallback when real pretrained encoders are unavailable.
    Same input string -> same vector, so downstream code (and demos) behave
    reproducibly even without model weights.
    """
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return v / (np.linalg.norm(v) + 1e-12)


class TextEncoder:
    """Wraps a sentence-level text encoder with a deterministic fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: Optional[int] = None):
        self.model_name = model_name
        self._model = None
        self._backend = "fallback"
        self.dim = dim or TEXT_EMBED_DIM_FALLBACK
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(model_name)
            self._backend = "sentence-transformers"
            self.dim = self._model.get_sentence_embedding_dimension()
        except Exception as exc:  # noqa: BLE001 - broad on purpose, this is an optional dep
            warnings.warn(
                f"TextEncoder: falling back to deterministic pseudo-embeddings "
                f"(sentence-transformers unavailable or model download failed: {exc}). "
                f"Install `sentence-transformers` and ensure network access to use a "
                f"real text encoder.",
                stacklevel=2,
            )

    def encode(self, texts: Sequence[Optional[str]]) -> List[Optional[np.ndarray]]:
        """Encode a batch of texts. `None` entries (missing text) map to `None`."""
        present_idx = [i for i, t in enumerate(texts) if t is not None and str(t).strip() != ""]
        out: List[Optional[np.ndarray]] = [None] * len(texts)

        if not present_idx:
            return out

        if self._backend == "sentence-transformers":
            batch = [texts[i] for i in present_idx]
            # batch_size caps how many texts sentence-transformers tokenizes/
            # embeds in one forward pass; encode() chunks internally, but we
            # still bound it explicitly here for parity with ImageEncoder's
            # manual chunking below (see its comment for why that one is a
            # hard requirement, not just tidiness).
            embs = self._model.encode(batch, normalize_embeddings=True, batch_size=32)
            for i, e in zip(present_idx, embs):
                out[i] = np.asarray(e, dtype=np.float64)
        else:
            for i in present_idx:
                out[i] = _deterministic_pseudo_embedding(f"text::{texts[i]}", self.dim)

        return out


class ImageEncoder:
    """Wraps a CLIP-style vision-language image encoder with a deterministic fallback.

    Images are accepted as either:
      - a PIL.Image.Image / numpy array (passed to a real CLIP processor), or
      - a plain string key (e.g. a file path or URL) used only for the
        deterministic fallback embedding when no real image data is on hand.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", dim: Optional[int] = None):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._backend = "fallback"
        self.dim = dim or IMAGE_EMBED_DIM_FALLBACK
        try:
            import torch  # type: ignore
            from transformers import CLIPModel, CLIPProcessor  # type: ignore

            self._model = CLIPModel.from_pretrained(model_name)
            self._processor = CLIPProcessor.from_pretrained(model_name)
            self._model.eval()
            self._torch = torch
            self._backend = "clip"
            self.dim = self._model.config.projection_dim
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"ImageEncoder: falling back to deterministic pseudo-embeddings "
                f"(transformers/CLIP unavailable or model download failed: {exc}). "
                f"Install `transformers`+`torch` and ensure network access to use a "
                f"real CLIP encoder.",
                stacklevel=2,
            )

    def encode(self, images: Sequence[Optional[object]]) -> List[Optional[np.ndarray]]:
        """Encode a batch of images. `None` entries (missing image) map to `None`."""
        present_idx = [i for i, v in enumerate(images) if v is not None]
        out: List[Optional[np.ndarray]] = [None] * len(images)

        if not present_idx:
            return out

        if self._backend == "clip":
            # Chunk into mini-batches rather than running the processor/model
            # on the whole `present_idx` set at once: CLIPProcessor stacks
            # every image into a single (N, 3, 224, 224) float tensor before
            # the forward pass, so an unbounded N scales memory linearly with
            # however many images happen to be present in a given fit() call
            # (not a fixed cost) - verified directly, this is what took down
            # a real run at N~1140 images on this machine. 32 keeps peak
            # memory bounded regardless of N.
            clip_batch_size = 32
            all_feats = []
            for start in range(0, len(present_idx), clip_batch_size):
                chunk_idx = present_idx[start: start + clip_batch_size]
                batch = [images[i] for i in chunk_idx]
                inputs = self._processor(images=batch, return_tensors="pt")
                with self._torch.no_grad():
                    outputs = self._model.get_image_features(**inputs)
                    # transformers >=4.49 returns a BaseModelOutputWithPooling
                    # (projected image embedding in `.pooler_output`) instead of
                    # a bare tensor; older versions return the tensor directly.
                    feats = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                all_feats.append(feats.cpu().numpy())
            feats = np.concatenate(all_feats, axis=0)
            for i, e in zip(present_idx, feats):
                out[i] = np.asarray(e, dtype=np.float64)
        else:
            for i in present_idx:
                key = images[i] if isinstance(images[i], str) else f"image_obj::{id(images[i])}::{i}"
                out[i] = _deterministic_pseudo_embedding(f"image::{key}", self.dim)

        return out


@dataclass
class MultimodalEncoder:
    """Convenience wrapper combining a TextEncoder and an ImageEncoder.

    Usage
    -----
    >>> enc = MultimodalEncoder()
    >>> e_t, e_v = enc.encode(texts=["hello world", None], images=[None, "img.jpg"])
    """

    text_encoder: TextEncoder = None
    image_encoder: ImageEncoder = None

    def __post_init__(self):
        if self.text_encoder is None:
            self.text_encoder = TextEncoder()
        if self.image_encoder is None:
            self.image_encoder = ImageEncoder()

    def encode(
        self,
        texts: Sequence[Optional[str]],
        images: Sequence[Optional[object]],
    ) -> tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        assert len(texts) == len(images), "texts and images must be the same length (one entry per node)"
        e_t = self.text_encoder.encode(texts)
        e_v = self.image_encoder.encode(images)
        return e_t, e_v
