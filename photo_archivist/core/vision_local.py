"""LIVE local vision backend: insightface face detection + optional CLIP vectors.

Selected by `config.yaml → vision.backend: local`. If insightface is not
installed, `vision.get_backend()` falls back to the mock backend automatically.

First run downloads the insightface `buffalo_l` model (~300 MB) into
~/.insightface/models/ — afterwards everything runs offline. Captions stay
empty on purpose unless a vision-language model is added: this backend only
claims what it can actually measure (faces, scores, CLIP embeddings).
"""
from __future__ import annotations

from .vision import VisionBackend, VisionResult


class LocalVision(VisionBackend):
    def __init__(self):
        import numpy as np  # noqa: F401  (insightface requires numpy)
        from insightface.app import FaceAnalysis
        self._np = np
        self._fa = FaceAnalysis(name="buffalo_l",
                                providers=["CPUExecutionProvider"])
        self._fa.prepare(ctx_id=-1, det_size=(640, 640))
        self._clip = None  # lazy: (model, preprocess) or False when unavailable

    def _clip_vec(self, path: str) -> list | None:
        if self._clip is False:
            return None
        try:
            if self._clip is None:
                import open_clip
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="laion2b_s34b_b79k")
                model.eval()
                self._clip = (model, preprocess)
            from PIL import Image
            model, preprocess = self._clip
            with Image.open(path) as im:
                tensor = preprocess(im.convert("RGB")).unsqueeze(0)
            import torch
            with torch.no_grad():
                vec = model.encode_image(tensor)
            return [float(x) for x in vec[0] / vec[0].norm()]
        except Exception:
            self._clip = False
            return None

    def describe(self, path: str, prime: dict) -> VisionResult:
        from PIL import Image
        with Image.open(path) as im:
            rgb = self._np.asarray(im.convert("RGB"))
        # insightface expects BGR
        faces = self._fa.get(rgb[:, :, ::-1])
        boxes, embeds = [], []
        for f in faces:
            boxes.append([float(x) for x in f.bbox] +
                         [float(getattr(f, "det_score", 0.0))])
            emb = getattr(f, "embedding", None)
            if emb is not None:
                embeds.append([float(x) for x in emb])
        return VisionResult(
            caption="",  # no VLM here — zero claims beats fabricated ones
            face_boxes=boxes, face_embeddings=embeds,
            image_vec=self._clip_vec(path),
            backend="local",
            confidences={"faces": len(boxes)})
