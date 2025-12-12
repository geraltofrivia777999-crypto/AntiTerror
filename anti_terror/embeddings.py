from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from loguru import logger
from torchvision import models, transforms
import cv2
from pathlib import Path
from typing import List, Tuple

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

from .config import EmbeddingConfig, select_device


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x


@dataclass
class EmbeddingSample:
    vectors: list[torch.Tensor]  # normalized embeddings history
    label: str
    image_path: str | None = None


class BagEmbedder:
    def __init__(self, cfg: EmbeddingConfig):
        device = select_device(cfg.device)
        if cfg.bag_model_name.lower() == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]))  # remove classifier
        backbone.eval().to(device)
        self.model = backbone
        self.device = device
        self.cfg = cfg
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        logger.info(f"Bag embedder ready on {device}")

    @torch.inference_mode()
    def __call__(self, crop: np.ndarray) -> torch.Tensor:
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        emb = self.model(tensor).flatten(1)
        emb = F.normalize(emb, dim=1)
        return emb.cpu().squeeze(0)


class FaceEmbedder:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.provider = cfg.face_provider.lower()
        if self.provider == "insightface" and FaceAnalysis is not None:
            # Try GPU first, fallback to CPU
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.model = FaceAnalysis(providers=providers)
            # det_size keeps speed; can tweak to (640,640) for better recall
            self.model.prepare(ctx_id=0 if select_device(cfg.device) == "cuda" else -1, det_size=(640, 640))
            logger.info("Face embedder ready (InsightFace)")
        else:
            device = select_device(cfg.device)
            self.mtcnn = MTCNN(image_size=160, device=device, thresholds=[0.7, 0.7, 0.8])
            self.model = InceptionResnetV1(pretrained=cfg.face_model).eval().to(device)
            self.device = device
            logger.info(f"Face embedder ready (FaceNet) on {device}")

    @torch.inference_mode()
    def __call__(self, frame: np.ndarray) -> List[Tuple[np.ndarray, float, torch.Tensor, np.ndarray]]:
        # returns list of (bbox_xyxy, score, embedding, face_crop)
        if self.provider == "insightface" and FaceAnalysis is not None:
            faces = self.model.get(frame)
            results = []
            for f in faces:
                box = f.bbox.astype(float)  # x1,y1,x2,y2
                score = float(f.det_score)
                if score < self.cfg.face_confidence:
                    continue
                emb = torch.tensor(f.normed_embedding, dtype=torch.float32)
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                face_crop = frame[y1:y2, x1:x2].copy()
                results.append((box, score, emb, face_crop))
            return results

        # FaceNet fallback
        boxes, probs = self.mtcnn.detect(frame)
        results = []
        if boxes is None:
            return results
        aligned = self.mtcnn(frame, save_path=None)
        if aligned is None:
            return results
        if isinstance(aligned, torch.Tensor):
            aligned = [aligned]
        embeddings = self.model(torch.stack(aligned).to(self.device))
        embeddings = F.normalize(embeddings, dim=1)
        for box, prob, emb in zip(boxes, probs, embeddings):
            if prob is None or prob < self.cfg.face_confidence:
                continue
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            face_crop = frame[y1:y2, x1:x2].copy()
            results.append((box.astype(float), float(prob), emb.cpu(), face_crop))
        return results


class EmbeddingStore:
    """In-memory store for cosine matching with optional crop persistence."""

    def __init__(self, max_history: int = 10):
        self.samples: Dict[str, EmbeddingSample] = {}
        self.counter = 0
        self.max_history = max_history

    def _new_label(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter:04d}"

    def get_vector(self, label: str) -> torch.Tensor | None:
        sample = self.samples.get(label)
        if sample:
            return sample.vectors[-1]
        return None

    def find_best(self, emb: torch.Tensor) -> tuple[str | None, float]:
        best_label = None
        best_score = -1.0
        for lbl, sample in self.samples.items():
            # use best similarity across history
            score = max(self.cosine(emb, vec) for vec in sample.vectors)
            if score > best_score:
                best_score = score
                best_label = lbl
        return best_label, best_score

    @staticmethod
    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    def match_or_create(
        self,
        emb: torch.Tensor,
        prefix: str,
        threshold: float,
        image: np.ndarray | None = None,
        save_dir: Path | None = None,
        force_threshold: float | None = None,
        create_threshold: float | None = None,
    ) -> tuple[str, bool, float]:
        """Return (label, created_new, best_score)."""
        if not self.samples:
            label = self._new_label(prefix)
            path = self._maybe_save(image, save_dir, label)
            self.samples[label] = EmbeddingSample(vectors=[emb], label=label, image_path=path)
            return label, True, 1.0
        best_label, best_score = self.find_best(emb)
        if best_label and best_score >= threshold:
            # update moving average for robustness
            self.samples[best_label].vectors.append(emb)
            self.samples[best_label].vectors = self.samples[best_label].vectors[-self.max_history :]
            return best_label, False, best_score
        if force_threshold is not None and best_label and best_score >= force_threshold:
            # Reuse closest ID even if slightly below threshold to avoid duplicating same person
            self.samples[best_label].vectors.append(emb)
            self.samples[best_label].vectors = self.samples[best_label].vectors[-self.max_history :]
            return best_label, False, best_score
        if create_threshold is not None and best_score < create_threshold:
            # Do not create a new ID if everything is too low; fallback to best_label to keep continuity
            if best_label:
                return best_label, False, best_score
        label = self._new_label(prefix)
        path = self._maybe_save(image, save_dir, label)
        self.samples[label] = EmbeddingSample(vectors=[emb], label=label, image_path=path)
        return label, True, best_score

    def _maybe_save(self, image: np.ndarray | None, save_dir: Path | None, label: str) -> str | None:
        if image is None or save_dir is None:
            return None
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = save_dir / f"{label}.jpg"
            cv2.imwrite(str(filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            return str(filename)
        except Exception as e:
            logger.warning(f"Failed to save image for {label}: {e}")
            return None
