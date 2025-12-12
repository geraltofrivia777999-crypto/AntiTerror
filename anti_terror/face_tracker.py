"""Face-specific tracking and re-identification module.

This module provides robust face tracking with embedding-based re-identification
to minimize duplicate ID creation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from supervision import Detections
from supervision.tracker.byte_tracker.core import ByteTrack

from .config import EmbeddingConfig, TrackingConfig


@dataclass
class FaceTrack:
    """Represents a tracked face with its embedding history."""
    track_id: int
    box: np.ndarray  # xyxy face bounding box
    score: float  # detection confidence

    # Identity
    person_id: str | None = None

    # Embedding state
    embedding: torch.Tensor | None = None
    embedding_history: list = field(default_factory=list)
    quality_history: list = field(default_factory=list)

    # Tracking state
    frames_seen: int = 0
    frames_since_embedding: int = 0
    last_seen_time: float = 0.0

    # Re-ID state
    candidate_id: str | None = None
    candidate_count: int = 0


class FaceGallery:
    """Gallery of known face identities with robust matching.

    Uses multiple strategies to minimize false new IDs:
    1. Centroid-based matching with quality weighting
    2. Temporal consistency checks
    3. Adaptive thresholds based on gallery size
    """

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.identities: Dict[str, FaceIdentity] = {}
        self.counter = 0
        self.last_creation_time = 0.0

    def _new_id(self) -> str:
        self.counter += 1
        return f"P_{self.counter:04d}"

    def match(self, embedding: torch.Tensor, quality: float = 1.0) -> Tuple[str | None, float]:
        """Find best matching identity for an embedding.

        Returns (identity_id, similarity_score) or (None, 0.0) if no match.
        """
        if not self.identities:
            return None, 0.0

        best_id = None
        best_score = -1.0

        for pid, identity in self.identities.items():
            score = identity.match_score(embedding)
            if score > best_score:
                best_score = score
                best_id = pid

        return best_id, best_score

    def match_or_create(
        self,
        embedding: torch.Tensor,
        quality: float,
        face_crop: np.ndarray | None = None,
        min_quality_for_create: float = 0.4
    ) -> Tuple[str, bool, float]:
        """Match embedding to existing identity or create new one.

        Very conservative about creating new IDs - multiple fallback checks.
        Returns (identity_id, was_created, match_score).
        """
        best_id, best_score = self.match(embedding, quality)

        # Adaptive threshold based on gallery size
        # More IDs = stricter matching to avoid duplicates
        base_threshold = self.cfg.face_similarity_threshold
        if len(self.identities) > 5:
            # Lower threshold when many IDs exist (be more permissive in matching)
            adaptive_threshold = base_threshold - 0.05 * min(len(self.identities) - 5, 5) / 5
        else:
            adaptive_threshold = base_threshold

        # Primary match
        if best_id and best_score >= adaptive_threshold:
            self.identities[best_id].add_embedding(embedding, quality)
            return best_id, False, best_score

        # Secondary match (force threshold)
        force_threshold = adaptive_threshold - self.cfg.face_force_match_margin
        if best_id and best_score >= force_threshold:
            self.identities[best_id].add_embedding(embedding, quality)
            return best_id, False, best_score

        # Tertiary match (very low threshold to avoid duplicates)
        min_threshold = 0.25
        if best_id and best_score >= min_threshold:
            logger.debug(f"Low-confidence match ({best_score:.3f}) to {best_id}")
            self.identities[best_id].add_embedding(embedding, quality)
            return best_id, False, best_score

        # Check cooldown before creating
        now = time.time()
        if (now - self.last_creation_time) < self.cfg.face_new_id_cooldown_s:
            if best_id:
                # Cooldown active - reuse best match anyway
                self.identities[best_id].add_embedding(embedding, quality)
                return best_id, False, best_score
            # No match and cooldown active - return None
            return "", False, 0.0

        # Quality check for new ID
        if quality < min_quality_for_create:
            if best_id:
                return best_id, False, best_score
            return "", False, 0.0

        # Create new identity
        new_id = self._new_id()
        self.identities[new_id] = FaceIdentity(new_id, embedding, quality)
        self.last_creation_time = now
        logger.info(f"Created new identity {new_id} (best_match: {best_id}, score: {best_score:.3f})")
        return new_id, True, 1.0

    def get_all_ids(self) -> List[str]:
        return list(self.identities.keys())


@dataclass
class FaceIdentity:
    """Represents a known face identity with embedding history."""
    identity_id: str
    embeddings: list = field(default_factory=list)
    qualities: list = field(default_factory=list)
    centroid: torch.Tensor | None = None
    created_time: float = field(default_factory=time.time)
    last_seen_time: float = field(default_factory=time.time)
    max_history: int = 30

    def __init__(self, identity_id: str, initial_embedding: torch.Tensor, quality: float = 1.0):
        self.identity_id = identity_id
        self.embeddings = [initial_embedding]
        self.qualities = [quality]
        self.centroid = initial_embedding.clone()
        self.created_time = time.time()
        self.last_seen_time = time.time()
        self.max_history = 30

    def add_embedding(self, embedding: torch.Tensor, quality: float = 1.0) -> None:
        """Add new embedding observation."""
        self.embeddings.append(embedding)
        self.qualities.append(quality)
        self.last_seen_time = time.time()

        # Trim history (keep highest quality)
        if len(self.embeddings) > self.max_history:
            # Sort by quality and keep best
            paired = list(zip(self.embeddings, self.qualities))
            paired.sort(key=lambda x: x[1], reverse=True)
            self.embeddings = [p[0] for p in paired[:self.max_history]]
            self.qualities = [p[1] for p in paired[:self.max_history]]

        self._update_centroid()

    def _update_centroid(self) -> None:
        """Recompute quality-weighted centroid."""
        if not self.embeddings:
            return

        weights = torch.tensor(self.qualities)
        weights = weights / weights.sum()

        stacked = torch.stack(self.embeddings)
        weighted_sum = (stacked * weights.unsqueeze(1)).sum(dim=0)
        self.centroid = F.normalize(weighted_sum, dim=0)

    def match_score(self, embedding: torch.Tensor) -> float:
        """Compute similarity score for an embedding."""
        if self.centroid is None:
            return 0.0

        # Primary: centroid similarity
        centroid_sim = float(F.cosine_similarity(
            embedding.unsqueeze(0),
            self.centroid.unsqueeze(0)
        ).item())

        # Secondary: best recent match
        recent = self.embeddings[-5:] if len(self.embeddings) >= 5 else self.embeddings
        recent_scores = [
            float(F.cosine_similarity(embedding.unsqueeze(0), e.unsqueeze(0)).item())
            for e in recent
        ]
        recent_max = max(recent_scores) if recent_scores else 0.0

        # Combined: 60% centroid + 40% recent (more weight to recent for temporal consistency)
        return 0.6 * centroid_sim + 0.4 * recent_max


class FaceTracker:
    """Specialized tracker for faces with embedding-based re-identification.

    Key features:
    1. ByteTrack for spatial tracking
    2. Embedding matching for re-identification
    3. Temporal consistency for stable IDs
    """

    def __init__(self, tracking_cfg: TrackingConfig, embedding_cfg: EmbeddingConfig):
        self.tracking_cfg = tracking_cfg
        self.embedding_cfg = embedding_cfg

        # Spatial tracker (ByteTrack)
        self.byte_tracker = ByteTrack(
            track_activation_threshold=0.5,  # Higher for faces
            lost_track_buffer=tracking_cfg.lost_track_buffer,
            minimum_matching_threshold=0.7,
            frame_rate=tracking_cfg.frame_rate,
            minimum_consecutive_frames=2,
        )

        # Identity gallery
        self.gallery = FaceGallery(embedding_cfg)

        # Active tracks
        self.tracks: Dict[int, FaceTrack] = {}

        logger.info("Initialized FaceTracker")

    def update(
        self,
        face_boxes: np.ndarray,
        face_scores: np.ndarray,
        face_embeddings: List[torch.Tensor],
        face_qualities: List[float],
        face_crops: List[np.ndarray]
    ) -> List[FaceTrack]:
        """Update tracker with new face detections.

        Args:
            face_boxes: Nx4 array of face bounding boxes (xyxy)
            face_scores: N detection confidence scores
            face_embeddings: N face embeddings
            face_qualities: N quality scores
            face_crops: N face crop images

        Returns:
            List of active FaceTrack objects with assigned person_ids
        """
        current_time = time.time()

        # Handle empty detections
        if len(face_boxes) == 0:
            self.byte_tracker.update_with_detections(Detections.empty())
            # Age out old tracks
            for tid in list(self.tracks.keys()):
                if current_time - self.tracks[tid].last_seen_time > 2.0:
                    del self.tracks[tid]
            return []

        # Run ByteTrack
        detections = Detections(
            xyxy=face_boxes,
            confidence=face_scores,
            class_id=np.zeros(len(face_boxes), dtype=int),  # All faces = class 0
        )
        tracked = self.byte_tracker.update_with_detections(detections)

        # Create detection index for matching
        det_index = {i: i for i in range(len(face_boxes))}

        result: List[FaceTrack] = []
        active_track_ids = set()

        for idx, track_id in enumerate(tracked.tracker_id):
            track_id = int(track_id)
            active_track_ids.add(track_id)

            box = tracked.xyxy[idx]
            score = float(tracked.confidence[idx])

            # Find corresponding detection (closest box)
            det_idx = self._find_detection_idx(box, face_boxes)
            if det_idx is None or det_idx >= len(face_embeddings):
                continue

            embedding = face_embeddings[det_idx]
            quality = face_qualities[det_idx]
            crop = face_crops[det_idx] if det_idx < len(face_crops) else None

            # Get or create track state
            if track_id not in self.tracks:
                self.tracks[track_id] = FaceTrack(
                    track_id=track_id,
                    box=box,
                    score=score
                )

            track = self.tracks[track_id]
            track.box = box
            track.score = score
            track.frames_seen += 1
            track.last_seen_time = current_time

            # Update embedding (EMA smoothing)
            if track.embedding is None:
                track.embedding = embedding
            else:
                alpha = self.embedding_cfg.ema_alpha
                # Quality-weighted alpha
                if self.embedding_cfg.ema_quality_weighted:
                    q_factor = min(quality / 0.6, 1.0)
                    new_weight = (1 - alpha) * (0.5 + 0.5 * q_factor)
                else:
                    new_weight = 1 - alpha
                track.embedding = F.normalize(
                    (1 - new_weight) * track.embedding + new_weight * embedding,
                    dim=0
                )

            track.embedding_history.append(embedding)
            track.quality_history.append(quality)
            if len(track.embedding_history) > 10:
                track.embedding_history = track.embedding_history[-10:]
                track.quality_history = track.quality_history[-10:]

            # Assign person_id
            if track.person_id is None:
                # New track - need to assign ID
                if track.frames_seen >= self.embedding_cfg.face_new_id_patience_frames:
                    # Enough frames seen - try to match or create
                    pid, created, match_score = self.gallery.match_or_create(
                        track.embedding,
                        quality=max(track.quality_history) if track.quality_history else quality,
                        face_crop=crop,
                        min_quality_for_create=self.embedding_cfg.min_face_quality
                    )
                    if pid:
                        track.person_id = pid
                else:
                    # Not enough frames - try to match existing only
                    best_id, best_score = self.gallery.match(track.embedding, quality)
                    if best_id and best_score >= self.embedding_cfg.face_similarity_threshold - 0.1:
                        track.person_id = best_id
                        self.gallery.identities[best_id].add_embedding(embedding, quality)
            else:
                # Existing track - verify and update
                current_id = track.person_id
                best_id, best_score = self.gallery.match(track.embedding, quality)

                # Update gallery with new observation
                if current_id in self.gallery.identities:
                    self.gallery.identities[current_id].add_embedding(embedding, quality)

                # Check if should switch ID (requires consistency)
                if best_id and best_id != current_id:
                    threshold = self.embedding_cfg.face_similarity_threshold
                    if best_score >= threshold + 0.1:  # Significantly better match
                        if track.candidate_id == best_id:
                            track.candidate_count += 1
                            if track.candidate_count >= self.embedding_cfg.face_switch_patience_frames:
                                logger.info(f"Track {track_id}: switching {current_id} -> {best_id}")
                                track.person_id = best_id
                                track.candidate_id = None
                                track.candidate_count = 0
                        else:
                            track.candidate_id = best_id
                            track.candidate_count = 1
                    else:
                        track.candidate_id = None
                        track.candidate_count = 0

            result.append(track)

        # Cleanup old tracks
        for tid in list(self.tracks.keys()):
            if tid not in active_track_ids:
                if current_time - self.tracks[tid].last_seen_time > 3.0:
                    del self.tracks[tid]

        return result

    def _find_detection_idx(self, track_box: np.ndarray, det_boxes: np.ndarray) -> Optional[int]:
        """Find detection index that best matches track box."""
        if len(det_boxes) == 0:
            return None

        # Compute IoU between track box and all detections
        best_iou = 0.0
        best_idx = None

        for i, det_box in enumerate(det_boxes):
            iou = self._compute_iou(track_box, det_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        return best_idx if best_iou > 0.3 else None

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0
