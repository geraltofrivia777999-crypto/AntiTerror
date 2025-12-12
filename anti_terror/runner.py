import argparse
import signal
import sys
import time
from typing import Dict

import cv2
import numpy as np
import torch
from loguru import logger

from .association import AssociationEngine
from .behavior import BehaviorAnalyzer
from .config import PipelineConfig, select_device
from .detection import Detector
from .embeddings import BagEmbedder, EmbeddingStore, FaceEmbedder, FaceQuality
from .events import EventSink
from .tracking import Tracker
from .video import open_video_source, read_frame, release

from dataclasses import dataclass, field


@dataclass
class PersonState:
    label: str | None = None
    ema_emb: torch.Tensor | None = None
    pending_label: str | None = None
    pending_count: int = 0
    init_frames: int = 0
    last_seen: float = 0.0
    # Quality tracking for adaptive EMA
    best_quality: float = 0.0
    quality_history: list = field(default_factory=list)  # List of recent quality scores
    consecutive_low_quality: int = 0  # Track poor detections


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        # adjust device globally
        cfg.detection.device = select_device(cfg.detection.device)
        cfg.embeddings.device = select_device(cfg.embeddings.device)

        self.cfg = cfg
        self.detector = Detector(cfg.detection)
        self.tracker = Tracker(cfg.tracking)
        self.bag_embedder = BagEmbedder(cfg.embeddings)
        self.face_embedder = FaceEmbedder(cfg.embeddings)
        self.bag_store = EmbeddingStore()
        self.face_store = EmbeddingStore()
        self.assoc = AssociationEngine(cfg.association)
        self.behavior = BehaviorAnalyzer(cfg.behavior)
        self.events = EventSink(cfg.events)
        self.cap = open_video_source(cfg.video_source)
        self.person_states: Dict[int, PersonState] = {}  # track_id -> state
        self.last_person_creation_ts: float = 0.0

    def process_frame(self, frame) -> None:
        detection = self.detector(frame)
        tracks = self.tracker.update(detection.boxes, detection.scores, detection.classes)

        person_tracks = [t for t in tracks if t.cls in self.cfg.detection.classes_person]
        bag_tracks = [t for t in tracks if t.cls in self.cfg.detection.classes_bag]

        # assign persistent IDs via embeddings
        bag_ids: Dict[int, str] = {}
        for bag in bag_tracks:
            x1, y1, x2, y2 = bag.box.astype(int)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            emb = self.bag_embedder(crop)
            label, created, _ = self.bag_store.match_or_create(
                emb,
                prefix="B",
                threshold=self.cfg.embeddings.bag_similarity_threshold,
                image=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                save_dir=self.cfg.persistence.bags_dir if self.cfg.persistence.save_bags else None,
            )
            bag_ids[bag.track_id] = label

        person_ids: Dict[int, str] = {}
        faces = self.face_embedder(frame)
        # match faces to person tracks by IoU of face bbox inside person bbox
        for p in person_tracks:
            state = self.person_states.get(p.track_id, PersonState())
            state.init_frames += 1
            state.last_seen = time.time()

            pid = None
            px1, py1, px2, py2 = p.box
            # pick best face inside person box - now includes quality
            candidates = []
            for face_data in faces:
                # New format: (bbox, score, embedding, face_crop, quality)
                if len(face_data) == 5:
                    (fx1, fy1, fx2, fy2), fscore, femb, fcrop, fquality = face_data
                else:
                    # Fallback for old format
                    (fx1, fy1, fx2, fy2), fscore, femb, fcrop = face_data
                    fquality = FaceQuality(detection_score=fscore)

                if fx1 >= px1 and fy1 >= py1 and fx2 <= px2 and fy2 <= py2:
                    candidates.append(((fx1, fy1, fx2, fy2), fscore, femb, fcrop, fquality))

            if candidates:
                # Select face with best quality, not just detection score
                best_candidate = max(candidates, key=lambda x: x[4].overall_quality)
                (_, fscore, femb, fcrop, fquality) = best_candidate
                quality_score = fquality.overall_quality

                # Track quality for this person
                state.quality_history.append(quality_score)
                if len(state.quality_history) > 10:
                    state.quality_history = state.quality_history[-10:]
                state.best_quality = max(state.best_quality, quality_score)

                # Adaptive EMA: weight by quality
                # High quality faces have more influence on the embedding
                ema_alpha = self.cfg.embeddings.ema_alpha
                if self.cfg.embeddings.ema_quality_weighted:
                    # Adjust alpha based on quality (higher quality = more weight to new)
                    quality_factor = min(quality_score / 0.7, 1.0)  # normalize around 0.7
                    new_weight = (1 - ema_alpha) * (0.5 + 0.5 * quality_factor)
                    old_weight = 1 - new_weight
                else:
                    old_weight = ema_alpha
                    new_weight = 1 - ema_alpha

                if state.ema_emb is None:
                    state.ema_emb = femb
                else:
                    state.ema_emb = torch.nn.functional.normalize(
                        old_weight * state.ema_emb + new_weight * femb, dim=0
                    )
                ref_emb = state.ema_emb

                best_label, best_score = self.face_store.find_best(ref_emb)

                # Get thresholds from config
                similarity_thresh = self.cfg.embeddings.face_similarity_threshold
                force_margin = self.cfg.embeddings.face_force_match_margin
                create_thresh = self.cfg.embeddings.face_create_threshold

                # Decide candidate label without creating new IDs eagerly
                if state.label:
                    pid = state.label
                    ref_vec = self.face_store.get_vector(state.label)
                    score_current = float(
                        torch.nn.functional.cosine_similarity(ref_emb.unsqueeze(0), ref_vec.unsqueeze(0)).item()
                    ) if ref_vec is not None else 0.0

                    # Update store with new embedding (quality-weighted)
                    self.face_store.add_embedding(state.label, femb, quality_score)

                    if score_current < similarity_thresh - force_margin:
                        # Consider switch only if consistent over several frames
                        if best_label and best_label != state.label and best_score >= similarity_thresh - force_margin:
                            if state.pending_label == best_label:
                                state.pending_count += 1
                            else:
                                state.pending_label = best_label
                                state.pending_count = 1
                            if state.pending_count >= self.cfg.embeddings.face_switch_patience_frames:
                                logger.info(f"Switching ID from {state.label} to {best_label} (score: {best_score:.3f})")
                                pid = best_label
                                state.label = pid
                                state.pending_label = None
                                state.pending_count = 0
                        else:
                            state.pending_label = None
                            state.pending_count = 0
                else:
                    # No label yet - be very conservative about creating new IDs
                    if best_label and best_score >= similarity_thresh:
                        # Clear match - reuse existing ID
                        pid = best_label
                        state.label = pid
                        self.face_store.add_embedding(pid, femb, quality_score)
                    elif best_label and best_score >= create_thresh:
                        # Moderate match - still prefer existing ID to avoid duplicates
                        pid = best_label
                        state.label = pid
                        self.face_store.add_embedding(pid, femb, quality_score)
                        logger.debug(f"Reusing {best_label} with moderate score {best_score:.3f}")
                    else:
                        # Low or no match - wait for patience frames before creating
                        if state.init_frames >= self.cfg.embeddings.face_new_id_patience_frames:
                            now = time.time()
                            # Enforce cooldown between new ID creation
                            if (now - self.last_person_creation_ts) >= self.cfg.embeddings.face_new_id_cooldown_s:
                                # Only create if quality is acceptable
                                if quality_score >= self.cfg.embeddings.min_face_quality:
                                    label, created, match_score = self.face_store.match_or_create(
                                        ref_emb,
                                        prefix="P",
                                        threshold=similarity_thresh,
                                        force_threshold=similarity_thresh - force_margin,
                                        create_threshold=create_thresh,
                                        quality=quality_score,
                                        image=fcrop,
                                        save_dir=self.cfg.persistence.faces_dir if self.cfg.persistence.save_faces else None,
                                    )
                                    if created:
                                        self.last_person_creation_ts = now
                                        logger.info(f"Created new person ID: {label}")
                                    pid = label
                                    state.label = pid
                                else:
                                    # Low quality - fallback to best label or track ID
                                    pid = best_label if best_label else f"P_{p.track_id:04d}"
                                    state.consecutive_low_quality += 1
                            else:
                                # Cooldown active - prefer existing ID
                                pid = best_label if best_label else state.label
                        else:
                            # Still waiting for patience - use best available
                            pid = best_label if best_label else state.label

                # Fallback if still None
                if pid is None:
                    pid = state.label if state.label else f"P_{p.track_id:04d}"
                state.label = pid
            else:
                # No face this frame; reuse previous ID if exists
                pid = state.label if state.label else f"P_{p.track_id:04d}"
                state.consecutive_low_quality = 0  # Reset counter when no face

            person_ids[p.track_id] = pid
            self.person_states[p.track_id] = state

        # prune stale person track states
        current_ids = {p.track_id for p in person_tracks}
        for tid in list(self.person_states.keys()):
            if tid not in current_ids:
                del self.person_states[tid]

        assignments = self.assoc.associate(person_tracks, bag_tracks)
        events = self.behavior.update(bag_tracks, bag_ids, person_ids, assignments)
        if events:
            self.events.emit(events)

        self._render(frame, person_tracks, bag_tracks, bag_ids, person_ids, assignments)

    def _render(self, frame, person_tracks, bag_tracks, bag_ids, person_ids, assignments):
        # lightweight on-frame debug overlay
        for p in person_tracks:
            x1, y1, x2, y2 = map(int, p.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = person_ids.get(p.track_id, f"P{p.track_id}")
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for b in bag_tracks:
            x1, y1, x2, y2 = map(int, b.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            label = bag_ids.get(b.track_id, f"B{b.track_id}")
            owner = assignments.get(b.track_id)
            owner_label = person_ids.get(owner, "?") if owner is not None else "?"
            cv2.putText(
                frame,
                f"{label}-> {owner_label}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
            )
        cv2.imshow("AntiTerror MVP", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    def run(self):
        logger.info("Starting pipeline. Press 'q' to exit.")
        try:
            while True:
                frame = read_frame(self.cap)
                if frame is None:
                    break
                self.process_frame(frame)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            release(self.cap)
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Anti-terror video analytics MVP")
    parser.add_argument("--source", type=str, default="0", help="Video source (index or path)")
    parser.add_argument("--camera-id", type=str, default="CAM_01", help="Camera identifier")
    parser.add_argument("--conf", type=float, default=None, help="Detection confidence override")
    parser.add_argument("--abandonment-timeout", type=float, default=None, help="Seconds to flag abandoned bag")
    return parser.parse_args()


def main():
    args = parse_args()
    source: str | int = int(args.source) if args.source.isdigit() else args.source
    cfg = PipelineConfig(video_source=source)
    cfg.events.camera_id = args.camera_id
    if args.conf is not None:
        cfg.detection.conf_threshold = args.conf
    if args.abandonment_timeout is not None:
        cfg.behavior.abandonment_timeout_s = args.abandonment_timeout
    pipeline = Pipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
