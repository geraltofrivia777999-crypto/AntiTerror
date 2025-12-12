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
from .embeddings import BagEmbedder, EmbeddingStore, FaceEmbedder
from .events import EventSink
from .tracking import Tracker
from .video import open_video_source, read_frame, release

from dataclasses import dataclass


@dataclass
class PersonState:
    label: str | None = None
    ema_emb: torch.Tensor | None = None
    pending_label: str | None = None
    pending_count: int = 0
    init_frames: int = 0
    last_seen: float = 0.0


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
            # pick best face inside person box
            candidates = []
            for (fx1, fy1, fx2, fy2), fscore, femb, fcrop in faces:
                if fx1 >= px1 and fy1 >= py1 and fx2 <= px2 and fy2 <= py2:
                    candidates.append(((fx1, fy1, fx2, fy2), fscore, femb, fcrop))
            if candidates:
                (_, fscore, femb, fcrop) = max(candidates, key=lambda x: x[1])
                # EMA embedding for stability
                if state.ema_emb is None:
                    state.ema_emb = femb
                else:
                    state.ema_emb = torch.nn.functional.normalize(0.8 * state.ema_emb + 0.2 * femb, dim=0)
                ref_emb = state.ema_emb

                best_label, best_score = self.face_store.find_best(ref_emb)

                # Decide candidate label without creating new IDs eagerly
                if state.label:
                    pid = state.label
                    ref_vec = self.face_store.get_vector(state.label)
                    score_current = float(
                        torch.nn.functional.cosine_similarity(ref_emb.unsqueeze(0), ref_vec.unsqueeze(0)).item()
                    ) if ref_vec is not None else 0.0
                    if score_current < self.cfg.embeddings.face_similarity_threshold - self.cfg.embeddings.face_force_match_margin:
                        # consider switch if consistent over several frames
                        if best_label and best_score >= self.cfg.embeddings.face_similarity_threshold - self.cfg.embeddings.face_force_match_margin:
                            if state.pending_label == best_label:
                                state.pending_count += 1
                            else:
                                state.pending_label = best_label
                                state.pending_count = 1
                            if state.pending_count >= self.cfg.embeddings.face_switch_patience_frames:
                                pid = best_label
                                state.label = pid
                                state.pending_label = None
                                state.pending_count = 0
                        else:
                            state.pending_label = None
                            state.pending_count = 0
                else:
                    # no label yet, gate creation by patience and score
                    if best_label and best_score >= self.cfg.embeddings.face_similarity_threshold:
                        pid = best_label
                        state.label = pid
                    elif best_label and best_score >= (self.cfg.embeddings.face_create_threshold - 0.05):
                        # reuse existing best label even если чуть ниже create_threshold
                        pid = best_label
                        state.label = pid
                    else:
                        # only create new ID if patience satisfied and score good enough
                        if (
                            state.init_frames >= self.cfg.embeddings.face_new_id_patience_frames
                            and best_score >= self.cfg.embeddings.face_create_threshold
                        ):
                            label, created, _ = self.face_store.match_or_create(
                                ref_emb,
                                prefix="P",
                                threshold=self.cfg.embeddings.face_similarity_threshold,
                                force_threshold=self.cfg.embeddings.face_similarity_threshold
                                - self.cfg.embeddings.face_force_match_margin,
                                create_threshold=self.cfg.embeddings.face_create_threshold,
                                image=fcrop,
                                save_dir=self.cfg.persistence.faces_dir if self.cfg.persistence.save_faces else None,
                            )
                            now = time.time()
                            if created and (now - self.last_person_creation_ts) < self.cfg.embeddings.face_new_id_cooldown_s and best_label:
                                label = best_label
                                created = False
                            if created:
                                self.last_person_creation_ts = now
                            pid = label
                            state.label = pid
                        else:
                            # fallback to best known label even if low score to preserve identity
                            pid = state.label if state.label else best_label
                # fallback if still None
                if pid is None:
                    pid = state.label if state.label else f"P_{p.track_id:04d}"
                state.label = pid
            else:
                # No face this frame; reuse previous ID if exists
                pid = state.label if state.label else f"P_{p.track_id:04d}"

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
