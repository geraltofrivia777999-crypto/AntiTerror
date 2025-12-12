from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .config import BehaviorConfig
from .association import bbox_center, iou


@dataclass
class BagState:
    bag_id: str
    last_box: np.ndarray
    last_update: float
    owner_person_id: Optional[str] = None
    last_owner_seen: float = field(default_factory=time.time)
    static_since: Optional[float] = None


class BehaviorAnalyzer:
    def __init__(self, cfg: BehaviorConfig):
        self.cfg = cfg
        self.bags: Dict[int, BagState] = {}  # key: bag track_id

    def update(
        self,
        bag_tracks,
        bag_ids: Dict[int, str],
        person_ids: Dict[int, str],
        assignments: Dict[int, int],
    ) -> list[dict]:
        now = time.time()
        events = []

        # Update bag states
        for bag in bag_tracks:
            state = self.bags.get(bag.track_id)
            if state is None:
                state = BagState(bag_id=bag_ids[bag.track_id], last_box=bag.box, last_update=now)
                self.bags[bag.track_id] = state
            else:
                # check static
                if iou(state.last_box, bag.box) >= self.cfg.static_iou_threshold:
                    state.static_since = state.static_since or now
                else:
                    state.static_since = None
                state.last_box = bag.box
                state.last_update = now

            if bag.track_id in assignments:
                pid = assignments[bag.track_id]
                state.owner_person_id = person_ids.get(pid)
                state.last_owner_seen = now
            elif state.owner_person_id:
                # owner not currently assigned
                pass

        # Detect abandoned bags
        for track_id, state in list(self.bags.items()):
            # discard stale tracks not updated
            if now - state.last_update > 3 * self.cfg.abandonment_timeout_s:
                del self.bags[track_id]
                continue

            if not state.static_since:
                continue
            if state.owner_person_id is None:
                # never linked; not enough info
                continue
            away_time = now - state.last_owner_seen
            static_time = now - state.static_since
            if away_time >= self.cfg.abandonment_timeout_s and static_time >= self.cfg.abandonment_timeout_s:
                events.append(
                    {
                        "type": "Abandoned Bag",
                        "bag_id": state.bag_id,
                        "person_id": state.owner_person_id,
                        "away_for_s": round(away_time, 2),
                        "static_for_s": round(static_time, 2),
                    }
                )
                # reset to avoid repeated spamming
                state.static_since = None
                state.last_owner_seen = now

        return events
