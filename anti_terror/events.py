import datetime as dt
from pathlib import Path
from typing import Iterable

from loguru import logger

from .config import EventConfig


class EventSink:
    def __init__(self, cfg: EventConfig):
        self.cfg = cfg
        if cfg.enable_file_logging:
            cfg.log_dir.mkdir(parents=True, exist_ok=True)
            logger.add(cfg.log_dir / "events.log", rotation="10 MB")

    def emit(self, events: Iterable[dict]):
        for event in events:
            timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            payload = {
                "timestamp": timestamp,
                "camera": self.cfg.camera_id,
                **event,
            }
            logger.opt(colors=True).info(
                f"[ALERT] {payload['timestamp']} | {payload.get('type','')} "
                f"PersonID: {payload.get('person_id','?')} BagID: {payload.get('bag_id','?')} "
                f"Camera: {payload['camera']} details={payload}"
            )
