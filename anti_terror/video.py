import cv2
from loguru import logger


def open_video_source(source: str | int):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    logger.info(f"Opened video source: {source}")
    return cap


def read_frame(cap):
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def release(cap):
    try:
        cap.release()
    except Exception:
        pass
