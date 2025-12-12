from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DetectionConfig:
    model_path: str = "yolov8s.pt"  # a bit heavier than nano for better bag/person fidelity
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str = "cuda"  # fallback handled at runtime
    classes_person: tuple[int, ...] = (0,)  # COCO: person
    classes_bag: tuple[int, ...] = (24, 26, 28)  # backpack, handbag, suitcase


@dataclass
class TrackingConfig:
    # ByteTrack params tuned for real-time webcam
    track_activation_threshold: float = 0.35
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    frame_rate: int = 30
    minimum_consecutive_frames: int = 1


@dataclass
class EmbeddingConfig:
    device: str = "cuda"
    # FaceNet fallback model (used when InsightFace unavailable)
    face_model: str = "vggface2"  # facenet-pytorch pretrained set

    # InsightFace model selection
    # Options: "buffalo_l" (best), "buffalo_m", "buffalo_s", "buffalo_sc"
    # buffalo_l: ArcFace R100 - highest accuracy, recommended for production
    # buffalo_s: ArcFace R34 - faster, good balance
    face_model_name: str = "buffalo_l"

    # Face detection parameters
    face_confidence: float = 0.65  # slightly lower to catch more faces
    face_provider: str = "insightface"  # "insightface" | "facenet"

    # Quality filtering (reduces noise from bad detections)
    min_face_size: int = 50  # minimum face size in pixels
    min_face_quality: float = 0.3  # minimum overall quality score [0-1]

    # ID creation timing (prevent rapid ID creation)
    face_new_id_cooldown_s: float = 5.0  # increased from 3.0 to reduce duplicates
    face_new_id_patience_frames: int = 5  # increased from 3 - wait longer before creating ID
    face_switch_patience_frames: int = 5  # increased from 3 - more stable ID assignment

    # Bag embedding config
    bag_model_name: str = "resnet50"
    bag_embedding_size: int = 2048
    bag_similarity_threshold: float = 0.7  # cosine similarity to attach to existing BagID

    # Face matching thresholds - CRITICAL for reducing duplicates
    # These are optimized for ArcFace/InsightFace embeddings
    face_similarity_threshold: float = 0.55  # lowered from 0.8 - ArcFace scores are typically lower
    face_force_match_margin: float = 0.15  # increased margin for more aggressive matching
    face_create_threshold: float = 0.35  # lowered - prefer reusing existing IDs

    # EMA smoothing parameters
    ema_alpha: float = 0.7  # weight for old embedding (0.7*old + 0.3*new)
    ema_quality_weighted: bool = True  # weight EMA by quality scores


@dataclass
class AssociationConfig:
    max_link_distance_px: int = 120
    iou_threshold: float = 0.2
    time_consistency_frames: int = 8


@dataclass
class BehaviorConfig:
    abandonment_timeout_s: float = 7.0  # suspicious if bag is static and owner gone for this long
    static_iou_threshold: float = 0.8  # bag considered static if current vs last bbox IoU
    owner_distance_px: int = 200  # person considered away if center distance exceeds this


@dataclass
class EventConfig:
    camera_id: str = "CAM_01"
    log_dir: Path = Path("logs")
    enable_file_logging: bool = True


@dataclass
class PersistenceConfig:
    save_faces: bool = True
    save_bags: bool = True
    faces_dir: Path = Path("storage/faces")
    bags_dir: Path = Path("storage/bags")


@dataclass
class PipelineConfig:
    video_source: str | int = 0  # default webcam
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    events: EventConfig = field(default_factory=EventConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)


def select_device(requested: str) -> str:
    """Pick device string depending on availability."""
    try:
        import torch

        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
    except Exception:
        return "cpu"
    return "cpu"
