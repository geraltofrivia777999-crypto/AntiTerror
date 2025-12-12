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
    face_model: str = "vggface2"  # facenet-pytorch pretrained set
    face_confidence: float = 0.7
    face_provider: str = "insightface"  # "insightface" | "facenet"
    face_new_id_cooldown_s: float = 3.0
    face_new_id_patience_frames: int = 3
    face_switch_patience_frames: int = 3
    bag_model_name: str = "resnet50"
    bag_embedding_size: int = 2048
    bag_similarity_threshold: float = 0.7  # cosine similarity to attach to existing BagID
    face_similarity_threshold: float = 0.8  # cosine similarity for PersonID reuse (stricter)
    face_force_match_margin: float = 0.1  # allow reuse if score >= threshold - margin to reduce dup splits
    face_create_threshold: float = 0.45  # lower barrier to reuse existing IDs


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
