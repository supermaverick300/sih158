"""Versioned, strict JSON data. Runtime meshes and Qt objects never enter here."""
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class AnalysisConfig(Record):
    interval: float = Field(default=1.0, ge=0.05, le=3600)
    blur_threshold: float = Field(default=35.0, ge=0, le=100000)
    duplicate_threshold: float = Field(default=2.5, ge=0, le=255)
    max_frames: int = Field(default=150, ge=1, le=2000)
    sampling_mode: Literal["Fixed", "Adaptive"] = "Fixed"
    min_features: int = Field(default=0, ge=0, le=2000)
    max_clipped_fraction: float = Field(default=1, ge=0, le=1)
    gps_spacing_m: float = Field(default=0, ge=0, le=1000)
    telemetry_offset_s: float = Field(default=0, ge=-86400, le=86400)
    telemetry_max_gap_s: float = Field(default=2, gt=0, le=60)


class GPSFix(Record):
    time: float = Field(ge=0)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    altitude_m: float


class PipelineConfig(Record):
    depth_method: Literal["COLMAP stereo", "Depth Anything V2"] = "COLMAP stereo"
    weights_path: str = "models/depth-anything-v2-small"
    device: Literal["Auto", "CPU", "CUDA"] = "Auto"
    align_gps: bool = False
    alignment_max_error_m: float = Field(default=5, gt=0, le=100)
    fusion_voxel_size: float = Field(default=0.05, gt=0, le=100)
    depth_stride: int = Field(default=4, ge=2, le=32)


class Settings(Record):
    theme: Literal["Dark", "Light"] = "Dark"
    autosave_ms: int = Field(default=800, ge=200, le=60000)
    default_project_directory: str = ""
    reconstruction_mode: Literal["Demo", "COLMAP"] = "COLMAP"
    reconstruction_output: Literal["Sparse cloud (CPU)", "Dense mesh (CUDA)"] = "Dense mesh (CUDA)"
    executable: str = ""
    background: str = Field(default="#101923", pattern=r"^#[0-9a-fA-F]{6}$")
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class Video(Record):
    path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    duration: float = Field(gt=0)
    size: int = Field(ge=0)


class Frame(Record):
    index: int = Field(ge=0)
    time: float = Field(ge=0)
    path: str
    thumbnail: str
    blur: float = Field(ge=0)
    accepted: bool
    reason: str = ""
    features: int = Field(default=0, ge=0)
    clipped_fraction: float = Field(default=0, ge=0, le=1)
    quality_score: float = Field(default=0, ge=0, le=100)
    motion_px: float = Field(default=0, ge=0)
    gps: GPSFix | None = None


class Transform(Record):
    position: tuple[float, float, float] = (0, 0, 0)
    rotation: tuple[float, float, float] = (0, 0, 0)
    scale: tuple[float, float, float] = (1, 1, 1)

    @model_validator(mode="after")
    def positive_scale(self):
        if any(s <= 0 or s > 10000 for s in self.scale):
            raise ValueError("scale: all axes must be greater than 0 and at most 10000")
        return self


class Model(Record):
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str = Field(min_length=1, max_length=200)
    path: str
    format: str
    vertices: int = Field(ge=0)
    faces: int = Field(ge=0)
    visible: bool = True
    status: str = "Ready"
    origin: Literal["Imported", "Demo", "COLMAP", "Depth Anything V2"] = "Imported"
    transform: Transform = Field(default_factory=Transform)


class Project(Record):
    schema_version: Literal[1] = 1
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    created: str = Field(default_factory=now)
    modified: str = Field(default_factory=now)
    video: Video | None = None
    analysis: AnalysisConfig = Field(default_factory=lambda: AnalysisConfig(sampling_mode="Adaptive", min_features=40, max_clipped_fraction=0.35))
    telemetry: list[GPSFix] = Field(default_factory=list)
    telemetry_source: str = ""
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    frames: list[Frame] = Field(default_factory=list)
    analysis_seconds: float = Field(default=0, ge=0)
    status: Literal["No video", "Ready for analysis", "Analysis running", "Ready for reconstruction", "Reconstruction running", "Scene ready", "Failed", "Cancelled"] = "No video"
    reconstruction_status: str = "Not started"
    models: list[Model] = Field(default_factory=list)
    settings: Settings = Field(default_factory=Settings)
    logs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_models(self):
        if len({m.id for m in self.models}) != len(self.models):
            raise ValueError("models: IDs must be unique")
        if any(b.time <= a.time for a, b in zip(self.telemetry, self.telemetry[1:])):
            raise ValueError("telemetry: timestamps must be strictly increasing")
        return self
