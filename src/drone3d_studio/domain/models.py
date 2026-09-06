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
    origin: Literal["Imported", "Demo", "COLMAP"] = "Imported"
    transform: Transform = Field(default_factory=Transform)


class Project(Record):
    schema_version: Literal[1] = 1
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    created: str = Field(default_factory=now)
    modified: str = Field(default_factory=now)
    video: Video | None = None
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
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
        return self
