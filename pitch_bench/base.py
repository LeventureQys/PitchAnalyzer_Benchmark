from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PitchResult:
    time: np.ndarray
    pitch: np.ndarray
    voiced: np.ndarray | None = None
    confidence: np.ndarray | None = None
    sample_rate: float = 48000.0

    def __post_init__(self):
        for arr_attr in ("time", "pitch"):
            val = getattr(self, arr_attr)
            if val is None:
                raise ValueError(f"'{arr_attr}' must not be None")
            setattr(self, arr_attr, np.asarray(val, dtype=np.float64))
        if self.voiced is not None:
            self.voiced = np.asarray(self.voiced, dtype=bool)
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=np.float64)

    @property
    def n_frames(self) -> int:
        return len(self.time)

    def get_voiced(self) -> np.ndarray:
        if self.voiced is not None:
            return self.voiced
        return self.pitch > 0.0


class PitchAlgorithm(ABC):
    name: str = "abstract"
    description: str = ""

    def __init__(self, f0_min: float = 50.0, f0_max: float = 600.0,
                 hop_size: int = 480, window_size: int = 2048, **kwargs):
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.hop_size = hop_size
        self.window_size = window_size
        self._params = kwargs

    @abstractmethod
    def compute(self, audio: np.ndarray, sr: int) -> PitchResult:
        ...

    def __repr__(self) -> str:
        return f"{self.name}(f0_min={self.f0_min}, f0_max={self.f0_max}, window={self.window_size}, hop={self.hop_size})"


@dataclass
class ProfileResult:
    algorithm_name: str
    total_time_sec: float
    avg_time_per_recording_sec: float
    max_memory_mb: float
    avg_memory_mb: float
    peak_memory_mb: float
    n_recordings: int
    settings: dict = field(default_factory=dict)
