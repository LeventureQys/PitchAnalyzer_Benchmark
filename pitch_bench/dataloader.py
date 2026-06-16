import json
import random
import wave
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np


@dataclass
class Recording:
    name: str
    signal_path: Path
    pitch_path: Path
    pitch_json_path: Path
    speaker: str
    sentence_type: str  # "sa", "sx", "si"

    @property
    def speaker_sex(self) -> str:
        return self.speaker[0]

    @property
    def speaker_id(self) -> str:
        return self.speaker[1:]


class PTDBLoader:
    def __init__(self, root: str = "datasets/PTDB_TUG"):
        self.root = Path(root)
        self._recording_paths: list[Path] = []
        self._scan()

    def _scan(self):
        self._recording_paths = sorted(
            p for p in self.root.iterdir()
            if p.is_dir() and any(f.suffix == ".wav" for f in p.iterdir())
        )

    @property
    def recording_names(self) -> list[str]:
        return [p.name for p in self._recording_paths]

    def __len__(self) -> int:
        return len(self._recording_paths)

    def __iter__(self) -> Iterator[Recording]:
        for rec_path in self._recording_paths:
            yield self._load_recording(rec_path)

    def __getitem__(self, idx: int) -> Recording:
        return self._load_recording(self._recording_paths[idx])

    def _load_recording(self, rec_path: Path) -> Recording:
        name = rec_path.name
        signal_path = rec_path / "signal.wav"
        pitch_path = rec_path / "pitch.npy"
        pitch_json_path = rec_path / "pitch.json"

        parts = name.split("_", 2)
        if len(parts) == 3:
            speaker = "_".join(parts[:2])
            sentence_type = parts[2][:2]
        else:
            speaker = parts[0]
            sentence_type = parts[1][:2] if len(parts) > 1 else "?"

        return Recording(
            name=name, signal_path=signal_path,
            pitch_path=pitch_path, pitch_json_path=pitch_json_path,
            speaker=speaker, sentence_type=sentence_type,
        )

    def load_audio(self, rec: Recording) -> Tuple[np.ndarray, int]:
        with wave.open(str(rec.signal_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            data = wf.readframes(n_frames)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)
        return samples, sr

    def load_pitch_gt(self, rec: Recording) -> Tuple[np.ndarray, np.ndarray, float]:
        gt = np.load(rec.pitch_path)
        with open(rec.pitch_json_path) as f:
            meta = json.load(f)
        pitch_sr = meta.get("samplerate", 99.24906132665832)
        return gt["time"].copy(), gt["pitch"].copy(), float(pitch_sr)

    def iter_recordings(self, subset: Optional[int] = None, speaker_sex: Optional[str] = None,
                        sentence_type: Optional[str] = None, shuffle: bool = False) -> Iterator[Recording]:
        recordings = list(self)
        if speaker_sex:
            recordings = [r for r in recordings if r.speaker_sex in speaker_sex]
        if sentence_type:
            recordings = [r for r in recordings if r.sentence_type == sentence_type]
        if shuffle:
            rng = random.Random(42)
            rng.shuffle(recordings)
        count = 0
        for rec in recordings:
            yield rec
            count += 1
            if subset and count >= subset:
                break
