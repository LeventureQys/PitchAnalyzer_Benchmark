import numpy as np
from pitch_bench.base import PitchAlgorithm, PitchResult


class PyWorldDIO(PitchAlgorithm):
    name = "PyWorld DIO"
    description = "DIO pitch estimator (WORLD vocoder) with StoneMask refinement"

    def __init__(self, f0_min=50.0, f0_max=600.0, hop_size=480, window_size=2048, **kwargs):
        super().__init__(f0_min, f0_max, hop_size, window_size, **kwargs)
        import pyworld
        self._pw = pyworld

    def compute(self, audio: np.ndarray, sr: int) -> PitchResult:
        audio = audio.astype(np.float64)
        frame_period = self.hop_size / sr * 1000.0
        f0, t = self._pw.dio(
            audio, sr,
            f0_floor=self.f0_min, f0_ceil=self.f0_max,
            frame_period=frame_period,
        )
        f0 = self._pw.stonemask(audio, f0, t, sr)
        voiced = f0 > 0.0
        f0[~voiced] = 0.0
        return PitchResult(
            time=t.astype(np.float64), pitch=f0.astype(np.float64),
            voiced=voiced, sample_rate=float(sr),
        )


class PyWorldHarvest(PitchAlgorithm):
    name = "PyWorld HARVEST"
    description = "HARVEST pitch estimator (WORLD vocoder)"

    def __init__(self, f0_min=50.0, f0_max=600.0, hop_size=480, window_size=2048, **kwargs):
        super().__init__(f0_min, f0_max, hop_size, window_size, **kwargs)
        import pyworld
        self._pw = pyworld

    def compute(self, audio: np.ndarray, sr: int) -> PitchResult:
        audio = audio.astype(np.float64)
        frame_period = self.hop_size / sr * 1000.0
        f0, t = self._pw.harvest(
            audio, sr,
            f0_floor=self.f0_min, f0_ceil=self.f0_max,
            frame_period=frame_period,
        )
        voiced = f0 > 0.0
        f0[~voiced] = 0.0
        return PitchResult(
            time=t.astype(np.float64), pitch=f0.astype(np.float64),
            voiced=voiced, sample_rate=float(sr),
        )


class TorchCrepe(PitchAlgorithm):
    name = "TorchCrepe"
    description = "CREPE deep-learning pitch estimator via torchcrepe"

    def __init__(self, f0_min=50.0, f0_max=600.0, hop_size=480, window_size=2048,
                 model="full", device="cpu", **kwargs):
        super().__init__(f0_min, f0_max, hop_size, window_size, **kwargs)
        self.model = model
        self.device = device
        import torchcrepe
        import torch
        self._tc = torchcrepe
        self._torch = torch

    def compute(self, audio: np.ndarray, sr: int) -> PitchResult:
        audio_t = self._torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(self.device)
        pitch, periodicity = self._tc.predict(
            audio_t, sr,
            hop_length=self.hop_size,
            fmin=self.f0_min, fmax=self.f0_max,
            model=self.model,
            return_periodicity=True,
            device=self.device,
        )
        pitch = pitch.squeeze(0).cpu().numpy().astype(np.float64)
        periodicity = periodicity.squeeze(0).cpu().numpy().astype(np.float64)
        voiced = pitch > 0.0
        pitch[~voiced] = 0.0
        n_frames = len(pitch)
        times = np.arange(n_frames, dtype=np.float64) * self.hop_size / sr
        return PitchResult(
            time=times, pitch=pitch, voiced=voiced,
            confidence=periodicity, sample_rate=float(sr),
        )


class LibrosaYIN(PitchAlgorithm):
    name = "Librosa YIN"
    description = "Probabilistic YIN pitch estimator via librosa"

    def __init__(self, f0_min=50.0, f0_max=600.0, hop_size=480, window_size=2048, **kwargs):
        super().__init__(f0_min, f0_max, hop_size, window_size, **kwargs)
        import librosa
        self._librosa = librosa

    def compute(self, audio: np.ndarray, sr: int) -> PitchResult:
        audio = audio.astype(np.float64)
        f0, voiced_flag, voiced_prob = self._librosa.pyin(
            audio, fmin=self.f0_min, fmax=self.f0_max, sr=sr,
            frame_length=self.window_size, hop_length=self.hop_size,
        )
        f0 = np.nan_to_num(f0, nan=0.0)
        voiced_flag = np.asarray(voiced_flag, dtype=bool)
        n_frames = len(f0)
        times = np.arange(n_frames, dtype=np.float64) * self.hop_size / sr
        return PitchResult(
            time=times, pitch=f0, voiced=voiced_flag,
            confidence=voiced_prob.astype(np.float64), sample_rate=float(sr),
        )
