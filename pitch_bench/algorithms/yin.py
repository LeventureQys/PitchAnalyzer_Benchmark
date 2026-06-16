import numpy as np
from pitch_bench.base import PitchAlgorithm, PitchResult
from pitch_bench.algorithms.time_domain import _frame_signal, _parabolic_interp


class YIN(PitchAlgorithm):
    name = "YIN"
    description = "YIN pitch estimator (de Cheveigne & Kawahara)"

    def __init__(self, f0_min: float = 50.0, f0_max: float = 600.0,
                 hop_size: int = 480, window_size: int = 2048,
                 threshold: float = 0.1, **kwargs):
        super().__init__(f0_min, f0_max, hop_size, window_size, **kwargs)
        self.threshold = threshold
        self.window_type = kwargs.pop("window_type", "hanning")

    def _difference_function(self, frame: np.ndarray) -> np.ndarray:
        W = len(frame)
        half = W // 2
        df = np.zeros(half, dtype=np.float64)
        for tau in range(half):
            diff = frame[:W - tau] - frame[tau:W]
            df[tau] = np.dot(diff, diff)
        return df

    def _cumulative_mean_normalized_diff(self, df: np.ndarray) -> np.ndarray:
        cmnd = np.ones_like(df, dtype=np.float64)
        cumsum = np.cumsum(df)
        cmnd[1:] = df[1:] * np.arange(1, len(df)) / np.maximum(cumsum[:-1], 1e-16)
        return cmnd

    def compute(self, audio: np.ndarray, sr: int) -> PitchResult:
        audio = audio.astype(np.float64)
        frames, n_frames = _frame_signal(audio, self.window_size, self.hop_size, self.window_type)
        min_lag = max(1, int(sr / self.f0_max))
        max_lag = min(int(sr / self.f0_min), self.window_size // 2 - 1)
        if max_lag <= min_lag:
            max_lag = min_lag + 1

        times = np.arange(n_frames, dtype=np.float64) * self.hop_size / sr
        pitch = np.zeros(n_frames, dtype=np.float64)
        voiced = np.zeros(n_frames, dtype=bool)
        energy = (frames ** 2).mean(axis=1)
        energy_thresh = energy.max() * 0.01 if energy.max() > 0 else 0

        for i in range(n_frames):
            if energy[i] < energy_thresh:
                continue
            df = self._difference_function(frames[i])
            cmnd = self._cumulative_mean_normalized_diff(df)
            search_end = min(max_lag, len(cmnd) - 1)
            search_start = max(min_lag, 1)

            below_indices = np.where(cmnd[search_start:search_end] < self.threshold)[0]
            if len(below_indices) == 0:
                valley = np.argmin(cmnd[search_start:search_end]) + search_start
            else:
                valley = below_indices[0] + search_start

            refined = _parabolic_interp(cmnd, valley)
            if refined > 1e-6:
                pitch[i] = sr / refined

        voiced = (pitch > self.f0_min) & (pitch < self.f0_max) & (energy > energy_thresh)
        pitch[~voiced] = 0.0
        return PitchResult(time=times, pitch=pitch, voiced=voiced, sample_rate=float(sr))


class pYIN(PitchAlgorithm):
    name = "pYIN"
    description = "Probabilistic YIN - simplified numpy version"

    def __init__(self, f0_min: float = 50.0, f0_max: float = 600.0,
                 hop_size: int = 480, window_size: int = 2048,
                 threshold: float = 0.1, n_candidates: int = 5, **kwargs):
        super().__init__(f0_min, f0_max, hop_size, window_size, **kwargs)
        self.threshold = threshold
        self.n_candidates = n_candidates
        self.window_type = kwargs.pop("window_type", "hanning")

    def _difference_function(self, frame: np.ndarray) -> np.ndarray:
        W = len(frame)
        half = W // 2
        df = np.zeros(half, dtype=np.float64)
        for tau in range(half):
            diff = frame[:W - tau] - frame[tau:W]
            df[tau] = np.dot(diff, diff)
        return df

    def _cmnd(self, df: np.ndarray) -> np.ndarray:
        cmnd = np.ones_like(df, dtype=np.float64)
        cumsum = np.cumsum(df)
        cmnd[1:] = df[1:] * np.arange(1, len(df)) / np.maximum(cumsum[:-1], 1e-16)
        return cmnd

    def compute(self, audio: np.ndarray, sr: int) -> PitchResult:
        audio = audio.astype(np.float64)
        frames, n_frames = _frame_signal(audio, self.window_size, self.hop_size, self.window_type)
        min_lag = max(1, int(sr / self.f0_max))
        max_lag = min(int(sr / self.f0_min), self.window_size // 2 - 1)
        if max_lag <= min_lag:
            max_lag = min_lag + 1

        times = np.arange(n_frames, dtype=np.float64) * self.hop_size / sr
        pitch = np.zeros(n_frames, dtype=np.float64)
        voiced = np.zeros(n_frames, dtype=bool)
        energy = (frames ** 2).mean(axis=1)
        energy_thresh = energy.max() * 0.01 if energy.max() > 0 else 0

        for i in range(n_frames):
            if energy[i] < energy_thresh:
                continue
            df = self._difference_function(frames[i])
            cmnd = self._cmnd(df)
            search_end = min(max_lag, len(cmnd))
            search_start = max(min_lag, 1)

            below = np.where(cmnd[search_start:search_end] < self.threshold)[0]
            if len(below) == 0:
                continue

            valley = below[0] + search_start
            refined = _parabolic_interp(cmnd, valley)
            if refined > 1e-6:
                pitch[i] = sr / refined
            voiced[i] = True

        pitch[~voiced] = 0.0
        return PitchResult(time=times, pitch=pitch, voiced=voiced, sample_rate=float(sr))
