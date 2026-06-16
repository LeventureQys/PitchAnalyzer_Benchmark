import numpy as np
from pitch_bench.base import PitchAlgorithm, PitchResult
from pitch_bench.config import WINDOW_TYPES


def _frame_signal(signal: np.ndarray, window_size: int, hop_size: int,
                  window_type: str = "hanning") -> np.ndarray:
    n_frames = (len(signal) - window_size) // hop_size + 1
    if n_frames <= 0:
        n_frames = 1
        window_size = len(signal)
        hop_size = window_size
    shape = (n_frames, window_size)
    strides = (signal.strides[0] * hop_size, signal.strides[0])
    frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides).copy()
    window_fn = _get_window(window_size, window_type)
    frames *= window_fn[np.newaxis, :]
    return frames, n_frames


def _get_window(size: int, wtype: str = "hanning") -> np.ndarray:
    wtype = wtype.lower()
    if wtype == "hanning":
        return np.hanning(size + 1)[:size]
    elif wtype == "hamming":
        return np.hamming(size)
    elif wtype == "blackman":
        return np.blackman(size)
    elif wtype in ("boxcar", "rectangular", "none"):
        return np.ones(size)
    else:
        return np.hanning(size + 1)[:size]


def _parabolic_interp(y: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(y) - 1:
        return float(idx)
    a = y[idx - 1]
    b = y[idx]
    c = y[idx + 1]
    denom = a - 2 * b + c
    if abs(denom) < 1e-12:
        return float(idx)
    return idx + 0.5 * (a - c) / denom


def _voiced_decision(pitch_candidates: np.ndarray, energy: np.ndarray,
                     energy_thresh_factor: float = 0.1,
                     silence_thresh_factor: float = 0.01) -> np.ndarray:
    if len(energy) == 0:
        return np.array([], dtype=bool)
    global_max = energy.max() if energy.max() > 0 else 1.0
    energy_thresh = global_max * energy_thresh_factor
    silence_thresh = global_max * silence_thresh_factor
    voiced = energy > energy_thresh
    silence = energy < silence_thresh
    top = silence.copy()
    top[:-1] = top[:-1] | silence[1:]
    top[1:] = top[1:] | silence[:-1]
    return voiced  # could refine further


class Autocorrelation(PitchAlgorithm):
    name = "ACF"
    description = "Autocorrelation Function (ACF) pitch estimator"

    def __init__(self, f0_min: float = 50.0, f0_max: float = 600.0,
                 hop_size: int = 480, window_size: int = 2048, **kwargs):
        super().__init__(f0_min, f0_max, hop_size, window_size, **kwargs)
        self.window_type = kwargs.pop("window_type", "hanning")
        self.energy_thresh = kwargs.pop("energy_thresh", 0.05)

    def compute(self, audio: np.ndarray, sr: int) -> PitchResult:
        audio = audio.astype(np.float64)
        frames, n_frames = _frame_signal(audio, self.window_size, self.hop_size, self.window_type)
        min_lag = int(sr / self.f0_max)
        max_lag = int(sr / self.f0_min)
        if max_lag >= self.window_size:
            max_lag = self.window_size - 1

        times = np.arange(n_frames, dtype=np.float64) * self.hop_size / sr
        pitch = np.zeros(n_frames, dtype=np.float64)
        voiced = np.zeros(n_frames, dtype=bool)
        energy = (frames ** 2).mean(axis=1)

        for i in range(n_frames):
            if energy[i] < 1e-12:
                continue
            frame = frames[i]
            corr = np.correlate(frame, frame, mode="full")
            acf = corr[len(corr)//2:len(corr)//2 + max_lag + 1]
            valid_range = slice(min_lag, max_lag + 1)
            peak_lag = int(np.argmax(acf[valid_range])) + min_lag
            refined_lag = _parabolic_interp(acf, peak_lag)
            if refined_lag > 0 and acf[peak_lag] > 0:
                pitch[i] = sr / refined_lag

        voiced = (pitch > self.f0_min) & (pitch < self.f0_max) & (energy > energy.max() * self.energy_thresh)
        pitch[~voiced] = 0.0
        return PitchResult(time=times, pitch=pitch, voiced=voiced, sample_rate=float(sr))


class AMDF(PitchAlgorithm):
    name = "AMDF"
    description = "Average Magnitude Difference Function pitch estimator"

    def __init__(self, f0_min: float = 50.0, f0_max: float = 600.0,
                 hop_size: int = 480, window_size: int = 2048, **kwargs):
        super().__init__(f0_min, f0_max, hop_size, window_size, **kwargs)
        self.window_type = kwargs.pop("window_type", "hanning")
        self.energy_thresh = kwargs.pop("energy_thresh", 0.05)

    def compute(self, audio: np.ndarray, sr: int) -> PitchResult:
        audio = audio.astype(np.float64)
        frames, n_frames = _frame_signal(audio, self.window_size, self.hop_size, self.window_type)
        min_lag = int(sr / self.f0_max)
        max_lag = int(sr / self.f0_min)
        if max_lag >= self.window_size:
            max_lag = self.window_size - 1

        times = np.arange(n_frames, dtype=np.float64) * self.hop_size / sr
        pitch = np.zeros(n_frames, dtype=np.float64)
        energy = (frames ** 2).mean(axis=1)

        for i in range(n_frames):
            if energy[i] < 1e-12:
                continue
            frame = frames[i]
            W = self.window_size
            amdf = np.zeros(max_lag + 1, dtype=np.float64)
            for lag in range(1, max_lag + 1):
                amdf[lag] = np.abs(frame[:W - lag] - frame[lag:W]).mean()
            valid_range = slice(min_lag, max_lag + 1)
            valley = int(np.argmin(amdf[valid_range])) + min_lag
            refined = _parabolic_interp(-amdf, valley)
            if refined > 0:
                pitch[i] = sr / refined

        voiced = (pitch > self.f0_min) & (pitch < self.f0_max) & (energy > energy.max() * self.energy_thresh)
        pitch[~voiced] = 0.0
        return PitchResult(time=times, pitch=pitch, voiced=voiced, sample_rate=float(sr))
