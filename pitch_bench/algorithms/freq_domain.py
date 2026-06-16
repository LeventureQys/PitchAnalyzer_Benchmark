import numpy as np
from pitch_bench.base import PitchAlgorithm, PitchResult
from pitch_bench.algorithms.time_domain import _frame_signal, _parabolic_interp


class Cepstrum(PitchAlgorithm):
    name = "Cepstrum"
    description = "Cepstral pitch estimation (peak in quefrency domain)"

    def __init__(self, f0_min: float = 50.0, f0_max: float = 600.0,
                 hop_size: int = 480, window_size: int = 2048,
                 nfft: int = 4096, **kwargs):
        super().__init__(f0_min, f0_max, hop_size, window_size, **kwargs)
        self.nfft = nfft or 2 ** int(np.ceil(np.log2(window_size)))
        self.window_type = kwargs.pop("window_type", "hanning")
        self.energy_thresh = kwargs.pop("energy_thresh", 0.02)

    def compute(self, audio: np.ndarray, sr: int) -> PitchResult:
        audio = audio.astype(np.float64)
        frames, n_frames = _frame_signal(audio, self.window_size, self.hop_size, self.window_type)
        min_quef = int(sr / self.f0_max)
        max_quef = int(sr / self.f0_min)
        nfft = self.nfft

        times = np.arange(n_frames, dtype=np.float64) * self.hop_size / sr
        pitch = np.zeros(n_frames, dtype=np.float64)
        voiced = np.zeros(n_frames, dtype=bool)
        energy = (frames ** 2).mean(axis=1)
        energy_thresh = energy.max() * self.energy_thresh if energy.max() > 0 else 0

        for i in range(n_frames):
            if energy[i] < energy_thresh:
                continue
            spectrum = np.abs(np.fft.rfft(frames[i], n=nfft))
            spectrum = np.maximum(spectrum, 1e-12)
            log_spectrum = np.log(spectrum)
            cepstrum_full = np.fft.irfft(log_spectrum)
            if min_quef >= len(cepstrum_full):
                continue
            quef_end = min(max_quef, len(cepstrum_full) - 1)
            quef_start = max(min_quef, 1)
            if quef_end <= quef_start:
                continue
            peak_idx = np.argmax(cepstrum_full[quef_start:quef_end]) + quef_start
            refined = _parabolic_interp(cepstrum_full, peak_idx)
            if refined > 1e-6:
                pitch[i] = sr / refined

        voiced = (pitch > self.f0_min) & (pitch < self.f0_max) & (energy > energy_thresh)
        pitch[~voiced] = 0.0
        return PitchResult(time=times, pitch=pitch, voiced=voiced, sample_rate=float(sr))


class HPS(PitchAlgorithm):
    name = "HPS"
    description = "Harmonic Product Spectrum pitch estimator"

    def __init__(self, f0_min: float = 50.0, f0_max: float = 600.0,
                 hop_size: int = 480, window_size: int = 2048,
                 n_harmonics: int = 5, nfft: int = 4096, **kwargs):
        super().__init__(f0_min, f0_max, hop_size, window_size, **kwargs)
        self.n_harmonics = n_harmonics
        self.nfft = nfft or 2 ** int(np.ceil(np.log2(window_size)))
        self.window_type = kwargs.pop("window_type", "hanning")
        self.energy_thresh = kwargs.pop("energy_thresh", 0.02)

    def compute(self, audio: np.ndarray, sr: int) -> PitchResult:
        audio = audio.astype(np.float64)
        frames, n_frames = _frame_signal(audio, self.window_size, self.hop_size, self.window_type)
        min_bin = int(self.f0_min * self.nfft / sr)
        max_bin = int(self.f0_max * self.nfft / sr)
        nfft = self.nfft

        times = np.arange(n_frames, dtype=np.float64) * self.hop_size / sr
        pitch = np.zeros(n_frames, dtype=np.float64)
        voiced = np.zeros(n_frames, dtype=bool)
        energy = (frames ** 2).mean(axis=1)
        energy_thresh = energy.max() * self.energy_thresh if energy.max() > 0 else 0

        for i in range(n_frames):
            if energy[i] < energy_thresh:
                continue
            mag = np.abs(np.fft.rfft(frames[i], n=nfft))
            hps_spectrum = mag.copy()[:max_bin]
            for h in range(2, self.n_harmonics + 1):
                downsampled = mag[::h]
                n_ds = min(len(downsampled), max_bin)
                hps_spectrum[:n_ds] *= downsampled[:n_ds]
            hps_spectrum[:min_bin] = 0
            peak_bin = np.argmax(hps_spectrum[min_bin:max_bin]) + min_bin
            refined = _parabolic_interp(hps_spectrum, peak_bin)
            if refined > 0:
                pitch[i] = sr * refined / nfft

        voiced = (pitch > self.f0_min) & (pitch < self.f0_max) & (energy > energy_thresh)
        pitch[~voiced] = 0.0
        return PitchResult(time=times, pitch=pitch, voiced=voiced, sample_rate=float(sr))


class SHS(PitchAlgorithm):
    name = "SHS"
    description = "Subharmonic Summation pitch estimator (Hermes 1988)"

    def __init__(self, f0_min: float = 50.0, f0_max: float = 600.0,
                 hop_size: int = 480, window_size: int = 2048,
                 n_harmonics: int = 15, compression: float = 0.5, nfft: int = 4096, **kwargs):
        super().__init__(f0_min, f0_max, hop_size, window_size, **kwargs)
        self.n_harmonics = n_harmonics
        self.compression = compression
        self.nfft = nfft or 2 ** int(np.ceil(np.log2(window_size)))
        self.window_type = kwargs.pop("window_type", "hanning")
        self.energy_thresh = kwargs.pop("energy_thresh", 0.02)

    def compute(self, audio: np.ndarray, sr: int) -> PitchResult:
        audio = audio.astype(np.float64)
        frames, n_frames = _frame_signal(audio, self.window_size, self.hop_size, self.window_type)
        min_bin = int(self.f0_min * self.nfft / sr)
        max_bin = int(self.f0_max * self.nfft / sr)
        nfft = self.nfft

        times = np.arange(n_frames, dtype=np.float64) * self.hop_size / sr
        pitch = np.zeros(n_frames, dtype=np.float64)
        voiced = np.zeros(n_frames, dtype=bool)
        energy = (frames ** 2).mean(axis=1)
        energy_thresh = energy.max() * self.energy_thresh if energy.max() > 0 else 0

        for i in range(n_frames):
            if energy[i] < energy_thresh:
                continue
            mag = np.abs(np.fft.rfft(frames[i], n=nfft))
            mag_compressed = mag ** self.compression
            shs = np.zeros(max_bin, dtype=np.float64)
            for h in range(1, self.n_harmonics + 1):
                weight = 0.84 ** (h - 1)
                if h == 1:
                    shs += weight * mag_compressed[:max_bin]
                else:
                    n = min(max_bin, len(mag_compressed) // h)
                    if n > 0:
                        shs[:n] += weight * mag_compressed[h::h][:n]
            shs[:min_bin] = 0
            peak_bin = np.argmax(shs[min_bin:max_bin]) + min_bin
            refined = _parabolic_interp(shs, peak_bin)
            if refined > 0:
                pitch[i] = sr * refined / nfft

        voiced = (pitch > self.f0_min) & (pitch < self.f0_max) & (energy > energy_thresh)
        pitch[~voiced] = 0.0
        return PitchResult(time=times, pitch=pitch, voiced=voiced, sample_rate=float(sr))
