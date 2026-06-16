import numpy as np

SAMPLE_RATE = 48000
AUDIO_EXT = ".wav"
PITCH_EXT = ".json"
PITCH_NPY_EXT = ".npy"
GT_PITCH_SR = 99.24906132665832
GT_TIME_STEP = 1.0 / GT_PITCH_SR

WINDOW_SIZE_SEC = 0.04644  # ~1 / 20ms pitch floor
HOP_SIZE_SEC = 0.010
WINDOW_SIZE = int(WINDOW_SIZE_SEC * SAMPLE_RATE)
HOP_SIZE = int(HOP_SIZE_SEC * SAMPLE_RATE)

F0_MIN_DEFAULT = 50.0
F0_MAX_DEFAULT = 600.0
F0_CANDIDATE_RESOLUTION = 0.25

NFFT_DEFAULT = 2048

SUBSET_SIZES = {"tiny": 50, "small": 200, "default": 0}

# metrics thresholds (as in MIREX/MIR eval)
GPE_THRESHOLD = 0.20

WINDOW_TYPES = {"hanning", "hamming", "blackman", "boxcar"}

SPECTRAL_THRESHOLD_DB = -40

PITCH_BINS_PER_OCTAVE = 48
PITCH_MIN_MIDI = 16  # ~20 Hz
PITCH_MAX_MIDI = 103  # ~6000 Hz
NUM_PITCH_CANDIDATES = (PITCH_MAX_MIDI - PITCH_MIN_MIDI) * PITCH_BINS_PER_OCTAVE


def midi_to_hz(midi: np.ndarray | float) -> np.ndarray | float:
    return 440.0 * 2.0 ** ((np.asarray(midi) - 69.0) / 12.0)


def hz_to_midi(hz: np.ndarray | float) -> np.ndarray | float:
    return 69.0 + 12.0 * np.log2(np.maximum(np.asarray(hz), 1e-10) / 440.0)
