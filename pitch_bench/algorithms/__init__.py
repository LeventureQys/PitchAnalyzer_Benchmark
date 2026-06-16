from pitch_bench.base import PitchAlgorithm
from pitch_bench.algorithms.time_domain import Autocorrelation, AMDF
from pitch_bench.algorithms.yin import YIN, pYIN
from pitch_bench.algorithms.freq_domain import Cepstrum, HPS, SHS
from pitch_bench.algorithms.external import PyWorldDIO, PyWorldHarvest, TorchCrepe, LibrosaYIN

_NATIVE_ALGORITHMS: dict[str, type[PitchAlgorithm]] = {
    "acf": Autocorrelation,
    "amdf": AMDF,
    "yin": YIN,
    "pyin": pYIN,
    "cepstrum": Cepstrum,
    "hps": HPS,
    "shs": SHS,
}

_EXTERNAL_ALGORITHMS: dict[str, dict] = {
    "torchcrepe": {"available": False, "cls": None},
    "crepe": {"available": False, "cls": None},
    "pyworld_dio": {"available": False, "cls": None},
    "pyworld_harvest": {"available": False, "cls": None},
    "praat": {"available": False, "cls": None},
    "librosa_yin": {"available": False, "cls": None},
    "aubio": {"available": False, "cls": None},
    "rmvpe": {"available": False, "cls": None},
}


def _check_external():
    try:
        import torchcrepe
        _EXTERNAL_ALGORITHMS["torchcrepe"].update(
            available=True, cls=TorchCrepe,
            version=getattr(torchcrepe, "__version__", "?"),
        )
    except ImportError:
        pass

    try:
        import crepe
        _EXTERNAL_ALGORITHMS["crepe"].update(
            available=True,
            version=getattr(crepe, "__version__", "?"),
        )
    except ImportError:
        pass

    try:
        import pyworld
        _EXTERNAL_ALGORITHMS["pyworld_dio"].update(available=True, cls=PyWorldDIO)
        _EXTERNAL_ALGORITHMS["pyworld_harvest"].update(available=True, cls=PyWorldHarvest)
    except ImportError:
        pass

    try:
        import parselmouth
        _EXTERNAL_ALGORITHMS["praat"].update(
            available=True,
            version=getattr(parselmouth, "__version__", "?"),
        )
    except ImportError:
        pass

    try:
        import librosa
        _EXTERNAL_ALGORITHMS["librosa_yin"].update(available=True, cls=LibrosaYIN)
    except ImportError:
        pass

    try:
        import aubio
        _EXTERNAL_ALGORITHMS["aubio"].update(available=True)
    except ImportError:
        pass

    try:
        import rmvpe
        _EXTERNAL_ALGORITHMS["rmvpe"].update(available=True)
    except ImportError:
        pass


_check_external()

EXTERNAL_STATUS = _EXTERNAL_ALGORITHMS
ALGORITHMS: dict = {**_NATIVE_ALGORITHMS, **_EXTERNAL_ALGORITHMS}


def get_algorithm(name: str, **kwargs) -> PitchAlgorithm:
    key = name.lower().strip()
    if key in _NATIVE_ALGORITHMS:
        return _NATIVE_ALGORITHMS[key](**kwargs)
    if key in _EXTERNAL_ALGORITHMS:
        info = _EXTERNAL_ALGORITHMS[key]
        if info.get("available") and info.get("cls") is not None:
            return info["cls"](**kwargs)
        raise ImportError(
            f"External algorithm '{name}' is not available. "
            f"Please install the required library."
        )
    raise ValueError(
        f"Unknown algorithm: '{name}'. "
        f"Available native: {list(_NATIVE_ALGORITHMS.keys())}. "
        f"External: { {k: v['available'] for k, v in _EXTERNAL_ALGORITHMS.items()} }"
    )


def list_algorithms() -> dict[str, list[str]]:
    available = list(_NATIVE_ALGORITHMS.keys())
    external = [k for k, v in _EXTERNAL_ALGORITHMS.items() if v.get("available")]
    unavailable = [k for k, v in _EXTERNAL_ALGORITHMS.items() if not v.get("available")]
    return {"native": available, "external_available": external, "external_unavailable": unavailable}
