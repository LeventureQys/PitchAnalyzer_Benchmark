import gc
import time
import psutil
import numpy as np
from typing import Optional
from pitch_bench.base import ProfileResult


def _get_memory_mb() -> float:
    try:
        proc = psutil.Process()
        return proc.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def profile_algorithm(algorithm, recordings: list, audio_cache: Optional[dict] = None) -> ProfileResult:
    total_time = 0.0
    memory_samples: list[float] = []
    peak_memory = 0.0
    n_success = 0

    if audio_cache is None:
        audio_cache = {}

    gc.collect()
    baseline_memory = _get_memory_mb()

    for rec in recordings:
        if rec.name in audio_cache:
            audio, sr = audio_cache[rec.name]
        else:
            from pitch_bench.dataloader import PTDBLoader
            loader = PTDBLoader()
            audio, sr = loader.load_audio(rec)
            audio_cache[rec.name] = (audio, sr)

        start_mem = _get_memory_mb()
        gc.collect()

        t_start = time.perf_counter()
        try:
            result = algorithm.compute(audio, sr)
            elapsed = time.perf_counter() - t_start
        except Exception:
            elapsed = time.perf_counter() - t_start

        total_time += elapsed

        end_mem = _get_memory_mb()
        sample_peak = max(start_mem, end_mem)
        peak_memory = max(peak_memory, sample_peak)
        memory_samples.append((start_mem + end_mem) / 2)
        n_success += 1

    avg_time = total_time / n_success if n_success > 0 else 0.0
    avg_memory = float(np.mean(memory_samples)) if memory_samples else 0.0
    peak_memory_relative = peak_memory - baseline_memory

    return ProfileResult(
        algorithm_name=algorithm.name,
        total_time_sec=total_time,
        avg_time_per_recording_sec=avg_time,
        max_memory_mb=round(peak_memory_relative, 2),
        avg_memory_mb=round(avg_memory - baseline_memory, 2),
        peak_memory_mb=round(peak_memory_relative, 2),
        n_recordings=n_success,
        settings={"f0_min": algorithm.f0_min, "f0_max": algorithm.f0_max,
                  "window_size": algorithm.window_size, "hop_size": algorithm.hop_size},
    )
