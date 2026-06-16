import gc
import os
import sys
import time
import platform
from typing import Optional
from pathlib import Path
import json

import numpy as np
from tqdm import tqdm

from pitch_bench.base import PitchAlgorithm, PitchResult, ProfileResult
from pitch_bench.dataloader import PTDBLoader, Recording
from pitch_bench.metrics import compute_metrics, aggregate_metrics
from pitch_bench.profiler import profile_algorithm, _get_memory_mb
from pitch_bench.algorithms import list_algorithms, get_algorithm, EXTERNAL_STATUS
from pitch_bench.config import SUBSET_SIZES


def run_benchmark(
    algorithms: list[str],
    subset: Optional[int] = None,
    subset_key: str = "default",
    output_dir: str = "benchmark_results",
    speaker_sex: Optional[str] = None,
    sentence_type: Optional[str] = None,
    verbose: bool = True,
    cache_audio: bool = True,
    shuffle: bool = False,
) -> dict:
    if subset is None and subset_key in SUBSET_SIZES:
        subset = SUBSET_SIZES[subset_key]
        if subset == 0:
            subset = None

    loader = PTDBLoader()
    recordings = list(loader.iter_recordings(subset=subset, speaker_sex=speaker_sex,
                                              sentence_type=sentence_type, shuffle=shuffle))
    if verbose:
        print(f"数据集: {len(recordings)} 条录音"
              + (f" (子集={subset})" if subset else " (全部)"))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_cache = {}
    if cache_audio:
        if verbose:
            print("预加载音频到内存...")
        for rec in tqdm(recordings, desc="加载音频", unit="文件"):
            audio, sr = loader.load_audio(rec)
            audio_cache[rec.name] = (audio, sr)

    results_all = {}

    gt_all_f0 = []
    gt_all_voiced = []
    gt_total_duration = 0.0
    for rec in tqdm(recordings, desc="加载GT数据", unit="文件", disable=not verbose) if verbose else recordings:
        gt_time, gt_pitch, gt_sr = loader.load_pitch_gt(rec)
        gt_voiced = gt_pitch > 0
        gt_all_f0.extend(gt_pitch[gt_voiced].tolist())
        gt_all_voiced.extend(gt_voiced.tolist())
        gt_total_duration += gt_time[-1] - gt_time[0] if len(gt_time) > 0 else 0
    gt_f0_arr = np.array(gt_all_f0) if gt_all_f0 else np.array([0.0])
    gt_stats = {
        "total_frames": len(gt_all_voiced),
        "voiced_frames": int(np.sum(gt_all_voiced)),
        "voiced_ratio": float(np.mean(gt_all_voiced)) if gt_all_voiced else 0.0,
        "f0_mean_hz": float(np.mean(gt_f0_arr)),
        "f0_std_hz": float(np.std(gt_f0_arr)),
        "f0_median_hz": float(np.median(gt_f0_arr)),
        "total_duration_sec": round(gt_total_duration, 1),
    }

    for algo_name in algorithms:
        if verbose:
            print(f"\n{'='*60}")
            print(f"算法: {algo_name}")
            print(f"{'='*60}")

        try:
            algorithm = get_algorithm(algo_name)
        except ValueError as e:
            if verbose:
                print(f"  跳过: {e}")
            continue

        algo_results = {}
        per_recording_metrics = []
        total_audio_sec = 0.0
        total_proc_time = 0.0
        peak_memory_mb = 0.0

        algo_results["config"] = {
            "name": algorithm.name,
            "description": algorithm.description,
            "f0_min": algorithm.f0_min,
            "f0_max": algorithm.f0_max,
            "window_size": algorithm.window_size,
            "hop_size": algorithm.hop_size,
        }

        gc.collect()
        memory_baseline = _get_memory_mb()

        if verbose:
            pbar = tqdm(recordings, desc=f"  处理中", unit="条")
        else:
            pbar = recordings

        for rec in pbar:
            if rec.name in audio_cache:
                audio, sr = audio_cache[rec.name]
            else:
                audio, sr = loader.load_audio(rec)

            gc.collect()
            mem_before = _get_memory_mb()

            t0 = time.perf_counter()
            try:
                result = algorithm.compute(audio, int(sr))
            except Exception as e:
                if verbose:
                    tqdm.write(f"  错误 [{rec.name}]: {e}")
                continue
            elapsed = time.perf_counter() - t0

            mem_after = _get_memory_mb()
            rec_peak = max(mem_before, mem_after) - memory_baseline
            peak_memory_mb = max(peak_memory_mb, rec_peak)

            duration_sec = len(audio) / sr
            rtf = elapsed / max(duration_sec, 1e-6)

            gt_time, gt_pitch, gt_sr = loader.load_pitch_gt(rec)

            metrics = compute_metrics(result.pitch, result.get_voiced(),
                                      gt_pitch, gt_time, result.time)
            metrics["recording"] = rec.name
            metrics["speaker"] = rec.speaker
            metrics["duration_sec"] = duration_sec
            metrics["proc_time_sec"] = elapsed
            metrics["rtf"] = rtf
            per_recording_metrics.append(metrics)

            total_audio_sec += duration_sec
            total_proc_time += elapsed

        agg = aggregate_metrics(per_recording_metrics)

        rtf_values = [m["rtf"] for m in per_recording_metrics if "rtf" in m]
        if rtf_values:
            rtf_arr = np.array(rtf_values)
            agg["rtf_mean"] = round(float(np.mean(rtf_arr)), 4)
            agg["rtf_std"] = round(float(np.std(rtf_arr)), 4)
            agg["rtf_median"] = round(float(np.median(rtf_arr)), 4)
            agg["rtf_min"] = round(float(np.min(rtf_arr)), 4)
            agg["rtf_max"] = round(float(np.max(rtf_arr)), 4)
        else:
            agg["rtf_mean"] = 0.0
            agg["rtf_std"] = 0.0
            agg["rtf_median"] = 0.0
            agg["rtf_min"] = 0.0
            agg["rtf_max"] = 0.0

        algo_results["aggregate_metrics"] = agg
        algo_results["per_recording_metrics"] = per_recording_metrics
        algo_results["summary"] = {
            "total_audio_seconds": round(total_audio_sec, 1),
            "total_proc_seconds": round(total_proc_time, 2),
            "real_time_factor": round(total_proc_time / max(total_audio_sec, 1e-6), 3),
            "peak_memory_mb": round(max(peak_memory_mb, 0), 2),
            "peak_memory_str": _format_memory(max(peak_memory_mb, 0)),
            "n_recordings_processed": len(per_recording_metrics),
        }

        if verbose:
            print_summary_line(algo_name, agg, algo_results["summary"])

        results_all[algo_name] = algo_results

    results_all["_metadata"] = {
        "system": _collect_system_info(),
        "dataset": _collect_dataset_info(loader, recordings),
        "ground_truth": gt_stats,
    }

    results_file = output_path / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    _save_json(str(results_file), results_all)
    if verbose:
        print(f"\n结果已保存至: {results_file}")

    return results_all


def _collect_system_info() -> dict:
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "未知",
        "python_version": sys.version.split()[0],
        "python_impl": platform.python_implementation(),
    }
    try:
        import psutil
        info["cpu_count_physical"] = psutil.cpu_count(logical=False) or 0
        info["cpu_count_logical"] = psutil.cpu_count(logical=True) or 0
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024**3), 1)
        info["ram_available_gb"] = round(mem.available / (1024**3), 1)
    except Exception:
        info["cpu_count_physical"] = os.cpu_count() or 0
        info["cpu_count_logical"] = os.cpu_count() or 0
        info["ram_total_gb"] = 0
        info["ram_available_gb"] = 0
    return info


def _collect_dataset_info(loader: PTDBLoader, recordings: list) -> dict:
    speakers = sorted(set(r.speaker for r in recordings))
    sexes = sorted(set(r.speaker_sex for r in recordings))
    sentence_types = sorted(set(r.sentence_type for r in recordings))
    durations = [len(loader.load_audio(r)[0]) / r.signal_path.stat().st_size
                 for r in recordings[:1]]  # placeholder, will compute from actual data
    return {
        "name": "PTDB-TUG",
        "source": "Graz University of Technology (TU Graz)",
        "description": "音高跟踪数据库 — 20位说话人朗读 TIMIT 句子的喉麦克风录音",
        "total_recordings_in_dataset": len(loader),
        "recordings_used": len(recordings),
        "speakers_used": speakers,
        "speaker_count": len(speakers),
        "sexes": sexes,
        "sentence_types": sentence_types,
        "sample_rate_hz": 48000,
        "bit_depth": 16,
    }


def _format_memory(mb: float) -> str:
    if mb <= 0:
        return "0.0 KB"
    if mb < 1.0:
        return f"{mb * 1024:.1f} KB"
    if mb < 1024.0:
        return f"{mb:.1f} MB"
    return f"{mb / 1024:.2f} GB"


def print_summary_line(name: str, agg: dict, summary: dict):
    print(f"\n  [{name}]")
    print(f"    录音数:               {summary['n_recordings_processed']}")
    print(f"    实时率:               {summary['real_time_factor']:.3f}x  (1x = 实时)")
    print(f"    峰值内存:             {_format_memory(max(summary['peak_memory_mb'], 0))}")
    print(f"    粗略音高误差 GPE:      {agg.get('gpe_rate_mean', 0)*100:.1f}%  (越低越好)")
    print(f"    精细音高误差 FPE:      {agg.get('fine_pitch_error_hz_mean', 0):.2f} Hz  (越低越好)")
    print(f"    平均绝对误差 MAE:      {agg.get('mean_abs_error_hz_mean', 0):.2f} Hz  (越低越好)")
    print(f"    综合帧错误 FFE:        {agg.get('f0_frame_error_mean', 0)*100:.1f}%  (越低越好)")
    print(f"    浊音检出率 VDR:        {agg.get('voicing_detection_rate_mean', 0)*100:.1f}%  (越高越好)")
    print(f"    清音误判率 VFA:        {agg.get('voicing_false_alarm_mean', 0)*100:.1f}%  (越低越好)")
    print(f"    浊音漏检率 VMR:        {agg.get('voicing_miss_rate_mean', 0)*100:.1f}%  (越低越好)")
    print(f"    相关性:               {agg.get('correlation_mean', 0):.3f}  (越接近1越好)")
    print(f"    实时率 RTF 均值:       {agg.get('rtf_mean', 0):.3f}x  (每录音均值)")
    print(f"    实时率 RTF 标准差:     {agg.get('rtf_std', 0):.3f}")


def _save_json(path: str, data: dict):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    with open(path, "w") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)
