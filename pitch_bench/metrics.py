import numpy as np
from pitch_bench.config import GPE_THRESHOLD, GT_TIME_STEP


def _resample_gt_to_target(gt_time: np.ndarray, gt_pitch: np.ndarray,
                           target_time: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(gt_time, target_time, side="right") - 1
    idx = np.clip(idx, 0, len(gt_time) - 2)
    t0 = gt_time[idx]
    t1 = gt_time[idx + 1]
    dt = t1 - t0
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(dt > 0, (target_time - t0) / dt, 0.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    p0 = gt_pitch[idx]
    p1 = gt_pitch[idx + 1]
    gt_resampled = p0 + alpha * (p1 - p0)
    return np.nan_to_num(gt_resampled)


def compute_metrics(est_pitch: np.ndarray, est_voiced: np.ndarray | None,
                    gt_pitch: np.ndarray, gt_time: np.ndarray,
                    est_time: np.ndarray) -> dict:

    if est_voiced is None:
        est_voiced = est_pitch > 0.0
    else:
        est_voiced = np.asarray(est_voiced, dtype=bool)
    gt_voiced = gt_pitch > 0.0

    gt_resampled = _resample_gt_to_target(gt_time, gt_pitch, est_time)
    gt_voiced_resampled = gt_resampled > 0.0

    n_total = len(est_pitch)
    if n_total == 0:
        return _empty_metrics()

    n_both_voiced = int(np.sum(est_voiced & gt_voiced_resampled))
    n_est_voiced = int(np.sum(est_voiced))
    n_gt_voiced = int(np.sum(gt_voiced_resampled))

    gpe_count = 0
    fpe_errors = []
    f0_diffs = []
    f0_gt_vals = []

    for i in range(n_total):
        if est_voiced[i] and gt_voiced_resampled[i]:
            rel_err = abs(est_pitch[i] - gt_resampled[i]) / max(gt_resampled[i], 1.0)
            if rel_err > GPE_THRESHOLD:
                gpe_count += 1
            else:
                fpe_errors.append(est_pitch[i] - gt_resampled[i])
            f0_diffs.append(est_pitch[i] - gt_resampled[i])
            f0_gt_vals.append(gt_resampled[i])
        elif est_voiced[i] and not gt_voiced_resampled[i]:
            gpe_count += 1

    if n_both_voiced > 0:
        gpe_rate = gpe_count / n_both_voiced
    else:
        gpe_rate = 1.0 if gpe_count > 0 else 0.0

    fine_pitch_error = float(np.std(fpe_errors)) if fpe_errors else 0.0
    mean_fine_pitch_error = float(np.mean(fpe_errors)) if fpe_errors else 0.0
    mean_abs_error = float(np.mean(np.abs(f0_diffs))) if f0_diffs else 0.0

    if n_both_voiced > 0:
        vdr = n_both_voiced / n_gt_voiced if n_gt_voiced > 0 else 0.0
    else:
        vdr = 0.0

    n_est_voiced_only = int(np.sum(est_voiced & ~gt_voiced_resampled))
    n_gt_voiced_only = int(np.sum(~est_voiced & gt_voiced_resampled))
    n_neither_voiced = int(np.sum(~est_voiced & ~gt_voiced_resampled))

    vfa = n_est_voiced_only / (n_est_voiced_only + n_neither_voiced) if (n_est_voiced_only + n_neither_voiced) > 0 else 0.0
    vmr = n_gt_voiced_only / n_gt_voiced if n_gt_voiced > 0 else 0.0

    ffe = (n_est_voiced_only + n_gt_voiced_only + gpe_count) / n_total if n_total > 0 else 0.0

    correlation = 0.0
    if n_both_voiced > 0:
        mask = est_voiced & gt_voiced_resampled
        if np.sum(mask) > 1:
            correlation = float(np.corrcoef(est_pitch[mask], gt_resampled[mask])[0, 1])
            correlation = 0.0 if np.isnan(correlation) else correlation

    return {
        "gpe_rate": gpe_rate,
        "fine_pitch_error_hz": fine_pitch_error,
        "mean_fine_pitch_error_hz": mean_fine_pitch_error,
        "mean_abs_error_hz": mean_abs_error,
        "f0_frame_error": ffe,
        "voicing_detection_rate": vdr,
        "voicing_false_alarm": vfa,
        "voicing_miss_rate": vmr,
        "correlation": correlation,
        "n_total_frames": n_total,
        "n_both_voiced": n_both_voiced,
        "n_est_voiced": n_est_voiced,
        "n_gt_voiced": n_gt_voiced,
        "gpe_count": gpe_count,
    }


def aggregate_metrics(all_metrics: list[dict]) -> dict:
    if not all_metrics:
        return _empty_aggregate()

    keys = ["gpe_rate", "fine_pitch_error_hz", "mean_abs_error_hz",
            "f0_frame_error", "voicing_detection_rate",
            "voicing_false_alarm", "voicing_miss_rate", "correlation"]

    agg = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m]
        agg[k + "_mean"] = float(np.mean(vals)) if vals else 0.0
        agg[k + "_std"] = float(np.std(vals)) if vals else 0.0

    total_frames = sum(m.get("n_total_frames", 0) for m in all_metrics)
    total_both_voiced = sum(m.get("n_both_voiced", 0) for m in all_metrics)
    total_gpe = sum(m.get("gpe_count", 0) for m in all_metrics)
    agg["total_recordings"] = len(all_metrics)
    agg["total_frames"] = total_frames
    agg["overall_gpe_rate"] = total_gpe / max(total_both_voiced, 1)
    return agg


def _empty_metrics() -> dict:
    return {
        "gpe_rate": 0.0, "fine_pitch_error_hz": 0.0,
        "mean_fine_pitch_error_hz": 0.0, "mean_abs_error_hz": 0.0,
        "f0_frame_error": 0.0, "voicing_detection_rate": 0.0,
        "voicing_false_alarm": 0.0, "voicing_miss_rate": 0.0,
        "correlation": 0.0, "n_total_frames": 0,
        "n_both_voiced": 0, "n_est_voiced": 0,
        "n_gt_voiced": 0, "gpe_count": 0,
    }


def _empty_aggregate() -> dict:
    keys = ["gpe_rate", "fine_pitch_error_hz", "mean_abs_error_hz",
            "f0_frame_error", "voicing_detection_rate",
            "voicing_false_alarm", "voicing_miss_rate", "correlation"]
    agg = {}
    for k in keys:
        agg[k + "_mean"] = 0.0
        agg[k + "_std"] = 0.0
    agg["total_recordings"] = 0
    agg["total_frames"] = 0
    agg["overall_gpe_rate"] = 0.0
    return agg
