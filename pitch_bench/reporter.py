import json
from pathlib import Path
from pitch_bench.algorithms import list_algorithms, EXTERNAL_STATUS


def _format_memory(mb: float) -> str:
    if mb <= 0:
        return "0.0 KB"
    if mb < 1.0:
        return f"{mb * 1024:.1f} KB"
    if mb < 1024.0:
        return f"{mb:.1f} MB"
    return f"{mb / 1024:.2f} GB"


def _sex_label(s: str) -> str:
    return {"F": "女", "M": "男"}.get(s, s)


def generate_report(results: dict, output_path: str = "benchmark_results/report.md"):
    meta = results.get("_metadata", {})
    sys_info = meta.get("system", {})
    ds_info = meta.get("dataset", {})

    lines = []
    lines.append("# 音高分析器基准测试报告")
    lines.append("")

    # ── 测试平台 ──
    lines.append("## 测试平台")
    lines.append("")
    lines.append(f"| 项目 | 详情 |")
    lines.append(f"|------|------|")
    lines.append(f"| 操作系统 | {sys_info.get('platform', '未知')} |")
    lines.append(f"| 系统版本 | {sys_info.get('system', '')} {sys_info.get('release', '')} |")
    lines.append(f"| 架构 | {sys_info.get('machine', '未知')} |")
    lines.append(f"| 处理器 | {sys_info.get('processor', '未知')} |")
    cpu_phys = sys_info.get("cpu_count_physical", 0)
    cpu_logi = sys_info.get("cpu_count_logical", 0)
    if cpu_phys and cpu_logi:
        lines.append(f"| CPU 核心 | {cpu_phys} 物理核 / {cpu_logi} 逻辑核 |")
    elif cpu_logi:
        lines.append(f"| CPU 逻辑核 | {cpu_logi} |")
    ram_total = sys_info.get("ram_total_gb", 0)
    if ram_total:
        lines.append(f"| 内存总量 | {ram_total} GB |")
    lines.append("")

    # ── 测试系统 ──
    lines.append("## 测试系统")
    lines.append("")
    lines.append(f"| 项目 | 详情 |")
    lines.append(f"|------|------|")
    lines.append(f"| Python 版本 | {sys_info.get('python_version', '未知')} ({sys_info.get('python_impl', '')}) |")
    lines.append(f"| 内置算法 | {', '.join(list_algorithms()['native'])} |")
    ext_avail = list_algorithms()["external_available"]
    ext_unavail = list_algorithms()["external_unavailable"]
    lines.append(f"| 外部可用 | {', '.join(ext_avail) if ext_avail else '无'} |")
    lines.append(f"| 外部不可用 | {', '.join(ext_unavail) if ext_unavail else '无'} |")
    n_algo = len([k for k in results if k != "_metadata"])
    lines.append(f"| 已测试算法数 | {n_algo} |")
    lines.append("")

    # ── 测试数据集 ──
    lines.append("## 测试数据集")
    lines.append("")
    lines.append(f"| 项目 | 详情 |")
    lines.append(f"|------|------|")
    lines.append(f"| 数据集名称 | {ds_info.get('name', '未知')} |")
    lines.append(f"| 数据来源 | {ds_info.get('source', '未知')} |")
    lines.append(f"| 数据描述 | {ds_info.get('description', '')} |")
    lines.append(f"| 录音格式 | {ds_info.get('sample_rate_hz', 0)} Hz / {ds_info.get('bit_depth', 0)}-bit |")
    lines.append(f"| 数据集总录音 | {ds_info.get('total_recordings_in_dataset', 0)} 条 |")
    lines.append(f"| 本次使用录音 | {ds_info.get('recordings_used', 0)} 条 |")
    speakers = ds_info.get("speakers_used", [])
    sexes = ds_info.get("sexes", [])
    stypes = ds_info.get("sentence_types", [])
    if speakers:
        sex_str = "/".join(_sex_label(s) for s in sexes) if sexes else "—"
        lines.append(f"| 说话人数 | {ds_info.get('speaker_count', len(speakers))} 人 ({sex_str}) |")
        lines.append(f"| 句子类型 | {', '.join(stypes) if stypes else '—'} |")
    lines.append("")

    # ── 真值统计 ──
    gt = meta.get("ground_truth", {})
    if gt:
        lines.append("## 真值 (Ground Truth) 统计")
        lines.append("")
        lines.append(f"| 项目 | 数值 |")
        lines.append(f"|------|------|")
        lines.append(f"| GT 总帧数 | {gt.get('total_frames', 0)} |")
        lines.append(f"| GT 浊音帧数 | {gt.get('voiced_frames', 0)} |")
        lines.append(f"| GT 浊音比例 | {gt.get('voiced_ratio', 0)*100:.1f}% |")
        lines.append(f"| GT 基频均值 | {gt.get('f0_mean_hz', 0):.1f} Hz |")
        lines.append(f"| GT 基频中位数 | {gt.get('f0_median_hz', 0):.1f} Hz |")
        lines.append(f"| GT 基频标准差 | {gt.get('f0_std_hz', 0):.1f} Hz |")
        lines.append(f"| GT 音频总时长 | {gt.get('total_duration_sec', 0):.1f} s |")
        lines.append("")

    if not results or n_algo == 0:
        lines.append("> 暂无测试结果。")
        lines.append("")
    else:
        # ── 主结果表 ──
        lines.append("## 基准测试结果")
        lines.append("")
        header = "| 算法 | GPE% | FPE (Hz) | MAE (Hz) | FFE% | 浊音检出% | 误判% | 实时率 | 峰值内存 |"
        sep =    "|------|------|----------|----------|------|-----------|-------|--------|----------|"
        lines.append(header)
        lines.append(sep)
        for algo_name, algo_data in results.items():
            if algo_name.startswith("_"):
                continue
            agg = algo_data.get("aggregate_metrics", {})
            summary = algo_data.get("summary", {})
            peak_mem = max(summary.get("peak_memory_mb", 0), 0)
            lines.append(
                f"| {algo_name} "
                f"| {agg.get('gpe_rate_mean', 0)*100:.1f} "
                f"| {agg.get('fine_pitch_error_hz_mean', 0):.2f} "
                f"| {agg.get('mean_abs_error_hz_mean', 0):.2f} "
                f"| {agg.get('f0_frame_error_mean', 0)*100:.1f} "
                f"| {agg.get('voicing_detection_rate_mean', 0)*100:.1f} "
                f"| {agg.get('voicing_false_alarm_mean', 0)*100:.1f} "
                f"| {summary.get('real_time_factor', 0):.3f}x "
                f"| {_format_memory(peak_mem)} |"
            )
        lines.append("")

        # ── RTF 详细统计表 ──
        lines.append("## 实时率 (RTF) 详细统计")
        lines.append("")
        lines.append("| 算法 | 均值 | 标准差 | 中位数 | 最小值 | 最大值 |")
        lines.append("|------|------|--------|--------|--------|--------|")
        for algo_name, algo_data in results.items():
            if algo_name.startswith("_"):
                continue
            agg = algo_data.get("aggregate_metrics", {})
            lines.append(
                f"| {algo_name} "
                f"| {agg.get('rtf_mean', 0):.3f}x "
                f"| {agg.get('rtf_std', 0):.3f} "
                f"| {agg.get('rtf_median', 0):.3f}x "
                f"| {agg.get('rtf_min', agg.get('rtf_mean', 0)):.3f}x "
                f"| {agg.get('rtf_max', agg.get('rtf_mean', 0)):.3f}x |"
            )
        lines.append("")

    # ── 指标说明 ──
    lines.append("## 指标说明")
    lines.append("")
    lines.append("| 指标 | 含义 | 方向 |")
    lines.append("|------|------|------|")
    lines.append("| **GPE** | 粗略音高误差率 (>20% 相对偏差的帧占比) | ↓ 越低越好 |")
    lines.append("| **FPE** | 精细音高误差标准差 (Hz, 仅 GPE 阈值内的帧) | ↓ 越低越好 |")
    lines.append("| **MAE** | 平均绝对误差 (Hz) | ↓ 越低越好 |")
    lines.append("| **FFE** | 综合 F0 帧错误率 | ↓ 越低越好 |")
    lines.append("| **浊音检出** | 浊音段正确检出比例 (VDR) | ↑ 越高越好 |")
    lines.append("| **误判** | 清音误判为浊音比例 (VFA) | ↓ 越低越好 |")
    lines.append("| **实时率** | 处理时间 / 音频时长 (< 1.0 表示快于实时) | ↓ 越低越好 |")
    lines.append("| **峰值内存** | 处理过程中最大内存占用 (自动缩放 KB/MB/GB) | ↓ 越低越好 |")
    lines.append("")

    report = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    return report


def print_summary(results: dict):
    report = generate_report(results, output_path="benchmark_results/report.md")
    print(report)
