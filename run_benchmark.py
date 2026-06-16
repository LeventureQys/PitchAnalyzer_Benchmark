#!/usr/bin/env python3
import argparse
from pitch_bench.runner import run_benchmark
from pitch_bench.reporter import print_summary
from pitch_bench.algorithms import list_algorithms


def main():
    algo_list = list_algorithms()
    native = algo_list["native"]

    parser = argparse.ArgumentParser(description="Pitch Analyzer Benchmark")
    parser.add_argument("--algorithms", nargs="+", default=None,
                        help=f"Algorithms to test. Available: {native} (default: all native)")
    parser.add_argument("--subset", type=int, default=None,
                        help="Limit number of recordings (default: all)")
    parser.add_argument("--subset-key", default="tiny",
                        choices=["tiny", "small", "default"],
                        help="Predefined subset size (default: tiny)")
    parser.add_argument("--output-dir", default="benchmark_results",
                        help="Output directory")
    parser.add_argument("--speaker-sex", default=None, choices=["F", "M"],
                        help="Filter by speaker sex")
    parser.add_argument("--sentence-type", default=None,
                        choices=["sa", "sx", "si"],
                        help="Filter by sentence type")
    parser.add_argument("--no-cache", action="store_true",
                        help="Don't pre-load audio into memory")
    parser.add_argument("--shuffle", action="store_true",
                        help="Randomly shuffle recordings before selecting subset")
    parser.add_argument("--list-algorithms", action="store_true",
                        help="List available algorithms and exit")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if args.list_algorithms:
        print("Native algorithms:")
        for a in native:
            print(f"  - {a}")
        ext_avail = algo_list["external_available"]
        if ext_avail:
            print("External available:")
            for a in ext_avail:
                print(f"  - {a}")
        ext_unavail = algo_list["external_unavailable"]
        if ext_unavail:
            print("External NOT available (pip install missing):")
            for a in ext_unavail:
                print(f"  - {a}")
        return

    if args.algorithms:
        algorithms = args.algorithms
    else:
        algorithms = native + algo_list["external_available"]
    verbose = not args.quiet

    results = run_benchmark(
        algorithms=algorithms,
        subset=args.subset,
        subset_key=args.subset_key,
        output_dir=args.output_dir,
        speaker_sex=args.speaker_sex,
        sentence_type=args.sentence_type,
        verbose=verbose,
        cache_audio=not args.no_cache,
        shuffle=args.shuffle,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
