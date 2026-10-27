"""Run reproducible Town05 receiving-gap cases with collector Python."""

import argparse
import subprocess
import sys
from pathlib import Path


CASES = (
    ("dense12_avoid_left_fast_rear", "avoid_left", "fast_rear", 12, .55, -1, 500),
    ("dense24_avoid_right_fast_rear", "avoid_right", "fast_rear", 24, .55, -1, 650),
    ("dense12_recover_left_parallel", "recover_left", "parallel", 12, 3.5, 100, 500),
    ("dense24_recover_right_parallel", "recover_right", "parallel", 24, 3.5, 100, 650),
    ("dense12_avoid_left_gap_opens", "avoid_left", "occupied", 12, .55, 120, 750),
    ("dense24_avoid_right_late_intrusion", "avoid_right", "late_intrusion", 24, .55, 100, 850),
    ("dense24_both_sides_blocked", "avoid_left", "occupied", 24, .55, -1, 220),
    ("dense24_early_recover_left", "recover_left", "none", 24, .55, -1, 300),
    ("dense24_early_recover_right", "recover_right", "none", 24, .55, -1, 300),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    recorder = Path(__file__).with_name("route_recovery_scenarios.py")
    for name, case, mode, density, offset, release, steps in CASES:
        if args.cases and name not in args.cases:
            continue
        print("scenario", name, flush=True)
        subprocess.run([sys.executable, str(recorder), "--case", case, "--traffic-mode", mode,
                        "--background-vehicles", str(density), "--recovery-offset", str(offset),
                        "--release-step", str(release), "--steps", str(steps), "--seed", "0",
                        "--reset-world", "--output", str(args.output / "records" / (name + ".json"))],
                       check=True)


if __name__ == "__main__":
    main()
