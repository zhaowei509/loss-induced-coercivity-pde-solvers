"""Command-line entry for Experiment 2 preflight, campaigns, and analysis."""

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    campaign = sub.add_parser("campaign")
    campaign.add_argument("--mode", choices=("smoke", "short_calibration",
                                             "calibration", "formal"),
                          required=True)
    campaign.add_argument("--output", type=Path, required=True)
    campaign.add_argument("--smoke-steps", type=int, default=5)
    campaign.add_argument("--resume", action="store_true",
                          help="Verify and retain completed runs, then continue an interrupted campaign.")
    campaign.add_argument("--plan", type=Path, default=ROOT / "EXPERIMENT2_PLAN.md")
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        from experiments.Ex2.checks import run_checks
        run_checks()
    elif args.command == "campaign":
        from experiments.Ex2.experiment import run_campaign
        result = run_campaign(args.output, args.mode, plan=args.plan,
                              smoke_steps=args.smoke_steps, resume=args.resume)
        print(f"experiment2_runs={len(result['records'])}")
    else:
        from experiments.Ex2.analysis import analyze
        report = analyze(args.root)
        print(f"experiment2_analyzed={report['run_count']}")


if __name__ == "__main__":
    main()
