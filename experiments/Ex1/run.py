"""Single command-line entry point for checks, campaigns, and read-only analysis."""

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--output", type=Path, required=True)
    campaign = subparsers.add_parser("campaign")
    campaign.add_argument("--mode", choices=("calibration", "formal"), required=True)
    campaign.add_argument("--output", type=Path, required=True)
    analysis = subparsers.add_parser("analyze")
    analysis.add_argument("--root", type=Path, required=True)
    for child in (check, campaign, analysis):
        child.add_argument("--plan", type=Path, default=ROOT / "EXPERIMENT1_PLAN.md")
    args = parser.parse_args()
    if args.command == "check":
        from experiments.Ex1.checks import run_checks
        report = run_checks(args.output, args.plan)
        print(f"preflight_passed={report['passed']}")
    elif args.command == "campaign":
        from experiments.Ex1.experiment import run_campaign
        summary = run_campaign(args.output, args.mode, args.plan)
        print(f"campaign_records={summary['raw_record_count']}")
    else:
        from experiments.Ex1.analysis import analyze
        summary = analyze(args.root, args.plan)
        print(f"analysis_records={summary['raw_record_count']}")


if __name__ == "__main__":
    raise SystemExit(main())
