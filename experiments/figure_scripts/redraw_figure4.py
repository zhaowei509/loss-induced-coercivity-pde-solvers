"""Redraw manuscript Figure 4 with equation references from LaTeX labels.

Only plotting labels are changed. Saved trajectories and historical figures
are preserved; this script does not run training or numerical evaluation.
After changes to equation numbering, rebuild main.tex, rerun this script,
and rebuild main.tex again to synchronize the external figure references.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.Ex1.analysis import plot_figure1


def main():
    labels = {"A1": "31a", "A2": "31b", "A3": "31c"}
    source = ROOT / "output/experiment1/formal_20260916_v2/runs"
    raws = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(source.rglob("raw.json"))
    ]
    if len(raws) != 44:
        raise ValueError(f"Expected 44 saved runs, found {len(raws)}")
    destination = ROOT / "output" / "paper_figures"
    destination.mkdir(parents=True, exist_ok=True)
    plot_figure1(raws, destination, equation_labels=labels)
    for suffix in (".pdf", ".png"):
        (destination / ("figure1_toy_convergence" + suffix)).replace(
            destination / ("Figure_4" + suffix)
        )


if __name__ == "__main__":
    main()
