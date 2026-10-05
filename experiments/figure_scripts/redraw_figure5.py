"""Redraw manuscript Figure 5 from saved data with the current loss notation.

This only changes figure labels; no training or numerical evaluation is run.
The original formal-result figures and manuscript materials remain untouched.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.Ex1.analysis import plot_figure2


def main():
    source = ROOT / "output/experiment1/formal_20260916_v2/runs"
    raws = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(source.rglob("raw.json"))
    ]
    if len(raws) != 44:
        raise ValueError(f"Expected 44 saved runs, found {len(raws)}")
    destination = ROOT / "output" / "paper_figures"
    destination.mkdir(parents=True, exist_ok=True)
    plot_figure2(raws, destination, loss_symbol=r"\mathcal{J}")
    for suffix in (".pdf", ".png"):
        (destination / ("figure2_general_case" + suffix)).replace(
            destination / ("Figure_5" + suffix)
        )


if __name__ == "__main__":
    main()
