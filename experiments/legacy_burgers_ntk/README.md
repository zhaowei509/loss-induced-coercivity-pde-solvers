# Figure 2: Burgers residual-NTK diagnostic

`rerun_burgers_ntk.py` is the reproducible source for Figure 2. It uses the settings preserved in the archived notebook, records the complete numerical trajectory, and exports both vector PDF and high-resolution PNG figures.

Run it from the repository root with:

```bash
python experiments/legacy_burgers_ntk/rerun_burgers_ntk.py \
  --output output/figure2_rerun_20261009_v1 \
  --outer-steps 1000 \
  --threads 8 \
  --publish-to output/paper_figures
```

The output directory contains:

- `trajectory.csv`: loss and relative NTK changes at every outer iteration.
- `training_samples.npz`: the sampled boundary and interior points.
- `metadata.json`: numerical settings, software versions, runtime, and data checksum.
- `Figure_2.pdf`: the vector figure used in the manuscript.
- `Figure_2.png`: a 1000 dpi preview.

`figure2_burgers_ntk.ipynb`, `ntk.png`, and `redraw_burgers_ntk.py` are retained as historical development materials. The current manuscript figure comes from `rerun_burgers_ntk.py` and the recorded output above.
