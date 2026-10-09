# Code and Results

This directory contains the code, formal numerical outputs, and derived figures supporting *Loss-Induced Coercivity and Local Unfreezing Conditions for Nonlinear Random-Feature PDE Solvers*. 

## Contents

- `experiments/`: code for Experiments 1--4, shared utilities, plotting scripts, and the Burgers residual-NTK experiment used for Figure 2.
- `output/`: immutable formal run records, including raw JSON/NPZ trajectories, summaries, tables, and figures.
- `output/paper_figures/`: final numerical figures named as in the manuscript and supplement.
- `requirements.txt`: Python dependencies used by the formal campaigns.
- `file_manifest.csv`: size, SHA-256 checksum, category, and origin of every included file.
- `source_provenance.csv`: comparison of source hashes recorded by the formal runs with the code available in this archive.
- `verify_archive.py`: inventory, checksum, syntax, and data-readability checks.

## Paper mapping

| Paper material | Code | Formal output |
| --- | --- | --- |
| Figure 2, Burgers residual-NTK diagnostic | `experiments/legacy_burgers_ntk/rerun_burgers_ntk.py` | `output/figure2_rerun_20261009_v1/` and `output/paper_figures/Figure_2.*` |
| Section 5.1, Figures 4--5, Appendix B.1, Supplement S3 | `experiments/Ex1/` | `output/experiment1/formal_20260916_v2/` and `output/experiment1/inexact_proximal_formal_20260921_v1/` |
| Section 5.2, Figure 6, Appendix B.2, Supplement S4.1--S4.3 | `experiments/Ex2/` | `output/experiment2/formal_20260917_v1/`, `c_supplement_formal_20260917_v1/`, and `deep_ritz_radial_scan_formal_20260921_v1/` |
| Table 1 and Supplement S4.4--S4.5 | `experiments/Ex3/` | `output/experiment3/formal_20260917_v2/` |
| Section 5.3, Figure 7, Appendix B.3, Supplement S5 | `experiments/Ex4/` | `output/experiment4/formal_20260918_v1/`, `controlled_certificate_formal_20260921_v1/`, `general_formal_20260921_v1/`, and `manufactured_formal_20260921_v2/` |
| Supplement S6 and Figure S2 | `experiments/Ex2/supplement/` and `experiments/Ex4/supplement/` | `application_scale_formal_20260921_v1/`, `target_frequency_formal_20260921_v1/`, and `application_scale_unfreezing_formal_20260921_v1/` |
| Consolidated figures and tables | `experiments/paper_materials.py` | `output/paper_experiment_materials_20260921_v1/` |

## Quick verification

The formal campaigns used Python 3.10.15, NumPy 2.1.2, SciPy 1.15.2, Matplotlib 3.10.1, and PyTorch 2.4.1 with CUDA 12.1 where noted in the per-campaign manifests. From this directory, install the listed packages and run:

```bash
python -m pip install -r requirements.txt
python verify_archive.py
```

The checker reads every compressed array, parses every JSON/CSV/notebook file, compiles every Python file, and verifies the complete inventory against `file_manifest.csv`. A CUDA-enabled PyTorch installation is needed only to rerun the CUDA campaigns; follow the PyTorch installation instructions for the target platform.

Figure 2 was rerun on an Intel macOS CPU with Python 3.11.7, NumPy 1.26.4, SciPy 1.11.4, Matplotlib 3.8.0, PyTorch 2.2.2, and pyDOE 0.3.8. Its complete settings, software versions, elapsed time, and checksums are recorded in `output/figure2_rerun_20261009_v1/metadata.json`.

## Reproducibility scope

The saved outputs, paper figures, and included source files are independently checksum-verifiable. Numerical settings are stored in the code and run manifests. The Figure 2 directory includes the complete 1,000-step numerical trajectory and sampled training points. The formal runners also hash internal planning `.md` files, which are intentionally omitted; therefore, this public archive preserves the experiments and their evidence but is not a one-command reconstruction of every original campaign directory.
