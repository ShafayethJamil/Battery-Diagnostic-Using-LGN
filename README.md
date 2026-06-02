# LGN Battery Diagnostics from Pulse Relaxation (HPPC)

Code and result files for the paper *"Lie Generator Networks Extract
EIS-Grade Battery Diagnostics from Pulse Relaxation Data"* (S. Jamil,
R. Kapadia). Preprint: [https://arxiv.org/abs/2605.15351].

Lie Generator Networks (LGN) extract electrochemical time constants from
~60 seconds of post-pulse voltage relaxation and reconstruct impedance-grade
diagnostics without dedicated EIS hardware. This repository contains the
reference implementation and the result files behind the paper's figures.

## Repository layout

Each dataset folder contains its run script and the result JSON files that the figures are built from:

- `KIT dataset/`        — degradation tracking and Nyquist reconstruction (`run_kit.py`)
- `TRI Dataset/`        — early-life prognosis (`run_tri_3d_warmstart.py`)
- `Samsung Dataset/`    — manufacturing quality control (`run_popp.py`)
- `Panasonic Dataset/`  — temperature/Arrhenius validation (`run_panasonic.py`)
- `Stanford_dataset/`   — window-scaling, model-order, and Nyquist analyses

The core model and fitting routine live in `run_degradation.py`

## Requirements

Python 3.10+, with:

```
torch  numpy  scipy  scikit-learn  pandas  matplotlib
```
Each script reads the cell data, fits LGN, and writes the result JSONs used
for the figures.

## Data

Raw cell data comes from the original public sources cited in the paper:
KIT, Toyota Research Institute, Samsung INR21700-50E, Panasonic NCR18650PF,
and Stanford SECL. Please cite those sources when using the data; this
repository's added value is the extracted time constants and derived results.

## Citing

If you use this code or data, please cite the paper (and the arXiv preprint
above).

## License

MIT — see `LICENSE`.
