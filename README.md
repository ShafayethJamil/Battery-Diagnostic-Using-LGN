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

Raw cell data is from the following public sources — download directly from each:

- **KIT** (cell aging): [link / DOI from paper ref.]
- **TRI** (aging matrix): [link / DOI from paper ref.]
- **Samsung INR21700-50E** (Popp et al.): https://doi.org/10.5281/zenodo.10891871
- **Panasonic NCR18650PF** (Kollmeyer): https://doi.org/10.17632/wykht8y7tg.1
- **Stanford SECL** (Pozzato et al.): [link / DOI from paper ref.]

This repository provides the run scripts and the extracted time constants /
result files; it does not redistribute the raw datasets. Cite the original
sources above when using the data.

## Citing

If you use this code or data, please cite the paper (and the arXiv preprint
above).

## License

MIT — see `LICENSE`.
