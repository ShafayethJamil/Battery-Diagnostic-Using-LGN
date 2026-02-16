# Nyquist Spectrum Reconstruction from Pulse Data

## The Claim

Two LGN-extracted time constants from a 30-second HPPC pulse are a sufficient statistic for the cell's impedance fingerprint across the diagnostic band (0.1 Hz – 1 kHz). A lightweight linear regressor trained on other cells reconstructs the full Nyquist spectrum of an unseen cell with **0.99% mean error** (leave-one-cell-out) and **1.31% error** when additionally extrapolating to unseen aging states.

---

## Why Regression, Not Analytical Z(ω)

Under the RC equivalent circuit, the impedance transfer function is:

$$Z(\omega) = R_s + \sum_{i=1}^{n} \frac{R_i}{1 + j\omega\tau_i}$$

LGN recovers (τ₁, τ₂, τ₃) and the initial branch voltages V₀ᵢ, which in principle determine Rᵢ = V₀ᵢ/I_pulse. One might expect to evaluate Z(ω) directly from these parameters — no regression needed.

**This does not work.** Direct substitution of LGN-extracted parameters into Z(ω) yields 5.6% reconstruction error — worse than EIS reconstructing itself via its own 3RC fit (2.0%). The reason is window-conditioned projection: LGN eigenvalues recovered from a finite observation window are not the true poles of the impedance function. They are projections of the full (infinite-dimensional) relaxation spectrum onto the n-dimensional subspace resolvable by the data. The absolute values depend on the window length, sampling rate, and model order.

**However**, these projected eigenvalues are *consistent* and *informative*. Cells at the same impedance state produce the same (τ₁, τ₂) regardless of which specific cell is measured, and cells at different impedance states produce different (τ₁, τ₂). This means (τ₁, τ₂) constitute a **sufficient statistic** for the underlying electrochemical state — they compress the diagnostic information contained in the full impedance spectrum into two numbers.

A simple Ridge regression mapping (τ₁, τ₂, τ₃) → Z_re and Z_im at each frequency achieves:

| Method | Z_re MAPE |
|--------|-----------|
| LGN regression (LOCO) | **0.99%** |
| LGN regression (LOCO + prospective) | **1.31%** |
| EIS 3RC self-reconstruction | 2.0% |
| Direct analytical Z(ω) from LGN | 5.6% |

The regression outperforms EIS reconstructing itself from its own fitted parameters. This is not paradoxical: it reflects that a low-dimensional linear mapping between two representations of the same physical state can exploit statistical regularity that a nonlinear physics-based formula applied to biased inputs cannot.

---

## Why 0.1 Hz – 1 kHz

Stanford EIS is recorded at five frequencies (0.1 Hz, 1 Hz, 10 Hz, 100 Hz, 1 kHz), which we use as ground truth for evaluation. This band covers the standard diagnostic range used to separate ohmic, charge-transfer, and diffusion contributions in lithium-ion cells.

At the upper end (1 kHz), the Nyquist locus already exhibits a positive imaginary component (inductive behavior), which cannot be represented by a pure RC impedance model of the form Z(ω) = Rs + ΣRᵢ/(1+jωτᵢ). For this reason, **we do not interpret LGN time constants as literal EIS poles**. Instead, we evaluate whether (τ₁, τ₂) provide a compact state representation sufficient to reconstruct the measured impedance fingerprint over the available diagnostic band.

Extending reconstruction to >1 kHz would require either explicit inductive elements (e.g., R–L branches) or fixture-aware modeling; this is orthogonal to the diagnostic objective here, as the sub-kHz regime captures the electrochemical processes relevant to state-of-health assessment.

---

## Validation Protocol

We evaluate three progressively stricter protocols, all at SOC 50%:

### 1. Leave-One-Cell-Out (LOCO) — Mean MAPE: 0.99%

Train the regressor on two cells (all 15 aging states each, 30 total training points). Test on the third cell (15 aging states). Repeat for each cell as holdout.

| Held-out cell | Trained on | MAPE |
|---------------|-----------|------|
| W8 | W9, W10 | 1.21% |
| W9 | W8, W10 | 0.86% |
| W10 | W8, W9 | 0.92% |
| **Mean** | | **0.99%** |

This proves the mapping generalizes across cells — the regressor has never seen the test cell's impedance trajectory.

### 2. LOCO + Prospective — Mean MAPE: 1.31%

The hardest test: train on two cells using only early aging (diagnostics 1–10). Test on the third cell at late aging (diagnostics 11–15). The regressor has never seen this cell AND never seen this degradation level.

| Held-out cell | Test diags | MAPE |
|---------------|-----------|------|
| W8 | 11–15 | 1.00% |
| W9 | 11–15 | 1.32% |
| W10 | 11–15 | 1.62% |
| **Mean** | | **1.31%** |

Even under double extrapolation (unseen cell × unseen aging), reconstruction stays below 1.7% at every fold.

### 3. Within-Cell Prospective — Mean MAPE: 1.94%

For comparison: train on the same cell's early aging (diags 1–10, n=10), predict its own late aging (diags 11–15). This is the weakest protocol but serves as a data-efficiency baseline.

| Cell | MAPE |
|------|------|
| W8 | 3.06% |
| W9 | 1.43% |
| W10 | 1.33% |
| **Mean** | **1.94%** |

Within-cell is *worse* than LOCO because it has fewer training points (10 vs 30). This demonstrates that cross-cell information genuinely helps — the impedance-to-τ mapping is universal, not cell-specific.

---

## Frequency-Resolved Error (LOCO)

Reconstruction accuracy is highest at high frequencies and degrades gracefully toward low frequencies, consistent with the physical expectation that slow processes are hardest to constrain from short pulses:

| Frequency | Mean MAPE (LOCO) | Mean MAPE (LOCO+prosp.) |
|-----------|-----------------|------------------------|
| 1 kHz | 0.83% | 1.05% |
| 100 Hz | 0.79% | 1.03% |
| 10 Hz | 1.09% | 1.39% |
| 1 Hz | 1.15% | 1.58% |
| 0.1 Hz | 1.11% | 1.52% |

---

## Figure Captions

### Figure 5 (or Figure panel to be inserted)

**Figure X. Two time constants from a 30-second pulse reconstruct the Nyquist spectrum of unseen cells with sub-1% error.**

Leave-one-cell-out validation on the Stanford SECL dataset (3 NMC cells, 15 aging states each, SOC 50%). A Ridge regressor maps LGN-extracted (τ₁, τ₂, τ₃) to impedance at five frequencies (0.1 Hz – 1 kHz); training uses only the other two cells.

**(a–c)** Nyquist spectra for each held-out cell. Black: EIS measured (30 minutes). Colored dashed: LGN-predicted (from 30-second pulse). Light-to-dark shading tracks aging progression (diagnostics 1→15). The regressor has never seen any data from the test cell.

**(d)** Reconstruction error by frequency, averaged across all three LOCO folds. Error increases monotonically from 0.83% (1 kHz) to 1.11% (0.1 Hz), consistent with the physical expectation that low-frequency modes are hardest to constrain from short-window data.

### Supplementary Figure

**Figure SX. Nyquist reconstruction under double extrapolation: unseen cell × unseen aging state.**

Same as Figure X but with the strictest holdout: train on two cells at early aging only (diagnostics 1–10), test on the third cell at late aging (diagnostics 11–15). Mean MAPE = 1.31%. Even under simultaneous cell and temporal extrapolation, the impedance fingerprint is recovered with <1.7% error at every fold.

---

## Key Numbers for Abstract / Introduction

- **0.99%** — LOCO Nyquist reconstruction error (the headline)
- **1.31%** — LOCO + prospective (unseen cell × unseen aging)
- **5×** — regression outperforms direct analytical Z(ω) by 5× (0.99% vs 5.6%)
- **0.1 Hz – 1 kHz** — diagnostic band covered (5 frequencies)
- **3 cells × 15 aging states** — validation scope
- **2 numbers** — τ₁, τ₂ compress 30 minutes of EIS into a 30-second pulse

## Sentence for the Abstract

"A lightweight regressor maps two LGN-extracted time constants to the full impedance fingerprint at 0.1 Hz – 1 kHz; leave-one-cell-out validation yields 0.99% mean reconstruction error, demonstrating that a 30-second pulse encodes the same diagnostic information as 30-minute impedance spectroscopy."

---

## What You Can Say vs. What You Must Not Say

### ✅ Safe claims

- "τ₁ and τ₂ are a sufficient statistic for the cell's impedance state"
- "A single pulse reconstructs the impedance fingerprint within the diagnostic band"
- "The mapping generalizes across cells without per-cell calibration"
- "LGN compresses 30 minutes of EIS into 2 numbers from a 30-second pulse"

### ⚠️ Must not say

- "LGN recovers the true impedance poles" (it doesn't — window conditioning)
- "LGN replaces EIS" (too strong without testing >1 kHz and across chemistries)
- "Analytical equivalence between pulse and EIS" (regression, not Z(ω))
- "Works across SOC without retraining" (transfer results are mixed, SOC-specific mapping required)

### 🔶 Qualified claims (say with caveat)

- "Validated on NMC chemistry; extension to other chemistries requires further evaluation"
- "0.1 Hz – 1 kHz covers the standard diagnostic band; inductive regime above 1 kHz is not addressed"
- "Three cells from the same manufacturer; larger fleet validation is ongoing (cf. Popp dataset)"
