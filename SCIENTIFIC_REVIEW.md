# Critical Scientific Review: LGN Battery Diagnostic Methodology

**Review Date:** 2026-02-22
**Repository:** LGN-Battery-Diagnostic
**Scope:** Mathematical formulation, implementation correctness, statistical methodology, validation design, and claims assessment across all 17 Python scripts, 5 datasets, and supporting documentation.

---

## Section 0: Executive Summary

### Overview of the LGN Approach

The Logarithmic Gaussian Network (LGN) approach estimates battery state-of-health (SOH) from brief HPPC (Hybrid Pulse Power Characterization) pulses, typically 10–30 seconds in duration. The method fits a linear state-space model A = S − D to the voltage relaxation transient, extracts eigenvalue-derived time constants τᵢ = −1/Re(λᵢ), and uses these as features for downstream tasks: Nyquist spectrum reconstruction via Ridge regression, SOH estimation via linear regression on log(τ), and degradation sensitivity analysis. The approach is validated on 5 datasets spanning Stanford (NMC 21700), KIT (NMC pouch), Panasonic (NCA 18650), Samsung (NMC 21700), and TRI (NMC 21700) cells.

### Top 5 Findings by Severity

1. **CRITICAL — Sample Size (n=3 cells):** All headline claims (0.64% SOH MAE, 0.99% Nyquist MAPE, 10× sensitivity) rest on 3 NMC cells from one manufacturer. The 95% CI on the 0.64% MAE is approximately ±0.6%, making it indistinguishable from ~1.2%. Cross-cell generalization claims require substantially larger validation sets.

2. **HIGH — Warm-Start Contamination in W10:** W10 uses warm-started optimization (`results_3d_W10_Warmstart.json`) while W8/W9 do not, creating an unfair comparison within LOCO. W10's per-cell MAE (0.95%) is notably the worst, but the warm-start bias could be operating in either direction.

3. **HIGH — Model Order Selection Circularity:** n=3 receives 9 initializations versus 4 for n=2, 6 for n=4, and 5 for n=5/n=6, biasing model selection toward n=3. The top-3 survival criterion guarantees n≥3 survives regardless of data evidence.

4. **HIGH — Window-Specific Initializations:** The 36s and 360s window sweeps use entirely different d_param initializations (τ₃ targets shifted by 2.0–2.5 in softplus space), biasing the "36s matches 3600s" comparison. The models are not asked the same question.

5. **HIGH — No Confidence Intervals on Any Headline Metric:** The 0.99% MAPE, 0.64% MAE, and ρ=0.911 are all point estimates without bootstrap CIs, standard errors, or hypothesis tests. Publication requires uncertainty quantification.

### Overall Assessment

**Publication-Readiness: 6.5/10 — Strong methodology with significant validation gaps.**

The core idea — extracting degradation-sensitive time constants from short HPPC pulses — is sound and novel. The implementation is competent, the impedance reconstruction result is genuinely impressive, and the capacitance-amplification physical argument is insightful. However, the work cannot support its generalization claims on n=3 cells, and several methodological choices (warm-start asymmetry, non-uniform initializations in model order selection, window-tuned d_params) introduce biases that must be addressed or disclosed before peer review.

### Key Strengths

- Novel capacitance-amplification insight (τ = RC captures multiplicative changes)
- Sub-1% Nyquist reconstruction from 10-second pulses is a striking result
- Clean state-space formulation enabling principled impedance transfer functions
- LOCO cross-validation is the correct protocol for battery SOH (stronger than random splits)
- Multiple-dataset coverage (5 chemistries/form factors) shows breadth of applicability

---

## Section 1: Mathematical Formulation Review

### 1.1 State-Space Model

**Files:** `Stanford_dataset/Result_data_files_with_codes/run_degradation.py:28–80`

The LGN model defines a continuous-time linear system:

```
A = S − D
η(t) = 1ᵀ exp(At) x₀
```

where S is skew-symmetric (encoding inter-state coupling) and D = diag(softplus(d_params)) enforces positive damping.

**Finding (HIGH): S is frozen at zero throughout training.**
At line 120, `model.s_params.requires_grad_(False)` disables gradient flow through S. Since S is initialized to zeros (line 34), it remains zero for all time. This means A = −D (purely diagonal), and the model reduces to a standard sum-of-exponentials: η(t) = Σᵢ x₀ᵢ·exp(−dᵢ·t). The "network" coupling physics claimed by "LGN" is structurally absent. The comment at line 120 acknowledges this: "freeze S: diagonal A, no identifiability issues."

*Impact:* The LGN label overstates the model's complexity. The method is a multi-restart nonlinear least-squares fit of a sum-of-exponentials, not a learned network. This does not invalidate the results, but the naming and framing should be more precise.

*Recommendation:* Acknowledge that S=0 is used in practice. Either demonstrate results with S≠0 or rename the approach to avoid implying network coupling.

**Finding (MEDIUM): C=B=1 in the impedance transfer function is not physically justified.**
At lines 161–177, the impedance shape function sets C = ones(1,n) and B = ones(n,1):

```python
C = np.ones((1, n))
B = np.ones((n, 1))
Z[k] = (C @ np.linalg.solve(1j * w * I_n - A, B)).item()
```

In a physical RC network, C and B would encode the topology of the equivalent circuit (which branches are driven by current, which contribute to voltage). Setting both to all-ones is an assumption that may introduce systematic bias, compensated downstream by the complex scale factor `a`. Since `a` absorbs both magnitude and phase rotation, this works empirically but obscures the physical interpretation.

*Recommendation:* Document that C=B=1 is a shape-only assumption compensated by the fitted scale factor. Consider whether learning C or B improves reconstruction.

### 1.2 Time Constant Extraction

**File:** `run_degradation.py:52–59`

```python
def get_time_constants(self, eps=1e-8):
    re = np.where(re < -eps, re, -eps)  # clamp near-zero modes
    taus = -1.0 / re
    return np.sort(taus)
```

**Finding (MEDIUM): Eigenvalue clamping to −eps creates artificial tau values for near-zero modes.**
When a mode has Re(λ) ≈ 0 (e.g., an unresolved slow process), the clamp at −1e-8 yields τ = 1e8 seconds (~3 years). This artificial value propagates into downstream features and correlations. The eps=1e-8 threshold is arbitrary — no sensitivity analysis is provided.

*Impact:* For the 3-state model, if the slowest mode is not well-constrained by the data, its reported τ₃ could be artificial. This may explain the high variance in τ₃ values seen across diagnostics.

*Recommendation:* Add a flag when clamping is triggered. Report which diagnostics had clamped eigenvalues. Consider excluding clamped modes from downstream analysis.

### 1.3 Impedance Scale Fitting

**File:** `run_degradation.py:180–202`

```python
def fit_scale_and_Rs(Z_shape, Z_data):
    """Fit Z_data ≈ a * Z_shape + Rs  (a complex, Rs real)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    ar, ai, Rs = beta
    a = ar + 1j * ai
    Rs = max(Rs, 0.0)
```

**Finding (MEDIUM): Complex scale factor `a` allows unphysical phase rotation.**
The complex `a` has 2 degrees of freedom (magnitude and phase), meaning the Z_shape can be rotated arbitrarily in the complex plane before matching Z_data. While this gives excellent fits, it means the physical interpretation of the impedance shape is lost — a rotated semicircle is no longer a physical RC response.

**Finding (LOW): No condition number check on the least-squares system.**
The system at line 196 uses `np.linalg.lstsq(X, y, rcond=None)`. For ill-conditioned X (e.g., when Z_shape has near-constant values across frequency), the fit is numerically unstable. No warning is issued.

*Recommendation:* Report |a| and arg(a) to assess the degree of correction. Flag fits where |arg(a)| > 15° as potentially unphysical. Add a condition number check with warning.

### 1.4 "Sufficient Statistic" Claim

**File:** `Stanford_dataset/Nyquist Reconstruction/nyquist_reconstruction_methods.md:5,19`

Line 5 states: *"Two LGN-extracted time constants from a 30-second HPPC pulse are a **sufficient statistic** for the cell's impedance fingerprint."*

Line 19 elaborates: *"This means (τ₁, τ₂) constitute a **sufficient statistic** for the underlying electrochemical state."*

**Finding (HIGH): The "sufficient statistic" claim is too strong.**
In statistics, a sufficient statistic T(X) satisfies the condition that P(X|T(X), θ) does not depend on θ — i.e., T captures ALL information in the data about the parameter. This requires a formal proof (typically via the factorization theorem). What the data show is that τ values are a *compact, low-dimensional representation* with high empirical accuracy (0.99% MAPE). This is an excellent result but does not constitute sufficiency in the statistical sense.

*Impact:* Reviewers familiar with mathematical statistics will flag this immediately. The claim is unnecessary — the empirical results are strong without it.

*Recommendation:* Replace "sufficient statistic" with "compact low-dimensional representation" or "informative summary statistic." The 0.99% reconstruction MAPE speaks for itself.

### 1.5 Capacitance Amplification Argument

**File:** `Stanford_dataset/EIS_LGN_Correlation_W8_W9_W10_V4/EIS_vs_LGN_Results.md:147–160`

Line 158: *"A 5% increase in R combined with a 50% increase in C produces a 58% increase in τ, but only a 5% increase in R as measured by EIS."*

**Finding (MEDIUM): The physical argument is sound but the specific numbers lack measurement support.**
The mathematical identity τ = RC correctly implies that τ captures multiplicative R×C changes. However, the assertion that C increases 50% while R increases only 5% is stated without direct measurement of C. EIS 2RC fitting (the eis_2rc_results.json data) recovers R and τ, not C directly. The C values would need to be derived as C = τ/R from the EIS fits, and their evolution over aging should be explicitly computed and reported.

*Recommendation:* Compute C = τ_EIS / R_EIS from the 2RC fits and plot C vs. aging to validate the 50% growth claim directly. If confirmed, this strengthens the argument considerably.

---

## Section 2: Implementation Quality Assessment

### 2.1 Training Loop

**File:** `run_degradation.py:91–155`

**Finding (MEDIUM): Gradient clipping at uniform 1.0 — not adaptive.**
Line 134: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`. The same clipping threshold applies to d_params (which span softplus values from ~0.05 to ~1000 in τ-space) and x0 (initial states). Parameters with different scales may require different clipping thresholds. Under-clipping d_params could slow convergence on slow modes; over-clipping x0 could cause instability.

**Finding (MEDIUM): No convergence criterion — all epochs always executed.**
Lines 128–136 run for exactly n_epochs iterations (3000 for Stanford, 2500 for KIT/TRI, 4000 for Panasonic) with no early stopping based on loss stagnation. ReduceLROnPlateau (line 125) reduces the learning rate but never terminates training. This wastes computation and could overfit on noisy diagnostics.

**Finding (LOW): Normalized MSE denominator uses mean(η²), not variance.**
Line 132: `loss = torch.mean((pred - eta_t)**2) / (torch.mean(eta_t**2) + 1e-12)`. Since η has non-zero mean (it's a decaying signal starting at η₀ ≈ −0.126V), mean(η²) ≠ var(η). This means the loss normalization includes the signal mean, making it scale-dependent in a non-standard way. This is not necessarily wrong but differs from the NRMSE reported in results (which normalizes by max|η|).

### 2.2 Warm-Start Mechanism

**File:** `run_stanford_3d_warmstart.py:94–98`

```python
if prev_d_params is not None:
    inits.insert(0, prev_d_params.clone().cpu())
```

And lines 107–111:
```python
if i_init == 0 and prev_x0 is not None and prev_d_params is not None:
    model.x0.data = prev_x0.clone().to(device)
```

**Finding (HIGH): Creates sequential path dependence between diagnostics.**
The warm-start mechanism prepends the previous diagnostic's optimized d_params as the first initialization for the next diagnostic, and also copies x0 for that initialization. This means the optimization trajectory of diagnostic k depends on the result of diagnostic k−1, creating a Markov chain of solutions. The extracted τ trajectory may partially reflect optimization path dynamics rather than physical degradation.

The warm-start chain is reset per cell (line 288: `prev_model = None`) but not per SOC. Within a cell's aging trajectory, all diagnostics are sequentially linked.

**No ablation exists comparing warm-start vs. cold-start.** Without this comparison, it is impossible to determine whether the smooth τ trajectories in the results reflect physical reality or warm-start smoothing.

*Recommendation:* Run a cold-start ablation on all Stanford W8/W9/W10 cells. Compare τ trajectories and SOH prediction accuracy. If warm-start and cold-start agree, this validates the approach. If they disagree, the warm-start bias must be quantified.

### 2.3 Multi-Restart Sufficiency

**Files:** `run_degradation.py:100–114`, `run_stanford_3d_warmstart.py:82–92`, `run_model_order.py:62–102`

| Dataset | n_states | Restart Count |
|---------|----------|---------------|
| Stanford (2D) | 2 | 4 |
| Stanford (3D) | 3 | 8 + warm-start |
| KIT | 3 | 4 + warm-start |
| Panasonic | 3 | 8 |
| TRI | 3 | 10 + warm-start |
| Samsung | 2 | (curve fit: 20; LGN inline: single) |

**Finding (MEDIUM): Restart counts are low by modern standards.**
Contemporary nonlinear optimization literature for multi-exponential fitting suggests 20–50+ restarts to reliably find the global minimum in noisy settings. The Stanford 2D model uses only 4 restarts. No check verifies that restarts converge to distinct solutions (some initializations may find the same local minimum).

*Recommendation:* Increase restarts to ≥20 for the headline Stanford results. Report the distribution of loss values across restarts to assess landscape complexity.

### 2.4 Subsampling Bias

**File:** `run_degradation.py:85–88`

```python
def _subsample_log(n_total, n_target):
    dense = np.arange(min(30, n_total))
    sparse = np.geomspace(1, n_total - 1, max(n_target - 30, 10)).astype(int)
    return np.unique(np.concatenate([dense, sparse]))
```

**Finding (MEDIUM): Includes the last point (t≈3600s) and overweights early times.**
The `geomspace(1, n_total-1, ...)` always includes the endpoint (index n_total-1), which corresponds to t≈3600s where η≈0. This biases the model toward fitting the asymptotic value precisely. Meanwhile, log-geometric spacing concentrates ~60% of points in the first 10% of the time window, overweighting fast dynamics relative to slow modes.

*Impact:* The model may underestimate the slowest τ because the long-time behavior is dominated by a single near-zero point, while the short-time behavior is oversampled.

*Recommendation:* Test sensitivity to subsampling strategy. Compare with uniform temporal sampling and with the last point excluded.

### 2.5 Hard-coded Values

**File:** `run_degradation.py:360`

```python
res['R_pulse'] = float(abs(eta_full[0]) / 4.85)  # η₀/I_pulse (assuming 4.85A)
```

**Finding (MEDIUM): I_pulse=4.85A is hard-coded.**
This value is correct for the Stanford INR21700-M50T cells but would be silently wrong for any other dataset. The KIT code (`run_kit.py:31`) correctly uses `I_PULSE_DEFAULT = 1.0` and computes per-measurement pulse current from the data (line 117). The TRI code uses a nominal 4.84 Ah capacity. The inconsistency suggests the codebase grew organically without a unified parameter management approach.

**Finding (LOW): Bare `except: continue` in EIS fitting.**

```python
# run_degradation.py:328
except:
    continue
```

This catches ALL exceptions including KeyboardInterrupt, MemoryError, and SystemExit. Convergence failures, numerical overflows, and genuine bugs are silently swallowed. The EIS analysis script (`eis_vs_lgn_analysis.py:110`) correctly uses `except Exception:` instead.

*Recommendation:* Replace `except:` with `except (ValueError, RuntimeError, np.linalg.LinAlgError):` throughout.

### 2.6 Window-Specific Initializations

**File:** `Stanford_dataset/Stanford window/run_window_sweep_v2.py:43–61`

```python
WINDOW_INITS = {
    36: [
        torch.tensor([2.0, -1.0, -3.0]),    # tau ~ [0.5, 3, 20]
        torch.tensor([1.5, -0.5, -2.5]),    # tau ~ [0.6, 2, 12]
        ...
    ],
    360: [
        torch.tensor([2.0, -2.0, -5.0]),    # tau ~ [0.5, 8, 150]
        torch.tensor([1.5, -1.5, -4.5]),    # tau ~ [0.6, 5, 90]
        ...
    ],
}
```

**Finding (HIGH): 36s and 360s windows use different d_param initializations explicitly tuned per window.**
The τ₂ initializations differ by ~1.0 in softplus space (3s vs 8s), and τ₃ differs by ~2.0–2.5 (20s vs 150s). This is defensible on physical grounds (longer windows can resolve slower modes), but it biases the comparison: the 36s model is given initialization hints appropriate for its observable timescales, and so is the 360s model. A fair comparison would use identical initializations for both windows and let the optimizer find what the data supports.

*Impact:* The claim that "36s matches 3600s accuracy" is weakened because the 36s model was given a priori guidance about which timescales to resolve.

*Recommendation:* Re-run window sweep with identical (superset) initializations for all window lengths. Report any degradation in 36s performance.

### 2.7 Inconsistent Model Orders Across Datasets

**Files:** All dataset scripts

| Dataset | n_states | Epochs | LR | Restarts | Warm-Start |
|---------|----------|--------|----|----------|------------|
| Stanford (main) | 2 | 3000 | 0.01 | 4 | No |
| Stanford (3D) | 3 | 4000 | 0.01 | 8+ws | Yes |
| KIT | 3 | 2500 | 0.01 | 4+ws | Yes |
| Panasonic | 3 | 4000 | 0.01 | 8 | No |
| Samsung | 2 | 300 | (cosine) | 1 | No |
| TRI | 3 | 2500 | 0.01 | 10+ws | Yes |

**Finding (HIGH): Hyperparameters differ per dataset with no justification or control experiment.**
Samsung uses n=2 with 300 epochs and cosine annealing; Panasonic uses n=3 with 4000 epochs and plateau scheduling; KIT uses 2500 epochs. No ablation study examines the sensitivity to these choices. The Samsung model is architecturally different (inline `LGN_SD_2D` class at `run_popp.py:241`) from the shared `LGN_Battery` class used elsewhere.

*Impact:* Cross-dataset claims are weakened because differences in results could stem from hyperparameter tuning rather than fundamental dataset characteristics.

*Recommendation:* Define a single default configuration and document per-dataset deviations with justification.

---

## Section 3: Statistical Methodology Review

### 3.1 Multiple Comparisons

**Files:** `run_degradation.py:510–540`, `eis_vs_lgn_analysis.py:241–259`

**Finding (MEDIUM): 100+ correlations computed without Bonferroni/FDR correction.**
The correlation analyses compute Spearman and Pearson correlations for every combination of LGN features (τ₁, τ₂, τ₃, d₁, d₂, d₃, R_pulse, η₀) against EIS markers (Z at 5+ frequencies) across multiple windows and cells. No multiple comparison correction is applied.

*Mitigating factor:* The reported p-values are extremely small (10⁻¹² to 10⁻²¹), so even Bonferroni correction with 100 tests would not change significance. The ρ > 0.7 threshold at line 521 provides informal correction. This is a documentation issue rather than an invalidating flaw.

*Recommendation:* Report the number of tests performed and note that corrections do not affect conclusions given the extreme p-values.

### 3.2 Sample Size

**Files:** All scripts

**Finding (CRITICAL): All headline claims rest on n=3 cells (W8, W9, W10).**

The three NMC INR21700-M50T cells from Stanford provide 45 total diagnostic observations (15 per cell). Key statistical consequences:

- **Spearman ρ SE ≈ 0.15–0.20 at n=15 per cell.** The reported ρ = −0.909 has an approximate 95% CI of [−0.97, −0.76].
- **LOCO R² highly unstable.** With 15 test points per fold, the R² = 0.900 for the τ₁ model has large sampling variance.
- **MAE uncertainty.** The 0.64% MAE has an approximate 95% CI of ±0.4–0.6% (based on the per-cell MAE spread: 0.41%, 0.55%, 0.95%). This means the true MAE is plausibly as high as ~1.2%.
- **Cross-cell generalization on 3 cells is fundamentally limited.** Three cells from one manufacturer, chemistry (NMC), and form factor (21700) cannot support claims about general battery diagnostics.

*Context:* Three-cell validation is not unprecedented — the NASA B0005/B0006/B0007 dataset is used in hundreds of published papers. However, top-tier publications increasingly require 10–300+ cells (Severson et al. 2019 used 124 cells; Attia et al. 2020 used 48).

*Recommendation:* All headline metrics must be accompanied by confidence intervals. The abstract/introduction should explicitly state n=3 as a limitation and frame generalization claims as preliminary.

### 3.3 Missing Confidence Intervals

**Files:** `soh_analysis.py:146–152`, `run_nyquist_complete.py:175–179`, `SOH_Analysis_Results.md`, `EIS_vs_LGN_Results.md`

**Finding (HIGH): No headline metric includes confidence intervals or bootstrap estimates.**

| Metric | Reported Value | Missing |
|--------|---------------|---------|
| Nyquist MAPE (LOCO) | 0.99% | CI, bootstrap SE |
| SOH MAE (LOCO) | 0.64% | CI, bootstrap SE |
| Spearman ρ (τ₁ vs SOH) | −0.909 | CI |
| Sensitivity amplification | 10× | Error bars |
| R² (SOH estimation) | 0.900 | CI |

The `soh_analysis.py` code (lines 146–152) computes only point estimates:
```python
mae = mean_absolute_error(y, pred)
r2 = r2_score(y, pred)
```

No bootstrap resampling, jackknife, or analytical standard error computation exists in any script.

*Recommendation:* Add bootstrap confidence intervals (1000+ resamples) for all headline metrics. At minimum, report the per-fold spread (which `nyquist_complete_summary.json` partially does: mean 1.27% ± 0.91% std for the 45-fold).

### 3.4 Non-Independence of Observations

**Finding (MEDIUM): Sequential diagnostics on the same cell are correlated.**
The 15 diagnostics per cell trace a monotonic degradation trajectory. Adjacent diagnostics (e.g., diag 5 and diag 6) are more similar than distant ones (diag 1 and diag 15). This violates the independence assumption underlying both Pearson/Spearman significance tests and the i.i.d. assumption in cross-validation error estimation.

LOCO correctly addresses the most critical independence violation (cell identity), but within each training fold, the ~30 observations from 2 cells contain strong temporal autocorrelation. Standard errors and p-values computed under independence are anti-conservative (too optimistic).

The warm-start mechanism (Section 2.2) creates additional dependency by linking optimization solutions across diagnostics.

*Recommendation:* Acknowledge temporal autocorrelation. Consider blocked bootstrap or autocorrelation-adjusted standard errors for within-cell metrics.

### 3.5 One-Point Folds

**File:** `Stanford_dataset/Nyquist Reconstruction/run_nyquist_complete.py:154–180`

```python
# 3 cells × 15 diagnostics = 45 double-blind folds
for ci, tc in enumerate(CELLS):
    for di, hd in enumerate(all_diags):
        train = [d for c in others for d in CD[c] if d['diag'] != hd]
        test = [d for d in CD[tc] if d['diag'] == hd]
```

**Finding (HIGH): The 45-fold Cell+Aging holdout uses exactly 1 test sample per fold.**
Each fold holds out one cell AND one diagnostic number, resulting in a single test spectrum. The MAPE from reconstructing one Nyquist spectrum is not statistically meaningful per-fold — it is a single-point estimate with no variance. The aggregate over 45 folds provides a valid mean estimate, but the fold-level results (reported in `nyquist_complete_summary.json` as ranging from 0.187% to 4.533%) have no individual statistical power.

*Recommendation:* Clarify that per-fold MAPEs are single-point evaluations. The aggregate 45-fold mean (1.27% ± 0.91%) is the meaningful quantity. Consider a simpler LOCO (3-fold) as the primary metric, with the 45-fold as supplementary.

---

## Section 4: Validation Design Assessment

### 4.1 LOCO with n=3 Cells

**Finding (HIGH): Statistically underpowered for generalization claims.**

Leave-one-cell-out with 3 cells is a 3-fold cross-validation. Each fold trains on 30 observations and tests on 15. While this is the correct protocol for battery SOH (preserving cell identity), the small fold count means:

- **High variance in per-fold estimates:** W8 MAE = 0.41%, W9 = 0.55%, W10 = 0.95% — nearly 2.3× range.
- **No statistical test for LGN vs. R_pulse:** The 0.64% vs. 2.33% MAE difference is evaluated visually, not via a paired hypothesis test (e.g., paired t-test or Wilcoxon signed-rank on per-sample absolute errors).

*Context:* Three-cell LOCO is used in hundreds of battery SOH papers. The NASA dataset (3–4 cells) has supported thousands of publications. However, this work makes stronger claims (sub-1% accuracy, 10× sensitivity) that require stronger evidence.

*Recommendation:* Add a paired statistical test for LGN vs. R_pulse at the per-observation level. Acknowledge n=3 limitation prominently.

### 4.2 Prospective Validation

**File:** `run_nyquist_complete.py:182–187`

```python
train = [d for c in others for d in CD[c] if d['diag'] <= DIAG_SPLIT]
test = [d for d in CD[tc] if d['diag'] > DIAG_SPLIT]
```

where `DIAG_SPLIT = 10`.

**Finding (MEDIUM): The LOCO+Prospective protocol conflates cell identity with temporal information.**
The protocol trains on diagnostics 1–10 of two cells and tests on diagnostics 11–15 of the third cell. This is a strong test (both cell and temporal generalization), but the split point at diagnostic 10 is arbitrary. No sensitivity analysis examines how the split point affects results (e.g., DIAG_SPLIT = 5, 8, 12).

*Recommendation:* Report results for 3–5 split points to demonstrate robustness.

### 4.3 Warm-Start Contamination in W10

**Files:** `Result_data_files_with_codes/W8_W9_W10_cells/SOC50/W10/results_3d_W10_Warmstart.json`, `SOH_Analysis_Results.md:58,60`

**Finding (HIGH): W10 uses warm-started results while W8/W9 do not.**

The SOH analysis results show:

| Cell | MAE | R² | Warm-Start? |
|------|-----|-----|-------------|
| W8 | 0.41% | 0.961 | No |
| W9 | 0.55% | 0.930 | No |
| W10 | 0.95% | 0.805 | **Yes** |

The documentation at `SOH_Analysis_Results.md:60` acknowledges: *"W10 has slightly higher error because it uses warm-start initialization (sequential diagnostics share initialization), creating a minor systematic offset."*

This is an unfair comparison within LOCO. When W10 is the test cell, its τ values were computed differently from the training data (W8, W9), potentially introducing a systematic offset that inflates test error. Conversely, when W10 is in the training set (testing W8 or W9), its warm-started τ values may provide a smoother training signal that benefits the model.

*Recommendation:* Re-run W10 without warm-start to create a consistent comparison. Alternatively, apply warm-start uniformly to all three cells and re-evaluate.

### 4.4 Model Order Selection Circularity

**File:** `Stanford_dataset/Stanford Model Order/run_model_order.py:62–102, 204–212`

**Finding (HIGH): Initialization counts are non-uniform across model orders.**

| n_states | Initializations | Composition |
|----------|----------------|-------------|
| 2 | 4 | 1 linspace + 3 custom |
| 3 | **9** | 1 linspace + **8 custom** |
| 4 | 6 | 1 linspace + 5 custom |
| 5 | 5 | 1 linspace + 4 custom |
| 6 | 5 | 1 linspace + 4 custom |

n=3 receives 9 initializations — 2.25× more than n=2 and 1.5× more than n=4. Since the model selection is based on fitting quality (NRMSE), giving n=3 more optimization attempts biases the comparison in its favor.

Additionally, the survival criterion at lines 204–212 guarantees that up to 3 modes survive:
```python
topk = sorted(range(len(modes)), key=lambda i: -energy_fracs[i])[:3]
m['survives'] = (m['status'] == 'IDENTIFIABLE' and
                (m['energy_frac'] > E_MIN or i in topk))
```

The `i in topk` condition ensures the top-3 energy modes survive regardless of the 3% energy threshold. For n≥3, this always selects exactly 3 modes. The model order "selection" is thus circular: n=3 gets the most restarts AND the survival criterion guarantees n=3.

From the JSON results, NRMSE does improve significantly from n=2 (mean 0.016) to n=3 (mean 0.005), but further improvement to n=4 (mean 0.003) and n=5/n=6 (mean 0.003) is marginal. This supports n≥3 but does not distinguish n=3 from n=4.

*Recommendation:* Re-run model order selection with equal initializations (e.g., 10 per model order). Remove the top-k override from the survival criterion. Use AIC/BIC for model selection instead of raw NRMSE.

### 4.5 Missing Nested Cross-Validation

**Files:** `run_nyquist_complete.py:51`, `run_nyquist_window_comparison.py:34`, `soh_analysis.py:130`

**Finding (MEDIUM): Ridge alpha=1e-4 and log-transform are fixed, not selected via inner CV loop.**

The Nyquist reconstruction uses `ALPHA = 1e-4` (effectively OLS) without cross-validating this choice. The SOH estimation uses `LinearRegression()` (equivalent to Ridge with α→0) with a log-transform on τ features (`np.log(X[train])`), also without validating the transform choice.

Since α=1e-4 is extremely small (essentially no regularization), the main risk is overfitting on the 30-sample training sets. With only 3 features (τ₁, τ₂, τ₃), this is unlikely to be severe, but it should be verified.

*Recommendation:* Add a brief sensitivity analysis: sweep α from 1e-6 to 1e0 and report the effect on LOCO MAPE. If results are stable, document this. If they vary, use inner CV.

---

## Section 5: Claims vs Evidence Audit

| # | Claim | Source | Verdict | Detail |
|---|-------|--------|---------|--------|
| 1 | 0.99% LOCO Nyquist reconstruction | nyquist_reconstruction_methods.md:25 | **SUPPORTED on 3 cells; needs CI** | Mean MAPE = 0.994% across 3 cells (W8: 1.21%, W9: 0.86%, W10: 0.92%). Strong result but no confidence interval. 45-fold variant gives 1.27% ± 0.91%. |
| 2 | 0.64% SOH MAE cross-cell | SOH_Analysis_Results.md:15 | **SUPPORTED but CI overlaps ~1.2%** | Per-cell: W8=0.41%, W9=0.55%, W10=0.95%. Range suggests true MAE could be higher with different cell samples. |
| 3 | 10× sensitivity amplification | EIS_vs_LGN_Results.md:145 | **SUPPORTED for W8/W9/W10** | W8: 10×, W9: 10×, W10: 12× amplification. Consistent across all 3 cells. Needs error bars on the per-cell amplification factors. |
| 4 | 100× beyond EIS bandwidth | EIS_vs_LGN_Results.md:180 | **SUPPORTED (definitional)** | τ₃ ≈ 670–1580s = 42–100× beyond EIS lowest frequency (0.01 Hz ≈ τ=16s). This follows from the timescale math and is not an empirical claim. |
| 5 | 3.7× more accurate than R_pulse | SOH_Analysis_Results.md:19 | **SUPPORTED (same CV protocol)** | 0.64% vs 2.33% MAE under identical LOCO. No paired hypothesis test provided. |
| 6 | τ are sufficient statistics | nyquist_reconstruction_methods.md:5,19 | **OVERSTATED** | Empirically supported (0.99% MAPE) but "sufficient statistic" has a precise statistical meaning not met here. Should use "informative summary" or "compact representation." |
| 7 | 36s matches 3600s accuracy | Stanford window analysis | **WEAKENED** | Biased by window-specific initializations (Section 2.6). The 36s model received d_param hints tuned for short-window timescales. |
| 8 | Generalizes across cells | Multiple files | **WEAKLY SUPPORTED** | Only 3 cells, same manufacturer/chemistry/form factor. LOCO is the right protocol but n=3 is insufficient for a strong generalization claim. |
| 9 | n=3 is optimal model order | Stanford Model Order analysis | **WEAKENED** | Biased initialization favoring n=3 (9 restarts vs 4–6 for others). NRMSE data actually suggests n≥3 with diminishing returns beyond n=4. See Section 4.4. |

---

## Section 6: Comparison to State-of-the-Art

### Quantitative Positioning

| Dimension | LGN (This Repo) | Best Published (Pulse-Based) | Best Published (Any Method) |
|-----------|-----------------|-----------------------------|-----------------------------|
| SOH MAE | 0.64% (cross-cell, n=3) | ~1–2% typical (EKF/UKF on pulse features) | 0.31–0.49% (deep learning on full cycling data, n=124 cells, Severson 2019) |
| Measurement time | 10 seconds | Minutes (relaxation voltage methods) to hours (slow-rate partial charge) | Full charge/discharge cycles (hours) |
| Equipment | Standard BMS hardware (current pulse + voltage measurement) | BMS / potentiostat | Potentiostat + thermal chamber |
| Cross-cell validation | LOCO (n=3 NMC cells) | Varies; many papers use same-cell train/test splits | 65–387 cells in top publications |
| Leading indicator | 10× amplification (novel) | Not commonly reported | Not commonly reported |

### Supporting Scripts

Three additional scripts provide supplementary functionality:

- **`run_nyquist_reconstruction.py`**: Implements the core Ridge regression reconstruction (τ → Z), including analytical Z(ω) evaluation via NNLS and the three validation protocols (LOCO, LOCO+Prospective, Within-Cell). Uses `RIDGE_ALPHA = 1e-4` (line 70), confirming the effectively-OLS regularization noted in Section 4.5.
- **`fig_headline_soh.py`**: Generates publication figures for SOH analysis (6-panel layout: τ₁ vs capacity trajectories, amplification factors, cross-cell parity plots, mode separation, τ₁ vs R_pulse, and per-cell MAE comparison). Point estimates only — no confidence bands plotted.
- **`extract_amplitudes.py`** (KIT dataset): Recovers per-mode amplitudes {aᵢ} from fixed τ values via linear least-squares (η(t) = Σ aᵢ·exp(−t/τᵢ)), then computes Rᵢ = |aᵢ|/I_pulse. Pure linear algebra — no hyperparameters or random initialization. Also extracts Rs from the voltage jump at the discharge-to-relaxation transition.

### DRT Methods (Most Direct Competitor)

Distribution of Relaxation Times (DRT) is the most natural comparison for pulse-based τ extraction. DRT deconvolves the impedance function into a continuous distribution γ(τ), while LGN fits a discrete sum of exponentials. Key differences:

- **DRT requires regularization** (Tikhonov or similar) to handle the ill-posed inverse problem. LGN uses direct nonlinear optimization.
- **DRT typically requires EIS data** (frequency-domain input), while LGN operates on time-domain pulse data.
- **DRT produces a full distribution** γ(τ), while LGN produces n discrete {τᵢ, aᵢ} pairs.

Recent work (Wan et al. 2015, Hahn et al. 2019, Paul et al. 2021) has applied DRT to time-domain data (galvanostatic relaxation), but these methods typically require longer observation windows (minutes to hours). LGN's ability to extract meaningful time constants from 10-second pulses, if validated on larger datasets, would represent a significant advance over time-domain DRT.

**No direct comparison with DRT methods exists in this repository.** This is a significant omission for positioning.

### EKF/UKF Observers

Standard BMS approaches use Extended Kalman Filters or Unscented Kalman Filters to jointly estimate SOC and SOH from operational data. These methods:

- Require a physics-based model (equivalent circuit or electrochemical)
- Operate continuously on operational data (not diagnostic pulses)
- Typically achieve 1–3% SOH accuracy

LGN addresses a different use case: periodic diagnostic checks from brief pulses. The methods are complementary, not competing.

### ML Baselines

Random Forest, XGBoost, and CNN-LSTM approaches on pulse features typically achieve 1–3% MAE on large datasets (50–300+ cells). LGN's 0.64% on 3 cells is not directly comparable due to the vastly different validation scales. On comparable 3-cell datasets, ML baselines have not been systematically benchmarked.

### Proper Positioning

LGN is competitive for **rapid pulse-based diagnostics** — extracting maximum information from minimal measurement time. It is not competing with full-cycle deep learning methods (which have more data and more cells) or continuous EKF observers (which have different use cases). The strongest positioning is: *"From a 10-second HPPC pulse, LGN extracts time constants that correlate with SOH at ρ=0.91 and reconstruct the Nyquist spectrum to 1% accuracy — enabling rapid, equipment-light diagnostics that complement existing BMS approaches."*

---

## Section 7: Cross-Dataset Generalization

### 7.1 No Cross-Dataset Prediction Experiment

**Finding (HIGH): 5 datasets use different model configurations, and no cross-dataset prediction exists.**
Each dataset is analyzed independently with dataset-specific hyperparameters (Section 2.7). At no point is a model trained on one dataset used to predict another. For example, no experiment trains on Stanford and tests on KIT, or vice versa. This means the "5-dataset validation" demonstrates only that the LGN fitting procedure works on each dataset individually — it does not validate transferability.

*Recommendation:* Design a cross-dataset experiment: train the Ridge regression (τ → SOH) on Stanford W8/W9/W10 and test on KIT cells, using a shared n=3 model with fixed hyperparameters.

### 7.2 TRI Has No EIS Ground Truth

**Finding (HIGH): The TRI dataset (EXP 374, ~400 cells) only validates τ vs. capacity correlation.**
The TRI analysis (`run_tri_3d_warmstart.py`) computes Spearman correlations between τ values and discharge capacity but has no EIS data for impedance reconstruction validation. The Nyquist reconstruction and EIS correlation claims are therefore supported only by the 3 Stanford cells.

### 7.3 KIT EIS Fits Hit Parameter Bounds

**Finding (MEDIUM): 42.7% of KIT entries have `eis_3rc_hit_bounds = true`.**
From `results_kit_gpu0.json`, 610 out of 1429 entries show the EIS 3RC fitting hitting parameter boundaries. Example: cell P001_1_S01_C10 at 0°C has `τ_eis_3rc = [0.0000111s, 0.01374s, 500.0s]` — the 500s boundary is clearly active. This means the EIS ground truth for nearly half of KIT data points is unreliable, potentially biasing the LGN-vs-EIS correlation analysis for this dataset.

*Recommendation:* Exclude entries with `eis_3rc_hit_bounds = true` from EIS correlation analyses, or flag them separately.

### 7.4 Samsung Uses OCV Proxy

**Finding (MEDIUM): Samsung SOH is evaluated against V_end (open-circuit voltage), not direct capacity measurement.**
The Samsung analysis (`run_popp.py`) uses `V_end` (voltage at end of relaxation) as an SOH proxy. While OCV correlates with SOC and indirectly with SOH, it is not a direct SOH measurement. The claimed LGN-τ correlations for Samsung are with OCV, not with independently measured capacity.

### 7.5 Panasonic: No Cross-Temperature Prediction

**Finding (MEDIUM): Panasonic tests one chemistry (NCA 18650PF) at 5 temperatures (−20°C to 25°C) but performs no cross-temperature prediction.**
Each temperature is analyzed independently (`run_panasonic.py` processes each temperature folder separately). No experiment trains on 25°C data and predicts at 0°C or −10°C. This misses an opportunity to validate temperature robustness.

---

## Section 8: Reproducibility Assessment

### 8.1 No Random Seeds

**Finding (HIGH): No random seeds are set in any of the 17 Python scripts.**
None of the scripts call `torch.manual_seed()`, `np.random.seed()`, or `random.seed()`. The LGN optimization uses random initialization via `torch.randn()` (in `run_model_order.py:100`: `jitter = torch.randn(n) * 0.5`), and PyTorch's Adam optimizer has stochastic behavior. Results are therefore non-deterministic across runs.

*Impact:* The reported metrics (0.99% MAPE, 0.64% MAE) could vary by ±0.1–0.5% between runs depending on initialization luck. Without seeds, exact reproduction is impossible.

*Recommendation:* Add `torch.manual_seed(42); np.random.seed(42)` at the top of each script. Report the seed used.

### 8.2 No Requirements File

**Finding (MEDIUM): No `requirements.txt`, `environment.yml`, or `pyproject.toml` exists.**
The code imports PyTorch, NumPy, SciPy, pandas, matplotlib, and scikit-learn, but specific versions are not documented. PyTorch's `matrix_exp` behavior changed between versions. SciPy's `minimize` default tolerances differ across versions.

*Recommendation:* Add `requirements.txt` with pinned versions. At minimum: `torch>=1.9`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`.

### 8.3 Hardcoded Paths

**Finding (MEDIUM): Scripts reference local directories and upload folders.**
Multiple scripts contain paths like `/home/...` or `../upload/` that are machine-specific. While the CSV data appears to be present in the repository, the path references in scripts may not match the repository structure.

*Recommendation:* Use relative paths from the repository root. Add a `DATA_ROOT` configuration variable.

### 8.4 Missing Data for 4 of 7 Stanford Cells

**Finding (LOW): Cells W4, W5, W7, and G1 have LGN results but no EIS correlation analysis.**
The `Result_data_files_with_codes/` directory contains results for W4, W5, V4, and G1, but only W8, W9, and W10 are used for the headline claims (EIS correlation, SOH estimation, Nyquist reconstruction). The selection of these 3 cells is not explained.

Looking at the available data: W4, W5, and G1 directories contain `results_3d_*.json` and `correlations_3d_*.json` files, and V4 is included in the EIS analysis (`eis_vs_lgn_analysis.py:122` includes V4). The reason W4, W5, W7, and G1 are excluded from headline analyses may be missing EIS data, but this is not documented.

*Recommendation:* Document why these cells were excluded. If EIS data is missing, state this explicitly.

---

## Section 9: Consolidated Findings Table

| ID | Section | Severity | Finding | File:Line | Recommendation |
|----|---------|----------|---------|-----------|----------------|
| F01 | 3.2 | CRITICAL | All headline claims based on n=3 cells, one manufacturer, one chemistry | All scripts | Add CI; acknowledge limitation; validate on 10+ cells |
| F02 | 1.1 | HIGH | S frozen at zero — model reduces to sum-of-exponentials, not a "network" | run_degradation.py:120 | Acknowledge S=0; rename or demonstrate S≠0 |
| F03 | 1.4 | HIGH | "Sufficient statistic" claim is too strong for the evidence | nyquist_reconstruction_methods.md:5,19 | Replace with "compact representation" |
| F04 | 2.2 | HIGH | Warm-start creates sequential path dependence; no ablation exists | run_stanford_3d_warmstart.py:94–98 | Run cold-start ablation |
| F05 | 2.6 | HIGH | Window-specific d_param initializations bias window comparison | run_window_sweep_v2.py:43–61 | Re-run with identical initializations |
| F06 | 2.7 | HIGH | Model order/hyperparameters differ per dataset without justification | All dataset scripts | Define default config; justify deviations |
| F07 | 3.3 | HIGH | No confidence intervals on any headline metric | soh_analysis.py:150–152 | Add bootstrap CIs |
| F08 | 3.5 | HIGH | 45-fold Cell+Aging holdout uses 1 test sample per fold | run_nyquist_complete.py:161–163 | Clarify statistical interpretation |
| F09 | 4.1 | HIGH | LOCO with n=3 is underpowered for generalization claims | soh_analysis.py:125–127 | Add paired hypothesis tests; expand dataset |
| F10 | 4.3 | HIGH | W10 uses warm-start; W8/W9 do not — asymmetric LOCO | results_3d_W10_Warmstart.json | Re-run W10 cold-start OR warm-start all cells |
| F11 | 4.4 | HIGH | Model order selection gives n=3 most restarts (9 vs 4–6) | run_model_order.py:67–79 | Equal restarts per n; use AIC/BIC |
| F12 | 7.1 | HIGH | No cross-dataset prediction experiment | All dataset scripts | Train Stanford → test KIT (or vice versa) |
| F13 | 7.2 | HIGH | TRI has no EIS ground truth | run_tri_3d_warmstart.py | Acknowledge limitation |
| F14 | 8.1 | HIGH | No random seeds — results non-deterministic | All scripts | Set seeds; report them |
| F15 | 1.2 | MEDIUM | Eigenvalue clamping at arbitrary eps=1e-8 creates artificial τ | run_degradation.py:57 | Flag clamped modes; sensitivity analysis |
| F16 | 1.3 | MEDIUM | Complex scale factor allows unphysical phase rotation | run_degradation.py:196–198 | Report |a|, arg(a); flag large rotations |
| F17 | 1.5 | MEDIUM | C increases 50% claim lacks direct measurement | EIS_vs_LGN_Results.md:158 | Compute C=τ/R from EIS data |
| F18 | 2.1 | MEDIUM | Uniform gradient clipping at 1.0 across all parameters | run_degradation.py:134 | Consider per-parameter adaptive clipping |
| F19 | 2.1 | MEDIUM | No convergence criterion; all epochs always run | run_degradation.py:128–136 | Add early stopping |
| F20 | 2.3 | MEDIUM | Only 4–10 restarts; literature suggests 20–50+ | run_degradation.py:100–106 | Increase to ≥20 restarts |
| F21 | 2.4 | MEDIUM | Subsampling includes last point; overweights early times | run_degradation.py:85–88 | Test sensitivity to subsampling |
| F22 | 2.5 | MEDIUM | I_pulse=4.85A hard-coded for Stanford only | run_degradation.py:360 | Parameterize or read from metadata |
| F23 | 3.1 | MEDIUM | 100+ correlations without multiple comparison correction | run_degradation.py:510–540 | Report test count; note p-values survive correction |
| F24 | 3.4 | MEDIUM | Within-cell observations temporally correlated | All scripts | Acknowledge; use autocorrelation-adjusted SEs |
| F25 | 4.2 | MEDIUM | Prospective split at diag 10 is arbitrary; no sensitivity analysis | run_nyquist_complete.py:52 | Sweep split points |
| F26 | 4.5 | MEDIUM | Ridge alpha=1e-4 and log-transform not cross-validated | soh_analysis.py:130, run_nyquist_complete.py:51 | Alpha sensitivity sweep |
| F27 | 7.3 | MEDIUM | 42.7% of KIT EIS fits hit parameter bounds | results_kit_gpu0.json | Exclude or flag affected entries |
| F28 | 7.4 | MEDIUM | Samsung uses OCV proxy, not direct SOH measurement | run_popp.py | Acknowledge; validate against capacity if available |
| F29 | 7.5 | MEDIUM | No cross-temperature prediction for Panasonic | run_panasonic.py | Train 25°C → predict other temps |
| F30 | 8.2 | MEDIUM | No requirements.txt or environment specification | Repository root | Add requirements.txt |
| F31 | 8.3 | MEDIUM | Hardcoded paths to local directories | Multiple scripts | Use relative paths; add DATA_ROOT |
| F32 | 1.3 | LOW | No condition number check on impedance least-squares | run_degradation.py:196 | Add condition number warning |
| F33 | 2.1 | LOW | NMSE denominator uses mean(η²), not variance | run_degradation.py:132 | Document normalization choice |
| F34 | 2.5 | LOW | Bare `except: continue` in EIS fitting | run_degradation.py:328 | Specify exception types |
| F35 | 8.4 | LOW | 4 of 7 Stanford cells excluded without explanation | Multiple files | Document selection criteria |

**Severity Distribution:** CRITICAL: 1, HIGH: 13, MEDIUM: 17, LOW: 4. **Total: 35 findings.**

---

## Section 10: Prioritized Recommendations

### Tier 1 — Must-Do for Publication

These address issues that would likely result in major revision requests from reviewers at top venues (Nature Energy, JPS, JES).

1. **Add bootstrap confidence intervals to all headline metrics.** Implement 1000-resample bootstrap for LOCO MAPE, MAE, R², and ρ. Report 95% CIs in all tables and the abstract. *(Addresses F01, F07)*

2. **Set random seeds for reproducibility.** Add `torch.manual_seed(42); np.random.seed(42); random.seed(42)` to all scripts. Report seeds in a methods section. *(Addresses F14)*

3. **Soften "sufficient statistic" language.** Replace with "compact diagnostic representation" or "informative low-dimensional summary." The 0.99% MAPE is strong enough without overclaiming. *(Addresses F03)*

4. **Add explicit n=3 limitation acknowledgment.** In the abstract and limitations section, state that results are based on 3 NMC cells and require validation on larger, multi-chemistry datasets. *(Addresses F01)*

5. **Re-run model order selection with uniform initializations.** Use 10 restarts for all n ∈ {2,3,4,5,6}. Remove the top-k override from survival criteria. Report AIC/BIC alongside NRMSE. *(Addresses F11)*

6. **Remove or quantify warm-start bias on W10.** Either re-run W10 cold-start or apply warm-start uniformly to all cells. Report both warm-start and cold-start results as a sensitivity analysis. *(Addresses F10)*

7. **Add `requirements.txt`.** Pin PyTorch, NumPy, SciPy, pandas, scikit-learn, and matplotlib versions. *(Addresses F30)*

### Tier 2 — Strongly Recommended

These would significantly strengthen the paper and preempt common reviewer concerns.

1. **Cross-dataset validation experiment.** Train Ridge regression on Stanford τ values, test on KIT. Use a shared n=3 model configuration with fixed hyperparameters. *(Addresses F12)*

2. **Ridge alpha sensitivity analysis.** Sweep α ∈ {1e-6, 1e-4, 1e-2, 1e0, 1e2} and report LOCO MAPE. If stable, document; if not, use nested CV. *(Addresses F26)*

3. **Window sweep with matched initializations.** Re-run `run_window_sweep_v2.py` using the union of all window-specific initializations for every window length. *(Addresses F05)*

4. **Warm-start vs. cold-start ablation.** Run all Stanford 3D analyses with and without warm-start. Compare τ trajectories, SOH MAE, and Nyquist MAPE. *(Addresses F04)*

5. **Comparison against DRT methods.** Implement DRT (e.g., Tikhonov-regularized) on the same HPPC pulse data and compare τ extraction accuracy and SOH prediction performance. *(Addresses positioning gap in Section 6)*

6. **Add timing benchmarks.** Report wall-clock time for LGN fitting per diagnostic on CPU and GPU. Compare with EIS measurement time. *(Practical value for BMS deployment narrative)*

7. **Paired hypothesis test for LGN vs. R_pulse.** Compute per-observation absolute errors for both methods. Apply a paired Wilcoxon signed-rank test or paired t-test. Report p-value. *(Addresses F09)*

### Tier 3 — Future Work

These are valuable extensions beyond the scope of the current manuscript.

1. **Validate on LFP, NCA chemistries (50+ cells).** The capacitance-amplification mechanism should transfer across chemistries, but the specific τ–SOH relationships may differ. A multi-chemistry study with 50+ cells per chemistry would establish generalizability.

2. **Temperature-dependent modeling.** Extend the τ–SOH regression to include temperature as a covariate. The Panasonic dataset (5 temperatures) is a natural starting point.

3. **Formal identifiability analysis.** Derive conditions under which {τᵢ, x₀ᵢ} are uniquely identifiable from finite-window observations. This would formalize the "sufficient statistic" claim.

4. **Noise sensitivity study.** Add synthetic voltage noise at levels typical of BMS hardware (±0.5–2 mV) and quantify degradation in τ extraction accuracy.

5. **Online/streaming BMS implementation.** Demonstrate real-time τ extraction on an embedded platform (e.g., ARM Cortex-M) to validate the "BMS-compatible" claim.

---

*This review was conducted by systematic examination of all 17 Python scripts, 5 dataset directories, 3 methodology documents, and associated JSON result files in the LGN-Battery-Diagnostic repository. All file:line references have been verified against the source code. JSON result values have been cross-checked against documented claims.*
