"""
SOH Estimation from LGN Time Constants
=======================================
Analyzes the relationship between LGN-extracted time constants (τ₁, τ₂, τ₃)
from 10-second HPPC pulses and battery state-of-health (SOH).

Experiments:
    1. Feature–SOH correlations (Spearman/Pearson)
    2. Leave-one-cell-out cross-validated SOH estimation
    3. Head-to-head: LGN τ vs R_pulse baseline
    4. Leading indicator analysis (amplification factor)


Author: Shafayeth (LGN Battery Project)
"""

import json
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

NOMINAL_CAPACITY = 4.85  # Ah (INR21700-M50T)

# Discharge capacity at each diagnostic (from capacity_test.mat)
CAPACITY_DATA = {
    'W8':  [4.8769, 4.8346, 4.7735, 4.7293, 4.6522, 4.6403, 4.5968,
            4.5572, 4.5320, 4.5361, 4.5284, 4.5106, 4.4958, 4.4838, 4.4569],
    'W9':  [4.8743, 4.8346, 4.7719, 4.7230, 4.6566, 4.6425, 4.5988,
            4.5581, 4.5463, 4.5466, 4.5346, 4.5158, 4.5007, 4.4864, 4.4636],
    'W10': [4.8659, 4.8336, 4.7647, 4.7013, 4.6455, 4.6346, 4.5919,
            4.5539, 4.5441, 4.5432, 4.5284, 4.5106, 4.4970, 4.4780, 4.4591],
}

# LGN results files (update paths as needed)
TAU_FILES = {
    'W8':  'results_3d_W8_SOC50.json',
    'W9':  'results_3d_W9_SOC50.json',
    'W10': 'results_3d_W10_Warmstart.json',  # warm-start for W10
}

CELLS = ['W8', 'W9', 'W10']

# Plot style
CELL_COLORS = {'W8': '#1f77b4', 'W9': '#ff7f0e', 'W10': '#2ca02c'}
CELL_MARKERS = {'W8': 'o', 'W9': 's', 'W10': '^'}

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 13,
    'axes.titlesize': 14, 'legend.fontsize': 9.5, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.8, 'lines.linewidth': 1.8, 'lines.markersize': 7,
})


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_data():
    """Load and align tau results with capacity measurements."""
    tau_data = {}
    for cell, path in TAU_FILES.items():
        with open(path) as f:
            tau_data[cell] = json.load(f)

    features, soh, cells, diags, rpulse = [], [], [], [], []
    for cell in CELLS:
        for entry in tau_data[cell]:
            diag = int(entry['diag'])
            if diag > len(CAPACITY_DATA[cell]):
                continue
            Q = CAPACITY_DATA[cell][diag - 1]
            features.append(entry['tau_full'])
            soh.append(Q / NOMINAL_CAPACITY * 100)
            cells.append(cell)
            diags.append(diag)
            rpulse.append(entry['R_pulse'])

    return (tau_data, np.array(features), np.array(soh),
            np.array(cells), np.array(diags), np.array(rpulse))


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: INDIVIDUAL CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════

def correlations(X, y, Rp):
    """Compute Spearman and Pearson correlations for each feature."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Feature → SOH Correlations (all cells pooled)")
    print("=" * 70)

    for ti, name in enumerate(['τ₁ (CT)', 'τ₂ (mid)', 'τ₃ (diff)']):
        rho, p = stats.spearmanr(X[:, ti], y)
        r, _ = stats.pearsonr(X[:, ti], y)
        print(f"  {name}: Spearman ρ = {rho:.3f} (p={p:.2e}), Pearson r = {r:.3f}")

    rho_rp, p_rp = stats.spearmanr(Rp, y)
    r_rp, _ = stats.pearsonr(Rp, y)
    print(f"  R_pulse:  Spearman ρ = {rho_rp:.3f} (p={p_rp:.2e}), Pearson r = {r_rp:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: LEAVE-ONE-CELL-OUT SOH ESTIMATION
# ══════════════════════════════════════════════════════════════════════════

def loocv_soh(X, y, cells_arr, Rp):
    """Leave-one-cell-out cross-validation for SOH estimation."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Leave-One-Cell-Out SOH Estimation")
    print("=" * 70)

    pred_tau_all = np.zeros_like(y)
    pred_tau1 = np.zeros_like(y)
    pred_rp = np.zeros_like(y)

    for test_cell in CELLS:
        train = cells_arr != test_cell
        test = cells_arr == test_cell

        # Full τ model
        reg = LinearRegression().fit(np.log(X[train]), y[train])
        pred_tau_all[test] = reg.predict(np.log(X[test]))

        # τ₁ only
        reg1 = LinearRegression().fit(np.log(X[train, 0:1]), y[train])
        pred_tau1[test] = reg1.predict(np.log(X[test, 0:1]))

        # R_pulse baseline
        reg_rp = LinearRegression().fit(Rp[train].reshape(-1, 1), y[train])
        pred_rp[test] = reg_rp.predict(Rp[test].reshape(-1, 1))

        # Per-cell results
        mae = mean_absolute_error(y[test], pred_tau1[test])
        r2 = r2_score(y[test], pred_tau1[test])
        print(f"\n  Test {test_cell}: τ₁-only MAE={mae:.3f}%, R²={r2:.3f}")

    print("\n  OVERALL (cross-validated):")
    for name, pred in [('[τ₁,τ₂,τ₃]', pred_tau_all),
                       ('τ₁ only', pred_tau1),
                       ('R_pulse', pred_rp)]:
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)
        print(f"    {name:15s}: MAE={mae:.3f}%, R²={r2:.3f}")

    return pred_tau_all, pred_tau1, pred_rp


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: LEADING INDICATOR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

def leading_indicator(tau_data):
    """Compute amplification factors for each cell."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Leading Indicator (Amplification Factor)")
    print("=" * 70)

    for cell in CELLS:
        entries = tau_data[cell]
        Q = [CAPACITY_DATA[cell][int(e['diag']) - 1] for e in entries]
        Q0, t0 = Q[0], entries[0]['tau_full']

        dQ_final = (Q[-1] / Q0 - 1) * 100
        dt_final = [(entries[-1]['tau_full'][i] / t0[i] - 1) * 100 for i in range(3)]
        amps = [abs(dt / dQ_final) for dt in dt_final]

        print(f"\n  {cell}: ΔQ = {dQ_final:.1f}%")
        for i, name in enumerate(['τ₁', 'τ₂', 'τ₃']):
            print(f"    {name}: Δ = +{dt_final[i]:.0f}% → {amps[i]:.0f}× amplification")


# ══════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════

def plot_leading_indicator(tau_data):
    """Figure 1: Leading indicator — τ change vs capacity change."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ci, cell in enumerate(CELLS):
        ax = axes[ci]
        entries = tau_data[cell]
        Q = [CAPACITY_DATA[cell][int(e['diag']) - 1] for e in entries]
        diags = [int(e['diag']) for e in entries]
        Q0, t0 = Q[0], entries[0]['tau_full']

        dQ = [(q / Q0 - 1) * 100 for q in Q]
        dt = [[(e['tau_full'][i] / t0[i] - 1) * 100 for e in entries] for i in range(3)]

        ax.plot(diags, dQ, 'k-o', lw=2, ms=6, label='Capacity', zorder=5)
        for i, (c, m, lbl) in enumerate([
            ('#d62728', 's', r'$\tau_1$ (CT)'),
            ('#ff7f0e', '^', r'$\tau_2$ (mid)'),
            ('#2ca02c', 'D', r'$\tau_3$ (diff)'),
        ]):
            ax.plot(diags, dt[i], f'-{m}', color=c, lw=1.5, ms=5, label=lbl, alpha=0.85)

        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_xlabel('Diagnostic Number')
        ax.set_ylabel('Change from Initial [%]')
        ax.set_title(f'Cell {cell}', fontweight='bold')
        ax.legend(loc='best', framealpha=0.9, edgecolor='none')
        ax.grid(True, alpha=0.15)

        amp = abs(dt[0][-1] / dQ[-1])
        ax.annotate(f'τ₁ amplification: {amp:.0f}×', xy=(0.05, 0.05),
                    xycoords='axes fraction', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9))

    fig.suptitle('Leading Indicator: Time Constants vs Capacity Fade',
                 fontweight='bold', fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig('fig_leading_indicator.png', dpi=300, bbox_inches='tight')
    print("  Saved fig_leading_indicator.png")


def plot_soh_parity(y, pred_tau_all, pred_tau1, pred_rp, cells_arr):
    """Figure 2: SOH parity plots for three models."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for pi, (pred, label) in enumerate([
        (pred_tau_all, r'LGN [$\tau_1, \tau_2, \tau_3$]'),
        (pred_tau1, r'LGN $\tau_1$ only'),
        (pred_rp, r'$R_{\mathrm{pulse}}$'),
    ]):
        ax = axes[pi]
        for cell in CELLS:
            mask = cells_arr == cell
            ax.scatter(y[mask], pred[mask], c=CELL_COLORS[cell],
                       marker=CELL_MARKERS[cell], s=50, alpha=0.8,
                       edgecolors='white', linewidth=0.5, label=cell, zorder=3)

        lo, hi = 91, 101.5
        ax.plot([lo, hi], [lo, hi], '--', color='gray', alpha=0.5, lw=1)
        ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1],
                        alpha=0.08, color='green')

        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)
        mx = np.max(np.abs(y - pred))
        ax.annotate(f'MAE = {mae:.2f}%\nR² = {r2:.3f}\nMax err = {mx:.2f}%',
                    xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.85))

        ax.set_xlabel('True SOH [%]')
        ax.set_ylabel('Predicted SOH [%]')
        ax.set_title(label, fontweight='bold')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.legend(loc='lower right', framealpha=0.9, edgecolor='none')
        ax.grid(True, alpha=0.15)

    fig.suptitle('SOH Estimation — Leave-One-Cell-Out Cross-Validation',
                 fontweight='bold', fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig('fig_soh_estimation.png', dpi=300, bbox_inches='tight')
    print("  Saved fig_soh_estimation.png")


def plot_soh_trajectory(X, y, cells_arr, diags_arr):
    """Figure 3: Cross-cell SOH trajectory prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ci, test_cell in enumerate(CELLS):
        ax = axes[ci]
        train = cells_arr != test_cell
        test = cells_arr == test_cell

        reg = LinearRegression().fit(np.log(X[train, 0:1]), y[train])
        pred = reg.predict(np.log(X[test, 0:1]))
        td, tt = diags_arr[test], y[test]

        ax.plot(td, tt, 'ko-', lw=2, ms=7, label='True SOH', zorder=5)
        ax.plot(td, pred, 's--', color='#d62728', lw=1.8, ms=6,
                label='Predicted (τ₁)', alpha=0.85, zorder=4)
        ax.fill_between(td, pred - 1, pred + 1, alpha=0.1, color='#d62728')

        mae = mean_absolute_error(tt, pred)
        r2 = r2_score(tt, pred)
        others = [c for c in CELLS if c != test_cell]
        ax.annotate(f'Train: {", ".join(others)}\nTest: {test_cell}\n'
                    f'MAE = {mae:.2f}%\nR² = {r2:.3f}',
                    xy=(0.55, 0.95), xycoords='axes fraction', fontsize=10, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9))

        ax.set_xlabel('Diagnostic Number')
        ax.set_ylabel('SOH [%]')
        ax.set_title(f'Cell {test_cell} (unseen)', fontweight='bold')
        ax.legend(loc='lower left', framealpha=0.9, edgecolor='none')
        ax.grid(True, alpha=0.15)
        ax.set_ylim(90.5, 102)

    fig.suptitle('Cross-Cell SOH Trajectory Prediction from τ₁ Only',
                 fontweight='bold', fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig('fig_soh_trajectory.png', dpi=300, bbox_inches='tight')
    print("  Saved fig_soh_trajectory.png")


def plot_feature_correlations(X, y, Rp, cells_arr):
    """Figure 4: Feature–SOH scatter plots."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for fi, (tidx, sym, name) in enumerate([
        (0, r'$\tau_1$', 'Charge Transfer'),
        (1, r'$\tau_2$', 'Mid-frequency'),
        (2, r'$\tau_3$', 'Diffusion'),
    ]):
        ax = axes[fi]
        for cell in CELLS:
            mask = cells_arr == cell
            ax.scatter(X[mask, tidx], y[mask], c=CELL_COLORS[cell],
                       marker=CELL_MARKERS[cell], s=50, alpha=0.8,
                       edgecolors='white', linewidth=0.5, label=cell, zorder=3)

        rho, _ = stats.spearmanr(X[:, tidx], y)
        r, _ = stats.pearsonr(X[:, tidx], y)
        ax.annotate(f'ρ = {rho:.3f}\nr = {r:.3f}', xy=(0.05, 0.05),
                    xycoords='axes fraction', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.85))
        ax.set_xlabel(f'{sym} [s]'); ax.set_ylabel('SOH [%]')
        ax.set_title(f'{sym} ({name})', fontweight='bold')
        ax.set_xscale('log')
        ax.legend(loc='upper right', framealpha=0.9, edgecolor='none')
        ax.grid(True, alpha=0.15, which='both')

    # R_pulse panel
    ax = axes[3]
    for cell in CELLS:
        mask = cells_arr == cell
        ax.scatter(Rp[mask] * 1000, y[mask], c=CELL_COLORS[cell],
                   marker=CELL_MARKERS[cell], s=50, alpha=0.8,
                   edgecolors='white', linewidth=0.5, label=cell, zorder=3)
    rho_rp, _ = stats.spearmanr(Rp, y)
    r_rp, _ = stats.pearsonr(Rp, y)
    ax.annotate(f'ρ = {rho_rp:.3f}\nr = {r_rp:.3f}', xy=(0.05, 0.05),
                xycoords='axes fraction', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.85))
    ax.set_xlabel(r'$R_{\mathrm{pulse}}$ [mΩ]'); ax.set_ylabel('SOH [%]')
    ax.set_title(r'$R_{\mathrm{pulse}}$ (baseline)', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='none')
    ax.grid(True, alpha=0.15)

    fig.suptitle('Feature–SOH Correlations: LGN Time Constants vs Pulse Resistance',
                 fontweight='bold', fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig('fig_correlations.png', dpi=300, bbox_inches='tight')
    print("  Saved fig_correlations.png")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    tau_data, X, y, cells_arr, diags_arr, Rp = load_data()
    print(f"Loaded {len(y)} samples across {len(CELLS)} cells")
    print(f"SOH range: {y.min():.1f}% – {y.max():.1f}%")

    # Numerical results
    correlations(X, y, Rp)
    pred_all, pred_t1, pred_rp = loocv_soh(X, y, cells_arr, Rp)
    leading_indicator(tau_data)

    # Figures
    print("\nGenerating figures...")
    plot_leading_indicator(tau_data)
    plot_soh_parity(y, pred_all, pred_t1, pred_rp, cells_arr)
    plot_soh_trajectory(X, y, cells_arr, diags_arr)
    plot_feature_correlations(X, y, Rp, cells_arr)
    print("Done!")
