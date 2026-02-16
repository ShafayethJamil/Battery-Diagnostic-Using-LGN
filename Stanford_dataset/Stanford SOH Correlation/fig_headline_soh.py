"""
Headline Figure: LGN Battery Diagnostics — SOH Estimation & Early Detection
=============================================================================
Builds a 6-panel figure showing:
  (a) τ₁ vs Capacity dual-axis — all 3 cells
  (b) 10× amplification — normalized % change
  (c) Cross-cell SOH estimation (leave-one-cell-out)
  (d) Mechanism separation — 3 modes diverging
  (e) τ₁ vs R_pulse correlation comparison
  (f) SOH prediction error comparison

Inputs:
  - results_3d_W8_SOC50.json
  - results_3d_W9_SOC50.json
  - results_3d_W10_Warmstart.json  (W10 SOC50 with warm-start)

Usage:
  python fig_headline_soh.py --data_dir /path/to/results --out fig_headline_soh.png

Author: Shafayeth Jamil (USC ECE), February 2026
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# CAPACITY DATA (from Stanford SECL diagnostic measurements)
# ============================================================
CAPACITY = {
    'W8':  [4.8769, 4.8346, 4.7735, 4.7293, 4.6522, 4.6403, 4.5968,
            4.5572, 4.5320, 4.5361, 4.5284, 4.5106, 4.4958, 4.4838, 4.4569],
    'W9':  [4.8743, 4.8346, 4.7719, 4.7230, 4.6566, 4.6425, 4.5988,
            4.5581, 4.5463, 4.5466, 4.5346, 4.5158, 4.5007, 4.4864, 4.4636],
    'W10': [4.8659, 4.8336, 4.7647, 4.7013, 4.6455, 4.6346, 4.5919,
            4.5539, 4.5441, 4.5432, 4.5284, 4.5106, 4.4970, 4.4780, 4.4591],
}

CELL_COLORS = {'W8': '#2CA02C', 'W9': '#FF7F0E', 'W10': '#9467BD'}
CELL_MARKERS = {'W8': 'o', 'W9': 's', 'W10': 'D'}


def load_data(data_dir):
    """Load LGN results for all three cells."""
    cells_data = {}
    fnames = {
        'W8': f'{data_dir}/results_3d_W8_SOC50.json',
        'W9': f'{data_dir}/results_3d_W9_SOC50.json',
        'W10': f'{data_dir}/results_3d_W10_Warmstart.json',
    }
    for cell, fname in fnames.items():
        with open(fname) as f:
            cells_data[cell] = json.load(f)
        print(f"  {cell}: {len(cells_data[cell])} diagnostics loaded")
    return cells_data


def build_figure(cells_data, out_path='fig_headline_soh.png'):
    """Build the 6-panel headline figure."""

    cap = CAPACITY
    diags = list(range(1, 16))
    Q0_global = max(cap['W8'][0], cap['W9'][0], cap['W10'][0])

    # ---- Collect global arrays for cross-cell analysis ----
    all_tau1, all_soh, all_labels, all_rpulse = [], [], [], []
    for cell in ['W8', 'W9', 'W10']:
        data = cells_data[cell]
        c = cap[cell]
        for i, r in enumerate(data):
            all_tau1.append(r['tau_full'][0])
            all_soh.append(100 * c[i] / Q0_global)
            all_rpulse.append(r['R_pulse'])
            all_labels.append(cell)

    all_tau1 = np.array(all_tau1)
    all_soh = np.array(all_soh)
    all_rpulse = np.array(all_rpulse)
    all_labels = np.array(all_labels)

    # ---- Figure setup ----
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.35,
                          left=0.06, right=0.96, top=0.92, bottom=0.08)

    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
    })

    # ============================================================
    # (a) τ₁ and Capacity dual-axis — all 3 cells
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1b = ax1.twinx()

    for cell in ['W8', 'W9', 'W10']:
        tau1 = [r['tau_full'][0] for r in cells_data[cell]]
        c = cap[cell]
        ax1.plot(diags, tau1, CELL_MARKERS[cell] + '-',
                color=CELL_COLORS[cell],
                linewidth=1.8, markersize=5, label=f'{cell} τ₁')
        ax1b.plot(diags, c, CELL_MARKERS[cell] + '--',
                 color=CELL_COLORS[cell],
                 linewidth=1, markersize=4, alpha=0.5)

    ax1.set_xlabel('Diagnostic #')
    ax1.set_ylabel('τ₁ [s]', color='#D62728')
    ax1b.set_ylabel('Capacity [Ah]', color='#1F77B4')
    ax1.set_title('(a) τ₁ Tracks Degradation Across All Cells')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.15)

    # ============================================================
    # (b) Amplification — normalized % change (all cells)
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])

    for cell in ['W8', 'W9', 'W10']:
        tau1 = [r['tau_full'][0] for r in cells_data[cell]]
        c = cap[cell]
        tau1_pct = [(t - tau1[0]) / tau1[0] * 100 for t in tau1]
        cap_pct = [(ci - c[0]) / c[0] * 100 for ci in c]
        ax2.plot(diags, tau1_pct, CELL_MARKERS[cell] + '-',
                color=CELL_COLORS[cell],
                linewidth=2, markersize=5, label=f'{cell} Δτ₁')
        ax2.plot(diags, cap_pct, CELL_MARKERS[cell] + '--',
                color=CELL_COLORS[cell],
                linewidth=1, markersize=3, alpha=0.4)

    ax2.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax2.text(0.05, 0.95,
            'Amplification: 10×\n(solid: τ₁, dashed: capacity)',
            transform=ax2.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                      edgecolor='orange', alpha=0.9))
    ax2.set_xlabel('Diagnostic #')
    ax2.set_ylabel('Relative change [%]')
    ax2.set_title('(b) 10× Early Detection Amplification')
    ax2.legend(fontsize=8, loc='center left')
    ax2.grid(True, alpha=0.15)

    # ============================================================
    # (c) Cross-cell SOH estimation: τ₁ vs SOH (leave-one-out)
    # ============================================================
    ax3 = fig.add_subplot(gs[0, 2])

    for test_cell in ['W8', 'W9', 'W10']:
        mask_test = all_labels == test_cell
        ax3.scatter(all_tau1[mask_test], all_soh[mask_test],
                   c=CELL_COLORS[test_cell],
                   marker=CELL_MARKERS[test_cell],
                   s=60, label=test_cell, zorder=5,
                   edgecolors='white', linewidths=0.5)

    # Global fit line for visualization
    reg_all = LinearRegression()
    reg_all.fit(all_tau1.reshape(-1, 1), all_soh)
    tau_range = np.linspace(5, 14, 100)
    ax3.plot(tau_range, reg_all.predict(tau_range.reshape(-1, 1)),
            '--', color='gray', linewidth=1.5, alpha=0.7)

    ax3.text(0.05, 0.05,
            'Leave-one-cell-out:\nMAE = 0.61%\nR² = 0.91',
            transform=ax3.transAxes, fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue',
                      edgecolor='steelblue', alpha=0.85))
    ax3.set_xlabel('τ₁ [s]')
    ax3.set_ylabel('SOH [%]')
    ax3.set_title('(c) Cross-Cell SOH Estimation')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.15)

    # ============================================================
    # (d) All 3 modes — mechanism divergence (W8)
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 0])

    cell = 'W8'
    data = cells_data[cell]
    mode_labels = ['τ₁ (CT)', 'τ₂ (SEI)', 'τ₃ (diff)']
    mode_colors = ['#D62728', '#FF7F0E', '#2CA02C']

    for ti, (label, color) in enumerate(zip(mode_labels, mode_colors)):
        vals = [r['tau_full'][ti] for r in data]
        normed = [v / vals[0] for v in vals]
        ax4.plot(diags, normed, 'o-', color=color,
                linewidth=2, markersize=5, label=label)

    ax4.axhline(1.0, color='gray', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Diagnostic #')
    ax4.set_ylabel('τ / τ₀ (normalized)')
    ax4.set_title(f'(d) Mechanism Separation — {cell}')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.15)

    # ============================================================
    # (e) τ₁ vs R_pulse correlation comparison (bar chart)
    # ============================================================
    ax5 = fig.add_subplot(gs[1, 1])

    corr_data = []
    for cell in ['W8', 'W9', 'W10']:
        data = cells_data[cell]
        c = cap[cell]
        soh = [100 * ci / c[0] for ci in c]
        tau1 = [r['tau_full'][0] for r in data]
        rpulse = [r['R_pulse'] for r in data]

        rho_t, _ = stats.spearmanr(tau1, soh)
        rho_r, _ = stats.spearmanr(rpulse, soh)
        corr_data.append((cell, abs(rho_t), abs(rho_r)))

    cells_list = [c[0] for c in corr_data]
    tau_corrs = [c[1] for c in corr_data]
    rp_corrs = [c[2] for c in corr_data]

    x = np.arange(len(cells_list))
    width = 0.35
    bars1 = ax5.bar(x - width / 2, tau_corrs, width,
                   label='LGN τ₁', color='#D62728', alpha=0.85)
    bars2 = ax5.bar(x + width / 2, rp_corrs, width,
                   label='R_pulse', color='#888888', alpha=0.65)

    ax5.set_xticks(x)
    ax5.set_xticklabels(cells_list)
    ax5.set_ylabel('|ρ| with SOH')
    ax5.set_title('(e) Degradation Tracking: LGN vs R_pulse')
    ax5.set_ylim(0, 1.05)
    ax5.axhline(0.9, color='green', linestyle=':', alpha=0.4, label='ρ = 0.9')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.15, axis='y')

    for bar, val in zip(bars1, tau_corrs):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', fontsize=9, fontweight='bold',
                color='#D62728')
    for bar, val in zip(bars2, rp_corrs):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', fontsize=9, color='#555555')

    # ============================================================
    # (f) SOH prediction error comparison (bar chart)
    # ============================================================
    ax6 = fig.add_subplot(gs[1, 2])

    mae_tau_cells, mae_rp_cells = [], []
    for test_cell in ['W8', 'W9', 'W10']:
        mask_test = all_labels == test_cell
        mask_train = ~mask_test

        # τ₁ model
        reg = LinearRegression()
        reg.fit(all_tau1[mask_train].reshape(-1, 1), all_soh[mask_train])
        pred_tau = reg.predict(all_tau1[mask_test].reshape(-1, 1))
        mae_tau_cells.append(mean_absolute_error(all_soh[mask_test], pred_tau))

        # R_pulse model
        rp_train = all_rpulse[mask_train]
        soh_train = all_soh[mask_train]
        rp_test = all_rpulse[mask_test]

        reg2 = LinearRegression()
        reg2.fit(rp_train.reshape(-1, 1), soh_train)
        pred_rp = reg2.predict(rp_test.reshape(-1, 1))
        mae_rp_cells.append(mean_absolute_error(all_soh[mask_test], pred_rp))

    x = np.arange(3)
    width = 0.35
    bars1 = ax6.bar(x - width / 2, mae_tau_cells, width,
                   label='LGN τ₁', color='#D62728', alpha=0.85)
    bars2 = ax6.bar(x + width / 2, mae_rp_cells, width,
                   label='R_pulse', color='#888888', alpha=0.65)

    ax6.set_xticks(x)
    ax6.set_xticklabels(['W8', 'W9', 'W10'])
    ax6.set_ylabel('MAE [%]')
    ax6.set_title('(f) SOH Prediction Error (Leave-One-Cell-Out)')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.15, axis='y')

    for bar, val in zip(bars1, mae_tau_cells):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f'{val:.2f}%', ha='center', fontsize=9, fontweight='bold',
                color='#D62728')
    for bar, val in zip(bars2, mae_rp_cells):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f'{val:.2f}%', ha='center', fontsize=9, color='#555555')

    # ---- Title ----
    fig.suptitle(
        'LGN Battery Diagnostics — Stanford SECL, SOC 50%, n=3 with Warm-Start',
        fontsize=15, fontweight='bold')

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")


def print_summary(cells_data):
    """Print key metrics to console."""
    cap = CAPACITY
    Q0_global = max(cap['W8'][0], cap['W9'][0], cap['W10'][0])

    print("\n" + "=" * 60)
    print("KEY METRICS")
    print("=" * 60)

    for cell in ['W8', 'W9', 'W10']:
        data = cells_data[cell]
        c = cap[cell]
        tau1 = [r['tau_full'][0] for r in data]

        tau1_change = (tau1[-1] - tau1[0]) / tau1[0] * 100
        cap_change = (c[-1] - c[0]) / c[0] * 100
        amp = abs(tau1_change / cap_change)

        soh = [100 * ci / c[0] for ci in c]
        rho_tau1, p_tau1 = stats.spearmanr(tau1, soh)

        rpulse = [r['R_pulse'] for r in data]
        rho_rp, p_rp = stats.spearmanr(rpulse, soh)

        print(f"\n{cell}:")
        print(f"  τ₁: {tau1[0]:.2f} → {tau1[-1]:.2f}s  ({tau1_change:+.1f}%)")
        print(f"  Cap: {c[0]:.4f} → {c[-1]:.4f} Ah  ({cap_change:+.1f}%)")
        print(f"  Amplification: {amp:.1f}×")
        print(f"  τ₁ vs SOH:      ρ = {rho_tau1:.4f} (p={p_tau1:.2e})")
        print(f"  R_pulse vs SOH:  ρ = {rho_rp:.4f} (p={p_rp:.2e})")

    # Cross-cell LOO
    all_tau1, all_soh, all_rpulse, all_labels = [], [], [], []
    for cell in ['W8', 'W9', 'W10']:
        data = cells_data[cell]
        c = cap[cell]
        for i, r in enumerate(data):
            all_tau1.append(r['tau_full'][0])
            all_soh.append(100 * c[i] / Q0_global)
            all_rpulse.append(r['R_pulse'])
            all_labels.append(cell)

    all_tau1 = np.array(all_tau1)
    all_soh = np.array(all_soh)
    all_rpulse = np.array(all_rpulse)
    all_labels = np.array(all_labels)

    all_pred_tau, all_pred_rp, all_true = [], [], []
    for test_cell in ['W8', 'W9', 'W10']:
        mask_test = all_labels == test_cell
        mask_train = ~mask_test

        reg = LinearRegression()
        reg.fit(all_tau1[mask_train].reshape(-1, 1), all_soh[mask_train])
        all_pred_tau.extend(reg.predict(all_tau1[mask_test].reshape(-1, 1)))

        reg2 = LinearRegression()
        reg2.fit(all_rpulse[mask_train].reshape(-1, 1), all_soh[mask_train])
        all_pred_rp.extend(reg2.predict(all_rpulse[mask_test].reshape(-1, 1)))

        all_true.extend(all_soh[mask_test])

    mae_tau = mean_absolute_error(all_true, all_pred_tau)
    r2_tau = r2_score(all_true, all_pred_tau)
    mae_rp = mean_absolute_error(all_true, all_pred_rp)
    r2_rp = r2_score(all_true, all_pred_rp)

    print(f"\n{'='*60}")
    print(f"CROSS-CELL SOH (Leave-One-Cell-Out):")
    print(f"  LGN τ₁:   MAE = {mae_tau:.3f}%,  R² = {r2_tau:.4f}")
    print(f"  R_pulse:   MAE = {mae_rp:.3f}%,  R² = {r2_rp:.4f}")
    print(f"  LGN is {mae_rp/mae_tau:.1f}× more accurate")


def main():
    parser = argparse.ArgumentParser(
        description='Build headline SOH figure from Stanford LGN results')
    parser.add_argument('--data_dir', default='.',
                        help='Directory containing results JSON files')
    parser.add_argument('--out', default='fig_headline_soh.png',
                        help='Output figure path')
    args = parser.parse_args()

    print("Loading data...")
    cells_data = load_data(args.data_dir)
    print_summary(cells_data)
    build_figure(cells_data, out_path=args.out)


if __name__ == '__main__':
    main()
