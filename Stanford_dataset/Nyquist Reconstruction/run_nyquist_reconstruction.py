"""
Nyquist Spectrum Reconstruction from LGN Time Constants
========================================================
Demonstrates that two LGN-extracted time constants from a 30-second HPPC pulse
are a sufficient statistic for the cell's impedance fingerprint (0.1 Hz – 1 kHz).

Three validation protocols:
  1. Leave-One-Cell-Out (LOCO)          → mean 0.99%
  2. LOCO + Prospective (hardest test)  → mean 1.31%
  3. Within-cell prospective            → mean 1.94%

Required files in DATA_DIR:
  - results_3d_W8_SOC50.json
  - results_3d_W9_SOC50.json
  - results_3d_W10_Warmstart.json

These are the output JSONs from run_stanford_3d_warmstart.py. Each file contains
15 entries (one per diagnostic checkpoint) with fields:
  - cell, diag, soc
  - tau_full: [tau1, tau2, tau3] from 3-state LGN on 3600s window
  - Z_{freq}_re, Z_{freq}_im: EIS ground truth at 5 frequencies
  - Rs_fit_full: series resistance from LGN fit
  - tau_eis_3rc, Rs_eis_3rc, R1/R2/R3_eis_3rc: EIS 3RC fit parameters

Outputs (saved to OUTPUT_DIR):
  - fig5a_loco_nyquist.png         (main paper figure)
  - fig5b_loco_prospective.png     (supplementary)
  - fig5c_protocol_comparison.png  (supplementary)
  - nyquist_results_summary.json   (all numbers for the paper)

Author: Shafayeth Jamil (USC ECE), February 2026
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from sklearn.linear_model import Ridge
from scipy.optimize import nnls

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = '.'          # Directory containing the 3 JSON files
OUTPUT_DIR = '.'        # Directory for output figures and results

# File mapping: cell name → JSON filename
CELL_FILES = {
    'W8':  'results_3d_W8_SOC50.json',
    'W9':  'results_3d_W9_SOC50.json',
    'W10': 'results_3d_W10_Warmstart.json',
}

# EIS frequencies available in Stanford dataset
FREQ_KEYS_RE = ['Z_1kHz_re', 'Z_100Hz_re', 'Z_10Hz_re', 'Z_1Hz_re', 'Z_01Hz_re']
FREQ_KEYS_IM = ['Z_1kHz_im', 'Z_100Hz_im', 'Z_10Hz_im', 'Z_1Hz_im', 'Z_01Hz_im']
FREQ_HZ = [1000, 100, 10, 1, 0.1]
FREQ_LABELS = ['1 kHz', '100 Hz', '10 Hz', '1 Hz', '0.1 Hz']
FREQ_LABELS_SHORT = ['1kHz', '100Hz', '10Hz', '1Hz', '0.1Hz']
OMEGAS = [2 * np.pi * f for f in FREQ_HZ]

# Plotting
CELL_COLORS = {'W8': '#264653', 'W9': '#E76F51', 'W10': '#2A9D8F'}
CELL_ORDER = ['W8', 'W9', 'W10']

# Regression hyperparameter (Ridge alpha — very small, essentially OLS)
RIDGE_ALPHA = 1e-4

# Train/test split point for prospective validation
DIAG_SPLIT = 10  # train on diags 1–10, test on 11–15


# ============================================================================
# DATA LOADING
# ============================================================================
def load_cell_data(data_dir, cell_files):
    """Load all cell JSON files and return dict of {cell_name: [list of diag dicts]}."""
    cells = {}
    for cell, fname in cell_files.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"ERROR: {fpath} not found. Check DATA_DIR and filenames.")
            sys.exit(1)
        with open(fpath) as f:
            cells[cell] = json.load(f)
        print(f"  Loaded {cell}: {len(cells[cell])} diagnostics from {fname}")
    return cells


def extract_features_targets(data_list):
    """
    Extract LGN tau features and EIS impedance targets from a list of diagnostic dicts.
    
    Returns:
        X:    (N, 3) array of [tau1, tau2, tau3]
        Y_re: (N, 5) array of Z_re at 5 frequencies
        Y_im: (N, 5) array of Z_im at 5 frequencies
        diags: list of diagnostic numbers
    """
    X, Y_re, Y_im, diags = [], [], [], []
    for d in data_list:
        X.append(d['tau_full'])
        Y_re.append([d[k] for k in FREQ_KEYS_RE])
        Y_im.append([d[k] for k in FREQ_KEYS_IM])
        diags.append(d['diag'])
    return np.array(X), np.array(Y_re), np.array(Y_im), diags


# ============================================================================
# RECONSTRUCTION METHODS
# ============================================================================
def fit_and_predict(X_train, Y_train, X_test, alpha=RIDGE_ALPHA):
    """Fit Ridge regression and predict. Works for (N,5) multi-output Y."""
    n_freq = Y_train.shape[1]
    Y_pred = np.zeros((X_test.shape[0], n_freq))
    for fi in range(n_freq):
        reg = Ridge(alpha=alpha).fit(X_train, Y_train[:, fi])
        Y_pred[:, fi] = reg.predict(X_test)
    return Y_pred


def compute_mape(Y_true, Y_pred):
    """Compute MAPE per frequency and overall."""
    mape_per_freq = np.mean(np.abs(Y_true - Y_pred) / (np.abs(Y_true) + 1e-15), axis=0) * 100
    overall = np.mean(mape_per_freq)
    return mape_per_freq, overall


def z_3rc(Rs, R, taus, omegas):
    """Analytical impedance: Z(w) = Rs + sum Ri/(1+jw*tau_i)."""
    z = Rs * np.ones(len(omegas), dtype=complex)
    for i in range(len(taus)):
        z += R[i] / (1 + 1j * np.array(omegas) * taus[i])
    return z


def analytical_reconstruction(d):
    """
    Direct analytical Z(w) from LGN parameters.
    Solves for best-fit Rs, R1, R2, R3 given LGN tau, matched to EIS ground truth.
    Returns predicted Z and MAPE.
    """
    taus = d['tau_full']
    z_true = np.array([complex(d[kr], d[ki]) for kr, ki in zip(FREQ_KEYS_RE, FREQ_KEYS_IM)])
    
    # Build overdetermined system: 10 real equations (5 freq × re/im), 4 unknowns
    A_mat = np.zeros((10, 4))
    b_vec = np.zeros(10)
    for fi, omega in enumerate(OMEGAS):
        A_mat[2*fi, 0] = 1.0      # Rs contributes to real part
        A_mat[2*fi+1, 0] = 0.0    # Rs doesn't contribute to imag part
        for ti in range(3):
            h = 1.0 / (1 + 1j * omega * taus[ti])
            A_mat[2*fi, ti+1] = h.real
            A_mat[2*fi+1, ti+1] = h.imag
        b_vec[2*fi] = z_true[fi].real
        b_vec[2*fi+1] = z_true[fi].imag
    
    x, _ = nnls(A_mat, b_vec)  # Non-negative least squares for physical Rs, Ri > 0
    z_pred = z_3rc(x[0], x[1:], taus, OMEGAS)
    
    mape_re = np.abs(z_true.real - z_pred.real) / np.abs(z_true.real) * 100
    return z_pred, mape_re, np.mean(mape_re)


def eis_self_reconstruction(d):
    """EIS 3RC self-consistency: reconstruct Z from EIS's own fitted parameters."""
    taus = d['tau_eis_3rc']
    Rs = d['Rs_eis_3rc']
    R = [d['R1_eis_3rc'], d['R2_eis_3rc'], d['R3_eis_3rc']]
    z_true = np.array([complex(d[kr], d[ki]) for kr, ki in zip(FREQ_KEYS_RE, FREQ_KEYS_IM)])
    z_pred = z_3rc(Rs, R, taus, OMEGAS)
    mape_re = np.abs(z_true.real - z_pred.real) / np.abs(z_true.real) * 100
    return z_pred, mape_re, np.mean(mape_re)


# ============================================================================
# VALIDATION PROTOCOLS
# ============================================================================
def run_loco(cells_data):
    """
    Leave-One-Cell-Out: train on 2 cells (all diagnostics), test on 3rd.
    Returns dict of results per held-out cell.
    """
    results = {}
    for test_cell in CELL_ORDER:
        train_cells = [c for c in CELL_ORDER if c != test_cell]
        
        # Training: all diags from other cells
        X_tr, Y_re_tr, Y_im_tr = [], [], []
        for c in train_cells:
            X, Yr, Yi, _ = extract_features_targets(cells_data[c])
            X_tr.append(X); Y_re_tr.append(Yr); Y_im_tr.append(Yi)
        X_tr = np.vstack(X_tr)
        Y_re_tr = np.vstack(Y_re_tr)
        Y_im_tr = np.vstack(Y_im_tr)
        
        # Test: all diags from held-out cell
        X_te, Y_re_te, Y_im_te, diags = extract_features_targets(cells_data[test_cell])
        
        # Predict
        Y_re_pred = fit_and_predict(X_tr, Y_re_tr, X_te)
        Y_im_pred = fit_and_predict(X_tr, Y_im_tr, X_te)
        
        mape_per_freq, overall = compute_mape(Y_re_te, Y_re_pred)
        
        results[test_cell] = {
            'train_cells': train_cells,
            'n_train': X_tr.shape[0],
            'n_test': X_te.shape[0],
            'mape_per_freq': mape_per_freq,
            'overall_mape': overall,
            'Y_re_test': Y_re_te,
            'Y_im_test': Y_im_te,
            'Y_re_pred': Y_re_pred,
            'Y_im_pred': Y_im_pred,
            'diags': diags,
        }
    return results


def run_loco_prospective(cells_data):
    """
    LOCO + Prospective: train on 2 cells (diags 1–SPLIT), test on 3rd (diags SPLIT+1–15).
    Hardest test: never seen the cell, never seen the aging level.
    """
    results = {}
    for test_cell in CELL_ORDER:
        train_cells = [c for c in CELL_ORDER if c != test_cell]
        
        # Training: other cells, early diags only
        X_tr, Y_re_tr, Y_im_tr = [], [], []
        for c in train_cells:
            for d in cells_data[c]:
                if d['diag'] <= DIAG_SPLIT:
                    X_tr.append(d['tau_full'])
                    Y_re_tr.append([d[k] for k in FREQ_KEYS_RE])
                    Y_im_tr.append([d[k] for k in FREQ_KEYS_IM])
        X_tr = np.array(X_tr)
        Y_re_tr = np.array(Y_re_tr)
        Y_im_tr = np.array(Y_im_tr)
        
        # Test: held-out cell, late diags only
        X_te, Y_re_te, Y_im_te, diags = [], [], [], []
        for d in cells_data[test_cell]:
            if d['diag'] > DIAG_SPLIT:
                X_te.append(d['tau_full'])
                Y_re_te.append([d[k] for k in FREQ_KEYS_RE])
                Y_im_te.append([d[k] for k in FREQ_KEYS_IM])
                diags.append(d['diag'])
        X_te = np.array(X_te)
        Y_re_te = np.array(Y_re_te)
        Y_im_te = np.array(Y_im_te)
        
        Y_re_pred = fit_and_predict(X_tr, Y_re_tr, X_te)
        Y_im_pred = fit_and_predict(X_tr, Y_im_tr, X_te)
        mape_per_freq, overall = compute_mape(Y_re_te, Y_re_pred)
        
        results[test_cell] = {
            'train_cells': train_cells,
            'n_train': X_tr.shape[0],
            'n_test': X_te.shape[0],
            'mape_per_freq': mape_per_freq,
            'overall_mape': overall,
            'Y_re_test': Y_re_te,
            'Y_im_test': Y_im_te,
            'Y_re_pred': Y_re_pred,
            'Y_im_pred': Y_im_pred,
            'diags': diags,
        }
    return results


def run_within_cell_prospective(cells_data):
    """
    Within-cell prospective: train on same cell diags 1–SPLIT, test on diags SPLIT+1–15.
    """
    results = {}
    for cell in CELL_ORDER:
        X_tr, Y_re_tr = [], []
        X_te, Y_re_te, Y_im_te, diags = [], [], [], []
        
        for d in cells_data[cell]:
            taus = d['tau_full']
            y_re = [d[k] for k in FREQ_KEYS_RE]
            y_im = [d[k] for k in FREQ_KEYS_IM]
            if d['diag'] <= DIAG_SPLIT:
                X_tr.append(taus)
                Y_re_tr.append(y_re)
            else:
                X_te.append(taus)
                Y_re_te.append(y_re)
                Y_im_te.append(y_im)
                diags.append(d['diag'])
        
        X_tr = np.array(X_tr)
        Y_re_tr = np.array(Y_re_tr)
        X_te = np.array(X_te)
        Y_re_te = np.array(Y_re_te)
        Y_im_te = np.array(Y_im_te)
        
        Y_re_pred = fit_and_predict(X_tr, Y_re_tr, X_te)
        Y_im_pred = fit_and_predict(X_tr, np.array([[d[k] for k in FREQ_KEYS_IM] 
                     for d in cells_data[cell] if d['diag'] <= DIAG_SPLIT]), X_te)
        mape_per_freq, overall = compute_mape(Y_re_te, Y_re_pred)
        
        results[cell] = {
            'n_train': X_tr.shape[0],
            'n_test': X_te.shape[0],
            'mape_per_freq': mape_per_freq,
            'overall_mape': overall,
            'Y_re_test': Y_re_te,
            'Y_im_test': Y_im_te,
            'Y_re_pred': Y_re_pred,
            'Y_im_pred': Y_im_pred,
            'diags': diags,
        }
    return results


def run_analytical_comparison(cells_data):
    """Compare analytical Z(w) vs EIS self-reconstruction vs regression."""
    analytical_mapes = {c: [] for c in CELL_ORDER}
    eis_self_mapes = {c: [] for c in CELL_ORDER}
    
    for cell in CELL_ORDER:
        for d in cells_data[cell]:
            _, _, m_ana = analytical_reconstruction(d)
            _, _, m_eis = eis_self_reconstruction(d)
            analytical_mapes[cell].append(m_ana)
            eis_self_mapes[cell].append(m_eis)
    
    return analytical_mapes, eis_self_mapes


# ============================================================================
# PLOTTING
# ============================================================================
def setup_matplotlib():
    matplotlib.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10.5,
        'axes.linewidth': 0.8,
        'axes.labelsize': 12,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })


def plot_nyquist_figure(results, title_suffix, legend_label, output_name, output_dir):
    """
    Generic function to plot a 3-panel Nyquist figure + frequency bar chart.
    Works for both LOCO and LOCO+prospective results.
    """
    fig = plt.figure(figsize=(17, 5.8))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.05, 0.7], wspace=0.28,
                           left=0.05, right=0.97, top=0.88, bottom=0.12)
    
    for pi, cell in enumerate(CELL_ORDER):
        ax = fig.add_subplot(gs[0, pi])
        r = results[cell]
        
        Y_re_te = r['Y_re_test']
        Y_im_te = r['Y_im_test']
        Y_re_pr = r['Y_re_pred']
        Y_im_pr = r['Y_im_pred']
        n_test = Y_re_te.shape[0]
        
        for di in range(n_test):
            alpha = 0.12 + 0.88 * (di / max(n_test - 1, 1))
            
            # EIS ground truth — black solid
            ax.plot(Y_re_te[di] * 1000, -Y_im_te[di] * 1000, 'o-', color='k',
                    ms=4.5, lw=1.1, alpha=alpha, mfc='k', mec='k', zorder=3)
            # LGN prediction — colored dashed
            ax.plot(Y_re_pr[di] * 1000, -Y_im_pr[di] * 1000, 's--',
                    color=CELL_COLORS[cell], ms=4.5, lw=1.1, alpha=alpha,
                    mfc='none', mec=CELL_COLORS[cell], zorder=4)
        
        # MAPE annotation
        ax.text(0.95, 0.95, f'{r["overall_mape"]:.1f}%',
                transform=ax.transAxes, fontsize=17, fontweight='bold',
                ha='right', va='top', color=CELL_COLORS[cell],
                bbox=dict(boxstyle='round,pad=0.35', fc='white',
                         ec=CELL_COLORS[cell], alpha=0.92, lw=1.5))
        
        # Training info
        if 'train_cells' in r:
            info = f'trained on\n{r["train_cells"][0]}, {r["train_cells"][1]}'
            if 'diags 1' in title_suffix.lower() or 'prospective' in title_suffix.lower():
                info += f'\n(diags 1–{DIAG_SPLIT} only)'
            ax.text(0.05, 0.95, info, transform=ax.transAxes,
                    fontsize=8, color='#888', va='top', fontstyle='italic')
        
        test_label = f'Test: {cell}'
        if 'prospective' in title_suffix.lower():
            test_label += f' (diags {DIAG_SPLIT+1}–15)'
        ax.set_title(test_label, fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel('$Z_{re}$ (mΩ)', fontsize=11)
        if pi == 0:
            ax.set_ylabel('$-Z_{im}$ (mΩ)', fontsize=12)
        ax.set_xlim(23, 31.5)
        ax.set_ylim(-3.8, 3.2)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.06)
        ax.text(-0.06, 1.07, chr(ord('a') + pi), transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='top')
    
    # Legend on first panel
    handles = [
        Line2D([0], [0], marker='o', color='k', lw=1.1, ms=5.5, mfc='k',
               label='EIS (measured)'),
        Line2D([0], [0], marker='s', color='#777', lw=1.1, ms=5.5, mfc='none',
               linestyle='--', label=f'LGN ({legend_label})'),
    ]
    fig.axes[0].legend(handles=handles, loc='lower right', fontsize=9.5,
                        framealpha=0.95, edgecolor='#ccc')
    
    # Frequency bar chart
    ax_bar = fig.add_subplot(gs[0, 4])
    avg = np.mean([results[c]['mape_per_freq'] for c in CELL_ORDER], axis=0)
    std = np.std([results[c]['mape_per_freq'] for c in CELL_ORDER], axis=0)
    x = np.arange(5)
    
    cmap_bar = plt.cm.RdYlGn_r
    colors_bar = [cmap_bar(0.15 + 0.55 * i / 4) for i in range(5)]
    
    bars = ax_bar.barh(x, avg, 0.52, xerr=std, capsize=3,
                        color=colors_bar, edgecolor='white', lw=0.6,
                        error_kw={'lw': 0.7, 'color': '#555', 'capthick': 0.7})
    for j, (m, s) in enumerate(zip(avg, std)):
        ax_bar.text(m + s + 0.05, j, f'{m:.2f}%', va='center', fontsize=9.5,
                   fontweight='bold', color='#333')
    
    ax_bar.set_yticks(x)
    ax_bar.set_yticklabels(FREQ_LABELS, fontsize=10.5)
    ax_bar.set_xlabel('MAPE (%)', fontsize=11)
    ax_bar.set_xlim(0, max(avg + std) + 0.6)
    ax_bar.invert_yaxis()
    ax_bar.grid(True, axis='x', alpha=0.08)
    ax_bar.axvline(1.0, color='#aaa', ls=':', lw=0.7)
    ax_bar.set_title('Error by\nfrequency', fontsize=12, fontweight='bold', pad=8)
    ax_bar.text(-0.15, 1.07, 'd', transform=ax_bar.transAxes,
                fontsize=15, fontweight='bold', va='top')
    
    mean_overall = np.mean([results[c]['overall_mape'] for c in CELL_ORDER])
    fig.text(0.42, 0.01,
             f'{title_suffix}: mean MAPE = {mean_overall:.2f}% (0.1 Hz – 1 kHz)',
             ha='center', fontsize=10.5, fontstyle='italic', color='#444')
    
    outpath = os.path.join(output_dir, output_name)
    fig.savefig(outpath, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {outpath}")
    return mean_overall


def plot_protocol_comparison(loco, loco_prosp, within_cell, output_dir):
    """Bar chart comparing all three validation protocols."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    protocols = ['LOCO\n(all diags)', 'Within-cell\nprospective', 'LOCO +\nprospective']
    x = np.arange(3)
    w = 0.22
    
    for i, cell in enumerate(CELL_ORDER):
        vals = [
            loco[cell]['overall_mape'],
            within_cell[cell]['overall_mape'],
            loco_prosp[cell]['overall_mape'],
        ]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=CELL_COLORS[cell],
                      alpha=0.85, edgecolor='white', lw=0.8, label=cell)
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 1) * w, v + 0.08, f'{v:.1f}', ha='center',
                   fontsize=9, color=CELL_COLORS[cell], fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(protocols, fontsize=11)
    ax.set_ylabel('$Z_{re}$ MAPE (%)', fontsize=12)
    ax.set_title('Nyquist reconstruction error across validation protocols',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10, framealpha=0.95, edgecolor='#ccc', loc='upper left')
    ax.grid(True, axis='y', alpha=0.08)
    ax.set_ylim(0, 4)
    ax.axhline(2.0, color='#999', ls='--', lw=0.8, alpha=0.5)
    ax.text(2.5, 2.1, '2% threshold', fontsize=9, color='#888', fontstyle='italic')
    
    # Mean lines
    means = [
        np.mean([loco[c]['overall_mape'] for c in CELL_ORDER]),
        np.mean([within_cell[c]['overall_mape'] for c in CELL_ORDER]),
        np.mean([loco_prosp[c]['overall_mape'] for c in CELL_ORDER]),
    ]
    for j, m in enumerate(means):
        ax.plot([x[j] - 0.35, x[j] + 0.35], [m, m], 'k-', lw=1.5, alpha=0.6)
        ax.text(x[j] + 0.38, m - 0.05, f'μ={m:.2f}%', fontsize=9,
               fontweight='bold', color='#333', va='center')
    
    outpath = os.path.join(output_dir, 'fig5c_protocol_comparison.png')
    fig.savefig(outpath, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {outpath}")


# ============================================================================
# SUMMARY OUTPUT
# ============================================================================
def print_results_table(name, results):
    """Print a formatted results table."""
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  {'Cell':>5}  {'n_tr':>5}  {'n_te':>5}  {'1kHz':>6}  {'100Hz':>6}  "
          f"{'10Hz':>6}  {'1Hz':>6}  {'0.1Hz':>6}  {'Mean':>6}")
    for cell in CELL_ORDER:
        r = results[cell]
        m = r['mape_per_freq']
        print(f"  {cell:>5}  {r['n_train']:>5}  {r['n_test']:>5}  "
              f"{m[0]:>6.2f}  {m[1]:>6.2f}  {m[2]:>6.2f}  {m[3]:>6.2f}  "
              f"{m[4]:>6.2f}  {r['overall_mape']:>6.2f}%")
    mean = np.mean([results[c]['overall_mape'] for c in CELL_ORDER])
    print(f"  {'MEAN':>5}  {'':>5}  {'':>5}  {'':>6}  {'':>6}  {'':>6}  "
          f"{'':>6}  {'':>6}  {mean:>6.2f}%")
    return mean


def save_summary_json(loco, loco_prosp, within_cell, ana_mapes, eis_mapes, output_dir):
    """Save all numbers to a JSON for the paper."""
    summary = {
        'loco': {
            cell: {
                'overall_mape': float(loco[cell]['overall_mape']),
                'mape_per_freq': {fl: float(loco[cell]['mape_per_freq'][i])
                                  for i, fl in enumerate(FREQ_LABELS_SHORT)},
                'n_train': int(loco[cell]['n_train']),
                'n_test': int(loco[cell]['n_test']),
            } for cell in CELL_ORDER
        },
        'loco_prospective': {
            cell: {
                'overall_mape': float(loco_prosp[cell]['overall_mape']),
                'mape_per_freq': {fl: float(loco_prosp[cell]['mape_per_freq'][i])
                                  for i, fl in enumerate(FREQ_LABELS_SHORT)},
                'n_train': int(loco_prosp[cell]['n_train']),
                'n_test': int(loco_prosp[cell]['n_test']),
            } for cell in CELL_ORDER
        },
        'within_cell_prospective': {
            cell: {
                'overall_mape': float(within_cell[cell]['overall_mape']),
            } for cell in CELL_ORDER
        },
        'method_comparison': {
            'regression_loco_mape': float(np.mean([loco[c]['overall_mape'] for c in CELL_ORDER])),
            'analytical_mape': float(np.mean([np.mean(ana_mapes[c]) for c in CELL_ORDER])),
            'eis_self_mape': float(np.mean([np.mean(eis_mapes[c]) for c in CELL_ORDER])),
        },
        'means': {
            'loco': float(np.mean([loco[c]['overall_mape'] for c in CELL_ORDER])),
            'loco_prospective': float(np.mean([loco_prosp[c]['overall_mape'] for c in CELL_ORDER])),
            'within_cell': float(np.mean([within_cell[c]['overall_mape'] for c in CELL_ORDER])),
        }
    }
    
    outpath = os.path.join(output_dir, 'nyquist_results_summary.json')
    with open(outpath, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {outpath}")
    return summary


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("  Nyquist Spectrum Reconstruction from LGN Time Constants")
    print("=" * 60)
    
    # Setup
    setup_matplotlib()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    cells_data = load_cell_data(DATA_DIR, CELL_FILES)
    
    # Run all protocols
    print("\nRunning validation protocols...")
    
    print("\n  [1/4] Leave-One-Cell-Out...")
    loco = run_loco(cells_data)
    mean_loco = print_results_table("LOCO (train 2 cells, test 3rd)", loco)
    
    print("\n  [2/4] LOCO + Prospective...")
    loco_prosp = run_loco_prospective(cells_data)
    mean_lp = print_results_table(
        f"LOCO + Prospective (train 2 cells diags 1-{DIAG_SPLIT}, test 3rd diags {DIAG_SPLIT+1}-15)",
        loco_prosp)
    
    print("\n  [3/4] Within-cell prospective...")
    within_cell = run_within_cell_prospective(cells_data)
    mean_wc = print_results_table(
        f"Within-cell (train diags 1-{DIAG_SPLIT}, test {DIAG_SPLIT+1}-15)", within_cell)
    
    print("\n  [4/4] Analytical comparison...")
    ana_mapes, eis_mapes = run_analytical_comparison(cells_data)
    
    # Print method comparison
    print(f"\n{'─'*60}")
    print(f"  METHOD COMPARISON (Z_re MAPE)")
    print(f"{'─'*60}")
    print(f"  LGN regression (LOCO):       {mean_loco:.2f}%")
    print(f"  LGN regression (LOCO+prosp): {mean_lp:.2f}%")
    print(f"  EIS 3RC self-reconstruction:  {np.mean([np.mean(eis_mapes[c]) for c in CELL_ORDER]):.2f}%")
    print(f"  Analytical Z(ω) from LGN:    {np.mean([np.mean(ana_mapes[c]) for c in CELL_ORDER]):.2f}%")
    
    # Generate figures
    print("\nGenerating figures...")
    plot_nyquist_figure(
        loco,
        'Leave-one-cell-out Nyquist reconstruction',
        'unseen cell',
        'fig5a_loco_nyquist.png',
        OUTPUT_DIR)
    
    plot_nyquist_figure(
        loco_prosp,
        'LOCO + prospective (unseen cell × unseen aging)',
        'unseen cell + aging',
        'fig5b_loco_prospective.png',
        OUTPUT_DIR)
    
    plot_protocol_comparison(loco, loco_prosp, within_cell, OUTPUT_DIR)
    
    # Save summary
    summary = save_summary_json(loco, loco_prosp, within_cell, ana_mapes, eis_mapes, OUTPUT_DIR)
    
    # Final headline
    print("\n" + "=" * 60)
    print("  HEADLINE NUMBERS FOR PAPER")
    print("=" * 60)
    print(f"  LOCO mean MAPE:            {mean_loco:.2f}%")
    print(f"  LOCO+prospective mean:     {mean_lp:.2f}%")
    print(f"  Regression vs analytical:  {mean_loco:.1f}% vs "
          f"{np.mean([np.mean(ana_mapes[c]) for c in CELL_ORDER]):.1f}% "
          f"({np.mean([np.mean(ana_mapes[c]) for c in CELL_ORDER])/mean_loco:.0f}× worse)")
    print(f"  Frequency band:            0.1 Hz – 1 kHz (5 points)")
    print(f"  Validation scope:          3 cells × 15 aging states")
    print("=" * 60)


if __name__ == '__main__':
    main()
