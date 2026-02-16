"""
Nyquist Spectrum Reconstruction — Complete Analysis & Visualization
====================================================================
Generates ALL figures demonstrating that LGN time constants from a 30-second
HPPC pulse are a sufficient statistic for the full impedance fingerprint.

Required files in DATA_DIR:
  - results_3d_W8_SOC50.json
  - results_3d_W9_SOC50.json  
  - results_3d_W10_Warmstart.json

Outputs 10 publication-quality figures + summary JSON.

Author: Shafayeth Jamil (USC ECE), February 2026
"""

import json, os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from sklearn.linear_model import Ridge
from scipy.optimize import nnls
from scipy import stats

# ============================================================================
# CONFIG
# ============================================================================
DATA_DIR = '.'
OUTPUT_DIR = './nyquist_figures'

CELL_FILES = {
    'W8':  'results_3d_W8_SOC50.json',
    'W9':  'results_3d_W9_SOC50.json',
    'W10': 'results_3d_W10_Warmstart.json',
}

FREQ_KEYS_RE = ['Z_1kHz_re', 'Z_100Hz_re', 'Z_10Hz_re', 'Z_1Hz_re', 'Z_01Hz_re']
FREQ_KEYS_IM = ['Z_1kHz_im', 'Z_100Hz_im', 'Z_10Hz_im', 'Z_1Hz_im', 'Z_01Hz_im']
FREQ_HZ = [1000, 100, 10, 1, 0.1]
FREQ_LABELS = ['1 kHz', '100 Hz', '10 Hz', '1 Hz', '0.1 Hz']
FREQ_SHORT = ['1kHz', '100Hz', '10Hz', '1Hz', '0.1Hz']
OMEGAS = [2 * np.pi * f for f in FREQ_HZ]

CELLS = ['W8', 'W9', 'W10']
CC = {'W8': '#264653', 'W9': '#E76F51', 'W10': '#2A9D8F'}
CC_LIGHT = {'W8': '#5C8A97', 'W9': '#F0A890', 'W10': '#6DC4B8'}

ALPHA = 1e-4
DIAG_SPLIT = 10

DPI = 250

# ============================================================================
# SETUP
# ============================================================================
def setup():
    matplotlib.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10.5,
        'axes.linewidth': 0.8, 'axes.labelsize': 12,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    })
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    cells = {}
    for cell, fname in CELL_FILES.items():
        fpath = os.path.join(DATA_DIR, fname)
        with open(fpath) as f:
            cells[cell] = json.load(f)
        print(f"  Loaded {cell}: {len(cells[cell])} diagnostics")
    return cells

# ============================================================================
# CORE FUNCTIONS
# ============================================================================
def get_xy(data_list):
    X = np.array([d['tau_full'] for d in data_list])
    Y_re = np.array([[d[k] for k in FREQ_KEYS_RE] for d in data_list])
    Y_im = np.array([[d[k] for k in FREQ_KEYS_IM] for d in data_list])
    diags = [d['diag'] for d in data_list]
    return X, Y_re, Y_im, diags

def fit_predict(X_tr, Y_tr, X_te):
    Y_pred = np.zeros((X_te.shape[0], Y_tr.shape[1]))
    for fi in range(Y_tr.shape[1]):
        reg = Ridge(alpha=ALPHA).fit(X_tr, Y_tr[:, fi])
        Y_pred[:, fi] = reg.predict(X_te)
    return Y_pred

def mape_overall(Y_true, Y_pred):
    return np.mean(np.abs(Y_true - Y_pred) / (np.abs(Y_true) + 1e-15)) * 100

def mape_per_freq(Y_true, Y_pred):
    return np.mean(np.abs(Y_true - Y_pred) / (np.abs(Y_true) + 1e-15), axis=0) * 100

def mape_per_sample(Y_true, Y_pred):
    return np.mean(np.abs(Y_true - Y_pred) / (np.abs(Y_true) + 1e-15), axis=1) * 100

def z_3rc(Rs, R, taus, omegas):
    z = Rs * np.ones(len(omegas), dtype=complex)
    for i in range(len(taus)):
        z += R[i] / (1 + 1j * np.array(omegas) * taus[i])
    return z

# ============================================================================
# ALL VALIDATION PROTOCOLS
# ============================================================================
def run_all_protocols(CD):
    """Run every protocol and return structured results."""
    results = {}
    all_diags = sorted(set(d['diag'] for d in CD['W8']))
    
    # --- 1. Cell-LOOCV ---
    cell_loco = {}
    for tc in CELLS:
        others = [c for c in CELLS if c != tc]
        X_tr, Y_re_tr, Y_im_tr = [], [], []
        for c in others:
            x, yr, yi, _ = get_xy(CD[c])
            X_tr.append(x); Y_re_tr.append(yr); Y_im_tr.append(yi)
        X_tr = np.vstack(X_tr); Y_re_tr = np.vstack(Y_re_tr); Y_im_tr = np.vstack(Y_im_tr)
        X_te, Y_re_te, Y_im_te, diags = get_xy(CD[tc])
        Yr_p = fit_predict(X_tr, Y_re_tr, X_te)
        Yi_p = fit_predict(X_tr, Y_im_tr, X_te)
        cell_loco[tc] = {
            'train_cells': others, 'n_train': len(X_tr), 'n_test': len(X_te),
            'Y_re_test': Y_re_te, 'Y_im_test': Y_im_te,
            'Y_re_pred': Yr_p, 'Y_im_pred': Yi_p, 'diags': diags,
            'mape_per_freq': mape_per_freq(Y_re_te, Yr_p),
            'mape_per_sample': mape_per_sample(Y_re_te, Yr_p),
            'overall_mape': mape_overall(Y_re_te, Yr_p),
        }
    results['cell_loco'] = cell_loco
    
    # --- 2. Aging-LOOCV (pooled across cells) ---
    aging_loco = {}
    for held_diag in all_diags:
        train = [d for c in CELLS for d in CD[c] if d['diag'] != held_diag]
        test = [d for c in CELLS for d in CD[c] if d['diag'] == held_diag]
        X_tr, Y_tr, _, _ = get_xy(train)
        X_te, Y_te, _, _ = get_xy(test)
        Y_p = fit_predict(X_tr, Y_tr, X_te)
        aging_loco[held_diag] = {
            'mape': mape_overall(Y_te, Y_p),
            'mape_per_freq': mape_per_freq(Y_te, Y_p),
            'n_test': len(test),
        }
    results['aging_loco'] = aging_loco
    
    # --- 3. Cell + Aging holdout (45 folds) ---
    ca_matrix = np.zeros((3, 15))
    ca_freq_matrix = np.zeros((3, 15, 5))
    ca_details = {}
    for ci, tc in enumerate(CELLS):
        others = [c for c in CELLS if c != tc]
        ca_details[tc] = {}
        for di, hd in enumerate(all_diags):
            train = [d for c in others for d in CD[c] if d['diag'] != hd]
            test = [d for d in CD[tc] if d['diag'] == hd]
            X_tr, Y_tr, _, _ = get_xy(train)
            X_te, Y_te, Yi_te, _ = get_xy(test)
            Y_p = fit_predict(X_tr, Y_tr, X_te)
            Yi_p = fit_predict(X_tr, np.array([[d[k] for k in FREQ_KEYS_IM] for d in train]), X_te)
            ca_matrix[ci, di] = mape_overall(Y_te, Y_p)
            ca_freq_matrix[ci, di] = mape_per_freq(Y_te, Y_p)
            ca_details[tc][hd] = {
                'mape': ca_matrix[ci, di],
                'Y_re_test': Y_te, 'Y_im_test': Yi_te,
                'Y_re_pred': Y_p, 'Y_im_pred': Yi_p,
            }
    results['cell_aging'] = {
        'matrix': ca_matrix, 'freq_matrix': ca_freq_matrix,
        'details': ca_details,
        'overall': float(np.mean(ca_matrix)),
        'std': float(np.std(ca_matrix)),
    }
    
    # --- 4. LOCO + Prospective ---
    loco_prosp = {}
    for tc in CELLS:
        others = [c for c in CELLS if c != tc]
        train = [d for c in others for d in CD[c] if d['diag'] <= DIAG_SPLIT]
        test = [d for d in CD[tc] if d['diag'] > DIAG_SPLIT]
        X_tr, Y_tr, Yi_tr, _ = get_xy(train)
        X_te, Y_te, Yi_te, diags = get_xy(test)
        Yr_p = fit_predict(X_tr, Y_tr, X_te)
        Yi_p = fit_predict(X_tr, Yi_tr, X_te)
        loco_prosp[tc] = {
            'train_cells': others, 'n_train': len(train), 'n_test': len(test),
            'Y_re_test': Y_te, 'Y_im_test': Yi_te,
            'Y_re_pred': Yr_p, 'Y_im_pred': Yi_p, 'diags': diags,
            'mape_per_freq': mape_per_freq(Y_te, Yr_p),
            'overall_mape': mape_overall(Y_te, Yr_p),
        }
    results['loco_prosp'] = loco_prosp
    
    # --- 5. Within-cell prospective ---
    within = {}
    for cell in CELLS:
        train = [d for d in CD[cell] if d['diag'] <= DIAG_SPLIT]
        test = [d for d in CD[cell] if d['diag'] > DIAG_SPLIT]
        X_tr, Y_tr, Yi_tr, _ = get_xy(train)
        X_te, Y_te, Yi_te, diags = get_xy(test)
        Yr_p = fit_predict(X_tr, Y_tr, X_te)
        Yi_p = fit_predict(X_tr, Yi_tr, X_te)
        within[cell] = {
            'n_train': len(train), 'n_test': len(test),
            'Y_re_test': Y_te, 'Y_im_test': Yi_te,
            'Y_re_pred': Yr_p, 'Y_im_pred': Yi_p, 'diags': diags,
            'mape_per_freq': mape_per_freq(Y_te, Yr_p),
            'overall_mape': mape_overall(Y_te, Yr_p),
        }
    results['within'] = within
    
    # --- 6. Analytical vs EIS self-reconstruction ---
    ana_all, eis_all = {c: [] for c in CELLS}, {c: [] for c in CELLS}
    for cell in CELLS:
        for d in CD[cell]:
            taus = d['tau_full']
            z_true = np.array([complex(d[kr], d[ki]) for kr, ki in zip(FREQ_KEYS_RE, FREQ_KEYS_IM)])
            # Analytical
            A_mat = np.zeros((10, 4)); b_vec = np.zeros(10)
            for fi, omega in enumerate(OMEGAS):
                A_mat[2*fi, 0] = 1.0
                for ti in range(3):
                    h = 1.0 / (1 + 1j * omega * taus[ti])
                    A_mat[2*fi, ti+1] = h.real; A_mat[2*fi+1, ti+1] = h.imag
                b_vec[2*fi] = z_true[fi].real; b_vec[2*fi+1] = z_true[fi].imag
            x_fit, _ = nnls(A_mat, b_vec)
            z_ana = z_3rc(x_fit[0], x_fit[1:], taus, OMEGAS)
            ana_all[cell].append(np.mean(np.abs(z_true.real - z_ana.real) / np.abs(z_true.real)) * 100)
            # EIS self
            z_eis = z_3rc(d['Rs_eis_3rc'], [d['R1_eis_3rc'], d['R2_eis_3rc'], d['R3_eis_3rc']],
                          d['tau_eis_3rc'], OMEGAS)
            eis_all[cell].append(np.mean(np.abs(z_true.real - z_eis.real) / np.abs(z_true.real)) * 100)
    results['analytical'] = ana_all
    results['eis_self'] = eis_all
    
    return results

# ============================================================================
# FIGURE 1: Cell-LOOCV Nyquist (MAIN PAPER)
# ============================================================================
def fig01_loco_nyquist(R):
    r = R['cell_loco']
    fig = plt.figure(figsize=(17, 5.8))
    gs = fig.add_gridspec(1, 5, width_ratios=[1,1,1,0.05,0.7], wspace=0.28,
                           left=0.05, right=0.97, top=0.88, bottom=0.12)
    for pi, cell in enumerate(CELLS):
        ax = fig.add_subplot(gs[0, pi])
        d = r[cell]
        n = d['Y_re_test'].shape[0]
        for di in range(n):
            a = 0.12 + 0.88*(di/max(n-1,1))
            ax.plot(d['Y_re_test'][di]*1e3, -d['Y_im_test'][di]*1e3, 'o-',
                    color='k', ms=4.5, lw=1.1, alpha=a, mfc='k', zorder=3)
            ax.plot(d['Y_re_pred'][di]*1e3, -d['Y_im_pred'][di]*1e3, 's--',
                    color=CC[cell], ms=4.5, lw=1.1, alpha=a, mfc='none', mec=CC[cell], zorder=4)
        ax.text(0.95, 0.95, f'{d["overall_mape"]:.1f}%', transform=ax.transAxes,
                fontsize=17, fontweight='bold', ha='right', va='top', color=CC[cell],
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=CC[cell], alpha=0.92, lw=1.5))
        ax.text(0.05, 0.95, f'trained on\n{d["train_cells"][0]}, {d["train_cells"][1]}',
                transform=ax.transAxes, fontsize=8, color='#888', va='top', fontstyle='italic')
        ax.set_title(f'Test: {cell}', fontsize=13, fontweight='bold', pad=8)
        ax.set_xlabel('$Z_{re}$ (mΩ)'); 
        if pi==0: ax.set_ylabel('$-Z_{im}$ (mΩ)', fontsize=12)
        ax.set_xlim(23,31.5); ax.set_ylim(-3.8,3.2); ax.grid(True, alpha=0.06)
        ax.text(-0.06,1.07, chr(ord('a')+pi), transform=ax.transAxes, fontsize=15, fontweight='bold')
    fig.axes[0].legend(handles=[
        Line2D([0],[0],marker='o',color='k',lw=1.1,ms=5.5,mfc='k',label='EIS (measured)'),
        Line2D([0],[0],marker='s',color='#777',lw=1.1,ms=5.5,mfc='none',ls='--',label='LGN (unseen cell)'),
    ], loc='lower right', fontsize=9.5, framealpha=0.95, edgecolor='#ccc')
    # Bar
    ax_b = fig.add_subplot(gs[0,4])
    avg = np.mean([r[c]['mape_per_freq'] for c in CELLS], axis=0)
    std = np.std([r[c]['mape_per_freq'] for c in CELLS], axis=0)
    x = np.arange(5)
    cbar = [plt.cm.RdYlGn_r(0.15+0.55*i/4) for i in range(5)]
    ax_b.barh(x, avg, 0.52, xerr=std, capsize=3, color=cbar, edgecolor='white',
              error_kw={'lw':0.7,'color':'#555','capthick':0.7})
    for j,(m,s) in enumerate(zip(avg,std)):
        ax_b.text(m+s+0.05, j, f'{m:.2f}%', va='center', fontsize=9.5, fontweight='bold', color='#333')
    ax_b.set_yticks(x); ax_b.set_yticklabels(FREQ_LABELS); ax_b.set_xlabel('MAPE (%)')
    ax_b.set_xlim(0,2.0); ax_b.invert_yaxis(); ax_b.grid(True,axis='x',alpha=0.08)
    ax_b.axvline(1.0,color='#aaa',ls=':',lw=0.7)
    ax_b.set_title('Error by\nfrequency', fontsize=12, fontweight='bold', pad=8)
    ax_b.text(-0.15,1.07,'d',transform=ax_b.transAxes,fontsize=15,fontweight='bold')
    mn = np.mean([r[c]['overall_mape'] for c in CELLS])
    fig.text(0.42,0.01, f'Leave-one-cell-out: mean MAPE = {mn:.2f}% (0.1 Hz – 1 kHz, 3 cells × 15 aging states)',
             ha='center', fontsize=10.5, fontstyle='italic', color='#444')
    fig.savefig(f'{OUTPUT_DIR}/fig01_loco_nyquist.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  [1/10] fig01_loco_nyquist.png — Cell-LOOCV mean {mn:.2f}%")

# ============================================================================
# FIGURE 2: LOCO + Prospective Nyquist
# ============================================================================
def fig02_loco_prospective(R):
    r = R['loco_prosp']
    fig = plt.figure(figsize=(17, 5.8))
    gs = fig.add_gridspec(1, 5, width_ratios=[1,1,1,0.05,0.7], wspace=0.28,
                           left=0.05, right=0.97, top=0.88, bottom=0.12)
    for pi, cell in enumerate(CELLS):
        ax = fig.add_subplot(gs[0, pi])
        d = r[cell]; n = d['Y_re_test'].shape[0]
        for di in range(n):
            a = 0.15 + 0.85*(di/max(n-1,1))
            ax.plot(d['Y_re_test'][di]*1e3, -d['Y_im_test'][di]*1e3, 'o-',
                    color='k', ms=5, lw=1.2, alpha=a, mfc='k', zorder=3)
            ax.plot(d['Y_re_pred'][di]*1e3, -d['Y_im_pred'][di]*1e3, 's--',
                    color=CC[cell], ms=5, lw=1.2, alpha=a, mfc='none', mec=CC[cell], zorder=4)
        ax.text(0.95, 0.95, f'{d["overall_mape"]:.1f}%', transform=ax.transAxes,
                fontsize=17, fontweight='bold', ha='right', va='top', color=CC[cell],
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=CC[cell], alpha=0.92, lw=1.5))
        ax.text(0.05,0.95, f'trained on\n{d["train_cells"][0]}, {d["train_cells"][1]}\n(diags 1–{DIAG_SPLIT} only)',
                transform=ax.transAxes, fontsize=8, color='#888', va='top', fontstyle='italic')
        ax.set_title(f'Test: {cell} (diags {DIAG_SPLIT+1}–15)', fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel('$Z_{re}$ (mΩ)')
        if pi==0: ax.set_ylabel('$-Z_{im}$ (mΩ)', fontsize=12)
        ax.set_xlim(23,31.5); ax.set_ylim(-3.8,3.2); ax.grid(True, alpha=0.06)
        ax.text(-0.06,1.07, chr(ord('a')+pi), transform=ax.transAxes, fontsize=15, fontweight='bold')
    fig.axes[0].legend(handles=[
        Line2D([0],[0],marker='o',color='k',lw=1.2,ms=5.5,mfc='k',label='EIS (measured)'),
        Line2D([0],[0],marker='s',color='#777',lw=1.2,ms=5.5,mfc='none',ls='--',label='LGN (unseen cell + aging)'),
    ], loc='lower right', fontsize=9.5, framealpha=0.95, edgecolor='#ccc')
    ax_b = fig.add_subplot(gs[0,4])
    avg = np.mean([r[c]['mape_per_freq'] for c in CELLS], axis=0)
    std = np.std([r[c]['mape_per_freq'] for c in CELLS], axis=0)
    x = np.arange(5)
    cbar = [plt.cm.RdYlGn_r(0.15+0.55*i/4) for i in range(5)]
    ax_b.barh(x, avg, 0.52, xerr=std, capsize=3, color=cbar, edgecolor='white',
              error_kw={'lw':0.7,'color':'#555','capthick':0.7})
    for j,(m,s) in enumerate(zip(avg,std)):
        ax_b.text(m+s+0.05, j, f'{m:.2f}%', va='center', fontsize=9.5, fontweight='bold', color='#333')
    ax_b.set_yticks(x); ax_b.set_yticklabels(FREQ_LABELS); ax_b.set_xlabel('MAPE (%)')
    ax_b.set_xlim(0,2.8); ax_b.invert_yaxis(); ax_b.grid(True,axis='x',alpha=0.08)
    ax_b.axvline(1.0,color='#aaa',ls=':',lw=0.7)
    ax_b.set_title('Error by\nfrequency', fontsize=12, fontweight='bold', pad=8)
    ax_b.text(-0.15,1.07,'d',transform=ax_b.transAxes,fontsize=15,fontweight='bold')
    mn = np.mean([r[c]['overall_mape'] for c in CELLS])
    fig.text(0.42,0.01, f'LOCO + prospective: mean MAPE = {mn:.2f}% (unseen cell × unseen aging)',
             ha='center', fontsize=10.5, fontstyle='italic', color='#444')
    fig.savefig(f'{OUTPUT_DIR}/fig02_loco_prospective.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  [2/10] fig02_loco_prospective.png — LOCO+prosp mean {mn:.2f}%")

# ============================================================================
# FIGURE 3: Protocol Comparison Bar Chart
# ============================================================================
def fig03_protocol_comparison(R):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    protocols = ['Cell-LOOCV', 'Aging-LOOCV\n(pooled)', 'Cell + Aging\nholdout',
                 'LOCO +\nprospective', 'Within-cell\nprospective']
    x = np.arange(5)
    
    # Compute per-cell values
    aging_per_cell = {}
    for c in CELLS:
        # aging LOOCV per cell: mean over all held-out diags
        am = []
        for hd, v in R['aging_loco'].items():
            # Need per-cell... use cell+aging matrix row instead
            pass
        aging_per_cell[c] = np.mean(R['cell_aging']['matrix'][CELLS.index(c)])
    
    w = 0.17
    for i, cell in enumerate(CELLS):
        vals = [
            R['cell_loco'][cell]['overall_mape'],
            np.mean([R['aging_loco'][hd]['mape'] for hd in R['aging_loco']]),  # same for all cells in pooled
            np.mean(R['cell_aging']['matrix'][i]),
            R['loco_prosp'][cell]['overall_mape'],
            R['within'][cell]['overall_mape'],
        ]
        bars = ax.bar(x + (i-1)*w, vals, w, color=CC[cell], alpha=0.85,
                      edgecolor='white', lw=0.8, label=cell)
        for j, v in enumerate(vals):
            ax.text(x[j]+(i-1)*w, v+0.06, f'{v:.1f}', ha='center', fontsize=8,
                   color=CC[cell], fontweight='bold')
    
    # Mean lines
    means = [
        np.mean([R['cell_loco'][c]['overall_mape'] for c in CELLS]),
        np.mean([R['aging_loco'][hd]['mape'] for hd in R['aging_loco']]),
        R['cell_aging']['overall'],
        np.mean([R['loco_prosp'][c]['overall_mape'] for c in CELLS]),
        np.mean([R['within'][c]['overall_mape'] for c in CELLS]),
    ]
    for j, m in enumerate(means):
        ax.plot([x[j]-0.32, x[j]+0.32], [m,m], 'k-', lw=1.8, alpha=0.5)
        ax.text(x[j]+0.35, m, f'μ={m:.2f}%', fontsize=9, fontweight='bold', color='#333', va='center')
    
    ax.set_xticks(x); ax.set_xticklabels(protocols, fontsize=10)
    ax.set_ylabel('$Z_{re}$ MAPE (%)', fontsize=12)
    ax.set_title('Nyquist reconstruction error across all validation protocols',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10, framealpha=0.95, loc='upper right')
    ax.grid(True, axis='y', alpha=0.08); ax.set_ylim(0, 4.5)
    ax.axhline(2.0, color='#999', ls='--', lw=0.8, alpha=0.5)
    ax.text(4.6, 2.08, '2%', fontsize=9, color='#888')
    
    fig.savefig(f'{OUTPUT_DIR}/fig03_protocol_comparison.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  [3/10] fig03_protocol_comparison.png")

# ============================================================================
# FIGURE 4: Cell+Aging Holdout Heatmap (45 folds)
# ============================================================================
def fig04_heatmap(R):
    mat = R['cell_aging']['matrix']
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=5)
    
    for ci in range(3):
        for di in range(15):
            v = mat[ci, di]
            color = 'white' if v > 3 else 'black'
            ax.text(di, ci, f'{v:.1f}', ha='center', va='center', fontsize=8.5,
                   fontweight='bold', color=color)
    
    ax.set_yticks(range(3)); ax.set_yticklabels(CELLS, fontsize=12, fontweight='bold')
    ax.set_xticks(range(15)); ax.set_xticklabels([f'{int(d)}' for d in range(1,16)], fontsize=9)
    ax.set_xlabel('Diagnostic number (aging state)', fontsize=12)
    ax.set_title(f'Cell + Aging holdout MAPE (%) — 45 double-blind folds\n'
                 f'Mean = {R["cell_aging"]["overall"]:.2f}% ± {R["cell_aging"]["std"]:.2f}%',
                 fontsize=13, fontweight='bold', pad=10)
    
    # Divider at diag 10 (prospective boundary)
    ax.axvline(9.5, color='white', ls='--', lw=2, alpha=0.8)
    ax.text(9.5, -0.7, '← early aging | late aging →', ha='center', fontsize=9,
           fontstyle='italic', color='#666')
    
    cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label('MAPE (%)', fontsize=11)
    
    fig.savefig(f'{OUTPUT_DIR}/fig04_cell_aging_heatmap.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  [4/10] fig04_cell_aging_heatmap.png — 45-fold heatmap")

# ============================================================================
# FIGURE 5: Parity Plots (Z_re predicted vs measured) for Cell-LOOCV
# ============================================================================
def fig05_parity(R):
    r = R['cell_loco']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'wspace': 0.25})
    
    for pi, cell in enumerate(CELLS):
        ax = axes[pi]
        d = r[cell]
        for fi in range(5):
            true = d['Y_re_test'][:, fi] * 1e3
            pred = d['Y_re_pred'][:, fi] * 1e3
            ax.scatter(true, pred, s=40, color=CC[cell], alpha=0.6, edgecolor='white',
                      lw=0.5, zorder=3, label=FREQ_LABELS[fi] if pi==0 else None)
        
        # Perfect line
        all_true = d['Y_re_test'].flatten() * 1e3
        all_pred = d['Y_re_pred'].flatten() * 1e3
        mn, mx = min(all_true.min(), all_pred.min()), max(all_true.max(), all_pred.max())
        pad = (mx - mn) * 0.05
        ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad], 'k--', lw=1, alpha=0.5, zorder=1)
        
        # ±1% band
        ax.fill_between([mn-pad, mx+pad],
                        [mn-pad - 0.01*(mx+pad), mx+pad - 0.01*(mx+pad)],
                        [mn-pad + 0.01*(mx+pad), mx+pad + 0.01*(mx+pad)],
                        alpha=0.08, color='green', zorder=0)
        
        # Stats
        rho, _ = stats.pearsonr(all_true, all_pred)
        ax.text(0.05, 0.95, f'ρ = {rho:.4f}\nMAPE = {d["overall_mape"]:.2f}%',
                transform=ax.transAxes, fontsize=10, fontweight='bold', va='top',
                bbox=dict(boxstyle='round', fc='white', alpha=0.9, ec='#ccc'))
        
        ax.set_xlabel('$Z_{re}$ measured (mΩ)', fontsize=11)
        if pi==0: ax.set_ylabel('$Z_{re}$ predicted (mΩ)', fontsize=12)
        ax.set_title(f'Test: {cell}', fontsize=12, fontweight='bold')
        ax.set_xlim(mn-pad, mx+pad); ax.set_ylim(mn-pad, mx+pad)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.06)
        ax.text(-0.06, 1.06, chr(ord('a')+pi), transform=ax.transAxes, fontsize=15, fontweight='bold')
    
    axes[0].legend(fontsize=9, loc='lower right', framealpha=0.95, title='Frequency', title_fontsize=9)
    fig.suptitle('Parity plots: Cell-LOOCV Nyquist reconstruction', fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(f'{OUTPUT_DIR}/fig05_parity_plots.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  [5/10] fig05_parity_plots.png")

# ============================================================================
# FIGURE 6: Aging-LOOCV Error vs Diagnostic Number
# ============================================================================
def fig06_aging_trajectory(R):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    diags_sorted = sorted(R['aging_loco'].keys())
    mapes = [R['aging_loco'][d]['mape'] for d in diags_sorted]
    diag_nums = [int(d) for d in diags_sorted]
    
    ax.bar(diag_nums, mapes, 0.6, color=[plt.cm.RdYlGn_r(0.15+0.6*m/max(mapes)) for m in mapes],
           edgecolor='white', lw=0.8)
    
    for d, m in zip(diag_nums, mapes):
        ax.text(d, m+0.05, f'{m:.2f}', ha='center', fontsize=8.5, fontweight='bold', color='#333')
    
    ax.axhline(np.mean(mapes), color='k', ls='--', lw=1.2, alpha=0.5)
    ax.text(15.5, np.mean(mapes)+0.05, f'μ = {np.mean(mapes):.2f}%', fontsize=10,
           fontweight='bold', color='#333')
    
    ax.axvline(DIAG_SPLIT + 0.5, color='#999', ls=':', lw=1, alpha=0.5)
    ax.text(DIAG_SPLIT + 0.7, max(mapes)*0.95, 'prospective\nboundary', fontsize=9,
           color='#888', fontstyle='italic')
    
    ax.set_xlabel('Held-out diagnostic number', fontsize=12)
    ax.set_ylabel('MAPE (%)', fontsize=12)
    ax.set_title('Aging-LOOCV: reconstruction error per held-out diagnostic\n'
                 '(pooled across 3 cells, trained on remaining 14 diagnostics × 3 cells)',
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(diag_nums)
    ax.set_ylim(0, max(mapes)*1.25)
    ax.grid(True, axis='y', alpha=0.08)
    
    fig.savefig(f'{OUTPUT_DIR}/fig06_aging_loocv.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  [6/10] fig06_aging_loocv.png")

# ============================================================================
# FIGURE 7: Analytical vs Regression vs EIS — Method Comparison
# ============================================================================
def fig07_method_comparison(R):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1, 0.7], 'wspace': 0.3})
    
    # Panel a: per-cell grouped bars
    methods = ['LGN regression\n(Cell-LOOCV)', 'EIS 3RC\nself-fit', 'Analytical\n$Z(\\omega)$']
    x = np.arange(3)
    w = 0.22
    
    for i, cell in enumerate(CELLS):
        vals = [
            R['cell_loco'][cell]['overall_mape'],
            np.mean(R['eis_self'][cell]),
            np.mean(R['analytical'][cell]),
        ]
        ax1.bar(x + (i-1)*w, vals, w, color=CC[cell], alpha=0.85, edgecolor='white', label=cell)
        for j, v in enumerate(vals):
            ax1.text(x[j]+(i-1)*w, v+0.1, f'{v:.1f}', ha='center', fontsize=9,
                    color=CC[cell], fontweight='bold')
    
    ax1.set_xticks(x); ax1.set_xticklabels(methods, fontsize=10.5)
    ax1.set_ylabel('$Z_{re}$ MAPE (%)', fontsize=12)
    ax1.set_title('Why regression, not analytical $Z(\\omega)$', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10); ax1.grid(True, axis='y', alpha=0.08); ax1.set_ylim(0, 8)
    ax1.text(-0.06, 1.06, 'a', transform=ax1.transAxes, fontsize=15, fontweight='bold')
    
    # Panel b: summary — overall means
    ax2_methods = ['LGN\nregression', 'EIS 3RC\nself-fit', 'Analytical\n$Z(\\omega)$']
    overall = [
        np.mean([R['cell_loco'][c]['overall_mape'] for c in CELLS]),
        np.mean([np.mean(R['eis_self'][c]) for c in CELLS]),
        np.mean([np.mean(R['analytical'][c]) for c in CELLS]),
    ]
    colors = ['#2A9D8F', '#E9C46A', '#C44E52']
    bars = ax2.bar(range(3), overall, 0.55, color=colors, edgecolor='white', lw=0.8)
    for i, (b, m) in enumerate(zip(bars, overall)):
        ax2.text(b.get_x()+b.get_width()/2, m+0.15, f'{m:.1f}%', ha='center',
                fontsize=13, fontweight='bold', color=colors[i])
    
    ax2.set_xticks(range(3)); ax2.set_xticklabels(ax2_methods, fontsize=10)
    ax2.set_ylabel('Overall MAPE (%)', fontsize=12)
    ax2.set_title('Method comparison', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 8); ax2.grid(True, axis='y', alpha=0.08)
    ax2.text(-0.08, 1.06, 'b', transform=ax2.transAxes, fontsize=15, fontweight='bold')
    
    # Arrow annotation
    ax2.annotate(f'{overall[2]/overall[0]:.0f}× worse', xy=(2, overall[2]-0.3),
                xytext=(0.5, overall[2]+1), fontsize=11, fontstyle='italic', color='#C44E52',
                arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.5), ha='center', fontweight='bold')
    
    fig.savefig(f'{OUTPUT_DIR}/fig07_method_comparison.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  [7/10] fig07_method_comparison.png")

# ============================================================================
# FIGURE 8: Error Distribution Histograms (all 45 cell+aging folds)
# ============================================================================
def fig08_error_distribution(R):
    mat = R['cell_aging']['matrix']
    all_mapes = mat.flatten()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'wspace': 0.3})
    
    # Panel a: histogram
    bins = np.arange(0, 5.5, 0.5)
    n, _, patches = ax1.hist(all_mapes, bins=bins, color='#2A9D8F', edgecolor='white',
                              lw=0.8, alpha=0.85)
    for p, left in zip(patches, bins):
        if left >= 2.0:
            p.set_facecolor('#E76F51')
            p.set_alpha(0.7)
    
    ax1.axvline(np.mean(all_mapes), color='k', ls='--', lw=1.5, label=f'Mean = {np.mean(all_mapes):.2f}%')
    ax1.axvline(np.median(all_mapes), color='#264653', ls=':', lw=1.5, label=f'Median = {np.median(all_mapes):.2f}%')
    ax1.axvline(2.0, color='#C44E52', ls='--', lw=1, alpha=0.7, label='2% threshold')
    
    ax1.set_xlabel('MAPE (%)', fontsize=12)
    ax1.set_ylabel('Count (of 45 folds)', fontsize=12)
    ax1.set_title('Error distribution: Cell + Aging holdout\n(45 double-blind folds)',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, framealpha=0.95)
    ax1.text(0.95, 0.95, f'{np.sum(all_mapes < 2)/len(all_mapes)*100:.0f}% < 2%\n'
             f'{np.sum(all_mapes < 1)/len(all_mapes)*100:.0f}% < 1%',
             transform=ax1.transAxes, fontsize=11, ha='right', va='top', fontweight='bold',
             bbox=dict(boxstyle='round', fc='white', ec='#ccc', alpha=0.9))
    ax1.text(-0.06, 1.06, 'a', transform=ax1.transAxes, fontsize=15, fontweight='bold')
    
    # Panel b: CDF
    sorted_m = np.sort(all_mapes)
    cdf = np.arange(1, len(sorted_m)+1) / len(sorted_m)
    ax2.plot(sorted_m, cdf*100, '-', color='#264653', lw=2.5)
    ax2.fill_between(sorted_m, 0, cdf*100, alpha=0.1, color='#264653')
    
    # Mark key thresholds
    for thresh, col in [(1.0, '#2A9D8F'), (2.0, '#E9C46A'), (3.0, '#E76F51')]:
        pct = np.sum(all_mapes <= thresh) / len(all_mapes) * 100
        ax2.axvline(thresh, color=col, ls='--', lw=1, alpha=0.7)
        ax2.plot(thresh, pct, 'o', color=col, ms=8, zorder=5)
        ax2.text(thresh+0.1, pct-5, f'{pct:.0f}% ≤ {thresh}%', fontsize=9.5,
                color=col, fontweight='bold')
    
    ax2.set_xlabel('MAPE (%)', fontsize=12)
    ax2.set_ylabel('Cumulative % of folds', fontsize=12)
    ax2.set_title('Cumulative distribution', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 5); ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.08)
    ax2.text(-0.06, 1.06, 'b', transform=ax2.transAxes, fontsize=15, fontweight='bold')
    
    fig.savefig(f'{OUTPUT_DIR}/fig08_error_distribution.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  [8/10] fig08_error_distribution.png")

# ============================================================================
# FIGURE 9: τ Space — shows cells occupy distinct but overlapping regions
# ============================================================================
def fig09_tau_space(CD):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={'wspace': 0.3})
    pairs = [(0,1,'τ₁','τ₂'), (0,2,'τ₁','τ₃'), (1,2,'τ₂','τ₃')]
    
    for pi, (i, j, li, lj) in enumerate(pairs):
        ax = axes[pi]
        for cell in CELLS:
            taus = np.array([d['tau_full'] for d in CD[cell]])
            diags = np.array([d['diag'] for d in CD[cell]])
            
            sc = ax.scatter(taus[:, i], taus[:, j], c=diags, cmap='viridis',
                           s=60, edgecolor=CC[cell], lw=1.5, zorder=3, vmin=1, vmax=15,
                           marker='o' if cell=='W8' else ('s' if cell=='W9' else 'D'))
            
            # Connect trajectory
            ax.plot(taus[:, i], taus[:, j], '-', color=CC[cell], alpha=0.3, lw=1)
            
            # Label start and end
            ax.annotate(f'{cell}\ndiag 1', xy=(taus[0,i], taus[0,j]),
                       fontsize=7, color=CC[cell], alpha=0.7,
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel(f'{li} (s)', fontsize=12)
        ax.set_ylabel(f'{lj} (s)', fontsize=12)
        ax.set_title(f'{li} vs {lj}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.06)
        ax.text(-0.06, 1.06, chr(ord('a')+pi), transform=ax.transAxes, fontsize=15, fontweight='bold')
    
    # Colorbar
    cb = plt.colorbar(sc, ax=axes[-1], shrink=0.8, pad=0.02)
    cb.set_label('Diagnostic #', fontsize=11)
    
    # Legend
    handles = [Line2D([0],[0],marker='o',color=CC['W8'],ls='',ms=8,mec=CC['W8'],mfc='none',label='W8'),
               Line2D([0],[0],marker='s',color=CC['W9'],ls='',ms=8,mec=CC['W9'],mfc='none',label='W9'),
               Line2D([0],[0],marker='D',color=CC['W10'],ls='',ms=8,mec=CC['W10'],mfc='none',label='W10')]
    axes[0].legend(handles=handles, fontsize=10, framealpha=0.95, loc='upper right')
    
    fig.suptitle('LGN time constant trajectories through τ-space during aging',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(f'{OUTPUT_DIR}/fig09_tau_space.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  [9/10] fig09_tau_space.png")

# ============================================================================
# FIGURE 10: Per-frequency MAPE across ALL protocols (comprehensive)
# ============================================================================
def fig10_frequency_breakdown(R):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    protocols = {
        'Cell-LOOCV': np.mean([R['cell_loco'][c]['mape_per_freq'] for c in CELLS], axis=0),
        'Aging-LOOCV': np.mean([R['aging_loco'][d]['mape_per_freq'] for d in R['aging_loco']], axis=0),
        'Cell+Aging': np.mean(R['cell_aging']['freq_matrix'].reshape(-1, 5), axis=0),
        'LOCO+prosp.': np.mean([R['loco_prosp'][c]['mape_per_freq'] for c in CELLS], axis=0),
        'Within-cell': np.mean([R['within'][c]['mape_per_freq'] for c in CELLS], axis=0),
    }
    
    x = np.arange(5)
    n_prot = len(protocols)
    w = 0.14
    colors = ['#2A9D8F', '#264653', '#E9C46A', '#E76F51', '#999999']
    
    for i, (name, vals) in enumerate(protocols.items()):
        offset = (i - n_prot/2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, color=colors[i], alpha=0.85,
                      edgecolor='white', lw=0.6, label=name)
    
    ax.set_xticks(x); ax.set_xticklabels(FREQ_LABELS, fontsize=11)
    ax.set_ylabel('$Z_{re}$ MAPE (%)', fontsize=12)
    ax.set_title('Frequency-resolved reconstruction error across all protocols',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=9.5, ncol=3, loc='upper left', framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.08)
    ax.axhline(1.0, color='#aaa', ls=':', lw=0.8)
    ax.axhline(2.0, color='#ccc', ls=':', lw=0.8)
    ax.set_ylim(0, 3.5)
    
    # Annotate trend
    ax.annotate('Error increases\nat low frequencies\n(expected: slow modes\nharder from short pulse)',
               xy=(4, np.mean(list(protocols.values()), axis=0)[4]),
               xytext=(3.2, 3.0), fontsize=9, fontstyle='italic', color='#666',
               arrowprops=dict(arrowstyle='->', color='#999', lw=1.2), ha='center')
    
    fig.savefig(f'{OUTPUT_DIR}/fig10_frequency_breakdown.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(); print(f"  [10/10] fig10_frequency_breakdown.png")

# ============================================================================
# SUMMARY JSON
# ============================================================================
def save_summary(R):
    aging_mapes = [R['aging_loco'][d]['mape'] for d in R['aging_loco']]
    s = {
        'headline': {
            'cell_loocv_mape': float(np.mean([R['cell_loco'][c]['overall_mape'] for c in CELLS])),
            'aging_loocv_mape': float(np.mean(aging_mapes)),
            'aging_loocv_std': float(np.std(aging_mapes)),
            'cell_aging_holdout_mape': R['cell_aging']['overall'],
            'cell_aging_holdout_std': R['cell_aging']['std'],
            'loco_prospective_mape': float(np.mean([R['loco_prosp'][c]['overall_mape'] for c in CELLS])),
            'within_cell_mape': float(np.mean([R['within'][c]['overall_mape'] for c in CELLS])),
            'analytical_mape': float(np.mean([np.mean(R['analytical'][c]) for c in CELLS])),
            'eis_self_mape': float(np.mean([np.mean(R['eis_self'][c]) for c in CELLS])),
        },
        'cell_loocv': {c: {
            'overall_mape': float(R['cell_loco'][c]['overall_mape']),
            'per_freq': {fl: float(R['cell_loco'][c]['mape_per_freq'][i]) for i, fl in enumerate(FREQ_SHORT)},
        } for c in CELLS},
        'loco_prospective': {c: {
            'overall_mape': float(R['loco_prosp'][c]['overall_mape']),
            'per_freq': {fl: float(R['loco_prosp'][c]['mape_per_freq'][i]) for i, fl in enumerate(FREQ_SHORT)},
        } for c in CELLS},
        'cell_aging_matrix': R['cell_aging']['matrix'].tolist(),
        'cell_aging_45fold_stats': {
            'mean': R['cell_aging']['overall'],
            'std': R['cell_aging']['std'],
            'min': float(R['cell_aging']['matrix'].min()),
            'max': float(R['cell_aging']['matrix'].max()),
            'pct_below_1': float(np.mean(R['cell_aging']['matrix'] < 1) * 100),
            'pct_below_2': float(np.mean(R['cell_aging']['matrix'] < 2) * 100),
            'pct_below_3': float(np.mean(R['cell_aging']['matrix'] < 3) * 100),
        },
    }
    outpath = f'{OUTPUT_DIR}/nyquist_complete_summary.json'
    with open(outpath, 'w') as f:
        json.dump(s, f, indent=2)
    print(f"\n  Summary → {outpath}")
    return s

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 65)
    print("  NYQUIST RECONSTRUCTION — COMPLETE ANALYSIS")
    print("=" * 65)
    
    setup()
    CD = load_data()
    
    print("\nRunning all validation protocols...")
    R = run_all_protocols(CD)
    
    print("\nGenerating 10 figures...")
    fig01_loco_nyquist(R)
    fig02_loco_prospective(R)
    fig03_protocol_comparison(R)
    fig04_heatmap(R)
    fig05_parity(R)
    fig06_aging_trajectory(R)
    fig07_method_comparison(R)
    fig08_error_distribution(R)
    fig09_tau_space(CD)
    fig10_frequency_breakdown(R)
    
    S = save_summary(R)
    
    print("\n" + "=" * 65)
    print("  HEADLINE NUMBERS")
    print("=" * 65)
    h = S['headline']
    print(f"  Cell-LOOCV:                 {h['cell_loocv_mape']:.2f}%")
    print(f"  Aging-LOOCV:                {h['aging_loocv_mape']:.2f}% ± {h['aging_loocv_std']:.2f}%")
    print(f"  Cell + Aging holdout:       {h['cell_aging_holdout_mape']:.2f}% ± {h['cell_aging_holdout_std']:.2f}%")
    print(f"  LOCO + Prospective:         {h['loco_prospective_mape']:.2f}%")
    print(f"  Within-cell prospective:    {h['within_cell_mape']:.2f}%")
    print(f"  Analytical Z(ω):            {h['analytical_mape']:.2f}%")
    print(f"  EIS 3RC self-fit:           {h['eis_self_mape']:.2f}%")
    print(f"\n  45-fold stats:")
    cs = S['cell_aging_45fold_stats']
    print(f"    {cs['pct_below_1']:.0f}% of folds < 1%")
    print(f"    {cs['pct_below_2']:.0f}% of folds < 2%")
    print(f"    {cs['pct_below_3']:.0f}% of folds < 3%")
    print(f"    Worst fold: {cs['max']:.2f}%")
    print(f"\n  Regression beats analytical by {h['analytical_mape']/h['cell_loocv_mape']:.0f}×")
    print("=" * 65)
    print(f"\n  All figures saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
