"""
Nyquist Reconstruction: 36s vs 360s vs 3600s — Complete Analysis + Figures
==========================================================================
Merges tau from window_sweep with EIS ground truth from results_3d,
runs Ridge regression protocols, and generates publication figures.

Required data files (all in same directory or specify paths):
  1. results_3d_W8_SOC50.json      — tau_full + EIS features for W8
  2. results_3d_W9_SOC50.json      — tau_full + EIS features for W9
  3. results_3d_W10_Warmstart.json  — tau_full + EIS features for W10
  4. window_sweep_SOC50.json        — tau_w36 + tau_w360 for all cells


Author: Shafayeth Jamil (USC ECE), February 2026
"""

import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy import stats

# ============================================================================
# CONFIG
# ============================================================================
FREQ_KEYS_RE = ['Z_1kHz_re', 'Z_100Hz_re', 'Z_10Hz_re', 'Z_1Hz_re', 'Z_01Hz_re']
FREQ_KEYS_IM = ['Z_1kHz_im', 'Z_100Hz_im', 'Z_10Hz_im', 'Z_1Hz_im', 'Z_01Hz_im']
FREQ_LABELS = ['1 kHz', '100 Hz', '10 Hz', '1 Hz', '0.1 Hz']
FREQ_SHORT = ['1kHz', '100Hz', '10Hz', '1Hz', '0.1Hz']
CELLS = ['W8', 'W9', 'W10']
CC = {'W8': '#264653', 'W9': '#E76F51', 'W10': '#2A9D8F'}
ALPHA = 1e-4
DIAG_SPLIT = 10
DPI = 300


# ============================================================================
# LOAD & MERGE
# ============================================================================
def load_and_merge(data_dir):
    """Merge tau from window_sweep with EIS features from results_3d."""
    cell_files = {
        'W8':  os.path.join(data_dir, 'results_3d_W8_SOC50.json'),
        'W9':  os.path.join(data_dir, 'results_3d_W9_SOC50.json'),
        'W10': os.path.join(data_dir, 'results_3d_W10_Warmstart.json'),
    }
    ws_file = os.path.join(data_dir, 'window_sweep_SOC50.json')

    # Load EIS features from results_3d
    eis_data = {}
    for cell, fpath in cell_files.items():
        with open(fpath) as f:
            data = json.load(f)
        for d in data:
            key = (cell, int(d['diag']))
            eis_data[key] = d
        print(f"  results_3d {cell}: {len(data)} diagnostics")

    # Load window sweep tau values
    with open(ws_file) as f:
        ws_data = json.load(f)
    print(f"  window_sweep: {len(ws_data)} entries")

    # Build merged datasets for each window
    merged = {wname: {c: [] for c in CELLS} for wname in ['w36', 'w360', 'full']}

    for r in ws_data:
        cell = r['cell']
        diag = int(r['diag'])
        key = (cell, diag)
        if key not in eis_data:
            print(f"  WARNING: no EIS for {cell} diag {diag}")
            continue
        eis = eis_data[key]

        for wname in ['w36', 'w360']:
            tau_key = f'tau_{wname}'
            if tau_key not in r or r[tau_key] is None:
                continue
            entry = {'cell': cell, 'diag': diag, 'tau': r[tau_key]}
            for k in FREQ_KEYS_RE + FREQ_KEYS_IM:
                entry[k] = eis[k]
            merged[wname][cell].append(entry)

    # Full (3600s) from results_3d directly
    for cell, fpath in cell_files.items():
        with open(fpath) as f:
            data = json.load(f)
        for d in data:
            entry = {'cell': cell, 'diag': int(d['diag']), 'tau': d['tau_full']}
            for k in FREQ_KEYS_RE + FREQ_KEYS_IM:
                entry[k] = d[k]
            merged['full'][cell].append(entry)

    for wname in merged:
        for cell in CELLS:
            merged[wname][cell].sort(key=lambda x: x['diag'])

    for wname in ['w36', 'w360', 'full']:
        counts = {c: len(merged[wname][c]) for c in CELLS}
        print(f"  Merged {wname}: {counts}")

    return merged


# ============================================================================
# CORE FUNCTIONS
# ============================================================================
def get_xy(data_list):
    X = np.array([d['tau'] for d in data_list])
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


# ============================================================================
# VALIDATION PROTOCOLS
# ============================================================================
def run_protocols(CD):
    """Run all protocols for one window's data."""
    results = {}
    all_diags = sorted(set(d['diag'] for d in CD['W8']))

    # --- 1. Cell-LOOCV ---
    cell_loco = {}
    for tc in CELLS:
        others = [c for c in CELLS if c != tc]
        X_tr, Y_re_tr = [], []
        for c in others:
            x, yr, _, _ = get_xy(CD[c])
            X_tr.append(x); Y_re_tr.append(yr)
        X_tr = np.vstack(X_tr); Y_re_tr = np.vstack(Y_re_tr)
        X_te, Y_re_te, _, diags = get_xy(CD[tc])
        Yr_p = fit_predict(X_tr, Y_re_tr, X_te)
        cell_loco[tc] = {
            'overall_mape': mape_overall(Y_re_te, Yr_p),
            'mape_per_freq': mape_per_freq(Y_re_te, Yr_p),
            'Y_re_test': Y_re_te, 'Y_re_pred': Yr_p, 'diags': diags,
            'train_cells': others,
        }
    results['cell_loco'] = cell_loco

    # --- 2. Cell + Aging holdout (45 folds) ---
    ca_matrix = np.zeros((3, len(all_diags)))
    ca_details = {}
    for ci, tc in enumerate(CELLS):
        others = [c for c in CELLS if c != tc]
        ca_details[tc] = {}
        for di, hd in enumerate(all_diags):
            train = [d for c in others for d in CD[c] if d['diag'] != hd]
            test = [d for d in CD[tc] if d['diag'] == hd]
            if len(test) == 0:
                ca_matrix[ci, di] = np.nan
                continue
            X_tr, Y_tr, _, _ = get_xy(train)
            X_te, Y_te, _, _ = get_xy(test)
            Y_p = fit_predict(X_tr, Y_tr, X_te)
            ca_matrix[ci, di] = mape_overall(Y_te, Y_p)
            ca_details[tc][hd] = {'Y_re_test': Y_te, 'Y_re_pred': Y_p}
    results['cell_aging'] = {
        'matrix': ca_matrix, 'details': ca_details,
        'overall': float(np.nanmean(ca_matrix)),
        'std': float(np.nanstd(ca_matrix)),
        'pct_below_2': float(np.nanmean(ca_matrix < 2) * 100),
        'max': float(np.nanmax(ca_matrix)),
    }

    # --- 3. LOCO + Prospective ---
    loco_prosp = {}
    for tc in CELLS:
        others = [c for c in CELLS if c != tc]
        train = [d for c in others for d in CD[c] if d['diag'] <= DIAG_SPLIT]
        test = [d for d in CD[tc] if d['diag'] > DIAG_SPLIT]
        if len(test) == 0:
            continue
        X_tr, Y_tr, _, _ = get_xy(train)
        X_te, Y_te, _, diags = get_xy(test)
        Yr_p = fit_predict(X_tr, Y_tr, X_te)
        loco_prosp[tc] = {
            'overall_mape': mape_overall(Y_te, Yr_p),
            'mape_per_freq': mape_per_freq(Y_te, Yr_p),
        }
    results['loco_prosp'] = loco_prosp

    return results


# ============================================================================
# FIGURE 1: Side-by-side heatmaps (36s vs 3600s)
# ============================================================================
def fig_heatmap_comparison(R_36, R_3600, out_dir):
    mat36 = R_36['cell_aging']['matrix']
    mat3600 = R_3600['cell_aging']['matrix']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7.5), gridspec_kw={'hspace': 0.45})

    for ax, mat, label, stats_str in [
        (ax1, mat36, '36 s (BMS-realistic)',
         f"mean {np.nanmean(mat36):.2f}% ± {np.nanstd(mat36):.2f}%  "
         f"({np.nanmean(mat36 < 2)*100:.0f}% of folds < 2%)"),
        (ax2, mat3600, '3600 s (lab-grade)',
         f"mean {np.nanmean(mat3600):.2f}% ± {np.nanstd(mat3600):.2f}%  "
         f"({np.nanmean(mat3600 < 2)*100:.0f}% of folds < 2%)")
    ]:
        im = ax.imshow(mat, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=5)
        for ci in range(3):
            for di in range(mat.shape[1]):
                v = mat[ci, di]
                if np.isnan(v):
                    continue
                color = 'white' if v > 3 else 'black'
                ax.text(di, ci, f'{v:.1f}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=color)
        ax.set_yticks(range(3))
        ax.set_yticklabels(CELLS, fontsize=12, fontweight='bold')
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels(range(1, mat.shape[1]+1), fontsize=9)
        ax.set_xlabel('Diagnostic number (aging state)', fontsize=11)
        ax.set_title(f'{label} — {stats_str}', fontsize=12, fontweight='bold', pad=8)
        ax.axvline(9.5, color='white', ls='--', lw=2, alpha=0.8)

    cb = plt.colorbar(im, ax=[ax1, ax2], shrink=0.6, pad=0.02)
    cb.set_label('MAPE (%)', fontsize=11)

    fig.suptitle('36 seconds matches 3600 seconds: cell+aging holdout heatmaps',
                 fontsize=14, fontweight='bold', y=0.98)

    # Panel labels
    ax1.text(-0.03, 1.08, 'a', transform=ax1.transAxes, fontsize=16, fontweight='bold')
    ax2.text(-0.03, 1.08, 'b', transform=ax2.transAxes, fontsize=16, fontweight='bold')

    path = f'{out_dir}/fig_36s_vs_3600s_heatmaps.png'
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {path}")
    return path


# ============================================================================
# FIGURE 2: Per-cell bar chart across windows (cell+aging holdout)
# ============================================================================
def fig_per_cell_window(all_R, out_dir):
    windows = ['w36', 'w360', 'full']
    wlabels = ['36 s', '360 s', '3600 s']

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(windows))
    w = 0.22

    for i, cell in enumerate(CELLS):
        vals = [np.nanmean(all_R[wn]['cell_aging']['matrix'][CELLS.index(cell)])
                for wn in windows]
        bars = ax.bar(x + (i-1)*w, vals, w, color=CC[cell], alpha=0.85,
                      edgecolor='white', lw=0.8, label=cell)
        for j, v in enumerate(vals):
            ax.text(x[j]+(i-1)*w, v+0.03, f'{v:.2f}', ha='center',
                    fontsize=9, color=CC[cell], fontweight='bold')

    # Mean lines
    for j, wn in enumerate(windows):
        m = all_R[wn]['cell_aging']['overall']
        ax.plot([x[j]-0.35, x[j]+0.35], [m, m], 'k-', lw=1.8, alpha=0.5)
        ax.text(x[j]+0.38, m, f'μ={m:.2f}%', fontsize=9, fontweight='bold',
                color='#333', va='center')

    ax.set_xticks(x)
    ax.set_xticklabels(wlabels, fontsize=12)
    ax.set_xlabel('Observation window length', fontsize=12)
    ax.set_ylabel('$Z_{re}$ MAPE (%)', fontsize=12)
    ax.set_title('Per-cell reconstruction error × window length\n'
                 '(cell+aging double-blind holdout)', fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=10, framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.08)
    ax.axhline(2.0, color='#C44E52', ls='--', lw=1, alpha=0.5)
    ax.set_ylim(0, 2.5)

    # Annotation
    m36 = all_R['w36']['cell_aging']['overall']
    m3600 = all_R['full']['cell_aging']['overall']
    delta = m36 - m3600
    ax.annotate(f'Only +{delta:.2f}% degradation\nfrom 3600s → 36s',
                xy=(0, max(all_R['w36']['cell_aging']['matrix'][CELLS.index('W10')].mean(),
                           all_R['w36']['cell_aging']['matrix'][CELLS.index('W8')].mean())),
                xytext=(0.8, 2.2), fontsize=10, fontstyle='italic', color=CC['W10'],
                arrowprops=dict(arrowstyle='->', color=CC['W10'], lw=1.5),
                fontweight='bold')

    path = f'{out_dir}/fig_per_cell_window.png'
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {path}")
    return path


# ============================================================================
# FIGURE 3: Protocol comparison across windows
# ============================================================================
def fig_protocol_comparison(all_R, out_dir):
    fig, ax = plt.subplots(figsize=(11, 5.5))

    windows = ['w36', 'w360', 'full']
    wlabels = ['36 s', '360 s', '3600 s']
    protocols = ['Cell-LOOCV', 'Cell+Aging\nholdout', 'LOCO+\nprospective']

    x = np.arange(len(protocols))
    w = 0.22
    wcolors = ['#E76F51', '#E9C46A', '#264653']

    for wi, (wn, wl, wc) in enumerate(zip(windows, wlabels, wcolors)):
        R = all_R[wn]
        loco = np.mean([R['cell_loco'][c]['overall_mape'] for c in CELLS])
        ca = R['cell_aging']['overall']
        lp = np.mean([R['loco_prosp'][c]['overall_mape']
                       for c in CELLS if c in R['loco_prosp']])
        vals = [loco, ca, lp]
        bars = ax.bar(x + (wi-1)*w, vals, w, color=wc, alpha=0.85,
                      edgecolor='white', lw=0.8, label=wl)
        for j, v in enumerate(vals):
            ax.text(x[j]+(wi-1)*w, v+0.03, f'{v:.2f}', ha='center',
                    fontsize=8.5, color=wc, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(protocols, fontsize=11)
    ax.set_ylabel('$Z_{re}$ MAPE (%)', fontsize=12)
    ax.set_title('Nyquist reconstruction error across validation protocols\n'
                 '36s vs 360s vs 3600s observation windows',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=10, framealpha=0.95, title='Window', title_fontsize=10)
    ax.grid(True, axis='y', alpha=0.08)
    ax.axhline(2.0, color='#999', ls='--', lw=0.8, alpha=0.5)
    ax.text(2.6, 2.08, '2% threshold', fontsize=9, color='#888', fontstyle='italic')
    ax.set_ylim(0, 2.5)

    path = f'{out_dir}/fig_protocol_comparison.png'
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {path}")
    return path


# ============================================================================
# FIGURE 4: Error distributions (36s vs 3600s histograms)
# ============================================================================
def fig_distribution(R_36, R_3600, out_dir):
    mat36 = R_36['cell_aging']['matrix'].flatten()
    mat3600 = R_3600['cell_aging']['matrix'].flatten()
    mat36 = mat36[~np.isnan(mat36)]
    mat3600 = mat3600[~np.isnan(mat3600)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'wspace': 0.3})
    bins = np.arange(0, 5.5, 0.5)

    for ax, data, label, color in [
        (ax1, mat36, 'Window = 36 s', '#E76F51'),
        (ax2, mat3600, 'Window = 3600 s', '#264653')
    ]:
        n, _, patches = ax.hist(data, bins=bins, color=color, edgecolor='white',
                                 lw=0.8, alpha=0.8)
        for p, left in zip(patches, bins):
            if left >= 2.0:
                p.set_alpha(0.4)

        ax.axvline(np.mean(data), color='k', ls='--', lw=1.5,
                   label=f'Mean = {np.mean(data):.2f}%')
        ax.axvline(np.median(data), color=color, ls=':', lw=1.5,
                   label=f'Median = {np.median(data):.2f}%')
        ax.axvline(2.0, color='#C44E52', ls='--', lw=1, alpha=0.5)

        ax.set_xlabel('MAPE (%)', fontsize=12)
        ax.set_ylabel('Count (of 45 folds)', fontsize=12)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, framealpha=0.95)
        ax.set_xlim(0, 5)

        pct2 = np.sum(data < 2) / len(data) * 100
        pct1 = np.sum(data < 1) / len(data) * 100
        ax.text(0.95, 0.95, f'{pct2:.0f}% < 2%\n{pct1:.0f}% < 1%',
                transform=ax.transAxes, fontsize=11, ha='right', va='top',
                fontweight='bold',
                bbox=dict(boxstyle='round', fc='white', ec='#ccc', alpha=0.9))

    ax1.text(-0.06, 1.06, 'a', transform=ax1.transAxes, fontsize=16, fontweight='bold')
    ax2.text(-0.06, 1.06, 'b', transform=ax2.transAxes, fontsize=16, fontweight='bold')

    fig.suptitle('Error distributions: BMS-realistic (36 s) vs lab-grade (3600 s)',
                 fontsize=14, fontweight='bold', y=1.02)

    path = f'{out_dir}/fig_distribution_36_vs_3600.png'
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {path}")
    return path


# ============================================================================
# FIGURE 5: Deployment summary (3 panels)
# ============================================================================
def fig_deployment_summary(all_R, out_dir):
    windows = ['w36', 'w360', 'full']
    wlabels = ['36 s', '360 s', '3600 s']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5),
                                          gridspec_kw={'wspace': 0.35})

    means = [all_R[w]['cell_aging']['overall'] for w in windows]
    stds = [all_R[w]['cell_aging']['std'] for w in windows]
    pct2 = [all_R[w]['cell_aging']['pct_below_2'] for w in windows]
    worsts = [all_R[w]['cell_aging']['max'] for w in windows]

    # Panel a: Mean error with envelope
    ax1.plot(range(len(windows)), means, 'o-', color='#264653', lw=2.5, ms=10,
             mfc='white', mew=2.5, zorder=5)
    ax1.fill_between(range(len(windows)),
                     [m-s for m, s in zip(means, stds)],
                     [m+s for m, s in zip(means, stds)],
                     alpha=0.15, color='#264653')
    for i, (m, s) in enumerate(zip(means, stds)):
        ax1.text(i, m-0.15, f'{m:.2f}%', ha='center', fontsize=11,
                fontweight='bold', color='#264653')
    ax1.axhline(2.0, color='#C44E52', ls='--', lw=1, alpha=0.5)
    ax1.set_xticks(range(len(windows)))
    ax1.set_xticklabels(wlabels, fontsize=11)
    ax1.set_ylabel('Mean MAPE (%)', fontsize=12)
    ax1.set_title('Mean error', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 3.0)
    ax1.grid(True, alpha=0.08)
    ax1.text(-0.08, 1.06, 'a', transform=ax1.transAxes, fontsize=16, fontweight='bold')

    # Panel b: Pass rate
    bar_colors = ['#E76F51', '#E9C46A', '#264653']
    bars = ax2.bar(range(len(windows)), pct2, 0.5, color=bar_colors,
                   edgecolor='white', lw=0.8, alpha=0.85)
    for i, v in enumerate(pct2):
        ax2.text(i, v+1, f'{v:.0f}%', ha='center', fontsize=12, fontweight='bold',
                color=bar_colors[i])
    ax2.axhline(85, color='#999', ls=':', lw=1, alpha=0.5)
    ax2.set_xticks(range(len(windows)))
    ax2.set_xticklabels(wlabels, fontsize=11)
    ax2.set_ylabel('% of 45 folds < 2%', fontsize=12)
    ax2.set_title('Pass rate (< 2% threshold)', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.text(-0.08, 1.06, 'b', transform=ax2.transAxes, fontsize=16, fontweight='bold')

    # Panel c: Worst case
    bars = ax3.bar(range(len(windows)), worsts, 0.5, color=bar_colors,
                   edgecolor='white', lw=0.8, alpha=0.85)
    for i, v in enumerate(worsts):
        ax3.text(i, v+0.1, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold',
                color=bar_colors[i])
    ax3.axhline(5.0, color='#C44E52', ls='--', lw=1, alpha=0.5)
    ax3.set_xticks(range(len(windows)))
    ax3.set_xticklabels(wlabels, fontsize=11)
    ax3.set_ylabel('Worst-fold MAPE (%)', fontsize=12)
    ax3.set_title('Worst case', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 6)
    ax3.text(-0.08, 1.06, 'c', transform=ax3.transAxes, fontsize=16, fontweight='bold')

    fig.suptitle('Deployment robustness: Nyquist reconstruction under BMS-realistic window constraints',
                 fontsize=14, fontweight='bold', y=1.03)

    path = f'{out_dir}/fig_deployment_summary.png'
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved {path}")
    return path


# ============================================================================
# SAVE SUMMARY JSON
# ============================================================================
def save_summary(all_R, out_dir):
    summary = {}
    for wn, wl in [('w36', '36s'), ('w360', '360s'), ('full', '3600s')]:
        R = all_R[wn]
        loco_mean = np.mean([R['cell_loco'][c]['overall_mape'] for c in CELLS])
        lp_mean = np.mean([R['loco_prosp'][c]['overall_mape']
                           for c in CELLS if c in R['loco_prosp']])
        summary[wl] = {
            'cell_loocv_mape': float(loco_mean),
            'cell_loocv_per_cell': {c: float(R['cell_loco'][c]['overall_mape']) for c in CELLS},
            'cell_aging_holdout_mape': R['cell_aging']['overall'],
            'cell_aging_holdout_std': R['cell_aging']['std'],
            'cell_aging_pct_below_2': R['cell_aging']['pct_below_2'],
            'cell_aging_worst_fold': R['cell_aging']['max'],
            'loco_prospective_mape': float(lp_mean),
            'loco_prospective_per_cell': {
                c: float(R['loco_prosp'][c]['overall_mape'])
                for c in CELLS if c in R['loco_prosp']
            },
            'cell_aging_matrix': R['cell_aging']['matrix'].tolist(),
        }

    path = f'{out_dir}/nyquist_window_summary.json'
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary → {path}")
    return summary


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='.')
    parser.add_argument('--out_dir', default='./nyquist_window_figures')
    args = parser.parse_args()

    print("=" * 70)
    print("  NYQUIST RECONSTRUCTION: 36s vs 360s vs 3600s")
    print("=" * 70)

    merged = load_and_merge(args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    # Run all protocols for each window
    all_R = {}
    for wname, label in [('w36', '36 seconds'), ('w360', '360 seconds'), ('full', '3600 seconds')]:
        CD = merged[wname]
        n_total = sum(len(CD[c]) for c in CELLS)
        if n_total == 0:
            print(f"\n  {label}: NO DATA, skipping")
            continue

        n_states = len(CD['W8'][0]['tau'])
        print(f"\n{'='*70}")
        print(f"  WINDOW: {label}  ({n_total} points, n_states={n_states})")
        print(f"{'='*70}")

        for cell in CELLS:
            taus = np.array([d['tau'] for d in CD[cell]])
            print(f"  {cell} tau ranges: "
                  f"[{taus[:,0].min():.1f}-{taus[:,0].max():.1f}], "
                  f"[{taus[:,1].min():.1f}-{taus[:,1].max():.1f}], "
                  f"[{taus[:,2].min():.1f}-{taus[:,2].max():.1f}]")

        R = run_protocols(CD)
        all_R[wname] = R

        # Print results
        print(f"\n  Cell-LOOCV:")
        for cell in CELLS:
            m = R['cell_loco'][cell]['overall_mape']
            pf = R['cell_loco'][cell]['mape_per_freq']
            print(f"    {cell}: {m:.2f}%  per-freq: {[f'{x:.2f}' for x in pf]}")
        mean_loco = np.mean([R['cell_loco'][c]['overall_mape'] for c in CELLS])
        print(f"    MEAN: {mean_loco:.2f}%")

        ca = R['cell_aging']
        print(f"\n  Cell+Aging holdout (45 folds):")
        print(f"    Mean: {ca['overall']:.2f}% ± {ca['std']:.2f}%")
        print(f"    {ca['pct_below_2']:.0f}% of folds < 2%")
        print(f"    Worst fold: {ca['max']:.2f}%")

        if R['loco_prosp']:
            print(f"\n  LOCO + Prospective:")
            for cell in CELLS:
                if cell in R['loco_prosp']:
                    m = R['loco_prosp'][cell]['overall_mape']
                    print(f"    {cell}: {m:.2f}%")
            mean_lp = np.mean([R['loco_prosp'][c]['overall_mape']
                               for c in CELLS if c in R['loco_prosp']])
            print(f"    MEAN: {mean_lp:.2f}%")

    # --- Head-to-head ---
    print(f"\n{'='*70}")
    print(f"  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Window':<12} {'Cell-LOOCV':>12} {'Cell+Aging':>12} {'LOCO+Prosp':>12}")
    print(f"  {'-'*48}")
    for wn, wl in [('w36', '36s'), ('w360', '360s'), ('full', '3600s')]:
        if wn not in all_R:
            continue
        R = all_R[wn]
        loco = np.mean([R['cell_loco'][c]['overall_mape'] for c in CELLS])
        ca = R['cell_aging']['overall']
        lp = np.mean([R['loco_prosp'][c]['overall_mape']
                       for c in CELLS if c in R['loco_prosp']])
        print(f"  {wl:<12} {loco:>11.2f}% {ca:>11.2f}% {lp:>11.2f}%")

    # --- Generate figures ---
    print(f"\nGenerating figures...")
    fig_heatmap_comparison(all_R['w36'], all_R['full'], args.out_dir)
    fig_per_cell_window(all_R, args.out_dir)
    fig_protocol_comparison(all_R, args.out_dir)
    fig_distribution(all_R['w36'], all_R['full'], args.out_dir)
    fig_deployment_summary(all_R, args.out_dir)

    # --- Save summary ---
    save_summary(all_R, args.out_dir)

    print(f"\n{'='*70}")
    print(f"  DONE — all figures in {args.out_dir}/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
