"""
Stanford 3D + Warm-Start: Three-Process Electrochemical Decomposition
=====================================================================
Runs LGN with n_states=3 on Stanford's 3600s relaxation window to resolve:
  τ₁ ~ 0.5s    (charge transfer at electrode-electrolyte interface)
  τ₂ ~ 20-30s  (SEI layer impedance)
  τ₃ ~ 200-500s (solid-state Li⁺ diffusion)

Key fix: Warm-start initialization from previous diagnostic prevents
collapse into degenerate local minima at later diagnostics.

Usage:
  python run_stanford_3d_warmstart.py --data_dir lgn_csv --cells W8 W9 W10 --device cuda

Requires: run_degradation.py in same directory (imported)

Author: Shafayeth Jamil (USC ECE), February 2026
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

import torch
import torch.nn as nn

# Import core functions from existing pipeline
from run_degradation import (
    LGN_Battery, train_lgn, lgn_to_impedance_shape,
    fit_scale_and_Rs, compare_impedance, safe_corr,
    fit_exponentials, fit_eis_randles,
    correlation_analysis, plot_degradation, 
    print_trajectory_table, print_impedance_summary,
    _subsample_log
)


# ============================================================================
# PATCHED TRAINING: 3D initializations that span CT → SEI → Diffusion
# ============================================================================
def train_lgn_3d(t_data, eta_data, n_epochs=4000, lr=0.01,
                 subsample=300, device='cpu', verbose=False,
                 prev_d_params=None, prev_x0=None):
    """
    Multi-restart LGN training specifically for 3-state battery model.

    Warm-start: if prev_d_params is provided (from previous diagnostic),
    it is added as the first initialization. This prevents collapse into
    degenerate local minima when the loss landscape shifts between diagnostics.

    Initialization strategy:
      softplus(d) ≈ d for d >> 0, softplus(d) ≈ exp(d) for d << 0
      τ = 1/softplus(d)

      d =  3.0 → τ ≈ 0.05s   (very fast, sub-CT)
      d =  2.0 → τ ≈ 0.5s    (charge transfer)
      d =  1.0 → τ ≈ 0.8s    (charge transfer / SEI boundary)
      d =  0.0 → τ ≈ 1.4s    (SEI)
      d = -1.0 → τ ≈ 3.3s    (SEI)
      d = -2.0 → τ ≈ 8s      (SEI)
      d = -3.0 → τ ≈ 20s     (SEI / slow)
      d = -4.0 → τ ≈ 55s     (slow SEI)
      d = -5.0 → τ ≈ 150s    (diffusion)
      d = -5.5 → τ ≈ 245s    (diffusion)
      d = -6.0 → τ ≈ 403s    (diffusion)
      d = -6.5 → τ ≈ 670s    (diffusion)
      d = -7.0 → τ ≈ 1097s   (very slow diffusion)

    We use 6 restarts covering different τ hypotheses:
    """
    n_states = 3

    idx = _subsample_log(len(t_data), subsample) if subsample else np.arange(len(t_data))
    t_t = torch.tensor(t_data[idx], dtype=torch.float64, device=device)
    eta_t = torch.tensor(eta_data[idx], dtype=torch.float64, device=device)

    # 8 diverse initializations spanning the full electrochemical range
    inits = [
        # (fast CT, SEI, diffusion) — our best guess from 2D results
        torch.tensor([2.0, -3.0, -5.5]),    # τ ~ [0.5, 20, 245]
        torch.tensor([1.5, -2.5, -6.0]),    # τ ~ [0.6, 12, 403]
        torch.tensor([2.5, -3.5, -5.0]),    # τ ~ [0.4, 33, 150]

        # Wider spread
        torch.tensor([3.0, -2.0, -6.5]),    # τ ~ [0.3, 8, 670]
        torch.tensor([1.0, -3.0, -6.0]),    # τ ~ [0.8, 20, 403]
        torch.tensor([2.0, -4.0, -5.5]),    # τ ~ [0.5, 55, 245]

        # Very wide range (safety net)
        torch.tensor([2.0, -2.0, -7.0]),    # τ ~ [0.5, 8, 1097]
        torch.tensor([1.0, -1.5, -5.0]),    # τ ~ [0.8, 5, 150]
    ]

    # Warm-start: prepend previous diagnostic's solution
    if prev_d_params is not None:
        inits.insert(0, prev_d_params.clone().cpu())
        if verbose:
            print(f"      [warm-start from previous diagnostic]")

    best_model, best_loss = None, float('inf')

    for i_init, init_d in enumerate(inits):
        model = LGN_Battery(n_states).double().to(device)
        model.s_params.requires_grad_(False)  # diagonal A
        model.d_params.data = init_d.clone().to(device)

        # Warm-start x0 from previous diagnostic for the first init
        if i_init == 0 and prev_x0 is not None and prev_d_params is not None:
            model.x0.data = prev_x0.clone().to(device)
        else:
            model.x0.data.fill_(eta_data[0] / n_states)

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=300, factor=0.5, min_lr=1e-5)

        run_best_loss, run_best_state = float('inf'), None
        for ep in range(n_epochs):
            opt.zero_grad()
            pred = model(t_t)
            loss = torch.mean((pred - eta_t)**2) / (torch.mean(eta_t**2) + 1e-12)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step(loss.item())
            if loss.item() < run_best_loss:
                run_best_loss = loss.item()
                run_best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if run_best_loss < best_loss:
            best_loss = run_best_loss
            model.load_state_dict(run_best_state)
            best_model = model

        if verbose:
            taus = model.get_time_constants()
            print(f"      init {i_init}: τ={np.round(taus,1)}  loss={run_best_loss:.4e}")

    best_model.eval()
    with torch.no_grad():
        eta_pred = best_model(t_t).cpu().numpy()
    nrmse = np.sqrt(np.mean((eta_pred - eta_data[idx])**2)) / (np.abs(eta_data).max() + 1e-12)

    if verbose:
        print(f"    BEST → τ = {np.round(best_model.get_time_constants(), 3)}  "
              f"NRMSE = {nrmse:.5f}")

    return best_model, nrmse, best_loss


# ============================================================================
# ANALYZE SINGLE DIAGNOSTIC (3D version)
# ============================================================================
def analyze_diagnostic_3d(t_relax, v_relax, freq, re_z, im_z,
                          cell='', diag=0, soc=0, device='cpu',
                          prev_model=None):
    """
    Analyze one diagnostic checkpoint with 3-state LGN.
    If prev_model is provided, uses warm-start initialization.
    Also runs 3-exponential curve fit and 3-RC EIS fit for comparison.
    """
    n_states = 3
    res = {'cell': cell, 'diag': diag, 'soc': soc}

    # ---- HPPC-derived features ----
    v_inf = v_relax[-1]
    eta_full = v_relax - v_inf
    res['eta0'] = float(eta_full[0])
    res['R_pulse'] = float(abs(eta_full[0]) / 4.85)
    res['v_end'] = float(v_inf)

    # ---- LGN-3D on multiple windows ----
    windows = {'full': len(t_relax), 'w300': 300, 'w100': 100}
    full_model = None  # save for warm-start chain
    for wname, wlen in windows.items():
        if isinstance(wlen, int) and wlen < len(t_relax):
            mask = t_relax <= wlen
            t_w = t_relax[mask]
            v_w = v_relax[mask]
            eta_w = v_w - v_w[-1]
        else:
            t_w = t_relax
            eta_w = eta_full

        # Use 3D training for full window, 2D for short windows
        # (short windows can't resolve 3 modes)
        if wname == 'full':
            # Extract warm-start params from previous diagnostic
            prev_d = prev_model.d_params.data if prev_model is not None else None
            prev_x = prev_model.x0.data if prev_model is not None else None
            model, nrmse, loss = train_lgn_3d(
                t_w, eta_w, n_epochs=4000, lr=0.01,
                subsample=min(300, len(t_w)), device=device, verbose=True,
                prev_d_params=prev_d, prev_x0=prev_x)
            full_model = model  # save for warm-start chain
        else:
            # Fall back to 2D for short windows
            from run_degradation import train_lgn
            model, nrmse, loss = train_lgn(
                t_w, eta_w, n_states=2, n_epochs=3000,
                lr=0.01, subsample=min(250, len(t_w)), device=device, verbose=False)

        taus = model.get_time_constants()
        d_rates = model.get_diagonal_damping()
        A = model.get_A_numpy()

        res[f'tau_{wname}'] = taus.tolist()
        res[f'd_{wname}'] = d_rates.tolist()
        res[f'decay_{wname}'] = model.get_mode_decay_rates().tolist()
        res[f'nrmse_{wname}'] = nrmse
        res[f'A_{wname}'] = A.tolist()
        res[f'n_states_{wname}'] = model.n

        # Impedance reconstruction
        Z_shape = lgn_to_impedance_shape(model, freq)
        Z_data = re_z + 1j * im_z
        band = (freq >= 0.05) & (freq <= 2000)
        a_fit, Rs_fit, Z_pred = fit_scale_and_Rs(Z_shape[band], Z_data[band])
        Z_pred_full = a_fit * Z_shape + Rs_fit
        comp = compare_impedance(Z_pred_full[band], Z_data[band], freq[band])
        res[f'z_re_corr_{wname}'] = comp['re_corr']
        res[f'z_im_corr_{wname}'] = comp['im_corr']
        res[f'z_nrmse_{wname}'] = comp['nrmse']
        res[f'Rs_fit_{wname}'] = float(Rs_fit)

    # ---- Curve fit baseline (3 exponentials) ----
    cf_taus, cf_nrmse, _ = fit_exponentials(t_relax, eta_full, n_exp=3)
    res['tau_cf'] = cf_taus.tolist() if cf_taus is not None else None
    res['nrmse_cf'] = cf_nrmse

    # Also run 2-exp curve fit for comparison
    cf2_taus, cf2_nrmse, _ = fit_exponentials(t_relax, eta_full, n_exp=2)
    res['tau_cf_2exp'] = cf2_taus.tolist() if cf2_taus is not None else None
    res['nrmse_cf_2exp'] = cf2_nrmse

    # ---- EIS ground truth: 3-RC fit ----
    eis_params_3rc, Z_fit_3rc = fit_eis_randles(freq, re_z, im_z, n_rc=3)
    if eis_params_3rc:
        res['tau_eis_3rc'] = sorted([eis_params_3rc[f'tau{i+1}'] for i in range(3)])
        res['Rs_eis_3rc'] = eis_params_3rc['Rs']
        for i in range(3):
            res[f'R{i+1}_eis_3rc'] = eis_params_3rc[f'R{i+1}']
    else:
        res['tau_eis_3rc'] = None

    # Also keep 2-RC EIS fit for comparison
    eis_params_2rc, Z_fit_2rc = fit_eis_randles(freq, re_z, im_z, n_rc=2)
    if eis_params_2rc:
        res['tau_eis_2rc'] = sorted([eis_params_2rc[f'tau{i+1}'] for i in range(2)])
        res['Rs_eis_2rc'] = eis_params_2rc['Rs']
        for i in range(2):
            res[f'R{i+1}_eis_2rc'] = eis_params_2rc[f'R{i+1}']
    else:
        res['tau_eis_2rc'] = None

    # ---- EIS scalar features (model-free, for correlation) ----
    Z_data = re_z + 1j * im_z
    for f_target, label in [(1000, 'Z_1kHz'), (100, 'Z_100Hz'),
                             (10, 'Z_10Hz'), (1, 'Z_1Hz'), (0.1, 'Z_01Hz')]:
        fi = np.argmin(np.abs(freq - f_target))
        res[f'{label}_re'] = float(re_z[fi])
        res[f'{label}_im'] = float(im_z[fi])
        res[f'{label}_mag'] = float(np.abs(Z_data[fi]))

    return res, full_model


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_stanford_3d(data_dir, cells, soc_target=50, device='cpu'):
    """Run 3D LGN on Stanford cells."""
    all_results = []

    for cell in cells:
        print(f"\n{'#'*70}")
        print(f"# CELL: {cell}  (3D LGN: charge-transfer + SEI + diffusion)")
        print(f"{'#'*70}")

        hppc = pd.read_csv(f'{data_dir}/{cell}_hppc_relaxation.csv')
        eis = pd.read_csv(f'{data_dir}/{cell}_eis.csv')
        diags = sorted(set(hppc['diag'].unique()) & set(eis['diag'].unique()))

        soc_col = f'{soc_target}pct'
        re_col = f're_z_ohm_{soc_col}'
        im_col = f'im_z_ohm_{soc_col}'

        prev_model = None  # warm-start chain: reset per cell

        for diag in diags:
            seg = hppc[(hppc['diag'] == diag) & (hppc['soc_pct'] == soc_target)]
            if len(seg) == 0:
                continue
            eis_seg = eis[eis['diag'] == diag]
            if re_col not in eis_seg.columns:
                continue

            print(f"\n  Diag {diag}:")
            r, full_model = analyze_diagnostic_3d(
                seg['time_s'].values, seg['voltage_V'].values,
                eis_seg['freq_Hz'].values,
                eis_seg[re_col].values,
                eis_seg[im_col].values,
                cell=cell, diag=diag, soc=soc_target, device=device,
                prev_model=prev_model)
            all_results.append(r)

            # Update warm-start chain
            prev_model = full_model

            # Print summary
            taus = r['tau_full']
            z_corr = r.get('z_re_corr_full', 0)
            tau_eis = r.get('tau_eis_3rc', [])
            print(f"    LGN-3D τ = {[round(t,2) for t in taus]}")
            print(f"    EIS-3RC τ = {[round(t,2) for t in tau_eis] if tau_eis else 'FAILED'}")
            print(f"    Z_re corr = {z_corr:.4f}, NRMSE = {r['nrmse_full']:.5f}")

    return all_results


def print_3d_summary(results):
    """Print comparison table: 3D LGN vs 3-RC EIS vs 2D LGN."""
    print(f"\n{'='*90}")
    print(f"3D DECOMPOSITION SUMMARY")
    print(f"{'='*90}")
    print(f"{'Cell':<5} {'Diag':>4} | {'LGN τ₁':>8} {'LGN τ₂':>8} {'LGN τ₃':>8} | "
          f"{'EIS τ₁':>8} {'EIS τ₂':>8} {'EIS τ₃':>8} | {'NRMSE':>7} {'Z_corr':>7}")
    print(f"{'':>10} | {'(CT)':>8} {'(SEI)':>8} {'(diff)':>8} | "
          f"{'(CT)':>8} {'(SEI)':>8} {'(diff)':>8} |")
    print('-' * 90)

    for r in sorted(results, key=lambda x: (x['cell'], x['diag'])):
        taus = r['tau_full']
        eis_taus = r.get('tau_eis_3rc', [None]*3)
        if eis_taus is None:
            eis_taus = [None]*3

        lgn_str = ' '.join([f'{t:8.2f}' for t in taus])
        eis_str = ' '.join([f'{t:8.2f}' if t else f'{"N/A":>8}' for t in eis_taus])

        print(f"{r['cell']:<5} {r['diag']:4.0f} | {lgn_str} | {eis_str} | "
              f"{r['nrmse_full']:7.5f} {r.get('z_re_corr_full',0):7.4f}")


def correlation_analysis_3d(results):
    """Compute correlations for 3D results — per-τ vs EIS markers."""
    cells = sorted(set(r['cell'] for r in results))
    print(f"\n{'='*80}")
    print(f"3D DEGRADATION CORRELATION ANALYSIS")
    print(f"{'='*80}")

    all_corr = {}

    for cell in cells:
        cr = sorted([r for r in results if r['cell'] == cell], key=lambda r: r['diag'])
        if len(cr) < 4:
            continue

        print(f"\n  ── Cell {cell} ({len(cr)} diagnostics) ──")

        # For each τ component
        for idx in range(3):
            tau_label = ['τ₁(CT)', 'τ₂(SEI)', 'τ₃(diff)'][idx]
            tau_vals = np.array([r['tau_full'][idx] for r in cr])

            for marker in ['Z_1kHz_re', 'Z_1Hz_re', 'Z_01Hz_re', 'Z_1kHz_mag']:
                marker_vals = np.array([r[marker] for r in cr])
                rho_s, p_s = stats.spearmanr(tau_vals, marker_vals)
                rho_p, p_p = stats.pearsonr(tau_vals, marker_vals)
                key = f"{cell}_tau_full[{idx}]_vs_{marker}"
                all_corr[key] = {
                    'spearman_r': rho_s, 'spearman_p': p_s,
                    'pearson_r': rho_p, 'pearson_p': p_p
                }

                sig = '***' if p_s < 0.001 else '**' if p_s < 0.01 else '*' if p_s < 0.05 else ''
                if abs(rho_s) > 0.5 or sig:
                    print(f"    {tau_label} vs {marker}: ρ_s={rho_s:+.3f}{sig}  ρ_p={rho_p:+.3f}")

        # R_pulse baseline
        rp = np.array([r['R_pulse'] for r in cr])
        z1k = np.array([r['Z_1kHz_re'] for r in cr])
        rho, p = stats.spearmanr(rp, z1k)
        print(f"    R_pulse vs Z_1kHz: ρ_s={rho:+.3f} (p={p:.4f})")
        all_corr[f"{cell}_R_pulse_vs_Z_1kHz"] = {'spearman_r': rho, 'spearman_p': p}

    return all_corr


# ============================================================================
# 2D vs 3D COMPARISON (THE MONEY FIGURE)
# ============================================================================
def run_2d_baseline(data_dir, cells, soc_target=50, device='cpu'):
    """Run standard 2D LGN for head-to-head comparison."""
    all_results = []
    soc_col = f'{soc_target}pct'
    re_col = f're_z_ohm_{soc_col}'
    im_col = f'im_z_ohm_{soc_col}'

    for cell in cells:
        hppc = pd.read_csv(f'{data_dir}/{cell}_hppc_relaxation.csv')
        eis = pd.read_csv(f'{data_dir}/{cell}_eis.csv')
        diags = sorted(set(hppc['diag'].unique()) & set(eis['diag'].unique()))

        for diag in diags:
            seg = hppc[(hppc['diag'] == diag) & (hppc['soc_pct'] == soc_target)]
            if len(seg) == 0:
                continue
            eis_seg = eis[eis['diag'] == diag]
            if re_col not in eis_seg.columns:
                continue

            t_relax = seg['time_s'].values
            v_relax = seg['voltage_V'].values
            v_inf = v_relax[-1]
            eta_full = v_relax - v_inf

            # 2D LGN
            model_2d, nrmse_2d, _ = train_lgn(
                t_relax, eta_full, n_states=2, n_epochs=3000,
                lr=0.01, subsample=250, device=device)

            # 2-exp curve fit
            cf_taus, cf_nrmse, _ = fit_exponentials(t_relax, eta_full, n_exp=2)

            all_results.append({
                'cell': cell, 'diag': diag,
                'tau_2d': model_2d.get_time_constants().tolist(),
                'nrmse_2d': nrmse_2d,
                'tau_cf_2exp': cf_taus.tolist() if cf_taus is not None else None,
                'nrmse_cf_2exp': cf_nrmse,
            })

    return all_results


def print_2d_vs_3d(results_3d, results_2d):
    """Head-to-head comparison table."""
    print(f"\n{'='*110}")
    print(f"2D vs 3D COMPARISON")
    print(f"{'='*110}")
    print(f"{'Cell':<5} {'Diag':>4} | {'2D τ₁':>7} {'2D τ₂':>7} {'':>7} {'NRMSE':>7} | "
          f"{'3D τ₁':>7} {'3D τ₂':>7} {'3D τ₃':>7} {'NRMSE':>7} | "
          f"{'EIS τ₁':>7} {'EIS τ₂':>7} {'EIS τ₃':>7}")
    print('-' * 110)

    for r3 in sorted(results_3d, key=lambda x: (x['cell'], x['diag'])):
        # Find matching 2D result
        r2 = next((r for r in results_2d
                    if r['cell'] == r3['cell'] and r['diag'] == r3['diag']), None)

        t3 = r3['tau_full']
        eis = r3.get('tau_eis_3rc', [None]*3) or [None]*3
        t2 = r2['tau_2d'] if r2 else [None, None]
        n2 = r2['nrmse_2d'] if r2 else None

        def fmt(v, w=7):
            return f'{v:{w}.2f}' if v is not None else f'{"N/A":>{w}}'

        print(f"{r3['cell']:<5} {r3['diag']:4.0f} | "
              f"{fmt(t2[0])} {fmt(t2[1])} {'':>7} {fmt(n2,7) if n2 else 'N/A':>7} | "
              f"{fmt(t3[0])} {fmt(t3[1])} {fmt(t3[2])} {r3['nrmse_full']:7.5f} | "
              f"{fmt(eis[0])} {fmt(eis[1])} {fmt(eis[2])}")


def plot_3d_results(results_3d, results_2d, out_dir):
    """Publication figures for 3D decomposition."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

    cells = sorted(set(r['cell'] for r in results_3d))
    colors = {'W8': '#1f77b4', 'W9': '#ff7f0e', 'W10': '#2ca02c',
              'V4': '#d62728', 'W3': '#9467bd'}

    # ---- Figure 1: Three τ trajectories + EIS ground truth ----
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    labels = ['τ₁ (charge transfer)', 'τ₂ (SEI)', 'τ₃ (diffusion)', 'Z @ 1kHz [mΩ]']

    for ci, cell in enumerate(cells):
        cr = sorted([r for r in results_3d if r['cell'] == cell], key=lambda r: r['diag'])
        diags = [r['diag'] for r in cr]
        c = colors.get(cell, f'C{ci}')

        for si in range(3):
            axes[si].plot(diags, [r['tau_full'][si] for r in cr],
                         'o-', color=c, markersize=5, linewidth=1.5, label=cell)
        axes[3].plot(diags, [r['Z_1kHz_re'] * 1000 for r in cr],
                     'o-', color=c, markersize=5, linewidth=1.5, label=cell)

    for si in range(3):
        axes[si].set_xlabel('Diagnostic #')
        axes[si].set_ylabel(f'τ{si+1} [s]')
        axes[si].set_title(labels[si])
        axes[si].grid(True, alpha=0.2)
    axes[3].set_xlabel('Diagnostic #')
    axes[3].set_ylabel('Z [mΩ]')
    axes[3].set_title(labels[3])
    axes[3].grid(True, alpha=0.2)
    axes[0].legend()

    fig.suptitle('3D LGN: Three-Process Electrochemical Decomposition', fontweight='bold', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/fig_3d_trajectories.png')
    plt.close(fig)

    # ---- Figure 2: LGN τ vs EIS τ alignment ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    tau_labels = ['τ₁ (CT)', 'τ₂ (SEI)', 'τ₃ (diff)']

    for si in range(3):
        lgn_vals, eis_vals = [], []
        for r in results_3d:
            eis_taus = r.get('tau_eis_3rc')
            if eis_taus and len(eis_taus) == 3:
                lgn_vals.append(r['tau_full'][si])
                eis_vals.append(eis_taus[si])

        if lgn_vals:
            lgn_arr, eis_arr = np.array(lgn_vals), np.array(eis_vals)
            axes[si].scatter(eis_arr, lgn_arr, c='steelblue', s=40, alpha=0.7)
            # Identity line
            lo = min(eis_arr.min(), lgn_arr.min()) * 0.8
            hi = max(eis_arr.max(), lgn_arr.max()) * 1.2
            axes[si].plot([lo, hi], [lo, hi], '--', color='gray', alpha=0.5)
            rho, p = stats.spearmanr(lgn_arr, eis_arr)
            axes[si].set_title(f'{tau_labels[si]}  ρ={rho:.3f}')
        else:
            axes[si].set_title(f'{tau_labels[si]}  (no EIS data)')

        axes[si].set_xlabel(f'EIS {tau_labels[si]} [s]')
        axes[si].set_ylabel(f'LGN {tau_labels[si]} [s]')
        axes[si].grid(True, alpha=0.2)

    fig.suptitle('LGN-3D vs EIS-3RC: Time Constant Alignment', fontweight='bold', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/fig_lgn_vs_eis_alignment.png')
    plt.close(fig)

    # ---- Figure 3: Per-τ correlation with degradation markers ----
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    markers_list = ['Z_1kHz_re', 'Z_1Hz_re', 'Z_01Hz_re']
    marker_labels = ['Z @ 1kHz [mΩ]', 'Z @ 1Hz [mΩ]', 'Z @ 0.1Hz [mΩ]']

    for si in range(3):
        for mi, (marker, mlabel) in enumerate(zip(markers_list, marker_labels)):
            ax = axes[si][mi]
            for ci, cell in enumerate(cells):
                cr = [r for r in results_3d if r['cell'] == cell]
                c = colors.get(cell, f'C{ci}')
                taus = [r['tau_full'][si] for r in cr]
                mvals = [r[marker] * 1000 for r in cr]
                ax.scatter(taus, mvals, color=c, s=30, label=cell, alpha=0.7)

            # Pooled correlation
            all_t = [r['tau_full'][si] for r in results_3d]
            all_m = [r[marker] * 1000 for r in results_3d]
            rho, p = stats.spearmanr(all_t, all_m)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            ax.set_title(f'τ{si+1} vs {mlabel.split("[")[0]} ρ={rho:+.3f}{sig}', fontsize=10)
            ax.set_xlabel(f'{tau_labels[si]} [s]')
            ax.set_ylabel(mlabel)
            ax.grid(True, alpha=0.2)
            if si == 0 and mi == 0:
                ax.legend(fontsize=8)

    fig.suptitle('3D LGN: Per-Mode Correlation with EIS Degradation Markers', fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/fig_3d_correlations.png')
    plt.close(fig)

    print(f"  3D plots saved to {out_dir}/")


# ============================================================================
# MULTI-SOC MERGE + CROSS-SOC ANALYSIS
# ============================================================================
def merge_soc_results(out_dir, socs=[20, 50, 80]):
    """Merge per-SOC result files into one combined analysis."""
    all_3d = []
    all_2d = []

    for soc in socs:
        soc_dir = f'{out_dir}/soc{soc}'
        f3d = f'{soc_dir}/results_3d.json'
        f2d = f'{soc_dir}/results_2d_baseline.json'
        if os.path.exists(f3d):
            with open(f3d) as f:
                data = json.load(f)
                for r in data:
                    r['soc'] = soc
                all_3d.extend(data)
                print(f"  SOC {soc}%: {len(data)} 3D results loaded")
        if os.path.exists(f2d):
            with open(f2d) as f:
                data = json.load(f)
                for r in data:
                    r['soc'] = soc
                all_2d.extend(data)

    return all_3d, all_2d


def cross_soc_analysis(results_3d):
    """Analyze consistency of τ decomposition across SOC levels."""
    cells = sorted(set(r['cell'] for r in results_3d))
    socs = sorted(set(r['soc'] for r in results_3d))

    print(f"\n{'='*80}")
    print(f"CROSS-SOC 3D ANALYSIS ({len(results_3d)} total experiments)")
    print(f"Cells: {cells}, SOCs: {socs}")
    print(f"{'='*80}")

    # Per-SOC τ summary
    for soc in socs:
        sr = [r for r in results_3d if r['soc'] == soc]
        if not sr:
            continue
        t1 = np.array([r['tau_full'][0] for r in sr])
        t2 = np.array([r['tau_full'][1] for r in sr])
        t3 = np.array([r['tau_full'][2] for r in sr])
        print(f"\n  SOC {soc}% (n={len(sr)}):")
        print(f"    τ₁(CT):   {t1.mean():.2f} ± {t1.std():.2f} s  [{t1.min():.2f} – {t1.max():.2f}]")
        print(f"    τ₂(SEI):  {t2.mean():.1f} ± {t2.std():.1f} s  [{t2.min():.1f} – {t2.max():.1f}]")
        print(f"    τ₃(diff): {t3.mean():.0f} ± {t3.std():.0f} s  [{t3.min():.0f} – {t3.max():.0f}]")

    # Per-SOC degradation correlations
    print(f"\n  ── Degradation correlations per SOC ──")
    for soc in socs:
        sr = [r for r in results_3d if r['soc'] == soc]
        if len(sr) < 4:
            continue
        for idx, label in enumerate(['τ₁(CT)', 'τ₂(SEI)', 'τ₃(diff)']):
            tvals = np.array([r['tau_full'][idx] for r in sr])
            for marker in ['Z_1kHz_re', 'Z_1Hz_re']:
                mvals = np.array([r[marker] for r in sr])
                rho, p = stats.spearmanr(tvals, mvals)
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                if abs(rho) > 0.3 or sig:
                    print(f"    SOC {soc}% {label} vs {marker}: ρ={rho:+.3f}{sig} (n={len(sr)})")

    # Cross-SOC τ consistency per cell per diag
    print(f"\n  ── τ consistency across SOC (same cell, same diag) ──")
    for cell in cells:
        for idx, label in enumerate(['τ₁(CT)', 'τ₂(SEI)', 'τ₃(diff)']):
            # For each diag, get τ at each SOC
            cr = [r for r in results_3d if r['cell'] == cell]
            diags = sorted(set(r['diag'] for r in cr))
            cvs = []
            for diag in diags:
                dr = [r for r in cr if r['diag'] == diag]
                if len(dr) >= 2:
                    tvals = [r['tau_full'][idx] for r in dr]
                    cv = np.std(tvals) / (np.mean(tvals) + 1e-12) * 100
                    cvs.append(cv)
            if cvs:
                print(f"    {cell} {label}: mean CV across SOC = {np.mean(cvs):.1f}%")


def plot_cross_soc(results_3d, out_dir):
    """Cross-SOC publication figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

    cells = sorted(set(r['cell'] for r in results_3d))
    socs = sorted(set(r['soc'] for r in results_3d))
    colors = {'W8': '#1f77b4', 'W9': '#ff7f0e', 'W10': '#2ca02c'}
    soc_markers = {20: 's', 50: 'o', 80: '^'}
    tau_labels = ['τ₁ (CT)', 'τ₂ (SEI)', 'τ₃ (diffusion)']

    # Figure: τ trajectories at each SOC, one row per τ component
    fig, axes = plt.subplots(3, len(socs) + 1, figsize=(5 * (len(socs) + 1), 12))

    for si in range(3):
        for soc_i, soc in enumerate(socs):
            ax = axes[si][soc_i]
            for cell in cells:
                cr = sorted([r for r in results_3d if r['cell'] == cell and r['soc'] == soc],
                           key=lambda r: r['diag'])
                if not cr:
                    continue
                diags = [r['diag'] for r in cr]
                tvals = [r['tau_full'][si] for r in cr]
                ax.plot(diags, tvals, 'o-', color=colors.get(cell, 'gray'),
                       marker=soc_markers.get(soc, 'o'), markersize=5,
                       linewidth=1.5, label=cell)
            ax.set_xlabel('Diagnostic #')
            ax.set_ylabel(f'{tau_labels[si]} [s]')
            ax.set_title(f'{tau_labels[si]} @ SOC={soc}%')
            ax.grid(True, alpha=0.2)
            if si == 0 and soc_i == 0:
                ax.legend()

        # EIS comparison column
        ax = axes[si][-1]
        marker_name = ['Z_1kHz_re', 'Z_1Hz_re', 'Z_01Hz_re'][si]
        marker_label = ['Z@1kHz', 'Z@1Hz', 'Z@0.1Hz'][si]
        for cell in cells:
            cr = sorted([r for r in results_3d if r['cell'] == cell and r['soc'] == 50],
                       key=lambda r: r['diag'])
            if cr:
                diags = [r['diag'] for r in cr]
                mvals = [r[marker_name] * 1000 for r in cr]
                ax.plot(diags, mvals, 'o-', color=colors.get(cell, 'gray'),
                       markersize=5, linewidth=1.5, label=cell)
        ax.set_xlabel('Diagnostic #')
        ax.set_ylabel(f'{marker_label} [mΩ]')
        ax.set_title(f'EIS {marker_label} (50% SOC)')
        ax.grid(True, alpha=0.2)

    fig.suptitle('3D LGN: Three-Process Decomposition Across SOC Levels', fontweight='bold', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/fig_cross_soc_3d.png')
    plt.close(fig)
    print(f"  Cross-SOC plot saved to {out_dir}/fig_cross_soc_3d.png")


# ============================================================================
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Stanford 3D LGN Decomposition')
    p.add_argument('--data_dir', default='lgn_csv')
    p.add_argument('--cells', nargs='+', default=['W8', 'W9', 'W10'])
    p.add_argument('--soc', type=int, default=None,
                   help='Single SOC to run. If not set, use --socs for multi-SOC.')
    p.add_argument('--socs', nargs='+', type=int, default=[20, 50, 80],
                   help='SOC levels for multi-GPU launch')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out_dir', default='results_stanford_3d')
    p.add_argument('--plot', action='store_true')
    p.add_argument('--skip_2d', action='store_true', help='Skip 2D baseline comparison')
    p.add_argument('--merge_only', action='store_true',
                   help='Skip fitting, just merge existing per-SOC results')
    args = p.parse_args()

    # ================================================================
    # MODE 1: Merge existing per-SOC results
    # ================================================================
    if args.merge_only:
        print("Merging per-SOC results...")
        all_3d, all_2d = merge_soc_results(args.out_dir, args.socs)
        if all_3d:
            print_3d_summary(all_3d)
            cross_soc_analysis(all_3d)
            correlation_analysis_3d(all_3d)
            if args.plot:
                plot_cross_soc(all_3d, args.out_dir)
                plot_3d_results(all_3d, all_2d, args.out_dir)
            # Save merged
            with open(f'{args.out_dir}/results_3d_all_socs.json', 'w') as f:
                json.dump(all_3d, f, indent=2)
            print(f"\n✓ Merged {len(all_3d)} results saved to {args.out_dir}/results_3d_all_socs.json")
        sys.exit(0)

    # ================================================================
    # MODE 2: Single SOC run (for per-GPU execution)
    # ================================================================
    if args.soc is not None:
        soc = args.soc
        soc_out = f'{args.out_dir}/soc{soc}'

        if args.device.startswith('cuda') and not torch.cuda.is_available():
            print("CUDA not available → CPU")
            args.device = 'cpu'

        print(f"Device: {args.device}")
        print(f"Cells: {args.cells}, SOC: {soc}%")
        print(f"Model: 3-state LGN-SD (charge transfer + SEI + diffusion)")
        print()

        # 3D run
        results_3d = run_stanford_3d(args.data_dir, args.cells,
                                      soc_target=soc, device=args.device)
        print_3d_summary(results_3d)
        corr = correlation_analysis_3d(results_3d)

        # 2D baseline
        results_2d = []
        if not args.skip_2d:
            print(f"\nRunning 2D baseline...")
            results_2d = run_2d_baseline(args.data_dir, args.cells,
                                          soc_target=soc, device=args.device)
            print_2d_vs_3d(results_3d, results_2d)

        # Save per-SOC
        os.makedirs(soc_out, exist_ok=True)
        serializable = []
        for r in results_3d:
            s = {}
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    s[k] = v.tolist()
                elif isinstance(v, (np.floating, np.integer)):
                    s[k] = float(v)
                elif v is None or isinstance(v, (int, float, str, list, dict, bool)):
                    s[k] = v
            serializable.append(s)

        with open(f'{soc_out}/results_3d.json', 'w') as f:
            json.dump(serializable, f, indent=2)
        with open(f'{soc_out}/correlations_3d.json', 'w') as f:
            json.dump({k: {kk: float(vv) for kk, vv in v.items()}
                       for k, v in corr.items()}, f, indent=2)
        if results_2d:
            with open(f'{soc_out}/results_2d_baseline.json', 'w') as f:
                json.dump(results_2d, f, indent=2)

        if args.plot:
            plot_3d_results(results_3d, results_2d, soc_out)

        print(f"\n✓ SOC {soc}% saved to {soc_out}/")
        sys.exit(0)

    # ================================================================
    # MODE 3: Launch all SOCs (prints commands for multi-GPU)
    # ================================================================
    print("=" * 70)
    print("MULTI-GPU MULTI-SOC LAUNCH")
    print("=" * 70)
    print()
    print("Run these 3 commands in separate terminals:")
    print()

    gpu_map = {0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
    cells_str = ' '.join(args.cells)

    for i, soc in enumerate(args.socs):
        gpu = gpu_map.get(i, f'cuda:{i}')
        cmd = (f"python run_stanford_3d_warmstart.py "
               f"--data_dir {args.data_dir} "
               f"--cells {cells_str} "
               f"--soc {soc} "
               f"--device {gpu} "
               f"--out_dir {args.out_dir} "
               f"--plot")
        print(f"  # GPU {i}: SOC {soc}%")
        print(f"  {cmd}")
        print()

    print("After all finish, merge with:")
    socs_str = ' '.join(str(s) for s in args.socs)
    print(f"  python run_stanford_3d.py --merge_only --socs {socs_str} --out_dir {args.out_dir} --plot")
    print()
    print("Or run all sequentially on one GPU:")
    print(f"  for soc in {' '.join(str(s) for s in args.socs)}; do")
    print(f"    python run_stanford_3d.py --data_dir {args.data_dir} --cells {cells_str} --soc $soc --device cuda:0 --out_dir {args.out_dir} --plot")
    print(f"  done")
    print(f"  python run_stanford_3d.py --merge_only --socs {socs_str} --out_dir {args.out_dir} --plot")
