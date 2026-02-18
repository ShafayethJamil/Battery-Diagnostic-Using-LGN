"""
TRI Aging Matrix: 3D LGN with Warm-Start
==========================================
Runs LGN-SD (n_states=3) on 40s HPPC relaxation traces from 19-50 TRI cells.
Warm-start from previous diagnostic prevents collapse in sequential measurements.

Key differences from Stanford:
  - 40s window (not 3600s) â†’ Ï„â‚ƒ max ~20s (diffusion), not 500s
  - 2.4 Hz sampling â†’ Ï„â‚ min ~0.2s
  - No EIS â†’ validate against capacity degradation
  - 9 SOC levels per diagnostic

Expected time constants (from curve-fit baseline):
  Ï„â‚ ~ 0.05-0.5s  (charge transfer â€” near sampling limit)
  Ï„â‚‚ ~ 1-3s       (SEI layer impedance)
  Ï„â‚ƒ ~ 12-22s     (solid-state diffusion)

Usage:
  # Single SOC on one GPU:
  python run_tri_3d_warmstart.py --data ALL_rest_traces.csv --soc 4 --device cuda:0

  # Multi-GPU launch (prints commands):
  python run_tri_3d_warmstart.py --data ALL_rest_traces.csv

  # After all SOCs finish:
  python run_tri_3d_warmstart.py --data ALL_rest_traces.csv --merge_only

Author: Shafayeth Jamil (USC ECE), February 2026
"""

import argparse, json, os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn


# ============================================================================
# LGN-SD MODEL (3D, self-contained)
# ============================================================================

class LGN_Battery_3D(nn.Module):
    """
    3-state LGN-SD for battery relaxation.
    A = S - D where S is skew-symmetric, D is positive diagonal.
    For diagonal-only mode: S=0, A = -D â†’ decoupled exponentials.
    
    Output: y(t) = c^T exp(At) x0
    """
    def __init__(self, n_states=3, diagonal=True):
        super().__init__()
        self.n = n_states
        self.diagonal = diagonal
        
        # D parameters: Ï„_i = 1/softplus(d_i)
        self.d_params = nn.Parameter(torch.zeros(n_states, dtype=torch.float64))
        
        # S parameters (off-diagonal coupling) â€” disabled in diagonal mode
        n_s = n_states * (n_states - 1) // 2
        self.s_params = nn.Parameter(torch.zeros(n_s, dtype=torch.float64))
        if diagonal:
            self.s_params.requires_grad_(False)
        
        # Initial state and output vector
        self.x0 = nn.Parameter(torch.ones(n_states, dtype=torch.float64) / n_states)
        self.c = nn.Parameter(torch.ones(n_states, dtype=torch.float64))
    
    def get_A(self):
        d = torch.nn.functional.softplus(self.d_params)
        D = torch.diag(d)
        
        if self.diagonal:
            return -D
        
        # Build skew-symmetric S
        S = torch.zeros(self.n, self.n, dtype=torch.float64, device=self.d_params.device)
        idx = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                S[i, j] = self.s_params[idx]
                S[j, i] = -self.s_params[idx]
                idx += 1
        return S - D
    
    def forward(self, t):
        if self.diagonal:
            # Analytical: y(t) = sum_i c_i * x0_i * exp(-softplus(d_i) * t)
            d = torch.nn.functional.softplus(self.d_params)       # (3,)
            decay = torch.exp(-d.unsqueeze(0) * t.unsqueeze(1))   # (N, 3)
            xt = decay * self.x0.unsqueeze(0)                      # (N, 3)
            return (xt * self.c.unsqueeze(0)).sum(dim=1)           # (N,)
        else:
            A = self.get_A()
            At = A.unsqueeze(0) * t.unsqueeze(1).unsqueeze(2)
            eAt = torch.matrix_exp(At)
            xt = eAt @ self.x0
            return xt @ self.c
    
    def get_time_constants(self):
        with torch.no_grad():
            A = self.get_A()
            eigs = torch.linalg.eigvals(A)
            taus = -1.0 / eigs.real
            return np.sort(taus.cpu().numpy())
    
    def get_diagonal_damping(self):
        with torch.no_grad():
            return torch.nn.functional.softplus(self.d_params).cpu().numpy()


# ============================================================================
# LOGARITHMIC SUBSAMPLING
# ============================================================================

def subsample_log(n, target):
    """Log-spaced subsampling: dense at start, sparse at end."""
    if n <= target:
        return np.arange(n)
    idx = np.unique(np.logspace(0, np.log10(n-1), target).astype(int))
    idx = np.clip(idx, 0, n-1)
    return np.unique(idx)


# ============================================================================
# 3D TRAINING WITH WARM-START
# ============================================================================

def train_lgn_3d(t_data, eta_data, n_epochs=2500, lr=0.01,
                 subsample=250, device='cpu', verbose=False,
                 prev_d_params=None, prev_x0=None, prev_c=None,
                 warm_only=False):
    """
    Multi-restart LGN training for 3-state model on 40s TRI relaxation.
    
    Warm-start: if prev_d_params provided, prepend as first initialization.
    warm_only: if True and prev_d_params exists, ONLY use warm-start (no diverse inits).
               This is ~10x faster and used after the first few diagnostics.
    
    Initialization strategy for 40s window:
      d =  2.0 â†’ Ï„ â‰ˆ 0.5s   (charge transfer)
      d =  0.0 â†’ Ï„ â‰ˆ 1.4s   (SEI)
      d = -1.0 â†’ Ï„ â‰ˆ 3.3s   (SEI / slow)
      d = -2.0 â†’ Ï„ â‰ˆ 8s     (slow SEI / fast diffusion)
      d = -2.5 â†’ Ï„ â‰ˆ 12s    (diffusion)
      d = -3.0 â†’ Ï„ â‰ˆ 20s    (diffusion)
      d = -3.5 â†’ Ï„ â‰ˆ 33s    (slow diffusion â€” near window limit)
    """
    n_states = 3
    
    idx = subsample_log(len(t_data), subsample)
    t_t = torch.tensor(t_data[idx], dtype=torch.float64, device=device)
    eta_t = torch.tensor(eta_data[idx], dtype=torch.float64, device=device)
    
    # Initializations: n_random_inits diverse + warm-start if available
    # Mirrors KIT approach: 10 random for first 3 diags, warm-start only after
    all_random = [
        # Core hypotheses spanning CT -> SEI -> diffusion (40s window)
        torch.tensor([2.0, -1.0, -3.0], dtype=torch.float64),    # tau ~ [0.5, 3, 20]
        torch.tensor([1.5, -0.5, -2.5], dtype=torch.float64),    # tau ~ [0.6, 2, 12]
        torch.tensor([3.0, -1.0, -3.5], dtype=torch.float64),    # tau ~ [0.3, 3, 33]
        torch.tensor([2.5, -1.5, -3.0], dtype=torch.float64),    # tau ~ [0.4, 5, 20]
        torch.tensor([1.0, 0.0, -2.5], dtype=torch.float64),     # tau ~ [0.8, 1.4, 12]
        # Extended range for robustness
        torch.tensor([2.0, -2.0, -3.5], dtype=torch.float64),    # tau ~ [0.5, 8, 33]
        torch.tensor([3.0, -0.5, -3.0], dtype=torch.float64),    # tau ~ [0.3, 2, 20]
        torch.tensor([1.0, -1.5, -3.0], dtype=torch.float64),    # tau ~ [0.8, 5, 20]
        torch.tensor([2.5, -0.5, -3.5], dtype=torch.float64),    # tau ~ [0.4, 2, 33]
        torch.tensor([1.5, -2.0, -2.5], dtype=torch.float64),    # tau ~ [0.6, 8, 12]
    ]
    n_ri = 0 if warm_only else len(all_random)
    inits = all_random[:n_ri]
    if prev_d_params is not None:
        inits.insert(0, prev_d_params.clone().cpu())
        if verbose:
            tau_prev = 1.0 / torch.nn.functional.softplus(prev_d_params.cpu()).numpy()
            label = "warm-only" if warm_only else f"warm-start + {n_ri} random"
            print(f"      [{label}: tau={np.round(np.sort(tau_prev),2)}]")
    
    best_model, best_loss = None, float('inf')
    
    for i_init, init_d in enumerate(inits):
        model = LGN_Battery_3D(n_states, diagonal=True).to(device)
        model.d_params.data = init_d.clone().to(device)
        
        # Warm-start x0 and c from previous diagnostic
        if i_init == 0 and prev_d_params is not None:
            if prev_x0 is not None:
                model.x0.data = prev_x0.clone().to(device)
            if prev_c is not None:
                model.c.data = prev_c.clone().to(device)
        else:
            model.x0.data.fill_(eta_data[0] / n_states)
            model.c.data.fill_(1.0)
        
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=200, factor=0.5, min_lr=1e-5)
        
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
        
        is_new_best = run_best_loss < best_loss
        if run_best_loss < best_loss:
            best_loss = run_best_loss
            model.load_state_dict(run_best_state)
            best_model = model
        
        if verbose:
            taus = model.get_time_constants()
            win_marker = ' â† BEST' if is_new_best else ''
            print(f"      init {i_init}: Ï„={np.round(taus,2)}  loss={run_best_loss:.4e}{win_marker}")
    
    best_model.eval()
    with torch.no_grad():
        eta_pred = best_model(t_t).cpu().numpy()
    nrmse = np.sqrt(np.mean((eta_pred - eta_data[idx])**2)) / (np.abs(eta_data).max() + 1e-12)
    
    if verbose:
        print(f"    BEST â†’ Ï„ = {np.round(best_model.get_time_constants(), 3)}  "
              f"NRMSE = {nrmse:.5f}")
    
    return best_model, nrmse, best_loss


# ============================================================================
# CURVE FIT BASELINE (3-exponential)
# ============================================================================

def fit_3exp(t, v):
    """Fit V(t) = V_inf - A1*exp(-t/Ï„1) - A2*exp(-t/Ï„2) - A3*exp(-t/Ï„3)"""
    v_start, v_end = v[0], v[-1]
    dv = v_end - v_start
    if dv < 1e-6:
        return None
    
    def model(t, V_inf, A1, tau1, A2, tau2, A3, tau3):
        return V_inf - A1*np.exp(-t/tau1) - A2*np.exp(-t/tau2) - A3*np.exp(-t/tau3)
    
    try:
        p0 = [v_end, dv*0.3, 0.5, dv*0.4, 5.0, dv*0.3, 15.0]
        bounds = ([v_start, 0, 0.05, 0, 0.3, 0, 1.0],
                  [v_end+0.01, dv*2, 10, dv*2, 50, dv*2, 200])
        popt, _ = curve_fit(model, t, v, p0=p0, bounds=bounds, maxfev=20000)
        resid = v - model(t, *popt)
        rmse = np.sqrt(np.mean(resid**2))
        
        # Sort by tau
        taus = [(popt[2], popt[1]), (popt[4], popt[3]), (popt[6], popt[5])]
        taus.sort(key=lambda x: x[0])
        return {
            'tau1': taus[0][0], 'tau2': taus[1][0], 'tau3': taus[2][0],
            'A1': taus[0][1], 'A2': taus[1][1], 'A3': taus[2][1],
            'V_inf': popt[0], 'rmse': rmse
        }
    except:
        return None


# ============================================================================
# PROCESS ONE CELL (all diagnostics with warm-start chain)
# ============================================================================

def process_cell(cell_data, cell_name, soc_target, device='cpu', verbose=True, n_epochs=2500):
    """
    Process all diagnostics for one cell at one SOC level.
    Uses warm-start chain: each diagnostic initializes from previous.
    """
    diags = sorted(cell_data['diag_num'].unique())
    
    if verbose:
        print(f"\n{'#'*60}")
        print(f"# CELL: {cell_name}  SOC: {soc_target}  ({len(diags)} diagnostics)")
        print(f"{'#'*60}")
    
    results = []
    prev_model = None  # warm-start chain
    diag_idx = 0       # counter for adaptive init strategy
    WARMUP_DIAGS = 3   # explore thoroughly for first 3, warm-only after
    
    for diag in diags:
        trace = cell_data[cell_data['diag_num'] == diag].sort_values('t_rel')
        
        if len(trace) < 20:
            continue
        
        t = trace['t_rel'].values
        v = trace['voltage'].values
        cap = trace['capacity'].iloc[0]
        
        # Compute overpotential: Î·(t) = V(t) - V_inf (negative, relaxing upward)
        # LGN convention: Î· = -(V - V_end) so Î· is positive and decaying
        v_end = v[-1]
        eta = -(v - v_end)  # positive, decaying to 0
        
        if eta[0] < 1e-6:
            continue
        
        if verbose:
            print(f"\n  Diag {diag} (cap={cap:.3f} Ah, {len(t)} pts, Î”V={1000*(v[-1]-v[0]):.1f} mV):")
        
        # --- LGN-3D fit ---
        # Adaptive inits: 10 random for diags 1-3, warm-start only for 4+
        use_warm_only = (diag_idx >= WARMUP_DIAGS) and (prev_model is not None)
        prev_d = prev_model.d_params.data if prev_model else None
        prev_x0 = prev_model.x0.data if prev_model else None
        prev_c = prev_model.c.data if prev_model else None
        
        model, nrmse, loss = train_lgn_3d(
            t, eta, n_epochs=n_epochs, lr=0.01,
            subsample=min(250, len(t)), device=device, verbose=verbose,
            prev_d_params=prev_d, prev_x0=prev_x0, prev_c=prev_c,
            warm_only=use_warm_only)
        
        taus = model.get_time_constants()
        prev_model = model  # update warm-start chain
        diag_idx += 1
        
        # --- Curve fit baseline ---
        cf = fit_3exp(t, v)
        
        # --- Store results ---
        res = {
            'cell': cell_name,
            'diag_num': int(diag),
            'soc_idx': int(soc_target),
            'capacity': float(cap),
            'soh': float(100 * cap / 4.84),  # TRI nominal capacity
            'n_pts': len(t),
            'delta_v_mV': float(1000 * (v[-1] - v[0])),
            'eta0': float(eta[0]),
            
            # LGN results
            'lgn_tau1': float(taus[0]),
            'lgn_tau2': float(taus[1]),
            'lgn_tau3': float(taus[2]),
            'lgn_nrmse': float(nrmse),
            'lgn_loss': float(loss),
            
            # Full model state (for reconstruction / warm-start)
            'd_params': model.d_params.data.cpu().numpy().tolist(),
            'x0': model.x0.data.cpu().numpy().tolist(),
            'c_params': model.c.data.cpu().numpy().tolist(),
            'A_matrix': model.get_A().detach().cpu().numpy().tolist(),
            'damping': model.get_diagonal_damping().tolist(),
        }
        
        # Curve fit results
        if cf:
            res['cf_tau1'] = cf['tau1']
            res['cf_tau2'] = cf['tau2']
            res['cf_tau3'] = cf['tau3']
            res['cf_rmse'] = cf['rmse']
        
        results.append(res)
        
        if verbose:
            cf_str = f"CF Ï„=[{cf['tau1']:.2f}, {cf['tau2']:.2f}, {cf['tau3']:.2f}]" if cf else "CF: FAILED"
            print(f"    LGN Ï„ = [{taus[0]:.3f}, {taus[1]:.3f}, {taus[2]:.3f}]  NRMSE={nrmse:.5f}")
            print(f"    {cf_str}")
    
    return results


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def correlation_analysis(results, label=""):
    """Compute per-cell and global correlations of Ï„ vs capacity."""
    df = pd.DataFrame(results)
    if len(df) < 5:
        print("Too few results for correlation analysis")
        return {}
    
    print(f"\n{'='*70}")
    print(f"DEGRADATION CORRELATION ANALYSIS {label}")
    print(f"{'='*70}")
    
    cells = sorted(df['cell'].unique())
    corr_results = {}
    
    # Global correlations
    print(f"\n--- Global correlations ({len(df)} data points) ---")
    for col in ['lgn_tau1', 'lgn_tau2', 'lgn_tau3', 'cf_tau1', 'cf_tau2', 'cf_tau3']:
        valid = df[[col, 'capacity']].dropna()
        if len(valid) > 5:
            rho, p = stats.spearmanr(valid[col], valid['capacity'])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {col:12s} vs capacity: Ï = {rho:+.3f}{sig}  (p={p:.2e})")
            corr_results[f'global_{col}'] = {'rho': float(rho), 'p': float(p)}
    
    # Per-cell correlations
    print(f"\n--- Per-cell correlations ---")
    for col in ['lgn_tau1', 'lgn_tau2', 'lgn_tau3', 'cf_tau3']:
        per_cell_rho = []
        for cell in cells:
            c = df[df['cell'] == cell][[col, 'capacity']].dropna()
            if len(c) >= 5:
                rho, p = stats.spearmanr(c[col], c['capacity'])
                if not np.isnan(rho):
                    per_cell_rho.append({'cell': cell, 'rho': rho, 'p': p, 'n': len(c)})
        
        if per_cell_rho:
            rhos = [x['rho'] for x in per_cell_rho]
            print(f"\n  {col}:")
            print(f"    Median |Ï| = {np.median(np.abs(rhos)):.3f}  "
                  f"(n={len(per_cell_rho)} cells)")
            for x in sorted(per_cell_rho, key=lambda x: -abs(x['rho']))[:5]:
                sig = '***' if x['p'] < 0.001 else '**' if x['p'] < 0.01 else '*' if x['p'] < 0.05 else ''
                print(f"    {x['cell'][:30]:30s}: Ï={x['rho']:+.3f}{sig} (n={x['n']})")
            
            corr_results[f'percell_{col}'] = {
                'median_abs_rho': float(np.median(np.abs(rhos))),
                'mean_abs_rho': float(np.mean(np.abs(rhos))),
                'n_cells': len(per_cell_rho),
                'details': per_cell_rho,
            }
    
    return corr_results


# ============================================================================
# MERGE SOC RESULTS
# ============================================================================

def merge_soc_results(out_dir, socs):
    """Merge per-SOC result files into one."""
    all_results = []
    all_corrs = {}
    for soc in socs:
        rpath = f'{out_dir}/soc{soc}/results_3d.json'
        cpath = f'{out_dir}/soc{soc}/correlations_3d.json'
        if os.path.exists(rpath):
            with open(rpath) as f:
                results = json.load(f)
            all_results.extend(results)
            print(f"  SOC {soc}: {len(results)} results loaded")
        if os.path.exists(cpath):
            with open(cpath) as f:
                corrs = json.load(f)
            all_corrs[f'soc{soc}'] = corrs
    
    print(f"\nTotal: {len(all_results)} results across {len(set(r['soc_idx'] for r in all_results))} SOC levels")
    return all_results, all_corrs


# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(results, out_dir):
    """Publication-quality figures for TRI LGN results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
    
    df = pd.DataFrame(results)
    cells = sorted(df['cell'].unique())
    n_cells = len(cells)
    
    # Color map
    cmap = plt.cm.tab20
    cell_colors = {c: cmap(i / max(n_cells-1, 1)) for i, c in enumerate(cells)}
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f'TRI Aging Matrix: LGN-3D Time Constant Extraction\n'
                 f'{n_cells} NCA 21700 cells, {len(df)} total fits',
                 fontsize=13, fontweight='bold', y=0.98)
    
    # (a) Ï„ trajectories over aging
    ax = axes[0, 0]
    for cell in cells:
        c = df[df['cell'] == cell].sort_values('diag_num')
        ax.plot(c['soh'], c['lgn_tau3'], '-', color=cell_colors[cell], 
                lw=1, alpha=0.7, markersize=3)
    ax.set_xlabel('SOH [%]')
    ax.set_ylabel('Ï„â‚ƒ (diffusion) [s]')
    ax.set_title('(a) Ï„â‚ƒ trajectories vs SOH')
    ax.grid(True, alpha=0.3)
    
    # (b) Ï„â‚‚ trajectories
    ax = axes[0, 1]
    for cell in cells:
        c = df[df['cell'] == cell].sort_values('diag_num')
        ax.plot(c['soh'], c['lgn_tau2'], '-', color=cell_colors[cell],
                lw=1, alpha=0.7, markersize=3)
    ax.set_xlabel('SOH [%]')
    ax.set_ylabel('Ï„â‚‚ (SEI) [s]')
    ax.set_title('(b) Ï„â‚‚ trajectories vs SOH')
    ax.grid(True, alpha=0.3)
    
    # (c) Global scatter: Ï„â‚ƒ vs capacity
    ax = axes[0, 2]
    for cell in cells:
        c = df[df['cell'] == cell]
        ax.scatter(c['capacity'], c['lgn_tau3'], s=15, color=cell_colors[cell], alpha=0.5)
    rho, p = stats.spearmanr(df['lgn_tau3'].dropna(), df.loc[df['lgn_tau3'].notna(), 'capacity'])
    ax.set_xlabel('Capacity [Ah]')
    ax.set_ylabel('Ï„â‚ƒ [s]')
    ax.set_title(f'(c) Ï„â‚ƒ vs Capacity â€” Ï = {rho:.3f}')
    ax.grid(True, alpha=0.3)
    
    # (d) LGN vs curve fit comparison
    ax = axes[1, 0]
    valid = df.dropna(subset=['lgn_tau3', 'cf_tau3'])
    ax.scatter(valid['cf_tau3'], valid['lgn_tau3'], s=15, alpha=0.5, color='steelblue')
    lims = [min(valid['cf_tau3'].min(), valid['lgn_tau3'].min()),
            max(valid['cf_tau3'].max(), valid['lgn_tau3'].max())]
    ax.plot(lims, lims, 'k--', alpha=0.3)
    rho, _ = stats.spearmanr(valid['cf_tau3'], valid['lgn_tau3'])
    ax.set_xlabel('Curve Fit Ï„â‚ƒ [s]')
    ax.set_ylabel('LGN Ï„â‚ƒ [s]')
    ax.set_title(f'(d) LGN vs Curve Fit Ï„â‚ƒ â€” Ï = {rho:.3f}')
    ax.grid(True, alpha=0.3)
    
    # (e) NRMSE distribution
    ax = axes[1, 1]
    ax.hist(df['lgn_nrmse'].dropna() * 100, bins=30, color='steelblue', edgecolor='white')
    ax.set_xlabel('NRMSE [%]')
    ax.set_ylabel('Count')
    med = df['lgn_nrmse'].median() * 100
    ax.axvline(med, color='red', ls='--', lw=2, label=f'Median: {med:.2f}%')
    ax.set_title(f'(e) LGN Fit Quality')
    ax.legend()
    
    # (f) Correlation comparison bar chart
    ax = axes[1, 2]
    methods = {}
    for col, label in [('lgn_tau1', 'LGN Ï„â‚'), ('lgn_tau2', 'LGN Ï„â‚‚'), 
                        ('lgn_tau3', 'LGN Ï„â‚ƒ'), ('cf_tau3', 'CF Ï„â‚ƒ')]:
        per_cell_rho = []
        for cell in cells:
            c = df[df['cell'] == cell][[col, 'capacity']].dropna()
            if len(c) >= 5:
                r, _ = stats.spearmanr(c[col], c['capacity'])
                if not np.isnan(r):
                    per_cell_rho.append(abs(r))
        if per_cell_rho:
            methods[label] = np.median(per_cell_rho)
    
    colors = ['#90CAF9', '#42A5F5', '#1565C0', '#EF5350']
    bars = ax.bar(range(len(methods)), list(methods.values()), color=colors[:len(methods)])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods.keys())
    ax.set_ylabel('Median per-cell |Ï| with capacity')
    ax.set_title('(f) Degradation correlation comparison')
    ax.set_ylim(0, 1.05)
    for bar, v in zip(bars, methods.values()):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}',
                ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f'{out_dir}/tri_lgn_3d_results.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {out_dir}/tri_lgn_3d_results.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    p = argparse.ArgumentParser(description='TRI Aging Matrix 3D LGN Pipeline')
    p.add_argument('--data', required=True, help='Path to ALL_rest_traces.csv')
    p.add_argument('--soc', type=int, default=None,
                   help='Single SOC index to run (0-8). If not set, prints multi-GPU commands.')
    p.add_argument('--socs', nargs='+', type=int, default=[2, 4, 7],
                   help='SOC indices: 2â‰ˆ80%%, 4â‰ˆ50%%, 7â‰ˆ20%% (matches Stanford)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out_dir', default='results_tri_3d')
    p.add_argument('--plot', action='store_true')
    p.add_argument('--merge_only', action='store_true')
    p.add_argument('--max_cells', type=int, default=None, help='Limit cells for testing')
    p.add_argument('--cell_start', type=int, default=None, help='Start cell index (for GPU splitting)')
    p.add_argument('--cell_end', type=int, default=None, help='End cell index exclusive (for GPU splitting)')
    p.add_argument('--n_epochs', type=int, default=2500)
    args = p.parse_args()
    
    # ================================================================
    # MODE 1: Merge existing per-SOC results
    # ================================================================
    if args.merge_only:
        print("Merging per-SOC results...")
        all_results, all_corrs = merge_soc_results(args.out_dir, args.socs)
        if all_results:
            corr = correlation_analysis(all_results, label="(ALL SOCs)")
            if args.plot:
                plot_results(all_results, args.out_dir)
            with open(f'{args.out_dir}/results_all_socs.json', 'w') as f:
                json.dump(all_results, f, indent=2)
            with open(f'{args.out_dir}/correlations_all_socs.json', 'w') as f:
                json.dump(corr, f, indent=2, default=str)
            print(f"\nâœ“ Merged {len(all_results)} results â†’ {args.out_dir}/")
        sys.exit(0)
    
    # ================================================================
    # MODE 2: Single SOC run
    # ================================================================
    if args.soc is not None:
        soc = args.soc
        
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            print("CUDA not available â†’ CPU")
            args.device = 'cpu'
        
        print(f"{'='*70}")
        print(f"TRI AGING MATRIX â€” 3D LGN WITH WARM-START")
        print(f"{'='*70}")
        print(f"Device: {args.device}")
        print(f"SOC index: {soc}")
        print(f"Model: 3-state LGN-SD (CT + SEI + diffusion)")
        print(f"Epochs: {args.n_epochs}")
        
        # Load data
        df = pd.read_csv(args.data)
        df_soc = df[df['soc_idx'] == soc]
        cells = sorted(df_soc['cell'].unique())
        
        if args.max_cells:
            cells = cells[:args.max_cells]
        
        # GPU splitting: select cell range
        if args.cell_start is not None or args.cell_end is not None:
            cs = args.cell_start or 0
            ce = args.cell_end or len(cells)
            cells = cells[cs:ce]
            print(f"Cell range: [{cs}:{ce}] → {len(cells)} cells")
        
        print(f"Cells: {len(cells)}")
        print(f"Total traces: {len(df_soc)}")
        
        # Process each cell
        all_results = []
        t_start = time.time()
        
        for ci, cell in enumerate(cells):
            cell_data = df_soc[df_soc['cell'] == cell]
            results = process_cell(cell_data, cell, soc, 
                                   device=args.device, verbose=True,
                                   n_epochs=args.n_epochs)
            all_results.extend(results)
            
            elapsed = time.time() - t_start
            rate = (ci + 1) / elapsed * 60
            remaining = (len(cells) - ci - 1) / rate if rate > 0 else 0
            print(f"\n  [{ci+1}/{len(cells)}] Done. {rate:.1f} cells/min, ~{remaining:.0f} min remaining")
        
        # Analysis
        corr = correlation_analysis(all_results, label=f"(SOC {soc})")
        
        # Save
        soc_out = f'{args.out_dir}/soc{soc}'
        os.makedirs(soc_out, exist_ok=True)
        
        # Suffix for GPU-split runs
        range_tag = ''
        if args.cell_start is not None or args.cell_end is not None:
            cs = args.cell_start or 0
            ce = args.cell_end or 999
            range_tag = f'_c{cs}-{ce}'
        
        # Serialize
        clean = []
        for r in all_results:
            cr = {}
            for k, v in r.items():
                if isinstance(v, (np.floating, np.integer)):
                    cr[k] = float(v)
                elif isinstance(v, np.ndarray):
                    cr[k] = v.tolist()
                else:
                    cr[k] = v
            clean.append(cr)
        
        with open(f'{soc_out}/results_3d{range_tag}.json', 'w') as f:
            json.dump(clean, f, indent=2)
        with open(f'{soc_out}/correlations_3d{range_tag}.json', 'w') as f:
            json.dump(corr, f, indent=2, default=str)
        
        if args.plot:
            plot_results(all_results, soc_out)
        
        elapsed = time.time() - t_start
        print(f"\n{'='*70}")
        print(f"âœ“ SOC {soc} complete: {len(all_results)} fits in {elapsed/60:.1f} min")
        print(f"  Saved to {soc_out}/")
        print(f"{'='*70}")
        sys.exit(0)
    
    # ================================================================
    # MODE 3: Print multi-GPU launch commands
    # ================================================================
    print("=" * 70)
    print("TRI 3D LGN â€” MULTI-GPU LAUNCH COMMANDS")
    print("=" * 70)
    print()
    
    gpu_map = {0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
    
    for i, soc in enumerate(args.socs):
        gpu = gpu_map.get(i % 4, f'cuda:{i % 4}')
        cmd = (f"python run_tri_3d_warmstart.py "
               f"--data {args.data} "
               f"--soc {soc} "
               f"--device {gpu} "
               f"--out_dir {args.out_dir} "
               f"--n_epochs {args.n_epochs} "
               f"--plot")
        print(f"  # GPU {i % 4}: SOC index {soc}")
        print(f"  {cmd}")
        print()
    
    print("After all finish, merge with:")
    socs_str = ' '.join(str(s) for s in args.socs)
    print(f"  python run_tri_3d_warmstart.py --data {args.data} --merge_only --socs {socs_str} --out_dir {args.out_dir} --plot")
    print()
    print("Or run sequentially on one GPU:")
    print(f"  for soc in {socs_str}; do")
    print(f"    python run_tri_3d_warmstart.py --data {args.data} --soc $soc --device cuda:0 --out_dir {args.out_dir} --plot")
    print(f"  done")
    print(f"  python run_tri_3d_warmstart.py --data {args.data} --merge_only --socs {socs_str} --out_dir {args.out_dir} --plot")


if __name__ == '__main__':
    main()
