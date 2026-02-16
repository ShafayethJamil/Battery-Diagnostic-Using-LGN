"""
LGN Battery: Degradation Tracking via Eigenvalue Trajectories (v3)
===================================================================
Fixes from code review:
  1. Window re-referencing (was computed but never applied)
  2. Impedance reconstruction via proper state-space transfer function
     Z_shape(jω) = Cᵀ(jωI - A)⁻¹B, then fit scale+Rs via least squares
  3. Complex least-squares scaling (not fragile single-point)
  4. Safe correlation guards
  5. No unnecessary eigendecomposition

Run:
  python run_degradation.py --cells W8 W9 W10 --device cuda
  python run_degradation.py --cells W8 W9 W10 V4 --n_states 3 --device cuda

Author: Shafayeth Jamil (USC ECE), February 2026
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

import torch
import torch.nn as nn


# ============================================================================
# 1. LGN MODEL (self-contained)
# ============================================================================
class LGN_Battery(nn.Module):
    """A = S - D, forward: η(t) = 1ᵀ exp(At) x₀"""
    def __init__(self, n_states=2):
        super().__init__()
        self.n = n_states
        n_skew = n_states * (n_states - 1) // 2
        self.s_params = nn.Parameter(torch.zeros(n_skew, dtype=torch.float64))
        self.d_params = nn.Parameter(torch.linspace(1.0, -3.0, n_states, dtype=torch.float64))
        self.x0 = nn.Parameter(torch.ones(n_states, dtype=torch.float64) * -0.01)

    def get_A(self):
        S = torch.zeros(self.n, self.n, dtype=torch.float64, device=self.s_params.device)
        idx = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                S[i, j] = self.s_params[idx]
                S[j, i] = -self.s_params[idx]
                idx += 1
        D = torch.diag(nn.functional.softplus(self.d_params))
        return S - D

    def get_A_numpy(self):
        return self.get_A().detach().cpu().numpy()

    def get_time_constants(self, eps=1e-8):
        """τᵢ = -1/Re(λᵢ). Guarded against near-zero eigenvalues."""
        A = self.get_A_numpy()
        eigs = np.linalg.eigvals(A)
        re = eigs.real
        re = np.where(re < -eps, re, -eps)  # clamp near-zero modes
        taus = -1.0 / re
        return np.sort(taus)

    def get_diagonal_damping(self):
        """Diagonal of D (softplus of d_params). NOT mode decay rates when S ≠ 0."""
        d = nn.functional.softplus(self.d_params).detach().cpu().numpy()
        return np.sort(d)

    def get_mode_decay_rates(self):
        """True mode decay rates: -Re(λᵢ) from eigenvalues of A."""
        A = self.get_A_numpy()
        eigs = np.linalg.eigvals(A)
        rates = -eigs.real
        return np.sort(rates)

    def forward(self, t_vec):
        A = self.get_A()
        C = torch.ones(self.n, dtype=torch.float64, device=A.device)
        At = A.unsqueeze(0) * t_vec.unsqueeze(1).unsqueeze(2)  # (T,n,n)
        eAt = torch.matrix_exp(At)
        x_t = torch.einsum('tij,j->ti', eAt, self.x0)
        return x_t @ C


# ============================================================================
# 2. TRAINING WITH MULTI-RESTART
# ============================================================================
def _subsample_log(n_total, n_target):
    dense = np.arange(min(30, n_total))
    sparse = np.geomspace(1, n_total - 1, max(n_target - 30, 10)).astype(int)
    return np.unique(np.concatenate([dense, sparse]))


def train_lgn(t_data, eta_data, n_states=2, n_epochs=3000, lr=0.01,
              subsample=250, device='cpu', verbose=False):
    """Multi-restart LGN training. Returns best model."""

    idx = _subsample_log(len(t_data), subsample) if subsample else np.arange(len(t_data))
    t_t = torch.tensor(t_data[idx], dtype=torch.float64, device=device)
    eta_t = torch.tensor(eta_data[idx], dtype=torch.float64, device=device)

    # Multiple D initializations covering plausible battery τ ranges
    if n_states == 2:
        inits = [
            torch.tensor([0.5, -2.5]),   # τ ~ [1.3, 12]
            torch.tensor([1.5, -1.5]),   # τ ~ [0.5, 4.7]
            torch.tensor([0.0, -3.5]),   # τ ~ [1.0, 33]
            torch.tensor([-0.5, -4.0]),  # τ ~ [1.6, 55]
        ]
    elif n_states == 3:
        inits = [
            torch.tensor([2.0, 0.0, -3.0]),
            torch.tensor([1.0, -1.0, -4.0]),
            torch.tensor([3.0, 0.5, -2.5]),
        ]
    else:
        inits = [torch.linspace(2.0, -4.0, n_states)]

    best_model, best_loss = None, float('inf')

    for init_d in inits:
        model = LGN_Battery(n_states).double().to(device)
        model.s_params.requires_grad_(False)  # freeze S: diagonal A, no identifiability issues
        model.d_params.data = init_d.clone().to(device)
        model.x0.data.fill_(eta_data[0] / n_states)

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

        if run_best_loss < best_loss:
            best_loss = run_best_loss
            model.load_state_dict(run_best_state)
            best_model = model

    best_model.eval()
    with torch.no_grad():
        eta_pred = best_model(t_t).cpu().numpy()
    nrmse = np.sqrt(np.mean((eta_pred - eta_data[idx])**2)) / (np.abs(eta_data).max() + 1e-12)

    if verbose:
        print(f"    τ = {np.round(best_model.get_time_constants(), 3)}  "
              f"NRMSE = {nrmse:.5f}  loss = {best_loss:.4e}")

    return best_model, nrmse, best_loss


# ============================================================================
# 3. IMPEDANCE RECONSTRUCTION (FIX #2: proper state-space transfer function)
# ============================================================================
def lgn_to_impedance_shape(model, freq):
    """Z_shape(jω) = Cᵀ (jωI - A)⁻¹ B with C=B=1.

    This gives the *shape* of the impedance. Absolute scale and Rs
    are fitted separately against EIS data.
    """
    A = model.get_A_numpy()
    n = A.shape[0]
    omega = 2 * np.pi * freq
    I_n = np.eye(n)
    C = np.ones((1, n))
    B = np.ones((n, 1))

    Z = np.zeros(len(freq), dtype=complex)
    for k, w in enumerate(omega):
        Z[k] = (C @ np.linalg.solve(1j * w * I_n - A, B)).item()
    return Z


def fit_scale_and_Rs(Z_shape, Z_data):
    """Fit Z_data ≈ a * Z_shape + Rs  (a complex, Rs real).

    Full complex least-squares: minimizes ||Z_pred - Z_data||² over
    both real and imaginary parts simultaneously.
    """
    zs, zd = Z_shape, Z_data

    # Stack real + imag equations:
    #   Re(zd) = ar*Re(zs) - ai*Im(zs) + Rs
    #   Im(zd) = ar*Im(zs) + ai*Re(zs) + 0
    X_re = np.column_stack([zs.real, -zs.imag, np.ones(len(zs))])
    X_im = np.column_stack([zs.imag,  zs.real, np.zeros(len(zs))])
    X = np.vstack([X_re, X_im])
    y = np.concatenate([zd.real, zd.imag])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    ar, ai, Rs = beta
    a = ar + 1j * ai
    Rs = max(Rs, 0.0)  # physical: Rs >= 0

    Z_pred = a * zs + Rs
    return a, float(Rs), Z_pred


def safe_corr(a, b):
    """Pearson correlation with NaN guard (fix #4)."""
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def compare_impedance(Z_pred, Z_data, freq, label=''):
    """Metrics between predicted and measured impedance."""
    rmse = np.sqrt(np.mean(np.abs(Z_pred - Z_data)**2))
    nrmse = rmse / (np.abs(Z_data).max() + 1e-15)
    re_corr = safe_corr(Z_pred.real, Z_data.real)
    im_corr = safe_corr(-Z_pred.imag, -Z_data.imag)

    if label:
        print(f"    {label}: Re corr={re_corr:.4f}  Im corr={im_corr:.4f}  "
              f"NRMSE={nrmse:.4f}  RMSE={rmse*1000:.3f} mΩ")

    return {'re_corr': re_corr, 'im_corr': im_corr,
            'nrmse': nrmse, 'rmse_mOhm': rmse * 1000}


# ============================================================================
# 4. CURVE FIT BASELINE (fixed bounds for negative overpotentials)
# ============================================================================
def fit_exponentials(t_data, eta_data, n_exp=2):
    """Sum-of-exponentials: η(t) = Σ aᵢ exp(-t/τᵢ)."""
    eta_max = np.abs(eta_data).max()
    if eta_max < 1e-10:
        return None, None, None

    def model(t, *p):
        out = np.zeros_like(t)
        for i in range(n_exp):
            out += p[2*i] * np.exp(-t / (p[2*i+1] + 1e-10))
        return out

    t_max = t_data[-1]
    a0 = eta_data[0] / n_exp

    if n_exp == 2:
        starts = [
            [a0*0.6, 5.0,  a0*0.4, 200.0],
            [a0*0.4, 10.0, a0*0.6, 500.0],
            [a0*0.7, 2.0,  a0*0.3, 100.0],
            [a0*0.3, 20.0, a0*0.7, 300.0],
        ]
    elif n_exp == 3:
        starts = [
            [a0, 2.0, a0, 30.0, a0, 300.0],
            [a0, 5.0, a0, 50.0, a0, 500.0],
        ]
    else:
        return None, None, None

    lo = [-eta_max*3] * (2*n_exp)
    hi = [ eta_max*3] * (2*n_exp)
    for i in range(n_exp):
        lo[2*i+1] = 0.01   # τ > 0
        hi[2*i+1] = t_max * 3

    best_popt, best_rmse = None, float('inf')
    for p0 in starts:
        try:
            popt, _ = curve_fit(model, t_data, eta_data, p0=p0,
                                bounds=(lo, hi), maxfev=100000, method='trf')
            rmse = np.sqrt(np.mean((model(t_data, *popt) - eta_data)**2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_popt = popt
        except Exception:
            continue

    if best_popt is None:
        return None, None, None

    taus = np.sort([best_popt[2*i+1] for i in range(n_exp)])
    nrmse = best_rmse / (eta_max + 1e-12)
    return taus, nrmse, best_rmse


# ============================================================================
# 5. EIS FITTING
# ============================================================================
def fit_eis_randles(freq, re_z, im_z, n_rc=2):
    """Multi-start Randles circuit fit."""
    from scipy.optimize import minimize

    Z_data = re_z + 1j * im_z
    Rs_est = re_z.min()
    dR = re_z.max() - re_z.min()

    def obj(p):
        Z = p[0] * np.ones(len(freq), dtype=complex)
        omega = 2 * np.pi * freq
        for i in range(n_rc):
            R, tau = p[1 + 2*i], p[2 + 2*i]
            Z += R / (1 + 1j * omega * tau)
        return np.sum(np.abs(Z - Z_data)**2)

    if n_rc == 2:
        starts = [[Rs_est, dR*f1, t1, dR*(1-f1), t2]
                   for f1 in [0.2, 0.4, 0.6]
                   for t1 in [0.05, 0.3, 1.0]
                   for t2 in [5.0, 30.0, 100.0]]
        bounds = [(0,None),(0,None),(1e-4,1e3),(0,None),(1e-4,1e3)]
    elif n_rc == 3:
        starts = [[Rs_est, dR*0.15, t1, dR*0.35, t2, dR*0.5, t3]
                   for t1 in [0.01, 0.05]
                   for t2 in [0.5, 5.0]
                   for t3 in [50, 200]]
        bounds = [(0,None),(0,None),(1e-5,1e3),(0,None),(1e-4,1e3),(0,None),(1e-3,1e4)]
    else:
        return None, None

    best, best_cost = None, float('inf')
    for p0 in starts:
        try:
            r = minimize(obj, p0, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 10000, 'ftol': 1e-15})
            if r.fun < best_cost:
                best_cost = r.fun
                best = r.x
        except:
            continue

    if best is None:
        return None, None

    params = {'Rs': best[0]}
    for i in range(n_rc):
        params[f'R{i+1}'] = best[1 + 2*i]
        params[f'tau{i+1}'] = best[2 + 2*i]

    Z_fit = best[0] * np.ones(len(freq), dtype=complex)
    omega = 2 * np.pi * freq
    for i in range(n_rc):
        Z_fit += best[1+2*i] / (1 + 1j * omega * best[2+2*i])

    return params, Z_fit


# ============================================================================
# 6. SINGLE DIAGNOSTIC ANALYSIS
# ============================================================================
def analyze_diagnostic(t_relax, v_relax, freq, re_z, im_z,
                       n_states=2, cell='', diag=0, soc=0, device='cpu'):
    """Analyze one diagnostic checkpoint. Returns dict of all metrics."""

    res = {'cell': cell, 'diag': diag, 'soc': soc}

    # ---- HPPC-derived features ----
    v_inf = v_relax[-1]
    eta_full = v_relax - v_inf
    res['eta0'] = float(eta_full[0])                         # initial overpotential
    res['R_pulse'] = float(abs(eta_full[0]) / 4.85)          # η₀/I_pulse (assuming 4.85A)
    res['v_end'] = float(v_inf)

    # ---- LGN on multiple windows ----
    windows = {'full': len(t_relax), 'w300': 300, 'w100': 100}
    for wname, wlen in windows.items():
        if isinstance(wlen, int) and wlen < len(t_relax):
            mask = t_relax <= wlen
            t_w = t_relax[mask]
            v_w = v_relax[mask]
            eta_w = v_w - v_w[-1]  # FIX #1: re-reference to END OF THIS WINDOW
        else:
            t_w = t_relax
            eta_w = eta_full

        model, nrmse, loss = train_lgn(
            t_w, eta_w, n_states=n_states, n_epochs=3000,
            lr=0.01, subsample=min(250, len(t_w)), device=device, verbose=False)

        taus = model.get_time_constants()
        d_rates = model.get_diagonal_damping()
        A = model.get_A_numpy()

        res[f'tau_{wname}'] = taus.tolist()
        res[f'd_{wname}'] = d_rates.tolist()
        res[f'decay_{wname}'] = model.get_mode_decay_rates().tolist()
        res[f'nrmse_{wname}'] = nrmse
        res[f'A_{wname}'] = A.tolist()

        # Impedance reconstruction (state-space transfer function)
        Z_shape = lgn_to_impedance_shape(model, freq)
        Z_data = re_z + 1j * im_z
        # Band mask: exclude inductive high-freq and unreliable low-freq tails
        band = (freq >= 0.05) & (freq <= 2000)
        a_fit, Rs_fit, Z_pred = fit_scale_and_Rs(Z_shape[band], Z_data[band])
        Z_pred_full = a_fit * Z_shape + Rs_fit  # predict on all freqs for plotting
        comp = compare_impedance(Z_pred_full[band], Z_data[band], freq[band])
        res[f'z_re_corr_{wname}'] = comp['re_corr']
        res[f'z_im_corr_{wname}'] = comp['im_corr']
        res[f'z_nrmse_{wname}'] = comp['nrmse']
        res[f'Rs_fit_{wname}'] = float(Rs_fit)

    # ---- Curve fit baseline ----
    cf_taus, cf_nrmse, _ = fit_exponentials(t_relax, eta_full, n_exp=n_states)
    res['tau_cf'] = cf_taus.tolist() if cf_taus is not None else None
    res['nrmse_cf'] = cf_nrmse

    # ---- EIS ground truth ----
    eis_params, Z_fit_eis = fit_eis_randles(freq, re_z, im_z, n_rc=n_states)
    if eis_params:
        res['tau_eis'] = sorted([eis_params[f'tau{i+1}'] for i in range(n_states)])
        res['Rs_eis'] = eis_params['Rs']
        for i in range(n_states):
            res[f'R{i+1}_eis'] = eis_params[f'R{i+1}']
    else:
        res['tau_eis'] = None

    # ---- EIS scalar features (model-free, for correlation) ----
    Z_data = re_z + 1j * im_z
    for f_target, label in [(1000, 'Z_1kHz'), (100, 'Z_100Hz'),
                             (10, 'Z_10Hz'), (1, 'Z_1Hz'), (0.1, 'Z_01Hz')]:
        fi = np.argmin(np.abs(freq - f_target))
        res[f'{label}_re'] = float(re_z[fi])
        res[f'{label}_im'] = float(im_z[fi])
        res[f'{label}_mag'] = float(np.abs(Z_data[fi]))

    return res


# ============================================================================
# 7. FULL PIPELINE
# ============================================================================
def run_degradation(data_dir, cells, soc_target=50, n_states=2, device='cpu'):
    all_results = []

    for cell in cells:
        print(f"\n{'#'*70}\n# CELL: {cell}\n{'#'*70}")
        hppc = pd.read_csv(f'{data_dir}/{cell}_hppc_relaxation.csv')
        eis = pd.read_csv(f'{data_dir}/{cell}_eis.csv')
        diags = sorted(set(hppc['diag'].unique()) & set(eis['diag'].unique()))

        for diag in diags:
            seg = hppc[(hppc['diag'] == diag) & (hppc['soc_pct'] == soc_target)]
            if len(seg) == 0:
                continue
            eis_seg = eis[eis['diag'] == diag]

            print(f"\n  Diag {diag} ...", end='', flush=True)
            r = analyze_diagnostic(
                seg['time_s'].values, seg['voltage_V'].values,
                eis_seg['freq_Hz'].values,
                eis_seg['re_z_ohm_50pct'].values,
                eis_seg['im_z_ohm_50pct'].values,
                n_states=n_states, cell=cell, diag=diag,
                soc=soc_target, device=device)
            all_results.append(r)

            # Live progress
            taus = r['tau_full']
            z_corr = r.get('z_re_corr_full', 0)
            print(f" τ={[round(t,1) for t in taus]}  Z_corr={z_corr:.3f}")

    return all_results


# ============================================================================
# 8. CORRELATION ANALYSIS (THE MONEY METRIC)
# ============================================================================
def correlation_analysis(results):
    """Compute correlations between LGN features and EIS degradation markers."""

    cells = sorted(set(r['cell'] for r in results))

    print(f"\n{'='*80}")
    print(f"DEGRADATION CORRELATION ANALYSIS")
    print(f"{'='*80}")

    # Features to correlate
    lgn_features = ['tau_full', 'tau_w300', 'tau_w100',
                    'd_full', 'd_w300', 'd_w100',
                    'decay_full', 'decay_w300',
                    'R_pulse', 'eta0']
    eis_markers = ['Z_1kHz_re', 'Z_1Hz_re', 'Z_01Hz_re', 'Z_1kHz_mag']

    all_correlations = {}

    for cell in cells:
        cr = sorted([r for r in results if r['cell'] == cell], key=lambda r: r['diag'])
        if len(cr) < 4:
            print(f"\n  {cell}: only {len(cr)} points, skipping correlation")
            continue

        print(f"\n  ── Cell {cell} ({len(cr)} diagnostics) ──")
        diags = [r['diag'] for r in cr]

        # Extract trajectories
        for feat in lgn_features:
            vals = [r.get(feat) for r in cr]
            if vals[0] is None:
                continue

            # For list features (tau, d), extract each component
            if isinstance(vals[0], list):
                for idx in range(len(vals[0])):
                    feat_name = f"{feat}[{idx}]"
                    feat_vals = np.array([v[idx] for v in vals])

                    for marker in eis_markers:
                        marker_vals = np.array([r[marker] for r in cr])

                        # Pearson correlation
                        r_pearson, p_pearson = stats.pearsonr(feat_vals, marker_vals)
                        # Spearman (rank) correlation — more robust
                        r_spearman, p_spearman = stats.spearmanr(feat_vals, marker_vals)

                        key = f"{cell}_{feat_name}_vs_{marker}"
                        all_correlations[key] = {
                            'pearson_r': r_pearson, 'pearson_p': p_pearson,
                            'spearman_r': r_spearman, 'spearman_p': p_spearman,
                        }

                        if abs(r_spearman) > 0.7:
                            sig = '***' if p_spearman < 0.001 else ('**' if p_spearman < 0.01 else '*')
                            print(f"    {feat_name:20s} vs {marker:12s}: "
                                  f"ρ={r_spearman:+.3f}{sig}  r={r_pearson:+.3f}")
            else:
                feat_vals = np.array(vals)
                for marker in eis_markers:
                    marker_vals = np.array([r[marker] for r in cr])
                    r_s, p_s = stats.spearmanr(feat_vals, marker_vals)
                    r_p, p_p = stats.pearsonr(feat_vals, marker_vals)

                    key = f"{cell}_{feat}_vs_{marker}"
                    all_correlations[key] = {
                        'pearson_r': r_p, 'pearson_p': p_p,
                        'spearman_r': r_s, 'spearman_p': p_s,
                    }
                    if abs(r_s) > 0.7:
                        sig = '***' if p_s < 0.001 else ('**' if p_s < 0.01 else '*')
                        print(f"    {feat:20s} vs {marker:12s}: "
                              f"ρ={r_s:+.3f}{sig}  r={r_p:+.3f}")

    # ---- Cross-cell summary ----
    print(f"\n  ── Cross-Cell Summary ──")
    # Pool all cells together
    all_diags = list(range(len(results)))
    for feat_base in ['tau_full', 'tau_w300', 'd_full', 'decay_full']:
        if results[0].get(feat_base) is None:
            continue
        n_comp = len(results[0][feat_base])
        for idx in range(n_comp):
            feat_vals = np.array([r[feat_base][idx] for r in results])
            for marker in ['Z_1kHz_re', 'Z_1Hz_re']:
                marker_vals = np.array([r[marker] for r in results])
                r_s, p_s = stats.spearmanr(feat_vals, marker_vals)
                r_p, p_p = stats.pearsonr(feat_vals, marker_vals)
                sig = '***' if p_s < 0.001 else ('**' if p_s < 0.01 else '*' if p_s < 0.05 else '')
                name = f"ALL {feat_base}[{idx}]"
                print(f"    {name:25s} vs {marker:12s}: "
                      f"ρ={r_s:+.3f}{sig}  r={r_p:+.3f}  (n={len(feat_vals)})")

    return all_correlations


# ============================================================================
# 9. SUMMARY TABLES
# ============================================================================
def print_trajectory_table(results):
    """Print eigenvalue trajectory table."""
    cells = sorted(set(r['cell'] for r in results))
    n = len(results[0]['tau_full'])

    print(f"\n{'='*90}")
    print(f"EIGENVALUE TRAJECTORIES")
    print(f"{'='*90}")

    for cell in cells:
        cr = sorted([r for r in results if r['cell'] == cell], key=lambda r: r['diag'])
        print(f"\n  Cell {cell}:")
        header = f"  {'Diag':>4} |"
        for i in range(n):
            header += f" {'τ'+str(i+1):>8}"
        header += f" | {'η₀[mV]':>8} {'R_p[mΩ]':>8} {'Z_1kHz':>8} {'Z_1Hz':>8}"
        print(header)
        print(f"  {'-'*len(header)}")

        for r in cr:
            line = f"  {r['diag']:>4} |"
            for i in range(n):
                line += f" {r['tau_full'][i]:8.2f}"
            line += f" | {r['eta0']*1000:8.2f} {r['R_pulse']*1000:8.2f}"
            line += f" {r['Z_1kHz_re']*1000:8.2f} {r['Z_1Hz_re']*1000:8.2f}"
            print(line)

        # Trend (first → last)
        if len(cr) > 1:
            for i in range(n):
                t0 = cr[0]['tau_full'][i]
                t1 = cr[-1]['tau_full'][i]
                pct = (t1 - t0) / abs(t0) * 100
                print(f"    τ{i+1}: {t0:.2f} → {t1:.2f}  ({pct:+.1f}%)")
            z0 = cr[0]['Z_1kHz_re']
            z1 = cr[-1]['Z_1kHz_re']
            print(f"    Z_1kHz: {z0*1000:.2f} → {z1*1000:.2f} mΩ  ({(z1-z0)/z0*100:+.1f}%)")


def print_impedance_summary(results):
    """Impedance reconstruction quality across all experiments."""
    print(f"\n{'='*70}")
    print(f"IMPEDANCE RECONSTRUCTION QUALITY")
    print(f"{'='*70}")

    for wname in ['full', 'w300', 'w100']:
        re_corrs = [r[f'z_re_corr_{wname}'] for r in results
                    if not np.isnan(r.get(f'z_re_corr_{wname}', np.nan))]
        im_corrs = [r[f'z_im_corr_{wname}'] for r in results
                    if not np.isnan(r.get(f'z_im_corr_{wname}', np.nan))]
        if re_corrs:
            wlabel = {'full': '3600s', 'w300': '300s', 'w100': '100s'}[wname]
            print(f"  {wlabel:>5}: Re corr = {np.mean(re_corrs):.4f} ± {np.std(re_corrs):.4f}  "
                  f"Im corr = {np.mean(im_corrs):.4f} ± {np.std(im_corrs):.4f}")


# ============================================================================
# 10. PLOTTING
# ============================================================================
def plot_degradation(results, out_dir):
    """Publication-quality degradation tracking figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

    cells = sorted(set(r['cell'] for r in results))
    n_tau = len(results[0]['tau_full'])
    colors = {'W8': '#1f77b4', 'W9': '#ff7f0e', 'W10': '#2ca02c',
              'V4': '#d62728', 'W3': '#9467bd', 'V5': '#8c564b'}

    # ---- Figure 1: Eigenvalue trajectories ----
    fig, axes = plt.subplots(1, n_tau + 1, figsize=(5 * (n_tau + 1), 4.5))

    for ci, cell in enumerate(cells):
        cr = sorted([r for r in results if r['cell'] == cell], key=lambda r: r['diag'])
        diags = [r['diag'] for r in cr]
        c = colors.get(cell, f'C{ci}')

        for si in range(n_tau):
            axes[si].plot(diags, [r['tau_full'][si] for r in cr],
                         'o-', color=c, markersize=5, linewidth=1.5, label=cell)
            axes[si].set_xlabel('Diagnostic #')
            axes[si].set_ylabel(f'τ{si+1} [s]')
            axes[si].set_title(f'Time Constant τ{si+1}')
            axes[si].grid(True, alpha=0.2)

        # EIS impedance for comparison
        axes[-1].plot(diags, [r['Z_1kHz_re'] * 1000 for r in cr],
                     'o-', color=c, markersize=5, linewidth=1.5, label=cell)

    axes[-1].set_xlabel('Diagnostic #')
    axes[-1].set_ylabel('Z @ 1kHz [mΩ]')
    axes[-1].set_title('EIS Impedance (ground truth)')
    axes[-1].grid(True, alpha=0.2)
    axes[0].legend()

    fig.suptitle('Degradation Tracking: LGN Eigenvalues vs EIS Impedance', fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/eigenvalue_trajectories.png')
    plt.close(fig)

    # ---- Figure 2: Scatter correlations ----
    fig, axes = plt.subplots(n_tau, 2, figsize=(10, 4.5 * n_tau), squeeze=False)

    for si in range(n_tau):
        for ci, cell in enumerate(cells):
            cr = [r for r in results if r['cell'] == cell]
            c = colors.get(cell, f'C{ci}')

            taus = [r['tau_full'][si] for r in cr]
            z_1k = [r['Z_1kHz_re'] * 1000 for r in cr]
            z_1 = [r['Z_1Hz_re'] * 1000 for r in cr]

            axes[si][0].scatter(taus, z_1k, color=c, s=30, label=cell, alpha=0.7)
            axes[si][1].scatter(taus, z_1, color=c, s=30, label=cell, alpha=0.7)

        axes[si][0].set_xlabel(f'τ{si+1} [s]')
        axes[si][0].set_ylabel('Z @ 1kHz [mΩ]')
        axes[si][0].grid(True, alpha=0.2)
        axes[si][1].set_xlabel(f'τ{si+1} [s]')
        axes[si][1].set_ylabel('Z @ 1Hz [mΩ]')
        axes[si][1].grid(True, alpha=0.2)

    axes[0][0].legend()
    fig.suptitle('LGN Time Constants vs EIS Impedance', fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/tau_vs_eis_scatter.png')
    plt.close(fig)

    print(f"  Plots saved to {out_dir}/")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='lgn_csv')
    p.add_argument('--cells', nargs='+', default=['W8', 'W9', 'W10'])
    p.add_argument('--soc', type=int, default=50)
    p.add_argument('--n_states', type=int, default=2)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out_dir', default='results_degradation')
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available → CPU")
        args.device = 'cpu'
    print(f"Device: {args.device}")
    print(f"Cells: {args.cells}, n_states={args.n_states}, SOC={args.soc}%")

    results = run_degradation(
        args.data_dir, args.cells, soc_target=args.soc,
        n_states=args.n_states, device=args.device)

    print_trajectory_table(results)
    print_impedance_summary(results)
    correlations = correlation_analysis(results)

    # Save everything
    os.makedirs(args.out_dir, exist_ok=True)
    serializable = []
    for r in results:
        s = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                s[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                s[k] = float(v)
            elif v is None or isinstance(v, (int, float, str, list, dict, bool)):
                s[k] = v
        serializable.append(s)
    with open(f'{args.out_dir}/results.json', 'w') as f:
        json.dump(serializable, f, indent=2)
    with open(f'{args.out_dir}/correlations.json', 'w') as f:
        json.dump({k: {kk: (float(vv) if isinstance(vv, float) else vv)
                       for kk, vv in v.items()} for k, v in correlations.items()}, f, indent=2)

    print(f"\nSaved to {args.out_dir}/")

    if args.plot:
        plot_degradation(results, args.out_dir)
