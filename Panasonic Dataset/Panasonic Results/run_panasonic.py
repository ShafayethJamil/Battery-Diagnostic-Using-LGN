"""
Panasonic NCA 18650PF: Cross-Chemistry & Temperature Validation
================================================================
Runs LGN with n_states=3 on Panasonic HPPC relaxation data at a single
temperature. Compares extracted time constants to EIS ground truth.

Model order: n=3 (validated by Stanford model order ablation).
  τ₁ ~ charge transfer
  τ₂ ~ SEI / mid-frequency
  τ₃ ~ solid-state diffusion

Auto-discovers files in the current directory:
  - One *HPPC*.mat file (HPPC pulse data)
  - Multiple *EIS*.csv files (one per SOC level)

Each temperature folder gets its own copy of this script.
Run five in parallel for full temperature sweep.


Requires: run_degradation.py in same directory (provides LGN_Battery)

Author: Shafayeth Jamil (USC ECE), February 2026
"""
import argparse, glob, json, os, re, sys
import numpy as np
import scipy.io as sio
from scipy import stats
from scipy.optimize import curve_fit

import torch
import torch.nn as nn

# Import core LGN model
from run_degradation import (
    LGN_Battery, _subsample_log
)


# ============================================================================
# SOC LEVEL MAPPING (from readme)
# ============================================================================
SOC_LEVELS_FULL = [100, 95, 90, 80, 70, 60, 50, 40, 30, 25, 20, 15, 10, 5, 0]


# ============================================================================
# LOG-DISTANCE TAU MATCHING
# ============================================================================
def match_taus_log(lgn_taus, eis_taus):
    """
    Match LGN taus to EIS taus by minimizing total log-distance.
    Uses brute-force permutation (optimal for 2-3 modes).
    
    Returns:
        matched_pairs: list of (lgn_tau, eis_tau) tuples, sorted by eis_tau
    """
    from itertools import permutations
    
    n = min(len(lgn_taus), len(eis_taus))
    lgn_arr = np.array(sorted(lgn_taus)[:n])
    eis_arr = np.array(sorted(eis_taus)[:n])
    
    best_cost = float('inf')
    best_perm = list(range(n))
    
    for perm in permutations(range(n)):
        cost = sum(abs(np.log(lgn_arr[i] + 1e-10) - np.log(eis_arr[perm[i]] + 1e-10))
                   for i in range(n))
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    
    matched = sorted([(float(lgn_arr[i]), float(eis_arr[best_perm[i]])) 
                      for i in range(n)], key=lambda x: x[1])
    return matched


# ============================================================================
# PARSE HPPC .mat FILE
# ============================================================================
def parse_hppc(mat_path, c_rate_idx=2):
    """
    Parse Panasonic HPPC .mat file and extract relaxation curves.
    
    The HPPC has 5 pulses per SOC level at C-rates: 0.5C, 1C, 2C, 4C, 6C.
    Each pulse is ~10s discharge followed by ~1200s rest.
    
    Args:
        mat_path: Path to HPPC .mat file
        c_rate_idx: Which C-rate to use (0=0.5C, 1=1C, 2=2C, 3=4C, 4=6C)
                    Default 2 (2C = 5.8A) for strong signal
    
    Returns:
        list of dicts with keys: soc, soc_idx, t_relax, v_relax, eta, etc.
    """
    mat = sio.loadmat(mat_path)
    meas = mat['meas'][0, 0]
    
    V = meas['Voltage'].flatten()
    I = meas['Current'].flatten()
    t = meas['Time'].flatten()
    T = meas['Battery_Temp_degC'].flatten()
    
    # Find all discharge pulse starts (current drops below -0.5A)
    pulse_starts = []
    for i in range(1, len(I)):
        if I[i] < -0.5 and I[i-1] > -0.5:
            pulse_starts.append(i)
    
    n_pulses = len(pulse_starts)
    n_soc = n_pulses // 5
    
    temp_mean = float(np.nanmean(T))
    
    print(f"  HPPC: {n_pulses} pulses, {n_soc} SOC levels, T={temp_mean:.0f}C")
    
    relaxations = []
    
    for soc_i in range(n_soc):
        pulse_idx = soc_i * 5 + c_rate_idx
        if pulse_idx >= n_pulses:
            break
        
        idx_start = pulse_starts[pulse_idx]
        
        # Find pulse end (current returns to ~0)
        idx_end = idx_start
        while idx_end < len(I) - 1 and I[idx_end] < -0.3:
            idx_end += 1
        
        I_pulse = float(np.mean(I[idx_start:idx_end]))
        v_before = float(V[idx_start - 1])
        
        # Relaxation ends at next pulse or end of data
        if pulse_idx + 1 < n_pulses:
            idx_relax_end = pulse_starts[pulse_idx + 1] - 1
        else:
            idx_relax_end = len(t) - 1
        
        t_relax = t[idx_end:idx_relax_end + 1] - t[idx_end]
        v_relax = V[idx_end:idx_relax_end + 1]
        
        if len(t_relax) < 100 or t_relax[-1] < 60:
            print(f"    SOC {soc_i}: skipping (relaxation only {t_relax[-1]:.0f}s)")
            continue
        
        v_inf = v_relax[-1]
        eta = v_relax - v_inf
        temp_relax = float(np.nanmean(T[idx_end:idx_relax_end + 1]))
        soc_pct = SOC_LEVELS_FULL[soc_i] if soc_i < len(SOC_LEVELS_FULL) else None
        
        relaxations.append({
            'soc': soc_pct,
            'soc_idx': soc_i,
            't_relax': t_relax,
            'v_relax': v_relax,
            'eta': eta,
            'v_before': v_before,
            'v_after': v_inf,
            'I_pulse': I_pulse,
            'temp': temp_relax,
            'pulse_duration': float(t[idx_end] - t[idx_start]),
            'relax_duration': float(t_relax[-1]),
            'n_samples': len(t_relax),
        })
        
        print(f"    SOC {soc_pct:>3}%: V={v_before:.3f}->{v_inf:.3f}V, "
              f"I={I_pulse:.1f}A, relax={t_relax[-1]:.0f}s, "
              f"eta0={eta[0]*1000:.1f}mV, {len(t_relax)} pts")
    
    return relaxations, temp_mean


# ============================================================================
# PARSE EIS CSV FILES
# ============================================================================
def parse_eis_file(csv_path):
    """
    Parse a single Panasonic EIS CSV file (semicolon-delimited).
    Returns freq, Zre, Zim (sorted by frequency), voltage.
    """
    lines = open(csv_path, 'r').readlines()
    
    header_line = None
    for i, line in enumerate(lines):
        if 'Zreal1' in line:
            header_line = i
            break
    
    if header_line is None:
        return None, None, None, None
    
    data_start = header_line + 2
    freqs, zre, zim = [], [], []
    voltage = None
    
    for line in lines[data_start:]:
        parts = line.strip().split(';')
        if len(parts) > 25:
            try:
                f = float(parts[24])   # ActFreq
                r = float(parts[22])   # Zreal1 [mOhm]
                x = float(parts[23])   # Zimg1 [mOhm]
                freqs.append(f)
                zre.append(r)
                zim.append(x)
                if voltage is None:
                    voltage = float(parts[8])
            except (ValueError, IndexError):
                pass
    
    if len(freqs) == 0:
        return None, None, None, None
    
    # Sort by frequency 
    freq = np.array(freqs)
    Zre = np.array(zre)
    Zim = np.array(zim)
    sort_idx = np.argsort(freq)
    
    return freq[sort_idx], Zre[sort_idx], Zim[sort_idx], voltage


def load_all_eis(eis_files):
    """Load all EIS files, sorted by file number = SOC index."""
    def extract_number(path):
        match = re.search(r'EIS(\d+)', os.path.basename(path))
        return int(match.group(1)) if match else 999
    
    eis_files = sorted(eis_files, key=extract_number)
    
    eis_data = []
    for fi, path in enumerate(eis_files):
        freq, zre, zim, voltage = parse_eis_file(path)
        if freq is None:
            continue
        
        soc_pct = SOC_LEVELS_FULL[fi] if fi < len(SOC_LEVELS_FULL) else None
        
        eis_data.append({
            'soc': soc_pct,
            'soc_idx': fi,
            'freq': freq,
            'Zre': zre,
            'Zim': zim,
            'voltage': voltage,
            'file': os.path.basename(path),
            'n_points': len(freq),
            'freq_min': float(freq.min()),
            'freq_max': float(freq.max()),
        })
        
        print(f"    EIS {fi+1:>2} (SOC {soc_pct:>3}%): {len(freq)} pts, "
              f"f=[{freq.min():.4f}, {freq.max():.0f}] Hz, V={voltage:.3f}V")
    
    return eis_data


# ============================================================================
# EIS RC FITTING
# ============================================================================
def fit_eis_nrc(freq, Zre, Zim, n_rc=2):
    """
    Fit n-RC Randles model to EIS data.
    Z(f) = Rs + sum Ri / (1 + j*2pi*f*tau_i)
    """
    Z_data = Zre + 1j * Zim
    
    # Sort by frequency for stable fitting 
    idx = np.argsort(freq)
    freq, Zre, Zim = freq[idx], Zre[idx], Zim[idx]
    Z_data = Z_data[idx]
    
    def Z_model(f, *params):
        Rs = params[0]
        Z = Rs * np.ones(len(f), dtype=complex)
        for k in range(n_rc):
            R = params[1 + 2*k]
            tau = params[2 + 2*k]
            Z += R / (1 + 1j * 2 * np.pi * f * tau)
        return np.concatenate([Z.real, Z.imag])
    
    Z_target = np.concatenate([Zre, Zim])
    
    Rs0 = Zre[np.argmax(freq)]
    R_total = max(Zre[np.argmin(freq)] - Rs0, 1.0)  # guard against negative
    
    # Widened tau bounds for cold temperature robustness 
    if n_rc == 2:
        p0 = [Rs0, R_total * 0.3, 0.01, R_total * 0.7, 1.0]
        bounds_lo = [0, 0, 1e-5, 0, 1e-4]
        bounds_hi = [Rs0 * 2, R_total * 3, 50, R_total * 3, 500]
    elif n_rc == 3:
        p0 = [Rs0, R_total * 0.2, 0.001, R_total * 0.3, 0.1, R_total * 0.5, 10]
        bounds_lo = [0, 0, 1e-6, 0, 1e-5, 0, 0.01]
        bounds_hi = [Rs0 * 2, R_total * 3, 5, R_total * 3, 100, R_total * 3, 1000]
    else:
        return None
    
    try:
        popt, _ = curve_fit(Z_model, freq, Z_target, p0=p0,
                           bounds=(bounds_lo, bounds_hi), maxfev=20000)
        
        result = {'Rs': float(popt[0])}
        for k in range(n_rc):
            result[f'R{k+1}'] = float(popt[1 + 2*k])
            result[f'tau{k+1}'] = float(popt[2 + 2*k])
        
        Z_pred = Z_model(freq, *popt)
        Z_pred_complex = Z_pred[:len(freq)] + 1j * Z_pred[len(freq):]
        residual = np.abs(Z_data - Z_pred_complex)
        result['nrmse'] = float(np.sqrt(np.mean(residual**2)) /
                                np.sqrt(np.mean(np.abs(Z_data)**2)))
        
        return result
    except Exception as e:
        return None


# ============================================================================
# LGN TRAINING (n=3)
# ============================================================================
def train_lgn_3d(t_data, eta_data, n_epochs=4000, lr=0.01,
                 subsample=300, device='cpu', verbose=False):
    """
    Multi-restart LGN training with n_states=3.
    Model order justified by Stanford ablation (n=3 = correct physics).
    Independent SOC fits -- no warm-start needed (no sequential collapse).
    """
    n_states = 3

    idx = _subsample_log(len(t_data), subsample) if subsample else np.arange(len(t_data))
    t_t = torch.tensor(t_data[idx], dtype=torch.float64, device=device)
    eta_t = torch.tensor(eta_data[idx], dtype=torch.float64, device=device)

    # 8 diverse initializations spanning CT -> SEI -> diffusion
    # 1200s window: identifiable band [3s, 400s]
    inits = [
        torch.tensor([2.0, -3.0, -5.5]),    # tau ~ [0.5, 20, 245]
        torch.tensor([1.5, -2.5, -6.0]),    # tau ~ [0.6, 12, 403]
        torch.tensor([2.5, -3.5, -5.0]),    # tau ~ [0.4, 33, 150]
        torch.tensor([3.0, -2.0, -6.5]),    # tau ~ [0.3, 8, 670]
        torch.tensor([1.0, -3.0, -6.0]),    # tau ~ [0.8, 20, 403]
        torch.tensor([2.0, -4.0, -5.5]),    # tau ~ [0.5, 55, 245]
        torch.tensor([2.0, -2.0, -7.0]),    # tau ~ [0.5, 8, 1097]
        torch.tensor([1.0, -1.5, -5.0]),    # tau ~ [0.8, 5, 150]
    ]

    best_model, best_loss = None, float('inf')

    for i_init, init_d in enumerate(inits):
        model = LGN_Battery(n_states).double().to(device)
        model.s_params.requires_grad_(False)  # diagonal A (ECM-equivalent)
        model.d_params.data = init_d.clone().to(device)
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
            #  reload best state before printing
            model.load_state_dict(run_best_state)
            taus = model.get_time_constants()
            print(f"      init {i_init}: tau={np.round(taus,1)}  loss={run_best_loss:.4e}")

    best_model.eval()
    with torch.no_grad():
        eta_pred = best_model(t_t).cpu().numpy()

    nrmse = float(np.sqrt(np.mean((eta_pred - eta_data[idx])**2)) /
                  (np.abs(eta_data).max() + 1e-12))

    return best_model, nrmse, best_loss


# ============================================================================
# ANALYZE ONE SOC CONDITION
# ============================================================================
def analyze_soc(relax_data, eis_data, device='cpu'):
    """
    Run LGN on one HPPC relaxation, compare to matched EIS.
    Returns dict with all results for this SOC level.
    """
    soc = relax_data['soc']
    res = {
        'soc': soc,
        'temp': relax_data['temp'],
        'v_before': relax_data['v_before'],
        'v_after': relax_data['v_after'],
        'I_pulse': relax_data['I_pulse'],
        'eta0': float(relax_data['eta'][0]),
        'relax_duration': relax_data['relax_duration'],
    }
    
    t_relax = relax_data['t_relax']
    eta = relax_data['eta']
    
    # ---- Run LGN-3D ----
    print(f"\n  SOC {soc}%: Running LGN (n=3)...")
    model, nrmse, loss = train_lgn_3d(
        t_relax, eta, n_epochs=4000, lr=0.01,
        subsample=min(300, len(t_relax)), device=device, verbose=True)
    
    taus = sorted(model.get_time_constants().tolist())
    res['tau_full'] = taus
    res['nrmse_lgn'] = nrmse
    res['loss_lgn'] = float(loss)
    
    print(f"    BEST -> tau = {np.round(taus, 3)}  NRMSE = {nrmse:.5f}")
    
    # ---- EIS fitting ----
    if eis_data is not None:
        freq = eis_data['freq']
        Zre = eis_data['Zre']
        Zim = eis_data['Zim']
        
        # Store raw EIS for Nyquist plotting
        res['eis_freq'] = freq.tolist()
        res['eis_Zre'] = Zre.tolist()
        res['eis_Zim'] = Zim.tolist()
        
        # 2-RC fit
        eis_2rc = fit_eis_nrc(freq, Zre, Zim, n_rc=2)
        if eis_2rc:
            res['tau_eis_2rc'] = sorted([eis_2rc['tau1'], eis_2rc['tau2']])
            res['Rs_eis_2rc'] = eis_2rc['Rs']
            res['R1_eis_2rc'] = eis_2rc['R1']
            res['R2_eis_2rc'] = eis_2rc['R2']
            res['nrmse_eis_2rc'] = eis_2rc['nrmse']
            
            # Log-distance matching 
            matched = match_taus_log(taus, res['tau_eis_2rc'])
            res['matched_2rc'] = matched
        
        # 3-RC fit
        eis_3rc = fit_eis_nrc(freq, Zre, Zim, n_rc=3)
        if eis_3rc:
            res['tau_eis_3rc'] = sorted([eis_3rc['tau1'], eis_3rc['tau2'], eis_3rc['tau3']])
            res['Rs_eis_3rc'] = eis_3rc['Rs']
            res['nrmse_eis_3rc'] = eis_3rc['nrmse']
            
            matched_3rc = match_taus_log(taus, res['tau_eis_3rc'])
            res['matched_3rc'] = matched_3rc
        
        # EIS scalar features at specific frequencies
        Z_complex = Zre + 1j * Zim
        for f_target, label in [(1000, 'Z_1kHz'), (100, 'Z_100Hz'),
                                 (10, 'Z_10Hz'), (1, 'Z_1Hz'), (0.1, 'Z_01Hz'),
                                 (0.01, 'Z_001Hz')]:
            fi = np.argmin(np.abs(freq - f_target))
            if abs(freq[fi] - f_target) / max(f_target, 0.001) < 0.5:
                res[f'{label}_re'] = float(Zre[fi])
                res[f'{label}_im'] = float(Zim[fi])
                res[f'{label}_mag'] = float(np.abs(Z_complex[fi]))
        
        print(f"    EIS 2RC: tau = {res.get('tau_eis_2rc', 'FAIL')}")
        print(f"    EIS 3RC: tau = {res.get('tau_eis_3rc', 'FAIL')}")
        if 'matched_2rc' in res:
            print(f"    Matched (2RC): {[(f'{l:.3f}', f'{e:.3f}') for l,e in res['matched_2rc']]}")
    else:
        print(f"    No matching EIS data for SOC {soc}%")
    
    return res


# ============================================================================
# PLOTTING (Proper Nyquist)
# ============================================================================
def make_plots(results, eis_by_soc, temp_label, out_dir='.'):
    """Generate summary plots for one temperature."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({'font.size': 11, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
    
    socs = [r['soc'] for r in results]
    
    # ---- Figure 1: tau vs SOC + parity + NRMSE ----
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    
    # Panel 1: LGN 3 taus vs SOC
    colors_lgn = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for ti in range(3):
        tau_vals = [r['tau_full'][ti] for r in results]
        label = ['tau1(CT)', 'tau2(SEI)', 'tau3(diff)'][ti]
        axes[0].plot(socs, tau_vals, 'o-', label=label, markersize=6,
                    color=colors_lgn[ti])
    axes[0].set_xlabel('SOC [%]')
    axes[0].set_ylabel('tau [s]')
    axes[0].set_title(f'LGN Time Constants @ {temp_label}')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)
    
    # Panel 2: EIS 2RC taus vs SOC
    has_eis = any('tau_eis_2rc' in r for r in results)
    if has_eis:
        for ti in range(2):
            tau_vals = [r['tau_eis_2rc'][ti] if 'tau_eis_2rc' in r else np.nan 
                       for r in results]
            label = ['tau_EIS1(fast)', 'tau_EIS2(slow)'][ti]
            axes[1].plot(socs, tau_vals, 's--', label=label, markersize=6)
        axes[1].set_xlabel('SOC [%]')
        axes[1].set_ylabel('tau [s]')
        axes[1].set_title(f'EIS 2RC Time Constants @ {temp_label}')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.2)
    
    # Panel 3: LGN vs EIS parity scatter (log-matched)
    lgn_matched, eis_matched = [], []
    for r in results:
        if 'matched_2rc' in r:
            for lgn_t, eis_t in r['matched_2rc']:
                lgn_matched.append(lgn_t)
                eis_matched.append(eis_t)
    
    if lgn_matched:
        axes[2].scatter(eis_matched, lgn_matched, c='steelblue', s=50, alpha=0.7)
        all_vals = lgn_matched + eis_matched
        lo, hi = min(all_vals) * 0.3, max(all_vals) * 3
        axes[2].plot([lo, hi], [lo, hi], '--', color='gray', alpha=0.5, label='y=x')
        rho, p = stats.spearmanr(lgn_matched, eis_matched)
        axes[2].set_xlabel('EIS tau [s]')
        axes[2].set_ylabel('LGN tau [s]')
        axes[2].set_title(f'LGN vs EIS (log-matched) rho={rho:.3f}')
        axes[2].set_xscale('log')
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].grid(True, alpha=0.2)
    
    # Panel 4: NRMSE vs SOC
    nrmse_vals = [r['nrmse_lgn'] for r in results]
    axes[3].plot(socs, nrmse_vals, 'o-', color='darkgreen', markersize=6)
    axes[3].set_xlabel('SOC [%]')
    axes[3].set_ylabel('NRMSE')
    axes[3].set_title(f'LGN Fit Quality @ {temp_label}')
    axes[3].grid(True, alpha=0.2)
    
    fig.suptitle(f'Panasonic NCA 18650PF -- {temp_label}', fontweight='bold', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/fig_panasonic_{temp_label.replace(chr(176),"").replace(" ","_")}.png')
    plt.close(fig)
    
    # ---- Figure 2: Proper Nyquist plots  ----
    n_soc = len(results)
    ncols = min(4, n_soc)
    nrows = (n_soc + ncols - 1) // ncols
    fig, axes_ny = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if n_soc == 1:
        axes_ny = np.array([axes_ny])
    axes_ny = axes_ny.flatten()
    
    for i, r in enumerate(results):
        ax = axes_ny[i]
        soc_val = r['soc']
        
        if 'eis_freq' in r:
            freq = np.array(r['eis_freq'])
            Zre = np.array(r['eis_Zre'])
            Zim = np.array(r['eis_Zim'])
            
            # Raw EIS Nyquist: Zre vs -Zim
            ax.plot(Zre, -Zim, 'o', color='steelblue', markersize=3,
                   alpha=0.7, label='EIS data')
            
            # Mark characteristic frequencies
            for f_mark, clr, lbl in [(1.0, 'red', '1Hz'),
                                      (0.1, 'orange', '0.1Hz'),
                                      (0.01, 'green', '0.01Hz')]:
                fi = np.argmin(np.abs(freq - f_mark))
                if abs(freq[fi] - f_mark) / max(f_mark, 0.001) < 0.5:
                    ax.plot(Zre[fi], -Zim[fi], 'x', color=clr, markersize=10,
                           markeredgewidth=2, label=lbl)
            
            # Mark LGN tau frequencies: f = 1/(2*pi*tau)
            taus = r['tau_full']
            marker_colors = ['purple', 'magenta', 'brown']
            for ti, tau in enumerate(taus):
                f_tau = 1.0 / (2 * np.pi * tau)
                fi = np.argmin(np.abs(freq - f_tau))
                # Only mark if frequency is within EIS range
                if freq.min() <= f_tau <= freq.max():
                    ax.plot(Zre[fi], -Zim[fi], 'D', color=marker_colors[ti],
                           markersize=8, markeredgewidth=1.5, markerfacecolor='none',
                           label=f'LGN tau{ti+1}={tau:.1f}s')
        
        ax.set_title(f'SOC {soc_val}%', fontsize=10)
        ax.set_xlabel('Zre [mOhm]')
        ax.set_ylabel('-Zim [mOhm]')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')
        if i == 0:
            ax.legend(fontsize=7, loc='upper left')
    
    for j in range(i + 1, len(axes_ny)):
        axes_ny[j].set_visible(False)
    
    fig.suptitle(f'Nyquist Plots -- {temp_label}', fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/fig_panasonic_nyquist_{temp_label.replace(chr(176),"").replace(" ","_")}.png')
    plt.close(fig)
    
    print(f"\n  Plots saved to {out_dir}/")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Panasonic NCA LGN Analysis')
    parser.add_argument('--device', default='cuda:0', help='torch device')
    parser.add_argument('--c_rate', type=int, default=2,
                        help='C-rate index (0=0.5C, 1=1C, 2=2C, 3=4C, 4=6C)')
    parser.add_argument('--skip_plot', action='store_true')
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # ---- Auto-discover files ----
    hppc_files = glob.glob('*HPPC*.mat') + glob.glob('*hppc*.mat') + \
                 glob.glob('*5Pulse*.mat') + glob.glob('*5pulse*.mat')
    hppc_files = list(set(hppc_files))
    
    eis_files = sorted(glob.glob('*EIS*.csv'))
    
    if not hppc_files:
        print("ERROR: No HPPC .mat file found in current directory!")
        sys.exit(1)
    
    hppc_file = hppc_files[0]
    print(f"\nHPPC file: {hppc_file}")
    print(f"EIS files: {len(eis_files)} found")
    
    # Detect temperature from filename
    fname = hppc_file.lower()
    temp_match = re.search(r'(n?\d+)degc', fname)
    if temp_match:
        t_str = temp_match.group(1)
        temp_label = t_str.replace('n', '-') + 'C'
    else:
        temp_label = 'unknown'
    print(f"Temperature: {temp_label}")
    
    # ---- Parse HPPC ----
    print(f"\n{'='*60}")
    print(f"Parsing HPPC ({['0.5C','1C','2C','4C','6C'][args.c_rate]} pulses)...")
    print(f"{'='*60}")
    relaxations, temp_measured = parse_hppc(hppc_file, c_rate_idx=args.c_rate)
    
    # ---- Parse EIS ----
    print(f"\n{'='*60}")
    print(f"Parsing EIS ({len(eis_files)} files)...")
    print(f"{'='*60}")
    if eis_files:
        eis_list = load_all_eis(eis_files)
    else:
        eis_list = []
        print("  No EIS files found -- will run LGN only")
    
    # Match by SOC
    eis_by_soc = {e['soc']: e for e in eis_list}
    
    # ---- Run LGN + EIS comparison ----
    print(f"\n{'='*60}")
    print(f"Running LGN-3D on {len(relaxations)} SOC levels...")
    print(f"{'='*60}")
    
    results = []
    for relax in relaxations:
        soc = relax['soc']
        eis = eis_by_soc.get(soc, None)
        r = analyze_soc(relax, eis, device=device)
        results.append(r)
    
    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"SUMMARY -- {temp_label} ({len(results)} SOC levels)")
    print(f"{'='*60}")
    
    print(f"\n{'SOC':>5} {'tau1(CT)':>9} {'tau2(SEI)':>9} {'tau3(diff)':>10} "
          f"{'EIS t1':>9} {'EIS t2':>9} {'NRMSE':>8}")
    print("-" * 65)
    
    for r in results:
        taus = r['tau_full']
        eis_taus = r.get('tau_eis_2rc', [None, None])
        eis_str1 = f"{eis_taus[0]:.4f}" if eis_taus and eis_taus[0] else "  ---"
        eis_str2 = f"{eis_taus[1]:.2f}" if eis_taus and len(eis_taus) > 1 else "  ---"
        
        print(f"{r['soc']:>4}%  {taus[0]:>8.3f}  {taus[1]:>8.2f}  "
              f"{taus[2]:>9.1f}  {eis_str1:>8}  {eis_str2:>8}  {r['nrmse_lgn']:>7.5f}")
    
    # ---- Save results ----
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
    
    out_name = f"results_panasonic_{temp_label.replace('-','n')}.json"
    with open(out_name, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {out_name}")
    
    # ---- Plots ----
    if not args.skip_plot:
        make_plots(results, eis_by_soc, temp_label)
    
    print(f"\nDone! {temp_label}, {len(results)} SOC levels analyzed.")


if __name__ == '__main__':
    main()
