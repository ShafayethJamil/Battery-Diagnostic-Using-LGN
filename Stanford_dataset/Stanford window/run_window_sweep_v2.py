"""
Window Sweep: n=3 LGN at 36s and 360s for Stanford SECL
========================================================
Self-contained script. Uses Stanford's LGN_Battery model (c=1, physically
principled: each state = branch voltage in ECM).

Initializations are tuned PER WINDOW:
  - 36s:  τ_max ~ 30s   (same regime as TRI 40s)
  - 360s: τ_max ~ 300s  (intermediate between BMS and full relaxation)



Requires: run_degradation.py (for LGN_Battery, lgn_to_impedance_shape, etc.)

Author: Shafayeth Jamil (USC ECE), February 2026
"""

import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from scipy import stats

import torch
import torch.nn as nn

# Import Stanford model + helpers (c=1, principled ECM interpretation)
from run_degradation import (
    LGN_Battery, lgn_to_impedance_shape,
    fit_scale_and_Rs, compare_impedance,
    _subsample_log
)


# ============================================================================
# PER-WINDOW INITIALIZATIONS
# ============================================================================
# softplus(d) mapping:
#   d =  3.0 -> tau ~ 0.05s    d = -1.0 -> tau ~ 3.3s     d = -4.0 -> tau ~ 55s
#   d =  2.0 -> tau ~ 0.5s     d = -2.0 -> tau ~ 8s        d = -5.0 -> tau ~ 150s
#   d =  1.0 -> tau ~ 0.8s     d = -2.5 -> tau ~ 12s       d = -5.5 -> tau ~ 245s
#   d =  0.0 -> tau ~ 1.4s     d = -3.0 -> tau ~ 20s       d = -6.0 -> tau ~ 403s

WINDOW_INITS = {
    36: [
        # 36s window: tau_3 must be < ~30s, same regime as TRI 40s data
        # (CT, SEI, diffusion) hypothesis
        torch.tensor([2.0, -1.0, -3.0]),    # tau ~ [0.5, 3, 20]
        torch.tensor([1.5, -0.5, -2.5]),    # tau ~ [0.6, 2, 12]
        torch.tensor([3.0, -1.0, -3.5]),    # tau ~ [0.3, 3, 33]
        torch.tensor([2.5, -1.5, -3.0]),    # tau ~ [0.4, 5, 20]
        torch.tensor([1.0,  0.0, -2.5]),    # tau ~ [0.8, 1.4, 12]
    ],
    360: [
        # 360s window: tau_3 can reach ~300s, tau_2 ~ 10-50s
        torch.tensor([2.0, -2.0, -5.0]),    # tau ~ [0.5, 8, 150]
        torch.tensor([1.5, -1.5, -4.5]),    # tau ~ [0.6, 5, 90]
        torch.tensor([2.5, -2.5, -5.5]),    # tau ~ [0.4, 12, 245]
        torch.tensor([3.0, -3.0, -5.0]),    # tau ~ [0.3, 20, 150]
        torch.tensor([1.0, -1.0, -4.0]),    # tau ~ [0.8, 3, 55]
    ],
}


# ============================================================================
# TRAINING: n=3 WITH WARM-START AND PER-WINDOW INITS
# ============================================================================
def train_lgn_3d_window(t_data, eta_data, window_sec,
                        n_epochs=2500, lr=0.01, subsample=250,
                        device='cpu', verbose=False,
                        prev_d_params=None, prev_x0=None):
    """
    Multi-restart LGN training for 3-state model, with initializations
    tuned to the observation window length.

    Uses Stanford LGN_Battery (c=1): eta(t) = 1^T exp(At) x0
    """
    n_states = 3

    idx = _subsample_log(len(t_data), subsample) if subsample else np.arange(len(t_data))
    t_t = torch.tensor(t_data[idx], dtype=torch.float64, device=device)
    eta_t = torch.tensor(eta_data[idx], dtype=torch.float64, device=device)

    # Get window-appropriate initializations
    inits = [init.clone() for init in WINDOW_INITS[window_sec]]

    # Warm-start: prepend previous diagnostic's solution
    if prev_d_params is not None:
        inits.insert(0, prev_d_params.clone().cpu())
        if verbose:
            print(f"      [warm-start from prev diag]")

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
            marker = ' <-- BEST' if is_new_best else ''
            print(f"      init {i_init}: tau={np.round(taus,2)}  "
                  f"loss={run_best_loss:.4e}{marker}")

    best_model.eval()
    with torch.no_grad():
        eta_pred = best_model(t_t).cpu().numpy()
    nrmse = np.sqrt(np.mean((eta_pred - eta_data[idx])**2)) / (np.abs(eta_data).max() + 1e-12)

    if verbose:
        print(f"    BEST -> tau = {np.round(best_model.get_time_constants(), 3)}  "
              f"NRMSE = {nrmse:.5f}")

    return best_model, nrmse, best_loss


# ============================================================================
# MAIN PIPELINE
# ============================================================================
WINDOWS = {
    'w36':  36,
    'w360': 360,
}


def run_window_sweep(data_dir, cells, soc_target, device='cpu'):
    """Run n=3 at 36s and 360s for all cells/diags at one SOC."""
    all_results = []

    for cell in cells:
        print(f"\n{'#'*70}")
        print(f"# CELL: {cell}  SOC: {soc_target}%  WINDOWS: 36s, 360s (n=3)")
        print(f"{'#'*70}")

        hppc = pd.read_csv(f'{data_dir}/{cell}_hppc_relaxation.csv')
        eis  = pd.read_csv(f'{data_dir}/{cell}_eis.csv')
        diags = sorted(set(hppc['diag'].unique()) & set(eis['diag'].unique()))

        soc_col = f'{soc_target}pct'
        re_col = f're_z_ohm_{soc_col}'
        im_col = f'im_z_ohm_{soc_col}'

        # Separate warm-start chain per window
        prev_models = {wname: None for wname in WINDOWS}

        for diag in diags:
            seg = hppc[(hppc['diag'] == diag) & (hppc['soc_pct'] == soc_target)]
            if len(seg) == 0:
                continue

            eis_seg = eis[eis['diag'] == diag]
            has_eis = re_col in eis_seg.columns and len(eis_seg) > 0
            freq = eis_seg['freq_Hz'].values if has_eis else None
            re_z = eis_seg[re_col].values if has_eis else None
            im_z = eis_seg[im_col].values if has_eis else None

            t_relax = seg['time_s'].values
            v_relax = seg['voltage_V'].values

            entry = {'cell': cell, 'diag': int(diag), 'soc': soc_target}

            for wname, wsec in WINDOWS.items():
                print(f"\n  Diag {int(diag)}, {wname} ({wsec}s):")

                # Truncate to window
                mask = t_relax <= wsec
                t_w = t_relax[mask]
                v_w = v_relax[mask]

                if len(t_w) < 10:
                    print(f"    WARNING: only {len(t_w)} points, skipping")
                    continue

                # Re-reference overpotential to end of THIS window
                eta_w = v_w - v_w[-1]

                # Get warm-start from previous diagnostic at this window
                prev_m = prev_models[wname]
                prev_d = prev_m.d_params.data if prev_m is not None else None
                prev_x = prev_m.x0.data if prev_m is not None else None

                model, nrmse, loss = train_lgn_3d_window(
                    t_w, eta_w, window_sec=wsec,
                    n_epochs=2500, lr=0.01,
                    subsample=min(250, len(t_w)),
                    device=device, verbose=True,
                    prev_d_params=prev_d, prev_x0=prev_x)

                prev_models[wname] = model

                taus = model.get_time_constants()
                d_rates = model.get_diagonal_damping()
                A = model.get_A_numpy()

                entry[f'tau_{wname}']   = taus.tolist()
                entry[f'd_{wname}']     = d_rates.tolist()
                entry[f'nrmse_{wname}'] = nrmse
                entry[f'A_{wname}']     = A.tolist()
                entry[f'n_pts_{wname}'] = int(mask.sum())
                entry[f't_max_{wname}'] = float(t_w[-1])

                # Impedance reconstruction (if EIS available)
                if freq is not None and len(freq) > 0:
                    try:
                        Z_shape = lgn_to_impedance_shape(model, freq)
                        Z_data = re_z + 1j * im_z
                        band = (freq >= 0.05) & (freq <= 2000)
                        a_fit, Rs_fit, Z_pred = fit_scale_and_Rs(
                            Z_shape[band], Z_data[band])
                        Z_pred_full = a_fit * Z_shape + Rs_fit
                        comp = compare_impedance(
                            Z_pred_full[band], Z_data[band], freq[band])
                        entry[f'z_re_corr_{wname}'] = comp['re_corr']
                        entry[f'z_im_corr_{wname}'] = comp['im_corr']
                        entry[f'z_nrmse_{wname}']   = comp['nrmse']
                        entry[f'Rs_fit_{wname}']     = float(Rs_fit)
                    except Exception as e:
                        print(f"    Impedance failed: {e}")

                print(f"    -> tau = {[round(t,2) for t in taus]}  NRMSE = {nrmse:.5f}")

            all_results.append(entry)

    return all_results


def print_summary(results):
    """Print comparison table across windows."""
    print(f"\n{'='*110}")
    print(f"WINDOW SWEEP SUMMARY -- n=3 diagonal LGN (c=1) at 36s and 360s")
    print(f"{'='*110}")
    print(f"{'Cell':<5} {'Diag':>4} {'SOC':>4} | "
          f"{'t1_36':>7} {'t2_36':>7} {'t3_36':>7} {'nrmse':>8} | "
          f"{'t1_360':>8} {'t2_360':>8} {'t3_360':>9} {'nrmse':>8}")
    print('-' * 110)

    for r in sorted(results, key=lambda x: (x['cell'], x['soc'], x['diag'])):
        t36  = r.get('tau_w36',  None)
        t360 = r.get('tau_w360', None)
        n36  = r.get('nrmse_w36',  None)
        n360 = r.get('nrmse_w360', None)

        def fmt_tau(taus, widths):
            if taus is None:
                return '   '.join(['N/A'.rjust(w) for w in widths])
            return ' '.join([f'{t:{w}.2f}' for t, w in zip(taus, widths)])

        t36_str  = fmt_tau(t36,  [7, 7, 7])
        t360_str = fmt_tau(t360, [8, 8, 9])
        n36_str  = f'{n36:8.5f}' if n36 is not None else '     N/A'
        n360_str = f'{n360:8.5f}' if n360 is not None else '     N/A'

        print(f"{r['cell']:<5} {r['diag']:4d} {r['soc']:4d} | "
              f"{t36_str} {n36_str} | {t360_str} {n360_str}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Window Sweep: n=3 at 36s and 360s')
    p.add_argument('--data_dir', default='lgn_csv')
    p.add_argument('--cells', nargs='+', default=['W8', 'W9', 'W10'])
    p.add_argument('--soc', type=int, required=True, help='SOC level (20, 50, or 80)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out_dir', default='results_window_sweep')
    args = p.parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available -> CPU")
        args.device = 'cpu'

    t0 = time.time()
    print(f"Device: {args.device}")
    print(f"Cells: {args.cells}, SOC: {args.soc}%")
    print(f"Windows: {WINDOWS}")
    print(f"Model: 3-state diagonal LGN, c=1 (Stanford LGN_Battery)")
    print(f"Training: 2500 epochs, 5+1 restarts, patience=200")
    print()

    results = run_window_sweep(args.data_dir, args.cells, args.soc, args.device)
    print_summary(results)

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = f'{args.out_dir}/window_sweep_SOC{args.soc}.json'

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

    with open(out_file, 'w') as f:
        json.dump(serializable, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n Saved {len(results)} results -> {out_file}")
    print(f"  Elapsed: {elapsed/60:.1f} min")
