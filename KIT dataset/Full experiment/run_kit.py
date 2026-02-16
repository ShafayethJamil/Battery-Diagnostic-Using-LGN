#!/usr/bin/env python3
"""
run_kit.py v2.1 — LGN-3D on KIT Battery Aging Dataset
=======================================================
Multi-segment training: shared A (→ shared τ), separate x₀ per segment.

KIT gives two relaxation segments per measurement:
  - Short relax (seq ~20-27): ~8 pts, ~10s, after 10s discharge
  - Long  relax (seq ~51-60): ~10 pts, ~60s, after 60s discharge


Author: Shafayeth Jamil (USC ECE), February 2026
"""
import argparse, csv, json, os, glob, sys, time
import numpy as np
from scipy import stats, optimize

import torch
import torch.nn as nn

from run_degradation import (
    LGN_Battery, lgn_to_impedance_shape,
    fit_scale_and_Rs, compare_impedance, safe_corr,
    fit_exponentials, _subsample_log
)


# ============================================================================
# CONSTANTS
# ============================================================================
I_PULSE_DEFAULT = 1.0
I_THRESH_OFF = 0.1       # |I| < 0.1A → relaxation
I_THRESH_ON  = 0.5       # |I| > 0.5A → pulse active
CU_GAP_S = 86400
N_STATES = 3
N_EPOCHS = 2500
N_INITS  = 4             # + warm-start = 5 total


# ============================================================================
# 1. DATA LOADING
# ============================================================================
def load_csv(path):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f, delimiter=';'):
            rows.append(row)
    return rows


def group_by_checkup(rows, soc=50, is_rt=1):
    filt = [r for r in rows
            if int(r['soc_nom']) == soc and int(r['is_rt']) == is_rt]
    if not filt:
        return []
    groups, cur = [], [filt[0]]
    for i in range(1, len(filt)):
        if float(filt[i]['timestamp_s']) - float(filt[i-1]['timestamp_s']) > CU_GAP_S:
            groups.append(cur)
            cur = [filt[i]]
        else:
            cur.append(filt[i])
    groups.append(cur)
    return groups


# ============================================================================
# 2. SEGMENT EXTRACTION — both relaxation segments 
# ============================================================================
def extract_relaxation_segments(points):
    """Extract clean post-pulse relaxation segments from one 61-point measurement.

    Returns list of (t, eta) tuples + metadata dict with i_pulses.
    """
    ts   = np.array([float(r['timestamp_s']) for r in points])
    vs   = np.array([float(r['v_raw_V'])     for r in points])
    amps = np.array([float(r['i_raw_A'])     for r in points])
    t    = ts - ts[0]

    relax_mask = np.abs(amps) < I_THRESH_OFF

    # Find contiguous relaxation runs
    raw_segments = []
    start = None
    for i in range(len(relax_mask)):
        if relax_mask[i]:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= 3:
                raw_segments.append((start, i))
            start = None
    if start is not None and len(relax_mask) - start >= 3:
        raw_segments.append((start, len(relax_mask)))

    # only keep relaxations preceded by actual current pulse
    segments = []
    for s, e in raw_segments:
        if s > 0:
            pre = amps[max(0, s-3):s]
            if np.any(np.abs(pre) > I_THRESH_ON):
                segments.append((s, e))

    # Build (t, eta) pairs + measure actual pulse current
    relax_pairs = []
    i_pulses = []
    for s, e in segments:
        t_seg = t[s:e] - t[s]
        v_seg = vs[s:e]
        v_inf = v_seg[-1]
        eta_seg = v_seg - v_inf

        # actual pulse current from preceding segment
        pre_start = max(0, s - 8)
        pre_amps = np.abs(amps[pre_start:s])
        active = pre_amps[pre_amps > I_THRESH_ON]
        i_pulse = float(np.median(active)) if len(active) > 0 else I_PULSE_DEFAULT

        relax_pairs.append((t_seg, eta_seg))
        i_pulses.append(i_pulse)

    meta = dict(
        n_segments=len(relax_pairs),
        total_pts=sum(len(p[0]) for p in relax_pairs),
        windows_s=[float(p[0][-1]) for p in relax_pairs],
        i_pulses=i_pulses,
    )
    return relax_pairs, meta


# ============================================================================
# 3. MULTI-SEGMENT LGN TRAINING — shared A, separate x₀
# ============================================================================
def train_lgn_multiseg(segments, n_epochs=N_EPOCHS, lr=0.01,
                       device='cpu', verbose=False,
                       prev_d_params=None, prev_x0s=None,
                       n_random_inits=N_INITS):
    """
    Multi-segment 3D LGN: fits shared A matrix across multiple relaxation
    segments, each with its own initial condition x₀.

    """
    n = N_STATES
    n_seg = len(segments)

    seg_tensors = []
    for t_s, eta_s in segments:
        seg_tensors.append((
            torch.tensor(t_s, dtype=torch.float64, device=device),
            torch.tensor(eta_s, dtype=torch.float64, device=device),
        ))

    # Initializations: n_random_inits diverse + warm-start if available
    all_random = [
        torch.tensor([2.0, -3.0, -5.5]),
        torch.tensor([1.5, -2.5, -6.0]),
        torch.tensor([2.5, -3.5, -5.0]),
        torch.tensor([3.0, -2.0, -6.5]),
        torch.tensor([1.0, -2.0, -5.0]),
        torch.tensor([2.0, -4.0, -6.0]),
        torch.tensor([3.5, -1.5, -5.5]),
        torch.tensor([1.5, -3.5, -7.0]),
        torch.tensor([2.5, -2.0, -4.5]),
        torch.tensor([1.0, -3.0, -6.5]),
    ]
    inits = all_random[:n_random_inits]
    if prev_d_params is not None:
        inits.insert(0, prev_d_params.clone().cpu())
        if verbose:
            print(f"      [warm-start + {n_random_inits} random inits]")

    best_model, best_x0s, best_loss = None, None, float('inf')

    for i_init, init_d in enumerate(inits):
        model = LGN_Battery(n).double().to(device)
        model.s_params.requires_grad_(False)
        model.d_params.data = init_d.clone().to(device)

        # warm-start x₀ from previous checkup
        x0_params = []
        for si, (t_s, eta_s) in enumerate(segments):
            if (i_init == 0 and prev_x0s is not None
                    and si < len(prev_x0s)):
                x0 = prev_x0s[si].clone().to(device)
            else:
                x0 = torch.ones(n, dtype=torch.float64, device=device) * (eta_s[0] / n)
            x0_params.append(nn.Parameter(x0))

        all_params = list(model.parameters()) + x0_params
        opt = torch.optim.Adam(all_params, lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=200, factor=0.5, min_lr=1e-5)

        run_best_loss, run_best_state, run_best_x0 = float('inf'), None, None

        for ep in range(n_epochs):
            opt.zero_grad()
            A = model.get_A()
            C = torch.ones(n, dtype=torch.float64, device=device)

            total_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
            for si, ((t_t, eta_t), x0) in enumerate(zip(seg_tensors, x0_params)):
                At = A.unsqueeze(0) * t_t.unsqueeze(1).unsqueeze(2)
                eAt = torch.matrix_exp(At)
                x_t = torch.einsum('tij,j->ti', eAt, x0)
                pred = x_t @ C
                #clamp denominator
                seg_loss = torch.mean((pred - eta_t)**2) / torch.mean(eta_t**2).clamp_min(1e-8)
                total_loss = total_loss + seg_loss

            total_loss = total_loss / n_seg
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
            sched.step(total_loss.item())

            if total_loss.item() < run_best_loss:
                run_best_loss = total_loss.item()
                run_best_state = {k: v.clone() for k, v in model.state_dict().items()}
                run_best_x0 = [x.data.clone() for x in x0_params]

        if run_best_loss < best_loss:
            best_loss = run_best_loss
            model.load_state_dict(run_best_state)
            best_model = model
            best_x0s = run_best_x0

        if verbose:
            model.load_state_dict(run_best_state)
            taus = model.get_time_constants()
            print(f"      init {i_init}: τ={np.round(taus, 1)}  loss={run_best_loss:.4e}")

    best_model.eval()

    # Per-segment NRMSE
    nrmses = []
    with torch.no_grad():
        A = best_model.get_A()
        C = torch.ones(n, dtype=torch.float64, device=device)
        for si, ((t_t, eta_t), x0) in enumerate(zip(seg_tensors, best_x0s)):
            At = A.unsqueeze(0) * t_t.unsqueeze(1).unsqueeze(2)
            eAt = torch.matrix_exp(At)
            x_t = torch.einsum('tij,j->ti', eAt, x0)
            pred = x_t @ C
            pred_np = pred.cpu().numpy()
            eta_np = segments[si][1]
            nrmse = np.sqrt(np.mean((pred_np - eta_np)**2)) / (np.abs(eta_np).max() + 1e-12)
            nrmses.append(nrmse)

    avg_nrmse = np.mean(nrmses)
    if verbose:
        print(f"    BEST → τ = {np.round(best_model.get_time_constants(), 4)}  "
              f"NRMSE = {avg_nrmse:.5f} (per-seg: {[round(n,5) for n in nrmses]})")

    # return best_x0s
    return best_model, best_x0s, avg_nrmse, best_loss


# ============================================================================
# 4. EIS 3-RC FITTER — widened for SiO
# ============================================================================
def fit_eis_randles_sio(freq, z_re, z_im, n_rc=3):
    """3-RC Randles fit with wider starts/bounds for SiO anode chemistry.
    Uses log-spaced τ starts spanning 1e-4 to 1e3 s.
    """
    Z_data = z_re + 1j * z_im
    Rs_est = z_re.min()
    dR = z_re.max() - z_re.min()

    def obj(p):
        Z = p[0] * np.ones(len(freq), dtype=complex)
        omega = 2 * np.pi * freq
        for i in range(n_rc):
            R, tau = p[1 + 2*i], p[2 + 2*i]
            Z += R / (1 + 1j * omega * tau)
        return np.sum(np.abs(Z - Z_data)**2)

    # Wider starts for SiO: τ spans 4 decades
    starts = [
        [Rs_est, dR*f1, t1, dR*f2, t2, dR*(1-f1-f2), t3]
        for f1 in [0.1, 0.3]
        for f2 in [0.2, 0.4]
        for t1 in [1e-4, 1e-3, 0.01]
        for t2 in [0.1, 1.0, 10.0]
        for t3 in [30, 100, 500]
        if f1 + f2 < 0.95
    ]
    bounds = [
        (0, None),                    # Rs
        (0, None), (1e-5, 1e4),       # R1, τ1
        (0, None), (1e-4, 1e4),       # R2, τ2
        (0, None), (1e-3, 1e4),       # R3, τ3
    ]

    best, best_cost = None, float('inf')
    for p0 in starts:
        try:
            r = optimize.minimize(obj, p0, method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 10000, 'ftol': 1e-15})
            if r.fun < best_cost:
                best_cost = r.fun
                best = r.x
        except Exception:
            continue

    if best is None:
        return None

    params = {'Rs': best[0]}
    taus_out = []
    for i in range(n_rc):
        params[f'R{i+1}'] = best[1 + 2*i]
        params[f'tau{i+1}'] = best[2 + 2*i]
        taus_out.append(best[2 + 2*i])

    # Quality gate: flag if any τ hit bounds
    params['hit_bounds'] = any(
        abs(t - 1e-5) < 1e-6 or abs(t - 1e4) < 1 for t in taus_out)

    return params


# ============================================================================
# 5. ANALYZE ONE MEASUREMENT
# ============================================================================
def analyze_measurement(relax_segments, freq, z_re, z_im,
                        cell_id, checkup, soc, temp_c, i_pulses,
                        device='cpu', prev_model=None, prev_x0s=None,
                        n_random_inits=N_INITS):
    """Run multi-segment LGN-3D on KIT pulse + compare to paired EIS."""
    main_seg = max(relax_segments, key=lambda s: s[0][-1])
    t_main, eta_main = main_seg

    i_pulse = float(np.mean(i_pulses)) if i_pulses else I_PULSE_DEFAULT

    res = dict(
        cell=cell_id, diag=checkup, soc=soc, temp=temp_c,
        eta0=float(eta_main[0]),
        R_pulse_mOhm=float(abs(eta_main[0]) / i_pulse * 1000),
        i_pulse_A=float(i_pulse),
        n_segments=len(relax_segments),
        total_pts=sum(len(s[0]) for s in relax_segments),
        relax_window_s=float(main_seg[0][-1]),
    )

    # ---- LGN-3D multi-segment ----
    prev_d = prev_model.d_params.data if prev_model else None

    model, x0s, nrmse, _ = train_lgn_multiseg(
        relax_segments,
        n_epochs=N_EPOCHS, lr=0.01, device=device, verbose=True,
        prev_d_params=prev_d, prev_x0s=prev_x0s,
        n_random_inits=n_random_inits)

    taus = model.get_time_constants()
    res['tau_full']   = taus.tolist()
    res['nrmse_full'] = float(nrmse)

    # ---- Impedance reconstruction ----
    Z_shape = lgn_to_impedance_shape(model, freq)
    Z_data  = z_re + 1j * z_im
    band    = (freq >= 0.05) & (freq <= 2000)

    if band.sum() >= 3:
        a_fit, Rs_fit, Z_pred = fit_scale_and_Rs(Z_shape[band], Z_data[band])
        Z_pred_full = a_fit * Z_shape + Rs_fit
        comp = compare_impedance(Z_pred_full[band], Z_data[band], freq[band])
        res.update({
            'z_re_corr':    comp['re_corr'],
            'z_im_corr':    comp['im_corr'],
            'z_nrmse':      comp['nrmse'],
            'z_rmse_mOhm':  comp['rmse_mOhm'],
            'Rs_fit':       float(Rs_fit),
        })

    # ---- EIS 3-RC baseline (widened for SiO) ----
    eis_params = fit_eis_randles_sio(freq, z_re, z_im, n_rc=3)
    if eis_params and not eis_params.get('hit_bounds', False):
        res['tau_eis_3rc'] = sorted([eis_params[f'tau{i+1}'] for i in range(3)])
        res['Rs_eis_3rc']  = eis_params['Rs']
        for i in range(3):
            res[f'R{i+1}_eis_3rc'] = eis_params[f'R{i+1}']
    else:
        res['tau_eis_3rc'] = None
        res['eis_3rc_hit_bounds'] = True

    # ---- Curve-fit baseline ----
    cf_taus, cf_nrmse, _ = fit_exponentials(main_seg[0], eta_main, n_exp=3)
    res['tau_cf']   = cf_taus.tolist() if cf_taus is not None else None
    res['nrmse_cf'] = cf_nrmse

    # ---- EIS scalar features ----
    for f_t, lab in [(1000,'Z_1kHz'),(100,'Z_100Hz'),(10,'Z_10Hz'),
                     (1,'Z_1Hz'),(0.1,'Z_01Hz'),(0.05,'Z_005Hz')]:
        fi = np.argmin(np.abs(freq - f_t))
        if np.abs(freq[fi] - f_t) / max(f_t, 0.01) < 0.5:
            res[f'{lab}_re']  = float(z_re[fi])
            res[f'{lab}_im']  = float(z_im[fi])
            res[f'{lab}_mag'] = float(np.abs(Z_data[fi]))

    return res, model, x0s


# ============================================================================
# 6. EIS HELPERS
# ============================================================================
def get_eis_spectrum(eis_checkup):
    valid = [r for r in eis_checkup
             if r.get('valid','0') == '1' and r['z_re_comp_mOhm'] != 'nan']
    if not valid:
        return None, None, None
    freq = np.array([float(r['freq_Hz'])          for r in valid])
    z_re = np.array([float(r['z_re_comp_mOhm'])   for r in valid]) / 1000.0
    z_im = np.array([float(r['z_im_comp_mOhm'])   for r in valid]) / 1000.0
    order = np.argsort(freq)[::-1]
    return freq[order], z_re[order], z_im[order]


# ============================================================================
# 7. PER-CELL PIPELINE
# ============================================================================
def run_kit_cell(pulse_path, eis_path, eoc_path=None,
                 soc=50, is_rt=1, device='cpu'):
    cell_id = os.path.basename(pulse_path) \
                .replace('cell_plsv2_','').replace('.csv','')

    print(f"\n{'#'*70}")
    print(f"# CELL: {cell_id}  |  SOC={soc}%  {'RT' if is_rt else 'OT'}")
    print(f"{'#'*70}")

    pulse_rows = load_csv(pulse_path)
    eis_rows   = load_csv(eis_path)

    pulse_cus = group_by_checkup(pulse_rows, soc, is_rt)
    eis_cus   = group_by_checkup(eis_rows,   soc, is_rt)
    n_cu = min(len(pulse_cus), len(eis_cus))
    print(f"  Checkups paired: {n_cu}")

    # Capacity data
    cap_list = []
    if eoc_path and os.path.exists(eoc_path):
        eoc = load_csv(eoc_path)
        cap_list = [(float(r['cap_aged_est_Ah']), float(r['soh_cap']))
                    for r in eoc
                    if r['cyc_charged'] == '0' and float(r['cap_aged_est_Ah']) > 1]

    temp_c = float(pulse_rows[0].get('age_temp', 25))

    results    = []
    prev_model = None
    prev_x0s   = None         # carry x₀ chain
    t_start    = time.time()

    for ci in range(n_cu):
        print(f"\n  ── CU {ci+1}/{n_cu} ──")

        relax_segs, meta = extract_relaxation_segments(pulse_cus[ci])

        if meta['total_pts'] < 6:
            print(f"    ⚠  Only {meta['total_pts']} relaxation pts, skip")
            continue

        print(f"    Segments: {meta['n_segments']}  |  "
              f"Total pts: {meta['total_pts']}  |  "
              f"Windows: {[f'{w:.0f}s' for w in meta['windows_s']]}  |  "
              f"I_pulse: {[f'{i:.3f}A' for i in meta['i_pulses']]}")

        # Paired EIS
        freq, z_re, z_im = get_eis_spectrum(eis_cus[ci])
        if freq is None or len(freq) < 5:
            print(f"    ⚠  Insufficient EIS data, skip")
            continue
        print(f"    EIS: {len(freq)} freqs, {freq.min():.2f}–{freq.max():.0f} Hz")

        # Adaptive inits: 10 random for CU 1-3, warm-start only for CU 4+
        WARMUP_CUS = 3
        if ci < WARMUP_CUS:
            n_ri = 10   # explore thoroughly
        else:
            n_ri = 0    # warm-start only (prev_model carries the basin)
        
        # Analyze
        res, model, x0s = analyze_measurement(
            relax_segs, freq, z_re, z_im,
            cell_id=cell_id, checkup=ci+1, soc=soc, temp_c=temp_c,
            i_pulses=meta['i_pulses'],
            device=device, prev_model=prev_model, prev_x0s=prev_x0s,
            n_random_inits=n_ri)

        if ci < len(cap_list):
            res['capacity_Ah'] = cap_list[ci][0]
            res['soh_pct']     = cap_list[ci][1]

        results.append(res)
        prev_model = model
        prev_x0s   = x0s    

        # Live summary
        t = res['tau_full']
        eis_t = res.get('tau_eis_3rc')
        print(f"    LGN  τ = [{t[0]:.4f}, {t[1]:.3f}, {t[2]:.2f}] s")
        if eis_t:
            print(f"    EIS  τ = [{eis_t[0]:.4f}, {eis_t[1]:.3f}, {eis_t[2]:.2f}] s")
        else:
            print(f"    EIS  3-RC: hit bounds (SiO baseline unreliable)")
        print(f"    Z_re ρ = {res.get('z_re_corr',0):.4f}  |  "
              f"fit NRMSE = {res['nrmse_full']:.5f}")

    elapsed = time.time() - t_start
    print(f"\n  ✓ {len(results)} CUs in {elapsed:.0f}s")
    return results


# ============================================================================
# 8. BATCH MODE
# ============================================================================
def run_kit_batch(pulse_dir, eis_dir, eoc_dir=None,
                  soc=50, is_rt=1, device='cpu',
                  max_cells=None, skip=0, out_path=None):
    pulse_files = sorted(glob.glob(os.path.join(pulse_dir, 'cell_plsv2_*.csv')))
    total = len(pulse_files)
    pulse_files = pulse_files[skip:]
    if max_cells:
        pulse_files = pulse_files[:max_cells]
    print(f"Found {total} total | skip {skip} | running {len(pulse_files)} cells")

    all_results = []
    for ci, pf in enumerate(pulse_files):
        cell_tag = os.path.basename(pf).replace('cell_plsv2_','').replace('.csv','')
        ef = os.path.join(eis_dir, f'cell_eisv2_{cell_tag}.csv')
        cf = os.path.join(eoc_dir, f'cell_eocv2_{cell_tag}.csv') if eoc_dir else None
        if not os.path.exists(ef):
            continue
        try:
            results = run_kit_cell(pf, ef, cf, soc=soc, is_rt=is_rt, device=device)
            all_results.extend(results)
        except Exception as e:
            print(f"  ✗ FAILED {cell_tag}: {e}")
            continue
        # Incremental save every 5 cells
        if out_path and (ci + 1) % 5 == 0:
            save_results(all_results, out_path)
            print(f"  [checkpoint: {len(all_results)} results saved]")
    return all_results


# ============================================================================
# 9. SUMMARY
# ============================================================================
def print_summary(results):
    if not results:
        print("No results.")
        return

    print(f"\n{'='*110}")
    print(f"  KIT LGN-3D RESULTS (multi-segment, v2.1)")
    print(f"{'='*110}")
    print(f"  {'Cell':<20} {'CU':>3} {'pts':>4} {'I_p':>5} | "
          f"{'τ₁(CT)':>8} {'τ₂(SEI)':>8} {'τ₃(diff)':>8} "
          f"| {'Z_re ρ':>7} {'NRMSE':>7} | {'SOH':>6}")
    print(f"  {'-'*106}")

    for r in results:
        t = r['tau_full']
        soh = f"{r['soh_pct']:.1f}%" if 'soh_pct' in r else '   —'
        print(f"  {r['cell']:<20} {r['diag']:3d} {r['total_pts']:4d} "
              f"{r.get('i_pulse_A', 1.0):5.2f} | "
              f"{t[0]:8.4f} {t[1]:8.3f} {t[2]:8.2f} | "
              f"{r.get('z_re_corr',0):7.4f} {r['nrmse_full']:7.5f} | {soh:>6}")

    z_corrs = [r['z_re_corr'] for r in results if 'z_re_corr' in r]
    nrmses  = [r['nrmse_full'] for r in results]

    print(f"  {'-'*106}")
    print(f"  {'MEAN':<33} | {'':>27} "
          f"| {np.mean(z_corrs):7.4f} {np.mean(nrmses):7.5f} |")

    # τ stability (CV across checkups)
    if len(results) >= 3:
        print(f"\n  τ stability (CV across checkups):")
        for idx, lab in enumerate(['τ₁(CT)', 'τ₂(SEI)', 'τ₃(diff)']):
            vals = [r['tau_full'][idx] for r in results]
            cv = np.std(vals) / (np.mean(vals) + 1e-12) * 100
            print(f"    {lab}: mean={np.mean(vals):.4f}  CV={cv:.1f}%")

    # τ vs SOH
    if any('soh_pct' in r for r in results) and len(results) >= 5:
        print(f"\n  τ vs SOH correlations:")
        sohs = [r['soh_pct'] for r in results if 'soh_pct' in r]
        for idx, lab in enumerate(['τ₁(CT)', 'τ₂(SEI)', 'τ₃(diff)']):
            tvals = [r['tau_full'][idx] for r in results if 'soh_pct' in r]
            rho, p = stats.spearmanr(tvals, sohs)
            sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
            print(f"    {lab} vs SOH: ρ_s = {rho:+.3f}{sig}")

    # τ vs EIS markers
    if len(results) >= 5:
        print(f"\n  τ vs EIS marker correlations:")
        for idx, lab in enumerate(['τ₁(CT)', 'τ₂(SEI)', 'τ₃(diff)']):
            tvals = [r['tau_full'][idx] for r in results]
            for marker in ['Z_1kHz_re', 'Z_1Hz_re', 'Z_01Hz_re']:
                mvals = [r.get(marker) for r in results]
                if all(v is not None for v in mvals):
                    rho, p = stats.spearmanr(tvals, mvals)
                    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else ''
                    if abs(rho) > 0.3 or sig:
                        print(f"    {lab} vs {marker}: ρ_s = {rho:+.3f}{sig}")

    # EIS baseline stats
    n_hit = sum(1 for r in results if r.get('eis_3rc_hit_bounds'))
    if n_hit:
        print(f"\n  ⚠ EIS 3-RC hit bounds on {n_hit}/{len(results)} checkups")

    print(f"{'='*110}")


def save_results(results, out_path):
    out = []
    for r in results:
        s = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):       s[k] = v.tolist()
            elif isinstance(v, np.floating):    s[k] = float(v)
            elif isinstance(v, np.integer):     s[k] = int(v)
            else:                               s[k] = v
        out.append(s)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ {len(out)} results → {out_path}")


# ============================================================================
# 10. CLI
# ============================================================================
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='LGN-3D on KIT (multi-segment v2.1)')
    p.add_argument('--pulse', help='cell_plsv2_*.csv')
    p.add_argument('--eis',   help='cell_eisv2_*.csv')
    p.add_argument('--eoc',   default=None)
    p.add_argument('--pulse_dir', help='Batch: pulse CSV directory')
    p.add_argument('--eis_dir',   help='Batch: EIS CSV directory')
    p.add_argument('--eoc_dir',   default=None)
    p.add_argument('--max_cells', type=int, default=None)
    p.add_argument('--skip', type=int, default=0, help='Skip first N cells (for multi-GPU)')
    p.add_argument('--soc',    type=int, default=50)
    p.add_argument('--rt',     type=int, default=1, choices=[0,1])
    p.add_argument('--device', default='cpu')
    p.add_argument('--out',    default=None)
    args = p.parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available → CPU")
        args.device = 'cpu'
    print(f"Device: {args.device}  |  SOC={args.soc}%  "
          f"{'RT' if args.rt else 'OT'}  |  "
          f"{N_INITS}+ws inits × {N_EPOCHS} epochs")

    if args.pulse_dir:
        default_out = args.out or f'results_kit_batch_SOC{args.soc}_skip{args.skip}.json'
        results = run_kit_batch(
            args.pulse_dir, args.eis_dir, args.eoc_dir,
            soc=args.soc, is_rt=args.rt, device=args.device,
            max_cells=args.max_cells, skip=args.skip,
            out_path=default_out)
    elif args.pulse and args.eis:
        results = run_kit_cell(
            args.pulse, args.eis, args.eoc,
            soc=args.soc, is_rt=args.rt, device=args.device)
        cell_id = os.path.basename(args.pulse) \
                    .replace('cell_plsv2_','').replace('.csv','')
        default_out = f'results_kit_{cell_id}_SOC{args.soc}.json'
    else:
        p.error("Provide --pulse + --eis, or --pulse_dir + --eis_dir")

    print_summary(results)
    save_results(results, args.out or default_out)
