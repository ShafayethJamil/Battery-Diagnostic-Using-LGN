#!/usr/bin/env python3
"""
extract_amplitudes.py — Recover R₁, R₂, R₃, Rs from stored τ + raw HPPC traces
=================================================================================
No GPU, no LGN model needed. Pure linear algebra. Runs in minutes on CPU.

For each record in your results JSON:
  1. Load the raw CSV, extract the same relaxation segments
  2. Rs: voltage jump at pulse→relax transition (V_relax[0] - V_load) / I
  3. R₁,R₂,R₃: fix τ from JSON, solve η(t) = Σ aᵢ·exp(-t/τᵢ) via lstsq
  4. Rᵢ = |aᵢ| / I_pulse

All features are 100% pulse-derived — zero EIS leakage.

Usage:
  python extract_amplitudes.py \
      --results results_kit_gpu0.json \
      --pulse_dir /path/to/pulse_csvs/ \
      --out results_kit_gpu0_augmented.json

Author: Shafayeth Jamil (USC ECE), February 2026
"""
import argparse, csv, json, os, sys
import numpy as np

# ============================================================================
# CONSTANTS — must match run_kit.py
# ============================================================================
I_THRESH_OFF = 0.1
I_THRESH_ON  = 0.5
CU_GAP_S     = 86400


# ============================================================================
# DATA LOADING — from run_kit.py
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
# SEGMENT + Rs EXTRACTION
# ============================================================================
def extract_segments_and_Rs(points):
    """Extract relaxation segments AND Rs from voltage jump.
    
    Returns:
        relax_pairs: list of (t, eta) arrays
        i_pulses: list of pulse currents
        Rs_values: list of Rs (Ohms) from each discharge→relax transition
    """
    vs   = np.array([float(r['v_raw_V']) for r in points])
    amps = np.array([float(r['i_raw_A']) for r in points])
    ts   = np.array([float(r['timestamp_s']) for r in points])
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

    # Keep only relaxations preceded by a real current pulse
    segments = []
    for s, e in raw_segments:
        if s > 0:
            pre = amps[max(0, s-3):s]
            if np.any(np.abs(pre) > I_THRESH_ON):
                segments.append((s, e))

    # Build (t, eta) + extract Rs from voltage jump
    relax_pairs = []
    i_pulses = []
    Rs_values = []

    for s, e in segments:
        t_seg = t[s:e] - t[s]
        v_seg = vs[s:e]
        v_inf = v_seg[-1]
        eta_seg = v_seg - v_inf

        # Pulse current from preceding segment
        pre_start = max(0, s - 8)
        pre_amps = np.abs(amps[pre_start:s])
        active = pre_amps[pre_amps > I_THRESH_ON]
        i_pulse = float(np.median(active)) if len(active) > 0 else 1.0

        # Rs from voltage jump: find last point under load before this relax
        # Look backwards from s for last sample with |I| > I_THRESH_ON
        j = s - 1
        while j >= 0 and abs(amps[j]) <= I_THRESH_ON:
            j -= 1

        if j >= 0 and abs(amps[j]) > I_THRESH_ON:
            V_load  = vs[j]
            I_load  = abs(amps[j])
            V_relax = vs[s]  # first relaxation sample
            Rs = abs(V_relax - V_load) / I_load  # Ohms
        else:
            Rs = np.nan

        relax_pairs.append((t_seg, eta_seg))
        i_pulses.append(i_pulse)
        Rs_values.append(Rs)

    return relax_pairs, i_pulses, Rs_values


# ============================================================================
# AMPLITUDE RECOVERY: lstsq with fixed τ
# ============================================================================
def recover_amplitudes(segments, taus):
    """η(t) = a₁·exp(-t/τ₁) + a₂·exp(-t/τ₂) + a₃·exp(-t/τ₃)
    
    Linear in [a₁, a₂, a₃]. One lstsq per segment, then average.
    """
    taus = np.array(taus)
    all_a = []

    for t_seg, eta_seg in segments:
        Phi = np.exp(-t_seg[:, None] / taus[None, :])
        a, _, _, _ = np.linalg.lstsq(Phi, eta_seg, rcond=None)
        all_a.append(a)

    a_avg = np.mean(all_a, axis=0) if all_a else np.zeros(len(taus))
    return all_a, a_avg


# ============================================================================
# PROCESS ONE CELL
# ============================================================================
def process_cell(pulse_csv, records, soc=50, is_rt=1):
    rows = load_csv(pulse_csv)
    checkups = group_by_checkup(rows, soc=soc, is_rt=is_rt)

    augmented = 0
    for rec in records:
        diag = rec['diag']
        ci = diag - 1

        if ci >= len(checkups):
            continue

        segments, i_pulses, Rs_values = extract_segments_and_Rs(checkups[ci])
        if not segments:
            continue

        taus = rec['tau_full']
        all_a, a_avg = recover_amplitudes(segments, taus)

        i_avg = np.mean(i_pulses) if i_pulses else 1.0

        # R values from amplitudes
        R_vals = np.abs(a_avg) / i_avg  # Ohms
        
        # Rs from voltage jump (average over discharge transitions)
        valid_Rs = [rs for rs in Rs_values if not np.isnan(rs)]
        Rs_jump = np.mean(valid_Rs) if valid_Rs else np.nan

        # Store everything (in Ohms)
        rec['x0_recovered']  = a_avg.tolist()
        rec['R1_pulse']      = float(R_vals[0])             # Ohms
        rec['R2_pulse']      = float(R_vals[1])
        rec['R3_pulse']      = float(R_vals[2])
        rec['Rs_jump']       = float(Rs_jump)               # Ohms
        rec['Rs_jump_mOhm']  = float(Rs_jump * 1000)        # mΩ
        rec['R1_pulse_mOhm'] = float(R_vals[0] * 1000)      # mΩ
        rec['R2_pulse_mOhm'] = float(R_vals[1] * 1000)
        rec['R3_pulse_mOhm'] = float(R_vals[2] * 1000)

        augmented += 1

    return augmented


# ============================================================================
# MAIN
# ============================================================================
def main():
    p = argparse.ArgumentParser(
        description='Extract R₁,R₂,R₃,Rs from stored τ + raw HPPC')
    p.add_argument('--results', required=True,
                   help='Input JSON (e.g. results_kit_gpu0.json)')
    p.add_argument('--pulse_dir', required=True,
                   help='Directory with cell_plsv2_*.csv files')
    p.add_argument('--soc', type=int, default=50)
    p.add_argument('--rt', type=int, default=1)
    p.add_argument('--out', default=None,
                   help='Output JSON (default: input_augmented.json)')
    args = p.parse_args()

    out_path = args.out or args.results.replace('.json', '_augmented.json')

    with open(args.results) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} records from {args.results}")

    # Group by cell
    cells = {}
    for r in results:
        c = r['cell']
        if c not in cells:
            cells[c] = []
        cells[c].append(r)

    print(f"Processing {len(cells)} cells...")
    total_aug = 0
    for ci, (cell_id, recs) in enumerate(sorted(cells.items())):
        pulse_csv = os.path.join(args.pulse_dir,
                                 f'cell_plsv2_{cell_id}.csv')
        if not os.path.exists(pulse_csv):
            print(f"  [{ci+1}] {cell_id}: CSV not found, skip")
            continue

        n = process_cell(pulse_csv, recs, soc=args.soc, is_rt=args.rt)
        total_aug += n

        if (ci + 1) % 20 == 0 or ci == 0:
            r = recs[0]
            if 'Rs_jump_mOhm' in r:
                print(f"  [{ci+1}/{len(cells)}] {cell_id}: {n} aug | "
                      f"Rs_jump={r['Rs_jump_mOhm']:.1f} "
                      f"Rs_fit={r.get('Rs_fit',0)*1000:.1f} mΩ | "
                      f"R=[{r['R1_pulse_mOhm']:.1f}, "
                      f"{r['R2_pulse_mOhm']:.1f}, "
                      f"{r['R3_pulse_mOhm']:.1f}] mΩ")

    # Save
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

    print(f"\n{'='*60}")
    print(f"  Augmented {total_aug}/{len(results)} records")
    print(f"  Output: {out_path}")
    print(f"  New fields: Rs_jump, R1/R2/R3_pulse (+ _mOhm versions)")
    print(f"{'='*60}")

    # Sanity check: Rs_jump vs Rs_fit
    aug = [r for r in results if 'Rs_jump_mOhm' in r and 'Rs_fit' in r]
    if aug:
        rj = np.array([r['Rs_jump_mOhm'] for r in aug])
        rf = np.array([r['Rs_fit'] * 1000 for r in aug])
        rho = np.corrcoef(rj, rf)[0, 1]
        ratio = np.median(rj / rf)
        print(f"\n  Rs_jump vs Rs_fit (EIS):")
        print(f"    corr = {rho:.4f}")
        print(f"    median ratio = {ratio:.4f}")
        print(f"    Rs_jump: {np.median(rj):.2f} mΩ (median)")
        print(f"    Rs_fit:  {np.median(rf):.2f} mΩ (median)")


if __name__ == '__main__':
    main()
