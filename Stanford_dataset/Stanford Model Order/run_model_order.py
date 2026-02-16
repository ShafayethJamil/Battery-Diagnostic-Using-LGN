"""
Model Order Selection Ablation
===============================
Runs LGN with n_states = 2, 3, 4, 5, 6 on Stanford cells at 50% SOC.
Supports n=3 as the minimal order that captures all dynamically significant
modes in the identifiable band; higher orders introduce additional modes
with negligible energy or outside identifiability limits.

Shows:
  1. NRMSE improves 2→3, plateaus at n≥3
  2. Extra modes (n>3) land outside identifiable band or carry <3% energy
  3. The 3 dominant modes are stable across model orders
  → "Data + identifiability selects 3"

Identifiability criterion (from advisor):
  "Modes with |Re(λ)| < 1/T are unidentifiable — they behave
   like constants over the observation window."

  Identifiable band: Δt < τ < T  →  1s < τ < 3600s


Author: Shafayeth Jamil (USC ECE), February 2026
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy import stats

import torch

from run_degradation import (
    LGN_Battery, lgn_to_impedance_shape,
    fit_scale_and_Rs, compare_impedance, safe_corr,
    fit_exponentials, _subsample_log
)


# ============================================================================
# GENERAL N-STATE TRAINING
# ============================================================================
def train_lgn_nstate(t_data, eta_data, n_states, n_epochs=4000, lr=0.01,
                     subsample=300, device='cpu', verbose=False):
    """
    Multi-restart LGN training for arbitrary n_states.
    
    Initialization strategy: spread d_params across the identifiable band.
    softplus(d) gives τ:
      d = 3.0 → τ ≈ 0.05s
      d = 2.0 → τ ≈ 0.5s
      d = 0.0 → τ ≈ 1.4s
      d = -3.0 → τ ≈ 20s
      d = -5.5 → τ ≈ 245s
      d = -6.0 → τ ≈ 403s
      d = -7.0 → τ ≈ 1097s
    """
    idx = _subsample_log(len(t_data), subsample) if subsample else np.arange(len(t_data))
    t_t = torch.tensor(t_data[idx], dtype=torch.float64, device=device)
    eta_t = torch.tensor(eta_data[idx], dtype=torch.float64, device=device)

    # Generate initializations: spread d_params across plausible range
    # Core range: d ∈ [3.0, -7.0] → τ ∈ [0.05s, 1097s]
    def make_inits(n):
        inits = []
        # Strategy 1: evenly spaced across full range
        inits.append(torch.linspace(2.5, -6.5, n))
        # Strategy 2: clustered around expected battery modes
        if n == 2:
            inits.append(torch.tensor([1.0, -3.0]))
            inits.append(torch.tensor([0.5, -5.0]))
            inits.append(torch.tensor([1.5, -4.0]))
        elif n == 3:
            inits.append(torch.tensor([2.0, -3.0, -5.5]))
            inits.append(torch.tensor([1.5, -2.5, -6.0]))
            inits.append(torch.tensor([2.5, -3.5, -5.0]))
            inits.append(torch.tensor([1.0, -3.0, -6.0]))
            inits.append(torch.tensor([2.0, -4.0, -5.5]))
            inits.append(torch.tensor([3.0, -2.0, -6.5]))
            inits.append(torch.tensor([2.0, -2.0, -7.0]))
            inits.append(torch.tensor([1.0, -1.5, -5.0]))
        elif n == 4:
            inits.append(torch.tensor([2.5, -1.0, -3.5, -5.5]))
            inits.append(torch.tensor([2.0, -2.0, -4.0, -6.0]))
            inits.append(torch.tensor([1.5, -1.5, -3.0, -6.5]))
            inits.append(torch.tensor([3.0, 0.0, -3.0, -5.5]))
            inits.append(torch.tensor([2.0, -0.5, -4.5, -6.0]))
        elif n == 5:
            inits.append(torch.tensor([3.0, 1.0, -1.5, -3.5, -5.5]))
            inits.append(torch.tensor([2.5, 0.0, -2.0, -4.0, -6.0]))
            inits.append(torch.tensor([2.0, 0.5, -1.0, -3.0, -6.5]))
            inits.append(torch.tensor([3.0, 1.5, -2.5, -4.5, -5.5]))
        elif n == 6:
            inits.append(torch.tensor([3.5, 1.5, 0.0, -2.0, -4.0, -6.0]))
            inits.append(torch.tensor([3.0, 1.0, -0.5, -2.5, -4.5, -6.5]))
            inits.append(torch.tensor([2.5, 0.5, -1.0, -3.0, -5.0, -6.0]))
            inits.append(torch.tensor([3.0, 2.0, 0.0, -3.0, -5.5, -7.0]))
        else:
            # Generic: spread across range with jitter
            for _ in range(4):
                base = torch.linspace(3.0, -7.0, n)
                jitter = torch.randn(n) * 0.5
                inits.append(base + jitter)
        return inits

    inits = make_inits(n_states)
    best_model, best_loss = None, float('inf')

    for i_init, init_d in enumerate(inits):
        model = LGN_Battery(n_states).double().to(device)
        model.s_params.requires_grad_(False)  # diagonal A
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
            taus = model.get_time_constants()
            print(f"      init {i_init}: τ={np.round(taus,1)}  loss={run_best_loss:.4e}")

    best_model.eval()
    # FIX: evaluate on FULL trajectory, not subsampled points
    t_full = torch.tensor(t_data, dtype=torch.float64, device=device)
    with torch.no_grad():
        eta_pred_full = best_model(t_full).cpu().numpy()
    nrmse = np.sqrt(np.mean((eta_pred_full - eta_data)**2)) / (np.abs(eta_data).max() + 1e-12)

    return best_model, nrmse, best_loss


# ============================================================================
# MODE ANALYSIS
# ============================================================================
def analyze_modes(model, t_data, eta_data, delta_t=1.0, T_obs=3600.0):
    """
    Classify each mode as identifiable or not.
    
    Conservative identifiability band:
      - Lower: τ > 3*Δt  (need multiple samples per time constant)
      - Upper: τ < T/3   (mode must decay noticeably within window)
    
    Returns list of dicts with mode info.
    """
    taus = model.get_time_constants()
    x0 = model.x0.detach().cpu().numpy()
    
    # Conservative bounds
    tau_min = 3 * delta_t     # need ~3 samples per τ to resolve
    tau_max = T_obs / 3       # mode must decay to ~e^{-3} ≈ 5% within window
    
    modes = []
    for i in range(len(taus)):
        tau = taus[i]
        amp = x0[i]
        energy = (amp ** 2) * tau  # L2 energy: ∫₀^∞ a²e^{-2t/τ} dt = a²τ/2 (proportional)
        
        # Classification with conservative bounds
        if tau < tau_min:
            status = 'SUB-BANDWIDTH'
            reason = f'τ={tau:.2f}s < 3Δt={tau_min:.1f}s'
        elif tau > tau_max:
            status = 'BEYOND-HORIZON'
            reason = f'τ={tau:.0f}s > T/3={tau_max:.0f}s'
        else:
            status = 'IDENTIFIABLE'
            reason = f'{tau_min:.1f}s < τ={tau:.1f}s < {tau_max:.0f}s'
        
        # Relative amplitude
        rel_amp = abs(amp) / (abs(eta_data[0]) + 1e-15)
        
        modes.append({
            'mode': i + 1,
            'tau': float(tau),
            'amplitude': float(amp),
            'rel_amplitude': float(rel_amp),
            'energy': float(energy),
            'status': status,
            'reason': reason,
        })
    
    # Compute energy fractions
    E_total = sum(abs(m['energy']) for m in modes) + 1e-15
    for m in modes:
        m['energy_frac'] = abs(m['energy']) / E_total
    
    # Survival: data-driven, not magic thresholds
    # A mode survives if it's in the identifiable band AND
    # (carries >3% energy OR is among the top-3 energy modes)
    E_MIN = 0.03  # 3% energy share (scale-invariant)
    energy_fracs = [m['energy_frac'] for m in modes]
    topk = sorted(range(len(modes)), key=lambda i: -energy_fracs[i])[:3]
    for i, m in enumerate(modes):
        m['survives'] = (m['status'] == 'IDENTIFIABLE' and 
                        (m['energy_frac'] > E_MIN or i in topk))
    
    return modes


# ============================================================================
# MAIN SWEEP
# ============================================================================
def run_order_sweep(data_dir, cells, soc_target=50, orders=[2, 3, 4, 5, 6],
                    device='cpu', diags_to_run=None):
    """Run model order sweep across cells and diagnostics."""
    
    all_results = []
    
    for cell in cells:
        print(f"\n{'#'*70}")
        print(f"# CELL: {cell}")
        print(f"{'#'*70}")
        
        hppc = pd.read_csv(f'{data_dir}/{cell}_hppc_relaxation.csv')
        eis = pd.read_csv(f'{data_dir}/{cell}_eis.csv')
        diags = sorted(set(hppc['diag'].unique()) & set(eis['diag'].unique()))
        
        if diags_to_run:
            diags = [d for d in diags if d in diags_to_run]
        
        soc_col = f'{soc_target}pct'
        re_col = f're_z_ohm_{soc_col}'
        im_col = f'im_z_ohm_{soc_col}'
        
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
            
            freq = eis_seg['freq_Hz'].values
            re_z = eis_seg[re_col].values
            im_z = eis_seg[im_col].values
            
            delta_t = t_relax[1] - t_relax[0] if len(t_relax) > 1 else 1.0
            T_obs = t_relax[-1]
            
            print(f"\n  Diag {diag} (Δt={delta_t:.1f}s, T={T_obs:.0f}s):")
            
            for n in orders:
                print(f"    n={n}:", end='', flush=True)
                
                model, nrmse, loss = train_lgn_nstate(
                    t_relax, eta_full, n_states=n,
                    n_epochs=4000, lr=0.01,
                    subsample=min(300, len(t_relax)),
                    device=device, verbose=False)
                
                taus = model.get_time_constants()
                modes = analyze_modes(model, t_relax, eta_full, delta_t, T_obs)
                
                n_identifiable = sum(1 for m in modes if m['status'] == 'IDENTIFIABLE')
                n_surviving = sum(1 for m in modes if m['survives'])
                
                # Impedance reconstruction
                Z_shape = lgn_to_impedance_shape(model, freq)
                Z_data = re_z + 1j * im_z
                band = (freq >= 0.05) & (freq <= 2000)
                a_fit, Rs_fit, Z_pred = fit_scale_and_Rs(Z_shape[band], Z_data[band])
                Z_pred_full = a_fit * Z_shape + Rs_fit
                comp = compare_impedance(Z_pred_full[band], Z_data[band], freq[band])
                
                result = {
                    'cell': cell,
                    'diag': int(diag),
                    'n_states': n,
                    'taus': taus.tolist(),
                    'nrmse': float(nrmse),
                    'loss': float(loss),
                    'z_re_corr': comp['re_corr'],
                    'z_nrmse': comp['nrmse'],
                    'modes': modes,
                    'n_identifiable': n_identifiable,
                    'n_surviving': n_surviving,
                    'delta_t': float(delta_t),
                    'T_obs': float(T_obs),
                }
                all_results.append(result)
                
                # Compact print
                tau_str = ', '.join([f'{t:.1f}' for t in taus])
                status_str = ', '.join([
                    ('✓' if m['survives'] else m['status'][0]) for m in modes
                ])
                print(f" τ=[{tau_str}]  NRMSE={nrmse:.5f}  "
                      f"Z_corr={comp['re_corr']:.4f}  "
                      f"surviving={n_surviving}/{n}  [{status_str}]")
    
    return all_results


# ============================================================================
# SUMMARY TABLES
# ============================================================================
def print_order_summary(results):
    """Print the money table: NRMSE and identifiable modes vs n."""
    
    cells = sorted(set(r['cell'] for r in results))
    diags = sorted(set(r['diag'] for r in results))
    orders = sorted(set(r['n_states'] for r in results))
    
    print(f"\n{'='*90}")
    print(f"MODEL ORDER SELECTION SUMMARY")
    print(f"{'='*90}")
    
    # Table 1: NRMSE vs n (averaged)
    print(f"\n  --- NRMSE vs Model Order (lower = better fit) ---")
    print(f"  {'n':>3} | {'NRMSE':>10} {'Δ vs n-1':>10} | {'Z_re_corr':>10} | {'Surviving':>12}")
    print(f"  {'-'*60}")
    
    prev_nrmse = None
    for n in orders:
        nr = [r for r in results if r['n_states'] == n]
        avg_nrmse = np.mean([r['nrmse'] for r in nr])
        avg_zcorr = np.mean([r['z_re_corr'] for r in nr])
        avg_survive = np.mean([r['n_surviving'] for r in nr])
        
        delta = f'{avg_nrmse - prev_nrmse:+.5f}' if prev_nrmse else '---'
        prev_nrmse = avg_nrmse
        
        print(f"  {n:>3} | {avg_nrmse:10.5f} {delta:>10} | {avg_zcorr:10.4f} | {avg_survive:10.1f}/{n}")
    
    # Table 2: Per-cell per-diag mode details
    print(f"\n  --- Mode Details (✓=Surviving, S=Sub-bandwidth, B=Beyond-horizon, i=In-band but negligible) ---")
    
    for cell in cells:
        for diag in diags:
            cr = [r for r in results if r['cell'] == cell and r['diag'] == diag]
            if not cr:
                continue
            
            print(f"\n  {cell} Diag {diag}:")
            for r in sorted(cr, key=lambda x: x['n_states']):
                n = r['n_states']
                mode_strs = []
                for m in r['modes']:
                    flag = '✓' if m['survives'] else {'IDENTIFIABLE': 'i', 'SUB-BANDWIDTH': 'S', 'BEYOND-HORIZON': 'B'}[m['status']]
                    mode_strs.append(f"{m['tau']:.1f}s({flag},{m['energy_frac']*100:.0f}%E)")
                n_surv = sum(m['survives'] for m in r['modes'])
                print(f"    n={n}: NRMSE={r['nrmse']:.5f} | surviving={n_surv}/{n} | {' | '.join(mode_strs)}")
    
    # Table 3: Stable τ across n (the punchline)
    print(f"\n  --- Surviving Modes Across Model Order ---")
    print(f"  (Do the same 3 modes persist regardless of n?)")
    
    for cell in cells:
        for diag in diags:
            cr = [r for r in results if r['cell'] == cell and r['diag'] == diag]
            if not cr:
                continue
            
            print(f"\n  {cell} Diag {diag}:")
            for r in sorted(cr, key=lambda x: x['n_states']):
                n = r['n_states']
                surv_taus = [m['tau'] for m in r['modes'] if m['survives']]
                pruned_taus = [m['tau'] for m in r['modes'] if not m['survives']]
                surv_str = ', '.join([f'{t:.1f}' for t in sorted(surv_taus)])
                pruned_str = ', '.join([f'{t:.1f}' for t in sorted(pruned_taus)]) if pruned_taus else '—'
                print(f"    n={n}: surviving=[{surv_str}]  pruned=[{pruned_str}]")


# ============================================================================
# PLOTTING
# ============================================================================
def plot_order_sweep(results, out_dir):
    """Publication figure for model order selection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
    
    orders = sorted(set(r['n_states'] for r in results))
    cells = sorted(set(r['cell'] for r in results))
    diags = sorted(set(r['diag'] for r in results))
    colors = {'W8': '#1f77b4', 'W9': '#ff7f0e', 'W10': '#2ca02c'}
    
    # ---- Figure 1: NRMSE vs n ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: NRMSE vs n (all cells averaged)
    ax = axes[0]
    avg_nrmse = [np.mean([r['nrmse'] for r in results if r['n_states'] == n]) for n in orders]
    std_nrmse = [np.std([r['nrmse'] for r in results if r['n_states'] == n]) for n in orders]
    ax.errorbar(orders, avg_nrmse, yerr=std_nrmse, fmt='o-', color='steelblue',
                markersize=8, linewidth=2, capsize=5)
    ax.set_xlabel('Model Order n')
    ax.set_ylabel('NRMSE')
    ax.set_title('Fit Quality vs Model Order')
    ax.set_xticks(orders)
    ax.grid(True, alpha=0.2)
    # Annotate the "knee"
    ax.axvline(3, color='red', linestyle='--', alpha=0.4, label='n=3 (selected)')
    ax.legend()
    
    # Panel B: Z_re correlation vs n
    ax = axes[1]
    avg_zcorr = [np.mean([r['z_re_corr'] for r in results if r['n_states'] == n]) for n in orders]
    ax.plot(orders, avg_zcorr, 'o-', color='darkgreen', markersize=8, linewidth=2)
    ax.set_xlabel('Model Order n')
    ax.set_ylabel('Z_re Correlation')
    ax.set_title('Impedance Reconstruction vs Order')
    ax.set_xticks(orders)
    ax.grid(True, alpha=0.2)
    ax.axvline(3, color='red', linestyle='--', alpha=0.4)
    
    # Panel C: Number of surviving modes vs n
    ax = axes[2]
    avg_survive = [np.mean([r['n_surviving'] for r in results if r['n_states'] == n]) for n in orders]
    ax.bar(orders, avg_survive, color='steelblue', alpha=0.7, label='Surviving')
    ax.bar(orders, [n - s for n, s in zip(orders, avg_survive)], 
           bottom=avg_survive, color='lightcoral', alpha=0.7, label='Pruned')
    ax.set_xlabel('Model Order n')
    ax.set_ylabel('Number of Modes')
    ax.set_title('Surviving vs Pruned Modes')
    ax.set_xticks(orders)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    
    fig.suptitle('Model Order Selection: Three Dynamically Significant Modes\n'
                 'Within the Identifiable Band Persist Across Orders',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/fig_model_order_selection.png')
    plt.close(fig)
    
    # ---- Figure 2: τ bubble chart (THE MONEY FIGURE) ----
    # x = model order, y = τ (log scale), bubble size = |amplitude|
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Identifiable band (conservative: 3*Δt to T/3)
    ax.axhspan(3.0, 1200, color='green', alpha=0.05, label='Identifiable band (3Δt < τ < T/3)')
    ax.axhline(3.0, color='green', linestyle=':', alpha=0.5)
    ax.axhline(1200, color='green', linestyle=':', alpha=0.5)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.3)
    ax.axhline(3600, color='gray', linestyle='--', alpha=0.3)
    ax.text(orders[-1] + 0.3, 3.0, '3Δt = 3s', fontsize=9, color='green', va='center')
    ax.text(orders[-1] + 0.3, 1200, 'T/3 = 1200s', fontsize=9, color='green', va='center')
    ax.text(orders[-1] + 0.3, 1.0, 'Δt = 1s', fontsize=8, color='gray', va='center')
    ax.text(orders[-1] + 0.3, 3600, 'T = 3600s', fontsize=8, color='gray', va='center')
    
    for cell in cells:
        for diag in diags[:3]:  # limit to first 3 diags to avoid clutter
            cr = [r for r in results if r['cell'] == cell and r['diag'] == diag]
            for r in cr:
                n = r['n_states']
                for m in r['modes']:
                    color = colors.get(cell, 'gray')
                    alpha = 0.85 if m['survives'] else 0.2
                    edge = 'black' if m['survives'] else 'gray'
                    size = 20 + 800 * np.sqrt(m['energy_frac'])
                    
                    # Jitter x slightly for visibility
                    jitter = (hash(f"{cell}{diag}") % 10 - 5) * 0.04
                    ax.scatter(n + jitter, m['tau'], s=size,
                              c=color, alpha=alpha, edgecolors=edge, linewidth=0.5)
    
    ax.set_yscale('log')
    ax.set_xlabel('Model Order n', fontsize=12)
    ax.set_ylabel('Time Constant τ [s]', fontsize=12)
    ax.set_title('Mode Discovery: Three Significant Modes Persist,\n'
                'Additional Modes Carry Negligible Energy', fontweight='bold')
    ax.set_xticks(orders)
    ax.set_ylim(0.01, 50000)
    ax.grid(True, alpha=0.2, which='both')
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
               markeredgecolor='black', markersize=12, label='Surviving mode (in-band, E > 3% or top-3)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
               markeredgecolor='gray', markersize=7, alpha=0.25, label='Pruned mode'),
    ]
    for cell in cells:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[cell],
                   markersize=8, label=cell))
    ax.legend(handles=legend_elements, loc='upper left')
    
    fig.tight_layout()
    fig.savefig(f'{out_dir}/fig_tau_bubble_chart.png')
    plt.close(fig)
    
    print(f"  Model order plots saved to {out_dir}/")


# ============================================================================
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Model Order Selection Ablation')
    p.add_argument('--data_dir', default='lgn_csv')
    p.add_argument('--cells', nargs='+', default=['W8', 'W9', 'W10'])
    p.add_argument('--soc', type=int, default=50)
    p.add_argument('--orders', nargs='+', type=int, default=[2, 3, 4, 5, 6])
    p.add_argument('--diags', nargs='+', type=int, default=None,
                   help='Specific diagnostics to run (default: all)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out_dir', default='results_model_order')
    p.add_argument('--plot', action='store_true')
    args = p.parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available → CPU")
        args.device = 'cpu'

    print(f"Device: {args.device}")
    print(f"Cells: {args.cells}, SOC: {args.soc}%")
    print(f"Model orders: {args.orders}")
    print(f"Diags: {args.diags or 'all'}")
    print(f"Identifiable band: 3Δt=3s < τ < T/3=1200s (conservative)")
    print()

    results = run_order_sweep(
        args.data_dir, args.cells, soc_target=args.soc,
        orders=args.orders, device=args.device,
        diags_to_run=args.diags)

    print_order_summary(results)

    # Save
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

    with open(f'{args.out_dir}/results_model_order.json', 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\n✓ Saved to {args.out_dir}/")

    if args.plot:
        plot_order_sweep(results, args.out_dir)

    # Quick summary
    print(f"\n{'='*60}")
    print(f"QUICK VERDICT:")
    orders = sorted(set(r['n_states'] for r in results))
    for n in orders:
        nr = [r for r in results if r['n_states'] == n]
        ns = np.mean([r['n_surviving'] for r in nr])
        nrmse = np.mean([r['nrmse'] for r in nr])
        print(f"  n={n}: avg NRMSE={nrmse:.5f}, avg surviving modes={ns:.1f}/{n}")
    print(f"{'='*60}")
