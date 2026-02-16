%% fig_soh_vs_eis_v5.m
%  ========================================================================
%  SOH Tracking + Mechanistic Decomposition (fleet-wide)
%
%  Panel (a): Median |ρ| bars — τ₁ vs R_pulse vs 5 EIS freqs
%  Panel (b): Dot strip — last/first ratio for R₁↑, τ₁↓, C₁↓↓ (194 cells)
%  Panel (c): Histogram — τ₁ |ρ| vs best EIS |ρ| distributions
%
%  Author: Shafayeth Jamil (USC ECE), February 2026
%  ========================================================================
clear; clc; close all;

%% ---- 1. LOAD DATA -------------------------------------------------------
fprintf('Loading augmented JSON files...\n');
records = {};
for gi = 0:3
    fname = sprintf('results_kit_gpu%d_aug.json', gi);
    if ~isfile(fname), warning('Not found: %s', fname); continue; end
    raw = jsondecode(fileread(fname));
    if iscell(raw)
        records = [records; raw(:)];
    else
        for k = 1:numel(raw)
            records{end+1,1} = raw(k);
        end
    end
end
fprintf('  Total records: %d\n', numel(records));

%% ---- 2. EXTRACT ARRAYS --------------------------------------------------
N_raw = numel(records);
keep = true(N_raw, 1);

cell_ids = strings(N_raw, 1);
diag_num = zeros(N_raw, 1);
soh      = zeros(N_raw, 1);
tau1     = zeros(N_raw, 1);
Rp       = zeros(N_raw, 1);
R1       = zeros(N_raw, 1);
Z1k      = zeros(N_raw, 1);
Z100     = zeros(N_raw, 1);
Z10      = zeros(N_raw, 1);
Z1       = zeros(N_raw, 1);
Z01      = zeros(N_raw, 1);

for i = 1:N_raw
    try
        r = records{i};
        if isempty(r.soh_pct) || isempty(r.tau_full) || ...
           isempty(r.R_pulse_mOhm) || isempty(r.R1_pulse_mOhm) || ...
           isempty(r.Z_1kHz_re) || isempty(r.Z_01Hz_re)
            keep(i) = false; continue;
        end
        if r.R1_pulse_mOhm <= 0
            keep(i) = false; continue;
        end
        cell_ids(i) = string(r.cell);
        diag_num(i) = r.diag;
        soh(i)      = r.soh_pct;
        tau1(i)     = r.tau_full(1);
        Rp(i)       = r.R_pulse_mOhm;
        R1(i)       = r.R1_pulse_mOhm;
        Z1k(i)      = r.Z_1kHz_re;
        Z100(i)     = r.Z_100Hz_re;
        Z10(i)      = r.Z_10Hz_re;
        Z1(i)       = r.Z_1Hz_re;
        Z01(i)      = r.Z_01Hz_re;
    catch
        keep(i) = false;
    end
end

cell_ids = cell_ids(keep);
diag_num = diag_num(keep);
soh  = soh(keep);
tau1 = tau1(keep);
Rp   = Rp(keep);
R1   = R1(keep);
Z1k  = Z1k(keep); Z100 = Z100(keep); Z10 = Z10(keep);
Z1   = Z1(keep);  Z01  = Z01(keep);

[unique_cells, ~, cidx] = unique(cell_ids);
n_cells = numel(unique_cells);
fprintf('  After filtering: %d records, %d cells\n', sum(keep), n_cells);

%% ---- 3. WITHIN-CELL SPEARMAN |ρ| ----------------------------------------
MIN_CUS = 5;

comp_labels = {'R_{pulse}', 'Z@1kHz', 'Z@100Hz', 'Z@10Hz', 'Z@1Hz', 'Z@0.1Hz'};
comp_data   = [Rp, Z1k, Z100, Z10, Z1, Z01];
n_comp = numel(comp_labels);

rho_tau1 = nan(n_cells, 1);
rho_comp = nan(n_cells, n_comp);
n_cus    = zeros(n_cells, 1);

for ci = 1:n_cells
    mask = (cidx == ci);
    n_cus(ci) = sum(mask);
    if n_cus(ci) < MIN_CUS, continue; end
    
    s = soh(mask);
    rho_tau1(ci) = abs(corr(tau1(mask), s, 'Type', 'Spearman'));
    for fi = 1:n_comp
        rho_comp(ci, fi) = abs(corr(comp_data(mask, fi), s, 'Type', 'Spearman'));
    end
end

valid = n_cus >= MIN_CUS;
n_valid = sum(valid);
best_eis_rho = max(rho_comp(valid, 2:6), [], 2);

%% ---- 4. COMPUTE LAST/FIRST RATIOS FOR ALL CELLS -------------------------
r1_ratio_all = nan(n_cells, 1);
t1_ratio_all = nan(n_cells, 1);
c1_ratio_all = nan(n_cells, 1);

for ci = 1:n_cells
    if n_cus(ci) < MIN_CUS, continue; end
    mask = (cidx == ci);
    [~, si] = sort(diag_num(mask));
    
    t1_vec = tau1(mask); t1_vec = t1_vec(si);
    r1_vec = R1(mask);   r1_vec = r1_vec(si);
    
    if r1_vec(1) <= 0 || t1_vec(1) <= 0, continue; end
    
    c1_first = t1_vec(1) / (r1_vec(1) / 1000);
    c1_last  = t1_vec(end) / (r1_vec(end) / 1000);
    
    r1_ratio_all(ci) = r1_vec(end) / r1_vec(1);
    t1_ratio_all(ci) = t1_vec(end) / t1_vec(1);
    c1_ratio_all(ci) = c1_last / c1_first;
end

valid_mech = ~isnan(r1_ratio_all);
r1_r = r1_ratio_all(valid_mech);
t1_r = t1_ratio_all(valid_mech);
c1_r = c1_ratio_all(valid_mech);
n_mech = sum(valid_mech);

fprintf('  Mechanistic decomposition: %d cells\n', n_mech);
fprintf('  R₁ increases: %.0f%%\n', mean(r1_r > 1)*100);
fprintf('  τ₁ decreases: %.0f%%\n', mean(t1_r < 1)*100);
fprintf('  C₁ decreases: %.0f%%\n', mean(c1_r < 1)*100);

%% ---- 5. FIGURE -----------------------------------------------------------
fig = figure('Units', 'centimeters', 'Position', [2 2 42 13]);
set(fig, 'Color', 'w');

col_pulse = [231 111 81] / 255;    % coral
col_eis   = [42 157 143] / 255;    % teal
col_hppc  = [233 196 106] / 255;   % gold
col_dark  = [38 70 83] / 255;      % navy
col_cap   = [69 123 157] / 255;    % blue

% ---- Panel (a): Median |ρ| bars ------------------------------------------
ax_a = subplot(1, 3, 1);
hold on;

med_rho = zeros(1 + n_comp, 1);
med_rho(1) = median(rho_tau1(valid));
for fi = 1:n_comp
    med_rho(fi+1) = median(rho_comp(valid, fi));
end

all_labels = [{'τ₁'}, comp_labels];
n_bars = numel(all_labels);
y_pos = (n_bars:-1:1)';
bar_colors = [col_pulse; col_hppc; repmat(col_eis, 5, 1)];

for fi = 1:n_bars
    barh(y_pos(fi), med_rho(fi), 0.6, ...
         'FaceColor', bar_colors(fi,:), 'EdgeColor', 'w', 'LineWidth', 1);
    text(med_rho(fi) + 0.008, y_pos(fi), sprintf('%.3f', med_rho(fi)), ...
         'FontSize', 10, 'FontWeight', 'bold', 'VerticalAlignment', 'middle');
end

%set(ax_a, 'YTick', y_pos, 'YTickLabel', all_labels, 'FontSize', 10);
xlabel('Median within-cell |ρ_s| with SOH', 'FontSize', 11);
title({'SOH tracking fidelity', sprintf('(%d cells, ≥%d checkups)', n_valid, MIN_CUS)}, ...
      'FontSize', 12, 'FontWeight', 'bold');
xlim([0, 1.12]);
ylim([0.4, n_bars + 0.6]);
%grid on; ax_a.GridAlpha = 0.1;
set(ax_a, 'YGrid', 'off');
box off;

patch_pulse = patch(nan, nan, col_pulse, 'EdgeColor', 'none');
patch_hppc  = patch(nan, nan, col_hppc, 'EdgeColor', 'none');
patch_eis   = patch(nan, nan, col_eis, 'EdgeColor', 'none');
legend([patch_pulse, patch_hppc, patch_eis], ...
       {'LGN τ₁ (40 s pulse)', 'HPPC R_{pulse} (seconds)', 'EIS (30 min)'}, ...
       'FontSize', 8, 'Location', 'southwest');

text(-0.16, 1.06, 'a', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold');

% ---- Panel (b): Dot strip — R₁, τ₁, C₁ ratios (log scale) ---------------
ax_b = subplot(1, 3, 2);
hold on;

% Jitter for visibility
jitter = @(n, spread) (rand(n, 1) - 0.5) * spread;

% Row positions
y_r1 = 3;
y_t1 = 2;
y_c1 = 1;

% Plot individual dots
scatter(r1_r, y_r1 + jitter(n_mech, 0.35), 12, col_eis, 'filled', ...
        'MarkerFaceAlpha', 0.4, 'MarkerEdgeColor', 'none');
scatter(t1_r, y_t1 + jitter(n_mech, 0.35), 12, col_pulse, 'filled', ...
        'MarkerFaceAlpha', 0.4, 'MarkerEdgeColor', 'none');
scatter(c1_r, y_c1 + jitter(n_mech, 0.35), 12, col_cap, 'filled', ...
        'MarkerFaceAlpha', 0.4, 'MarkerEdgeColor', 'none');

% Median markers (big diamonds)
plot(median(r1_r), y_r1, 'd', 'Color', col_eis, 'MarkerFaceColor', col_eis, ...
     'MarkerSize', 10, 'LineWidth', 1.5);
plot(median(t1_r), y_t1, 'd', 'Color', col_pulse, 'MarkerFaceColor', col_pulse, ...
     'MarkerSize', 10, 'LineWidth', 1.5);
plot(median(c1_r), y_c1, 'd', 'Color', col_cap, 'MarkerFaceColor', col_cap, ...
     'MarkerSize', 10, 'LineWidth', 1.5);

% Reference line at ratio = 1 (no change)
xline(1, '-', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.5);

% Median annotations
text(median(r1_r)*1.15, y_r1 + 0.3, sprintf('%.1f×', median(r1_r)), ...
     'FontSize', 10, 'FontWeight', 'bold', 'Color', col_eis);
text(median(t1_r)*0.6, y_t1 + 0.3, sprintf('%.2f×', median(t1_r)), ...
     'FontSize', 10, 'FontWeight', 'bold', 'Color', col_pulse);
text(median(c1_r)*0.5, y_c1 + 0.3, sprintf('%.2f×', median(c1_r)), ...
     'FontSize', 10, 'FontWeight', 'bold', 'Color', col_cap);

% Percentage annotations on far side
text(0.97, y_r1, sprintf('%.0f%% ↑', mean(r1_r>1)*100), ...
     'Units', 'data', 'FontSize', 9, 'Color', col_eis, ...
     'HorizontalAlignment', 'right', ...
     'VerticalAlignment', 'middle');
text(0.016, y_t1, sprintf('%.0f%% ↓', mean(t1_r<1)*100), ...
     'FontSize', 9, 'Color', col_pulse, ...
     'HorizontalAlignment', 'left', ...
     'VerticalAlignment', 'middle');
text(0.016, y_c1, sprintf('%.0f%% ↓', mean(c1_r<1)*100), ...
     'FontSize', 9, 'Color', col_cap, ...
     'HorizontalAlignment', 'left', ...
     'VerticalAlignment', 'middle');

set(ax_b, 'XScale', 'log');
set(ax_b, 'YTick', [1 2 3], ...
    'YTickLabel', {'C₁ = τ₁/R₁', 'τ₁', 'R₁'}, 'FontSize', 10);
xlabel('Last / first checkup ratio', 'FontSize', 11);
title({'Mechanistic decomposition', sprintf('(%d cells, from pulse only)', n_mech)}, ...
      'FontSize', 12, 'FontWeight', 'bold');
xlim([0.005 15]);
ylim([0.4 3.6]);
grid on; ax_b.GridAlpha = 0.1;
set(ax_b, 'YGrid', 'off');
box off;

% Arrow annotations
text(5, 3.45, 'R grows →', 'FontSize', 9, 'FontWeight', 'bold', ...
     'Color', col_eis, 'HorizontalAlignment', 'center');
text(0.08, 0.55, '← C collapses', 'FontSize', 9, 'FontWeight', 'bold', ...
     'Color', col_cap, 'HorizontalAlignment', 'center');

text(-0.14, 1.06, 'b', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold');

% ---- Panel (c): Overlaid histograms — τ₁ vs best EIS --------------------
ax_c = subplot(1, 3, 3);
hold on;

edges = 0:0.025:1.025;
histogram(best_eis_rho, edges, 'FaceColor', col_eis, 'EdgeColor', 'w', ...
          'FaceAlpha', 0.6, 'LineWidth', 0.8);
histogram(rho_tau1(valid), edges, 'FaceColor', col_pulse, 'EdgeColor', 'w', ...
          'FaceAlpha', 0.7, 'LineWidth', 0.8);

xline(median(rho_tau1(valid)), '--', 'Color', col_pulse, 'LineWidth', 2);
xline(median(best_eis_rho), '--', 'Color', col_eis, 'LineWidth', 2);

xlabel('Within-cell |ρ_s| with SOH', 'FontSize', 11);
ylabel('Number of cells', 'FontSize', 11);
title({'|ρ| distribution', sprintf('(%d cells)', n_valid)}, ...
      'FontSize', 12, 'FontWeight', 'bold');

legend({sprintf('Best EIS (med %.3f)', median(best_eis_rho)), ...
        sprintf('τ₁ pulse (med %.3f)', median(rho_tau1(valid)))}, ...
       'FontSize', 9, 'Location', 'northwest');

xlim([0 1.05]);
%grid on; ax_c.GridAlpha = 0.1;
set(ax_c, 'XGrid', 'off');
box off;
text(-0.14, 1.06, 'c', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold');

%% ---- 6. SAVE -------------------------------------------------------------
exportgraphics(fig, 'fig_soh_vs_eis.png', 'Resolution', 300);
exportgraphics(fig, 'fig_soh_vs_eis.pdf', 'ContentType', 'vector');
fprintf('Saved: fig_soh_vs_eis.png + .pdf\n');