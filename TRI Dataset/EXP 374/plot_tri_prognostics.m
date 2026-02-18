%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Figure 2: Fleet-Scale Prognostics
%%% 374 Cells, 207 Protocols, 40-Second Pulses
%%%
%%% Panel layout:
%%%   (a) Fan plot - tau3 slope stratifies lifetime
%%%   (b) Spearman boxplots - tau tracks SOH per cell
%%%   (c) SOH distribution - dataset scale
%%%   (d) Pairwise accuracy - systematic proof
%%%   (e) Pareto: measurement time vs MAE (540x speedup)
%%%   (f) Parity plot K=7 - deliverable
%%%
%%% Requires: results_3d_c0-88.json, results_3d_c88-176.json,
%%%           results_3d_c176-264.json, results_3d_c264-400.json
%%%
%%% Author: Shafayeth Jamil (USC ECE), February 2026
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc;

%% ===== Load all 4 GPU batches =====
files = {'results_3d_c0-88.json', 'results_3d_c88-176.json', ...
         'results_3d_c176-264.json', 'results_3d_c264-400.json'};

all_results = [];
for fi = 1:length(files)
    raw = jsondecode(fileread(files{fi}));
    all_results = [all_results; raw];
end
fprintf('Loaded %d fits\n', length(all_results));

%% ===== Parse fields =====
N = length(all_results);
cell_ids  = {all_results.cell}';
diag_nums = [all_results.diag_num]';
soh       = [all_results.soh]';
capacity  = [all_results.capacity]';
tau1      = [all_results.lgn_tau1]';
tau2      = [all_results.lgn_tau2]';
tau3      = [all_results.lgn_tau3]';
eta0      = [all_results.eta0]';

unique_cells = unique(cell_ids);
n_cells = length(unique_cells);
fprintf('%d unique cells\n', n_cells);

%% ===== Per-cell diagnostics =====
tau2_rho  = nan(n_cells, 1);
tau3_rho  = nan(n_cells, 1);
final_soh = nan(n_cells, 1);
n_diags   = nan(n_cells, 1);

for ci = 1:n_cells
    mask = strcmp(cell_ids, unique_cells{ci});
    s = soh(mask); t2 = tau2(mask); t3 = tau3(mask);
    dn = diag_nums(mask);
    [~, idx] = sort(dn);
    s = s(idx); t2 = t2(idx); t3 = t3(idx);
    
    n_diags(ci) = length(s);
    final_soh(ci) = s(end);
    
    if length(s) >= 5
        tau2_rho(ci) = abs(corr(s, t2, 'Type', 'Spearman'));
        tau3_rho(ci) = abs(corr(s, t3, 'Type', 'Spearman'));
    end
end

valid = ~isnan(tau2_rho);
med2 = median(tau2_rho(valid));
med3 = median(tau3_rho(valid));
fprintf('tau2 median |rho| = %.3f\n', med2);
fprintf('tau3 median |rho| = %.3f\n', med3);

%% ===== K=3 features (for fan + pairwise) =====
K3 = 3;
tau3_slope_K3 = nan(n_cells, 1);
cap_slope_K3  = nan(n_cells, 1);
cap_start     = nan(n_cells, 1);
has_K3        = false(n_cells, 1);

for ci = 1:n_cells
    mask = strcmp(cell_ids, unique_cells{ci});
    s = soh(mask); c = capacity(mask);
    t3 = tau3(mask); dn = diag_nums(mask);
    [~, idx] = sort(dn);
    s = s(idx); c = c(idx); t3 = t3(idx);
    
    if length(s) < 7, continue; end
    has_K3(ci) = true;
    
    dn_k = (0:K3-1)';
    p3 = polyfit(dn_k, t3(1:K3), 1);
    pc = polyfit(dn_k, c(1:K3), 1);
    tau3_slope_K3(ci) = p3(1);
    cap_slope_K3(ci) = pc(1);
    cap_start(ci) = c(1);
end

%% ===== Fan plot data: cap-matched terciles =====
cap_med = median(cap_start(has_K3));
matched = has_K3 & abs(cap_start - cap_med) < 0.15;
t3_slopes_matched = tau3_slope_K3(matched);
t33 = prctile(t3_slopes_matched, 33);
t67 = prctile(t3_slopes_matched, 67);

top_idx = find(matched & tau3_slope_K3 > t67);
bot_idx = find(matched & tau3_slope_K3 <= t33);
fprintf('Fan plot: %d top third, %d bottom third\n', length(top_idx), length(bot_idx));


%% ===== Pairwise accuracy =====
gaps = [10 15 20 25 30 35 40];
n_gaps = length(gaps);
pw_n   = zeros(n_gaps, 1);
pw_t3  = zeros(n_gaps, 1);
pw_cap = zeros(n_gaps, 1);

k3_idx = find(has_K3);
nk3 = length(k3_idx);

for ii = 1:nk3
    for jj = ii+1:nk3
        ci = k3_idx(ii); cj = k3_idx(jj);
        if abs(cap_start(ci) - cap_start(cj)) > 0.1, continue; end
        if abs(cap_slope_K3(ci) - cap_slope_K3(cj)) > 0.02, continue; end
        
        soh_gap = abs(final_soh(ci) - final_soh(cj));
        if soh_gap < 10, continue; end
        
        % Who is worse?
        if final_soh(ci) < final_soh(cj)
            worse_is_i = true;
        else
            worse_is_i = false;
        end
        
        t3_says_i = tau3_slope_K3(ci) < tau3_slope_K3(cj);
        cap_says_i = cap_slope_K3(ci) < cap_slope_K3(cj);
        
        for gi = 1:n_gaps
            if soh_gap >= gaps(gi)
                pw_n(gi) = pw_n(gi) + 1;
                if t3_says_i == worse_is_i
                    pw_t3(gi) = pw_t3(gi) + 1;
                end
                if cap_says_i == worse_is_i
                    pw_cap(gi) = pw_cap(gi) + 1;
                end
            end
        end
    end
end

fprintf('Pairwise at ΔSOH>30%%: %d pairs, τ₃=%.1f%%, cap=%.1f%%\n', ...
    pw_n(gaps==30), 100*pw_t3(gaps==30)/pw_n(gaps==30), 100*pw_cap(gaps==30)/pw_n(gaps==30));

%% ===== LOOCV Prognostics =====
Ks = [3 5 7 10];
mae_tau = nan(length(Ks), 1);
mae_cap = nan(length(Ks), 1);
mae_eta = nan(length(Ks), 1);
rho_tau = nan(length(Ks), 1);
n_cells_K = nan(length(Ks), 1);

pred_tau_K7 = []; pred_cap_K7 = []; true_K7 = [];

for ki = 1:length(Ks)
    K = Ks(ki);
    X_tau = []; X_cap = []; X_eta = []; y = [];
    
    for ci = 1:n_cells
        mask = strcmp(cell_ids, unique_cells{ci});
        s = soh(mask); c = capacity(mask);
        t1 = tau1(mask); t2 = tau2(mask); t3 = tau3(mask);
        e0 = eta0(mask); dn = diag_nums(mask);
        [~, idx] = sort(dn);
        s = s(idx); c = c(idx);
        t1 = t1(idx); t2 = t2(idx); t3 = t3(idx); e0 = e0(idx);
        
        if length(s) < max(K, 5), continue; end
        
        dn_k = (0:K-1)';
        p1 = polyfit(dn_k, t1(1:K), 1);
        p2 = polyfit(dn_k, t2(1:K), 1);
        p3 = polyfit(dn_k, t3(1:K), 1);
        pc = polyfit(dn_k, c(1:K), 1);
        pe = polyfit(dn_k, e0(1:K), 1);
        
        X_tau = [X_tau; p1(1) p2(1) p3(1) t1(K) t2(K) t3(K)];
        X_cap = [X_cap; pc(1) c(K)];
        X_eta = [X_eta; e0(K) pe(1)];
        y = [y; s(end)];
    end
    
    n_cells_K(ki) = length(y);
    [mae_tau(ki), rho_tau(ki), pt] = ridge_loocv(X_tau, y);
    [mae_cap(ki), ~, pc_pred]      = ridge_loocv(X_cap, y);
    [mae_eta(ki), ~, ~]             = ridge_loocv(X_eta, y);
    
    if K == 7
        pred_tau_K7 = pt;
        pred_cap_K7 = pc_pred;
        true_K7 = y;
    end
    
    fprintf('K=%d (%d cells): tau=%.2f%%  cap=%.2f%%  eta=%.2f%%\n', ...
        K, length(y), mae_tau(ki), mae_cap(ki), mae_eta(ki));
end

%% ===== COLORS =====
C_tau3 = [0.204 0.596 0.859];
C_tau2 = [0.180 0.800 0.443];
C_cap  = [0.498 0.549 0.553];
C_eta  = [0.608 0.349 0.714];
C_ok   = [0.153 0.682 0.376];
C_bad  = [0.753 0.224 0.169];

%% ===== FIGURE =====
fig = figure('Units', 'inches', 'Position', [0.5 0.5 18 11], 'Color', 'w');

% ================================================================
% (a) FAN PLOT
% ================================================================
ax1 = subplot(2, 3, 1); hold on;

% Bottom tercile spaghetti (red)
for ii = 1:length(bot_idx)
    ci = bot_idx(ii);
    mask = strcmp(cell_ids, unique_cells{ci});
    dn_c = diag_nums(mask); s_c = soh(mask);
    [~, idx] = sort(dn_c);
    plot(dn_c(idx), s_c(idx), '-', 'Color', [C_bad 0.12], 'LineWidth', 0.7, ...
        'HandleVisibility', 'off');
end

% Top tercile spaghetti (green)
for ii = 1:length(top_idx)
    ci = top_idx(ii);
    mask = strcmp(cell_ids, unique_cells{ci});
    dn_c = diag_nums(mask); s_c = soh(mask);
    [~, idx] = sort(dn_c);
    plot(dn_c(idx), s_c(idx), '-', 'Color', [C_ok 0.12], 'LineWidth', 0.7, ...
        'HandleVisibility', 'off');
end

% Median trajectories (bold) for each tercile
% Collect all (diag_num, soh) per group, bin by diag_num, take median
max_dn = 0;
for ii = [bot_idx(:); top_idx(:)]'
    mask = strcmp(cell_ids, unique_cells{ii});
    max_dn = max(max_dn, max(diag_nums(mask)));
end
dn_axis = 0:max_dn;

% Bottom tercile median
soh_bot_mat = nan(length(bot_idx), length(dn_axis));
for ii = 1:length(bot_idx)
    ci = bot_idx(ii);
    mask = strcmp(cell_ids, unique_cells{ci});
    dn_c = diag_nums(mask); s_c = soh(mask);
    [~, idx] = sort(dn_c); dn_c = dn_c(idx); s_c = s_c(idx);
    for jj = 1:length(dn_c)
        col = find(dn_axis == dn_c(jj), 1);
        if ~isempty(col), soh_bot_mat(ii, col) = s_c(jj); end
    end
end
mu_bot = median(soh_bot_mat, 1, 'omitnan');
n_bot  = sum(~isnan(soh_bot_mat), 1);
mu_bot(n_bot < length(bot_idx)*0.5) = NaN;  % truncate at <50% survival

% Top tercile median
soh_top_mat = nan(length(top_idx), length(dn_axis));
for ii = 1:length(top_idx)
    ci = top_idx(ii);
    mask = strcmp(cell_ids, unique_cells{ci});
    dn_c = diag_nums(mask); s_c = soh(mask);
    [~, idx] = sort(dn_c); dn_c = dn_c(idx); s_c = s_c(idx);
    for jj = 1:length(dn_c)
        col = find(dn_axis == dn_c(jj), 1);
        if ~isempty(col), soh_top_mat(ii, col) = s_c(jj); end
    end
end
mu_top = median(soh_top_mat, 1, 'omitnan');
n_top  = sum(~isnan(soh_top_mat), 1);
mu_top(n_top < length(top_idx)*0.5) = NaN;  % truncate at <50% survival

% Plot bold median lines
plot(dn_axis, mu_bot, '-', 'Color', C_bad, 'LineWidth', 3.5);
plot(dn_axis, mu_top, '-', 'Color', C_ok,  'LineWidth', 3.5);

% K=3 window
patch([-.3 2.3 2.3 -.3], [20 20 105 105], 'b', ...
    'FaceAlpha', 0.08, 'EdgeColor', 'none');
yline(80, '--', 'Color', [0.5 0.5 0.5 0.4]);
xlabel('Diagnostic \#', 'FontSize', 11);
ylabel('SOH (%)', 'FontSize', 11);
title('(a) \tau_3 slope at K=3 stratifies lifetime', 'FontSize', 12, 'FontWeight', 'bold');
ylim([20 105]);
legend({sprintf('Low \\tau_3 slope (n=%d)', length(bot_idx)), ...
        sprintf('High \\tau_3 slope (n=%d)', length(top_idx))}, ...
    'FontSize', 8, 'Location', 'southwest');
grid on; set(gca, 'GridAlpha', 0.15); box on;

% ================================================================
% (b) SPEARMAN BOXPLOTS
% ================================================================
ax2 = subplot(2, 3, 2);
grp = [ones(sum(valid),1); 2*ones(sum(valid),1)];
bp = boxplot([tau2_rho(valid); tau3_rho(valid)], grp, ...
    'Widths', 0.5, 'Symbol', '');
h = findobj(gca, 'Tag', 'Box');
box_colors = [C_tau3; C_tau2];  % reversed (MATLAB quirk)
for j = 1:length(h)
    patch(get(h(j),'XData'), get(h(j),'YData'), box_colors(j,:), 'FaceAlpha', 0.7);
end
hold on;
text(1, med2+0.02, sprintf('%.3f', med2), ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
text(2, med3+0.02, sprintf('%.3f', med3), ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
yline(0.9, '--', 'Color', [0.5 0.5 0.5 0.4]);
set(gca, 'XTickLabel', {'\tau_2 (SEI)', '\tau_3 (diff)'}, 'FontSize', 11);
ylabel('Per-cell |\rho(\tau, SOH)|', 'FontSize', 11);
title(sprintf('(b) Diagnostic: \\tau tracks SOH (%d cells)', sum(valid)), ...
    'FontSize', 12, 'FontWeight', 'bold');
ylim([0.4 1.05]); box on;
grid on; set(gca, 'GridAlpha', 0.15);

% ================================================================
% (c) SOH DISTRIBUTION
% ================================================================
ax3 = subplot(2, 3, 3);
fs_valid = final_soh(valid);
histogram(fs_valid, 25, 'FaceColor', C_tau3, 'FaceAlpha', 0.7, 'EdgeColor', 'w');
hold on;
xline(80, '--r', 'LineWidth', 1.5);
xline(median(fs_valid), '--k', 'LineWidth', 1.5);
xlabel('Final SOH (%)', 'FontSize', 11);
ylabel('Cell count', 'FontSize', 11);
n_below80 = sum(fs_valid < 80);
title(sprintf('(c) Dataset: %d cells, 207 protocols', sum(valid)), ...
    'FontSize', 12, 'FontWeight', 'bold');
legend({'', '80% EOL', sprintf('Median=%.0f%%', median(fs_valid))}, ...
    'FontSize', 9, 'Location', 'northwest');
text(0.05, 0.9, sprintf('%d cells below 80%%', n_below80), ...
    'Units', 'normalized', 'FontSize', 10, 'FontWeight', 'bold');
box on; grid on; set(gca, 'GridAlpha', 0.15);

% ================================================================
% (d) PAIRWISE ACCURACY
% ================================================================
ax4 = subplot(2, 3, 4);
x = 1:n_gaps;
w = 0.3;
t3_acc = 100 * pw_t3 ./ pw_n;
cap_acc = 100 * pw_cap ./ pw_n;

bar(x - w/2, t3_acc, w, 'FaceColor', C_tau3, 'FaceAlpha', 0.85, 'EdgeColor', 'w'); hold on;
bar(x + w/2, cap_acc, w, 'FaceColor', C_cap, 'FaceAlpha', 0.85, 'EdgeColor', 'w');
yline(50, '--k', 'LineWidth', 1, 'Alpha', 0.5);

set(gca, 'XTick', x);
set(gca, 'XTickLabel', arrayfun(@(g) sprintf('>%d%%', g), gaps, 'UniformOutput', false));
xlabel('Final SOH gap threshold', 'FontSize', 11);
ylabel('Pairwise accuracy (%)', 'FontSize', 11);
title('(d) "Who dies first?" \textemdash{} cap-matched pairs, K=3', ...
    'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'latex');
legend({'\tau_3 slope', 'Cap slope', 'Random'}, 'FontSize', 9, 'Location', 'northwest');
ylim([40 90]);

for i = 1:n_gaps
    text(i - w/2, t3_acc(i)+1.2, sprintf('%.0f%%', t3_acc(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 8, ...
        'FontWeight', 'bold', 'Color', C_tau3);
    text(i, 42, sprintf('n=%d', pw_n(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 7, 'Color', [0 0 0 0.5]);
end
grid on; set(gca, 'GridAlpha', 0.15); box on;

% ================================================================
% (e) PARETO: MEASUREMENT TIME vs MAE
% ================================================================
% Times calibrated from van Vlijmen et al., Energy Environ. Sci., 2025,
% 18, 6641-6654. DOI: 10.1039/D4EE05609D
% Diagnostic cycle: reset + HPPC (INL standard) + RPT at 0.2C, 1C, 2C
% HPPC per INL: 10s discharge, 40s rest, 10s charge per SOC level
%
% LGN: K x 40s rest windows only
% Capacity: requires 0.2C RPT per diagnostic
%   0.2C on 4.84 Ah cell = 5h discharge + CC-CV charge + settling ~ 6h
% Severson 2019: 100 fast-charge cycles ~ 25 hours

ax5 = subplot(2, 3, 5);

% --- Measurement times ---
time_lgn = Ks(:) * 40 / 60;              % minutes
time_cap = Ks(:) * 360;                  % 6 hr per diagnostic, in minutes
time_severson = 1500;                     % ~25 hours in minutes

% --- Severson benchmark (Nature Energy 2019, 9.1% MAE on LFP cells) ---
C_sev = [0.753 0.224 0.169];

% LGN tau curve
loglog(time_lgn, mae_tau, 'o-', 'Color', C_tau3, 'LineWidth', 2.5, ...
    'MarkerSize', 11, 'MarkerFaceColor', C_tau3, ...
    'MarkerEdgeColor', 'w'); hold on;

% Capacity curve
loglog(time_cap, mae_cap, 's--', 'Color', C_cap, 'LineWidth', 2, ...
    'MarkerSize', 9, 'MarkerFaceColor', C_cap, ...
    'MarkerEdgeColor', 'w');

% Severson point
loglog(time_severson, 9.1, 'p', 'Color', C_sev, 'MarkerSize', 16, ...
    'MarkerFaceColor', C_sev, 'MarkerEdgeColor', 'w', 'LineWidth', 1.5);

% --- Annotations: LGN K labels ---
for i = 1:length(Ks)
    text(time_lgn(i)*1.25, mae_tau(i), sprintf('K=%d', Ks(i)), ...
        'FontSize', 8, 'FontWeight', 'bold', 'Color', C_tau3);
end

% --- Annotations: Capacity K labels (first & last only) ---
text(time_cap(1)*1.15, mae_cap(1)+0.3, sprintf('K=%d', Ks(1)), ...
    'FontSize', 8, 'Color', C_cap);
text(time_cap(end)*1.15, mae_cap(end)-0.3, sprintf('K=%d', Ks(end)), ...
    'FontSize', 8, 'Color', C_cap);

% --- Annotations: Severson ---
text(time_severson*0.25, 9.1+0.6, 'Severson 2019', ...
    'FontSize', 8, 'FontWeight', 'bold', 'Color', C_sev);

% --- Speedup arrow: LGN K=7 vs Capacity K=7 ---
ki7 = find(Ks == 7);
speedup = time_cap(ki7) / time_lgn(ki7);
mid_x = sqrt(time_lgn(ki7) * time_cap(ki7));
mid_y = (mae_tau(ki7) + mae_cap(ki7)) / 2;
text(mid_x, mid_y - 0.7, sprintf('%d\\times faster', round(speedup)), ...
    'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.153 0.682 0.376], ...
    'HorizontalAlignment', 'center', ...
    'BackgroundColor', 'w', 'EdgeColor', [0.153 0.682 0.376], ...
    'Margin', 3);

% --- Axis formatting ---
set(gca, 'XScale', 'log', 'YScale', 'linear');
xlabel('Total measurement time (minutes)', 'FontSize', 11);
ylabel('LOOCV MAE (% SOH)', 'FontSize', 11);
title('(e) Accuracy vs. measurement burden', ...
    'FontSize', 12, 'FontWeight', 'bold');
legend({'LGN \tau (40s pulse)', 'Capacity (0.2C RPT)', 'Severson 2019'}, ...
    'FontSize', 9, 'Location', 'northeast');
xlim([0.8 80000]); ylim([5 15]);
grid on; set(gca, 'GridAlpha', 0.15); box on;

% Human-readable top ticks
xt = [1 5 60 360 1440 15000];
xtl = {'1 min', '5 min', '1 hr', '6 hr', '24 hr', ''};
ax5_top = axes('Position', ax5.Position, 'XAxisLocation', 'top', ...
    'YTick', [], 'XScale', 'log', 'XLim', [0.8 80000], ...
    'XTick', xt, 'XTickLabel', xtl, 'FontSize', 8, ...
    'Color', 'none', 'Box', 'off');

% ================================================================
% (f) PARITY PLOT K=7
% ================================================================
ax6 = subplot(2, 3, 6);
ki7 = find(Ks == 7);

% Capacity (gray, behind)
scatter(true_K7, pred_cap_K7, 15, C_cap, 'filled', 'MarkerFaceAlpha', 0.3); hold on;
% Tau (blue, on top)
scatter(true_K7, pred_tau_K7, 15, C_tau3, 'filled', 'MarkerFaceAlpha', 0.4);
% Parity + band
plot([20 100], [20 100], 'k-', 'LineWidth', 1.5);
fill([20 100 100 20], [15 95 105 25], [0.5 0.5 0.5], ...
    'FaceAlpha', 0.06, 'EdgeColor', 'none');

% Compute Spearman rho for legend
rho_tau_K7 = corr(true_K7, pred_tau_K7, 'Type', 'Spearman');
rho_cap_K7 = corr(true_K7, pred_cap_K7, 'Type', 'Spearman');

xlabel('True final SOH (%)', 'FontSize', 11);
ylabel('Predicted final SOH (%)', 'FontSize', 11);
title(sprintf('(f) LOOCV K=7 (%d cells)', n_cells_K(ki7)), ...
    'FontSize', 12, 'FontWeight', 'bold');
legend({sprintf('Capacity (MAE=%.1f%%, \\rho=%.2f)', mae_cap(ki7), rho_cap_K7), ...
        sprintf('\\tau only (MAE=%.1f%%, \\rho=%.2f)', mae_tau(ki7), rho_tau_K7)}, ...
    'FontSize', 9, 'Location', 'northwest');
xlim([20 100]); ylim([20 100]);
axis square; grid on; set(gca, 'GridAlpha', 0.15); box on;

%% ===== Save =====
sgtitle('Figure 2: Fleet-Scale Prognostics — 374 Cells, 207 Protocols, 40-Second Pulses', ...
    'FontSize', 14, 'FontWeight', 'bold');
set(fig, 'PaperPositionMode', 'auto');
exportgraphics(fig, 'fig2_fleet_prognostics.png', 'Resolution', 300);
fprintf('\nSaved fig2_fleet_prognostics.png\n');

%% ===== Ridge LOOCV helper =====
function [mae, rho, preds] = ridge_loocv(X, y)
    N = length(y);
    preds = zeros(N, 1);
    for i = 1:N
        X_tr = X; X_tr(i,:) = [];
        y_tr = y; y_tr(i) = [];
        X_te = X(i,:);
        
        mu = mean(X_tr); sd = std(X_tr);
        sd(sd == 0) = 1;
        X_tr_s = (X_tr - mu) ./ sd;
        X_te_s = (X_te - mu) ./ sd;
        
        b = ridge(y_tr, X_tr_s, 1, 0);
        preds(i) = b(1) + X_te_s * b(2:end);
    end
    mae = mean(abs(preds - y));
    rho = corr(preds, y, 'Type', 'Spearman');
end