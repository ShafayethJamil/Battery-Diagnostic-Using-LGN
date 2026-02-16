%% fig_nyquist_reconstruction.m
%  ========================================================================
%  Impedance Spectrum Reconstruction from a Single 40 s Pulse
%  228-cell cross-cell LOOCV  |  7 ECM parameters  |  zero EIS required
%
%  Panel (a): Per-frequency MAPE bar chart
%  Panel (b): Parity plot — predicted vs measured Re(Z)
%  Panel (c): Per-cell MAPE histogram
%
%  Requires: results_kit_gpu{0,1,2,3}_aug.json in current directory
%  Author: Shafayeth Jamil (USC ECE), February 2026
%  ========================================================================
clear; clc; close all;

%% ---- 1. LOAD DATA -------------------------------------------------------
fprintf('Loading augmented JSON files...\n');
records = {};
for gi = 0:3
    fname = sprintf('results_kit_gpu%d_aug.json', gi);
    if ~isfile(fname)
        warning('File not found: %s', fname); continue;
    end
    raw = jsondecode(fileread(fname));
    if iscell(raw)
        records = [records; raw(:)]; %#ok<AGROW>
    else
        for k = 1:numel(raw)
            records{end+1,1} = raw(k); %#ok<AGROW>
        end
    end
end
fprintf('  Total records: %d\n', numel(records));

%% ---- 2. FILTER & BUILD MATRICES -----------------------------------------
fprintf('Filtering and building matrices...\n');

freq_labels = {'1 kHz', '100 Hz', '10 Hz', '1 Hz', '0.1 Hz'};

% Pre-allocate
N_raw = numel(records);
cell_ids_raw = strings(N_raw, 1);
X_raw = zeros(N_raw, 7);
Y_raw = zeros(N_raw, 5);
keep = true(N_raw, 1);

for i = 1:N_raw
    try
        r = records{i};
        % Check all required fields exist and are non-empty
        if isempty(r.tau_full) || isempty(r.Rs_jump_mOhm) || ...
           isempty(r.R1_pulse_mOhm) || isempty(r.R2_pulse_mOhm) || ...
           isempty(r.R3_pulse_mOhm) || isempty(r.Z_1kHz_re) || ...
           isempty(r.Z_100Hz_re) || isempty(r.Z_10Hz_re) || ...
           isempty(r.Z_1Hz_re) || isempty(r.Z_01Hz_re)
            keep(i) = false;
            continue;
        end
        cell_ids_raw(i) = string(r.cell);
        X_raw(i,:) = [r.Rs_jump_mOhm, ...
                      r.tau_full(1), r.tau_full(2), r.tau_full(3), ...
                      r.R1_pulse_mOhm, r.R2_pulse_mOhm, r.R3_pulse_mOhm];
        Y_raw(i,:) = [r.Z_1kHz_re, r.Z_100Hz_re, r.Z_10Hz_re, ...
                      r.Z_1Hz_re, r.Z_01Hz_re];
    catch
        keep(i) = false;
    end
end

% Apply filter
X = X_raw(keep, :);
Y = Y_raw(keep, :);
cell_ids = cell_ids_raw(keep);
N = sum(keep);

[unique_cells, ~, cell_idx] = unique(cell_ids);
n_cells = numel(unique_cells);
fprintf('  After filtering: %d records, %d cells\n', N, n_cells);

%% ---- 3. LEAVE-ONE-CELL-OUT CROSS-VALIDATION -----------------------------
fprintf('Running LOOCV across %d cells...\n', n_cells);

alpha = 1e-4;
Y_pred_all = zeros(N, 5);

for ci = 1:n_cells
    if mod(ci, 50) == 0
        fprintf('  Cell %d/%d...\n', ci, n_cells);
    end
    
    test_mask  = (cell_idx == ci);
    train_mask = ~test_mask;
    
    X_tr = X(train_mask, :);
    Y_tr = Y(train_mask, :);
    X_te = X(test_mask, :);
    
    % Ridge: w = (X'X + aI)^-1 X'y  (with intercept via augmented X)
    X_tr_aug = [X_tr, ones(sum(train_mask), 1)];
    X_te_aug = [X_te, ones(sum(test_mask), 1)];
    reg_mat = X_tr_aug' * X_tr_aug + alpha * eye(size(X_tr_aug, 2));
    
    for fi = 1:5
        w = reg_mat \ (X_tr_aug' * Y_tr(:, fi));
        Y_pred_all(test_mask, fi) = X_te_aug * w;
    end
end

fprintf('  Done.\n');

%% ---- 4. COMPUTE METRICS -------------------------------------------------
% Per-frequency MAPE
mape_per_freq = zeros(1, 5);
sem_per_freq  = zeros(1, 5);
for fi = 1:5
    sample_mape = abs(Y_pred_all(:,fi) - Y(:,fi)) ./ (abs(Y(:,fi)) + 1e-15) * 100;
    mape_per_freq(fi) = mean(sample_mape);
    sem_per_freq(fi)  = std(sample_mape) / sqrt(numel(sample_mape));
end
overall_mape = mean(mape_per_freq);

% Per-cell MAPE
cell_mape = zeros(n_cells, 1);
for ci = 1:n_cells
    mask = (cell_idx == ci);
    cell_mape(ci) = mean(abs(Y_pred_all(mask,:) - Y(mask,:)) ...
                        ./ (abs(Y(mask,:)) + 1e-15), 'all') * 100;
end

% R squared
r_val = corrcoef(Y(:), Y_pred_all(:));
R2 = r_val(1,2)^2;

% Print summary
fprintf('\n============================================================\n');
fprintf('  HEADLINE: 7-param ECM  |  %d-cell LOOCV\n', n_cells);
fprintf('============================================================\n');
fprintf('  Overall MAPE: %.2f%%\n', overall_mape);
for fi = 1:5
    fprintf('    %8s: %.2f%%\n', freq_labels{fi}, mape_per_freq(fi));
end
fprintf('  R^2 = %.4f\n', R2);
fprintf('  Median cell MAPE: %.2f%%\n', median(cell_mape));
fprintf('  Worst cell MAPE:  %.2f%%\n', max(cell_mape));
fprintf('  Cells < 2%%: %.0f%%\n', mean(cell_mape < 2) * 100);
fprintf('  Cells < 3%%: %.0f%%\n', mean(cell_mape < 3) * 100);
fprintf('  Cells < 5%%: %.0f%%\n', mean(cell_mape < 5) * 100);
fprintf('============================================================\n\n');

%% ---- 5. FIGURE -----------------------------------------------------------
fig = figure('Units', 'centimeters', 'Position', [2 2 42 13]);
set(fig, 'Color', 'w');

colors_freq = [42 157 143;
               87 184 148;
               139 195 74;
               244 162 97;
               231 111 81] / 255;

dark_blue = [38 70 83] / 255;

% ---- Panel (a): Per-frequency MAPE bars -----------------------------------
ax_a = subplot(1, 3, 1);
hold on;

y_pos = (1:5)';
for fi = 1:5
    barh(y_pos(fi), mape_per_freq(fi), 0.6, ...
         'FaceColor', colors_freq(fi,:), 'EdgeColor', 'w', 'LineWidth', 1.2);
    text(mape_per_freq(fi) + 0.06, y_pos(fi), ...
         sprintf('%.2f%%', mape_per_freq(fi)), ...
         'FontSize', 10, 'FontWeight', 'bold', 'VerticalAlignment', 'middle');
end

xline(2.0, ':', 'Color', [0.8 0.8 0.8], 'LineWidth', 0.8);
set(ax_a, 'YTick', y_pos, 'YTickLabel', freq_labels, 'FontSize', 10.5);
xlabel('MAPE (%)', 'FontSize', 11);
title('Reconstruction error by frequency', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0, max(mape_per_freq) * 1.55]);
ylim([0.4, 5.6]);
%grid on; ax_a.GridAlpha = 0.12;
set(ax_a, 'YGrid', 'off');
box off;
text(-0.14, 1.06, 'a', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold');

% ---- Panel (b): Parity plot ----------------------------------------------
ax_b = subplot(1, 3, 2);
hold on;

for fi = 1:5
    scatter(Y(:,fi)*1000, Y_pred_all(:,fi)*1000, 8, ...
            colors_freq(fi,:), 'filled', 'MarkerFaceAlpha', 0.2);
end

lims = [min(Y(:))*1000*0.92, max(Y(:))*1000*1.05];
plot(lims, lims, 'k-', 'LineWidth', 1.2, 'Color', [0 0 0 0.4]);
xlim(lims); ylim(lims);
axis square;

xlabel('Measured Re(Z) (m\Omega)', 'FontSize', 11);
ylabel('Predicted Re(Z) (m\Omega)', 'FontSize', 11);
title('Predicted vs measured impedance', 'FontSize', 12, 'FontWeight', 'bold');
legend(freq_labels, 'Location', 'northwest', 'FontSize', 8);
%grid on; ax_b.GridAlpha = 0.1;
box off;

text(0.97, 0.06, sprintf('R^2 = %.4f', R2), 'Units', 'normalized', ...
     'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'right', ...
     'BackgroundColor', 'w', 'EdgeColor', [0.8 0.8 0.8], 'Margin', 4);
text(-0.14, 1.06, 'b', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold');

% ---- Panel (c): Per-cell MAPE histogram -----------------------------------
ax_c = subplot(1, 3, 3);
hold on;

histogram(cell_mape, 30, 'FaceColor', dark_blue, 'EdgeColor', 'w', ...
          'LineWidth', 0.8, 'FaceAlpha', 0.85);

med_val = median(cell_mape);
p90_val = prctile(cell_mape, 90);
xline(med_val, '--', 'Color', [231 111 81]/255, 'LineWidth', 2.2);
xline(p90_val, ':',  'Color', [233 196 106]/255, 'LineWidth', 1.8);

xlabel('Mean MAPE across spectrum (%)', 'FontSize', 11);
ylabel('Number of cells', 'FontSize', 11);
title(sprintf('Distribution across %d cells', n_cells), ...
      'FontSize', 12, 'FontWeight', 'bold');
legend({sprintf('Median: %.2f%%', med_val), ...
        sprintf('90th pctl: %.2f%%', p90_val)}, ...
       'FontSize', 9.5, 'Location', 'northeast');
%grid on; ax_c.GridAlpha = 0.1;
set(ax_c, 'XGrid', 'off');
box off;

pct2 = mean(cell_mape < 2) * 100;
pct3 = mean(cell_mape < 3) * 100;
pct5 = mean(cell_mape < 5) * 100;
stats_str = sprintf('%.0f%% < 2%%\n%.0f%% < 3%%\n%.0f%% < 5%%', pct2, pct3, pct5);
text(0.97, 0.95, stats_str, 'Units', 'normalized', ...
     'FontSize', 10.5, 'FontWeight', 'bold', ...
     'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', ...
     'BackgroundColor', 'w', 'EdgeColor', [0.8 0.8 0.8], 'Margin', 4);
text(-0.14, 1.06, 'c', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold');

%% ---- 6. SAVE -------------------------------------------------------------
exportgraphics(fig, 'fig_nyquist_headline.png', 'Resolution', 300);
exportgraphics(fig, 'fig_nyquist_headline.pdf', 'ContentType', 'vector');
fprintf('Saved: fig_nyquist_headline.png + .pdf\n');