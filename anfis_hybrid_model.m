%% =========================================================
%  ANFIS & HYBRID (HEC-HMS + ANFIS Residual) MODEL
%  Multi-Event Rainfall-Runoff Modeling
%  4 Events Train | 1 Event Test (Event-Based)
%
%  Column structure (each CSV):
%    Col 1: Time_min  — elapsed time in minutes (5-min resolution,
%                       HEC-HMS output interpolated to 5-min grid)
%    Col 2: Rainfall  — rainfall depth (mm per 5-min interval)
%    Col 3: Qobs      — observed discharge (m3/s)
%    Col 4: Qhms      — HEC-HMS simulated discharge (m3/s)
%                       (originally 20-min, linearly interpolated to 5-min)
%
%  Events:
%    1 — 30 Jun 1988  (61  steps)  — paper calibration event
%    2 — 02 May 1990  (108 steps)
%    3 — 03 Jun 1995  (72  steps)
%    4 — 23 Mar 1998  (48  steps)
%    5 — 01 Jun 2005  (174 steps)  — paper validation event (default test)
%
%  Usage:
%    1. Place all CSV files in the MATLAB working directory
%    2. Set TEST_EVENT_INDEX below
%    3. Run the script
%    4. Results are displayed as figures and a summary table
% =========================================================

clear; clc; close all;

%% =========================================================
%  USER SETTINGS - EDIT THIS SECTION
% =========================================================

% Provide file paths for all 5 events (CSV)
% Column order MUST be: Time_min | Rainfall | Qobs | Qhms
FILE_LIST = {
    'event_1988_30Jun.csv',   % Event 1: 30 Jun 1988 (61 steps,  5-min)
    'event_1990_02May.csv',   % Event 2: 02 May 1990 (108 steps, 5-min)
    'event_1995_03Jun.csv',   % Event 3: 03 Jun 1995 (72 steps,  5-min)
    'event_1998_23Mar.csv',   % Event 4: 23 Mar 1998 (48 steps,  5-min)
    'event_2005_01Jun.csv'    % Event 5: 01 Jun 2005 (174 steps, 5-min) ← test
};

% Index of the test event (1 to 5)
% Default: Event 5 (2005) — consistent with the reference paper validation
TEST_EVENT_INDEX = 5;

% File type: 'csv' or 'excel'
FILE_TYPE = 'csv';

% ANFIS settings (matching Temelli & Tombul methodology)
LAG           = 2;      % Lag order: inputs P(t),Q(t),P(t-1),Q(t-1)
N_EPOCHS      = 50;     % Number of training epochs
CLUSTER_RAD   = 0.8;    % Subtractive clustering influence radius
VAL_RATIO     = 0.20;   % Internal validation ratio (last 20% of training data)

% Figure saving options
SAVE_FIGURES  = false;
FIG_FORMAT    = 'png';  % 'png' or 'pdf'
OUTPUT_DIR    = 'results';

%% =========================================================
%  HELPER FUNCTIONS
% =========================================================

% --- Performance metrics ---
calc_metrics = @(obs, sim) struct(...
    'NSE',   1 - sum((obs-sim).^2) / sum((obs-mean(obs)).^2), ...
    'RMSE',  sqrt(mean((obs-sim).^2)), ...
    'MAE',   mean(abs(obs-sim)), ...
    'KGE',   1 - sqrt((corr(obs,sim)-1)^2 + (std(sim)/std(obs)-1)^2 + (mean(sim)/mean(obs)-1)^2), ...
    'PBIAS', 100 * sum(obs-sim) / sum(obs) ...
);

% --- Min-max normalization ---
normalize   = @(x, mn, mx) (x - mn) / (mx - mn + 1e-12);
denormalize = @(xn, mn, mx) xn * (mx - mn + 1e-12) + mn;

%% =========================================================
%  1. DATA LOADING
% =========================================================

fprintf('=== Loading event data ===\n');
events = cell(1, 5);

for i = 1:5
    if strcmpi(FILE_TYPE, 'csv')
        raw = readmatrix(FILE_LIST{i});
    else
        raw = readmatrix(FILE_LIST{i});  % Works for both CSV and Excel
    end

    % Assign columns
    events{i}.time   = raw(:, 1);
    events{i}.P      = raw(:, 2);
    events{i}.Qobs   = raw(:, 3);
    events{i}.Qhms   = raw(:, 4);
    events{i}.name   = sprintf('Event %d', i);

    fprintf('  %s: %d time steps loaded\n', events{i}.name, length(events{i}.time));
end

%% =========================================================
%  2. TRAIN / TEST SPLIT
% =========================================================

test_idx  = TEST_EVENT_INDEX;
train_idx = setdiff(1:5, test_idx);

fprintf('\n=== Training Events: %s | Test Event: %d ===\n', ...
    num2str(train_idx), test_idx);

%% =========================================================
%  3. NORMALIZATION
%     Each event is normalized using its own min-max range
%     (consistent with the reference paper methodology)
% =========================================================

for i = 1:5
    mn_P = min(events{i}.P);    mx_P = max(events{i}.P);
    mn_Q = min(events{i}.Qobs); mx_Q = max(events{i}.Qobs);

    events{i}.P_n    = normalize(events{i}.P,    mn_P, mx_P);
    events{i}.Qobs_n = normalize(events{i}.Qobs, mn_Q, mx_Q);
    events{i}.Qhms_n = normalize(events{i}.Qhms, mn_Q, mx_Q);

    % Store scaling parameters for denormalization
    events{i}.mn_Q = mn_Q;
    events{i}.mx_Q = mx_Q;
end

%% =========================================================
%  4. LAG FEATURE CONSTRUCTION
%
%  Standalone ANFIS:
%    Inputs:  P(t), Q(t), P(t-1), Q(t-1)   [4 inputs, LAG=2]
%    Output:  Q(t+1)
%
%  Hybrid ANFIS (residual correction):
%    Inputs:  P(t), Qhms(t), Qobs(t), P(t-1), Qhms(t-1), Qobs(t-1)  [6 inputs]
%    Output:  residual(t+1) = Qobs(t+1) - Qhms(t+1)
%    Reference: "input variables include normalized precipitation,
%    HEC-HMS simulated discharge, and lagged discharge values" (Temelli & Tombul)
% =========================================================

function [X, Y] = build_lag_features(P_n, Q_n, lag)
    % For standalone ANFIS: Q_n = normalized observed discharge
    N = length(P_n);
    X = [];
    Y = [];
    for t = lag:(N-1)
        row = [];
        for k = 0:(lag-1)
            row = [row, P_n(t-k), Q_n(t-k)];
        end
        X = [X; row];
        Y = [Y; Q_n(t+1)];
    end
end

function [X, Y] = build_hybrid_features(P_n, Qhms_n, Qobs_n, res_n, lag)
    % For hybrid ANFIS: inputs = P, Qhms, Qobs (all lagged)
    % Target = residual at next step
    N = length(P_n);
    X = [];
    Y = [];
    for t = lag:(N-1)
        row = [];
        for k = 0:(lag-1)
            row = [row, P_n(t-k), Qhms_n(t-k), Qobs_n(t-k)];
        end
        X = [X; row];
        Y = [Y; res_n(t+1)];
    end
end

%% =========================================================
%  5A. STANDALONE ANFIS - TRAINING
% =========================================================

fprintf('\n=== Standalone ANFIS Training ===\n');

% Concatenate all training events
X_train_anfis = [];
Y_train_anfis = [];

for i = train_idx
    [Xi, Yi] = build_lag_features(events{i}.P_n, events{i}.Qobs_n, LAG);
    X_train_anfis = [X_train_anfis; Xi];
    Y_train_anfis = [Y_train_anfis; Yi];
end

% Internal validation split (last 20%)
n_train = size(X_train_anfis, 1);
n_val   = floor(n_train * VAL_RATIO);
n_fit   = n_train - n_val;

X_fit = X_train_anfis(1:n_fit, :);
Y_fit = Y_train_anfis(1:n_fit);
X_val = X_train_anfis(n_fit+1:end, :);
Y_val = Y_train_anfis(n_fit+1:end);

% Generate FIS via subtractive clustering
train_data_anfis = [X_fit, Y_fit];
fis_anfis = genfis2(X_fit, Y_fit, CLUSTER_RAD);

fprintf('  Number of fuzzy rules: %d\n', length(fis_anfis.Rules));

% Train ANFIS
opt = anfisOptions(...
    'InitialFIS',          fis_anfis, ...
    'EpochNumber',         N_EPOCHS, ...
    'ValidationData',      [X_val, Y_val], ...
    'DisplayANFISInformation', 0, ...
    'DisplayErrorValues',  0, ...
    'DisplayStepSize',     0, ...
    'DisplayFinalResults', 0);

[fis_anfis_trained, train_error, ~, fis_anfis_best, val_error] = ...
    anfis([X_fit, Y_fit], opt);

fprintf('  Final epoch - Training RMSE: %.6f\n', train_error(end));
fprintf('  Final epoch - Validation RMSE: %.6f\n', val_error(end));

%% =========================================================
%  5B. HYBRID MODEL - TRAINING (ANFIS residual correction)
% =========================================================

fprintf('\n=== Hybrid Model Training ===\n');

X_train_hyb = [];
Y_train_hyb = [];

for i = train_idx
    % Residual target: observed minus HEC-HMS (normalized)
    res_n = events{i}.Qobs_n - events{i}.Qhms_n;
    % Inputs: P(t), Qhms(t), Qobs(t) at each lag step
    [Xi, Yi] = build_hybrid_features(events{i}.P_n, events{i}.Qhms_n, ...
                                      events{i}.Qobs_n, res_n, LAG);
    X_train_hyb = [X_train_hyb; Xi];
    Y_train_hyb = [Y_train_hyb; Yi];
end

% Internal validation split
n_train_h = size(X_train_hyb, 1);
n_val_h   = floor(n_train_h * VAL_RATIO);
n_fit_h   = n_train_h - n_val_h;

X_fit_h = X_train_hyb(1:n_fit_h, :);
Y_fit_h = Y_train_hyb(1:n_fit_h);
X_val_h = X_train_hyb(n_fit_h+1:end, :);
Y_val_h = Y_train_hyb(n_fit_h+1:end);

fis_hyb = genfis2(X_fit_h, Y_fit_h, CLUSTER_RAD);

fprintf('  Number of fuzzy rules (Hybrid): %d\n', length(fis_hyb.Rules));

opt_h = anfisOptions(...
    'InitialFIS',          fis_hyb, ...
    'EpochNumber',         N_EPOCHS, ...
    'ValidationData',      [X_val_h, Y_val_h], ...
    'DisplayANFISInformation', 0, ...
    'DisplayErrorValues',  0, ...
    'DisplayStepSize',     0, ...
    'DisplayFinalResults', 0);

[fis_hyb_trained, train_error_h, ~, fis_hyb_best, val_error_h] = ...
    anfis([X_fit_h, Y_fit_h], opt_h);

fprintf('  Final epoch - Training RMSE: %.6f\n', train_error_h(end));
fprintf('  Final epoch - Validation RMSE: %.6f\n', val_error_h(end));

%% =========================================================
%  6. PREDICTION ON TEST EVENT
% =========================================================

ev   = events{test_idx};
mn_Q = ev.mn_Q;
mx_Q = ev.mx_Q;

% --- Build test features ---
% Standalone ANFIS: inputs = P(t), Qobs(t) lagged
[X_test_a, Y_test_a] = build_lag_features(ev.P_n, ev.Qobs_n, LAG);

% Hybrid ANFIS: inputs = P(t), Qhms(t), Qobs(t) lagged; target = residual
res_test_n = ev.Qobs_n - ev.Qhms_n;
[X_test_h, Y_test_h] = build_hybrid_features(ev.P_n, ev.Qhms_n, ev.Qobs_n, res_test_n, LAG);

% --- Predictions (normalized space) ---
Qsim_anfis_n = evalfis(fis_anfis_best, X_test_a);
Qres_hyb_n   = evalfis(fis_hyb_best,   X_test_h);

% HEC-HMS aligned to lag offset
Qhms_test_n  = ev.Qhms_n(LAG+1:end);
Qhyb_n       = Qhms_test_n + Qres_hyb_n;

% Observed (aligned)
Qobs_test_n  = Y_test_a;

% Denormalize to m3/s
Qobs_raw = denormalize(Qobs_test_n,  mn_Q, mx_Q);
Qanfis   = denormalize(Qsim_anfis_n, mn_Q, mx_Q);
Qhyb     = denormalize(Qhyb_n,      mn_Q, mx_Q);
Qhms_raw = denormalize(Qhms_test_n,  mn_Q, mx_Q);

time_test = ev.time(LAG+1:end);

%% =========================================================
%  7. ALL EVENTS — FULL PERFORMANCE TABLE (all 3 models)
% =========================================================

fprintf('\n');
fprintf('=================================================================\n');
fprintf('  FULL PERFORMANCE SUMMARY — ALL EVENTS (normalised discharge)\n');
fprintf('=================================================================\n');
fprintf('%-20s %-10s %7s %7s %7s %7s %8s\n', ...
    'Event','Model','NSE','RMSE','MAE','KGE','PBIAS');
fprintf('%s\n', repmat('-',1,75));

all_event_idx = 1:5;

for i = all_event_idx
    ev_i = events{i};

    % --- HEC-HMS (no lag offset needed — direct comparison) ---
    % Use full normalized series (no lag trim for HEC-HMS standalone)
    Qobs_i_full = ev_i.Qobs_n;
    Qhms_i_full = ev_i.Qhms_n;
    m_hms_i = calc_metrics(Qobs_i_full, Qhms_i_full);

    % --- ANFIS ---
    [Xi_a, Yi_a] = build_lag_features(ev_i.P_n, ev_i.Qobs_n, LAG);
    if any(i == train_idx)
        % Training event: use trained FIS
        Qsim_i_n = evalfis(fis_anfis_best, Xi_a);
    else
        % Test event
        Qsim_i_n = evalfis(fis_anfis_best, Xi_a);
    end
    m_anfis_i = calc_metrics(Yi_a, Qsim_i_n);

    % --- Hybrid ---
    res_i_n = ev_i.Qobs_n - ev_i.Qhms_n;
    [Xi_h, ~] = build_hybrid_features(ev_i.P_n, ev_i.Qhms_n, ev_i.Qobs_n, res_i_n, LAG);
    Qres_i_n  = evalfis(fis_hyb_best, Xi_h);
    Qhms_i_lag = ev_i.Qhms_n(LAG+1:end);
    Qhyb_i_n   = Qhms_i_lag + Qres_i_n;
    Qobs_i_lag = Yi_a;
    m_hyb_i   = calc_metrics(Qobs_i_lag, Qhyb_i_n);

    % Phase label
    if i == test_idx
        phase = '[TEST]';
    else
        phase = '[TRAIN]';
    end
    event_label = sprintf('%s %s', ev_i.name, phase);

    % Print row for each model
    fprintf('%-20s %-10s %7.4f %7.4f %7.4f %7.4f %8.2f\n', ...
        event_label, 'HEC-HMS', m_hms_i.NSE, m_hms_i.RMSE, m_hms_i.MAE, m_hms_i.KGE, m_hms_i.PBIAS);
    fprintf('%-20s %-10s %7.4f %7.4f %7.4f %7.4f %8.2f\n', ...
        '', 'ANFIS', m_anfis_i.NSE, m_anfis_i.RMSE, m_anfis_i.MAE, m_anfis_i.KGE, m_anfis_i.PBIAS);
    fprintf('%-20s %-10s %7.4f %7.4f %7.4f %7.4f %8.2f\n', ...
        '', 'Hybrid', m_hyb_i.NSE, m_hyb_i.RMSE, m_hyb_i.MAE, m_hyb_i.KGE, m_hyb_i.PBIAS);
    fprintf('%s\n', repmat('-',1,75));
end

%% =========================================================
%  8. TEST EVENT SUMMARY (quick reference)
% =========================================================

m_anfis = calc_metrics(Qobs_test_n, Qsim_anfis_n);
m_hyb   = calc_metrics(Qobs_test_n, Qhyb_n);
m_hms   = calc_metrics(Qobs_test_n, Qhms_test_n);

fprintf('\n=== TEST EVENT QUICK SUMMARY (normalised discharge values) ===\n');
fprintf('%-12s %8s %8s %8s %8s %8s\n', 'Model','NSE','RMSE','MAE','KGE','PBIAS');
fprintf('%-12s %8.4f %8.4f %8.4f %8.4f %8.2f\n', 'HEC-HMS', ...
    m_hms.NSE, m_hms.RMSE, m_hms.MAE, m_hms.KGE, m_hms.PBIAS);
fprintf('%-12s %8.4f %8.4f %8.4f %8.4f %8.2f\n', 'ANFIS', ...
    m_anfis.NSE, m_anfis.RMSE, m_anfis.MAE, m_anfis.KGE, m_anfis.PBIAS);
fprintf('%-12s %8.4f %8.4f %8.4f %8.4f %8.2f\n', 'Hybrid', ...
    m_hyb.NSE, m_hyb.RMSE, m_hyb.MAE, m_hyb.KGE, m_hyb.PBIAS);

%% =========================================================
%  9. FIGURES
% =========================================================

if SAVE_FIGURES && ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

%--- Figure 1: ANFIS Training Progress ---
figure('Name','ANFIS Training Progress','Position',[100 100 700 400]);
plot(1:N_EPOCHS, train_error, 'b-',  'LineWidth', 1.5); hold on;
plot(1:N_EPOCHS, val_error,   'r-',  'LineWidth', 1.5);
xlabel('Epoch'); ylabel('RMSE');
title(sprintf('ANFIS Training Progress | Lag=%d', LAG));
legend('Training RMSE','Internal Validation RMSE','Location','NorthEast');
grid on;
if SAVE_FIGURES
    saveas(gcf, fullfile(OUTPUT_DIR, ['anfis_training.' FIG_FORMAT]));
end

%--- Figure 2: ANFIS Validation Hydrograph (normalised) ---
figure('Name','ANFIS Validation Hydrograph','Position',[100 100 800 450]);
plot(time_test, Qobs_test_n,  'k-',  'LineWidth', 1.5); hold on;
plot(time_test, Qsim_anfis_n, 'r--', 'LineWidth', 1.5);
xlabel('Time (min)'); ylabel('Normalised Discharge (-)');
title(sprintf('Validation Hydrograph (Test Event) | NSE=%.4f | RMSE=%.4f', ...
    m_anfis.NSE, m_anfis.RMSE));
legend('Observed','ANFIS','Location','NorthEast');
grid on;
if SAVE_FIGURES
    saveas(gcf, fullfile(OUTPUT_DIR, ['anfis_validation.' FIG_FORMAT]));
end

%--- Figure 3: Hybrid Model Validation Hydrograph (normalised) ---
figure('Name','Hybrid Model Validation','Position',[100 100 800 450]);
plot(time_test, Qobs_test_n, 'k-',  'LineWidth', 1.5); hold on;
plot(time_test, Qhms_test_n, 'b--', 'LineWidth', 1.5);
plot(time_test, Qhyb_n,      'r-',  'LineWidth', 1.5);
xlabel('Time (min)'); ylabel('Normalised Discharge (-)');
title(sprintf('Hybrid Model Performance | NSE=%.4f | RMSE=%.4f', ...
    m_hyb.NSE, m_hyb.RMSE));
legend('Observed','HEC-HMS','Hybrid','Location','NorthEast');
grid on;
if SAVE_FIGURES
    saveas(gcf, fullfile(OUTPUT_DIR, ['hybrid_validation.' FIG_FORMAT]));
end

%--- Figure 4: Discharge hydrograph in physical units (m3/s) ---
figure('Name','Hydrograph m3/s','Position',[100 100 800 450]);
plot(time_test, Qobs_raw,  'k-',  'LineWidth', 1.5); hold on;
plot(time_test, Qanfis,    'r--', 'LineWidth', 1.5);
plot(time_test, Qhyb,      'b-',  'LineWidth', 1.5);
xlabel('Time (min)'); ylabel('Discharge (m^3/s)');
title(sprintf('Validation Hydrograph | Q_{peak,obs}=%.4f | Q_{peak,hyb}=%.4f m^3/s', ...
    max(Qobs_raw), max(Qhyb)));
legend('Observed','ANFIS','Hybrid','Location','NorthEast');
grid on;
if SAVE_FIGURES
    saveas(gcf, fullfile(OUTPUT_DIR, ['hydrograph_m3s.' FIG_FORMAT]));
end

%--- Figure 5: Scatter plot – Observed vs. Simulated (ANFIS) ---
figure('Name','Scatter Plot','Position',[100 100 500 500]);
scatter(Qobs_test_n, Qsim_anfis_n, 40, 'r', 'filled'); hold on;
lims = [0, max(max(Qobs_test_n), max(Qsim_anfis_n))*1.05];
plot(lims, lims, 'k--', 'LineWidth', 1.2);
xlabel('Observed Normalised Discharge (-)');
ylabel('Simulated Normalised Discharge (-)');
title('Observed vs. Simulated Discharge (ANFIS – Test Event)');
legend(sprintf('ANFIS  R^2=%.4f', m_anfis.NSE), '1:1 Line', 'Location','NorthWest');
axis([lims lims]); grid on;
if SAVE_FIGURES
    saveas(gcf, fullfile(OUTPUT_DIR, ['scatter_anfis.' FIG_FORMAT]));
end

%--- Figure 6: Hybrid ANFIS Training Progress ---
figure('Name','Hybrid ANFIS Training Progress','Position',[100 100 700 400]);
plot(1:N_EPOCHS, train_error_h, 'b-', 'LineWidth', 1.5); hold on;
plot(1:N_EPOCHS, val_error_h,   'r-', 'LineWidth', 1.5);
xlabel('Epoch'); ylabel('RMSE');
title(sprintf('Hybrid ANFIS Training Progress | Lag=%d', LAG));
legend('Training RMSE','Internal Validation RMSE','Location','NorthEast');
grid on;
if SAVE_FIGURES
    saveas(gcf, fullfile(OUTPUT_DIR, ['hybrid_training.' FIG_FORMAT]));
end

fprintf('\n=== Completed ===\n');
fprintf('Peak discharge – Observed: %.4f m3/s | ANFIS: %.4f m3/s | Hybrid: %.4f m3/s\n', ...
    max(Qobs_raw), max(Qanfis), max(Qhyb));
