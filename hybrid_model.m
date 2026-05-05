% hybrid_model.m
% ANFIS-Based Hybrid Rainfall-Runoff Model
%
% Combines a physics-based routing model (HEC-HMS) with an ANFIS
% residual-correction layer to improve discharge predictions.
%
% Inputs:
%   train_1988.xlsx  - Training data (1988 storm event)
%   test_2005.xlsx   - Test data     (2005 storm event)
%
% Columns expected in each file:
%   minute  - time step index
%   preNOM  - normalized precipitation
%   Qnom    - normalized observed discharge
%   hed     - normalized HEC-HMS simulated discharge
%
% Outputs:
%   Console: NSE and RMSE for the hybrid model on the test set
%   Figure:  Observed vs. HEC-HMS vs. Hybrid discharge time series

clc; clear; close all;

%% 1. Load data
train = readtable('train_1988.xlsx');
test  = readtable('test_2005.xlsx');

% Column values are stored as strings in some Excel exports
P_train = str2double(string(train.preNOM));
Q_train = str2double(string(train.Qnom));
H_train = str2double(string(train.hed));

P_test  = str2double(string(test.preNOM));
Q_test  = str2double(string(test.Qnom));
H_test  = str2double(string(test.hed));

%% 2. Feature engineering
% One-step lagged discharge captures temporal autocorrelation
Qlag_train = [NaN; Q_train(1:end-1)];
Qlag_test  = [NaN; Q_test(1:end-1)];

% Residual = observed - HEC-HMS; this is what ANFIS will learn to predict
res_train = Q_train - H_train;
res_test  = Q_test  - H_test;

%% 3. Remove rows with NaN (first row lost due to lag)
data_train = [P_train, H_train, Qlag_train, res_train];
data_train = data_train(~any(isnan(data_train), 2), :);
train_in  = data_train(:, 1:3);
train_out = data_train(:, 4);

data_test = [P_test, H_test, Qlag_test, res_test];
data_test = data_test(~any(isnan(data_test), 2), :);
test_in  = data_test(:, 1:3);
test_out = data_test(:, 4);

%% 4. Build and train ANFIS
% genfis2 constructs an initial Sugeno FIS via subtractive clustering.
% The influence radius (0.8) controls the number of fuzzy rules generated.
fis = genfis2(train_in, train_out, 0.8);

opt = anfisOptions('InitialFIS', fis);
opt.EpochNumber = 50;

anfis_model = anfis([train_in, train_out], opt);

%% 5. Hybrid prediction on the test set
% ANFIS predicts the residual; add it back onto the HEC-HMS baseline
res_pred = evalfis(anfis_model, test_in);

Q_hybrid = test_in(:, 2) + res_pred;          % HEC-HMS + predicted residual
Q_obs    = test_out    + test_in(:, 2);        % reconstruct observed Q

%% 6. Performance metrics
NSE  = 1 - sum((Q_obs - Q_hybrid).^2) / sum((Q_obs - mean(Q_obs)).^2);
RMSE = sqrt(mean((Q_obs - Q_hybrid).^2));

fprintf('--- Test Set Performance (2005 Event) ---\n');
fprintf('NSE  : %.4f\n', NSE);
fprintf('RMSE : %.6f\n', RMSE);

%% 7. Plot results
figure('Name', 'Hybrid Model Performance', 'NumberTitle', 'off');
plot(Q_obs,          'k',   'LineWidth', 2); hold on;
plot(test_in(:, 2),  'b--', 'LineWidth', 2);
plot(Q_hybrid,       'r',   'LineWidth', 2);
legend('Observed', 'HEC-HMS', 'Hybrid', 'Location', 'best');
title(sprintf('Hybrid Model — Test Event 2005   (NSE = %.3f)', NSE));
xlabel('Time Step (5-min intervals)');
ylabel('Normalized Discharge');
grid on;
