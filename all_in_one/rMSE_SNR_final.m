clear; clc; close all;

snr_db = [10 20 30 40 50];
burnIn = 63;
% groupA = ["./final_data_log/localization/MC20/MLE_10db/Log.mat", ...
%           "./final_data_log/localization/MC20/MLE_20db/Log.mat", ...
%           "./final_data_log/localization/MC20/MLE_30db/Log.mat", ...
%           "./final_data_log/localization/MC20/MLE_40db/Log.mat", ...
%           "./final_data_log/localization/MC20/MLE_50db/Log.mat"];
% 
% groupB = ["./final_data_log/localization/MC20/MAP_10db/Log.mat", ...
%           "./final_data_log/localization/MC20/MAP_20db/Log.mat", ...
%           "./final_data_log/localization/MC20/MAP_30db/Log.mat", ...
%           "./final_data_log/localization/MC20/MAP_40db/Log.mat", ...
%           "./final_data_log/localization/MC20/MAP_50db/Log.mat"];


groupA = ["./data_log/rMSE_SNR/MAP_5db_tracking/Log.mat", ...
           "./data_log/rMSE_SNR/MAP_15db_tracking/Log.mat", ...
           "./data_log/rMSE_SNR/MAP_30db_tracking/Log.mat", ...
           "./data_log/rMSE_SNR/MAP_50db_tracking/Log.mat"];

groupB = ["./data_log/rMSE_SNR/MLE_5db_tracking/Log.mat", ...
           "./data_log/rMSE_SNR/MLE_15db_tracking/Log.mat", ...
           "./data_log/rMSE_SNR/MLE_30db_tracking/Log.mat", ...
           "./data_log/rMSE_SNR/MLE_50db_tracking/Log.mat"];



% groupC = ["../ADMM-EKF/data_log/dkf_mc20_10db/Log.mat",...
%           "../ADMM-EKF/data_log/dkf_mc2_20db/Log.mat",...
%           "../ADMM-EKF/data_log/dkf_mc20_30db/Log.mat",...
%           "../ADMM-EKF/data_log/dkf_mc2_40db/Log.mat",...
%           "../ADMM-EKF/data_log/dkf_mc2_50db/Log.mat"];

groupC= ["../ADMM-EKF/data_log/MC3/dkf_10db/Log.mat",...
    "../ADMM-EKF/data_log/MC5/dkf_20db/Log.mat",...
    "../ADMM-EKF/data_log/MC5/dkf_30db/Log.mat"...
    "../ADMM-EKF/data_log/MC5/dkf_40db/Log.mat"...
    ,"../ADMM-EKF/data_log/MC5/dkf_50db/Log.mat"];



groups = {groupA, groupB, groupC};
groupNames = ["MLE", "MAP", "EKF"];

nState = 4;
rmseCA = nan(numel(groups), numel(snr_db), nState);
rmseDA = nan(numel(groups), numel(snr_db), nState);

for g = 1:numel(groups)
    files = groups{g};

    for i = 1:numel(files)
        if ~isfile(files(i))
            warning("File not found: %s", files(i));
            continue;
        end

        S = load(files(i));
        try
            Log = S.Log;
        catch
            Log = S.log;
        end
        [ca, da] = compute_rmse_from_log(Log, burnIn);
        rmseCA(g,i,:) = ca;
        rmseDA(g,i,:) = da;
    end
end

stateLabels = {'RMSE_x', 'RMSE_y', 'RMSE_{v_x}', 'RMSE_{v_y}'};

figure('Color','w');
tiledlayout(4,1,'TileSpacing','compact','Padding','compact');

colors = lines(numel(groups));

for s = 1:nState
    nexttile; hold on; grid on; box on;

    for g = 1:numel(groups)
        semilogy(snr_db, squeeze(rmseDA(g,:,s)), '-o', ...
            'Color', colors(g,:), ...
            'LineWidth', 1.4, ...
            'MarkerSize', 4, ...
            'DisplayName', "D-" + groupNames(g));

        semilogy(snr_db, squeeze(rmseCA(g,:,s)), '--o', ...
            'Color', colors(g,:), ...
            'LineWidth', 1.4, ...
            'MarkerSize', 4, ...
            'DisplayName', "C-" + groupNames(g));
    end

    ylabel(stateLabels{s}, 'Interpreter','tex');
    set(gca, 'YScale', 'log');

    if s == 1
        legend('Location','northeast');
    end

    if s == nState
        xlabel('SNR (dB)');
    else
        set(gca,'XTickLabel',[]);
    end
end


function [rmseCA, rmseDA] = compute_rmse_from_log(Log, burnIn)
    R = Log.Results;

    useRaw = isfield(R, 'estimations_CA_raw') && ...
             isfield(R, 'estimations_DA_raw') && ...
             isfield(R, 'true_params_raw');

    if useRaw
        CA = R.estimations_CA_raw;
        DA = R.estimations_DA_raw;
        TR = R.true_params_raw;
    else
        CA = R.estimations_CA;
        DA = R.estimations_DA;
        TR = R.true_params;
    end

    e2CA = [];
    e2DA = [];

    for k = 1:numel(TR)
        if useRaw
            [~, tIdx] = ind2sub(size(TR), k);

            if tIdx <= burnIn
                continue;
            end
        end

        if isempty(TR{k}) || isempty(CA{k}) || isempty(DA{k})
            continue;
        end

        e2CA = [e2CA, squared_error_by_state(CA{k}, TR{k})];
        e2DA = [e2DA, squared_error_by_state(DA{k}, TR{k})];
    end

    rmseCA = sqrt(mean(e2CA, 2, 'omitnan'));
    rmseDA = sqrt(mean(e2DA, 2, 'omitnan'));
end


function e2 = squared_error_by_state(est, truth)
    truth = squeeze(double(truth));
    truth = truth(:);
    truth = truth(1:4);

    est = squeeze(double(est));
    sz = size(est);

    stateDim = find(sz == 4, 1, 'first');

    if isempty(stateDim)
        error("Cannot find state dimension of size 4. est size = %s", mat2str(sz));
    end

    order = [stateDim, setdiff(1:ndims(est), stateDim, 'stable')];
    est = permute(est, order);
    est = reshape(est, 4, []);

    e = est - truth;
    e2 = e.^2;
end 