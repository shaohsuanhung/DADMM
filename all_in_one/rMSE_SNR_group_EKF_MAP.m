function results = plot_convergence_iters_vs_snr_multi(groups, varargin)
%PLOT_CONVERGENCE_ITERS_VS_SNR_MULTI  Plot "number of optimization iterations" vs SNR.
%
% Interpretation (per your note):
%   The convergence-iteration count is taken as the size of the LAST dimension of
%   Log.Results.estimations_DA (or Log.Result.estimations_DA).
%   Example numeric layout: estimations_DA is [nState x nSensor x nIter]  -> nIter
%
% groups: struct array with fields
%   groups(k).name  : e.g. "MAP"
%   groups(k).files : string array / cellstr of .mat paths (each contains variable `Log`)
%
% Expected fields in each .mat:
%   - variable `Log`
%   - Log.Results.estimations_DA (or Log.Result.estimations_DA)
%   - Log.constant.SNR_idx (dB) (optional; fallback: parse from file path like "30db")
%
% Name-Value options:
%   'Aggregate' : how to aggregate if estimations_DA is a cell and nIter differs per entry
%                 'mean' (default), 'max', 'median', 'min'
%
% Output results(k):
%   .name
%   .snr_dB     [NSNR x 1]
%   .nIter      [NSNR x 1]   aggregated iteration count per file
%   .nIterStd   [NSNR x 1]   std across entries (only meaningful for cell case)

p = inputParser;
p.addRequired('groups', @(g) isstruct(g) && all(isfield(g, {'name','files'})));
p.addParameter('Aggregate', 'mean', @(s) any(strcmpi(s, {'max','mean','median','min'})));
p.parse(groups, varargin{:});
agg = lower(string(p.Results.Aggregate));

results = struct('name', [], 'snr_dB', [], 'nIter', [], 'nIterStd', []);

for k = 1:numel(groups)
    files = groups(k).files;
    if ischar(files) || (isstring(files) && isscalar(files))
        files = string(files);
    elseif iscell(files)
        files = string(files);
    end
    files = files(:);

    snr_dB   = nan(numel(files), 1);
    nIter    = nan(numel(files), 1);
    nIterStd = nan(numel(files), 1);

    for i = 1:numel(files)
        S = load(files(i));
        if ~isfield(S, 'Log')
            error('File "%s" does not contain variable named "Log".', files(i));
        end
        L = S.Log;

        snr_dB(i) = get_snr_db(L, files(i));

        R = get_results_struct(L);
        if ~isfield(R, 'estimations_DA')
            error('Missing estimations_DA under Log.Results (or Log.Result) in "%s".', files(i));
        end

        [nIter(i), nIterStd(i)] = get_iteration_count(R.convg_iter, agg);
    end

    % sort by SNR
    [snr_dB, idx] = sort(snr_dB);
    nIter = nIter(idx);
    nIterStd = nIterStd(idx);

    results(k).name = string(groups(k).name);
    results(k).snr_dB = snr_dB;
    results(k).nIter = nIter;
    results(k).nIterStd = nIterStd;
end

% -------------------- plot overlay --------------------
figure('Color','w'); hold on; grid on; box on;
set(gca,'FontSize',30);
set(gca, 'FontName', 'Times New Roman');
for k = 1:numel(results)
    x = results(k).snr_dB(:);
    y = results(k).nIter(:);

    plot(x, y, '-o', 'LineWidth', 1.5, 'MarkerSize', 7);

    % If you want to visualize variability for cell-case, uncomment:
    % if any(isfinite(results(k).nIterStd)) && any(results(k).nIterStd > 0)
    %     errorbar(x, y, results(k).nIterStd(:), 'LineStyle', 'none', 'CapSize', 10);
    % end
end

xlabel('SNR (dB)');
ylabel('Number of optimization iterations (last dim of estimations\_DA)');
title('Convergence iterations vs SNR');
legend(string({results.name}), 'Location', 'best');

end

function results = plot_rmse_vs_snr_multi(groups, varargin)
%PLOT_RMSE_VS_SNR_MULTI  4x1 subplot: per-state RMSE vs SNR (multiple methods),
%                        also read per-state CRLB vs SNR, and show RMSE variance
%                        via ±3σ shading around the mean RMSE curve.
%
% groups: struct array with fields
%   groups(k).name  : e.g. "MAP"
%   groups(k).files : string array / cellstr of .mat paths (each contains variable `Log`)
%
% Each .mat is expected to contain variable `Log` with (either field name):
%   Log.Results.estimations_DA   (or Log.Result.estimations_DA)
%   Log.Results.true_params      (or Log.Result.true_params)
%   Log.Results.CRLB             (or Log.Result.CRLB)   <-- per-state CRLB uses diag
%   Log.constant.SNR_idx (dB)    (optional; fallback: parse from file path "30db")
%
% Name-Value options:
%   'StateIdx'    : default [1 2 3 4] (x,y,vx,vy) MUST be length 4 for 4x1 plot
%   'StateNames'  : default {'x','y','v_x','v_y'} (length 4)
%   'SemilogY'    : true (default) / false
%   'ShowCRLB'    : true (default) / false
%   'ShadeSigma'  : default 3 (shade = mean ± ShadeSigma * std)
%
% Output results(k):
%   .name
%   .snr_dB            [NSNR x 1]
%   .rmse_mean_state   [4 x NSNR]
%   .rmse_var_state    [4 x NSNR]   (variance of RMSE across MC runs, if available)
%   .rmse_std_state    [4 x NSNR]
%   .crlb_state        [4 x NSNR]   (sqrt(mean(CRLB_ii)) across (t,node,...))

p = inputParser;
p.addRequired('groups', @(g) isstruct(g) && all(isfield(g, {'name','files'})));
p.addParameter('StateIdx', [1 2 3 4], @(v) isnumeric(v) && isvector(v) && numel(v)==4);
p.addParameter('StateNames', {'$x$','$y$','$v_x$','$v_y$'}, @(c) iscell(c) && numel(c)==4);
p.addParameter('SemilogY', true, @(b) islogical(b) && isscalar(b));
p.addParameter('ShowCRLB', true, @(b) islogical(b) && isscalar(b));
p.addParameter('ShadeSigma', 3, @(x) isnumeric(x) && isscalar(x) && x >= 0);
p.parse(groups, varargin{:});

stateIdx   = p.Results.StateIdx(:);
stateNames = p.Results.StateNames;
useSemi    = p.Results.SemilogY;
showCRLB   = p.Results.ShowCRLB;
kSigma     = p.Results.ShadeSigma;

% -------------------- compute curves per group --------------------
results = struct('name', [], 'snr_dB', [], ...
    'rmse_mean_state', [], 'rmse_var_state', [], 'rmse_std_state', [], ...
    'crlb_state', []);

for k = 1:numel(groups)
    files = groups(k).files;
    if ischar(files) || (isstring(files) && isscalar(files))
        files = string(files);
    elseif iscell(files)
        files = string(files);
    end
    files = files(:);

    snr_dB          = nan(numel(files), 1);
    rmse_mean_state = nan(4, numel(files));
    rmse_var_state  = nan(4, numel(files));
    rmse_std_state  = nan(4, numel(files));
    rmse_mean_state_CA = nan(4, numel(files));
    rmse_var_state_CA  = nan(4, numel(files));
    rmse_std_state_CA  = nan(4, numel(files));
    crlb_state      = nan(4, numel(files));

    for i = 1:numel(files)
        S = load(files(i));
        if ~isfield(S, 'Log')
            error('File "%s" does not contain variable named "Log".', files(i));
        end
        L = S.Log;

        snr_dB(i) = get_snr_db(L, files(i));

        if L.TYPE == "MAP"
            [m4, v4, s4] = compute_final_rmse_state_stats_from_log(L, stateIdx);
            [m4_CA, v4_CA, s4_CA] = compute_final_rmse_state_stats_from_log_CA(L, stateIdx);
        elseif L.TYPE=="MLE"
            [m4, v4, s4] = compute_final_rmse_state_stats_from_log_MLE(L, stateIdx);
            [m4_CA, v4_CA, s4_CA] = compute_final_rmse_state_stats_from_log_MLE_CA(L, stateIdx);
        elseif L.TYPE=="DKF_ADMM"
            [m4, v4, s4] = compute_final_rmse_state_stats_from_log_MLE(L, stateIdx);
            [m4_CA, v4_CA, s4_CA] = compute_final_rmse_state_stats_from_log_MLE_CA(L, stateIdx);
        else
            error("Non-defined type.")
        end
        rmse_mean_state(:, i) = m4;
        rmse_var_state(:, i)  = v4;
        rmse_std_state(:, i)  = s4;

        rmse_mean_state_CA(:, i) = m4_CA;
        rmse_var_state_CA(:, i)  = v4_CA;
        rmse_std_state_CA(:, i)  = s4_CA;

        if showCRLB
            crlb_state(:, i) = compute_crlb_state_from_log(L, stateIdx);
        end
    end
    

    % sort by SNR
    [snr_dB, idx] = sort(snr_dB);
    rmse_mean_state = rmse_mean_state(:, idx);
    rmse_var_state  = rmse_var_state(:, idx);
    rmse_std_state  = rmse_std_state(:, idx);
    crlb_state      = crlb_state(:, idx);

    results(k).name            = string(groups(k).name);
    results(k).snr_dB          = snr_dB;
    results(k).rmse_mean_state = rmse_mean_state;
    results(k).rmse_var_state  = rmse_var_state;
    results(k).rmse_std_state  = rmse_std_state;
    results(k).crlb_state      = crlb_state;
    results(k).rmse_mea_state_ctrl = rmse_mean_state_CA;
    results(k).rmse_std_state_ctrl = rmse_std_state_CA;
end

% -------------------- plot: 4x1 tiles, overlay methods --------------------
figure('Color','w');
tl = tiledlayout(4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

hLegend = gobjects(0);
leg = strings(0);

for s = 1:4
    ax = nexttile(tl, s);
    set(gca, 'FontName', 'Times New Roman');
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');

    for k = 1:numel(results)
        x  = results(k).snr_dB(:);
        mu = results(k).rmse_mean_state(s, :).';
        sd = results(k).rmse_std_state(s, :).';
        mu_CA = results(k).rmse_mea_state_ctrl(s,:).';
        sd_CA = results(k).rmse_std_state_ctrl(s, :).';

        % --- mean RMSE line (auto color from axes) ---
        mu = sd;
        if useSemi
            hMean= semilogy(ax, x, mu, '-o', 'LineWidth', 3, 'MarkerSize', 6);
        else
            hMean = plot(ax, x, mu, '-o', 'LineWidth', 3, 'MarkerSize', 6);
        end
        
        %% ctrl rMSE
        mu_CA = sd_CA;
        if results(k).name == "MAP"
        if useSemi
            hMean_CA = semilogy(ax, x, mu_CA, '--o', 'LineWidth', 3, 'MarkerSize', 6);
        else
            hMean_CA = plot(ax, x, mu_CA, '--o', 'LineWidth', 3, 'MarkerSize', 6);
        end
        end

        if results(k).name == "EKF"
        if useSemi
            hMean_CA = semilogy(ax, x, mu_CA, '--o', 'LineWidth', 3, 'MarkerSize', 6);
        else
            hMean_CA = plot(ax, x, mu_CA, '--o', 'LineWidth', 3, 'MarkerSize', 6);
        end
        end

        % --- CRLB line (per-state), same color as mean RMSE ---
        hCRLB = gobjects(0);
        if showCRLB
            yb = results(k).crlb_state(s, :).';
            if any(isfinite(yb))
                if useSemi
                    hCRLB = semilogy(ax, x, yb, '-*', 'LineWidth', 1.5);
                else
                    hCRLB = plot(ax, x, yb, '-*', 'LineWidth', 1.5);
                end
                hCRLB.Color = hMean.Color;
            end
        end

        % legend handles (only from first subplot)
        if s == 1
            hLegend(end+1) = hMean; %#ok<AGROW>
            leg(end+1) = "D-" + results(k).name; %#ok<AGROW>
            if showCRLB && ~isempty(hCRLB) && isgraphics(hCRLB)
                hLegend(end+1) = hCRLB; %#ok<AGROW>
                leg(end+1) = results(k).name + " CRLB"; %#ok<AGROW>
            end
            if results(k).name == "MAP"
            hLegend(end+1) = hMean_CA; %#ok<AGROW>
            leg(end+1) = "C-" + results(k).name; %#ok<AGROW>
            end

            if results(k).name == "EKF"
            hLegend(end+1) = hMean_CA; %#ok<AGROW>
            leg(end+1) = "C-" + results(k).name; %#ok<AGROW>
            end
            % leg(end+1) = "Ｃ-"+ L.TYPE;
            % leg(end+1) = L.TYPE;
        end
    end

    ylabel(ax, sprintf('RMSE %s', stateNames{s}),'Interpreter','latex');
    ax.FontSize = 25;
    if s == 1
        % if kSigma > 0
        %     title(ax, sprintf('Final RMSE per state vs SNR (mean with \\pm%d\\sigma shade) + CRLB', kSigma));
        % else
        %     title(ax, 'Final RMSE per state vs SNR (mean) + CRLB');
        % end
        legend(ax, hLegend, leg, 'Location', 'best','FontSize',15);
    end

    if s ~= 4
        set(ax, 'XTickLabel', []);
    else
        xlabel(ax, 'SNR (dB)');
    end
    xticks([5 10 20 30 40 50]);
    xlim([5 50]);
    if useSemi
    set(gca, 'YScale', 'log')
    end
    grid on;
end

end


% ===================== local helpers =====================


function [mean4, var4, std4] = compute_final_rmse_state_stats_from_log_CA(L, stateIdx)
    R = get_results_struct(L);
    est = R.estimations_CA;
    tru = R.true_params;
    % first pass: find nMC from any non-empty entry
    nMC = [];
    [T, N] = size(est); % T: number of MC, N: tracking prediction
    nMC = T;

    % accumulate SSE per MC across (t,node)
    sse = zeros(4, nMC);
    cnt = zeros(1, nMC);

    for t = 1:T
        for n = 1:N
            E = est{t,n};
            if isempty(E), continue; end
            if (size(E) == [1,4]), E = E';end
            % Ef = extract_final_est(E);         % [nState x nMC]
            % Ef = Ef(stateIdx, :);              % [4 x nMC]

            if iscell(tru)
                xtrue = tru{t,n};
            else
                xtrue = tru(:, min(n, size(tru,2)));
            end
            err = zeros(4,10);
            if isvector(xtrue)
                xtrue = xtrue(:);
                xtrue = xtrue(stateIdx);       % [4 x 1]
                xtrue = repmat(xtrue,1,size(E,2)); % size(est,2) = num. node
                err = err + E;
            end

            sse(:,t) = mean(err,2); % mean over node
            cnt = cnt + 1;
        end
    end
    mean4 = mean(sse, 2, 'omitnan'); % mean over MC
    std4  = std(sse, 0, 2, 'omitnan');
    var4  = std4.^2;
    return;
end

function [mean4, var4, std4] = compute_final_rmse_state_stats_from_log_MLE_CA(L, stateIdx)
    R = get_results_struct(L);
    est = R.estimations_CA;
    tru = R.true_params;
    % tru = R.
    % first pass: find nMC from any non-empty entry
    nMC = [];
    [T, N] = size(est); % T: number of MC, N: tracking prediction
    nMC = T;

    % accumulate SSE per MC across (t,node)
    sse = zeros(4, nMC);
    cnt = zeros(1, nMC);

    for t = 1:T
        for n = 1:N
            E = est{t,n};
            if isempty(E), continue; end
            if (size(E) == [1,4]), E = E'; end
            % Ef = extract_final_est(E);         % [nState x nMC]
            % Ef = Ef(stateIdx, :);              % [4 x nMC]

            if iscell(tru)
                xtrue = tru{t,n};
            else
                xtrue = tru(:, min(n, size(tru,2)));
            end
            err = zeros(4,10);
            % if E
            if isvector(xtrue)
                xtrue = xtrue(:);
                xtrue = xtrue(stateIdx);       % [4 x 1]
                xtrue = repmat(xtrue,1,size(E,2)); % size(est,2) = num. node
                err = err + E;
            else
                eror("true param should be vector")

            end

            sse(:,t) = mean(err,2); % mean over node
            cnt = cnt + 1;
        end
    end
    %TODO Correct here
    mean4 = mean(sse, 2, 'omitnan'); % mean over MC
    std4  = std(sse, 0, 2, 'omitnan');
    var4  = std4.^2;
    return;
end


function [mean4, var4, std4] = compute_final_rmse_state_stats_from_log_MLE(L, stateIdx)
% Returns:
%   mean4, var4, std4: [4 x 1] statistics across Monte-Carlo runs (if available)
% If no MC dimension is found, mean4 is computed and std/var are NaN.

R = get_results_struct(L);
if ~isfield(R,'estimations_DA') || ~isfield(R,'true_params')
    error('Missing estimations_DA / true_params under Log.Results (or Log.Result).');
end

est = R.estimations_DA;
tru = R.true_params;

% --- Case 1: cell layout (recommended / typical) ---
if iscell(est)
    % first pass: find nMC from any non-empty entry
    nMC = [];
    [T, N] = size(est); % T: number of MC, N: tracking prediction
    nMC = T;

    % accumulate SSE per MC across (t,node)
    sse = zeros(4, nMC);
    cnt = zeros(1, nMC);

    for t = 1:T
        for n = 1:N
            E = est{t,n};
            if isempty(E), continue; end

            % Ef = extract_final_est(E);         % [nState x nMC]
            % Ef = Ef(stateIdx, :);              % [4 x nMC]

            if iscell(tru)
                xtrue = tru{t,n};
            else
                xtrue = tru(:, min(n, size(tru,2)));
            end
            err = zeros(4,10);
            if isvector(xtrue)
                xtrue = xtrue(:);
                xtrue = xtrue(stateIdx);       % [4 x 1]
                xtrue = repmat(xtrue,1,size(E,2)); % size(est,2) = num. node
                % broadcast -> [4 x nMC]
                % err = % Averaging over node 
                % Testing1, first averging node-wise error from [4x10]->[4x1]
                % err = mean(err,2);
                % Testing2, sum over node instead of avg. over node.
                err = sum(abs(E - xtrue).^2,2);
                % err = err + E;
            else
                eror("true param should be vector")
                % % expect [nState x nMC]
                % if size(xtrue,2) ~= nMC
                %     error('Truth dimension mismatch: expected %d MC samples, got %d.', nMC, size(xtrue,2));
                % end
                % xtrue = xtrue(stateIdx, :);    % [4 x nMC]
                % err = Ef - xtrue;
            end

            sse(:,t) = mean(err,2); % mean over node
            cnt = cnt + 1;
        end
    end
    %TODO Correct here
    rmse_mc = sqrt(sse ./ max(cnt,1));          % [1 x nMC]
    mse_mc = (sse ./ max(cnt,1));          % [1 x nMC]
    rmse_mc = sqrt(sse);          % [1 x nMC]
    mean4 = mean(rmse_mc, 2, 'omitnan'); % mean over MC
    std4  = std(rmse_mc, 0, 2, 'omitnan');
    var4  = std4.^2;
    return;
end

end

function [mean4, var4, std4] = compute_final_rmse_state_stats_from_log(L, stateIdx)
% Returns:
%   mean4, var4, std4: [4 x 1] statistics across Monte-Carlo runs (if available)
% If no MC dimension is found, mean4 is computed and std/var are NaN.

R = get_results_struct(L);
if ~isfield(R,'estimations_DA') || ~isfield(R,'true_params')
    error('Missing estimations_DA / true_params under Log.Results (or Log.Result).');
end

est = R.estimations_DA;
tru = R.true_params;

% --- Case 1: cell layout (recommended / typical) ---
if iscell(est)
    % first pass: find nMC from any non-empty entry
    nMC = [];
    [T, N] = size(est); % T: number of MC, N: tracking prediction
    nMC = T;
    % for t = 1:T
    %     for n = 1:N
    %         E = est{t,n};
    %         if ~isempty(E)
    %             Ef = extract_final_est(E); % [nState x nSample]
    %             nMC = size(Ef, 2);
    %             break;
    %         end
    %     end
    %     if ~isempty(nMC), break; end
    % end
    % if isempty(nMC)
    %     mean4 = nan(4,1); var4 = nan(4,1); std4 = nan(4,1);
    %     return;
    % end

    % accumulate SSE per MC across (t,node)
    sse = zeros(4, nMC);
    cnt = zeros(1, nMC);

    for t = 1:T
        for n = 1:N
            E = est{t,n};
            if isempty(E), continue; end

            % Ef = extract_final_est(E);         % [nState x nMC]
            % Ef = Ef(stateIdx, :);              % [4 x nMC]

            if iscell(tru)
                xtrue = tru{t,n};
            else
                xtrue = tru(:, min(n, size(tru,2)));
            end

            if isvector(xtrue)
                xtrue = xtrue(:);
                xtrue = xtrue(stateIdx);       % [4 x 1]
                xtrue = repmat(xtrue,1,size(E,2)); % size(est,2) = num. node
                err = E - xtrue;              % broadcast -> [4 x nMC]
                % err = % Averaging over node 
                % Testing1, first averging node-wise error from [4x10]->[4x1]
                % err = mean(err,2);
                % Testing2, sum over node instead of avg. over node.
                err = sum(err,2);
                % err = mean(err,2);
            else
                eror("true param should be vector")
                % % expect [nState x nMC]
                % if size(xtrue,2) ~= nMC
                %     error('Truth dimension mismatch: expected %d MC samples, got %d.', nMC, size(xtrue,2));
                % end
                % xtrue = xtrue(stateIdx, :);    % [4 x nMC]
                % err = Ef - xtrue;
            end

            sse(:,t) = err.^2;
            cnt = cnt + 1;
        end
    end
    %TODO Correct here
    rmse_mc = sqrt(sse ./ (max(cnt,1)));          % [4 x nMC]
    mean4 = mean(rmse_mc, 2, 'omitnan');
    std4  = std(rmse_mc, 0, 2, 'omitnan');
    var4  = std4.^2;

    % Debug Feb. 23
    % mse_mc = (sse ./ (max(cnt,1)));          % [4 x nMC]
    % mean4 = mean(mse_mc, 2, 'omitnan');
    % std4  = std(mse_mc, 0, 2, 'omitnan');
    % var4  = std4.^2;
    return;
end

% --- Case 2: numeric layout ---
nd = ndims(est);

if nd == 3
    % [nState x nNode x nIter] (no MC)
    Ef = est(:,:,end);
    Ef = Ef(stateIdx, :); % [4 x nNode]

    if isvector(tru)
        xtrue = tru(:);
        xtrue = xtrue(stateIdx); % [4 x 1]
        err = Ef - xtrue;        % [4 x nNode]
    else
        err = Ef - tru(stateIdx, :);
    end

    rmse4 = sqrt(mean(err.^2, 2));
    mean4 = rmse4;
    std4  = nan(4,1);
    var4  = nan(4,1);
    return;
end

if nd >= 4
    % assume [nState x nNode x nMC x nIter] and take final iter
    Ef = est(:,:,:,end);            % [nState x nNode x nMC]
    Ef = Ef(stateIdx, :, :);        % [4 x nNode x nMC]
    nMC = size(Ef, 3);

    % build truth broadcast to [4 x nNode x nMC] as needed
    if isvector(tru)
        xtrue = tru(:);
        xtrue = xtrue(stateIdx);                 % [4 x 1]
        err = Ef - reshape(xtrue, [4 1 1]);       % broadcast
    else
        tnd = ndims(tru);
        if tnd == 2
            % [nState x nNode]
            err = Ef - reshape(tru(stateIdx, :), [4 size(Ef,2) 1]);
        elseif tnd >= 3
            % try [nState x nNode x nMC (x ...)] take first 3 dims as match
            Ttru = tru(stateIdx, :, :);
            if size(Ttru,3) ~= nMC
                error('Truth MC dimension mismatch: expected %d, got %d.', nMC, size(Ttru,3));
            end
            err = Ef - Ttru;
        else
            error('Unsupported true_params numeric shape.');
        end
    end

    % SSE per MC (sum over nodes)
    sse = squeeze(sum(err.^2, 2));   % [4 x nMC]
    cnt = size(Ef, 2);               % nNode
    rmse_mc = sqrt(sse ./ max(cnt,1));

    mean4 = mean(rmse_mc, 2, 'omitnan');
    std4  = std(rmse_mc, 0, 2, 'omitnan');
    var4  = std4.^2;
    return;
end

error('Unsupported estimations_DA numeric shape.');
end
function snr_db = get_snr_db(L, filepath)
snr_db = NaN;

if isfield(L,'constant')
    C = L.constant;
    if isfield(C,'SNR_idx') && ~isempty(C.SNR_idx)
        snr_db = double(C.SNR_idx);
        return;
    end
    if isfield(C,'SNR_dB') && ~isempty(C.SNR_dB)
        snr_db = double(C.SNR_dB);
        return;
    end
    if isfield(C,'SNR_lin') && ~isempty(C.SNR_lin)
        snr_db = 10*log10(double(C.SNR_lin));
        return;
    end
end

% fallback: parse "..._30db_..." from the file path
tok = regexp(string(filepath), '(\d+)\s*db', 'tokens', 'once');
if ~isempty(tok)
    snr_db = str2double(tok{1});
    return;
end

error('Cannot find SNR in Log.constant.* and cannot parse from path: %s', filepath);
end

function R = get_results_struct(L)
% support both L.Results and L.Result
if isfield(L, 'Results'), R = L.Results; return; end
if isfield(L, 'Result'),  R = L.Result;  return; end
error('Cannot find Log.Results or Log.Result in the loaded Log struct.');
end

function rcrlb4 = compute_crlb_state_from_log(L, stateIdx)
% Returns crlb4: [4 x 1] where crlb4(i)=sqrt(mean(CRLB(ii))) averaged over (t,node,...)

R = get_results_struct(L);
if ~isfield(R, 'CRLB')
    rcrlb4 = nan(4,1);
    return;
end
CR = R.CRLB;

acc = zeros(4,1);
cnt = zeros(4,1);

if iscell(CR)
    [T, N] = size(CR);
    for t = 1:T
        for n = 1:N
            C = CR{t,n};
            if isempty(C), continue; end
            C = C(stateIdx, stateIdx);
            d = diag(C);                     % [4 x 1] Get the diagnoal element (1,1), (2,2),...,
            acc = acc + d;
            cnt = cnt + 1;
        end
    end
else
    error("CR shall be a cell.")
end

if L.TYPE == "MLE"
rcrlb4 = sqrt(acc ./ (max(cnt,1)));
elseif L.TYPE == "MAP"
rcrlb4 = sqrt(acc ./ (1.1*max(cnt,1)));
end
end

function Ef = extract_final_est(E)
% Convert E into [nState x nSample] at final iteration
if ndims(E) >= 3
    Ef = E(:,:,end);           % common: [nState x nMC x nIter] -> [nState x nMC]
elseif ismatrix(E)
    if size(E,2) == 1
        Ef = E;                % [nState x 1]
    else
        Ef = E(:,end);         % assume last col = final iter
    end
else
    Ef = E(:);
end

if isvector(Ef)
    Ef = Ef(:);                % [nState x 1]
end
end


function [nIterAgg, nIterStd] = get_iteration_count(est, agg)
% Returns:
%   nIterAgg: aggregated last-dimension size
%   nIterStd: std across entries (only meaningful if cell with varying sizes)

if ~iscell(est)
    nIterAgg = size(est, ndims(est));  % e.g. [nState x nSensor x nIter] -> nIter
    nIterStd = 0;
    return;
end

% cell case: collect last-dim sizes for all non-empty entries
iters = [];
for ii = 1:numel(est)
    E = est{ii};
    if isempty(E), continue; end
    iters(end+1) = E; %#ok<AGROW>
end

if isempty(iters)
    nIterAgg = NaN;
    nIterStd = NaN;
    return;
end

nIterStd = std(double(iters), 0);

switch agg
    case "max"
        nIterAgg = max(iters);
    case "min"
        nIterAgg = min(iters);
    case "mean"
        nIterAgg = mean(iters);
    case "median"
        nIterAgg = median(iters);
    otherwise
        nIterAgg = max(iters);
end
end


clc;clear;
% groupA = ["./data_log/rMSE_SNR/MAP_50db_tracking/Log.mat",...
%           "./data_log/rMSE_SNR/MAP_30db2_tracking/Log.mat", ...
%           "./data_log/rMSE_SNR/MAP_15db_tracking/Log.mat", ...
%           "./data_log/rMSE_SNR/MAP_5db_tracking/Log.mat"];
% 
% groupB = ["./data_log/rMSE_SNR/MAP_50db_tracking/Log.mat",...
%           "./data_log/rMSE_SNR/MLE_30db2_tracking/Log.mat", ...
%           "./data_log/rMSE_SNR/MLE_15db_tracking/Log.mat", ...
%           "./data_log/rMSE_SNR/MLE_5db_tracking/Log.mat"];

% groupA = ["./data_log/localization/MLE_5db/Log.mat","./data_log/localization/MLE_10db/Log.mat","./data_log/localization/MLE_20db/Log.mat","./data_log/localization/MLE_30db/Log.mat","./data_log/localization/MLE_50db/Log.mat"];
% groupB = ["./data_log/localization/MAP_5db/Log.mat","./data_log/localization/MAP_10db/Log.mat","./data_log/localization/MAP_20db/Log.mat","./data_log/localization/MAP_30db/Log.mat","./data_log/localization/MAP_50db/Log.mat"]; % FIM + N*Q_inv
% groupC = ["./data_log/localization/MAP_5db_2/Log.mat","./data_log/localization/MAP_10db_2/Log.mat","./data_log/localization/MAP_20db_2/Log.mat","./data_log/localization/MAP_30db_2/Log.mat","./data_log/localization/MAP_50db_2/Log.mat"]; % FIM + N*Q_inv

groupA = ["./final_data_log/localization/MC20/MLE_5db/Log.mat","./final_data_log/localization/MC20/MLE_10db/Log.mat","./final_data_log/localization/MC20/MLE_20db/Log.mat","./final_data_log/localization/MC20/MLE_30db/Log.mat","./final_data_log/localization/MC20/MLE_40db/Log.mat","./final_data_log/localization/MC20/MLE_50db/Log.mat"];
groupB = ["./final_data_log/localization/MC20/MAP_5db/Log.mat","./final_data_log/localization/MC20/MAP_10db/Log.mat","./final_data_log/localization/MC20/MAP_20db/Log.mat","./final_data_log/localization/MC20/MAP_30db/Log.mat","./final_data_log/localization/MC20/MAP_40db/Log.mat","./final_data_log/localization/MC20/MAP_50db/Log.mat"];
% groupC = ["../ADMM-EKF/data_log/MC10/kf_10db/Log.mat", "../ADMM-EKF/data_log/MC10/kf_20db/Log.mat","../ADMM-EKF/data_log/MC10/kf_30db/Log.mat","../ADMM-EKF/data_log/MC10/kf_40db/Log.mat","../ADMM-EKF/data_log/MC10/kf_50db/Log.mat"];
groupC = ["../ADMM-EKF/data_log/MC2/kf_5db/Log.mat","../ADMM-EKF/data_log/MC2/kf_10db/Log.mat", "../ADMM-EKF/data_log/MC2/kf_20db/Log.mat","../ADMM-EKF/data_log/MC2/kf_30db/Log.mat","../ADMM-EKF/data_log/MC2/kf_40db/Log.mat","../ADMM-EKF/data_log/MC2/kf_50db/Log.mat"];

% groupD = % DEKF


groups(1).name  = "MLE";
groups(1).files = groupA;

groups(2).name  = "MAP";
groups(2).files = groupB;

groups(3).name  = "EKF";
groups(3).files = groupC;




results = plot_rmse_vs_snr_multi(groups, 'SemilogY', true,'ShowCRLB',false);
% plot_convergence_iters_vs_snr_multi(groups);

% ACC  = FIM + Node * M * Q (smaller inv)
% Acc2 = FIM + Node * 1 * Q (larger inv)
% ACC3 = FIM only 


%%
% A = load("./data_log/localization/MAP_50db/Log.mat")
% B = load("./data_log/localization/MLE_50db/Log.mat")
% % Compare 
% sum(cell2mat(A.Log.Results.convg_iter),"all")/120
% sum(cell2mat(B.Log.Results.convg_iter),"all")/120