function [snr_dB, rmse] = plot_rmse_vs_snr(matFiles, varargin)
%PLOT_RMSE_VS_SNR  Read one/many .mat log files and plot final RMSE vs SNR.
%
% Assumptions (matches your example log.mat):
%   - Each .mat contains variable `log`
%   - SNR is stored in: log.constant.SNR_idx (dB)   (fallback: 10*log10(SNR_lin))
%   - Estimates in:     log.Results.estimations_DA
%   - Truth in:         log.Results.true_params
%
% This code supports both:
%   (A) cell array layout (example): estimations_DA{t,node} is [nState x nMC x nIter]
%   (B) numeric layout: estimations_DA is [nState x nNode x nIter]
%
% Options (Name,Value):
%   'Method'    : 'euclidean' (default) or 'component'
%                 euclidean: RMSE = sqrt(mean(||e||^2)) over samples
%                 component: RMSE = sqrt(mean(e.^2)) over all components
%   'StateIdx'  : indices of state dimensions to include (default: [])
%                 [] means use all states.
%   'SemilogY'  : true (default) / false
%
% Usage:
%   plot_rmse_vs_snr();                       % uses all *.mat in current folder
%   plot_rmse_vs_snr("logs");                 % all *.mat in folder
%   plot_rmse_vs_snr(["a.mat","b.mat"]);      % explicit list

% -------------------- parse inputs --------------------
p = inputParser;
p.addOptional('matFiles', [], @(x) isempty(x) || ischar(x) || isstring(x) || iscell(x) || isstring(x));
p.addParameter('Method',  'euclidean', @(s) any(strcmpi(s, {'euclidean','component'})));
p.addParameter('StateIdx', [], @(v) isempty(v) || (isnumeric(v) && isvector(v)));
p.addParameter('SemilogY', true, @(b) islogical(b) && isscalar(b));
p.parse(matFiles, varargin{:});
method   = lower(string(p.Results.Method));
stateIdx = p.Results.StateIdx;
useSemi  = p.Results.SemilogY;

% normalize matFiles into a cellstr list
fileList = normalize_file_list(p.Results.matFiles);

% -------------------- loop files --------------------
snr_dB = nan(numel(fileList), 1);
rmse   = nan(numel(fileList), 1);

for i = 1:numel(fileList)
    S = load(fileList{i});

    % find the log struct
    if isfield(S, 'log')
        L = S.log;
    else
        error('File "%s" does not contain variable named "log".', fileList{i});
    end

    % SNR (dB)
    snr_dB(i) = get_snr_db(L);

    % RMSE from estimations_DA vs true_params at FINAL iteration
    rmse(i) = compute_final_rmse(L, method, stateIdx);
end

% sort by SNR
[snr_dB, idx] = sort(snr_dB);
rmse = rmse(idx);

% -------------------- plot --------------------
figure('Color','w'); hold on; grid on; box on;
if useSemi
    semilogy(snr_dB, rmse, '-o', 'LineWidth', 1.5, 'MarkerSize', 7);
else
    plot(snr_dB, rmse, '-o', 'LineWidth', 1.5, 'MarkerSize', 7);
end
xlabel('SNR (dB)');
ylabel(sprintf('RMSE (%s)', method));
title('Final RMSE vs SNR');
end

% ======================================================================

function fileList = normalize_file_list(matFiles)
if nargin < 1 || isempty(matFiles)
    d = dir('*.mat');
    fileList = fullfile({d.folder}, {d.name});
    fileList = fileList(:);
    if isempty(fileList)
        error('No .mat files found in current folder.');
    end
    return;
end

if ischar(matFiles) || (isstring(matFiles) && isscalar(matFiles))
    matFiles = char(matFiles);
    if isfolder(matFiles)
        d = dir(fullfile(matFiles, '*.mat'));
        fileList = fullfile({d.folder}, {d.name});
        fileList = fileList(:);
        if isempty(fileList)
            error('No .mat files found in folder: %s', matFiles);
        end
    else
        fileList = {matFiles};
    end
    return;
end

if isstring(matFiles)
    fileList = cellstr(matFiles(:));
elseif iscell(matFiles)
    fileList = matFiles(:);
else
    error('Unsupported matFiles input type.');
end
end

function snr_db = get_snr_db(L)
snr_db = NaN;
if isfield(L, 'constant')
    C = L.constant;
    if isfield(C, 'SNR_idx') && ~isempty(C.SNR_idx)
        snr_db = double(C.SNR_idx);
        return;
    end
    if isfield(C, 'SNR_lin') && ~isempty(C.SNR_lin)
        snr_db = 10*log10(double(C.SNR_lin));
        return;
    end
end
error('Could not find SNR in log.constant.SNR_idx or log.constant.SNR_lin');
end

function r = compute_final_rmse(L, method, stateIdx)
if ~isfield(L, 'Results')
    error('log.Results missing.');
end
R = L.Results;

if ~isfield(R, 'estimations_DA') || ~isfield(R, 'true_params')
    error('log.Results.estimations_DA or log.Results.true_params missing.');
end

est = R.estimations_DA;
tru = R.true_params;

acc = 0.0;
cnt = 0.0;

if iscell(est)
    [T, N] = size(est);
    for t = 1:T
        for n = 1:N
            E = est{t,n};
            if isempty(E), continue; end

            % final estimate: make it [nState x nSample]
            Ef = final_slice_to_state_by_sample(E);

            % truth vector: [nState x 1]
            if iscell(tru)
                xtrue = tru{t,n};
            else
                % if tru is numeric but est is cell, try best effort
                xtrue = tru(:, min(n, size(tru,2)));
            end
            xtrue = xtrue(:);

            % optional state selection
            if ~isempty(stateIdx)
                Ef    = Ef(stateIdx, :);
                xtrue = xtrue(stateIdx);
            end

            err = Ef - xtrue;  % implicit expansion

            if method == "euclidean"
                % each column is one sample
                acc = acc + sum(sum(err.^2, 1));   % sum over samples of ||e||^2
                cnt = cnt + size(err, 2);
            else
                % component-wise
                acc = acc + sum(err(:).^2);
                cnt = cnt + numel(err);
            end
        end
    end

else
    % numeric layout: [nState x nNode x nIter] (or similar)
    if ndims(est) < 3
        error('Numeric estimations_DA expected to be 3D: [nState x nNode x nIter].');
    end
    Ef = est(:,:,end); % [nState x nNode]

    % truth layout could be [nState x nNode] or [nState x 1]
    if iscell(tru)
        error('true_params is cell but estimations_DA is numeric; please adapt mapping.');
    end
    if isvector(tru)
        Xtrue = tru(:);                 % [nState x 1]
        err = Ef - Xtrue;               % expands to [nState x nNode]
    else
        Xtrue = tru;                    % [nState x nNode]
        err = Ef - Xtrue;
    end

    if ~isempty(stateIdx)
        err = err(stateIdx, :);
    end

    if method == "euclidean"
        acc = sum(sum(err.^2, 1));
        cnt = size(err, 2);
    else
        acc = sum(err(:).^2);
        cnt = numel(err);
    end
end

r = sqrt(acc / max(cnt, 1));
end

function Ef = final_slice_to_state_by_sample(E)
% Convert E (various shapes) -> Ef = [nState x nSample] at FINAL iteration.
sz = size(E);

if ndims(E) >= 3
    % most common: [nState x nMC x nIter]
    Ef = E(:,:,end);
elseif ismatrix(E)
    % could be [nState x nIter] or [nState x nMC] or [nState x 1]
    if sz(2) == 1
        Ef = E;                  % [nState x 1]
    else
        Ef = E(:, end);          % assume last column corresponds to final iter
        Ef = Ef(:);              % [nState x 1]
    end
else
    Ef = E(:);                   % fallback: [nState x 1]
end

% ensure 2D [nState x nSample]
if isvector(Ef)
    Ef = Ef(:);                  % [nState x 1]
end
end



function [snr_dB, rmse_state] = plot_rmse_states_subplot(matFiles, varargin)
%PLOT_RMSE_STATES_SUBPLOT  Plot 4x1 subplots of final RMSE per state vs SNR.
%
% IMPORTANT:
%   This function REUSES helper functions from the previous reply:
%     - normalize_file_list
%     - get_snr_db
%     - final_slice_to_state_by_sample
%   So place this function in the SAME .m file as your previous function
%   (below it), or otherwise make those helpers available on your MATLAB path.
%
% Expected fields:
%   log.Results.estimations_DA
%   log.Results.true_params
%   log.constant.SNR_idx   (or SNR_lin)
%
% Options (Name,Value):
%   'StateIdx'   : default [1 2 3 4]  (x, y, vx, vy)
%   'StateNames' : default {'x','y','v_x','v_y'}
%   'SemilogY'   : true (default) / false

% -------------------- parse inputs --------------------
p = inputParser;
p.addOptional('matFiles', [], @(x) isempty(x) || ischar(x) || isstring(x) || iscell(x));
p.addParameter('StateIdx', [1 2 3 4], @(v) isnumeric(v) && isvector(v) && numel(v)==4);
p.addParameter('StateNames', {'x','y','v_x','v_y'}, @(c) iscell(c) && numel(c)==4);
p.addParameter('SemilogY', true, @(b) islogical(b) && isscalar(b));
p.parse(matFiles, varargin{:});

stateIdx   = p.Results.StateIdx(:);
stateNames = p.Results.StateNames;
useSemi    = p.Results.SemilogY;

fileList = normalize_file_list(p.Results.matFiles);

% -------------------- loop files --------------------
nF = numel(fileList);
snr_dB = nan(nF, 1);
rmse_state = nan(4, nF);

for i = 1:nF
    S = load(fileList{i});
    if ~isfield(S, 'log')
        error('File "%s" does not contain variable named "log".', fileList{i});
    end
    L = S.log;

    snr_dB(i) = get_snr_db(L);

    est = L.Results.estimations_DA;
    tru = L.Results.true_params;

    acc = zeros(4,1);
    cnt = zeros(4,1);

    if iscell(est)
        [T, N] = size(est);
        for t = 1:T
            for n = 1:N
                E = est{t,n};
                if isempty(E), continue; end

                Ef = final_slice_to_state_by_sample(E);  % [nState x nSample]

                if iscell(tru)
                    xtrue = tru{t,n};
                else
                    xtrue = tru(:, min(n, size(tru,2)));
                end
                xtrue = xtrue(:);

                % select requested states (4)
                Ef_sel = Ef(stateIdx, :);
                x_sel  = xtrue(stateIdx);

                err = Ef_sel - x_sel;  % [4 x nSample]

                acc = acc + sum(err.^2, 2);
                cnt = cnt + size(err,2);
            end
        end
    else
        % numeric layout: [nState x nNode x nIter]
        if ndims(est) < 3
            error('Numeric estimations_DA expected 3D: [nState x nNode x nIter].');
        end
        Ef = est(:,:,end);  % [nState x nNode]

        if isvector(tru)
            err = Ef - tru(:);  % expands to [nState x nNode]
        else
            err = Ef - tru;
        end

        err = err(stateIdx, :); % [4 x nNode]
        acc = acc + sum(err.^2, 2);
        cnt = cnt + size(err,2);
    end

    rmse_state(:, i) = sqrt(acc ./ max(cnt,1));
end

% sort by SNR
[snr_dB, idx] = sort(snr_dB);
rmse_state = rmse_state(:, idx);

% -------------------- plot (4x1) --------------------
figure('Color','w');
for k = 1:4
    subplot(4,1,k); hold on; grid on; box on;

    if useSemi
        semilogy(snr_dB, rmse_state(k,:), '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
    else
        plot(snr_dB, rmse_state(k,:), '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
    end

    ylabel(sprintf('RMSE %s', stateNames{k}));
    if k == 1
        title('Final RMSE per state vs SNR');
    end
    if k == 4
        xlabel('SNR (dB)');
    else
        set(gca, 'XTickLabel', []);
    end
end
end

%%% Main function

matfiles = ["./data_log/rMSE_SNR/MAP_50db_tracking/log.mat","./data_log/rMSE_SNR/MAP_30db_tracking/log.mat","./data_log/rMSE_SNR/MAP_15db_tracking/log.mat","./data_log/rMSE_SNR/MAP_5db_tracking/log.mat"]
matfiles2 = ["./data_log/rMSE_SNR/MLE_50db_tracking/log.mat","./data_log/rMSE_SNR/MLE_30db_tracking/log.mat","./data_log/rMSE_SNR/MLE_15db_tracking/log.mat","./data_log/rMSE_SNR/MLE_5db_tracking/log.mat"]
% plot_rmse_vs_snr(matfiles, 'StateIdx', [], 'Method', 'euclidean');
plot_rmse_states_subplot(matfiles2);
plot_rmse_states_subplot(matfiles);