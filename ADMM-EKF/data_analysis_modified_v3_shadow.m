%% data_analysis.m
% Plot RMSE vs time for multiple Log.mat files with different SNR values.
% Put this file in the project root folder, then run:
%
%   data_analysis
%
% The script will automatically search for:
%   ./data_log/**/Log*.mat
% and, if nothing is found there, it will search from the current folder.
%
% Expected fields:
%   Log.Results.estimations_DA_raw
%   Log.Results.true_params_raw
%   Log.Results.CRLB
%
% For your uploaded examples:
%   Log.mat    -> SNR_idx = 30 dB
%   Log(1).mat -> SNR_idx = 50 dB
%
% Plot meaning:
%   solid line : RMSE(t)
%   shadow     : RMSE(t) +/- 3 * std(abs(error samples))

clc; clear; close all;

%% ---------------- user settings ----------------
rootDir = fileparts(mfilename('fullpath'));
if isempty(rootDir)
    rootDir = pwd;
end

% Recommended: keep your logs under folders such as:
%   data_log/kf_10db/Log.mat
%   data_log/kf_20db/Log.mat
%   data_log/kf_30db/Log.mat
%   data_log/kf_50db/Log.mat
%
% Leave this empty for automatic search.
manualLogFiles = strings(0,1);

% Plot settings similar to the attached figure
plotOpt.UseRaw              = true;
plotOpt.StateIdx            = [1 2 3 4];
plotOpt.StateNames          = {'x','y','v_x','v_y'};
plotOpt.SemilogY            = true;     % attached figure is linear y-axis
plotOpt.UseDistributedNodes = false;     % false: estimations_DA_raw, true: estimations_DA_nodes_raw
plotOpt.UseTimeStep         = true;
plotOpt.TimeScale           = 1.0;       % change to 4 if you want 16 samples to span about 0.6 s
plotOpt.YLim                = [1e-5 10];     % set [] for automatic y limits
plotOpt.XLim                = [];        % set [] for automatic x limits
plotOpt.LineWidth           = 2.0;
plotOpt.FontSize            = 18;
plotOpt.FontName            = 'Times New Roman';
plotOpt.SaveFigure          = true;
plotOpt.OutputFigureName    = 'rmse_vs_time_snr_3std_shadow.png';
plotOpt.ShowStdShadow       = true;
plotOpt.StdMultiplier       = 3;
plotOpt.ShadowAlpha         = 0.15;

%% ---------------- collect logs ----------------
if ~isempty(manualLogFiles)
    logFiles = manualLogFiles(:);
else
    logFiles = find_log_files(rootDir);
end

if isempty(logFiles)
    error(['No Log*.mat files found. Put the logs under data_log/**/Log.mat, ', ...
           'or set manualLogFiles at the top of this script.']);
end

curves = make_curves_from_logs(logFiles);
fprintf('Found %d Log file(s):\n', numel(curves));
for i = 1:numel(curves)
    fprintf('  %s -> %s\n', curves(i).name, curves(i).file);
end

%% ---------------- plot ----------------
results = plot_rmse_snr_vs_time(curves, plotOpt);


%% ========================================================================
%                               functions
% ========================================================================

function logFiles = find_log_files(rootDir)
% Prefer ./data_log first. If absent, search from rootDir.

    candidates = strings(0,1);

    dataLogDir = fullfile(rootDir, 'data_log/MC10');
    if isfolder(dataLogDir)
        d = dir(fullfile(dataLogDir, '**', 'Log*.mat'));
        candidates = [candidates; fullfile(string({d.folder})', string({d.name})')];
    end

    if isempty(candidates)
        d = dir(fullfile(rootDir, '**', 'Log*.mat'));
        candidates = [candidates; fullfile(string({d.folder})', string({d.name})')];
    end

    % Remove duplicates and files inside hidden/cache folders
    candidates = unique(candidates, 'stable');
    keep = true(size(candidates));
    for i = 1:numel(candidates)
        f = candidates(i);
        keep(i) = isfile(f) ...
            && ~contains(f, [filesep '.git' filesep]) ...
            && ~contains(f, [filesep '__MACOSX' filesep]);
    end

    logFiles = candidates(keep);
end

function opt = set_default_options(opt)
% Fill missing plot options. This ƒkeeps the code compatible with older MATLAB
% releases and allows calling plot_rmse_snr_vs_time(curves, plotOpt).

    if ~isfield(opt, 'UseRaw'),              opt.UseRaw = true; end
    if ~isfield(opt, 'StateIdx'),            opt.StateIdx = [1 2 3 4]; end
    if ~isfield(opt, 'StateNames'),          opt.StateNames = {'x','y','v_x','v_y'}; end
    if ~isfield(opt, 'SemilogY'),            opt.SemilogY = false; end
    if ~isfield(opt, 'UseDistributedNodes'), opt.UseDistributedNodes = false; end
    if ~isfield(opt, 'UseTimeStep'),         opt.UseTimeStep = true; end
    if ~isfield(opt, 'TimeScale'),           opt.TimeScale = 1.0; end
    if ~isfield(opt, 'YLim'),                opt.YLim = []; end
    if ~isfield(opt, 'XLim'),                opt.XLim = []; end
    if ~isfield(opt, 'LineWidth'),           opt.LineWidth = 2.0; end
    if ~isfield(opt, 'FontSize'),            opt.FontSize = 18; end
    if ~isfield(opt, 'FontName'),            opt.FontName = 'Times New Roman'; end
    if ~isfield(opt, 'SaveFigure'),          opt.SaveFigure = false; end
    if ~isfield(opt, 'OutputFigureName'),    opt.OutputFigureName = 'rmse_vs_time_snr_3std_shadow.png'; end
    if ~isfield(opt, 'ShowStdShadow'),       opt.ShowStdShadow = true; end
    if ~isfield(opt, 'StdMultiplier'),       opt.StdMultiplier = 3; end
    if ~isfield(opt, 'ShadowAlpha'),         opt.ShadowAlpha = 0.15; end
end

function curves = make_curves_from_logs(logFiles)
% Build curve list, read SNR from each Log.mat, sort by SNR.

    curves = struct('name', {}, 'file', {}, 'snr_dB', {});
    for i = 1:numel(logFiles)
        f = string(logFiles(i));

        try
            S = load(f, 'Log');
            if isfield(S, 'Log')
                L = S.Log;
            else
                S = load(f);
                if isfield(S, 'log')
                    L = S.log;
                else
                    warning('Skipping %s: no Log/log variable.', f);
                    continue;
                end
            end
        catch ME
            warning('Skipping %s: %s', f, ME.message);
            continue;
        end

        snr = try_get_snr_db(L, f);
        if isnan(snr)
            [~, nm, ~] = fileparts(f);
            curveName = string(nm);
        else
            curveName = sprintf('%gdb', snr);
        end

        curves(end+1).name = string(curveName); %#ok<AGROW>
        curves(end).file    = f;
        curves(end).snr_dB  = snr;
    end

    if isempty(curves)
        error('No valid Log/log variables were found in the selected files.');
    end

    % Sort by SNR if available
    snrs = [curves.snr_dB];
    if any(~isnan(snrs))
        [~, idx] = sort(snrs);
        curves = curves(idx);
    end
end

function results = plot_rmse_snr_vs_time(curves, opt)
% Plot four state RMSE curves:
%   x, y, v_x, v_y
% Each curve is one SNR.
%
% NOTE:
% Do NOT use a MATLAB "arguments" block here because this function is called
% as plot_rmse_snr_vs_time(curves, plotOpt), where plotOpt is a normal struct.

    if nargin < 2 || isempty(opt)
        opt = struct();
    end

    opt = set_default_options(opt);

    results = struct('name', {}, 'file', {}, 'snr_dB', {}, ...
                     't', {}, 'rmse_state', {}, 'rmse_std_state', {}, 'crlb_state', {});

    for i = 1:numel(curves)
        S = load(curves(i).file);

        if isfield(S, 'Log')
            L = S.Log;
        elseif isfield(S, 'log')
            L = S.log;
        else
            error('File "%s" does not contain variable Log or log.', curves(i).file);
        end

        R = get_results_struct(L);

        if opt.UseRaw
            if opt.UseDistributedNodes
                estField = pick_field(R, ["estimations_DA_nodes_raw"]);
            else
                estField = pick_field(R, ["estimations_DA_raw", "estimation_DA_raw"]);
            end
            truField  = pick_field(R, ["true_params_raw"]);
            crlbField = pick_field(R, ["CRLB"]);
        else
            estField  = pick_field(R, ["estimations_DA"]);
            truField  = pick_field(R, ["true_params"]);
            crlbField = pick_field(R, ["CRLB"]);
        end

        [tvec, rmse_state, rmse_std_state, crlb_state] = compute_rmse_crlb_vs_time( ...
            L, R.(estField), R.(truField), R.(crlbField), ...
            opt.StateIdx, opt.UseTimeStep, opt.TimeScale);

        results(i).name       = curves(i).name;
        results(i).file       = curves(i).file;
        results(i).snr_dB     = curves(i).snr_dB;
        results(i).t          = tvec;
        results(i).rmse_state     = rmse_state;
        results(i).rmse_std_state = rmse_std_state;
        results(i).crlb_state     = crlb_state;
    end

    % Figure shape similar to attached figure
    fig = figure('Color', 'w', 'Units', 'pixels', 'Position', [100 80 850 1000]);
    tl = tiledlayout(fig, 4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    hLegend = gobjects(0);
    leg = strings(0);

    for s = 1:4
        ax = nexttile(tl, s);
        hold(ax, 'on');
        grid(ax, 'on');
        box(ax, 'on');
        set(ax, 'FontName', opt.FontName, 'FontSize', opt.FontSize);

        for i = 1:numel(results)
            t = results(i).t(:);
            y = results(i).rmse_state(s, :).';
            yStd = results(i).rmse_std_state(s, :).';

            % Avoid complex round-off and invalid values
            y = real(y);
            yStd = real(yStd);
            y(~isfinite(y)) = NaN;
            yStd(~isfinite(yStd)) = NaN;

            % First plot the solid mean/RMSE curve so MATLAB assigns the color.
            if opt.SemilogY
                y(y <= 0) = NaN;
                h = semilogy(ax, t, y, '-', 'LineWidth', opt.LineWidth);
            else
                h = plot(ax, t, y, '-', 'LineWidth', opt.LineWidth);
            end

            % Then add mean ± 3 std shadow using the same color.
            if opt.ShowStdShadow
                if opt.SemilogY
                    % Log-scale-safe error shadow
                    band = opt.StdMultiplier * yStd;
                    yLow  = max(y - band, realmin('double'));
                    yHigh = max(y + band, realmin('double'));
                else
                    % Linear-scale error shadow
                    band = opt.StdMultiplier * yStd;
                    yLow  = max(y - band, 0);
                    yHigh = y + band;
                end

                valid = isfinite(t) & isfinite(yLow) & isfinite(yHigh);
                if any(valid)
                    hp = patch(ax, ...
                        [t(valid); flipud(t(valid))], ...
                        [yLow(valid); flipud(yHigh(valid))], ...
                        h.Color, ...
                        'FaceAlpha', opt.ShadowAlpha, ...
                        'EdgeColor', 'none', ...
                        'HandleVisibility', 'off');
                    uistack(hp, 'bottom');
                end
            end

            if s == 1
                hLegend(end+1) = h; %#ok<AGROW>
                leg(end+1) = string(results(i).name); %#ok<AGROW>
            end
        end

        ylabel(ax, sprintf('RMSE %s', opt.StateNames{s}), ...
               'FontName', opt.FontName, 'FontSize', opt.FontSize + 2);
        
        if opt.SemilogY
            set(ax, 'YScale', 'log');
        end
        if ~isempty(opt.YLim)
            ylim(ax, opt.YLim);
        end

        if ~isempty(opt.XLim)
            xlim(ax, opt.XLim);
        else
            allT = cell2mat(arrayfun(@(r) r.t(:), results, 'UniformOutput', false));
            allT = allT(isfinite(allT));
            
            if isempty(allT)
                xlim(ax, [0 1]);
            elseif min(allT) == max(allT)
                xlim(ax, [allT(1)-0.5, allT(1)+0.5]);
            else
                xlim(ax, [min(allT), max(allT)]);
            end
        end

        if s < 4
            set(ax, 'XTickLabel', []);
        else
            xlabel(ax, 'time (s)', 'FontName', opt.FontName, 'FontSize', opt.FontSize + 2);
        end

        if s == 1
            lgd = legend(ax, hLegend, leg, 'Location', 'northeast', ...
                         'FontSize', opt.FontSize - 3, 'Box', 'on');
            lgd.NumColumns = 2;
        end

        % set(gca, 'YScale','log')
        % xlim(opt.XLim)
    end

    if opt.SaveFigure
        exportgraphics(fig, opt.OutputFigureName, 'Resolution', 300);
        fprintf('Saved figure: %s\n', opt.OutputFigureName);
    end
end

function R = get_results_struct(L)
    if isfield(L, 'Results')
        R = L.Results;
    elseif isfield(L, 'Result')
        R = L.Result;
    else
        error('Cannot find Log.Results or Log.Result.');
    end
end

function field = pick_field(S, candidates)
    field = "";
    for k = 1:numel(candidates)
        c = string(candidates(k));
        if isfield(S, c)
            field = c;
            return;
        end
    end
    error('Missing required field. Tried: %s', strjoin(string(candidates), ", "));
end

function snr_db = try_get_snr_db(L, filepath)
    snr_db = NaN;

    try
        if isfield(L, 'constant')
            C = L.constant;

            if isfield(C, 'SNR_idx') && ~isempty(C.SNR_idx)
                snr_db = double(C.SNR_idx);
                return;
            end

            if isfield(C, 'SNR_dB') && ~isempty(C.SNR_dB)
                snr_db = double(C.SNR_dB);
                return;
            end

            if isfield(C, 'SNR_lin') && ~isempty(C.SNR_lin)
                snr_db = 10 * log10(double(C.SNR_lin));
                return;
            end
        end

        tok = regexp(lower(string(filepath)), '(\d+)\s*db', 'tokens', 'once');
        if ~isempty(tok)
            snr_db = str2double(tok{1});
        end
    catch
        snr_db = NaN;
    end
end

function [tvec, rmse_state, rmse_std_state, crlb_state] = compute_rmse_crlb_vs_time( ...
    L, est, tru, CRLB, stateIdx, useDt, timeScale)
% Supports cell arrays sized:
%   [Nmc x T]
%   [Nnode x T]
%   [T x Nmc]
%
% Each estimate cell can be:
%   [nState x 1]             final estimate
%   [nState x nNode]         node estimates
%   [nState x nNode x nIter] node history; final iteration is used
%
% RMSE(t,state) is computed over all available MC/node samples.

    if ~iscell(est) || ~iscell(tru)
        error('estimations and true_params must be cell arrays.');
    end

    [nA, nB] = size(est);

    % In your logs, raw fields are [mc x time].
    % In older logs, fields may be [node x time].
    % Usually time dimension is the larger second dimension.
    if nB >= nA
        T = nB;
        N = nA;
        getE = @(n,t) est{n,t};
        getX = @(n,t) tru{n,t};
        getC = @(n,t) CRLB{n,t};
    else
        T = nA;
        N = nB;
        getE = @(n,t) est{t,n};
        getX = @(n,t) tru{t,n};
        getC = @(n,t) CRLB{t,n};
    end

    ns = numel(stateIdx);
    rmse_state = nan(ns, T);
    rmse_std_state = nan(ns, T);
    crlb_state = nan(ns, T);

    for t = 1:T
        errSamples = [];
        crlbDiagSamples = [];

        for n = 1:N
            E = getE(n,t);
            X = getX(n,t);

            if isempty(E) || isempty(X)
                continue;
            end

            Ef = extract_final_est(E);  % [nState x nSamples]
            if size(Ef,1) < max(stateIdx)
                continue;
            end
            Ef = Ef(stateIdx, :);

            xtrue = X(:);
            if numel(xtrue) < max(stateIdx)
                continue;
            end
            xtrue = xtrue(stateIdx);

            err = Ef - xtrue;           % MATLAB implicit expansion
            errSamples = [errSamples, err]; %#ok<AGROW>

            if iscell(CRLB)
                C = getC(n,t);
                if ~isempty(C) && all(size(C) >= [max(stateIdx), max(stateIdx)])
                    C = C(stateIdx, stateIdx);
                    crlbDiagSamples = [crlbDiagSamples, diag(C)]; %#ok<AGROW>
                end
            end
        end

        if ~isempty(errSamples)
            % Solid line: RMSE over all available MC/node samples.
            rmse_state(:,t) = sqrt(mean(errSamples.^2, 2, 'omitnan'));

            % Shadow: +/- 3 std of state-wise absolute error samples.
            % For one state, sqrt(error^2) = abs(error), i.e. the per-sample RMSE.
            rmse_std_state(:,t) = std(abs(errSamples), 0, 2, 'omitnan');
        end

        if ~isempty(crlbDiagSamples)
            crlb_state(:,t) = sqrt(mean(crlbDiagSamples, 2, 'omitnan'));
        end
    end

    dt = 1;
    if useDt
        if isfield(L, 'constant') && isfield(L.constant, 'time_step') && ~isempty(L.constant.time_step)
            dt = double(L.constant.time_step);
        elseif isfield(L, 'time_step') && ~isempty(L.time_step)
            dt = double(L.time_step);
        end
    end

    tvec = (0:(T-1)) * dt * timeScale;
end

function Ef = extract_final_est(E)
% Convert estimate to [nState x nSamples].
    if ndims(E) >= 3
        Ef = E(:,:,end);      % final ADMM/consensus iteration
    else
        Ef = E;
    end

    if isvector(Ef)
        Ef = Ef(:);
    end

    % If accidentally transposed, make state dimension rows.
    if size(Ef,1) < 4 && size(Ef,2) >= 4
        Ef = Ef.';
    end
end
