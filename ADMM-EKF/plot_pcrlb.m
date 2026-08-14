function results = plot_pcrlb(curves, varargin)
%PLOT_PCRLB  Plot RMSE and PCRLB versus time from one or more Log.mat files.
%
% One Log.mat corresponds to ONE method/curve color:
%   - solid line  : rMSE(t)
%   - dashed line : CRLB(t)
%   (same color for the same Log.mat)
%
% Input "curves" (recommended):
%   curves(i).name : e.g. "MAP", "MLE"
%   curves(i).file : path to Log.mat
%
% Also accepts: curves = string array / cellstr of file paths
% (then name defaults to the filename)
%
% Reads (either field name):
%   Log.Results.estimations_DA_raw   (or Log.Result.estimations_DA_raw)
%   Log.Results.true_params_raw      (or Log.Result.true_params_raw)
%   Log.Results.CRLB                (or Log.Result.CRLB)
%
% rMSE(t) is computed per state by averaging over:
%   - number of nodes
%   - number of Monte-Carlo samples (columns inside each cell)
%
% CRLB(t) per state is computed as:
%   sqrt( mean_over_nodes( diag(CRLB_cell)(state) ) )
%
% Options (Name,Value):
%   'StateIdx'    : default [1 2 3 4] (x,y,vx,vy), MUST be length 4 for 4x1 plot
%   'StateNames'  : default {'x','y','v_x','v_y'}
%   'SemilogY'    : false (default) / true
%   'UseRaw'      : true (default) use *_raw fields; false uses estimations_DA / true_params (if you have time axis there)
%   'TimeDim'     : 'auto' (default) | 'cols' | 'rows'
%                  For cell arrays:
%                    - 'cols' means size = [Nnode x T]
%                    - 'rows' means size = [T x Nnode]
%   'UseTimeStep' : true (default) use Log.constant.time_step if present, else index
%
% Output results(i):
%   .name
%   .file
%   .snr_dB          (if available)
%   .t               [1 x T]
%   .rmse_state      [4 x T]
%   .crlb_state      [4 x T]

p = inputParser;
p.addRequired('curves');
p.addParameter('StateIdx', [1 2 3 4], @(v) isnumeric(v) && isvector(v) && numel(v)==4);
p.addParameter('StateNames', {'$x$','$y$','$v_x$','$v_y$'}, @(c) iscell(c) && numel(c)==4);
p.addParameter('SemilogY', false, @(b) islogical(b) && isscalar(b));
p.addParameter('UseRaw', true, @(b) islogical(b) && isscalar(b));
p.addParameter('TimeDim', 'auto', @(s) any(strcmpi(string(s), ["auto","cols","rows"])));
p.addParameter('UseTimeStep', true, @(b) islogical(b) && isscalar(b));
p.parse(curves, varargin{:});

stateIdx   = p.Results.StateIdx(:);
stateNames = p.Results.StateNames;
useSemi    = p.Results.SemilogY;
useRaw     = p.Results.UseRaw;
timeDim    = lower(string(p.Results.TimeDim));
useDt      = p.Results.UseTimeStep;

% --- normalize curves input ---
curves = normalize_curves_input(curves);

% --- compute for each file ---
results = struct('name', [], 'file', [], 'snr_dB', [], 't', [], ...
    'rmse_state', [], 'crlb_state', [], 'std_state', []);
for i = 1:numel(curves)
    S = load(curves(i).file);

    if isfield(S,'Log')
        L = S.Log;
    elseif isfield(S,'log')
        L = S.log;
    else
        error('File "%s" does not contain variable "Log" or "log".', curves(i).file);
    end

    R = get_results_struct(L);

    % select fields
    if useRaw
        estField  = pick_field(R, ["estimations_DA_raw", "estimation_DA_raw"]);
        truField  = pick_field(R, "true_params_raw");
        crlbField = pick_field(R, "CRLB");
    else
        estField  = pick_field(R, "estimations_DA");
        truField  = pick_field(R, "true_params");
        crlbField = pick_field(R, "CRLB");
    end

    est  = R.(estField);
    tru  = R.(truField);
    CRLB = R.(crlbField);

    [tvec, rmse_state, crlb_state,std_state] = compute_rmse_crlb_vs_time( ...
        L, est, tru, CRLB, stateIdx, timeDim, useDt);

    results(i).name = curves(i).name;
    results(i).file = curves(i).file;
    results(i).snr_dB = try_get_snr_db(L, curves(i).file);
    results(i).t = tvec;
    results(i).rmse_state = rmse_state;
    results(i).std_state = std_state;
    results(i).crlb_state = crlb_state;
end

% --- plot (4x1) ---
figure('Color','w');
tl = tiledlayout(4,1,'TileSpacing','compact','Padding','compact');
% set(gca,'FontSize',30);
hLegend = gobjects(0);
leg = strings(0);

for s = 1:4
    ax = nexttile(tl, s);
    hold(ax,'on'); grid(ax,'on'); box(ax,'on');
    set(gca, 'FontName', 'Times New Roman');
    for i = 1:numel(results)
        t = results(i).t(:);
        y_rmse = results(i).rmse_state(s,:).';
        y_crlb = results(i).crlb_state(s,:).';

        if useSemi
            % y_rmse = max(y_rmse, realmin('double'));
            % noise = (1e-3)*(1*rand(size(y_std)));
            % if results(i).snr_dB == 10
            %     noise = 6e-4;
            % elseif results(i).snr_dB == 20
            %     noise = 5e-4;
            % else
            %     noise = 1e-4;
            % end
            y_rmse = max(y_rmse, realmin('double'));
            y_crlb = max(y_crlb, realmin('double'));
            h1 = semilogy(ax, t, y_rmse, '-s', 'LineWidth', 1.5);
            h1.MarkerSize = 15;
            h1.MarkerIndices = 1:5:length(t);
        else
            h1 = plot(ax, t, y_rmse, '-', 'LineWidth', 2);
        end

        % CRLB same color, dashed
        if useSemi
            h2 = semilogy(ax, t, y_crlb, '--*', 'LineWidth', 1.5);
            h2.MarkerSize = 10;
        else
            h2 = plot(ax, t, y_crlb, '--', 'LineWidth', 2);
        end
        h2.Color = h1.Color;

        h2.MarkerIndices = 1:5:length(t);

        % legend only from first subplot
        if s == 1
            labelBase = results(i).name;
            hLegend(end+1) = h1; %#ok<AGROW>
            leg(end+1) = labelBase + " D-EKF"; %#ok<AGROW>
            hLegend(end+1) = h2; %#ok<AGROW>
            leg(end+1) = labelBase + " PCRLB"; %#ok<AGROW>
        end
    end

    % ylabel(ax, sprintf('RMSE %s', stateNames{s}),'Interpreter','latex');
    ylabel(ax, sprintf('RMSE %s', stateNames{s}),'Interpreter','latex');
    ax.FontSize = 25;
    if s == 1
        % title(ax, 'rMSE and CRLB vs time');
        ldg = legend(ax, hLegend, leg, 'Location','best','FontSize',15);
        ldg.NumColumns = 2;
    end

    if s ~= 4
        set(ax,'XTickLabel', []);
    else
        xlabel(ax,'time (s)');
    end
    % xticks([5 10 20 30 40 50]);
    % xlim([0 t(size(t,1))]);
    % xlim([0.5 t(size(t,1))]);
    grid on;
end

end

% ========================= helpers =========================
function curves = normalize_curves_input(curves)
% Accept:
%   - struct array with fields name,file
%   - string array / cellstr of files
if isstruct(curves)
    if ~all(isfield(curves, {'name','file'}))
        error('If curves is struct, it must have fields: name, file.');
    end
    for i=1:numel(curves)
        curves(i).name = string(curves(i).name);
        curves(i).file = string(curves(i).file);
    end
    return;
end

if ischar(curves) || (isstring(curves) && isscalar(curves))
    curves = string(curves);
end
if iscell(curves)
    curves = string(curves);
end
curves = curves(:);

tmp = struct('name', [], 'file', []);
tmp = repmat(tmp, numel(curves), 1);
for i = 1:numel(curves)
    tmp(i).file = curves(i);
    [~, nm, ~] = fileparts(curves(i));
    tmp(i).name = string(nm);
end
curves = tmp;
end

function R = get_results_struct(L)
if isfield(L,'Results'), R = L.Results; return; end
if isfield(L,'Result'),  R = L.Result;  return; end
error('Cannot find Log.Results or Log.Result.');
end

function field = pick_field(S, candidates)
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
    if isfield(L,'constant')
        C = L.constant;
        if isfield(C,'SNR_idx') && ~isempty(C.SNR_idx)
            snr_db = double(C.SNR_idx); return;
        end
        if isfield(C,'SNR_dB') && ~isempty(C.SNR_dB)
            snr_db = double(C.SNR_dB); return;
        end
        if isfield(C,'SNR_lin') && ~isempty(C.SNR_lin)
            snr_db = 10*log10(double(C.SNR_lin)); return;
        end
    end
    tok = regexp(string(filepath), '(\d+)\s*db', 'tokens', 'once');
    if ~isempty(tok), snr_db = str2double(tok{1}); end
catch
    snr_db = NaN;
end
end

function [tvec, rmse_state, crlb_state, std_state] = ...
    compute_rmse_crlb_vs_time(L, est, tru, CRLB, stateIdx, timeDim, useDt)
% est, tru, CRLB are expected to be cell arrays sized [Nnode x T] or [T x Nnode]
if ~iscell(est) || ~iscell(tru) || ~iscell(CRLB)
    error('For time curves, estimations/true/CRLB must be cell arrays (nodes x time or time x nodes).');
end

[nA, nB] = size(est);
switch timeDim
    case "cols"  % [N x T]
        N = nA; T = nB;
        getE = @(n,t) est{n,t};
        getX = @(n,t) tru{n,t};
        getC = @(n,t) CRLB{n,t};
    case "rows"  % [T x N]
        T = nA; N = nB;
        getE = @(n,t) est{t,n};
        getX = @(n,t) tru{t,n};
        getC = @(n,t) CRLB{t,n};
    otherwise % auto
        if nB > nA
            % typical: nodes << time => [N x T]
            N = nA; T = nB;
            getE = @(n,t) est{n,t};
            getX = @(n,t) tru{n,t};
            getC = @(n,t) CRLB{n,t};
        else
            % fallback: [T x N]
            T = nA; N = nB;
            getE = @(n,t) est{t,n};
            getX = @(n,t) tru{t,n};
            getC = @(n,t) CRLB{t,n};
        end
end

ns = numel(stateIdx);
rmse_state = nan(ns, T);
std_state = nan(ns, T);
crlb_state = nan(ns, T);

for t = 1:T % Num of Tracked time
    sumSq = zeros(ns,1);
    cnt   = 0;

    sumDiag = zeros(ns,1);
    cntC    = 0;
    absErrSamples = [];
    
    for n = 1:N % Num pf MC
        E = getE(n,t);
        X = getX(n,t);
        C = getC(n,t);

        if isempty(E) || isempty(X)
            continue;
        end

        % --- estimates: allow [nState x nMC] or [nState x nMC x nIter]
        Ef = extract_final_est(E);                % [nState x nSample]
        if size(Ef,1) < max(stateIdx), continue; end
        Ef = Ef(stateIdx, :);                   % [ns x nSample]

        % --- truth: usually [nState x 1]
        X = X(:);
        if numel(X) < max(stateIdx), continue; end
        xtrue = X(stateIdx);                    % [ns x 1]

        err = Ef - xtrue;                       % broadcast
        sumSq = sumSq + sum(err.^2, 2);
        absErrSamples = [absErrSamples, abs(err)]; %#ok<AGROW>
        cnt   = cnt + size(err,2);

        % --- CRLB: [nState x nState]
        if ~isempty(C) && ...
                all(size(C) >= [max(stateIdx), max(stateIdx)])
            C = C(stateIdx, stateIdx);
            sumDiag = sumDiag + diag(C);
            cntC = cntC + 1;
        end
    end

    if cnt > 0
        rmse_state(:,t) = sqrt(sumSq ./ cnt);
        std_state(:,t) = std(absErrSamples, 0, 2, 'omitnan');
    end
    if cntC > 0
        crlb_state(:,t) = sqrt(sumDiag ./ cntC);
    end
end

% --- time vector ---
dt = 1;
if useDt && isfield(L,'constant') && isfield(L.constant,'time_step') && ~isempty(L.constant.time_step)
    dt = double(L.constant.time_step);
end
tvec = (0:(T-1)) * dt;
end

function Ef = extract_final_est(E)
% Convert E into [nState x nSample] at FINAL optimization iteration (if exists).
% Supports:
%   [1 x nState] / [nState x 1] -> column state vector
%   [nState x nSample]           -> as-is
%   [nSample x nState]           -> transpose
%   [nState x nSample x nIter]   -> take (:,:,end)
if ~ismatrix(E)
    Ef = E(:,:,end);
else
    Ef = E;
end

if isvector(Ef)
    Ef = Ef(:);
elseif size(Ef,1) < 4 && size(Ef,2) >= 4
    Ef = Ef.';
end
end

% Example:
% curves(1).name = "30 dB";
% curves(1).file = "./data_log/verify_CRLB4/dkf_30db_n10_3/Log.mat";
% results = plot_pcrlb(curves, 'UseRaw', true, ...
%     'SemilogY', true, 'TimeDim', 'auto');
