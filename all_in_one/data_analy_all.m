clc;clear;close all;
% plot_tracking_log("./data_log/tracking_MLE_mc5_same_vel/log.mat");
% plot_tracking_compare(["./data_log/tracking_MAP_mc10_2/log.mat","./data_log/tracking_MLE_mc10_2/log.mat"],"method_names", ["MAP","MLE"])
% plot_tracking_compare_mse_by_state(["./data_log/tracking_MAP_mc10/log.mat","./data_log/tracking_MLE_mc10/log.mat"]);
plot_tracking_compare_mse_centralized_vs_decentralized(["./data_log/tracking_MAP_mc10/log.mat"]);

function S = plot_tracking_log(matPath, varargin)
%PLOT_TRACKING_LOG  Read ONE .mat containing Results cell(MC,T) and plot shaded bands.
%
% Expected .mat content:
%   Results (struct) with fields (cell arrays of size MC x T):
%     Results.primal_residauls
%     Results.dual_residauls
%     Results.estimations_DA
%     Results.true_params
%     Results.convg_iter
%     Results.estimations_CA
%     Results.ADMM_setting    (not used for plotting here)
%
% Usage:
%   plot_tracking_log("log.mat");
%   plot_tracking_log("log.mat","t_pick",5);
%   plot_tracking_log("log.mat","time_s",0:0.1:2.9);  % length must equal T

%% options
p = inputParser;
p.addParameter("t_pick", 1, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter("time_s", [], @(x) isempty(x) || isnumeric(x));
p.addParameter("title_prefix", "", @(s) isstring(s) || ischar(s));
p.parse(varargin{:});
opt = p.Results;

%% load
data = load(matPath).log;
if ~isfield(data, "Results")
    error("'%s' does not contain variable 'Results'.", matPath);
end
res = data.Results;

needFields = ["primal_residauls","dual_residauls","estimations_DA","true_params","convg_iter","estimations_CA"];
for k = 1:numel(needFields)
    if ~isfield(res, needFields(k))
        error("Results missing field: %s", needFields(k));
    end
end

Pcell  = res.primal_residauls;   % cell(MC,T)
Dcell  = res.dual_residauls;     % cell(MC,T)
Ecell  = res.estimations_DA;     % cell(MC,T)
Tcell  = res.true_params;        % cell(MC,T)
Ccell  = res.convg_iter;         % cell(MC,T)
CAcell = res.estimations_CA;     % cell(MC,T)

[MC, T] = size(Tcell);

%% time axis
if ~isempty(opt.time_s)
    t = opt.time_s(:);
    if numel(t) ~= T
        error("time_s length must match TRACK_TIME. Got %d, expected %d.", numel(t), T);
    end
else
    t = (0:T-1).';
end

%% compute errors over time for each MC
errCA = nan(T, MC);
errDA = nan(T, MC);
convg = nan(T, MC);

for mc = 1:MC
    for tt = 1:T
        xtrue = Tcell{mc, tt};
        if isempty(xtrue), continue; end
        xtrue = to_state4(xtrue);

        xca = res.estimations_CA{mc, tt};
        if ~isempty(xca)
            xca = to_state4(xca);
            errCA(tt, mc) = norm(xca - xtrue, 2);
        end

        xda = Ecell{mc, tt};
        if isempty(xda), continue; end
        X = to_3d(xda); % 4 x Nnode x Niter
        if isempty(X), continue; end

        I = get_convg_iter(Ccell{mc, tt}, size(X,3));
        convg(tt, mc) = I;

        slice = X(:,:,I);
        xbar  = mean(slice, 2, "omitnan");
        errDA(tt, mc) = norm(xbar - xtrue, 2);
    end
end

%% output
S = struct();
S.matPath = string(matPath);
S.Results = res;
S.MC = MC; S.T = T; S.t = t;
S.errCA = errCA;
S.errDA = errDA;
S.convg = convg;

%% Figure 1: error over time (reference-like shaded, log-y)
figure('Name','Tracking error over time (shaded)','Color','w');
hold on; grid on;
set(gca, 'YScale','log');

plot_mean_shaded(t, errCA, [1 0 0], '-', 2.2, 0.15, 'CA');
plot_mean_shaded(t, errDA, [0 0 1], '-', 2.2, 0.15, 'DA (mean over nodes)');

xlabel('Time t [s]');
ylabel('Tracking error (L2 norm)');
ttl = "Tracking error over time";
if strlength(string(opt.title_prefix)) > 0
    ttl = string(opt.title_prefix) + " - " + ttl;
end
title(ttl, 'Interpreter','tex');
legend('Location','northeast', 'Interpreter','tex');
apply_times_new_roman(gcf, 18, 14);

%% Figure 2: convg_iter over time (shaded)
if ~all(isnan(convg(:)))
    figure('Name','ADMM convergence iteration over time (shaded)','Color','w');
    hold on; grid on;

    plot_mean_shaded(t, convg, [0 0 0], '-', 2.0, 0.12, 'convg\_iter');

    xlabel('Time t [s]');
    ylabel('ADMM convergence iteration');
    title('ADMM convergence iteration over time', 'Interpreter','tex');
    legend('Location','northeast', 'Interpreter','tex');
    apply_times_new_roman(gcf, 18, 14);
end

%% Figure 3: residual curves at chosen time index (across MC, shaded, log-y)
t_pick = round(opt.t_pick);
t_pick = max(1, min(T, t_pick));

[Pmat, Dmat, itMax] = stack_residuals_at_time(Pcell, Dcell, t_pick);

if itMax > 0
    figure('Name',sprintf('Residual curves at time index %d (shaded)', t_pick), 'Color','w');
    tiledlayout(1,2);

    nexttile; hold on; grid on; set(gca,'YScale','log');
    x = (1:itMax).';
    plot_mean_shaded(x, Pmat, [0 0 0], '-', 2.0, 0.15, 'Primal');
    xlabel('ADMM iteration'); ylabel('Primal residual');
    title(sprintf('Primal residual (t=%d)', t_pick), 'Interpreter','tex');

    nexttile; hold on; grid on; set(gca,'YScale','log');
    plot_mean_shaded(x, Dmat, [0 0 0], '-', 2.0, 0.15, 'Dual');
    xlabel('ADMM iteration'); ylabel('Dual residual');
    title(sprintf('Dual residual (t=%d)', t_pick), 'Interpreter','tex');

    apply_times_new_roman(gcf, 18, 14);
end

end

% function S = plot_tracking_compare(matFiles, varargin)
% %PLOT_TRACKING_COMPARE  Compare multiple methods from multiple .mat files in ONE figure.
% %
% % Each .mat should contain:
% %   Results (struct) with fields cell(MC,T):
% %     Results.estimations_DA, Results.true_params, Results.convg_iter
% %     Results.estimations_CA (optional if you want metric="CA")
% %
% % Usage:
% %   plot_tracking_compare(["mle.mat","mmse_1e-4.mat","mmse_1e-5.mat"]);
% %   plot_tracking_compare({"a.mat","b.mat"}, "metric","DA", "time_s", 0:0.1:2.9);
% %   plot_tracking_compare(matFiles, "method_names", ["MLE","MMSE 1e-4","MMSE 1e-5"]);
% %
% % Options:
% %   metric: "DA" (default) or "CA"
% %   time_s: time axis vector length T (optional)
% %   method_names: string array same length as matFiles (optional)
% %   alpha: shaded alpha (default 0.15)
% %   line_width: mean line width (default 2.2)
% %
% % Output S:
% %   S.methods(i).name, .t, .err (T x MC), .mu, .std
% 
% %% ---- parse inputs
% if ischar(matFiles) || isstring(matFiles)
%     matFiles = string(matFiles);
% elseif iscell(matFiles)
%     matFiles = string(matFiles);
% else
%     error("matFiles must be string array, char, or cellstr.");
% end
% matFiles = matFiles(:);
% 
% p = inputParser;
% p.addParameter("metric", "DA", @(s) any(strcmpi(string(s), ["DA","CA"])));
% p.addParameter("time_s", [], @(x) isempty(x) || isnumeric(x));
% p.addParameter("method_names", [], @(x) isempty(x) || isstring(x) || iscell(x));
% p.addParameter("alpha", 0.15, @(x) isnumeric(x) && isscalar(x));
% p.addParameter("line_width", 2.2, @(x) isnumeric(x) && isscalar(x));
% p.addParameter("title_text", "Tracking error over time", @(s) isstring(s) || ischar(s));
% p.parse(varargin{:});
% opt = p.Results;
% 
% metric = upper(string(opt.metric));
% 
% % method names
% if isempty(opt.method_names)
%     methodNames = strings(numel(matFiles),1);
%     for i = 1:numel(matFiles)
%         [~, bn, ~] = fileparts(matFiles(i));
%         methodNames(i) = string(bn);
%     end
% else
%     methodNames = string(opt.method_names);
%     methodNames = methodNames(:);
%     if numel(methodNames) ~= numel(matFiles)
%         error("method_names length must match matFiles length.");
%     end
% end
% 
% %% ---- load and compute err(t,mc) per method
% S = struct();
% S.metric = metric;
% S.files = matFiles;
% S.method_names = methodNames;
% S.methods = repmat(struct("name","","t",[],"err",[],"mu",[],"std",[]), numel(matFiles), 1);
% 
% for i = 1:numel(matFiles)
%     data = load(matFiles(i)).log;
%     if ~isfield(data, "Results")
%         error("File %s does not contain variable 'Results'.", matFiles(i));
%     end
%     R = data.Results;
% 
%     % required fields
%     if ~isfield(R,"true_params") || ~isfield(R,"estimations_DA") || ~isfield(R,"convg_iter")
%         error("Results in %s must contain true_params, estimations_DA, convg_iter.", matFiles(i));
%     end
%     if metric == "CA" && ~isfield(R,"estimations_CA")
%         error("metric='CA' requires Results.estimations_CA in %s.", matFiles(i));
%     end
% 
%     Tcell  = R.true_params;        % cell(MC,T)
%     Ecell  = R.estimations_DA;     % cell(MC,T)
%     Ccell  = R.convg_iter;         % cell(MC,T)
%     if metric == "CA"
%         CAc   = R.estimations_CA;  % cell(MC,T)
%     end
% 
%     [MC, T] = size(Tcell);
% 
%     % time axis
%     if ~isempty(opt.time_s)
%         t = opt.time_s(:);
%         if numel(t) ~= T
%             error("time_s length must equal TRACK_TIME (T=%d) for file %s.", T, matFiles(i));
%         end
%     else
%         t = (0:T-1).';
%     end
% 
%     err = nan(T, MC);
% 
%     for mc = 1:MC
%         for tt = 1:T
%             xtrue = Tcell{mc,tt};
%             if isempty(xtrue), continue; end
%             xtrue = to_state4(xtrue);
% 
%             if metric == "CA"
%                 xhat = CAc{mc,tt};
%                 if isempty(xhat), continue; end
%                 xhat = to_state4(xhat);
%                 err(tt,mc) = norm(xhat - xtrue, 2);
% 
%             else % metric == "DA"
%                 xda = Ecell{mc,tt};
%                 if isempty(xda), continue; end
%                 X = to_3d(xda); % 4 x Nnode x Niter
%                 if isempty(X), continue; end
% 
%                 I = get_convg_iter(Ccell{mc,tt}, size(X,3));
%                 slice = X(:,:,I);                 % 4 x Nnode
%                 xbar  = mean(slice, 2, "omitnan");% 4 x 1
%                 err(tt,mc) = norm(xbar - xtrue, 2);
%             end
%         end
%     end
% 
%     S.methods(i).name = methodNames(i);
%     S.methods(i).t = t;
%     S.methods(i).err = err;
%     S.methods(i).mu = mean(err, 2, "omitnan");
%     S.methods(i).std = std(err, 0, 2, "omitnan");
% end
% %% ---- plot ALL methods in ONE figure (reference-like shaded)
% figure('Name','Compare methods (shaded)','Color','w');
% hold on; grid on;
% set(gca, 'YScale','log');
% 
% for i = 1:numel(S.methods)
%     t   = S.methods(i).t;
%     err = S.methods(i).err;
% 
%     % Use MATLAB default color order by just plotting a dummy line to get color,
%     % but we want patch first -> get next color from axes colororder:
%     c = get_next_color(gca);
% 
%     plot_mean_shaded(t, err, c, '-', opt.line_width, opt.alpha, S.methods(i).name);
% end
% 
% xlabel('Time t [s]');
% ylabel(sprintf('%s tracking error (L2 norm)', metric));
% title(string(opt.title_text), 'Interpreter','tex');
% legend('Location','northeast', 'Interpreter','tex');
% 
% apply_times_new_roman(gcf, 18, 14);
% end

function S = plot_tracking_compare(matFiles, varargin)
%PLOT_TRACKING_COMPARE  Compare multiple methods from multiple .mat files and output 4 figures.
%
% Each .mat file should contain:
%   Results (struct) with fields cell(MC,T):
%     Results.primal_residauls
%     Results.dual_residauls
%     Results.estimations_DA
%     Results.true_params
%     Results.convg_iter
%     Results.estimations_CA
%
% Usage:
%   plot_tracking_compare(["mle.mat","mmse_1e-4.mat"], "method_names", ["MLE","MMSE 1e-4"]);
%   plot_tracking_compare(files, "metric","DA", "time_s", 0:0.1:2.9, "t_pick", 5);
%
% Options:
%   metric: "DA" (default) or "CA"  -> for the tracking error figure
%   method_names: legend names (default = file base names)
%   time_s: time axis vector length T (optional; else 0:T-1)
%   t_pick: time index for residual curve plots (default 1)
%   alpha: shaded alpha (default 0.15)
%   line_width: mean line width (default 2.2)
%   y_scale_err: "log" (default) or "linear" for error figure
%
% Output S: struct containing per-method computed matrices.

%% ---- inputs
if ischar(matFiles) || isstring(matFiles)
    matFiles = string(matFiles);
elseif iscell(matFiles)
    matFiles = string(matFiles);
else
    error("matFiles must be string array / cellstr / char.");
end
matFiles = matFiles(:);

p = inputParser;
p.addParameter("metric", "DA", @(s) any(strcmpi(string(s), ["DA","CA"])));
p.addParameter("time_s", [], @(x) isempty(x) || isnumeric(x));
p.addParameter("method_names", [], @(x) isempty(x) || isstring(x) || iscell(x));
p.addParameter("t_pick", 1, @(x) isnumeric(x) && isscalar(x) && x >= 1);
p.addParameter("alpha", 0.15, @(x) isnumeric(x) && isscalar(x));
p.addParameter("line_width", 2.2, @(x) isnumeric(x) && isscalar(x));
p.addParameter("y_scale_err", "log", @(s) any(strcmpi(string(s), ["log","linear"])));
p.parse(varargin{:});
opt = p.Results;

metric = upper(string(opt.metric));

% method names
if isempty(opt.method_names)
    methodNames = strings(numel(matFiles),1);
    for i = 1:numel(matFiles)
        [~, bn, ~] = fileparts(matFiles(i));
        methodNames(i) = string(bn);
    end
else
    methodNames = string(opt.method_names);
    methodNames = methodNames(:);
    if numel(methodNames) ~= numel(matFiles)
        error("method_names length must match matFiles length.");
    end
end

%% ---- load + compute per-method series (T x MC)
S = struct();
S.files = matFiles;
S.method_names = methodNames;
S.metric = metric;
S.methods = repmat(struct( ...
    "name","", ...
    "t",[], ...
    "err",[], ...         % T x MC (selected metric)
    "convg",[], ...       % T x MC
    "P_res_cell",[], ...  % MC x T (original)
    "D_res_cell",[], ...  % MC x T (original)
    "t_pick",[], ...
    "Pmat",[], ...        % itMax x MC at t_pick
    "Dmat",[] ...         % itMax x MC at t_pick
), numel(matFiles), 1);

for i = 1:numel(matFiles)
    data = load(matFiles(i)).log;
    if ~isfield(data,"Results")
        error("File %s missing variable 'Results'.", matFiles(i));
    end
    R = data.Results;

    % required fields for all plots
    need = ["primal_residauls","dual_residauls","true_params","estimations_DA","convg_iter"];
    for k = 1:numel(need)
        if ~isfield(R, need(k))
            error("Results in %s missing field: %s", matFiles(i), need(k));
        end
    end
    if metric == "CA" && ~isfield(R,"estimations_CA")
        error("metric='CA' requires Results.estimations_CA in %s.", matFiles(i));
    end

    Pcell  = R.primal_residauls;    % cell(MC,T)
    Dcell  = R.dual_residauls;      % cell(MC,T)
    Tcell  = R.true_params;         % cell(MC,T)
    Ecell  = R.estimations_DA;      % cell(MC,T)
    Ccell  = R.convg_iter;          % cell(MC,T)
    if metric == "CA"
        CAc = R.estimations_CA;     % cell(MC,T)
    end

    [MC, T] = size(Tcell);

    % time axis
    if ~isempty(opt.time_s)
        t = opt.time_s(:);
        if numel(t) ~= T
            error("time_s length must equal TRACK_TIME (T=%d) for file %s.", T, matFiles(i));
        end
    else
        t = (0:T-1).';
    end

    % compute tracking error (T x MC) + convg (T x MC)
    err   = nan(T, MC);
    convg = nan(T, MC);

    for mc = 1:MC
        for tt = 1:T
            xtrue = Tcell{mc,tt};
            if isempty(xtrue), continue; end
            xtrue = to_state4(xtrue);

            if metric == "CA"
                xhat = CAc{mc,tt};
                if isempty(xhat), continue; end
                xhat = to_state4(xhat);
                err(tt,mc) = norm(xhat - xtrue, 2);
            else
                xda = Ecell{mc,tt};
                if isempty(xda), continue; end
                X = to_3d(xda);
                if isempty(X), continue; end
                I = get_convg_iter(Ccell{mc,tt}, size(X,3));
                convg(tt,mc) = I;
                xhat = mean(X(:,:,I), 2, "omitnan");
                err(tt,mc) = norm(xhat - xtrue, 2);
            end

            % convg for CA case still useful (if stored)
            if metric == "CA"
                xda = Ecell{mc,tt};
                if ~isempty(xda)
                    X = to_3d(xda);
                    if ~isempty(X)
                        I = get_convg_iter(Ccell{mc,tt}, size(X,3));
                        convg(tt,mc) = I;
                    end
                end
            end
        end
    end

    % residual matrices at t_pick
    t_pick = max(1, min(T, round(opt.t_pick)));
    [Pmat, Dmat, itMax] = stack_residuals_at_time(Pcell, Dcell, t_pick); %#ok<ASGLU>

    S.methods(i).name = methodNames(i);
    S.methods(i).t = t;
    S.methods(i).err = err;
    S.methods(i).convg = convg;
    S.methods(i).P_res_cell = Pcell;
    S.methods(i).D_res_cell = Dcell;
    S.methods(i).t_pick = t_pick;
    S.methods(i).Pmat = Pmat;
    S.methods(i).Dmat = Dmat;
end

%% ---- Figure 1: Tracking error over time (all methods in one)
fig1 = figure('Name','Tracking error over time (compare)','Color','w');
hold on; grid on;
if strcmpi(opt.y_scale_err,"log")
    set(gca,'YScale','log');
end

co = colororder;
nC = size(co,1);

for i = 1:numel(S.methods)
    c = co(mod(i-1,nC)+1,:);
    plot_mean_shaded_ax(gca, S.methods(i).t, S.methods(i).err, c, '-', opt.line_width, opt.alpha, S.methods(i).name);
end

xlabel('Time t [s]');
ylabel(sprintf('%s tracking error (L2 norm)', metric));
title(sprintf('Tracking error over time (%s)', metric), 'Interpreter','tex');
legend('Location','northeast', 'Interpreter','tex');
apply_times_new_roman(fig1, 18, 14);

%% ---- Figure 2: convg_iter over time (all methods in one)
fig2 = figure('Name','ADMM convg_iter over time (compare)','Color','w');
hold on; grid on;

for i = 1:numel(S.methods)
    c = co(mod(i-1,nC)+1,:);
    plot_mean_shaded_ax(gca, S.methods(i).t, S.methods(i).convg, c, '-', 2.0, 0.12, S.methods(i).name);
end

xlabel('Time t [s]');
ylabel('ADMM convergence iteration');
title('ADMM convergence iteration over time', 'Interpreter','tex');
legend('Location','northeast', 'Interpreter','tex');
apply_times_new_roman(fig2, 18, 14);

%% ---- Figure 3: primal residual curves at t_pick (all methods in one)
fig3 = figure('Name',sprintf('Primal residual at t\\_pick=%d (compare)', S.methods(1).t_pick), 'Color','w');
hold on; grid on;
set(gca,'YScale','log');

for i = 1:numel(S.methods)
    Pmat = S.methods(i).Pmat;
    if isempty(Pmat), continue; end
    x = (1:size(Pmat,1)).';
    c = co(mod(i-1,nC)+1,:);
    plot_mean_shaded_ax(gca, x, Pmat, c, '-', opt.line_width, opt.alpha, S.methods(i).name);
end

xlabel('ADMM iteration');
ylabel('Primal residual');
title(sprintf('Primal residual (t\\_pick=%d)', S.methods(1).t_pick), 'Interpreter','tex');
legend('Location','northeast', 'Interpreter','tex');
apply_times_new_roman(fig3, 18, 14);

%% ---- Figure 4: dual residual curves at t_pick (all methods in one)
fig4 = figure('Name',sprintf('Dual residual at t\\_pick=%d (compare)', S.methods(1).t_pick), 'Color','w');
hold on; grid on;
set(gca,'YScale','log');

for i = 1:numel(S.methods)
    Dmat = S.methods(i).Dmat;
    if isempty(Dmat), continue; end
    x = (1:size(Dmat,1)).';
    c = co(mod(i-1,nC)+1,:);
    plot_mean_shaded_ax(gca, x, Dmat, c, '-', opt.line_width, opt.alpha, S.methods(i).name);
end

xlabel('ADMM iteration');
ylabel('Dual residual');
title(sprintf('Dual residual (t\\_pick=%d)', S.methods(1).t_pick), 'Interpreter','tex');
legend('Location','northeast', 'Interpreter','tex');
apply_times_new_roman(fig4, 18, 14);

end
function S = plot_tracking_compare_mse_by_state(matFiles, varargin)
%PLOT_TRACKING_COMPARE_MSE_BY_STATE  Compare multiple .mat methods; plot MSE per state in 2x2 tiledlayout.
%
% Each .mat should contain Results struct with cell(MC,T):
%   Results.true_params      (MC,T) each -> 4x1 or 1x4
%   Results.estimations_DA   (MC,T) each -> 4 x Nnode x Niter (or nested cell form)
%   Results.convg_iter       (MC,T) each -> scalar (or empty)
%   Results.estimations_CA   (MC,T) each -> 4x1 or 1x4 (only if metric="CA")
%
% Usage:
%   plot_tracking_compare_mse_by_state(["mle.mat","mmse_1e-4.mat"], ...
%       "method_names", ["MLE","MMSE 1e-4"], "metric","DA", "time_s", 0:0.1:2.9);

%% ---- parse inputs
if ischar(matFiles) || isstring(matFiles)
    matFiles = string(matFiles);
elseif iscell(matFiles)
    matFiles = string(matFiles);
else
    error("matFiles must be string array, char, or cellstr.");
end
matFiles = matFiles(:);

p = inputParser;
p.addParameter("metric", "DA", @(s) any(strcmpi(string(s), ["DA","CA"])));
p.addParameter("time_s", [], @(x) isempty(x) || isnumeric(x));
p.addParameter("method_names", [], @(x) isempty(x) || isstring(x) || iscell(x));
p.addParameter("alpha", 0.15, @(x) isnumeric(x) && isscalar(x));
p.addParameter("line_width", 2.2, @(x) isnumeric(x) && isscalar(x));
p.addParameter("y_scale", "log", @(s) any(strcmpi(string(s), ["log","linear"])));
p.parse(varargin{:});
opt = p.Results;

metric = upper(string(opt.metric));
stateNames = ["x","y","v_x","v_y"];

% method names
if isempty(opt.method_names)
    methodNames = strings(numel(matFiles),1);
    for i = 1:numel(matFiles)
        [~, bn, ~] = fileparts(matFiles(i));
        methodNames(i) = string(bn);
    end
else
    methodNames = string(opt.method_names);
    methodNames = methodNames(:);
    if numel(methodNames) ~= numel(matFiles)
        error("method_names length must match matFiles length.");
    end
end

%% ---- load and compute squared error per state: SE(state, t, mc)
S = struct();
S.metric = metric;
S.files = matFiles;
S.method_names = methodNames;
S.methods = repmat(struct("name","","t",[],"SE",[]), numel(matFiles), 1);

for i = 1:numel(matFiles)
    data = load(matFiles(i)).log;
    if ~isfield(data, "Results")
        error("File %s does not contain variable 'Results'.", matFiles(i));
    end
    R = data.Results;

    if ~isfield(R,"true_params") || ~isfield(R,"estimations_DA") || ~isfield(R,"convg_iter")
        error("Results in %s must contain true_params, estimations_DA, convg_iter.", matFiles(i));
    end
    if metric == "CA" && ~isfield(R,"estimations_CA")
        error("metric='CA' requires Results.estimations_CA in %s.", matFiles(i));
    end

    Tcell = R.true_params;       % cell(MC,T)
    Ecell = R.estimations_DA;    % cell(MC,T)
    Ccell = R.convg_iter;        % cell(MC,T)
    if metric == "CA"
        CAc = R.estimations_CA;  % cell(MC,T)
    end

    [MC, T] = size(Tcell);

    % time axis
    if ~isempty(opt.time_s)
        t = opt.time_s(:);
        if numel(t) ~= T
            error("time_s length must equal TRACK_TIME (T=%d) for file %s.", T, matFiles(i));
        end
    else
        t = (0:T-1).';
    end

    % SE: 4 x T x MC
    SE = nan(4, T, MC);

    for mc = 1:MC
        for tt = 1:T
            xtrue = Tcell{mc,tt};
            if isempty(xtrue), continue; end
            xtrue = to_state4(xtrue);

            if metric == "CA"
                xhat = CAc{mc,tt};
                if isempty(xhat), continue; end
                xhat = to_state4(xhat);
            else
                xda = Ecell{mc,tt};
                if isempty(xda), continue; end
                X = to_3d(xda); % 4 x Nnode x Niter OR nested cell
                if isempty(X), continue; end

                I = get_convg_iter(Ccell{mc,tt}, size(X,3));
                slice = X(:,:,I);
                xhat  = mean(slice, 2, "omitnan"); % 4 x 1
            end

            e = xhat - xtrue;
            SE(:,tt,mc) = sqrt(e.^2);
        end
    end

    S.methods(i).name = methodNames(i);
    S.methods(i).t = t;
    S.methods(i).SE = SE;
end

%% ---- ONE figure, 2x2 subplots (tiledlayout)
fig = figure('Name','MSE by state (2x2, shaded)','Color','w');
tl = tiledlayout(fig, 2, 2, 'TileSpacing','compact', 'Padding','compact');

% Keep consistent color per method across all subplots:
co = colororder;                % default axes color order
nC = size(co,1);

for s = 1:4
    ax = nexttile(tl, s);
    hold(ax, 'on'); grid(ax, 'on');
    if strcmpi(opt.y_scale, "log")
        set(ax, 'YScale','log');
    end

    for i = 1:numel(S.methods)
        t = S.methods(i).t;
        Y = squeeze(S.methods(i).SE(s,:,:)); % T x MC
        c = co(mod(i-1, nC)+1, :);             % fixed per method
        plot_mean_shaded_ax(ax, t, Y, c, '-', opt.line_width, opt.alpha, S.methods(i).name);
    end

    title(ax, sprintf('MSE of %s', stateNames(s)), 'Interpreter','tex');
    xlabel(ax, 'Time t [s]');
    ylabel(ax, sprintf('MSE(%s)', stateNames(s)));
end

% single legend for whole figure
% lg = legend(tl, methodNames, 'Location','northoutside', 'Orientation','horizontal', 'Interpreter','tex');

lg = legend(methodNames, 'Location','northoutside', 'Orientation','horizontal', 'Interpreter','tex');
% lg.Box = 'off';

title(tl, sprintf('State-wise MSE over time (%s)', metric), 'Interpreter','tex');

apply_times_new_roman(fig, 18, 12);

end

%% ========================= helpers =========================
function x = to_state4(v)
if isempty(v), x = []; return; end
if iscell(v), v = cell2mat(v(:)); end
v = double(v(:));
if numel(v) ~= 4
    error("State vector must have 4 elements, got %d.", numel(v));
end
x = v;
end

function v = to_vec(x)
if isempty(x), v = []; return; end
if isnumeric(x), v = double(x(:)); return; end
if iscell(x), v = double(cell2mat(x(:))); v = v(:); return; end
error("to_vec: unsupported type %s", class(x));
end

function I = get_convg_iter(c, nIter)
if isempty(c)
    I = nIter;
elseif iscell(c)
    if isempty(c{1}), I = nIter; else, I = c{1}; end
else
    I = c;
end
if isempty(I) || ~isnumeric(I) || ~isscalar(I) || isnan(I) || I <= 0
    I = nIter;
end
I = round(I);
I = max(1, min(nIter, I));
end

function X = to_3d(eDA)
% Convert estimations_DA entry to numeric 3D: [4 x Nnode x Niter]
if isempty(eDA), X = []; return; end
if isnumeric(eDA), X = double(eDA); return; end
if ~iscell(eDA), error("to_3d: unsupported type %s", class(eDA)); end

nState = numel(eDA);
if nState == 0, X = []; return; end

nNodes = numel(eDA{1});
nIter  = numel(eDA{1}{1});

X = nan(nState, nNodes, nIter);
for i = 1:nState
    for j = 1:nNodes
        vv = to_vec(eDA{i}{j});
        K = min(nIter, numel(vv));
        X(i,j,1:K) = vv(1:K);
    end
end
end

function plot_mean_shaded(x, Y, colorRGB, lineStyle, lineW, alphaVal, displayName)
Y = double(Y);
mu = mean(Y, 2, 'omitnan');
sg = std(Y, 0, 2, 'omitnan');

yLo = max(mu - sg, eps);
yHi = max(mu + sg, eps);
mu  = max(mu, eps);

Xpatch = [x(:); flipud(x(:))];
Ypatch = [yHi(:); flipud(yLo(:))];

patch(Xpatch, Ypatch, colorRGB, ...
    'FaceAlpha', alphaVal, ...
    'EdgeColor', 'none', ...
    'HandleVisibility', 'off');
plot(x, mu, 'Color', colorRGB, ...
    'LineStyle', lineStyle, ...
    'LineWidth', lineW, ...
    'DisplayName', displayName);
end

function apply_times_new_roman(fig, titleSize, axisSize)
if nargin < 2, titleSize = 18; end
if nargin < 3, axisSize = 14; end

set(fig, 'Color','w');
set(findall(fig, '-property','FontName'), 'FontName', 'Times New Roman');

ax = findall(fig, 'Type','axes');
for k = 1:numel(ax)
    ax(k).FontName = 'Times New Roman';
    ax(k).FontSize = axisSize;
    ax(k).LineWidth = 1.0;

    ax(k).XLabel.FontName = 'Times New Roman';
    ax(k).YLabel.FontName = 'Times New Roman';

    if isprop(ax(k), 'Title') && ~isempty(ax(k).Title)
        ax(k).Title.FontName = 'Times New Roman';
        ax(k).Title.FontSize = titleSize;
    end
end

lg = findall(fig, 'Type','legend');
for k = 1:numel(lg)
    lg(k).FontName = 'Times New Roman';
    lg(k).FontSize = axisSize;
end
end

function [Pmat, Dmat, itMax] = stack_residuals_at_time(Pcell, Dcell, t_pick)
[MC, ~] = size(Pcell);
iters = zeros(MC,1);

for mc = 1:MC
    pr = Pcell{mc, t_pick};
    du = Dcell{mc, t_pick};
    if ~isempty(pr)
        iters(mc) = numel(to_vec(pr));
    elseif ~isempty(du)
        iters(mc) = numel(to_vec(du));
    else
        iters(mc) = 0;
    end
end

itMax = max(iters);
if itMax <= 0
    Pmat = []; Dmat = [];
    return;
end

Pmat = nan(itMax, MC);
Dmat = nan(itMax, MC);

for mc = 1:MC
    pr = Pcell{mc, t_pick};
    du = Dcell{mc, t_pick};

    if ~isempty(pr)
        v = to_vec(pr);
        Pmat(1:numel(v), mc) = v;
    end
    if ~isempty(du)
        v = to_vec(du);
        Dmat(1:numel(v), mc) = v;
    end
end
end
function c = get_next_color(ax)
% Return next color from axes ColorOrder, cycling by ColorOrderIndex
co = ax.ColorOrder;
idx = ax.ColorOrderIndex;
c = co(idx, :);
ax.ColorOrderIndex = idx + 1;
end

function plot_mean_shaded_ax(ax, x, Y, colorRGB, lineStyle, lineW, alphaVal, displayName)
Y = double(Y);
mu = mean(Y, 2, 'omitnan');
sg = std(Y, 0, 2, 'omitnan');

% log-safe clamp
yLo = max(mu - sg, eps);
yHi = max(mu + sg, eps);
mu  = max(mu, eps);

Xpatch = [x(:); flipud(x(:))];
Ypatch = [yHi(:); flipud(yLo(:))];

patch(ax, Xpatch, Ypatch, colorRGB, ...
    'FaceAlpha', alphaVal, ...
    'EdgeColor', 'none', ...
    'HandleVisibility', 'off');

plot(ax, x, mu, 'Color', colorRGB, ...
    'LineStyle', lineStyle, ...
    'LineWidth', lineW, ...
    'DisplayName', displayName);
end



function S = plot_tracking_compare_mse_centralized_vs_decentralized(matFiles, varargin)
%PLOT_TRACKING_COMPARE_MSE_CENTRALIZED_VS_DECENTRALIZED
% Compare multiple .mat methods and plot per-state error for:
%   - Decentralized: Results.estimations_DA + Results.convg_iter
%   - Centralized  : Results.estimations_CA
%
% Produces ONE figure with 2x2 tiledlayout, each subplot has both curves.
%
% Expected per .mat:
%   log.Results.true_params      cell(MC,T) each -> 4x1 or 1x4
%   log.Results.estimations_DA   cell(MC,T) each -> 4 x Nnode x Niter (or common nested cell forms)
%   log.Results.convg_iter       cell(MC,T) each -> scalar or empty
%   log.Results.estimations_CA   cell(MC,T) each -> 4x1 or 1x4
%
% Usage:
%   plot_tracking_compare_mse_centralized_vs_decentralized(["mle.mat","mmse.mat"], ...
%       "method_names", ["MLE","MMSE"], "time_s", 0:0.1:2.9, "y_scale","log");
%
% Options:
%   "time_s"          : time vector length T (default: 0:T-1)
%   "method_names"    : names for legend
%   "alpha"           : shade alpha (0 disables shading)
%   "line_width"      : line width
%   "y_scale"         : "log" or "linear"
%   "error_metric"    : "MSE" (default) or "RMSE"
%   "style_dec"       : line style for decentralized (default "-")
%   "style_cen"       : line style for centralized (default "--")
%   "shade_mode"      : "both" (default) | "dec_only" | "cen_only" | "none"
%   "legend_location" : e.g. "southoutside" (default)

%% ---- parse inputs
if ischar(matFiles) || isstring(matFiles)
    matFiles = string(matFiles);
elseif iscell(matFiles)
    matFiles = string(matFiles);
else
    error("matFiles must be string array, char, or cellstr.");
end
matFiles = matFiles(:);

p = inputParser;
p.addParameter("time_s", [], @(x) isempty(x) || isnumeric(x));
p.addParameter("method_names", [], @(x) isempty(x) || isstring(x) || iscell(x));
p.addParameter("alpha", 0.15, @(x) isnumeric(x) && isscalar(x));
p.addParameter("line_width", 2.2, @(x) isnumeric(x) && isscalar(x));
p.addParameter("y_scale", "log", @(s) any(strcmpi(string(s), ["log","linear"])));
p.addParameter("error_metric", "MSE", @(s) any(strcmpi(string(s), ["MSE","RMSE"])));
p.addParameter("style_dec", "-", @(s) ischar(s) || isstring(s));
p.addParameter("style_cen", "--", @(s) ischar(s) || isstring(s));
p.addParameter("shade_mode", "both", @(s) any(strcmpi(string(s), ["both","dec_only","cen_only","none"])));
p.addParameter("legend_location", "southoutside", @(s) ischar(s) || isstring(s));
p.parse(varargin{:});
opt = p.Results;

stateNames = ["x","y","v_x","v_y"];
errMetric  = upper(string(opt.error_metric));
shadeMode  = lower(string(opt.shade_mode));

% method names
if isempty(opt.method_names)
    methodNames = strings(numel(matFiles),1);
    for i = 1:numel(matFiles)
        [~, bn, ~] = fileparts(matFiles(i));
        methodNames(i) = string(bn);
    end
else
    methodNames = string(opt.method_names);
    methodNames = methodNames(:);
    if numel(methodNames) ~= numel(matFiles)
        error("method_names length must match matFiles length.");
    end
end

%% ---- load and compute per-state error for both modes
S = struct();
S.files = matFiles;
S.method_names = methodNames;
S.error_metric = errMetric;

S.methods = repmat(struct( ...
    "name","", "t",[], ...
    "DEC", struct("E",[]), ...
    "CEN", struct("E",[]) ...
), numel(matFiles), 1);

for i = 1:numel(matFiles)
    tmp = load(matFiles(i));
    if ~isfield(tmp, "log") || ~isfield(tmp.log, "Results")
        error("File %s must contain variable 'log.Results'.", matFiles(i));
    end
    R = tmp.log.Results;

    mustHave = ["true_params","estimations_DA","convg_iter","estimations_CA"];
    for f = mustHave
        if ~isfield(R, f)
            error("Results in %s must contain '%s'.", matFiles(i), f);
        end
    end

    Tcell = R.true_params;     % cell(MC,T)
    Edec  = R.estimations_DA;  % cell(MC,T)
    Citer = R.convg_iter;      % cell(MC,T)
    Ecen  = R.estimations_CA;  % cell(MC,T)

    [MC, T] = size(Tcell);

    % time axis
    if ~isempty(opt.time_s)
        t = opt.time_s(:);
        if numel(t) ~= T
            error("time_s length must equal T=%d for file %s.", T, matFiles(i));
        end
    else
        t = (0:T-1).';
    end

    % E: 4 x T x MC  (either MSE or RMSE later)
    E_DEC = nan(4, T, MC);
    E_CEN = nan(4, T, MC);

    for mc = 1:MC
        for tt = 1:T
            xtrue = Tcell{mc,tt};
            if isempty(xtrue), continue; end
            xtrue = to_state4(xtrue);

            % ---- centralized
            xhatC = Ecen{mc,tt};
            if ~isempty(xhatC)
                xhatC = to_state4(xhatC);
                eC = xhatC - xtrue;
                E_CEN(:,tt,mc) = sqrt(eC.^2)/10;  % store squared error
            end

            % ---- decentralized
            xda = Edec{mc,tt};
            if ~isempty(xda)
                X = to_3d(xda);           % 4 x Nnode x Niter
                if ~isempty(X)
                    I = get_convg_iter(Citer{mc,tt}, size(X,3));
                    slice = X(:,:,I);     % 4 x Nnode
                    xhatD = mean(slice, 2, "omitnan"); % 4 x 1
                    eD = xhatD - xtrue;
                    E_DEC(:,tt,mc) = sqrt(eD.^2)/10;          % store squared error
                end
            end
        end
    end

    % convert to RMSE if requested (still keep MC dimension)
    if errMetric == "RMSE"
        E_DEC = sqrt(E_DEC);
        E_CEN = sqrt(E_CEN);
    end

    S.methods(i).name = methodNames(i);
    S.methods(i).t    = t;
    S.methods(i).DEC.E = E_DEC;
    S.methods(i).CEN.E = E_CEN;
end

%% ---- plot: ONE figure, 2x2
fig = figure('Name', sprintf('%s by state (Centralized vs Decentralized)', errMetric), 'Color','w');
tl = tiledlayout(fig, 2, 2, 'TileSpacing','compact', 'Padding','compact');

co = colororder;  nC = size(co,1);

% build legend entries (method x mode)
legendHandles = gobjects(0,1);
legendLabels  = strings(0,1);

for s = 1:4
    ax = nexttile(tl, s);
    hold(ax, 'on'); grid(ax, 'on');
    if strcmpi(opt.y_scale, "log"), set(ax, 'YScale','log'); end

    for i = 1:numel(S.methods)
        c = co(mod(i-1, nC)+1, :);
        t = S.methods(i).t;

        Yd = squeeze(S.methods(i).DEC.E(s,:,:)); % T x MC
        Yc = squeeze(S.methods(i).CEN.E(s,:,:)); % T x MC

        % decentralized (solid)
        [hD] = plot_mean_shaded_ax(ax, t, Yd, c, char(opt.style_dec), opt.line_width, ...
            (opt.alpha>0) && any(strcmp(shadeMode, ["both","dec_only"])), opt.alpha, ...
            S.methods(i).name + " (Decentralized)");

        % centralized (dashed)
        [hC] = plot_mean_shaded_ax(ax, t, Yc, c, char(opt.style_cen), opt.line_width, ...
            (opt.alpha>0) && any(strcmp(shadeMode, ["both","cen_only"])), opt.alpha*0.8, ...
            S.methods(i).name + " (Centralized)");

        if s == 1
            if ~isempty(hD), legendHandles(end+1,1) = hD; legendLabels(end+1,1) = S.methods(i).name + " (Decentralized)"; end %#ok<AGROW>
            if ~isempty(hC), legendHandles(end+1,1) = hC; legendLabels(end+1,1) = S.methods(i).name + " (Centralized)"; end %#ok<AGROW>
        end
    end

    title(ax, sprintf('%s of %s', errMetric, stateNames(s)), 'Interpreter','tex');
    xlabel(ax, 'Time t [s]');
    ylabel(ax, sprintf('%s(%s)', errMetric, stateNames(s)));
end

if ~isempty(legendHandles)
    lg = legend(legendHandles, legendLabels, 'Location', char(opt.legend_location));
    lg.Interpreter = 'none';
end

%% =================== helpers ===================
function x = to_state4(xin)
    x = xin;
    if isrow(x), x = x.'; end
    x = x(:);
    if numel(x) ~= 4
        error("State must have 4 elements; got %d.", numel(x));
    end
end

function X = to_3d(xda)
% Return 4 x Nnode x Niter from common representations.
    if isempty(xda)
        X = [];
        return;
    end

    if isnumeric(xda)
        if ndims(xda) == 3
            X = xda;
            return;
        elseif ismatrix(xda)
            % assume 4 x Niter (single node) OR 4 x Nnode (single iter)
            if size(xda,1) == 4
                % interpret as 4 x Niter (single node)
                X = reshape(xda, 4, 1, size(xda,2));
                return;
            elseif size(xda,2) == 4
                X = reshape(xda.', 4, 1, size(xda,1));
                return;
            end
        end
        error("Unsupported numeric estimations_DA shape.");
    end

    if iscell(xda)
        % Case A: 1xNnode, each cell is 4xNiter
        if isvector(xda) && all(cellfun(@(z) isnumeric(z) && ~isempty(z), xda))
            Nnode = numel(xda);
            nI = zeros(Nnode,1);
            for k=1:Nnode
                zk = xda{k};
                if size(zk,1) ~= 4 && size(zk,2) == 4
                    zk = zk.'; % make 4 x Niter
                end
                if size(zk,1) ~= 4, error("Cell node %d must be 4xNiter.", k); end
                nI(k) = size(zk,2);
                xda{k} = zk;
            end
            Niter = max(nI);
            X = nan(4, Nnode, Niter);
            for k=1:Nnode
                zk = xda{k};
                X(:,k,1:size(zk,2)) = zk;
            end
            return;
        end

        % Case B: cell(Nnode,Niter), each cell is 4x1
        if ismatrix(xda)
            [Nnode, Niter] = size(xda);
            X = nan(4, Nnode, Niter);
            for k=1:Nnode
                for it=1:Niter
                    z = xda{k,it};
                    if isempty(z), continue; end
                    z = to_state4(z);
                    X(:,k,it) = z;
                end
            end
            return;
        end

        error("Unsupported cell estimations_DA format.");
    end

    error("Unsupported estimations_DA type.");
end

function I = get_convg_iter(ci, Niter)
% Robust convergence-iter selection (1..Niter)
    if isempty(ci)
        I = Niter;
        return;
    end
    if iscell(ci), ci = ci{1}; end
    if isempty(ci) || ~isfinite(ci)
        I = Niter;
        return;
    end
    I = round(double(ci));
    I = max(1, min(Niter, I));
end

function hLine = plot_mean_shaded_ax(ax, t, Y, colorRGB, ls, lw, doShade, alphaVal, dispName)
% Y is T x MC. Plot mean +/- std (omitnan).
    hLine = gobjects(0);

    if isempty(Y) || all(isnan(Y(:)))
        return;
    end

    mu = mean(Y, 2, "omitnan");
    sd = std(Y, 0, 2, "omitnan");

    if doShade && alphaVal > 0
        lo = mu - sd;
        hi = mu + sd;
        x = [t(:); flipud(t(:))];
        y = [lo(:); flipud(hi(:))];
        patch(ax, x, y, colorRGB, 'FaceAlpha', alphaVal, 'EdgeColor', 'none', 'HandleVisibility','off');
    end

    hLine = plot(ax, t, mu, 'LineStyle', ls, 'LineWidth', lw, 'Color', colorRGB, ...
        'DisplayName', char(dispName));
end

end
