clc;clear;close all;

plot_tracking_logs("./data_log/MAP_0109");

function plot_tracking_logs(root_dir)
% plot_tracking_logs  Read all tracking_*/log.json and plot key series.
%
% Usage:
%   plot_tracking_logs(pwd)      % if you run it in the root folder
%   plot_tracking_logs("path/to/root")

if nargin < 1
    root_dir = pwd;
end

% ---- find all log.json under tracking_*
d = dir(fullfile(root_dir, "tracking_*", "log.json"));
if isempty(d)
    error("No log.json found under %s/tracking_*/log.json", root_dir);
end

log_paths = fullfile({d.folder}, {d.name});
fprintf("Found %d logs.\n", numel(log_paths));

% ---- read all logs and parse
runs = cell(numel(log_paths), 1);
for i = 1:numel(log_paths)
    runs{i} = read_one_tracking_log(log_paths{i});
end

% ---- plot: XY trajectory (per run)
figure('Name','XY trajectory (per run)');
tiledlayout("flow");
for i = 1:numel(runs)
    r = runs{i};
    nexttile;
    hold on; grid on;
    if ~isempty(r.true)
        plot(r.true(:,1), r.true(:,2), '-o', 'DisplayName', 'True');
    end
    if ~isempty(r.ca)
        plot(r.ca(:,1), r.ca(:,2), '-x', 'DisplayName', 'CA');
    end
    if ~isempty(r.da_mean)
        plot(r.da_mean(:,1), r.da_mean(:,2), '-.', 'DisplayName', 'DA mean');
    end
    xlabel('x'); ylabel('y');
    title(sprintf("Run %d: %s", i, r.run_name), 'Interpreter','none');
    legend('Location','best');
end

% ---- plot: error norm over time (per run)
figure('Name','Error norm over time (per run)');
tiledlayout("flow");
for i = 1:numel(runs)
    r = runs{i};
    nexttile;
    hold on; grid on;

    if isempty(r.true)
        title(sprintf("Run %d: (empty)", i));
        continue;
    end

    err_ca = vecnorm(r.ca - r.true, 2, 2);
    err_da = vecnorm(r.da_mean - r.true, 2, 2);

    plot(r.time_idx, err_ca, '-o', 'DisplayName','CA ||e||_2');
    plot(r.time_idx, err_da, '-o', 'DisplayName','DA mean ||e||_2');
    xlabel('time index');
    ylabel('||estimate - true||_2');
    title(sprintf("Run %d: %s", i, r.run_name), 'Interpreter','none');
    legend('Location','best');
end

% ---- plot: convergence iteration over time (per run)
figure('Name','ADMM convg_iter over time (per run)');
tiledlayout("flow");
for i = 1:numel(runs)
    r = runs{i};
    nexttile;
    grid on;
    if ~isempty(r.convg_iter)
        plot(r.time_idx, r.convg_iter, '-o');
        xlabel('time index');
        ylabel('convg\_iter');
    end
    title(sprintf("Run %d: %s", i, r.run_name), 'Interpreter','none');
end

% ---- plot: residual curves (pick one run + first valid time index)
pick_run = 1;
r = runs{pick_run};
if ~isempty(r.first_valid_time_for_residual)
    t0 = r.first_valid_time_for_residual;
    pr = r.primal_residuals{t0};
    dr = r.dual_residuals{t0};

    figure('Name','Residual curves (example)');
    tiledlayout(1,2);

    nexttile; grid on;
    plot(1:numel(pr), pr, '-o');
    xlabel('ADMM iteration'); ylabel('primal residual');
    title(sprintf("Primal residual (run %d, time=%d)", pick_run, t0));

    nexttile; grid on;
    plot(1:numel(dr), dr, '-o');
    xlabel('ADMM iteration'); ylabel('dual residual');
    title(sprintf("Dual residual (run %d, time=%d)", pick_run, t0));
else
    fprintf("No residual curves found in run %d.\n", pick_run);
end

fprintf("Done plotting.\n");

end


function R = read_one_tracking_log(json_path)
% read_one_tracking_log  Parse one tracking log.json into numeric arrays.

txt = fileread(json_path);
J = jsondecode(txt);

R = struct();
if isfield(J, "RUN_NAME"); R.run_name = string(J.RUN_NAME);
else; R.run_name = string(json_path);
end

if ~isfield(J, "Results")
    error("JSON has no field 'Results': %s", json_path);
end
res = J.Results;

% --- helper: turn cell/list to numeric (T x 4), but skip invalid entries
true_list = res.true_params;
T = numel(true_list);

valid = false(T,1);
for t = 1:T
    x = true_list{t};
    valid(t) = isnumeric(x) && numel(x) == 4;
end
time_idx = find(valid);

% true / CA
true_mat = zeros(numel(time_idx), 4);
ca_mat   = zeros(numel(time_idx), 4);

for k = 1:numel(time_idx)
    t = time_idx(k);
    true_mat(k,:) = res.true_params{t}(:).';
    ca_mat(k,:)   = res.estimations_CA{t}(:).';
end

% --- DA: estimations_DA{t} is expected to be 4 x N x I
da_mean = zeros(numel(time_idx), 4);
convg_iter = zeros(numel(time_idx), 1);

% if convg_iter missing/empty, fallback by residual length
has_convg = isfield(res, "convg_iter");

for k = 1:numel(time_idx)
    t = time_idx(k);

    est = res.estimations_DA{t};  % should be numeric 3D
    if ~isnumeric(est) || ndims(est) < 3
        da_mean(k,:) = NaN;
        convg_iter(k) = 0;
        continue;
    end

    % determine convergence iteration
    I = 0;
    if has_convg
        c = res.convg_iter{t};
        if isnumeric(c) && isscalar(c) && c > 0
            I = c;
        end
    end
    if I == 0 && isfield(res,"primal_residauls")
        pr = res.primal_residauls{t};
        if isnumeric(pr)
            I = numel(pr);
        elseif iscell(pr)
            I = numel(pr);
        end
    end
    if I <= 0
        I = size(est, 3);
    end
    I = min(max(I,1), size(est,3));
    convg_iter(k) = I;

    % take mean across nodes at iteration I
    slice = est(:,:,I);          % 4 x N
    da_mean(k,:) = mean(slice, 2).';
end

% --- residual curves (store as cell arrays indexed by time t)
primal_residuals = cell(T,1);
dual_residuals   = cell(T,1);
first_valid_time_for_residual = [];

if isfield(res, "primal_residauls")
    for t = 1:T
        pr = res.primal_residauls{t};
        if isnumeric(pr)
            primal_residuals{t} = pr(:);
        elseif iscell(pr)
            primal_residuals{t} = cell2mat(pr(:));
        else
            primal_residuals{t} = [];
        end
    end
end

if isfield(res, "dual_residauls")
    for t = 1:T
        dr = res.dual_residauls{t};
        if isnumeric(dr)
            dual_residuals{t} = dr(:);
        elseif iscell(dr)
            dual_residuals{t} = cell2mat(dr(:));
        else
            dual_residuals{t} = [];
        end
    end
end

% pick first time index that has residuals
for k = 1:numel(time_idx)
    t = time_idx(k);
    if ~isempty(primal_residuals{t}) && ~isempty(dual_residuals{t})
        first_valid_time_for_residual = t;
        break;
    end
end

% --- output struct
R.time_idx = time_idx;
R.true = true_mat;
R.ca = ca_mat;
R.da_mean = da_mean;
R.convg_iter = convg_iter;

R.primal_residuals = primal_residuals;
R.dual_residuals = dual_residuals;
R.first_valid_time_for_residual = first_valid_time_for_residual;

R.json_path = string(json_path);

end
