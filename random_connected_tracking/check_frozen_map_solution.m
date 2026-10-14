function Comparison = check_frozen_map_solution(Run)
%CHECK_FROZEN_MAP_SOLUTION Compare first-scan ADMM with its frozen affine MAP.
% Uses MC1/scan1 only. The replay changes only the ADMM cap to 100; the
% reference solves the saved affine model's normal equations directly.
% Only new runs with saved first_order_affine model/prior metadata are valid.
% Historical nonlinear runs require their matching historical source.
% Saves frozen_map_comparison.csv and frozen_map_comparison.mat beside Run.

% Reject historical runs before path changes, solver replay, or output writes.
if ~isfield(Run, 'Results') || ...
        ~isfield(Run.Results, 'MeasurementModel') || ...
        ~isscalar(string(Run.Results.MeasurementModel)) || ...
        string(Run.Results.MeasurementModel) ~= "first_order_affine"
    error('RadarTracking:FrozenMAPModel', ...
        ['This diagnostic requires a new first_order_affine saved run. ', ...
         'Historical nonlinear runs require their matching historical source; ', ...
         'they cannot be replayed or overwritten by this diagnostic.']);
end
if string(Run.Method) ~= "EKF" || string(Run.Results.Config.ADMMObjectiveType) ~= "MAP"
    error('RadarTracking:FrozenMAPMethod', 'Supply an EKF run with MAP correction.');
end
if isempty(Run.OutputFolder) || ~isfolder(Run.OutputFolder)
    error('RadarTracking:FrozenMAPOutput', 'Run.OutputFolder must be an existing saved-run folder.');
end
project_root = fileparts(mfilename('fullpath'));
previous_path = path;
path_cleanup = onCleanup(@() path(previous_path)); %#ok<NASGU>
addpath(fullfile(project_root, 'ADMM_EKF'));
previous_rng = rng;
rng_cleanup = onCleanup(@() rng(previous_rng)); %#ok<NASGU>

R = Run.Results;
cfg = R.Config;
network = Run.network_topo;
env = R.env;
ut = TrackingADMMUtils;
models = TrackingModels;
neighbors = ut.get_neighbors(network.adj_matrix, network.numNodes);
num_nodes = network.numNodes;
required_metadata = {'linearization_DA_raw', ...
    'prior_mean_common_raw', 'prior_cov_common_raw'};
for k = 1:numel(required_metadata)
    name = required_metadata{k};
    if ~isfield(R, name) || ~iscell(R.(name)) || isempty(R.(name)) || ...
            isempty(R.(name){1, 1})
        error('RadarTracking:FrozenMAPModel', ...
            'The affine run is missing saved MC1/scan1 metadata: %s.', name);
    end
end
model = R.linearization_DA_raw{1, 1};
prior_mean = R.prior_mean_common_raw{1, 1}(:);
prior_cov = R.prior_cov_common_raw{1, 1};
validate_frozen_model(model, prior_mean, prior_cov, num_nodes);

% The public replay API constructs its model from the common prior. Verify
% that construction matches the saved model before running any optimizer.
replay_model = models.linearize_measurements(prior_mean, network, env);
assert_same_model(model, replay_model);

range_meas = R.range_meas_raw{1};
doppler_meas = R.doppler_meas_raw{1};
range_first = reshape(range_meas(1, :, 1), 1, num_nodes);
doppler_first = reshape(doppler_meas(1, :, 1), 1, num_nodes);
z = [range_first; doppler_first];

% Use the prior actually saved for this correction, not an inferred x0/P0.
x_pred_nodes = repmat({prior_mean}, 1, num_nodes);
P_pred_nodes = repmat({prior_cov}, 1, num_nodes);
[short_mean, short_nodes, ~, short_info] = tracking_consensus_admm_update( ...
    x_pred_nodes, P_pred_nodes, range_meas, doppler_meas, 1, ...
    network, neighbors, env, models, ut, cfg.ADMMObjectiveType, ...
    R.MeasurementMode, cfg.PreWhiten, 100, cfg.ADMMAbsTolerance, ...
    cfg.ADMMRelTolerance, cfg.ADMMPenalty, cfg.ADMMRhoUpdateInterval, ...
    cfg.ADMMRhoBalanceMu, cfg.ADMMRhoScale, 0, false);
saved_mean = R.estimations_DA_raw{1, 1}(:);

% Independent physical-unit objective: one common prior, each measurement
% once, and the saved affine intercept/Jacobian held fixed for every x.
objective = @(x) affine_map_cost(x, prior_mean, prior_cov, z, model, env.Sigma);
[reference, reference_covariance, reference_information, reference_rhs] = ...
    affine_map_reference(prior_mean, prior_cov, z, model, env.Sigma);
[reference_value, reference_gradient] = objective(reference);
reference_exitflag = 1; % Compatibility field: successful direct solve.
reference_output = struct('algorithm', 'normal-equations direct solve', ...
    'iterations', 0, 'firstorderopt', norm(reference_gradient), ...
    'message', 'Unique affine MAP minimizer obtained by a direct linear solve.');

states = [saved_mean, short_mean, reference];
objective_values = zeros(3, 1);
gradient_norms = zeros(3, 1);
utility_differences = zeros(3, 1);
for k = 1:3
    [objective_values(k), gradient] = objective(states(:, k));
    gradient_norms(k) = norm(gradient);
    if cfg.PreWhiten
        utility_value = ut.MAP(states(:, k), range_first, doppler_first, ...
            {prior_mean}, {prior_cov}, model, env.Sigma_filter, env.pre_whit_L);
    else
        utility_value = ut.MAP(states(:, k), range_first, doppler_first, ...
            {prior_mean}, {prior_cov}, model, env.Sigma);
    end
    % Both definitions omit state-independent Gaussian normalizers.
    utility_differences(k) = utility_value - objective_values(k);
end
if any(~isfinite(utility_differences)) || ...
        any(abs(utility_differences) > 1e-8 * max(1, abs(objective_values)))
    error('RadarTracking:FrozenMAPObjective', ...
        'The diagnostic objective does not match the existing MAP utility.');
end

objective_gaps = objective_values - reference_value;
relative_gaps = objective_gaps / max(1, abs(reference_value));
position_gaps = vecnorm(states(1:2, :) - reference(1:2), 2, 1).';
velocity_gaps = vecnorm(states(3:4, :) - reference(3:4), 2, 1).';
primal_ratios = [ ...
    R.primal_residauls_raw{1, 1}(end) / R.admm_eps_pri_raw{1, 1}(end); ...
    short_info.primal_residual(end) / short_info.eps_pri(end); NaN];
dual_ratios = [ ...
    R.dual_residauls_raw{1, 1}(end) / R.admm_eps_dual_raw{1, 1}(end); ...
    short_info.dual_residual(end) / short_info.eps_dual(end); NaN];
iterations = [R.convg_iter_raw{1, 1}; short_info.iter; reference_output.iterations];
stopping_rule_met = [logical(R.admm_converged_raw{1, 1}); short_info.converged; false];
local_nonpositive_flags = [R.fmincon_failure_count_raw{1, 1}; ...
    short_info.fmincon_failure_count; NaN];
last_local_exitflags = [string(mat2str(R.fmincon_exitflag_raw{1, 1}.')); ...
    string(mat2str(short_info.fmincon_exitflag.')); "not applicable"];
solver_status = [string(R.admm_stop_reason_raw{1, 1}); ...
    string(short_info.stop_reason); "direct solve"];
Comparison.Table = table(["SavedADMM"; "Replay100"; "CentralizedAffineMAP"], ...
    iterations, stopping_rule_met, primal_ratios, dual_ratios, ...
    objective_values, objective_gaps, relative_gaps, position_gaps, velocity_gaps, ...
    gradient_norms, local_nonpositive_flags, last_local_exitflags, solver_status, ...
    'VariableNames', {'Method', 'Iterations', 'ADMMStoppingRuleMet', ...
    'PrimalRatio', 'DualRatio', 'Objective', 'ObjectiveGap', 'RelativeObjectiveGap', ...
    'PositionGapM', 'VelocityGapMPerS', 'ObjectiveGradientNorm', ...
    'LocalNonpositiveFlags', 'LastLocalExitflags', 'SolverStatus'});
Comparison.MonteCarloTrial = 1;
Comparison.Scan = 1;
Comparison.MeasurementModel = 'first_order_affine';
Comparison.Linearization = model;
Comparison.PriorMean = prior_mean;
Comparison.PriorCovariance = prior_cov;
Comparison.Measurements = z;
Comparison.PhysicalMeasurementCovariance = env.Sigma;
Comparison.RadarPositions = network.radar_pos;
Comparison.Adjacency = network.adj_matrix;
Comparison.Config = cfg;
Comparison.SavedADMM.Mean = saved_mean;
Comparison.SavedADMM.Nodes = R.estimations_DA_nodes_raw{1, 1};
Comparison.Replay100.Mean = short_mean;
Comparison.Replay100.Nodes = short_nodes;
Comparison.Replay100.Info = short_info;
Comparison.Reference.State = reference;
Comparison.Reference.Covariance = reference_covariance;
Comparison.Reference.Information = reference_information;
Comparison.Reference.IncrementNormalEquationRHS = reference_rhs;
Comparison.Reference.Objective = reference_value;
Comparison.Reference.Iterations = 0;
Comparison.Reference.Status = 'direct solve';
Comparison.Reference.Exitflag = reference_exitflag;
Comparison.Reference.SolverOutput = reference_output;
Comparison.ObjectiveCrosscheckDifference = utility_differences;
Comparison.ObjectiveDefinition = ...
    ['0.5*(x-prior_mean)''*prior_cov^(-1)*(x-prior_mean) + ', ...
     '0.5*sum_n residual_n''*Sigma^(-1)*residual_n; ', ...
     'residual_n = z_n - model.value(:,n) - model.H(:,:,n)*(x-model.anchor); ', ...
     'state-independent constants omitted'];
Comparison.RelativeGapDefinition = '(Objective - ReferenceObjective)/max(1,abs(ReferenceObjective))';
Comparison.Note = ...
    ['The centralized affine MAP reference uses the saved raw physical model ', ...
     'and common prior. Its positive-definite normal equations give the unique ', ...
     'minimizer of that frozen quadratic objective, up to numerical precision. ', ...
     'This is not a nonlinear MAP reference or a reference for other scans. ', ...
     'Reference.Exitflag=1 denotes a successful direct solve, not fmincon. ', ...
     'ADMMStoppingRuleMet is not applicable to the centralized row (reported false).'];
writetable(Comparison.Table, fullfile(Run.OutputFolder, 'frozen_map_comparison.csv'));
save(fullfile(Run.OutputFolder, 'frozen_map_comparison.mat'), 'Comparison');
disp(Comparison.Table);
fprintf('Frozen MC1/scan1 comparison saved in %s\n', Run.OutputFolder);
end

function [value, gradient] = affine_map_cost(x, prior_mean, prior_cov, z, model, Sigma)
delta = x(:) - prior_mean;
weighted_delta = prior_cov \ delta;
value = 0.5 * delta.' * weighted_delta;
gradient = weighted_delta;
for n = 1:size(z, 2)
    H = model.H(:, :, n);
    prediction = model.value(:, n) + H * (x(:) - model.anchor(:));
    residual = z(:, n) - prediction;
    weighted_residual = Sigma \ residual;
    value = value + 0.5 * residual.' * weighted_residual;
    gradient = gradient - H.' * weighted_residual;
end
end

function [reference, covariance, information, rhs] = ...
        affine_map_reference(prior_mean, prior_cov, z, model, Sigma)
% Solve for the increment from the anchor to avoid large affine offsets.
nx = numel(prior_mean);
information = prior_cov \ eye(nx);
rhs = prior_cov \ (prior_mean - model.anchor(:));
for n = 1:size(z, 2)
    H = model.H(:, :, n);
    information = information + H.' * (Sigma \ H);
    rhs = rhs + H.' * (Sigma \ (z(:, n) - model.value(:, n)));
end
information = 0.5 * (information + information.');
reference = model.anchor(:) + information \ rhs;
covariance = information \ eye(nx);
covariance = 0.5 * (covariance + covariance.');
end

function validate_frozen_model(model, prior_mean, prior_cov, num_nodes)
if ~isstruct(model) || ~isscalar(model) || ...
        ~all(isfield(model, {'anchor', 'value', 'H'})) || ...
        numel(prior_mean) ~= 4 || numel(model.anchor) ~= 4 || ...
        ~isequal(size(prior_cov), [4, 4]) || ...
        ~isequal(size(model.value), [2, num_nodes]) || ...
        size(model.H, 1) ~= 2 || size(model.H, 2) ~= 4 || ...
        size(model.H, 3) ~= num_nodes || ndims(model.H) > 3 || ...
        any(~isfinite([prior_mean(:); prior_cov(:); model.anchor(:); ...
        model.value(:); model.H(:)]))
    error('RadarTracking:FrozenMAPModel', ...
        'Saved MC1/scan1 affine model and prior metadata have invalid dimensions or values.');
end
if norm(prior_cov - prior_cov.', 'fro') > ...
        1e-12 * max(1, norm(prior_cov, 'fro'))
    error('RadarTracking:FrozenMAPModel', ...
        'The saved common prior covariance must be symmetric positive definite.');
end
[~, flag] = chol(prior_cov);
if flag ~= 0
    error('RadarTracking:FrozenMAPModel', ...
        'The saved common prior covariance must be symmetric positive definite.');
end
end

function assert_same_model(saved_model, replay_model)
fields = {'anchor', 'value', 'H'};
for k = 1:numel(fields)
    name = fields{k};
    saved_value = saved_model.(name);
    replay_value = replay_model.(name);
    if ~isequal(size(saved_value), size(replay_value)) || ...
            norm(saved_value(:) - replay_value(:)) > ...
            1e-12 * max(1, norm(saved_value(:)))
        error('RadarTracking:FrozenMAPModel', ...
            ['The current source does not reproduce saved affine model field %s. ', ...
             'Use the source matching this saved run before replaying it.'], name);
    end
end
end
