function Report = benchmark_batch_linearization(repetitions)
%BENCHMARK_BATCH_LINEARIZATION Time a frozen local solve, not ADMM rounds.
% Both models use analytic gradients, identical data, priors, penalties and
% fmincon settings. They represent different objectives away from the anchor.
% Nothing is plotted or saved; the caller may save the returned report.

if nargin == 0
    repetitions = 3;
end
validateattributes(repetitions, {'numeric'}, {'scalar','integer','>=',3,'<=',5});
project_root = fileparts(mfilename('fullpath'));
previous_path = path;
path_cleanup = onCleanup(@() path(previous_path));
addpath(fullfile(project_root, 'all_in_one'), fullfile(project_root, 'common'));
cfg = tracking_config('original');
network = make_connected_random_graph(cfg.Graph, cfg.Radar);
node = 1;
sensor_ids = sort([node, network.neighbors{node}]);
radars = network.radar_pos(sensor_ids, :);
num_sensors = numel(sensor_ids);
num_samples = 64;
lambda = 3e8 / 10e9;
snr_linear = 10^(50/10);
range_variance = 3*(3e8)^2 / (8*pi^2*(10e6)^2*snr_linear);
doppler_variance = 3*(20e6)^2 / (pi^2*snr_linear*num_samples^3);
cross_covariance = 1e-6 * sqrt(range_variance*doppler_variance);
Sigma = [range_variance, cross_covariance; cross_covariance, doppler_variance];

truth = [-20; -15; 8; 5];
anchor = truth + [2.5; -2; 1; -0.75];
initial_state = anchor;
model = BatchADMMUtils.linearize_measurements(anchor, radars, lambda);
stream = RandStream('mt19937ar', 'Seed', 20261015);
noise_factor = chol(Sigma, 'lower');
range_measurements = zeros(num_samples, num_sensors);
doppler_measurements = zeros(num_samples, num_sensors);
for j = 1:num_sensors
    prediction = physical_measurement(truth, radars(j,:), lambda);
    samples = prediction + noise_factor * randn(stream, 2, num_samples);
    range_measurements(:,j) = samples(1,:).';
    doppler_measurements(:,j) = samples(2,:).';
end

% Use the same retained batch prior and augmented-cost convention on both
% sides. Nonzero duals and edge offsets exercise all local quadratic terms.
prior_mean = repmat({anchor + [0.75; -0.5; 0.25; -0.15]}, 1, num_sensors);
prior_covariance = repmat({diag([range_variance, range_variance, ...
    doppler_variance, doppler_variance])}, 1, num_sensors);
penalty = [100, 100, 15, 15];
duals = cell(1, network.numNodes);
edge_states = cell(1, network.numNodes);
duals{node} = zeros(4, network.numNodes);
edge_states{node} = zeros(4, network.numNodes);
for j = network.neighbors{node}
    duals{node}(:,j) = [0.5; -0.3; 0.2; -0.1] * j;
    edge_states{node}(:,j) = anchor + [0.08; -0.05; 0.03; -0.02] * j;
end

affine = @(x) BatchADMMUtils.posteriorWithConsensus(x, ...
    range_measurements, doppler_measurements, prior_mean, prior_covariance, ...
    model, Sigma, node, network.neighbors, duals, edge_states, penalty);
nonlinear = @(x) nonlinear_local_cost(x, range_measurements, ...
    doppler_measurements, radars, lambda, Sigma, prior_mean, prior_covariance, ...
    node, network.neighbors, duals, edge_states, penalty);
[affine_anchor_value, affine_anchor_gradient] = affine(anchor);
[nonlinear_anchor_value, nonlinear_anchor_gradient] = nonlinear(anchor);
anchor_value_gap = abs(affine_anchor_value - nonlinear_anchor_value);
anchor_gradient_gap = norm(affine_anchor_gradient - nonlinear_anchor_gradient, inf);
assert(anchor_value_gap <= 1e-10 * max(1, abs(nonlinear_anchor_value)) && ...
    anchor_gradient_gap <= 1e-10 * max(1, norm(nonlinear_anchor_gradient, inf)), ...
    'The two objectives must agree in value and gradient at the anchor.');

% Independently assemble Q*x = b for the affine problem, including the
% legacy prior coefficient and c.^2/4 penalty; no production gradient used.
[Q, b] = normal_equations(model, range_measurements, doppler_measurements, ...
    Sigma, prior_mean, prior_covariance, node, network.neighbors, ...
    duals, edge_states, penalty);
reference_state = Q \ b;
[reference_value, reference_gradient] = affine(reference_state);
reference_scale = max(1, norm(Q,inf)*norm(reference_state,inf) + norm(b,inf));
assert(norm(reference_gradient,inf) / reference_scale < 1e-9, ...
    'Independent affine normal equations disagree with the objective gradient.');

options = optimoptions('fmincon', 'Display', 'off', ...
    'SpecifyObjectiveGradient', true, 'ScaleProblem', true, ...
    'OptimalityTolerance', 1e-8, 'FunctionTolerance', 1e-10, ...
    'StepTolerance', 1e-10, 'MaxIterations', 200, 'MaxFunctionEvaluations', 2000);
objectives = {nonlinear, affine};
methods = ["Nonlinear"; "Fixed affine"];
for m = 1:2
    fmincon(objectives{m}, initial_state, [], [], [], [], [], [], [], options);
end

elapsed = zeros(repetitions,2);
function_count = zeros(repetitions,2);
iterations = zeros(repetitions,2);
exitflags = zeros(repetitions,2);
objective_values = zeros(repetitions,2);
gradient_inf = zeros(repetitions,2);
states = zeros(4,repetitions,2);
relative_affine_stationarity = nan(repetitions,2);
affine_state_gap = nan(repetitions,2);
affine_objective_gap = nan(repetitions,2);
for trial = 1:repetitions
    order = 1:2;
    if mod(trial,2) == 0
        order = 2:-1:1;
    end
    for m = order
        timer = tic;
        [x, value, flag, output] = fmincon(objectives{m}, initial_state, ...
            [], [], [], [], [], [], [], options);
        elapsed(trial,m) = toc(timer);
        [~, gradient] = objectives{m}(x);
        states(:,trial,m) = x;
        function_count(trial,m) = output.funcCount;
        iterations(trial,m) = output.iterations;
        exitflags(trial,m) = flag;
        objective_values(trial,m) = value;
        gradient_inf(trial,m) = norm(gradient,inf);
        if m == 2
            relative_affine_stationarity(trial,m) = norm(gradient,inf) / ...
                max(1, norm(Q,inf)*norm(x,inf) + norm(b,inf));
            affine_state_gap(trial,m) = norm(x-reference_state);
            affine_objective_gap(trial,m) = abs(value-reference_value) / ...
                max(1,abs(reference_value));
        end
    end
end

Report.Summary = table(methods, median(elapsed,1).', median(function_count,1).', ...
    median(iterations,1).', min(exitflags,[],1).', max(exitflags,[],1).', ...
    median(gradient_inf,1).', median(relative_affine_stationarity,1).', ...
    median(affine_state_gap,1).', median(affine_objective_gap,1).', ...
    'VariableNames', {'Model','MedianSeconds','MedianFunctionCount','MedianIterations', ...
    'MinExitFlag','MaxExitFlag','MedianGradientInf','MedianRelativeAffineStationarity', ...
    'MedianAffineReferenceStateGap','MedianAffineReferenceRelativeObjectiveGap'});
Report.NonlinearToAffineTimeRatio = median(elapsed(:,1)) / median(elapsed(:,2));
Report.Trials = struct('Seconds',elapsed, 'FunctionCount',function_count, ...
    'Iterations',iterations, 'ExitFlags',exitflags, 'ObjectiveValues',objective_values, ...
    'GradientInf',gradient_inf, 'States',states);
Report.AffineReference = struct('State',reference_state, 'Objective',reference_value, ...
    'GradientInf',norm(reference_gradient,inf), ...
    'RelativeStationarity',norm(reference_gradient,inf)/reference_scale, ...
    'Hessian',Q, 'RightHandSide',b);
Report.AnchorAgreement = struct('ValueGap',anchor_value_gap,'GradientInfGap',anchor_gradient_gap);
Report.Input = struct('Node',node, 'SensorIDs',sensor_ids, 'RadarPositions',radars, ...
    'Truth',truth, 'Anchor',anchor, 'InitialState',initial_state, 'Sigma',Sigma, ...
    'RangeMeasurements',range_measurements, 'DopplerMeasurements',doppler_measurements, ...
    'PriorMeans',{prior_mean}, 'PriorCovariances',{prior_covariance}, ...
    'Penalty',penalty, 'Duals',{duals}, 'EdgeStates',{edge_states}, ...
    'GraphConfig',cfg.Graph, 'RadarConfig',cfg.Radar);
Report.SolverOptions = options;
Report.Scope = "Frozen local solves only; different objectives away from the anchor. " + ...
    "Time ratio is not an ADMM communication-round comparison. " + ...
    "Both models use the corrected Doppler sign/full covariance and retained legacy prior/penalty.";
end

function [value, gradient] = nonlinear_local_cost(x, ranges, dopplers, radars, ...
        lambda, Sigma, prior_means, prior_covariances, node, neighbors, duals, edges, penalty)
value = 0;
gradient = zeros(4,1);
for j = 1:size(radars,1)
    [prediction, H] = physical_measurement(x, radars(j,:), lambda);
    residual = [ranges(:,j).'; dopplers(:,j).'] - prediction;
    weighted_residual = Sigma \ residual;
    value = value + 0.5 * sum(residual .* weighted_residual, 'all');
    gradient = gradient - H.' * sum(weighted_residual,2);
end
[prior, prior_gradient] = BatchADMMUtils.state_prior_distribution( ...
    x, prior_means, prior_covariances, numel(prior_means));
[consensus, consensus_gradient] = BatchADMMUtils.consensus_cost( ...
    x, node, neighbors, duals, edges, penalty);
value = value + prior + consensus;
gradient = gradient + prior_gradient + consensus_gradient;
end

function [prediction, H] = physical_measurement(x, radar, lambda)
displacement = x(1:2) - radar(:);
range = norm(displacement);
assert(range > 1e-9, 'The benchmark encountered a zero-range measurement.');
direction = displacement / range;
velocity = x(3:4);
radial_velocity = direction.' * velocity;
prediction = [range; radial_velocity/lambda];
H = [direction.', zeros(1,2); ...
    (velocity-direction*radial_velocity).'/(lambda*range), direction.'/lambda];
end

function [Q, b] = normal_equations(model, ranges, dopplers, Sigma, ...
        means, covariances, node, neighbors, duals, edges, penalty)
Q = zeros(4);
b = zeros(4,1);
M = size(ranges,1);
for j = 1:size(model.value,2)
    H = model.H(:,:,j);
    intercept = model.value(:,j) - H*model.anchor;
    observation_sum = [sum(ranges(:,j)); sum(dopplers(:,j))] - M*intercept;
    Q = Q + M * H.' * (Sigma \ H);
    b = b + H.' * (Sigma \ observation_sum);
end
for j = 1:numel(means)
    weight = 1 / sqrt((2*pi)^4 * det(covariances{j}));
    prior_precision = (2*weight/numel(means)) * (covariances{j} \ eye(4));
    Q = Q + prior_precision;
    b = b + prior_precision*means{j};
end
penalty_hessian = diag(penalty(:).^2/2);
for j = neighbors{node}
    Q = Q + penalty_hessian;
    b = b + penalty_hessian*edges{node}(:,j) - duals{node}(:,j);
end
end
