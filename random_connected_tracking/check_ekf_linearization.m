function Report = check_ekf_linearization()
%CHECK_EKF_LINEARIZATION Deterministic formula checks; no simulation or files.
project_root = fileparts(mfilename('fullpath'));
previous_path = path;
path_cleanup = onCleanup(@() path(previous_path));
addpath(fullfile(project_root, 'ADMM_EKF'));

anchor = [0.35; -0.22; 1.8; -0.7];
network.radar_pos = [0 0; 4 -2; -3 5; 5 6];
network.numNodes = size(network.radar_pos, 1);
env.lambda = 0.42;
env.Sigma = [0.16 0.054; 0.054 0.09];
env.PRE_WHITEN = false;
N = network.numNodes;
M = 3;
model = TrackingModels.linearize_measurements(anchor, network, env);
saved_model = model;
assert(isequal(size(model.anchor), [4 1]) && isequal(model.anchor, anchor));
assert(isequal(size(model.value), [2 N]) && isequal(size(model.H), [2 4 N]));
physical_value = physical_measurement(anchor, network, env);
Report.AnchorError = relative_error(model.value, physical_value);
assert(Report.AnchorError < 1e-12, 'The affine model must equal the physical model at its anchor.');

% Independent physical formula uses positive target-minus-radar and 2/lambda.
numeric_H = zeros(2, 4, N);
for d = 1:4
    step = 1e-6 * max(1, abs(anchor(d)));
    delta = zeros(4, 1);
    delta(d) = step;
    derivative = (physical_measurement(anchor+delta, network, env) - ...
        physical_measurement(anchor-delta, network, env)) / (2*step);
    numeric_H(:,d,:) = reshape(derivative, [2 1 N]);
end
Report.JacobianRelativeError = relative_error(model.H, numeric_H);
assert(Report.JacobianRelativeError < 2e-7, 'Physical Jacobian finite difference failed.');
direction = [0.4; -0.3; 0.2; 0.5];
steps = [0.01 0.005 0.0025];
taylor_error = zeros(size(steps));
for k = 1:numel(steps)
    candidate = anchor + steps(k)*direction;
    error = physical_measurement(candidate, network, env) - affine_measurement(candidate, model);
    taylor_error(k) = norm(error(:));
end
Report.TaylorErrorRatios = taylor_error(1:end-1) ./ taylor_error(2:end);
assert(all(Report.TaylorErrorRatios > 3.5 & Report.TaylorErrorRatios < 4.5), ...
    'Halving the perturbation must reduce Taylor error by about four.');

target.target_state = repmat(reshape(anchor, [1 1 4]), [1 M 1]);
target.target_position = target.target_state(:,:,1:2);
[true_range, true_doppler, true_measurements] = TrackingADMMUtils.gt_data_generation_tracking_ekf( ...
    zeros(1,N,M), zeros(1,N,M), zeros(1,N,2*M), target, network, env, M, 1);
for k = 1:M
    generated = [reshape(true_range(1,:,k),1,N); reshape(true_doppler(1,:,k),1,N)];
    assert(relative_error(generated, physical_value) < 1e-12, 'Generator must match the anchor and 2/lambda factor.');
    assert(relative_error(squeeze(true_measurements(1,:,2*k-1:2*k))', physical_value) < 1e-12);
end
Report.GeneratorAnchorError = relative_error(generated, physical_value);

% Inputs stay raw even when optional L whitens the complete residual and H.
range = repmat(model.value(1,:), M, 1) + 0.1*reshape(sin(1:M*N), M, N);
doppler = repmat(model.value(2,:), M, 1) + 0.2*reshape(cos(0.7*(1:M*N)), M, N);
x = anchor + [0.18; -0.11; 0.6; -0.25];
Sigma = env.Sigma;
L = chol(Sigma, 'lower') \ eye(2);
[expected_f, expected_g, likelihood_H, likelihood_rhs] = direct_likelihood(x, range, doppler, model, Sigma);
mu = {anchor+[0.1; -0.2; 0.05; -0.1], anchor+[-0.2; 0.1; -0.1; 0.15]};
sigma = {diag([0.4 0.6 0.8 1]), diag([0.6 0.9 1.1 0.5])};
[prior_f, prior_g] = direct_prior(x, mu, sigma);
neighbors = {[2 4], [1 3 4], 2, [1 2]};
n = 2;
c = [0.4 0.8 1.1 0.6];
Nu = cell(1,N);
z = cell(1,N);
for j = 1:N
    Nu{j} = 0.2*reshape(sin((1:4*N)+j), 4, N);
    z{j} = repmat(anchor, 1, N) + 0.1*reshape(cos((1:4*N)+j), 4, N);
end
[penalty_f, penalty_g] = direct_consensus(x, n, neighbors, Nu, z, c);
raw = {@(q) TrackingADMMUtils.logLikelihood(q, range, doppler, model, Sigma), ...
    @(q) TrackingADMMUtils.MAP(q, range, doppler, mu, sigma, model, Sigma), ...
    @(q) TrackingADMMUtils.logLikelihoodWithConsensus(q, range, doppler, model, Sigma, n, neighbors, Nu, z, c), ...
    @(q) TrackingADMMUtils.posteriorWithConsensus(q, range, doppler, mu, sigma, model, Sigma, n, neighbors, Nu, z, c)};
white = {@(q) TrackingADMMUtils.logLikelihood(q, range, doppler, model, eye(2), L), ...
    @(q) TrackingADMMUtils.MAP(q, range, doppler, mu, sigma, model, eye(2), L), ...
    @(q) TrackingADMMUtils.logLikelihoodWithConsensus(q, range, doppler, model, eye(2), n, neighbors, Nu, z, c, L), ...
    @(q) TrackingADMMUtils.posteriorWithConsensus(q, range, doppler, mu, sigma, model, eye(2), n, neighbors, Nu, z, c, L)};
expected_values = [expected_f, expected_f+prior_f, expected_f+penalty_f, expected_f+prior_f+penalty_f];
expected_gradients = [expected_g, expected_g+prior_g, expected_g+penalty_g, expected_g+prior_g+penalty_g];
Report.ObjectiveNames = {'Likelihood', 'MAP', 'LikelihoodConsensus', 'PosteriorConsensus'};
Report.GradientRelativeErrors = zeros(4,2); % Columns: raw, optionally whitened.
Report.WhiteningRelativeErrors = zeros(4,1);
for k = 1:4
    [value, analytic] = raw{k}(x);
    [white_value, white_analytic] = white{k}(x);
    assert(isequal(size(analytic), [4 1]) && isequal(size(white_analytic), [4 1]));
    assert(relative_error(value, expected_values(k)) < 1e-11, 'Objective coefficient check failed.');
    assert(relative_error(analytic, expected_gradients(:,k)) < 1e-11, 'Analytic coefficient check failed.');
    Report.WhiteningRelativeErrors(k) = max(relative_error(value, white_value), relative_error(analytic, white_analytic));
    assert(Report.WhiteningRelativeErrors(k) < 1e-11, 'Whitening must preserve objective value and gradient.');
    Report.GradientRelativeErrors(k,:) = [relative_error(analytic, finite_gradient(raw{k},x)), ...
        relative_error(white_analytic, finite_gradient(white{k},x))];
end
assert(all(Report.GradientRelativeErrors(:) < 2e-6), 'Objective finite difference gradient failed.');

% Self-only measurements plus N shares of precision P^-1/N give one prior.
P = diag([0.7 0.9 1.1 0.6]) + 0.03*ones(4);
[global_f, global_g] = TrackingADMMUtils.MAP(x, range, doppler, {anchor}, {P}, model, Sigma);
sum_f = 0;
sum_g = zeros(4,1);
for j = 1:N
    local_model = struct('anchor', anchor, 'value', model.value(:,j), 'H', model.H(:,:,j));
    [local_f, local_g] = TrackingADMMUtils.MAP(x, range(:,j), doppler(:,j), ...
        {anchor}, {N*P}, local_model, Sigma);
    sum_f = sum_f + local_f;
    sum_g = sum_g + local_g;
end
Report.PriorPartitionRelativeError = max(relative_error(global_f,sum_f), relative_error(global_g,sum_g));
assert(Report.PriorPartitionRelativeError < 1e-11, 'Local MAP costs must sum to the single-prior global MAP cost.');

% Frozen likelihood has one constant Hessian, including far from its anchor.
x_minimum = likelihood_H \ likelihood_rhs;
[minimum_f, minimum_g] = raw{1}(x_minimum);
assert(norm(minimum_g) < 1e-8, 'Frozen likelihood normal-equation solution must be stationary.');
offset = [0.2; -0.3; 0.4; -0.1];
[offset_f, offset_g] = raw{1}(x_minimum+offset);
assert(relative_error(offset_f-minimum_f, 0.5*offset'*likelihood_H*offset) < 1e-10);
assert(relative_error(offset_g-minimum_g, likelihood_H*offset) < 1e-10);

% One scan: normal equations and the independent Joseph update must agree.
scan_range = range(1,:);
scan_doppler = doppler(1,:);
observations = reshape([scan_range; scan_doppler], [], 1);
[~, ~, J_measurement, b_measurement] = direct_likelihood(anchor, scan_range, scan_doppler, model, Sigma);
J = (P \ eye(4)) + J_measurement;
x_normal = J \ ((P\anchor)+b_measurement);
P_normal = J \ eye(4);
H = reshape(permute(model.H, [1 3 2]), 2*N, 4);
R = kron(eye(N), Sigma);
K = P*H' / (H*P*H'+R);
x_joseph = anchor + K*(observations-model.value(:));
A = eye(4)-K*H;
P_joseph = A*P*A' + K*R*K';
assert(relative_error(x_normal, x_joseph) < 1e-11);
assert(relative_error(P_normal, P_joseph) < 1e-11);
models_ut = TrackingModels;
[x_ekf, P_ekf, returned_model] = tracking_centralized_ekf_update(anchor, P, observations, network, env, models_ut);
[x_two, P_two] = tracking_centralized_ekf_update(anchor, P, observations, network, env, models_ut);
assert(relative_error(returned_model.anchor, anchor) < 1e-12);
assert(relative_error(returned_model.value, model.value) < 1e-12);
assert(relative_error(returned_model.H, model.H) < 1e-12);
assert(relative_error(x_two, x_ekf) < 1e-12 && relative_error(P_two, P_ekf) < 1e-12);
Report.CentralStateRelativeError = relative_error(x_ekf, x_normal);
Report.CentralCovarianceRelativeError = relative_error(P_ekf, P_normal);
assert(Report.CentralStateRelativeError < 1e-11 && Report.CentralCovarianceRelativeError < 1e-11, ...
    'Central EKF must equal the same frozen linearized MAP solution and Joseph covariance.');
[~, map_gradient] = TrackingADMMUtils.MAP(x_normal, scan_range, scan_doppler, {anchor}, {P}, model, Sigma);
Report.MAPStationarityNorm = norm(map_gradient);
assert(Report.MAPStationarityNorm < 1e-8);
env.PRE_WHITEN = true;
env.pre_whit_L = L;
[x_white, P_white, white_model] = tracking_centralized_ekf_update(anchor, P, observations, network, env, models_ut);
Report.CentralWhiteningRelativeError = max(relative_error(x_white,x_ekf), relative_error(P_white,P_ekf));
assert(Report.CentralWhiteningRelativeError < 1e-11);
assert(relative_error(white_model.value, model.value) < 1e-12 && relative_error(white_model.H, model.H) < 1e-12, ...
    'Returned model must remain physical and unwhitened.');
assert(isequaln(model, saved_model), 'Objective evaluation must preserve the frozen model.');
Report.Passed = true;
fprintf('EKF linearization checks passed: Jacobian %.2g; max gradient %.2g; central state/covariance %.2g / %.2g.\n', ...
    Report.JacobianRelativeError, max(Report.GradientRelativeErrors(:)), ...
    Report.CentralStateRelativeError, Report.CentralCovarianceRelativeError);
end

function y = physical_measurement(x, network, env)
displacement = x(1:2)' - network.radar_pos;
distance = sqrt(sum(displacement.^2, 2));
y = [distance'; (2*(displacement*x(3:4)) ./ (env.lambda*distance))'];
end

function y = affine_measurement(x, model)
y = model.value;
for j = 1:size(y, 2)
    y(:,j) = y(:,j) + model.H(:,:,j)*(x-model.anchor);
end
end

function [f, g, Hessian, rhs] = direct_likelihood(x, range, doppler, model, Sigma)
f = 0;
g = zeros(4,1);
Hessian = zeros(4);
rhs = zeros(4,1);
prediction = affine_measurement(x, model);
for j = 1:size(range, 2)
    H = model.H(:,:,j);
    for m = 1:size(range, 1)
        observation = [range(m,j); doppler(m,j)];
        residual = prediction(:,j)-observation;
        f = f + 0.5*residual'*(Sigma\residual);
        g = g + H'*(Sigma\residual);
        Hessian = Hessian + H'*(Sigma\H);
        rhs = rhs + H'*(Sigma\(observation-model.value(:,j)+H*model.anchor));
    end
end
end

function [f, g] = direct_prior(x, mu, sigma)
f = 0;
g = zeros(4,1);
for j = 1:numel(mu)
    delta = x-mu{j};
    f = f + 0.5*delta'*(sigma{j}\delta);
    g = g + sigma{j}\delta;
end
end

function [f, g] = direct_consensus(x, n, neighbors, Nu, z, c)
f = 0;
g = zeros(4,1);
for j = neighbors{n}
    delta = x-z{n}(:,j);
    f = f + Nu{n}(:,j)'*delta + 0.5*sum(c(:).*delta.^2);
    g = g + Nu{n}(:,j) + c(:).*delta;
end
end

function gradient = finite_gradient(objective, x)
gradient = zeros(size(x));
for d = 1:numel(x)
    step = 1e-6*max(1, abs(x(d)));
    delta = zeros(size(x));
    delta(d) = step;
    gradient(d) = (objective(x+delta)-objective(x-delta))/(2*step);
end
end

function value = relative_error(actual, expected)
value = norm(actual(:)-expected(:)) / max(1, norm(expected(:)));
end
