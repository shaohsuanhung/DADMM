function Report = check_batch_linearization()
%CHECK_BATCH_LINEARIZATION Small deterministic checks; no tracking run or files.
% Independently checks the physical derivative and frozen quadratic objectives.
project_root = fileparts(mfilename('fullpath'));
previous_path = path;
path_cleanup = onCleanup(@() path(previous_path));
addpath(fullfile(project_root, 'all_in_one'));

anchor = [0.35; -0.22; 1.8; -0.7];
radars = [0 0; 4 -2; -3 5; 5 6]; % First range is < 1 m: no range clipping.
lambda = 0.42;
N = size(radars, 1);
M = 3;
model = BatchADMMUtils.linearize_measurements(anchor, radars, lambda);
saved_model = model;
assert(isequal(size(model.anchor), [4 1]) && isequal(model.anchor, anchor));
assert(isequal(size(model.value), [2 N]) && isequal(size(model.H), [2 4 N]));
Report.AnchorError = relative_error(model.value, physical_measurement(anchor, radars, lambda));
assert(Report.AnchorError < 1e-12, 'The affine model must be exact at its anchor.');

% Differentiate an independent physical formula, including the Doppler sign.
numeric_H = zeros(2, 4, N);
for d = 1:4
    step = 1e-6 * max(1, abs(anchor(d)));
    delta = zeros(4, 1);
    delta(d) = step;
    derivative = (physical_measurement(anchor+delta, radars, lambda) - ...
        physical_measurement(anchor-delta, radars, lambda)) / (2*step);
    numeric_H(:,d,:) = reshape(derivative, [2 1 N]);
end
Report.JacobianRelativeError = relative_error(model.H, numeric_H);
assert(Report.JacobianRelativeError < 2e-7, 'Physical Jacobian finite difference failed.');

direction = [0.4; -0.3; 0.2; 0.5];
steps = [0.01 0.005 0.0025];
taylor_error = zeros(size(steps));
for k = 1:numel(steps)
    candidate = anchor + steps(k)*direction;
    error = physical_measurement(candidate, radars, lambda) - affine_measurement(candidate, model);
    taylor_error(k) = norm(error(:));
end
Report.TaylorErrorRatios = taylor_error(1:end-1) ./ taylor_error(2:end);
assert(all(Report.TaylorErrorRatios > 3.5 & Report.TaylorErrorRatios < 4.5), ...
    'Halving the perturbation must reduce Taylor error by about four.');
assert_throws(@() BatchADMMUtils.linearize_measurements([radars(1,:)'; 1; 2], radars, lambda));
assert_throws(@() BatchADMMUtils.linearize_measurements([radars(1,:)'+[5e-10;0]; 1; 2], radars, lambda));

% Correlated, unequal variances catch omitted off-diagonal terms and scaling.
Sigma = [0.16 0.054; 0.054 0.09];
range = repmat(model.value(1,:), M, 1) + 0.1*reshape(sin(1:M*N), M, N);
doppler = repmat(model.value(2,:), M, 1) + 0.2*reshape(cos(0.7*(1:M*N)), M, N);
x = anchor + [0.18; -0.11; 0.6; -0.25];
[f, gradient] = BatchADMMUtils.logLikelihood(x, range, doppler, model, Sigma);
[expected_f, expected_gradient, Hessian, rhs] = direct_likelihood(x, range, doppler, model, Sigma);
Report.LikelihoodRelativeError = relative_error(f, expected_f);
assert(Report.LikelihoodRelativeError < 1e-12, 'Likelihood must equal direct Mahalanobis loss.');
assert(relative_error(gradient, expected_gradient) < 1e-12, 'Likelihood gradient disagrees with direct formula.');

% Transform the original two-vector simultaneously, including its Jacobian.
L = chol(Sigma, 'lower') \ eye(2);
white_model = BatchADMMUtils.linearize_measurements(anchor, radars, lambda, L);
white_range = zeros(M, N);
white_doppler = zeros(M, N);
for j = 1:N
    assert(relative_error(white_model.value(:,j), L*model.value(:,j)) < 1e-12);
    assert(relative_error(white_model.H(:,:,j), L*model.H(:,:,j)) < 1e-12);
    transformed = L * [range(:,j)'; doppler(:,j)'];
    white_range(:,j) = transformed(1,:)';
    white_doppler(:,j) = transformed(2,:)';
end
[white_f, white_gradient] = BatchADMMUtils.logLikelihood(x, white_range, white_doppler, white_model, eye(2));
Report.WhiteningRelativeError = max(relative_error(f, white_f), relative_error(gradient, white_gradient));
assert(Report.WhiteningRelativeError < 1e-11, 'Prewhitening must preserve the loss and gradient.');

% Preserve and test the existing prior coefficient and squared penalty weights.
mu = cell(1, N);
sigma = cell(1, N);
Nu = cell(1, N);
z = cell(1, N);
neighbors = {[2 4], [1 3 4], 2, [1 2]};
n = 2;
c = [0.4 0.8 1.1 0.6];
for j = 1:N
    mu{j} = anchor + 0.04*j*[1; -2; 0.5; -1];
    v = [1; -0.5; 0.3; 0.2];
    sigma{j} = diag([0.4 0.6 0.8 1]+0.05*j) + 0.02*(v*v');
    Nu{j} = 0.2*reshape(sin((1:4*N)+j), 4, N);
    z{j} = repmat(anchor, 1, N) + 0.1*reshape(cos((1:4*N)+j), 4, N);
end
[prior_f, prior_g] = direct_prior(x, mu, sigma);
[penalty_f, penalty_g] = direct_consensus(x, n, neighbors, Nu, z, c);
objectives = {@(q) BatchADMMUtils.logLikelihood(q, range, doppler, model, Sigma), ...
    @(q) BatchADMMUtils.MAP(q, range, doppler, mu, sigma, model, Sigma), ...
    @(q) BatchADMMUtils.logLikelihoodWithConsensus(q, range, doppler, model, Sigma, n, neighbors, Nu, z, c), ...
    @(q) BatchADMMUtils.posteriorWithConsensus(q, range, doppler, mu, sigma, model, Sigma, n, neighbors, Nu, z, c)};
expected_values = [expected_f, expected_f+prior_f, expected_f+penalty_f, expected_f+prior_f+penalty_f];
expected_gradients = [expected_gradient, expected_gradient+prior_g, ...
    expected_gradient+penalty_g, expected_gradient+prior_g+penalty_g];
Report.ObjectiveNames = {'Likelihood', 'MAP', 'LikelihoodConsensus', 'PosteriorConsensus'};
Report.GradientRelativeErrors = zeros(1, numel(objectives));
for k = 1:numel(objectives)
    [value, analytic_gradient] = objectives{k}(x);
    assert(isequal(size(analytic_gradient), [4 1]), 'Gradients must be column vectors.');
    assert(relative_error(value, expected_values(k)) < 1e-11, 'Objective coefficient check failed.');
    assert(relative_error(analytic_gradient, expected_gradients(:,k)) < 1e-11, 'Analytic coefficient check failed.');
    numeric_gradient = finite_gradient(objectives{k}, x);
    Report.GradientRelativeErrors(k) = relative_error(analytic_gradient, numeric_gradient);
    assert(Report.GradientRelativeErrors(k) < 2e-6, 'Objective finite difference gradient failed.');
end

% This is an independently assembled frozen quadratic, not a nonlinear solve.
assert(min(eig(Hessian)) > 1e-8, 'Test geometry must identify all four state components.');
x_minimum = Hessian \ rhs;
[minimum_f, minimum_gradient] = objectives{1}(x_minimum);
Report.MinimumGradientNorm = norm(minimum_gradient);
assert(Report.MinimumGradientNorm < 1e-8, 'Closed-form quadratic minimizer must be stationary.');
offset = [0.2; -0.3; 0.4; -0.1];
[offset_f, offset_g] = objectives{1}(x_minimum+offset);
assert(relative_error(offset_f-minimum_f, 0.5*offset'*Hessian*offset) < 1e-10);
assert(relative_error(offset_g-minimum_gradient, Hessian*offset) < 1e-10, ...
    'The objective must retain the original anchor and constant Hessian.');
assert(isequaln(model, saved_model), 'Objective evaluation must not change the frozen model.');
Report.Passed = true;
fprintf('Batch linearization checks passed: Jacobian %.2g; max gradient error %.2g; whitening %.2g.\n', ...
    Report.JacobianRelativeError, max(Report.GradientRelativeErrors), Report.WhiteningRelativeError);
end

function y = physical_measurement(x, radars, lambda)
displacement = x(1:2)' - radars;
distance = sqrt(sum(displacement.^2, 2));
y = [distance'; ((displacement*x(3:4)) ./ (lambda*distance))'];
end

function y = affine_measurement(x, model)
y = model.value;
for j = 1:size(y, 2)
    y(:,j) = y(:,j) + model.H(:,:,j)*(x-model.anchor);
end
end

function [f, g, Hessian, rhs] = direct_likelihood(x, range, doppler, model, Sigma)
f = 0;
g = zeros(4, 1);
Hessian = zeros(4);
rhs = zeros(4, 1);
predicted = affine_measurement(x, model);
for j = 1:size(range, 2)
    H = model.H(:,:,j);
    for m = 1:size(range, 1)
        y = [range(m,j); doppler(m,j)];
        residual = predicted(:,j)-y;
        f = f + 0.5*residual'*(Sigma\residual);
        g = g + H'*(Sigma\residual);
        Hessian = Hessian + H'*(Sigma\H);
        rhs = rhs + H'*(Sigma\(y-model.value(:,j)+H*model.anchor));
    end
end
end

function [f, g] = direct_prior(x, mu, sigma)
f = 0;
g = zeros(4, 1);
for j = 1:numel(mu)
    delta = x-mu{j};
    coefficient = 1 / (numel(mu)*sqrt((2*pi)^4*det(sigma{j})));
    f = f + coefficient*delta'*(sigma{j}\delta);
    g = g + 2*coefficient*(sigma{j}\delta);
end
end

function [f, g] = direct_consensus(x, n, neighbors, Nu, z, c)
f = 0;
g = zeros(4, 1);
for j = neighbors{n}
    delta = x-z{n}(:,j);
    f = f + Nu{n}(:,j)'*delta + sum(((c(:)/2).*delta).^2);
    g = g + Nu{n}(:,j) + (c(:).^2/2).*delta;
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

function assert_throws(action)
did_throw = false;
try
    action();
catch
    did_throw = true;
end
assert(did_throw, 'A linearization anchor at or within 1e-9 m of a radar must be rejected.');
end
