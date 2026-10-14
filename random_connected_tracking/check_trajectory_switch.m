function check_trajectory_switch()
%CHECK_TRAJECTORY_SWITCH Check both truth choices without running estimators.
project_root = fileparts(mfilename('fullpath'));
previous_path = path;
path_cleanup = onCleanup(@() path(previous_path)); %#ok<NASGU>
addpath(fullfile(project_root, 'common'), fullfile(project_root, 'all_in_one'), ...
    fullfile(project_root, 'ADMM_EKF'));
before_rng = rng;
cfg = trajectory_config();
dt = 0.01;
num_samples = round(cfg.Duration / dt) + 1;
assert(strcmp(cfg.Type, 'reference'), 'The default must retain the reference path.');

% Compare with the established standalone reference generator and velocity
% rule, rather than comparing two calls through the new dispatcher alone.
reference = make_tracking_truth(cfg, num_samples, dt);
opts = struct('seed', cfg.Seed, 'randomizeShape', cfg.RandomizeShape, ...
    'M', cfg.NumWaypoints, 'shapeJitterX', cfg.ShapeJitterX, ...
    'shapeJitterY', cfg.ShapeJitterY, 'denseN', cfg.DenseSamples, ...
    'lateralStd', cfg.LateralStd, 'lateralTau', cfg.LateralCorrelationTime, ...
    'longMode', cfg.LongMode);
position = gen_ref_trajectory(num_samples, cfg.Bounds, dt, cfg.Speed, opts);
expected = [position, gradient(position(:, 1), dt), gradient(position(:, 2), dt)];
assert(isequal(reference.reference.states, expected), ...
    'Selecting reference must preserve the original reference numerics.');
assert(strcmp(reference.reference.generator, 'gen_ref_trajectory'));
assert(strcmp(reference.trajectory_type, 'reference'));
check_prefixes(cfg, reference, dt);

% An old saved configuration has neither the selector nor nonlinear options.
legacy_cfg = rmfield(cfg, {'Type', 'Nonlinear'});
legacy = make_tracking_truth(legacy_cfg, num_samples, dt);
assert(isequal(legacy.target_state, reference.target_state));
assert(isequal(legacy.target_position, reference.target_position));
assert(isequal(legacy.time, reference.time));
assert(isequal(legacy.reference, reference.reference));
assert(strcmp(legacy.trajectory_type, 'reference'));

cfg.Type = 'nonlinear';
nonlinear = make_tracking_truth(cfg, num_samples, dt);
states = nonlinear.reference.states;
assert(strcmp(nonlinear.reference.generator, 'generateNonlinearTrajectory'));
assert(strcmp(nonlinear.trajectory_type, 'nonlinear'));
assert(isequal(states(1, 1:2), cfg.Nonlinear.StartPosition));
assert(isequal(states(end, 1:2), cfg.Nonlinear.EndPosition));
assert(all(isfinite(states(:))), 'Nonlinear states must remain finite.');
assert(rank(states(:, 1:2) - states(1, 1:2), 1e-8) == 2, ...
    'The nonlinear path must curve.');
assert(any(vecnorm(diff(states(:, 3:4)), 2, 2) > 1e-6), ...
    'Nonlinear truth must supply varying per-sample velocities.');
assert(~isequal(states, reference.reference.states), ...
    'The selector must change the generated truth.');
check_prefixes(cfg, nonlinear, dt);

% With no noise or endpoint correction, speed/heading have closed forms and
% position increments use the previous sample's velocity (forward Euler).
raw_cfg = cfg;
raw_cfg.Nonlinear.UseRandomness = false;
raw_cfg.Nonlinear.EnforceEndpoint = false;
raw = make_tracking_truth(raw_cfg, num_samples, dt);
p = raw_cfg.Nonlinear;
time = (0:num_samples-1).' * dt;
speed = max(p.InitialSpeed * (1 + p.SpeedAmplitude * ...
    sin(2*pi*p.SpeedFrequency*time)), 0.1);
heading = deg2rad(p.InitialHeadingDeg) + deg2rad(p.TurnAmplitudeDeg) * ...
    sin(2*pi*p.TurnFrequency*time);
velocity = [speed .* cos(heading), speed .* sin(heading)];
expected_position = p.StartPosition + [zeros(1, 2); cumsum(dt * velocity(1:end-1, :), 1)];
assert_close(raw.reference.states(:, 3:4), velocity, 1e-11, ...
    'Uncorrected velocity must follow the analytic speed/heading profile.');
assert_close(raw.reference.states(:, 1:2), expected_position, 1e-9, ...
    'Uncorrected positions must follow forward Euler integration.');
assert_close(diff(raw.reference.states(:, 1:2)), dt * velocity(1:end-1, :), 1e-11, ...
    'Euler position increments must use the previous sample velocity.');

% Endpoint position and velocity corrections have different analytic forms.
% In particular, the corrected velocity is not a finite-difference gradient.
tau = time / cfg.Duration;
endpoint_error = p.EndPosition - raw.reference.states(end, 1:2);
alpha = 3*tau.^2 - 2*tau.^3;
alpha_dot = (6*tau - 6*tau.^2) / cfg.Duration;
expected_corrected = raw.reference.states + [alpha*endpoint_error, alpha_dot*endpoint_error];
assert_close(states, expected_corrected, 1e-10, ...
    'Endpoint correction must retain the original smoothstep and analytic derivative.');
assert(isequal(states([1 end], 3:4), raw.reference.states([1 end], 3:4)), ...
    'Smoothstep correction must preserve both endpoint velocities.');

% The stochastic path must be repeatable without consuming the caller RNG.
noisy_cfg = cfg;
noisy_cfg.Nonlinear.SpeedNoiseStd = 1.2;
noisy_cfg.Nonlinear.HeadingNoiseStdDeg = 3;
noisy = make_tracking_truth(noisy_cfg, num_samples, dt);
repeated = make_tracking_truth(noisy_cfg, num_samples, dt);
assert(isequal(noisy.reference.states, repeated.reference.states));
assert(~isequal(noisy.reference.states, states), 'Nonzero noise must affect the path.');
assert(isequal(noisy.reference.states(1, 1:2), p.StartPosition));
assert(isequal(noisy.reference.states(end, 1:2), p.EndPosition));
check_prefixes(noisy_cfg, noisy, dt);
different_cfg = noisy_cfg;
different_cfg.Seed = noisy_cfg.Seed + 1;
different = make_tracking_truth(different_cfg, num_samples, dt);
assert(~isequal(noisy.reference.states, different.reference.states), ...
    'Different trajectory seeds must change a stochastic path.');
noisy_cfg.Nonlinear.UseRandomness = false;
disabled = make_tracking_truth(noisy_cfg, num_samples, dt);
assert(isequal(disabled.reference.states, states), ...
    'UseRandomness=false must ignore nonzero noise settings.');

% Check both branches and scales. Poison the legacy initial-velocity field
% to make accidental reuse of that constant visible even at the first sample.
check_doppler(reference);
check_doppler(nonlinear);

bad_cfg = cfg;
bad_cfg.Type = 'unsupported';
assert_throws(@() make_tracking_truth(bad_cfg, 8, dt), 'Unknown trajectory type');
for duration = [-1, 0, 1.5*dt]
    bad_cfg = cfg;
    bad_cfg.Duration = duration;
    assert_throws(@() make_tracking_truth(bad_cfg, 1, dt), 'Invalid duration');
end
assert_throws(@() make_tracking_truth(cfg, num_samples+1, dt), 'Reference too short');
assert_throws(@() make_tracking_truth(cfg, 0, dt), 'Zero samples');
assert_throws(@() make_tracking_truth(cfg, 1.5, dt), 'Fractional samples');
assert_throws(@() make_tracking_truth(cfg, 8, 0), 'Zero time step');
assert(isequal(before_rng, rng), 'Trajectory checks must leave the caller RNG unchanged.');
fprintf(['Trajectory switch checks passed: legacy reference equivalence, nonlinear ', ...
    'Euler/smoothstep states, full-duration prefixes, seeded noise, Doppler and guards.\n']);
end

function check_prefixes(cfg, full, dt)
num_reference_samples = size(full.reference.states, 1);
assert(isequal(size(full.reference.states), [num_reference_samples, 4]));
assert(isequal(size(full.reference.time), [num_reference_samples, 1]));
assert(full.reference.time(1) == 0 && full.reference.time(end) == cfg.Duration);
for count = [1, 2, 8, 64, num_reference_samples]
    prefix = make_tracking_truth(cfg, count, dt);
    assert(isequal(prefix.reference, full.reference), ...
        'Short runs must retain the complete fixed-duration reference.');
    assert(isequal(size(prefix.target_state), [1, count, 4]));
    assert(isequal(size(prefix.target_position), [1, count, 2]));
    assert(isequal(prefix.target_state, full.target_state(:, 1:count, :)), ...
        'Short states must be an exact prefix of the full reference.');
    assert(isequal(prefix.target_position, prefix.target_state(:, :, 1:2)));
    assert(isequal(prefix.time, full.reference.time(1:count)));
    assert(isequal(prefix.true_params, full.reference.states(1, :)));
    assert(isequal(prefix.initial_position, full.reference.states(1, 1:2)));
    assert(isequal(prefix.trajectory_seed, cfg.Seed));
    assert(isequal(prefix.trajectory_config, cfg));
end
end

function check_doppler(full)
indices = [1, 2, 8, 64, round(size(full.reference.states, 1)/2)];
states = full.reference.states(indices, :);
count = numel(indices);
target = full;
target.target_state = reshape(states, [1, count, 4]);
target.target_position = reshape(states(:, 1:2), [1, count, 2]);
target.true_params(3:4) = [12345, -54321];
network.numNodes = 2;
network.radar_pos = [0, 0; 20, 0];
env.lambda = 0.03;
[batch_range, batch_doppler, batch_measurements] = BatchADMMUtils.gt_data_generation_tracking( ...
    zeros(1, 2, count), zeros(1, 2, count), zeros(1, 2, 2*count), target, network, env, count, 1);
[ekf_range, ekf_doppler, ekf_measurements] = TrackingADMMUtils.gt_data_generation_tracking_ekf( ...
    zeros(1, 2, count), zeros(1, 2, count), zeros(1, 2, 2*count), target, network, env, count, 1);
for k = 1:count
    for n = 1:network.numNodes
        delta = states(k, 1:2) - network.radar_pos(n, :);
        range = norm(delta);
        expected = dot(states(k, 3:4), delta) / (range * env.lambda);
        assert_close(batch_range(1, n, k), range, 1e-10, 'Batch range mismatch.');
        assert_close(ekf_range(1, n, k), range, 1e-10, 'EKF range mismatch.');
        assert_close(batch_doppler(1, n, k), expected, 1e-9, 'Batch must retain 1/lambda.');
        assert_close(ekf_doppler(1, n, k), 2*expected, 1e-9, 'EKF must retain 2/lambda.');
    end
end
assert(isequal(batch_measurements(:, :, 1:2:end), batch_range));
assert(isequal(batch_measurements(:, :, 2:2:end), batch_doppler));
assert(isequal(ekf_measurements(:, :, 1:2:end), ekf_range));
assert(isequal(ekf_measurements(:, :, 2:2:end), ekf_doppler));
end

function assert_close(actual, expected, tolerance, message)
assert(isequal(size(actual), size(expected)), message);
assert(all(abs(actual(:) - expected(:)) <= tolerance), message);
end

function assert_throws(action, label)
threw = false;
try
    action();
catch
    threw = true;
end
assert(threw, '%s must be rejected.', label);
end
