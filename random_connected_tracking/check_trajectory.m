function check_trajectory()
%CHECK_TRAJECTORY Check shared truth independently of estimator convergence.
project_root = fileparts(mfilename('fullpath'));
previous_path = path;
cleanup = onCleanup(@() path(previous_path));
addpath(fullfile(project_root, 'common'), fullfile(project_root, 'all_in_one'), ...
    fullfile(project_root, 'ADMM_EKF'));
cfg = trajectory_config();
dt = 0.01;
num_samples = round(cfg.Duration / dt) + 1;
before_rng = rng;
target = make_tracking_truth(cfg, num_samples, dt);
states = reshape(target.target_state, num_samples, 4);
assert(all(states(:, 1) >= cfg.Bounds(1) & states(:, 1) <= cfg.Bounds(2)));
assert(all(states(:, 2) >= cfg.Bounds(3) & states(:, 2) <= cfg.Bounds(4)));
assert(rank(states(:, 1:2) - states(1, 1:2), 1e-8) == 2, 'Reference must curve.');
assert(any(vecnorm(diff(states(:, 3:4)), 2, 2) > 1e-6), 'Truth velocity must vary.');
for prefix_length = [1 2 8 64]
    prefix = make_tracking_truth(cfg, prefix_length, dt);
    assert(isequal(prefix.target_state, target.target_state(:, 1:prefix_length, :)), ...
        'Short runs must use a prefix, without compressing the full curve.');
end

% Velocities are derived on the full curve, before selecting any short run.
assert(isequal(states(:, 3), gradient(states(:, 1), dt)));
assert(isequal(states(:, 4), gradient(states(:, 2), dt)));
% With lateral wandering disabled the original baseline traverses once,
% remains monotone and reaches the bounding-box corners in clip mode.
cfg.LateralStd = 0;
base = make_tracking_truth(cfg, num_samples, dt);
base_states = reshape(base.target_state, num_samples, 4);
assert(all(diff(base_states(:, 1:2)) >= -1e-10, 'all'));
assert(norm(base_states(1, 1:2) - cfg.Bounds([1 3])) < 1e-10);
assert(norm(base_states(end, 1:2) - cfg.Bounds([2 4])) < 1e-10);

% Seeded curve jitter and lateral wandering preserve the caller RNG.
cfg.LateralStd = 0.25;
first = make_tracking_truth(cfg, 64, dt);
second = make_tracking_truth(cfg, 64, dt);
assert(isequal(first.target_state, second.target_state));
assert(isequal(before_rng, rng));
cfg.Seed = cfg.Seed + 1;
different = make_tracking_truth(cfg, 64, dt);
assert(~isequal(first.target_state, different.target_state));
% The standalone helper supports both documented bounds formats.
matrix_bounds = [cfg.Bounds(1:2); cfg.Bounds(3:4)];
vector_pos = gen_ref_trajectory(64, cfg.Bounds, dt, cfg.Speed, struct('seed', cfg.Seed));
matrix_pos = gen_ref_trajectory(64, matrix_bounds, dt, cfg.Speed, struct('seed', cfg.Seed));
assert(isequal(vector_pos, matrix_pos));

% Both generators must read per-sample velocity; keep their original scales.
test_target = make_tracking_truth(trajectory_config(), 4, dt);
network.numNodes = 2;
network.radar_pos = [0 0; 20 0];
env.lambda = 0.03;
[~, batch_doppler] = BatchADMMUtils.gt_data_generation_tracking( ...
    zeros(1, 2, 4), zeros(1, 2, 4), zeros(1, 2, 8), test_target, network, env, 4, 1);
[~, ekf_doppler] = TrackingADMMUtils.gt_data_generation_tracking_ekf( ...
    zeros(1, 2, 4), zeros(1, 2, 4), zeros(1, 2, 8), test_target, network, env, 4, 1);
for k = 1:4
    state = reshape(test_target.target_state(1, k, :), 4, 1);
    for n = 1:2
        delta = state(1:2) - network.radar_pos(n, :).';
        expected = dot(state(3:4), delta) / (norm(delta) * env.lambda);
        assert(abs(batch_doppler(1, n, k) - expected) < 1e-10);
        assert(abs(ekf_doppler(1, n, k) - 2*expected) < 1e-10);
    end
end
fprintf('Trajectory checks passed: bounded gen_ref path, shared prefixes, gradient velocity, Doppler and RNG isolation.\n');
end
