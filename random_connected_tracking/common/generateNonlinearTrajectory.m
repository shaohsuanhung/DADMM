function [t, states, info] = generateNonlinearTrajectory(opts)
%GENERATENONLINEARTRAJECTORY Smooth nonlinear 2D ground-truth trajectory.
%   states(k,:) = [x, y, vx, vy], with samples at t = 0:dt:T.
%   T must be an integer multiple of dt. Generate the full reference duration
%   and take a prefix for shorter runs, so noise and endpoint correction agree.
%   Options retain the names and defaults of the original ADMM-EKF utility.

arguments
    opts.T (1,1) double {mustBeReal, mustBeFinite, mustBePositive} = 30.0
    opts.dt (1,1) double {mustBeReal, mustBeFinite, mustBePositive} = 0.1
    opts.start_pos double = [0.0, 0.0]

    opts.initial_speed (1,1) double {mustBeReal, mustBeFinite, mustBeNonnegative} = 5.0
    opts.initial_heading_deg (1,1) double {mustBeReal, mustBeFinite} = 45.0
    opts.speed_amp (1,1) double {mustBeReal, mustBeFinite} = 0.4
    opts.speed_freq (1,1) double {mustBeReal, mustBeFinite, mustBeNonnegative} = 0.15
    opts.turn_amp_deg (1,1) double {mustBeReal, mustBeFinite} = 60.0
    opts.turn_freq (1,1) double {mustBeReal, mustBeFinite, mustBeNonnegative} = 0.08

    opts.seed (1,1) double {mustBeReal, mustBeFinite, mustBeInteger, mustBeNonnegative} = 42
    opts.use_randomness (1,1) logical = false
    opts.speed_noise_std (1,1) double {mustBeReal, mustBeFinite, mustBeNonnegative} = 0.0
    opts.heading_noise_std_deg (1,1) double {mustBeReal, mustBeFinite, mustBeNonnegative} = 0.0
    opts.smooth_noise_window (1,1) double {mustBeReal, mustBeFinite} = 15

    opts.enforce_endpoint (1,1) logical = false
    opts.end_pos double = [NaN, NaN]
end

validateattributes(opts.start_pos, {'double'}, ...
    {'real', 'finite', 'vector', 'numel', 2});
validateattributes(opts.end_pos, {'double'}, {'real', 'vector', 'numel', 2});
if opts.enforce_endpoint && any(~isfinite(opts.end_pos))
    error('NonlinearTrajectory:Endpoint', ...
        'A finite end_pos is required when enforce_endpoint is true.');
end
if opts.seed > 2^32 - 1
    error('NonlinearTrajectory:Seed', 'seed must be between 0 and 2^32-1.');
end

step_count = opts.T / opts.dt;
num_steps = round(step_count);
if ~isfinite(step_count) || num_steps < 1 || ...
        abs(step_count - num_steps) > 1e-10 * max(1, step_count)
    error('NonlinearTrajectory:Duration', ...
        'T must be a positive integer multiple of dt.');
end
t = (0:num_steps).' * opts.dt;
t(end) = opts.T;
K = numel(t);
tau = t / opts.T;

caller_rng = rng;
rng_cleanup = onCleanup(@() rng(caller_rng)); %#ok<NASGU>
rng(opts.seed, 'twister');

% Sinusoidal speed and heading profiles from the original utility.
speed = opts.initial_speed * ...
    (1 + opts.speed_amp * sin(2 * pi * opts.speed_freq * t));
heading = deg2rad(opts.initial_heading_deg) + ...
    deg2rad(opts.turn_amp_deg) * sin(2 * pi * opts.turn_freq * t);

if opts.use_randomness
    speed_noise = opts.speed_noise_std * randn(K, 1);
    heading_noise = deg2rad(opts.heading_noise_std_deg) * randn(K, 1);
    w = max(1, round(opts.smooth_noise_window));
    kernel = ones(w, 1) / w;
    speed = speed + conv(speed_noise, kernel, 'same');
    heading = heading + conv(heading_noise, kernel, 'same');
end

% Retain the original minimum speed and forward Euler position integration.
speed = max(speed, 0.1);
vx = speed .* cos(heading);
vy = speed .* sin(heading);
x = zeros(K, 1);
y = zeros(K, 1);
x(1) = opts.start_pos(1);
y(1) = opts.start_pos(2);
for k = 2:K
    x(k) = x(k-1) + vx(k-1) * opts.dt;
    y(k) = y(k-1) + vy(k-1) * opts.dt;
end

if opts.enforce_endpoint
    endpoint_error = opts.end_pos(:) - [x(end); y(end)];
    % Smoothstep preserves the start and both endpoint velocities. Its
    % analytic derivative supplies the original velocity correction.
    alpha = 3 * tau.^2 - 2 * tau.^3;
    alpha_dot = (6 * tau - 6 * tau.^2) / opts.T;
    x = x + alpha * endpoint_error(1);
    y = y + alpha * endpoint_error(2);
    vx = vx + alpha_dot * endpoint_error(1);
    vy = vy + alpha_dot * endpoint_error(2);
    x(end) = opts.end_pos(1);
    y(end) = opts.end_pos(2);
end

states = [x, y, vx, vy];
info.speed = hypot(vx, vy);
info.heading = atan2(vy, vx);
info.heading_deg = rad2deg(info.heading);
info.vx = vx;
info.vy = vy;
if opts.enforce_endpoint
    info.endpoint_error_before_correction = endpoint_error;
    info.desired_end_pos = opts.end_pos;
    info.actual_end_pos = states(end, 1:2);
end
end
