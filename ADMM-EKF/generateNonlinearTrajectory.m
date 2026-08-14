% function [t, states, info] = generateNonlinearTrajectory(opts)
% % generateNonlinearTrajectory
% %
% % Generate a reproducible nonlinear 2D target trajectory.
% %
% % State format:
% %   states(k, :) = [x, y, vx, vy]
% %
% % Outputs:
% %   t      : K x 1 time vector
% %   states : K x 4 state matrix [x, y, vx, vy]
% %   info   : struct containing speed, heading, acceleration, turn_rate
% 
% arguments
%     opts.T double = 30.0                      % total duration [s]
%     opts.dt double = 0.1                      % sampling interval [s]
%     opts.start_pos double = [0.0, 0.0]        % initial position [x0, y0]
%     opts.initial_speed double = 5.0           % initial speed [m/s]
%     opts.initial_heading_deg double = 45.0    % initial heading [deg]
%     opts.seed double = 42                     % random seed
% 
%     opts.nonlinear_turn_strength double = 1.0
%     opts.nonlinear_accel_strength double = 1.0
% 
%     opts.random_accel_std double = 0.2        % random acceleration std
%     opts.random_turn_std_deg double = 2.0     % random turn-rate std [deg/s]
% end
% 
% % Set random seed for reproducibility
% rng(opts.seed);
% 
% % Time vector
% t = (opts.dt:opts.dt:opts.T).';
% K = length(t);
% 
% % Allocate memory
% x = zeros(K, 1);
% y = zeros(K, 1);
% vx = zeros(K, 1);
% vy = zeros(K, 1);
% 
% speed = zeros(K, 1);
% heading = zeros(K, 1);
% 
% % Initial condition
% x(1) = opts.start_pos(1);
% y(1) = opts.start_pos(2);
% 
% speed(1) = opts.initial_speed;
% heading(1) = opts.initial_heading_deg * pi / 180;
% 
% vx(1) = speed(1) * cos(heading(1));
% vy(1) = speed(1) * sin(heading(1));
% 
% % Smooth nonlinear acceleration profile
% base_accel = opts.nonlinear_accel_strength * ...
%     (0.8 * sin(2 * pi * 0.08 * t) + ...
%      0.4 * sin(2 * pi * 0.17 * t + 0.8));
% 
% % Smooth nonlinear turn-rate profile
% base_turn_rate = opts.nonlinear_turn_strength * ...
%     (0.35 * sin(2 * pi * 0.05 * t) + ...
%      0.20 * sin(2 * pi * 0.11 * t + 1.2));
% 
% % Randomness
% random_accel = opts.random_accel_std * randn(K, 1);
% random_turn = (opts.random_turn_std_deg * pi / 180) * randn(K, 1);
% 
% accel = base_accel + random_accel;
% turn_rate = base_turn_rate + random_turn;
% 
% % Generate trajectory
% for k = 2:K
% 
%     % Update speed
%     speed(k) = speed(k-1) + accel(k-1) * opts.dt;
% 
%     % Avoid negative speed
%     speed(k) = max(speed(k), 0.1);
% 
%     % Update heading
%     heading(k) = heading(k-1) + turn_rate(k-1) * opts.dt;
% 
%     % Convert speed and heading to velocity
%     vx(k) = speed(k) * cos(heading(k));
%     vy(k) = speed(k) * sin(heading(k));
% 
%     % Integrate position
%     x(k) = x(k-1) + vx(k) * opts.dt;
%     y(k) = y(k-1) + vy(k) * opts.dt;
% end
% 
% % State matrix
% states = [x, y, vx, vy];
% 
% % Additional information
% info.speed = speed;
% info.heading = heading;
% info.accel = accel;
% info.turn_rate = turn_rate;
% 
% end

function [t, states, info] = generateNonlinearTrajectory(opts)
% generateNonlinearTrajectory
%
% Generate a smooth nonlinear 2D target trajectory.
%
% State format:
%   states(k, :) = [x, y, vx, vy]
%
% This version supports:
%   - sampling interval dt
%   - total duration T
%   - specified start point
%   - optional specified end point
%   - optional smooth randomness

arguments
    opts.T double = 30.0                       % total duration [s]
    opts.dt double = 0.1                       % sampling interval [s]
    opts.start_pos double = [0.0, 0.0]         % initial position [x0, y0]

    opts.initial_speed double = 5.0            % initial speed [m/s]
    opts.initial_heading_deg double = 45.0     % initial heading [deg]

    opts.speed_amp double = 0.4                % speed variation amplitude
    opts.speed_freq double = 0.15              % speed oscillation frequency [Hz]

    opts.turn_amp_deg double = 60.0            % heading variation amplitude [deg]
    opts.turn_freq double = 0.08               % heading oscillation frequency [Hz]

    opts.seed double = 42                      % random seed
    opts.use_randomness logical = false        % true or false

    opts.speed_noise_std double = 0.0          % speed noise std [m/s]
    opts.heading_noise_std_deg double = 0.0    % heading noise std [deg]

    opts.smooth_noise_window double = 15       % smoothing window for random noise

    opts.enforce_endpoint logical = false      % whether to force final position
    opts.end_pos double = [NaN, NaN]           % desired final position [xT, yT]
end

% Set random seed
rng(opts.seed);

% Time vector
t = (opts.dt:opts.dt:opts.T).';
K = length(t);

% Normalized time, from 0 to 1
tau = t / opts.T;

% Nominal nonlinear speed profile
speed = opts.initial_speed * ...
    (1.0 + opts.speed_amp * sin(2 * pi * opts.speed_freq * t));

% Nominal nonlinear heading profile
heading = deg2rad(opts.initial_heading_deg) + ...
    deg2rad(opts.turn_amp_deg) * sin(2 * pi * opts.turn_freq * t);

% Optional smooth randomness
if opts.use_randomness

    speed_noise = opts.speed_noise_std * randn(K, 1);
    heading_noise = deg2rad(opts.heading_noise_std_deg) * randn(K, 1);

    % Smooth the noise to avoid unrealistic jagged motion
    w = max(1, round(opts.smooth_noise_window));
    kernel = ones(w, 1) / w;

    speed_noise = conv(speed_noise, kernel, 'same');
    heading_noise = conv(heading_noise, kernel, 'same');

    speed = speed + speed_noise;
    heading = heading + heading_noise;
end

% Avoid negative or zero speed
speed = max(speed, 0.1);

% Velocity components
vx = speed .* cos(heading);
vy = speed .* sin(heading);

% Integrate position
x = zeros(K, 1);
y = zeros(K, 1);

x(1) = opts.start_pos(1);
y(1) = opts.start_pos(2);

for k = 2:K
    x(k) = x(k-1) + vx(k-1) * opts.dt;
    y(k) = y(k-1) + vy(k-1) * opts.dt;
end

% ============================================================
% Enforce endpoint if requested
% ============================================================
if opts.enforce_endpoint

    if any(isnan(opts.end_pos))
        error('If enforce_endpoint = true, you must provide opts.end_pos.');
    end

    desired_end = opts.end_pos(:);
    current_end = [x(end); y(end)];

    endpoint_error = desired_end - current_end;

    % Smoothstep correction:
    % alpha(0) = 0, alpha(1) = 1
    % alpha_dot(0) = 0, alpha_dot(1) = 0
    %
    % This preserves the starting point and smoothly shifts the path
    % so that the final point exactly matches end_pos.
    alpha = 3 * tau.^2 - 2 * tau.^3;

    % Derivative of alpha with respect to time
    alpha_dot = (6 * tau - 6 * tau.^2) / opts.T;

    % Correct position
    x = x + alpha * endpoint_error(1);
    y = y + alpha * endpoint_error(2);

    % Correct velocity consistently
    vx = vx + alpha_dot * endpoint_error(1);
    vy = vy + alpha_dot * endpoint_error(2);

end

% State matrix
states = [x, y, vx, vy];

% Extra information
info.speed = sqrt(vx.^2 + vy.^2);
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