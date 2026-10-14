function target = make_tracking_truth(cfg, num_samples, dt)
%MAKE_TRACKING_TRUTH Shared selectable truth with a CV estimator.
% Generate the full reference once, then select the requested prefix.
% Shortening a run must not change either position or velocity samples.

validateattributes(num_samples, {'numeric'}, {'scalar', 'integer', 'positive'});
validateattributes(dt, {'numeric'}, {'scalar', 'positive', 'finite'});
validateattributes(cfg.Duration, {'numeric'}, {'scalar', 'positive', 'finite'});
num_intervals = round(cfg.Duration / dt);
if num_intervals < 1 || abs(num_intervals * dt - cfg.Duration) > 1e-9 * max(1, cfg.Duration)
    error('RadarTracking:TrajectoryDuration', 'Trajectory.Duration must be a positive integer multiple of TimeStep.');
end
num_reference_samples = num_intervals + 1;
if num_samples > num_reference_samples
    error('RadarTracking:TrajectoryTooShort', ...
        'Run needs %.3f s; increase Trajectory.Duration (currently %.3f s).', ...
        (num_samples - 1) * dt, cfg.Duration);
end

% Configurations saved before the selector was added use the reference path.
trajectory_type = "reference";
if isfield(cfg, 'Type')
    trajectory_type = string(validatestring(cfg.Type, {'reference', 'nonlinear'}));
end

switch trajectory_type
    case "reference"
        opts.seed = cfg.Seed;
        opts.randomizeShape = cfg.RandomizeShape;
        opts.M = cfg.NumWaypoints;
        opts.shapeJitterX = cfg.ShapeJitterX;
        opts.shapeJitterY = cfg.ShapeJitterY;
        opts.denseN = cfg.DenseSamples;
        opts.lateralStd = cfg.LateralStd;
        opts.lateralTau = cfg.LateralCorrelationTime;
        opts.longMode = cfg.LongMode;
        position = gen_ref_trajectory(num_reference_samples, cfg.Bounds, dt, cfg.Speed, opts);
        velocity = [gradient(position(:, 1), dt), gradient(position(:, 2), dt)];
        time = (0:num_intervals).' * dt;
        states = [position, velocity];
        generator = 'gen_ref_trajectory';
        motion_model = 'Original ADMM-EKF gen_ref_trajectory bounded curve';
        velocity_method = 'gradient(position, dt) on the complete reference before selecting a prefix';

    case "nonlinear"
        p = cfg.Nonlinear;
        [time, states] = generateNonlinearTrajectory( ...
            T=cfg.Duration, dt=dt, seed=cfg.Seed, ...
            start_pos=p.StartPosition, end_pos=p.EndPosition, ...
            initial_speed=p.InitialSpeed, initial_heading_deg=p.InitialHeadingDeg, ...
            speed_amp=p.SpeedAmplitude, speed_freq=p.SpeedFrequency, ...
            turn_amp_deg=p.TurnAmplitudeDeg, turn_freq=p.TurnFrequency, ...
            use_randomness=p.UseRandomness, speed_noise_std=p.SpeedNoiseStd, ...
            heading_noise_std_deg=p.HeadingNoiseStdDeg, ...
            smooth_noise_window=p.SmoothNoiseWindow, enforce_endpoint=p.EnforceEndpoint);
        position = states(:, 1:2);
        generator = 'generateNonlinearTrajectory';
        motion_model = 'ADMM-EKF sinusoidal speed/heading trajectory with optional endpoint correction';
        velocity_method = 'Speed/heading components plus analytic smoothstep endpoint correction; Euler position integration';
end

target.time = time(1:num_samples);
target.target_state = reshape(states(1:num_samples, :), [1, num_samples, 4]);
target.target_position = reshape(position(1:num_samples, :), [1, num_samples, 2]);
target.true_params = states(1, :);
target.initial_position = states(1, 1:2);
target.trajectory_seed = cfg.Seed;
target.trajectory_config = cfg;
target.trajectory_type = trajectory_type;
target.motion_model = motion_model;
target.reference.time = time;
target.reference.states = states;
target.reference.generator = generator;
target.reference.velocity_method = velocity_method;
end
