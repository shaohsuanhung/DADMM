function cfg = trajectory_config()
%TRAJECTORY_CONFIG Shared truth settings for both original ADMM-EKF paths.
% This is the ground-truth path, independent of the estimator's CV dynamics.
% Keep Duration fixed when shortening a run: short runs use a prefix.
cfg.Type = "reference";            % "reference" or "nonlinear".
cfg.Duration = 30;                 % Full reference duration [s], independent of run length.
cfg.Seed = 42;                     % Local trajectory RNG; independent of measurement noise.

% gen_ref_trajectory: bounded reference curve with optional wandering.
cfg.Bounds = [-30, 20, -30, 25];   % [xmin xmax ymin ymax], original v5 call.
cfg.Speed = 50;                    % Nominal speed [m/s]; longMode="clip" can reduce it.
cfg.RandomizeShape = true;
cfg.NumWaypoints = 18;
cfg.ShapeJitterX = 0.012;
cfg.ShapeJitterY = 0.012;           % Actual ADMM-EKF implementation default.
cfg.DenseSamples = 4000;
cfg.LateralStd = 0.25;             % Correlated lateral wandering [m].
cfg.LateralCorrelationTime = 1.5;  % Seconds.
cfg.LongMode = "clip";             % Traverse the curve once if the request is too long.

% generateNonlinearTrajectory: original v5 sinusoidal speed/heading example.
% This path is not clipped to Bounds. Endpoint correction can change speed.
cfg.Nonlinear.StartPosition = [-30, -30];
cfg.Nonlinear.EndPosition = [20, 20];
cfg.Nonlinear.InitialSpeed = 50;              % Nominal speed before endpoint correction [m/s].
cfg.Nonlinear.InitialHeadingDeg = 45;
cfg.Nonlinear.SpeedAmplitude = 0.4;           % Fraction of InitialSpeed.
cfg.Nonlinear.SpeedFrequency = 0.15;          % Hz.
cfg.Nonlinear.TurnAmplitudeDeg = 50;
cfg.Nonlinear.TurnFrequency = 0.1;            % Hz.
cfg.Nonlinear.UseRandomness = true;
cfg.Nonlinear.SpeedNoiseStd = 0;              % m/s; zero matches the original v5 example.
cfg.Nonlinear.HeadingNoiseStdDeg = 0;
cfg.Nonlinear.SmoothNoiseWindow = 15;         % Samples.
cfg.Nonlinear.EnforceEndpoint = true;
end
