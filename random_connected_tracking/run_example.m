% Start here: change METHOD, PROFILE and TRAJECTORY, then press Run.
% Methods: "MAP", "MLE", "EKF". Profiles: "quick", "original", "smoke".
METHOD = "EKF";
PROFILE = "original";
TRAJECTORY = "nonlinear"; % "original", "smooth", "another", or "nonlinear".

cfg = tracking_config(PROFILE);
cfg.Graph.Mode = "random_connected";             % "ring" or "random_connected".
cfg.Graph.NumNodes = 10;
cfg.Graph.EdgeProbability = 0.35;     % Used only for random_connected.
cfg.Graph.Seed = 20261013;            % Used only for random_connected.

% Circle + ring: each radar connects to the two nearest nodes, left and right.
cfg.Radar.Mode = "random_disk";       % "random_disk" randomizes positions independently.
cfg.Radar.Radius = 30;            % Deployment region radius [m].
cfg.Radar.Center = [0, 0];
cfg.Radar.Seed = 20261014;        % Used only for random_disk; edges stay fixed.

% Shared truth selection; all estimators still use CV prediction.
cfg.Trajectory.Duration = 30;      % Fixed reference horizon [s]; quick uses a prefix.
cfg.Trajectory.Bounds = [-30, 20, -30, 25];

switch TRAJECTORY
    case "original"
        cfg.Trajectory.Type = "reference"; % Original curve with lateral wandering.
    case "smooth"
        cfg.Trajectory.Type = "reference";
        cfg.Trajectory.RandomizeShape = false;
        cfg.Trajectory.LateralStd = 0;     % Smooth curve, not a straight line.
    case "another"
        cfg.Trajectory.Type = "reference";
        cfg.Trajectory.Seed = 43;          % Try 44, 45, ... for other curves.
    case "nonlinear"
        cfg.Trajectory.Type = "nonlinear"; % Sinusoidal speed and heading; optional endpoint.
    otherwise
        error('Unknown trajectory: %s', TRAJECTORY);
end
% Nonlinear-path examples (only used when TRAJECTORY = "nonlinear"):
% cfg.Trajectory.Nonlinear.InitialSpeed = 50;
% cfg.Trajectory.Nonlinear.TurnAmplitudeDeg = 50;
% cfg.Trajectory.Nonlinear.EndPosition = [20, 20];
% cfg.EKF.ProcessAccelerationVariance = 0; % CV process-noise tuning, not truth motion.

% CV-PCRLB reference; saved in Run.Results.PCRLB_Bayes.
cfg.EKF.CalculatePCRLB = true;
cfg.EKF.PCRLBNumStateSamples = 2000; % State samples for expected measurement information.

% Other examples:
% cfg.Batch.SNRdB = 40;
% cfg.EKF.SNRdB = 40;
% cfg.Batch.MaxIterations = 1000;
% cfg.EKF.ADMMMaxIterations = 2000;
% cfg.MakePlots = false;

Run = run_tracking(METHOD, cfg);
