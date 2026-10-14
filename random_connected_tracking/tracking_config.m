function cfg = tracking_config(profile)
%TRACKING_CONFIG Editable experiment settings; no simulation is run here.
% "quick": short exploratory run. "original": original iteration/time limits.
% "smoke": tiny execution check, deliberately not a convergence experiment.

if nargin == 0
    profile = "quick";
end
profile = validatestring(profile, {'quick', 'original', 'smoke'});
project_root = fileparts(mfilename('fullpath'));
previous_path = path;
cleanup = onCleanup(@() path(previous_path));
addpath(fullfile(project_root, 'all_in_one'), fullfile(project_root, 'ADMM_EKF'), ...
    fullfile(project_root, 'common'));

cfg.Profile = string(profile);
cfg.Graph.Mode = "random_connected"; % "ring" connects each node to its predecessor/successor.
cfg.Graph.NumNodes = 10;
cfg.Graph.EdgeProbability = 0.35;
cfg.Graph.Seed = 20261013;
cfg.Graph.MaxAttempts = 10000;

% Shared physical radar layout, independent of communication edges.
cfg.Radar.Mode = "random_disk";  % "circle" restores evenly spaced positions.
cfg.Radar.Radius = 30;           % Deployment region radius [m].
cfg.Radar.Center = [0, 0];       % [x, y] in meters.
cfg.Radar.Seed = 20261014;       % Change only this seed to move the radars.
cfg.Batch = batch_default_config();
cfg.EKF = ekf_default_config();
cfg.Trajectory = trajectory_config(); % Shared nonlinear truth, separate from CV prediction.

cfg.SaveResults = true;
cfg.MakePlots = true;
cfg.FigureVisible = false;
cfg.Plot = tracking_plot_config(); % Fonts, markers, sizes and manuscript exports.
cfg.OutputRoot = fullfile(project_root, 'results');

switch profile
    case 'quick'
        cfg.Batch.NumMonteCarlo = 1;
        cfg.Batch.NumBatches = 3;
        cfg.Batch.SamplesPerBatch = 16;
        cfg.Batch.MaxIterations = 100;
        cfg.EKF.NumMonteCarlo = 1;
        cfg.EKF.NumCPIPerMeasurement = 8;
        cfg.EKF.TrackTime = 1;
        cfg.EKF.ADMMMaxIterations = 100;
    case 'smoke'
        cfg.Batch.NumMonteCarlo = 1;
        cfg.Batch.NumBatches = 1;
        cfg.Batch.SamplesPerBatch = 4;
        cfg.Batch.MaxIterations = 5;
        cfg.EKF.NumMonteCarlo = 1;
        cfg.EKF.NumCPIPerMeasurement = 2;
        cfg.EKF.TrackTime = 1;
        cfg.EKF.ADMMMaxIterations = 5;
        cfg.EKF.Verbose = false;
end
end
