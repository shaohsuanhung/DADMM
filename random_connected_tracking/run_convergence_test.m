% Longer convergence test for the current random-layout EKF entry.
% Keep graph, radar positions, truth, CV model and stopping tolerances fixed.
% This entry now solves fixed first-order consensus MAP, with a central EKF reference.
% Previously saved nonlinear runs require their original source to reproduce.
cfg = tracking_config("original");
cfg.Profile = "convergence_long";

cfg.EKF.NumMonteCarlo = 2;
cfg.EKF.NumCPIPerMeasurement = 128; % 0..1.27 s at the existing dt=0.01.
cfg.EKF.TrackTime = 1;
cfg.EKF.ADMMMaxIterations = 5000;
cfg.EKF.ADMMAbsTolerance = 1e-5; % Pin the stricter tolerances used in this test.
cfg.EKF.ADMMRelTolerance = 1e-5;
cfg.EKF.ADMMVerboseEvery = 500;
cfg.EKF.Verbose = true;
cfg.EKF.CalculatePCRLB = false;

% NumDopplerSamples stays 64: it controls noise variance, not run length.
% The complete 30-second truth needs 3001 updates and takes much longer.
Run = run_tracking("EKF", cfg);
metrics = summarize_ekf_convergence(Run);
