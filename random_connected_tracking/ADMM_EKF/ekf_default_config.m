function cfg = ekf_default_config()
%EKF_DEFAULT_CONFIG CV estimator defaults for the shared nonlinear truth.
cfg.NumMonteCarlo = 1;
cfg.Seed = 43;
cfg.NumCPIPerMeasurement = 64;
cfg.TrackTime = 10;
cfg.TimeStep = 0.01;
cfg.NumDopplerSamples = 64;
cfg.SNRdB = 40;
cfg.PreWhiten = false;
cfg.ProcessAccelerationVariance = 100; % Inherited CV process variance; not tuned.
cfg.CalculatePCRLB = true; % Compute the assumed-CV PCRLB reference.
cfg.PCRLBNumStateSamples = 2000;
cfg.PCRLBSeed = 100043;
cfg.ADMMObjectiveType = "MAP";
cfg.ADMMMaxIterations = 2000;
cfg.ADMMAbsTolerance = 1e-3;
cfg.ADMMRelTolerance = 1e-3;
cfg.ADMMPenalty = [100, 100, 15, 15];
cfg.ADMMRhoUpdateInterval = 10;
cfg.ADMMRhoBalanceMu = 10;
cfg.ADMMRhoScale = 2;
cfg.ADMMVerboseEvery = 150;
cfg.Verbose = true;
end