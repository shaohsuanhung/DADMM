function cfg = batch_default_config()
%BATCH_DEFAULT_CONFIG Editable settings retained from all_in_one/MAP_tracking.
% These are the original full-run sizes. The caller may override fields for
% a short smoke run without changing the estimator formulas.
cfg.NumMonteCarlo = 1;
cfg.Seed = 43;
cfg.SNRdB = 50;
cfg.NumBatches = 30;
cfg.SamplesPerBatch = 64;
cfg.TimeStep = 0.01;          % Seconds per original measurement sample.
cfg.MaxIterations = 1000;    % ADMM cap, not a promise of convergence.
cfg.Tolerance = 1e-3;        % Original effective primal-residual threshold.
cfg.PreWhiten = false;
cfg.Verbose = false;
cfg.ComputeLegacyBound = false; % Original helper is not a validated PCRLB.
cfg.InitialState = [-30; -30; 10; 10];
cfg.PriorMean = [-30; -30; 14; 14];
cfg.ProcessAccelerationVariance = 1e-2; % Original CV acceleration variance.
end
