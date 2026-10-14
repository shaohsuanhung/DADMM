function Report = run_batch_linearization_test()
%RUN_BATCH_LINEARIZATION_TEST Validate affine costs and run bounded MAP/MLE checks.
% Run from this folder. All numerical work uses MATLAB, with fixed seeds.
project_root = fileparts(mfilename('fullpath'));
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss_SSS'));
output_folder = fullfile(project_root, 'results', ['batch_linearization_check_' timestamp]);
mkdir(output_folder);
diary(fullfile(output_folder, 'matlab_run.log'));
diary_cleanup = onCleanup(@() diary('off'));
Report.MATLABVersion = version;
Report.OutputFolder = output_folder;
Report.Formulas = check_batch_linearization();

% Cover both methods, whitening branches, and the second-batch CV anchor.
cfg = tracking_config('smoke');
cfg.SaveResults = false;
cfg.MakePlots = false;
cfg.Batch.NumBatches = 2;
cfg.Batch.MaxIterations = 3;
before_rng = rng;
methods = ["MAP", "MLE"];
for method = methods
    for whiten = [false true]
        cfg.Batch.PreWhiten = whiten;
        Run = run_tracking(method, cfg);
        R = Run.Results;
        assert(isequal(before_rng, rng), 'The caller RNG must be restored.');
        assert(all(isfinite(cell2mat(R.consensus_estimates)), 'all'));
        for k = 1:cfg.Batch.NumBatches
            assert(isequal(R.linearization_CA{1,k}.anchor, R.warm_starts_CA{1,k}));
            models = R.linearization_DA{1,k};
            for n = 1:cfg.Graph.NumNodes
                assert(isequal(models{n}.anchor, R.warm_starts_DA{1,k}(:,n)));
                assert(isequal(models{n}.anchor, models{1}.anchor), ...
                    'Every distributed node must use the same batch anchor.');
            end
        end
        assert(norm(R.linearization_CA{1,2}.anchor - R.F_batch*R.estimations_CA{1,1}(:)) < 1e-10);
        assert(norm(R.linearization_DA{1,2}{1}.anchor - R.F_batch*R.consensus_estimates{1,1}) < 1e-10);
        if ~whiten
            raw = R;
        else
            assert(isequal(raw.prior_covariances_CA, R.prior_covariances_CA), ...
                'Measurement whitening must not change state prior covariance.');
            assert(norm(raw.estimations_CA{1,1}-R.estimations_CA{1,1}) < 1e-4, ...
                'Physical and whitened first-batch centralized solutions must agree.');
        end
    end
end
Report.IntegrationPassed = true;

% Preserve the original 64-pulse noise setting while bounding tracking length.
cfg = tracking_config('original');
cfg.Profile = "linearized_check";
cfg.Batch.NumMonteCarlo = 1;
cfg.Batch.NumBatches = 3;
cfg.Batch.MaxIterations = 300;
cfg.OutputRoot = output_folder;
cfg.MakePlots = true;
cfg.FigureVisible = false;
Report.Config = cfg;
Report.RunFolders = strings(2,1);
elapsed = zeros(2,1);
stopped = zeros(2,1);
iterations = zeros(2,3);
final_primal = zeros(2,3);
for i = 1:numel(methods)
    Run = run_tracking(methods(i), cfg);
    Report.RunFolders(i) = string(Run.OutputFolder);
    elapsed(i) = Run.ElapsedSeconds;
    stopped(i) = Run.ConvergedSteps;
    iterations(i,:) = cell2mat(Run.Results.convg_iter);
    final_primal(i,:) = cellfun(@(x) x(end), Run.Results.primal_residauls);
end
Report.Tracking = table(methods(:), elapsed, stopped, iterations, final_primal, ...
    'VariableNames', {'Method','ElapsedSeconds','PrimalStops','Iterations','FinalPrimal'});
Report.Benchmark = benchmark_batch_linearization();
save(fullfile(output_folder, 'validation.mat'), 'Report');
writetable(Report.Tracking, fullfile(output_folder, 'tracking_checks.csv'));
writetable(Report.Benchmark.Summary, fullfile(output_folder, 'local_solver_benchmark.csv'));
disp(Report.Tracking);
disp(Report.Benchmark.Summary);
fprintf('Saved linearization validation: %s\n', output_folder);
fprintf('Batch stopping still checks only primal residual; this is not a convergence proof.\n');
end
