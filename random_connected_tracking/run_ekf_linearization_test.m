function Report = run_ekf_linearization_test()
%RUN_EKF_LINEARIZATION_TEST Formula checks and bounded recursive EKF validation.
% The longer run uses 16 updates x 2 trials; it is not the full trajectory.
project_root = fileparts(mfilename('fullpath'));
previous_path = path;
path_cleanup = onCleanup(@() path(previous_path));
addpath(fullfile(project_root, 'ADMM_EKF'));
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss_SSS'));
output_folder = fullfile(project_root, 'results', ['ekf_linearization_check_' timestamp]);
mkdir(output_folder);
diary(fullfile(output_folder, 'matlab_run.log'));
diary_cleanup = onCleanup(@() diary('off'));
Report.MATLABVersion = version;
Report.Formulas = check_ekf_linearization();

% Short execution checks cover both objective branches and whitening modes.
cfg = tracking_config('smoke');
cfg.SaveResults = false;
cfg.MakePlots = false;
cfg.EKF.ADMMMaxIterations = 3;
before_rng = rng;
for objective = ["MAP", "MLE"]
    cfg.EKF.ADMMObjectiveType = objective;
    for whiten = [false true]
        cfg.EKF.PreWhiten = whiten;
        Run = run_tracking('EKF', cfg);
        assert(isequal(before_rng, rng), 'The runner must restore caller RNG.');
        assert(all(isfinite(cell2mat(Run.Results.estimations_DA_raw)), 'all'));
        verify_posteriors(Run);
    end
end
% Historical nonlinear runs cannot be replayed as if they used the new cost.
rejected = false;
try
    check_frozen_map_solution(struct('Results', struct()));
catch exception
    rejected = strcmp(exception.identifier, 'RadarTracking:FrozenMAPModel');
end
assert(rejected, 'The frozen diagnostic must reject historical model metadata.');
Report.BranchChecksPassed = true;

cfg = tracking_config('original');
cfg.Profile = "ekf_linearized_check";
cfg.EKF.NumMonteCarlo = 2;
cfg.EKF.NumCPIPerMeasurement = 16;
cfg.EKF.TrackTime = 1;
cfg.EKF.Verbose = false;
cfg.EKF.ADMMVerboseEvery = 0;
cfg.EKF.CalculatePCRLB = false;
cfg.OutputRoot = output_folder;
cfg.FigureVisible = false;
Run = run_tracking('EKF', cfg);
Report.RunFolder = string(Run.OutputFolder);
Report.TrackingSeconds = Run.ElapsedSeconds;
Report.Convergence = summarize_ekf_convergence(Run);
Report.PosteriorChecks = verify_posteriors(Run);
Report.Frozen = check_frozen_map_solution(Run);

% Check first-scan optimizer accuracy with stricter residual thresholds.
R = Run.Results;
network = Run.network_topo;
N = network.numNodes;
e = R.Config;
[state, ~, ~, info] = tracking_consensus_admm_update( ...
    repmat({R.prior_mean_common_raw{1,1}},1,N), ...
    repmat({R.prior_cov_common_raw{1,1}},1,N), ...
    R.range_meas_raw{1}, R.doppler_meas_raw{1}, 1, network, network.neighbors, ...
    R.env, TrackingModels, TrackingADMMUtils, "MAP", "self_only", e.PreWhiten, ...
    3000, 1e-5, 1e-5, e.ADMMPenalty, e.ADMMRhoUpdateInterval, ...
    e.ADMMRhoBalanceMu, e.ADMMRhoScale, 0, false);
Report.StrictFirstScan.State = state;
Report.StrictFirstScan.Info = info;
Report.StrictFirstScan.ReferenceStateGap = norm(state-Report.Frozen.Reference.State);
assert(info.converged && info.fmincon_failure_count == 0, ...
    'Strict first-scan ADMM must pass both residual tests without solver failures.');
assert(Report.StrictFirstScan.ReferenceStateGap < 0.01, ...
    'Strict first-scan ADMM must approach the independent affine MAP solution.');
save(fullfile(output_folder, 'validation.mat'), 'Report');
writetable(Report.PosteriorChecks, fullfile(output_folder, 'posterior_checks.csv'));
fprintf('EKF linearization: %d/%d updates passed; %.3f s tracking time.\n', ...
    Run.ConvergedSteps, Run.TotalSteps, Run.ElapsedSeconds);
fprintf('Strict first scan: %d iterations; affine-reference state gap %.6g.\n', ...
    info.iter, Report.StrictFirstScan.ReferenceStateGap);
fprintf('Validation saved: %s\n', output_folder);
end

function checks = verify_posteriors(Run)
R = Run.Results;
[num_mc, num_steps] = size(R.estimations_DA_raw);
values = zeros(num_mc*num_steps, 5);
N = Run.network_topo.numNodes;
for mc = 1:num_mc
    for k = 1:num_steps
        model = R.linearization_DA_raw{mc,k};
        mu = R.prior_mean_common_raw{mc,k};
        assert(isequal(model.anchor, mean(R.predicted_DA_raw{mc,k},2)));
        assert(isequal(model.anchor, mu));
        assert(isequal(R.linearization_CA_raw{mc,k}.anchor, R.predicted_CA_raw{mc,k}));
        J = zeros(4);
        if string(R.Config.ADMMObjectiveType) == "MAP"
            J = R.prior_cov_common_raw{mc,k} \ eye(4);
        end
        rhs = J*mu;
        for n = 1:N
            H = model.H(:,:,n);
            y = [R.range_meas_raw{mc}(1,n,k); R.doppler_meas_raw{mc}(1,n,k)];
            J = J + H.'*(R.env.Sigma\H);
            rhs = rhs + H.'*(R.env.Sigma\(y-model.value(:,n)+H*model.anchor));
        end
        expected_P = J\eye(4);
        reference = J\rhs;
        actual_P = R.estimations_DA_sigma_raw{mc,k};
        covariance_gap = norm(actual_P-expected_P,'fro')/max(1,norm(expected_P,'fro'));
        assert(covariance_gap < 1e-9, 'Posterior covariance must use the frozen objective Jacobian.');
        assert(min(eig((actual_P+actual_P')/2)) > 0);
        mean_state = R.estimations_DA_raw{mc,k}(:);
        values((mc-1)*num_steps+k,:) = [mc,k,covariance_gap, ...
            norm(mean_state(1:2)-reference(1:2)), norm(mean_state(3:4)-reference(3:4))];
    end
end
checks = array2table(values, 'VariableNames', ...
    {'MC','Epoch','RelativeCovarianceGap','PositionGapToAffineMAP','VelocityGapToAffineMAP'});
end
