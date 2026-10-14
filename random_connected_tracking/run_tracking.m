function Run = run_tracking(method, cfg)
%RUN_TRACKING Run one legacy estimator on a fixed connected topology.
% Run = run_tracking("MAP", tracking_config("quick"));
% All methods share the selected ring or seeded connected random graph.

if nargin < 1
    method = "MAP";
end
if nargin < 2
    cfg = tracking_config();
end
method = string(validatestring(upper(string(method)), {'MAP', 'MLE', 'EKF'}));
project_root = fileparts(mfilename('fullpath'));
previous_path = path;
path_cleanup = onCleanup(@() path(previous_path));
addpath(fullfile(project_root, 'common'), fullfile(project_root, 'all_in_one'), ...
    fullfile(project_root, 'ADMM_EKF'));
previous_rng = rng;
rng_cleanup = onCleanup(@() rng(previous_rng));

if exist('fmincon', 'file') ~= 2 || exist('mvnrnd', 'file') ~= 2
    error('RadarTracking:MissingToolbox', ...
        'This baseline needs Optimization Toolbox and Statistics and Machine Learning Toolbox.');
end
network_topo = make_connected_random_graph(cfg.Graph, cfg.Radar);
fprintf('%s | graph=%s, N=%d, edges=%d, lambda2=%.4g\n', method, ...
    network_topo.graph_mode, network_topo.numNodes, network_topo.numEdges, ...
    network_topo.algebraic_connectivity);
if network_topo.graph_mode == "random_connected"
    fprintf('Random graph seed: %d; edge probability: %g\n', ...
        network_topo.graph_seed, network_topo.edge_probability);
end
fprintf('Radar layout: %s, radius=%g m, position seed=%d\n', ...
    cfg.Radar.Mode, cfg.Radar.Radius, cfg.Radar.Seed);
fprintf('Profile: %s. Inspect convergence before interpreting estimates.\n', cfg.Profile);

timer = tic;
if method == "EKF"
    num_samples = cfg.EKF.NumCPIPerMeasurement * cfg.EKF.TrackTime;
    target = make_tracking_truth(cfg.Trajectory, num_samples, cfg.EKF.TimeStep);
    Results = run_ekf_tracking(cfg.EKF, network_topo, target);
else
    num_samples = cfg.Batch.SamplesPerBatch * cfg.Batch.NumBatches;
    target = make_tracking_truth(cfg.Trajectory, num_samples, cfg.Batch.TimeStep);
    Results = run_batch_tracking(cfg.Batch, network_topo, method, target);
end
Run.Method = method;
Run.Config = cfg;
Run.network_topo = network_topo;
Run.Results = Results;
Run.ElapsedSeconds = toc(timer);
Run.MATLABVersion = version;
Run.TruthMotion = target.motion_model;
if method == "MLE"
    Run.EstimatorDynamics = 'CV warm start; measurement-only objective';
elseif method == "MAP"
    Run.EstimatorDynamics = 'CV prior-mean prediction; legacy fixed covariance proxy';
else
    Run.EstimatorDynamics = 'CV prediction; first-order affine measurement correction';
end
Run.OutputFolder = '';
if method == "EKF"
    converged = cell2mat(Results.admm_converged_raw);
    Run.StopCriterion = 'Primal and dual residual thresholds (inherited v5)';
    Run.LocalSolverNonpositiveFlags = sum(cell2mat(Results.fmincon_failure_count_raw), 'all');
    Run.SolverFlagScope = 'All local solves across all ADMM iterations';
else
    converged = Results.converged;
    Run.StopCriterion = 'Primal residual only (inherited batch baseline)';
    Run.LocalSolverNonpositiveFlags = sum(cellfun(@(flags) nnz(flags <= 0), ...
        Results.solver_exitflags_DA), 'all');
    Run.SolverFlagScope = 'Final ADMM iteration of each step only (legacy batch log)';
end
Run.ConvergedSteps = nnz(converged);
Run.TotalSteps = numel(converged);
fprintf('Stopping rule met: %d/%d steps across all trials (%.2f s).\n', ...
    Run.ConvergedSteps, Run.TotalSteps, Run.ElapsedSeconds);
fprintf('Local fmincon nonpositive exit flags: %d. Scope: %s.\n', ...
    Run.LocalSolverNonpositiveFlags, Run.SolverFlagScope);

if cfg.SaveResults
    if ~isfolder(cfg.OutputRoot)
        mkdir(cfg.OutputRoot);
    end
    timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss_SSS'));
    output_folder = fullfile(cfg.OutputRoot, sprintf('%s_%s_%s', ...
        method, cfg.Profile, timestamp));
    % Avoid overwriting a previous run even on a timestamp collision.
    suffix = 1;
    candidate = output_folder;
    while isfolder(candidate)
        candidate = sprintf('%s_%d', output_folder, suffix);
        suffix = suffix + 1;
    end
    output_folder = candidate;
    mkdir(output_folder);
    Run.OutputFolder = output_folder;
    save(fullfile(output_folder, 'results.mat'), 'Run', '-v7.3');
    writematrix(network_topo.adj_matrix, fullfile(output_folder, 'adjacency.csv'));
    writematrix(network_topo.edge_list, fullfile(output_folder, 'edges.csv'));
    writematrix(network_topo.radar_pos, fullfile(output_folder, 'radar_positions.csv'));
    write_summary(Run);
end

if cfg.MakePlots
    plot_tracking_results(Run, cfg.FigureVisible);
end
if cfg.SaveResults
    fprintf('Saved: %s\n', Run.OutputFolder);
end
end

function write_summary(Run)
filename = fullfile(Run.OutputFolder, 'summary.txt');
fid = fopen(filename, 'w');
if fid == -1
    error('RadarTracking:SaveFailed', 'Cannot write %s.', filename);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'Method: %s\nProfile: %s\nMATLAB: %s\n', ...
    Run.Method, Run.Config.Profile, Run.MATLABVersion);
fprintf(fid, 'Truth: %s\nEstimator dynamics: %s\nTrajectory seed: %d\n', ...
    Run.TruthMotion, Run.EstimatorDynamics, Run.Config.Trajectory.Seed);
fprintf(fid, 'Reference duration: %.3f s (short runs use a prefix)\n', Run.Config.Trajectory.Duration);
if Run.Method == "EKF"
    fprintf(fid, 'Measurement model: %s\nLinearization: %s\n', ...
        Run.Results.MeasurementModel, Run.Results.LinearizationPolicy);
end
fprintf(fid, 'Radar layout: %s\nRadar position seed: %d\nDeployment radius: %g m\nDeployment center: %s m\n', ...
    Run.Config.Radar.Mode, Run.Config.Radar.Seed, Run.Config.Radar.Radius, ...
    mat2str(Run.Config.Radar.Center));
fprintf(fid, 'Graph model: %s\nNodes: %d\nEdges: %d\n', ...
    Run.network_topo.graph_model, Run.network_topo.numNodes, Run.network_topo.numEdges);
if Run.network_topo.graph_mode == "random_connected"
    fprintf(fid, 'Graph seed: %d\nEdge probability: %g\nGeneration attempts: %d\n', ...
        Run.network_topo.graph_seed, Run.network_topo.edge_probability, ...
        Run.network_topo.generation_attempts);
end
fprintf(fid, 'Degrees: %s\nLambda2: %.12g\n', ...
    mat2str(diag(Run.network_topo.degree_matrix).'), ...
    Run.network_topo.algebraic_connectivity);
fprintf(fid, 'Stopping rule met: %d/%d steps\nRule: %s\nElapsed seconds: %.6f\n', ...
    Run.ConvergedSteps, Run.TotalSteps, Run.StopCriterion, Run.ElapsedSeconds);
fprintf(fid, 'Local fmincon nonpositive exit flags: %d\nFlag scope: %s\n', ...
    Run.LocalSolverNonpositiveFlags, Run.SolverFlagScope);
fprintf(fid, '\nSmoke/quick runs are execution checks, not validated research results.\n');
fprintf(fid, 'See README.md for retained estimator conventions and comparison limits.\n');
end
