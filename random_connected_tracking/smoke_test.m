function smoke_test()
%SMOKE_TEST Exercise all methods, irregular graphs, recursive steps and saving.
% Uses small iteration caps; successful execution does not prove convergence.
check_graph();
check_radar_positions();
check_trajectory();
cfg = tracking_config('smoke');
cfg.Batch.NumMonteCarlo = 2;
cfg.Batch.NumBatches = 2;
cfg.Batch.MaxIterations = 3;
cfg.EKF.NumMonteCarlo = 2;
cfg.EKF.CalculatePCRLB = true;
cfg.EKF.PCRLBNumStateSamples = 20;
before_rng = rng;
methods = ["MAP", "MLE", "EKF"];
previous_adjacency = [];
shared_radar_positions = [];
shared_reference = [];
for method = methods
    Run = run_tracking(method, cfg);
    if method == "EKF"
        estimates = Run.Results.estimations_DA_raw;
        assert(isequal(size(Run.Results.PCRLB_Bayes), [4, 4, 2]));
        assert(contains(Run.Results.PCRLB_status, 'reference only'));
        target = Run.Results.target;
        for mc = 1:2
            expected = Run.Results.F * Run.Results.estimations_DA_nodes_raw{mc, 1};
            assert(norm(Run.Results.predicted_DA_raw{mc, 2} - expected, 'fro') < 1e-10);
            expected = Run.Results.F * Run.Results.estimations_CA_raw{mc, 1}(:);
            assert(norm(Run.Results.predicted_CA_raw{mc, 2} - expected) < 1e-10);
        end
    else
        estimates = Run.Results.consensus_estimates;
        assert(isequal(Run.Results.targets{1}, Run.Results.targets{2}), ...
            'Measurement-noise trials must share the prescribed truth.');
        target = Run.Results.targets{1};
        for mc = 1:2
            expected = Run.Results.F_batch * Run.Results.consensus_estimates{mc, 1}(:);
            assert(norm(Run.Results.predicted_means_DA{mc, 2}(:, 1) - expected) < 1e-10);
            expected = Run.Results.F_batch * Run.Results.estimations_CA{mc, 1}(:);
            assert(norm(Run.Results.predicted_means_CA{mc, 2} - expected) < 1e-10);
        end
    end
    assert(isequal(size(estimates), [2 2]));
    assert(all(cellfun(@(x) numel(x) == 4 && all(isfinite(x(:))), estimates(:))));
    assert(isequal(before_rng, rng), 'The runner must restore the caller RNG.');
    if ~isempty(previous_adjacency)
        assert(isequal(previous_adjacency, Run.network_topo.adj_matrix));
    end
    previous_adjacency = Run.network_topo.adj_matrix;
    if ~isempty(shared_radar_positions)
        assert(isequal(shared_radar_positions, Run.network_topo.radar_pos));
    end
    shared_radar_positions = Run.network_topo.radar_pos;
    if ~isempty(shared_reference)
        assert(isequal(shared_reference, target.reference.states));
    end
    shared_reference = target.reference.states;
    saved = load(fullfile(Run.OutputFolder, 'results.mat'), 'Run');
    assert(isequaln(saved.Run.Results, Run.Results));
    positions = readmatrix(fullfile(Run.OutputFolder, 'radar_positions.csv'));
    assert(norm(positions - Run.network_topo.radar_pos, 'fro') < 1e-10);
    assert(isfile(fullfile(Run.OutputFolder, 'overview.png')));
    assert(isfile(fullfile(Run.OutputFolder, 'convergence.png')));
    assert(isfile(fullfile(Run.OutputFolder, 'truth_trajectory.png')));
    assert(isfile(fullfile(Run.OutputFolder, 'radar_layout.png')));
end
fprintf('Smoke checks passed for MAP, MLE and EKF. Inspect saved stopping flags.\n');
end
