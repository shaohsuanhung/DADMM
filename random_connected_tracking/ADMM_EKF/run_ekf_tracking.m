function Results = run_ekf_tracking(cfg, network_topo, target)
%RUN_EKF_TRACKING First-order global consensus-ADMM EKF from the v5 runner.
% Geometry, graph and a shared nonlinear truth are supplied by the caller.
% The estimator retains its constant-velocity prediction model. Every node
% owns one measurement; its affine model stays fixed within each ADMM solve.
% This function performs no plotting, saving, path changes or workspace clearing.
    if nargin < 3
        error('Tracking:MissingTruth', 'Supply the shared target as the third input.');
    end
    defaults = ekf_default_config();
    if nargin < 1 || isempty(cfg), cfg = defaults; end
    fields = fieldnames(defaults);
    for field_index = 1:numel(fields)
        name = fields{field_index};
        if ~isfield(cfg, name), cfg.(name) = defaults.(name); end
    end
    validateattributes(cfg.NumMonteCarlo, {'numeric'}, {'scalar','integer','positive'});
    validateattributes(cfg.NumCPIPerMeasurement, {'numeric'}, {'scalar','integer','positive'});
    validateattributes(cfg.TrackTime, {'numeric'}, {'scalar','integer','positive'});
    validateattributes(cfg.ADMMMaxIterations, {'numeric'}, {'scalar','integer','positive'});
    validateattributes(cfg.TimeStep, {'numeric'}, {'scalar','positive','finite'});
    validateattributes(cfg.ProcessAccelerationVariance, {'numeric'}, ...
        {'scalar','real','nonnegative','finite'});
    ut = TrackingADMMUtils;
    models_ut = TrackingModels;
    neighbors = ut.get_neighbors(network_topo.adj_matrix, network_topo.numNodes);
    if network_topo.numNodes > 1 && numel(unique(conncomp(graph(network_topo.adj_matrix)))) ~= 1
        error('Tracking:DisconnectedGraph', 'Consensus ADMM requires a connected graph.');
    end
    NUM_CPI_PER_MEA = cfg.NumCPIPerMeasurement;
    NUM_DOPPLER_ESTIMATION_SAMPLES = cfg.NumDopplerSamples;
    TRACK_TIME = cfg.TrackTime;
    dt = cfg.TimeStep;
    NUM_TAR = 1;
    M_total = NUM_CPI_PER_MEA * TRACK_TIME;
    if ~isequal(size(target.target_state), [1, M_total, 4]) || ...
            ~isequal(size(target.target_position), [1, M_total, 2]) || ...
            numel(target.time) ~= M_total
        error('Tracking:TruthSize', ...
            'Shared truth must contain exactly %d states and matching positions/times.', M_total);
    end
    validateattributes(target.target_state, {'numeric'}, {'real','finite'});
    validateattributes(target.true_params, {'numeric'}, {'vector','numel',4,'real','finite'});
    if ~isequal(target.true_params(:), reshape(target.target_state(1, 1, :), 4, 1))
        error('Tracking:TruthInitialState', 'target.true_params must be the first truth state.');
    end
    if M_total > 1 && any(abs(diff(target.time(:)) - dt) > 1e-10 * max(1, dt))
        error('Tracking:TruthTimeStep', 'Shared truth timestamps must use cfg.TimeStep.');
    end
    env.SNR_idx = cfg.SNRdB;
    num_monte_carlo = cfg.NumMonteCarlo;
    seed0 = cfg.Seed;
    PCRLB_NUM_STATE_SAMPLES = cfg.PCRLBNumStateSamples;
    PCRLB_SEED = cfg.PCRLBSeed;
    PRE_WHITEN = cfg.PreWhiten;
    DEBUG = cfg.Verbose;
    MEASUREMENT_MODE = "self_only";
    ADMM_OBJECTIVE_TYPE = cfg.ADMMObjectiveType;
    ADMM_MAX_ITER = cfg.ADMMMaxIterations;
    ADMM_TOLERANCE = cfg.ADMMAbsTolerance;
    ADMM_REL_TOLERANCE = cfg.ADMMRelTolerance;
    ADMM_C_PENALTY = cfg.ADMMPenalty;
    ADMM_RHO_UPDATE_INTERVAL = cfg.ADMMRhoUpdateInterval;
    ADMM_RHO_BALANCE_MU = cfg.ADMMRhoBalanceMu;
    ADMM_RHO_SCALE = cfg.ADMMRhoScale;
    ADMM_VERBOSE_EVERY = cfg.ADMMVerboseEvery;
    rng(seed0, "twister");
    SNAPSHOT_T = randi(M_total);

%% ---------------- Result Log -------------
Results = struct();
Results.primal_residauls = cell(num_monte_carlo, TRACK_TIME);
Results.dual_residauls   = cell(num_monte_carlo, TRACK_TIME);
Results.estimations_DA   = cell(num_monte_carlo, TRACK_TIME);
Results.true_params      = cell(num_monte_carlo, TRACK_TIME);
Results.estimations_CA   = cell(num_monte_carlo, TRACK_TIME);
Results.convg_iter       = cell(num_monte_carlo, TRACK_TIME);
Results.estimations_DA_sigma = cell(num_monte_carlo, TRACK_TIME);
Results.estimations_CA_sigma = cell(num_monte_carlo, TRACK_TIME);

Results.primal_residauls_raw     = cell(num_monte_carlo, M_total);
Results.dual_residauls_raw       = cell(num_monte_carlo, M_total);
Results.estimations_DA_raw       = cell(num_monte_carlo, M_total);
Results.estimations_DA_nodes_raw = cell(num_monte_carlo, M_total);
Results.estimations_DA_nodes_hist_raw = cell(num_monte_carlo, M_total);
Results.estimations_DA_sigma_raw = cell(num_monte_carlo, M_total);
Results.P_nodes_raw              = cell(num_monte_carlo, M_total);
Results.P_dadmm_raw              = cell(num_monte_carlo, M_total);
Results.true_params_raw          = cell(num_monte_carlo, M_total);
Results.estimations_CA_raw       = cell(num_monte_carlo, M_total);
Results.estimations_CA_sigma_raw = cell(num_monte_carlo, M_total);
Results.predicted_DA_raw         = cell(num_monte_carlo, M_total);
Results.predicted_CA_raw         = cell(num_monte_carlo, M_total);
Results.linearization_DA_raw     = cell(num_monte_carlo, M_total);
Results.linearization_CA_raw     = cell(num_monte_carlo, M_total);
Results.prior_mean_common_raw    = cell(num_monte_carlo, M_total);
Results.prior_cov_common_raw     = cell(num_monte_carlo, M_total);
Results.convg_iter_raw           = cell(num_monte_carlo, M_total);
Results.admm_converged_raw       = cell(num_monte_carlo, M_total);
Results.admm_eps_pri_raw         = cell(num_monte_carlo, M_total);
Results.admm_eps_dual_raw        = cell(num_monte_carlo, M_total);
Results.fmincon_exitflag_raw     = cell(num_monte_carlo, M_total);
Results.fmincon_failure_count_raw = cell(num_monte_carlo, M_total);
Results.fmincon_max_iterations_raw = cell(num_monte_carlo, M_total);
Results.admm_stop_reason_raw     = cell(num_monte_carlo, M_total);
Results.admm_c_penalty_final_raw = cell(num_monte_carlo, M_total);
Results.admm_c_penalty_history_raw = cell(num_monte_carlo, M_total);
Results.CRLB                     = cell(num_monte_carlo, M_total);
Results.range_meas_raw           = cell(num_monte_carlo, 1);
Results.doppler_meas_raw         = cell(num_monte_carlo, 1);
Results.measurement_seed         = zeros(num_monte_carlo, 1);

%% ---------------- Environment / measurement noise ----------------
F = [1 0 dt 0;
     0 1 0 dt;
     0 0 1  0;
     0 0 0  1];

env.c = 3e8;
env.lambda = env.c / 10e9;
env.time_step = dt;

env.B  = 10e6 * ones(1, network_topo.numNodes);
env.fs = 2 * env.B;


env.SNR_lin = 10^(env.SNR_idx/10);

env.range_var   = (3 * env.c^2) / (8 * pi^2 * env.B(1)^2 * env.SNR_lin);
env.doppler_var = (3 * (env.fs(1)^2)) / ...
                  (pi^2 * env.SNR_lin * ...
                   NUM_DOPPLER_ESTIMATION_SAMPLES^3);
env.rho = 0.0;

measurement_cross_cov = env.rho * sqrt(env.range_var * env.doppler_var);
env.Sigma = [env.range_var,          measurement_cross_cov;
             measurement_cross_cov, env.doppler_var];

env.PRE_WHITEN = PRE_WHITEN;
if env.PRE_WHITEN
    % L*Sigma*L' = I
    chol_lower        = chol(env.Sigma, 'lower');
    env.pre_whit_L    = chol_lower \ eye(2);
    env.Sigma_filter  = eye(2);
else
    env.pre_whit_L    = eye(2);
    env.Sigma_filter  = env.Sigma;
end

%% ---------------- DKF / CV model ----------------
% The prescribed truth may turn and accelerate; prediction still uses CV.
% The configurable process variance retains the source default of zero.
% It is an experiment setting, not a tuned value for the nonlinear truth.
env.Q = cfg.ProcessAccelerationVariance * ...
              [dt^4/4, 0,      dt^3/2, 0;
                0,      dt^4/4, 0,      dt^3/2;
                dt^3/2, 0,      dt^2,   0;
                0,      dt^3/2, 0,      dt^2];

P0 = diag([1e4, 1e4, 1e6, 1e6]);
% P0 = diag([1e2, 1e2, 1e6, 1e6]);
% Both filters start from the same prior, centered at the first truth state.
x0 = target.true_params(:);

%% ---------------- Optional assumed-CV reference ----------------
if cfg.CalculatePCRLB
    [PCRLB_all, PCRLB_info] = tracking_pcrlb( ...
        x0, P0, F, env.Q, network_topo.radar_pos, ...
        env.lambda, env.Sigma, M_total, PCRLB_NUM_STATE_SAMPLES, ...
        'Seed', PCRLB_SEED);
    Results.PCRLB_status = ...
        "assumed-CV reference only, not a bound for prescribed nonlinear truth";
else
    PCRLB_all = [];
    PCRLB_info = struct();
    Results.PCRLB_status = "disabled for prescribed nonlinear truth";
end
Results.PCRLB_Bayes = PCRLB_all;

%% ---------------- Monte Carlo simulation ----------------
for mc = 1:num_monte_carlo
    rng(seed0 + mc, "twister");
    if cfg.Verbose, fprintf("\n[MC %d/%d]\n", mc, num_monte_carlo); end

    x_dadmm = cell(1, network_topo.numNodes);
    P_dadmm = cell(1, network_topo.numNodes);
    for n = 1:network_topo.numNodes
        % Every node estimates the same global target state.
        x_dadmm{n} = x0;
        P_dadmm{n} = P0;
    end

    % Centralized EKF baseline uses the same prior as the distributed filter.
    x_ca = x0;
    P_ca = P0;

    %% ---------------- Synthetic measurements ----------------
    range_true   = zeros(NUM_TAR, network_topo.numNodes, M_total);
    doppler_true = zeros(NUM_TAR, network_topo.numNodes, M_total);
    meas_true    = zeros(NUM_TAR, network_topo.numNodes, 2*M_total);

    [range_true, doppler_true, meas_true] = ut.gt_data_generation_tracking_ekf( ...
        range_true, doppler_true, meas_true, target, network_topo, env, ...
        M_total, NUM_TAR);

    [range_meas, doppler_meas, ~] = ut.add_measurement_noise( ...
        range_true, doppler_true, M_total, NUM_TAR, network_topo.numNodes, env);
    Results.range_meas_raw{mc} = range_meas;
    Results.doppler_meas_raw{mc} = doppler_meas;
    Results.measurement_seed(mc) = seed0 + mc;

    %% ---------------- Run recursive simulation ----------------
    for kBurst = 1:TRACK_TIME
        for inst = 1:NUM_CPI_PER_MEA
            t = (kBurst-1)*NUM_CPI_PER_MEA + inst;
            if cfg.Verbose, fprintf("t=%d/%d\n", t, M_total); end

            % ------------------------------------------------------------
            % 1) Prediction at all nodes
            % ------------------------------------------------------------
            x_pred_nodes = cell(1, network_topo.numNodes);
            P_pred_nodes = cell(1, network_topo.numNodes);
            for n = 1:network_topo.numNodes
                if t == 1
                    % target_state(:,:,1) is the initial state, so the first
                    % measurement update must not be preceded by prediction.
                    x_pred_nodes{n} = x_dadmm{n};
                    P_pred_nodes{n} = P_dadmm{n};
                else
                    [x_pred_nodes{n}, P_pred_nodes{n}] = predict_cv( ...
                        x_dadmm{n}, P_dadmm{n}, F, env.Q);
                end
            end
            Results.predicted_DA_raw{mc, t} = cat(2, x_pred_nodes{:});

            % ------------------------------------------------------------
            % 2) Fixed first-order consensus ADMM measurement update
            %
            % Each node solves a quadratic local MAP/MLE cost with fmincon
            % and an analytic gradient. All nodes share one predicted anchor:
            %
            %   local negative log-posterior / likelihood
            %   + sum_j Nu_{n,j}'(theta_n - z_{n,j})
            %   + sum_j 0.5*(theta_n-z_{n,j})'*C*(theta_n-z_{n,j})
            %
            % This follows the structure in your reference snippet.
            % ------------------------------------------------------------
            [x_consensus, x_nodes_admm, P_nodes_admm, admm_info] = ...
                tracking_consensus_admm_update( ...
                    x_pred_nodes, P_pred_nodes, range_meas, doppler_meas, t, ...
                    network_topo, neighbors, env, models_ut, ut, ...
                    ADMM_OBJECTIVE_TYPE, MEASUREMENT_MODE, PRE_WHITEN, ...
                    ADMM_MAX_ITER, ADMM_TOLERANCE, ADMM_REL_TOLERANCE, ...
                    ADMM_C_PENALTY, ADMM_RHO_UPDATE_INTERVAL, ...
                    ADMM_RHO_BALANCE_MU, ADMM_RHO_SCALE, ...
                    ADMM_VERBOSE_EVERY, DEBUG);

            if DEBUG && ~admm_info.converged
                warning('ADMM_EKF:ADMMMaxIterations', ...
                    ['ADMM did not satisfy both residual tolerances at ', ...
                     'MC %d, t %d (r %.3e/%.3e, s %.3e/%.3e).'], ...
                    mc, t, admm_info.primal_residual(end), ...
                    admm_info.eps_pri(end), admm_info.dual_residual(end), ...
                    admm_info.eps_dual(end));
            end
            if DEBUG && admm_info.fmincon_failure_count > 0
                warning('ADMM_EKF:FminconFailure', ...
                    '%d local fmincon solves reported exitflag <= 0.', ...
                    admm_info.fmincon_failure_count);
            end

            % Feed the ADMM posterior back to the distributed filters.
            % x_nodes_admm(:,n) is retained so the recursion remains
            % distributed if ADMM stops before exact consensus.
            for n = 1:network_topo.numNodes
                x_dadmm{n} = x_nodes_admm(:, n);
                P_dadmm{n} = P_nodes_admm{n};
            end

            % All nodes receive the covariance of the same fixed affine
            % posterior. Preserve the existing mean-of-nodes log structure.
            P_consensus = zeros(4);
            for n = 1:network_topo.numNodes
                P_consensus = P_consensus + P_nodes_admm{n};
            end
            P_consensus = symmetrize(P_consensus / network_topo.numNodes);
           
% ------------------------------------------------------------
            % 4) Centralized EKF baseline using all radar measurements once
            % ------------------------------------------------------------
            if t == 1
                x_ca_pred = x_ca;
                P_ca_pred = P_ca;
            else
                [x_ca_pred, P_ca_pred] = predict_cv(x_ca, P_ca, F, env.Q);
            end
            Results.predicted_CA_raw{mc, t} = x_ca_pred;
            idx_all = (1:network_topo.numNodes).';
            y_all = stack_measurements(idx_all, range_meas, doppler_meas, t);
            [x_ca, P_ca, model_ca] = tracking_centralized_ekf_update(x_ca_pred,P_ca_pred,y_all,network_topo,env,models_ut);
            % ------------------------------------------------------------
            % 5) Logging
            % ------------------------------------------------------------
            true_params_k = squeeze(target.target_state(1,t,:))';

            Results.estimations_DA_raw{mc, t}       = x_consensus(:).';
            Results.estimations_DA_nodes_raw{mc, t} = x_nodes_admm;
            Results.estimations_DA_nodes_hist_raw{mc, t} = admm_info.x_hist;
            Results.estimations_DA_sigma_raw{mc, t} = P_consensus;
            Results.P_nodes_raw{mc, t}              = cat(3, P_nodes_admm{:});
            Results.P_dadmm_raw{mc, t}              = cat(3, P_dadmm{:});
            Results.true_params_raw{mc, t}          = true_params_k;
            Results.convg_iter_raw{mc, t}           = admm_info.iter;
            Results.admm_converged_raw{mc, t}       = admm_info.converged;
            Results.admm_eps_pri_raw{mc, t}         = admm_info.eps_pri;
            Results.admm_eps_dual_raw{mc, t}        = admm_info.eps_dual;
            Results.fmincon_exitflag_raw{mc, t}     = admm_info.fmincon_exitflag;
            Results.fmincon_failure_count_raw{mc, t} = ...
                admm_info.fmincon_failure_count;
            Results.fmincon_max_iterations_raw{mc, t} = ...
                admm_info.fmincon_max_iterations;
            Results.admm_stop_reason_raw{mc, t}     = admm_info.stop_reason;
            Results.admm_c_penalty_final_raw{mc, t} = ...
                admm_info.c_penalty_final;
            Results.admm_c_penalty_history_raw{mc, t} = ...
                admm_info.c_penalty_history;
            Results.primal_residauls_raw{mc, t}     = admm_info.primal_residual;
            Results.dual_residauls_raw{mc, t}       = admm_info.dual_residual;

            Results.estimations_CA_raw{mc, t}       = x_ca(:).';
            Results.estimations_CA_sigma_raw{mc, t} = P_ca;
            Results.linearization_DA_raw{mc, t} = admm_info.linearization;
            Results.linearization_CA_raw{mc, t} = model_ca;
            Results.prior_mean_common_raw{mc, t} = admm_info.prior_mean_common;
            Results.prior_cov_common_raw{mc, t} = admm_info.prior_cov_common;

            % Keep the legacy cell field for existing plotting/log code.
            % The Bayesian bound is independent of measurement-noise MC.
            if cfg.CalculatePCRLB
                Results.CRLB{mc, t} = PCRLB_all(:,:,t);
            end

            if DEBUG && (t == 1 || mod(t, 64) == 0)
                fprintf("  ADMM iter=%d, r=%.3e, s=%.3e, est=[%.2f %.2f %.2f %.2f]\n", ...
                    admm_info.iter, admm_info.primal_residual(end), ...
                    admm_info.dual_residual(end), x_consensus);
            end
        end

        % Save burst-level values at the last CPI sample in the burst.
        t_idx = kBurst * NUM_CPI_PER_MEA;
        Results.estimations_DA{mc, kBurst}       = Results.estimations_DA_raw{mc, t_idx};
        Results.estimations_CA{mc, kBurst}       = Results.estimations_CA_raw{mc, t_idx};
        Results.true_params{mc, kBurst}          = Results.true_params_raw{mc, t_idx};
        Results.convg_iter{mc, kBurst}           = Results.convg_iter_raw{mc, t_idx};
        Results.primal_residauls{mc, kBurst}     = Results.primal_residauls_raw{mc, t_idx};
        Results.dual_residauls{mc, kBurst}       = Results.dual_residauls_raw{mc, t_idx};
        Results.estimations_DA_sigma{mc, kBurst} = Results.estimations_DA_sigma_raw{mc, t_idx};
        Results.estimations_CA_sigma{mc, kBurst} = Results.estimations_CA_sigma_raw{mc, t_idx};

        if cfg.Verbose
        fprintf("Burst %d done. ADMM-DKF=[%.2f %.2f %.2f %.2f], CKF=[%.2f %.2f %.2f %.2f]\n", ...
            kBurst, Results.estimations_DA{mc, kBurst}, Results.estimations_CA{mc, kBurst});
        end
    end
end

Results.Config = cfg;
Results.network_topo = network_topo;
Results.env = env;
Results.target = target;
Results.F = F;
Results.P0 = P0;
Results.x0 = x0;
Results.PCRLB_info = PCRLB_info;
Results.SnapshotT = SNAPSHOT_T;
Results.measurement_multiplicity = ones(network_topo.numNodes, 1);
Results.MeasurementMode = MEASUREMENT_MODE;
Results.SourceRunner = 'run_sim_dkf_consensus_admm_v5.m';
Results.DynamicsModel = 'CV';
Results.MeasurementModel = "first_order_affine";
Results.LinearizationPolicy = "Fixed per update at the common DA predicted mean; CA uses its own predicted mean";
Results.CovarianceModel = "Information update using the same fixed Jacobians as the affine objective";
Results.DopplerConvention = "Target-minus-radar; 2/lambda; physical observations remain nonlinear";
Results.trajectory_seed = target.trajectory_seed;
Results.time = target.time(:);
Results.TrajectoryReference = target.reference;
end

function [x_pred, P_pred] = predict_cv(x, P, F, Q)
    x_pred = F * x;
    P_pred = F * P * F' + Q;
    P_pred = symmetrize(P_pred);
end

function y_bar = stack_measurements(idx_set, range_meas, doppler_meas, t)
    y_bar = zeros(2*numel(idx_set), 1);
    for a = 1:numel(idx_set)
        j = idx_set(a);
        y_bar(2*a-1) = range_meas(1, j, t);
        y_bar(2*a)   = doppler_meas(1, j, t);
    end
end

function A = symmetrize(A)
    A = 0.5 * (A + A');
end
