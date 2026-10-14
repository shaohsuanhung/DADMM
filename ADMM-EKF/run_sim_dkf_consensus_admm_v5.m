clc; clear; close all;

% This repository contains several classes/functions with identical names.
% Refuse to run if MATLAB resolves a different ADMM-EKF implementation.
script_dir = fileparts(mfilename('fullpath'));
dependency_names = {'ADMM_utils', 'models', 'make_figs', ...
    'dkf_neighbor_update', 'centralized_ekf_update_stack', ...
    'calculate_pcrlb_v5'};
for dependency_index = 1:numel(dependency_names)
    dependency_name = dependency_names{dependency_index};
    expected_dependency = fullfile(script_dir, [dependency_name, '.m']);
    resolved_dependency = which(dependency_name);
    if ~strcmp(resolved_dependency, expected_dependency)
        error('ADMM_EKF:PathShadowing', ...
            ['MATLAB resolved %s to:\n  %s\nExpected:\n  %s\n', ...
             'Change the current folder to ADMM-EKF, run clear classes, ', ...
             'and retry.'], ...
            dependency_name, resolved_dependency, expected_dependency);
    end
end

ut        = ADMM_utils;
models_ut = models;

DEBUG      = true;
PRE_WHITEN = false;   % handled consistently in MAP and EKF updates
SAVE_LOG   = true;
Log_DIR    = "./data_log/verify_CRLB4";
RUN_NAME   = "dkf_consensus_admm";
TYPE       = "DKF_ADMM";

%% ========================================================================
% Distributed EKF + Consensus ADMM
%
% The ADMM step follows the nonlinear fmincon-based consensus structure
% with Nu, edge-wise z, and local MAP/MLE objectives.
%
% This script keeps the state-space model and simulation parameters from
% run_sim_dkf_neighbor_consensus.m.  The difference is that the final
% agreement step is solved by consensus ADMM, not by consensus.m or
% consensus_covariance.m.
%
% Required files in the same folder/path:
%   ADMM_utils.m
%   models.m
%   make_figs.m
%   centralized_ekf_update_stack.m
%   calculate_pcrlb_v5.m
%
% Not required:
%   consensus.m
%   consensus_covariance.m
%% ========================================================================

%% ---------------- P1: Simulation parameters ----------------
NUM_CPI_PER_MEA = 64;
% Number of fast-time samples used to form each Doppler estimate. Keep this
% separate from the number of recursive tracking epochs.
NUM_DOPPLER_ESTIMATION_SAMPLES = 64;
TRACK_TIME      = 1;
dt              = 1e-2;
NUM_TAR         = 1;
M_total         = NUM_CPI_PER_MEA * TRACK_TIME;
env.SNR_idx = 40;
% -----------------------
% MC config
% -----------------------
num_monte_carlo = 2;
seed0           = 43;

% Bayesian PCRLB uses an independent state Monte Carlo expectation.  This
% is intentionally separate from num_monte_carlo, which controls the main
% simulation's measurement-noise trials.
PCRLB_NUM_STATE_SAMPLES = 2000;
PCRLB_SEED              = seed0 + 100000;

%% ---------------- ADMM config ----------------
% The ADMM part follows the nonlinear consensus-ADMM style in which each
% node solves a local MLE/MAP problem with fmincon and an edge-wise
% consensus penalty. No consensus.m or consensus_covariance.m is used.
%
% For a recursive EKF-style simulation, MAP is the natural choice because
% the predicted state/covariance at each node provides the prior. MLE is
% kept as an option for debugging o for the first step if desired.
% Each physical measurement must occur exactly once in the sum of local
% consensus objectives. A closed-neighborhood EKF is a different estimator.
MEASUREMENT_MODE     = "local_neighbor";
ADMM_OBJECTIVE_TYPE  = "MAP";              % "MAP" or "MLE"

ADMM_MAX_ITER        = 2000;
ADMM_TOLERANCE       = 1e-5;  % absolute residual tolerance
ADMM_REL_TOLERANCE   = 1e-5;
ADMM_C_PENALTY       = [100, 100, 15, 15];
ADMM_RHO_UPDATE_INTERVAL = 10;
ADMM_RHO_BALANCE_MU      = 10;
ADMM_RHO_SCALE           = 2;
ADMM_VERBOSE_EVERY   = 150;

% Randomly pick one sample index for the ADMM node-convergence snapshot.
% This is selected once so that the plotted snapshot corresponds to one
% concrete recursive EKF time step.
rng(seed0, "twister");
SNAPSHOT_T = randi(M_total);
% SNAPSHOT_T = 3;
fprintf("Random ADMM convergence snapshot will be plotted at t=%d/%d.\n", ...
SNAPSHOT_T, M_total);

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

all_tracking_params_raw    = cell(num_monte_carlo, M_total);
all_tracking_params_ca_raw = cell(num_monte_carlo, M_total);

%% ---------------- Network topology ----------------
network_topo.numNodes = 10;
theta = linspace(0, 2*pi, network_topo.numNodes+1);
network_topo.theta = theta(1:end-1);
network_topo.radius = 30;
network_topo.com_rad_CR = 30;

network_topo.radar_pos = network_topo.radius * ...
    [cos(network_topo.theta); sin(network_topo.theta)]';
network_topo.labels = arrayfun(@(i) sprintf("N%d", i), ...
    1:network_topo.numNodes, "UniformOutput", false);
fig_ut = make_figs(network_topo.numNodes);

[network_topo.adj_matrix, network_topo.degree_matrix, ...
 network_topo.laplacian_matrix, network_topo.inc_matrix, ...
 network_topo.weights_matrix] = ut.calculate_all_graph_matrix( ...
    network_topo.radar_pos, network_topo.com_rad_CR, network_topo.numNodes);

neighbors = ut.get_neighbors(network_topo.adj_matrix, network_topo.numNodes);
degree_vector = diag(network_topo.degree_matrix).';
fprintf('Topology: com_rad=%g, edges=%d, degree=%s\n', ...
    network_topo.com_rad_CR, nnz(triu(network_topo.adj_matrix)), ...
    mat2str(degree_vector));

if network_topo.numNodes > 1
    graph_components = conncomp(graph(network_topo.adj_matrix));
    if numel(unique(graph_components)) ~= 1
        error('ADMM_EKF:DisconnectedGraph', ...
            ['Consensus ADMM requires a connected graph. ', ...
             'Increase network_topo.com_rad_CR.']);
    end
end

measurement_multiplicity = zeros(network_topo.numNodes, 1);
for n = 1:network_topo.numNodes
    idx_set_n = get_measurement_index_set(n, network_topo, MEASUREMENT_MODE);
    measurement_multiplicity(idx_set_n) = ...
        measurement_multiplicity(idx_set_n) + 1;
end
if any(measurement_multiplicity ~= 1)
    error('ADMM_EKF:MeasurementDoubleCounting', ...
        ['The sum of local objectives must contain every physical ', ...
         'measurement exactly once. Multiplicities: %s'], ...
        mat2str(measurement_multiplicity.'));
end
% %% Test on different connectivity
% NUM_NEI_NODE = 2;


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

%% ---------------- Target ----------------
target.initial_position = [-30, -30];
target.speed = 50;
target.angle_degrees = 45;
target.direction = [cosd(target.angle_degrees), sind(target.angle_degrees)];
target.true_params = [target.initial_position(1), target.initial_position(2), ...
                      target.speed * target.direction(1), ...
                      target.speed * target.direction(2)];

target.target_position = zeros(NUM_TAR, M_total, 2);
target.target_position(1, 1, :) = target.initial_position;

DYNA_NOISE = false;
for t = 2:M_total
    target.target_position(1, t, :) = squeeze(target.target_position(1, t-1, :))' + ...
                                      target.speed * target.direction * dt;
    if DYNA_NOISE
        target.target_position(1, t, :) = target.target_position(1, t, :) + ...
            0.1 .* reshape((2*rand(1, 2)-1), 1, 1, 2);
    end
end

target_position_matrix = reshape(target.target_position, [M_total, 2]);
target_velocity_matrix = repmat(target.speed * target.direction, M_total, 1);
target.target_state = reshape( ...
    [target_position_matrix, target_velocity_matrix], [1, M_total, 4]);

% Same randomized reference trajectory used in run_sim_dkf_neighbor_consensus.m.
% pos = ut.gen_ref_trajectory(M_total, [-30 20 -30 25], dt, target.speed);
% vx = gradient(pos(:,1), dt); vy = gradient(pos(:,2), dt);
% target.target_state = reshape([pos, [vx,vy]],[1,M_total,4]);
% target.target_position = reshape(pos, [1, M_total, 2]);


%Other trajectory
% [t, states, info] = generateNonlinearTrajectory( ...
%     T = M_total*dt, ...
%     dt = dt, ...
%     start_pos = [-30.0, -30.0], ...
%     initial_speed =target.speed, ...
%     initial_heading_deg = 45.0, ...
%     end_pos = [20.0, 20.0], ...
%     enforce_endpoint = true, ...
%     speed_amp = 0.4, ...
%     speed_freq = 0.15, ...
%     turn_amp_deg = 50.0, ...
%     turn_freq = 0.1, ...
%     use_randomness = true ...
% );
% target.target_state = reshape(states,[1,M_total,4]);
% target.target_position = reshape(states(:,1:2), [1, M_total, 2]);
% Ensure target trajectory
% fig_ut.plot_setup(target.target_position, network_topo);
%% ---------------- DKF / CV model ----------------
% The generated truth is deterministic constant velocity, so Q=0 keeps the
% estimator and PCRLB on the same transition model. If nonzero acceleration
% noise is desired, inject the same Gaussian process into target_state.
PROCESS_ACCELERATION_VARIANCE = 0;
env.Q = PROCESS_ACCELERATION_VARIANCE * ...
              [dt^4/4, 0,      dt^3/2, 0;
                0,      dt^4/4, 0,      dt^3/2;
                dt^3/2, 0,      dt^2,   0;
                0,      dt^3/2, 0,      dt^2];

P0 = diag([1e4, 1e4, 1e6, 1e6]);
% P0 = diag([1e2, 1e2, 1e6, 1e6]);
% Use one immutable global prior mean for both distributed and centralized
% filters. Change this single value when testing initialization robustness.
x0 = target.true_params(:);

%% ---------------- Bayesian expected-FIM PCRLB ----------------
% Compute the complete bound once.  The helper samples x0 ~ N(mu0,P0) and
% w ~ N(0,Q), averages H'*(Sigma\H) before inversion, and treats the first
% measurement as an observation of the initial state.  Always pass the
% physical covariance env.Sigma because the helper Jacobian is unwhitened.
mu0_pcrlb = x0;
[PCRLB_all, PCRLB_info] = calculate_pcrlb_v5( ...
    mu0_pcrlb, P0, F, env.Q, network_topo.radar_pos, ...
    env.lambda, env.Sigma, M_total, PCRLB_NUM_STATE_SAMPLES, ...
    'Seed', PCRLB_SEED);

% Canonical, nonduplicated representation: [state x state x time].
Results.PCRLB_Bayes = PCRLB_all;

%% ---------------- Monte Carlo simulation ----------------
for mc = 1:num_monte_carlo
    rng(seed0 + mc, "twister");
    fprintf("\n[MC %d/%d]\n", mc, num_monte_carlo);

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

    % Store node-wise post-ADMM estimates for the same state-trace plots
    % used in run_sim_dkf_neighbor_consensus.m.
    all_est  = cell(M_total, network_topo.numNodes);
    all_pred = cell(M_total, network_topo.numNodes);

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
            fprintf("t=%d/%d\n", t, M_total);

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
                all_pred{t,n} = x_pred_nodes{n};
            end

            % ------------------------------------------------------------
            % 2) Nonlinear consensus ADMM measurement update
            %
            % Each node solves a local MAP/MLE objective with fmincon:
            %
            %   local negative log-posterior / likelihood
            %   + sum_j Nu_{n,j}'(theta_n - z_{n,j})
            %   + sum_j 0.5*(theta_n-z_{n,j})'*C*(theta_n-z_{n,j})
            %
            % This follows the structure in your reference snippet.
            % ------------------------------------------------------------
            [x_consensus, x_nodes_admm, P_nodes_admm, admm_info] = ...
                consensus_admm_nonlinear_tracking( ...
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
                % x_dadmm{n} = x_consensus; % Prior as consensus result
                P_dadmm{n} = P_nodes_admm{n};
                all_est{t,n} = x_nodes_admm(:, n);
            end

            % For logging, use the mean of node-wise posterior covariances
            % as an approximate covariance summary. The nonlinear fmincon
            % ADMM step itself does not directly return a covariance matrix.
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
            idx_all = (1:network_topo.numNodes).';
            y_all = stack_measurements(idx_all, range_meas, doppler_meas, t);
            [x_ca, P_ca] = centralized_ekf_update_stack(x_ca_pred,P_ca_pred,y_all,network_topo,env,models_ut);
            % ------------------------------------------------------------
            % 5) Logging
            % ------------------------------------------------------------
            true_params_k = squeeze(target.target_state(1,t,:))';

            % Column vectors are used for compatibility with
            % make_figs.plot_trajectory_and_network(...).
            all_tracking_params_raw{mc, t}    = x_consensus(:);
            all_tracking_params_ca_raw{mc, t} = x_ca(:);

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

            % Keep the legacy cell field for existing plotting/log code.
            % The Bayesian bound is independent of measurement-noise MC.
            Results.CRLB{mc, t} = PCRLB_all(:,:,t);

            if DEBUG && (t == 1 || mod(t, 64) == 0)
                fprintf("  ADMM iter=%d, r=%.3e, s=%.3e, est=[%.2f %.2f %.2f %.2f]\n", ...
                    admm_info.iter, admm_info.primal_residual(end), ...
                    admm_info.dual_residual(end), x_consensus);
            end
            if t == SNAPSHOT_T
                    snapshot = admm_info.x_hist;
                    % snapshot = permute(snapshot,[2,1,3]);
                    % snapshot_true = [target.target_position(1,t,1),target.target_position(1,t,2),14.141,14.141];
                    snapshot_true = true_params_k;
                    snapshot_ctrl = x_ca; % Get the result of Ctrl from CKF
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

        fprintf("Burst %d done. ADMM-DKF=[%.2f %.2f %.2f %.2f], CKF=[%.2f %.2f %.2f %.2f]\n", ...
            kBurst, Results.estimations_DA{mc, kBurst}, Results.estimations_CA{mc, kBurst});
    end
    %% Plotting
    % fig_ut.plot_trajectory_and_network(target.target_position, all_tracking_params_raw,network_topo);
    % fig_ut.plot_trajectory_and_network(target.target_position, all_tracking_params_ca_raw,network_topo);
    % fig_ut.plot_converge_mse_across_node_withCentrl(snapshot,snapshot_true,network_topo,snapshot_ctrl);
end

%% ---------------- Save log ----------------
if SAVE_LOG
    Log = struct();
    Log.RUN_NAME = RUN_NAME;
    Log.TYPE = TYPE;
    Log.NUM_TAR = NUM_TAR;
    Log.NUM_CPI_PER_MEA = NUM_CPI_PER_MEA;
    Log.NUM_DOPPLER_ESTIMATION_SAMPLES = ...
        NUM_DOPPLER_ESTIMATION_SAMPLES;
    Log.TRACK_TIME = TRACK_TIME;
    Log.mc = num_monte_carlo;
    Log.network_topo = network_topo;
    Log.constant = env;
    Log.target = target;
    Log.time_step = dt;
    Log.PRE_WHITEN = PRE_WHITEN;
    Log.MEASUREMENT_MODE = MEASUREMENT_MODE;
    Log.ADMM_OBJECTIVE_TYPE = ADMM_OBJECTIVE_TYPE;
    Log.ADMM_MAX_ITER = ADMM_MAX_ITER;
    Log.ADMM_TOLERANCE = ADMM_TOLERANCE;
    Log.ADMM_REL_TOLERANCE = ADMM_REL_TOLERANCE;
    Log.ADMM_C_PENALTY = ADMM_C_PENALTY;
    Log.ADMM_RHO_UPDATE_INTERVAL = ADMM_RHO_UPDATE_INTERVAL;
    Log.ADMM_RHO_BALANCE_MU = ADMM_RHO_BALANCE_MU;
    Log.ADMM_RHO_SCALE = ADMM_RHO_SCALE;
    Log.seed0 = seed0;
    Log.PCRLB_SEED = PCRLB_SEED;
    Log.P0 = P0;
    Log.x0 = x0;
    Log.F = F;
    Log.DYNA_NOISE = DYNA_NOISE;
    Log.PROCESS_ACCELERATION_VARIANCE = PROCESS_ACCELERATION_VARIANCE;
    Log.measurement_multiplicity = measurement_multiplicity;
    Log.created_at = char(datetime('now', 'Format', ...
        'yyyy-MM-dd HH:mm:ss.SSS Z'));
    Log.matlab_version = version;
    Log.computer = computer;
    source_paths = {which(mfilename), which('ADMM_utils'), ...
        which('models'), which('make_figs'), which('dkf_neighbor_update'), ...
        which('centralized_ekf_update_stack'), ...
        which('calculate_pcrlb_v5'), which('plot_pcrlb')};
    Log.source_provenance = collect_source_provenance(source_paths);
    Log.PCRLB_info = PCRLB_info;
    Log.Results = Results;
    configured_run_name = sprintf('%s_snr%g_cr%g', ...
        RUN_NAME, env.SNR_idx, network_topo.com_rad_CR);
    save_mat_Log(Log_DIR, configured_run_name, Log);
end

%% ---------------- Plot quick check ----------------
% These plots intentionally follow the structure in
% run_sim_dkf_neighbor_consensus.m:
%   1) node-wise DKF/EKF state traces,
%   2) target trajectory and network geometry,
%   3) node convergence across optimization iterations at one sample time.

if DEBUG
    start_idx = 1;
    tEnd = M_total;
    time_idx = 1:TRACK_TIME*NUM_CPI_PER_MEA;

    figure;
    subplot(2,2,1); hold on;
    for n = 1:network_topo.numNodes
        plot(start_idx:tEnd, cellfun(@(x) x(1), all_est(start_idx:tEnd,n)), ...
            'DisplayName', "post." + network_topo.labels{n});
    end
    plot(time_idx, target.target_state(1,:,1), ...
        'DisplayName', 'GT', 'LineStyle', '--', 'Color', 'k');
    title("DKF + ADMM X"); legend('show'); grid on; box on;

    subplot(2,2,2); hold on;
    for n = 1:network_topo.numNodes
        plot(start_idx:tEnd, cellfun(@(x) x(2), all_est(start_idx:tEnd,n)), ...
            'DisplayName', "post." + network_topo.labels{n});
    end
    plot(time_idx, target.target_state(1,:,2), ...
        'DisplayName', 'GT', 'LineStyle', '--', 'Color', 'k');
    title("DKF + ADMM Y"); legend('show'); grid on; box on;

    subplot(2,2,3); hold on;
    for n = 1:network_topo.numNodes
        plot(start_idx:tEnd, cellfun(@(x) x(3), all_est(start_idx:tEnd,n)), ...
            'DisplayName', "post." + network_topo.labels{n});
    end
    % yline(target.true_params(3), 'k--', 'GT');
    plot(time_idx, target.target_state(1,:,3), ...
        'DisplayName', 'GT', 'LineStyle', '--', 'Color', 'k');
    title("DKF + ADMM Vx"); legend('show'); grid on; box on;

    subplot(2,2,4); hold on;
    for n = 1:network_topo.numNodes
        plot(start_idx:tEnd, cellfun(@(x) x(4), all_est(start_idx:tEnd,n)), ...
            'DisplayName', "post." + network_topo.labels{n});
    end
    % yline(target.true_params(4), 'k--', 'GT');
    plot(time_idx, target.target_state(1,:,4), ...
        'DisplayName', 'GT', 'LineStyle', '--', 'Color', 'k');
    title("DKF + ADMM Vy"); legend('show'); grid on; box on;
end

if DEBUG
    % Trajectory and communication-network plot.
    fig_ut.plot_trajectory_and_network(target.target_position, ...
                            all_tracking_params_raw, network_topo);
    title("Distributed EKF + Consensus ADMM");

    % Randomly selected ADMM convergence snapshot.
    snapshot = Results.estimations_DA_nodes_hist_raw{1, SNAPSHOT_T};
    snapshot_true = Results.true_params_raw{1, SNAPSHOT_T};
    snapshot_ctrl = Results.estimations_CA_raw{1, SNAPSHOT_T};
    fig_ut.plot_converge_across_node_withCentrl( ...
        snapshot, snapshot_true, network_topo, snapshot_ctrl(:));
    sgtitle(sprintf( ...
        "ADMM node convergence at random time stamp t = %d", SNAPSHOT_T));
end
%% ========================================================================
% Local helper functions
%% ========================================================================
function [x_pred, P_pred] = predict_cv(x, P, F, Q)
    x_pred = F * x;
    P_pred = F * P * F' + Q;
    P_pred = symmetrize(P_pred);
end

function idx_set = get_measurement_index_set(n, network_topo, mode)
    switch string(mode)
        case "local_neighbor"
            % Explicitly include self so an isolated node never loses its
            % own measurement. This mode is for local estimators, not the
            % graph-wide consensus objective used by this script.
            idx_set = unique([n, ...
                find(network_topo.adj_matrix(n, :) > 0)], 'stable').';
        case "self_only"
            idx_set = n;
        otherwise
            error("Unknown MEASUREMENT_MODE: %s", string(mode));
    end
end

function y_bar = stack_measurements(idx_set, range_meas, doppler_meas, t)
    y_bar = zeros(2*numel(idx_set), 1);
    for a = 1:numel(idx_set)
        j = idx_set(a);
        y_bar(2*a-1) = range_meas(1, j, t);
        y_bar(2*a)   = doppler_meas(1, j, t);
    end
end

function [x_bar, x_nodes, P_nodes, info] = consensus_admm_nonlinear_tracking( ...
    x_pred_nodes, P_pred_nodes, range_meas, doppler_meas, t, ...
    network_topo, neighbors, env, models_ut, ut, objective_type, ...
    measurement_mode, PRE_WHITEN, max_iter, abs_tolerance, ...
    rel_tolerance, c_penalty, rho_update_interval, rho_balance_mu, ...
    rho_scale, verbose_every, DEBUG)

    N  = network_topo.numNodes;
    nx = numel(x_pred_nodes{1});

    if string(measurement_mode) ~= "self_only"
        error('ADMM_EKF:ConsensusObjectivePartition', ...
            ['Graph-wide consensus ADMM requires self_only measurements ', ...
             'so every physical measurement is counted exactly once.']);
    end
    if rho_update_interval < 1 || rho_update_interval ~= floor(rho_update_interval) || ...
            rho_balance_mu <= 1 || rho_scale <= 1
        error('ADMM_EKF:RhoAdaptationConfig', ...
            ['rho_update_interval must be a positive integer; ', ...
             'rho_balance_mu and rho_scale must be greater than one.']);
    end

    ADMM = ut.Initialized_ADMM(network_topo);
    ADMM.max_iter  = max_iter;
    ADMM.tolerance = abs_tolerance;
    ADMM.c_penalty = c_penalty(:).';

    ADMM.initial_values = zeros(nx, N);
    for n = 1:N
        ADMM.initial_values(:,n) = x_pred_nodes{n};
    end

    [prior_mean_common, prior_cov_common] = ...
        common_predicted_prior(x_pred_nodes, P_pred_nodes);

    % Sum_n (1/N)*prior(x) reconstructs one common predicted prior.
    prior_mean = {prior_mean_common};
    prior_cov  = {N * prior_cov_common};

    range_local = cell(1, N);
    doppler_local = cell(1, N);
    radarpos_local = cell(1, N);
    numNodes_local = zeros(1, N);
    for n = 1:N
        [range_local{n}, doppler_local{n}, radarpos_local{n}, ...
            numNodes_local(n)] = local_measurement_for_admm( ...
                n, range_meas, doppler_meas, t, network_topo, measurement_mode);
        if numNodes_local(n) ~= 1
            error('ADMM_EKF:LocalMeasurementCount', ...
                'Node %d must own exactly one measurement.', n);
        end
    end

    % Edge-variable initialization.
    for n = 1:N
        for j = neighbors{n}
            z0 = 0.5 * (ADMM.initial_values(:,n) + ADMM.initial_values(:,j));
            ADMM.update_z{n}(:,j)      = z0;
            ADMM.update_z_prev{n}(:,j) = z0;
            ADMM.Nu{n}(:,j)            = zeros(nx,1);
            ADMM.Nu_prev{n}(:,j)       = zeros(nx,1);
        end
    end

    all_estimations = ADMM.initial_values;
    iteration = 0;
    stop_reason = "max_iter";

    ADMM.primal_residual_all = zeros(1, max_iter);
    ADMM.dual_residual_all = zeros(1, max_iter);
    ADMM.eps_pri_all = zeros(1, max_iter);
    ADMM.eps_dual_all = zeros(1, max_iter);
    ADMM.c_penalty_all = zeros(nx, max_iter);
    ADMM.all_estimations_every_iter = zeros(nx, N, max_iter);
    ADMM.primal_residual_by_para = cell(1, max_iter);
    ADMM.dual_residual_by_para = cell(1, max_iter);

    fmincon_exitflag = zeros(N,1);
    fmincon_failure_count = 0;
    fmincon_max_iterations = 0;

    num_directed_edges = sum(cellfun(@numel, neighbors));
    num_primal_entries = nx * num_directed_edges;

    while ~ADMM.converged && iteration < ADMM.max_iter
        iteration = iteration + 1;

        % ------------------------------------------------------------
        % 1) x update using lambda^k and z^k.
        % ------------------------------------------------------------
        for n = 1:N
            switch string(objective_type)
                case "MLE"
                    if PRE_WHITEN
                        fun = @(p) ut.logLikelihoodWithConsensus(p, ...
                            range_local{n}, doppler_local{n}, radarpos_local{n}, ...
                            numNodes_local(n), 1, env.lambda, env.Sigma_filter, ...
                            n, neighbors, ADMM.Nu_prev, ADMM.initial_values, ...
                            ADMM.update_z_prev, ADMM.c_penalty, env.pre_whit_L);
                    else
                        fun = @(p) ut.logLikelihoodWithConsensus(p, ...
                            range_local{n}, doppler_local{n}, radarpos_local{n}, ...
                            numNodes_local(n), 1, env.lambda, env.Sigma_filter, ...
                            n, neighbors, ADMM.Nu_prev, ADMM.initial_values, ...
                            ADMM.update_z_prev, ADMM.c_penalty);
                    end

                case "MAP"
                    if PRE_WHITEN
                        fun = @(p) ut.posteriorWithConsensus(p, ...
                            range_local{n}, doppler_local{n}, prior_mean, prior_cov, ...
                            radarpos_local{n}, numNodes_local(n), 1, env.lambda, ...
                            env.Sigma_filter, n, neighbors, ADMM.Nu_prev, ...
                            ADMM.initial_values, ADMM.update_z_prev, ...
                            ADMM.c_penalty, env.pre_whit_L);
                    else
                        fun = @(p) ut.posteriorWithConsensus(p, ...
                            range_local{n}, doppler_local{n}, prior_mean, prior_cov, ...
                            radarpos_local{n}, numNodes_local(n), 1, env.lambda, ...
                            env.Sigma_filter, n, neighbors, ADMM.Nu_prev, ...
                            ADMM.initial_values, ADMM.update_z_prev, ...
                            ADMM.c_penalty);
                    end

                otherwise
                    error('Wrong ADMM_OBJECTIVE_TYPE setting: %s', ...
                        string(objective_type));
            end

            [x_opt, ~, exitflag, solver_output] = fmincon( ...
                fun, ADMM.initial_values(:,n), [], [], [], [], ...
                ADMM.lb(:), ADMM.ub(:), [], ADMM.solver);
            all_estimations(:,n) = x_opt(:);
            fmincon_exitflag(n) = exitflag;
            fmincon_failure_count = fmincon_failure_count + (exitflag <= 0);
            if isfield(solver_output, 'iterations')
                fmincon_max_iterations = max( ...
                    fmincon_max_iterations, solver_output.iterations);
            end
        end

        ADMM.all_estimations_every_iter(:,:,iteration) = all_estimations;

        % ------------------------------------------------------------
        % 2) z update using x^(k+1) and lambda^k.
        % ------------------------------------------------------------
        c_inv = 1 ./ ADMM.c_penalty(:);
        for n = 1:N
            for j = neighbors{n}
                ADMM.update_z{n}(:,j) = 0.5 * ( ...
                    all_estimations(:,n) + all_estimations(:,j) + ...
                    c_inv .* (ADMM.Nu_prev{n}(:,j) + ...
                              ADMM.Nu_prev{j}(:,n)));
            end
        end

        % ------------------------------------------------------------
        % 3) Unscaled dual update using x^(k+1) and z^(k+1).
        % ------------------------------------------------------------
        for n = 1:N
            for j = neighbors{n}
                ADMM.Nu{n}(:,j) = ADMM.Nu_prev{n}(:,j) + ...
                    ADMM.c_penalty(:) .* ...
                    (all_estimations(:,n) - ADMM.update_z{n}(:,j));
            end
        end

        % Standard stacked primal and dual residuals.
        primal_sq = 0;
        dual_sq = 0;
        x_norm_sq = 0;
        z_norm_sq = 0;
        lambda_norm_sq = 0;
        primal_by_node = zeros(nx, N);
        dual_by_node = zeros(nx, N);

        for n = 1:N
            lambda_sum = zeros(nx,1);
            dual_node = zeros(nx,1);
            for j = neighbors{n}
                r_edge = all_estimations(:,n) - ADMM.update_z{n}(:,j);
                s_edge = ADMM.c_penalty(:) .* ...
                    (ADMM.update_z{n}(:,j) - ADMM.update_z_prev{n}(:,j));

                primal_sq = primal_sq + sum(r_edge.^2);
                x_norm_sq = x_norm_sq + sum(all_estimations(:,n).^2);
                z_norm_sq = z_norm_sq + sum(ADMM.update_z{n}(:,j).^2);
                primal_by_node(:,n) = primal_by_node(:,n) + abs(r_edge);
                dual_node = dual_node + s_edge;
                lambda_sum = lambda_sum + ADMM.Nu{n}(:,j);
            end
            % Standard ADMM dual residual is A' * C * B * delta-z.  Its
            % node-n block is the signed sum over incident edge variables,
            % not the root-sum-square of separate edge contributions.
            dual_sq = dual_sq + sum(dual_node.^2);
            dual_by_node(:,n) = abs(dual_node);
            lambda_norm_sq = lambda_norm_sq + sum(lambda_sum.^2);
        end

        primal_residual = sqrt(primal_sq);
        dual_residual = sqrt(dual_sq);
        eps_pri = sqrt(num_primal_entries) * abs_tolerance + ...
            rel_tolerance * max(sqrt(x_norm_sq), sqrt(z_norm_sq));
        eps_dual = sqrt(N * nx) * abs_tolerance + ...
            rel_tolerance * sqrt(lambda_norm_sq);

        ADMM.primal_residual_all(iteration) = primal_residual;
        ADMM.dual_residual_all(iteration) = dual_residual;
        ADMM.eps_pri_all(iteration) = eps_pri;
        ADMM.eps_dual_all(iteration) = eps_dual;
        ADMM.c_penalty_all(:,iteration) = ADMM.c_penalty(:);
        ADMM.primal_residual_by_para{iteration} = primal_by_node;
        ADMM.dual_residual_by_para{iteration} = dual_by_node;

        if DEBUG && verbose_every > 0 && mod(iteration, verbose_every) == 0
            fprintf(['\rADMM iter %d, r %.3e/%.3e, ', ...
                     's %.3e/%.3e\n'], ...
                iteration, primal_residual, eps_pri, dual_residual, eps_dual);
        end

        ADMM.converged = primal_residual <= eps_pri && ...
                         dual_residual <= eps_dual;
        if ADMM.converged
            stop_reason = "converged";
        elseif mod(iteration, rho_update_interval) == 0
            % Residual balancing for an unscaled dual variable.
            if primal_residual > rho_balance_mu * dual_residual
                ADMM.c_penalty = rho_scale * ADMM.c_penalty;
            elseif dual_residual > rho_balance_mu * primal_residual
                ADMM.c_penalty = ADMM.c_penalty / rho_scale;
            end
        end

        ADMM.initial_values = all_estimations;
        ADMM.update_z_prev = ADMM.update_z;
        ADMM.Nu_prev = ADMM.Nu;
    end

    ADMM.primal_residual_all = ADMM.primal_residual_all(1:iteration);
    ADMM.dual_residual_all = ADMM.dual_residual_all(1:iteration);
    ADMM.eps_pri_all = ADMM.eps_pri_all(1:iteration);
    ADMM.eps_dual_all = ADMM.eps_dual_all(1:iteration);
    ADMM.c_penalty_all = ADMM.c_penalty_all(:,1:iteration);
    ADMM.all_estimations_every_iter = ...
        ADMM.all_estimations_every_iter(:,:,1:iteration);

    x_nodes = all_estimations;
    x_bar = mean(all_estimations, 2);

    % Gauss-Newton/Laplace covariance of the same global MAP posterior:
    % one common predicted prior and every physical measurement once.
    if string(objective_type) == "MAP"
        J_prior = prior_cov_common \ eye(nx);
    else
        % An MLE state update contains no recursive prior information.
        J_prior = zeros(nx, nx);
    end
    J_measurement = zeros(nx, nx);
    for n = 1:N
        [~, Hn] = models_ut.MeasureModelAndJacobian( ...
            x_bar, n, network_topo, env);
        J_measurement = J_measurement + Hn' * (env.Sigma \ Hn);
    end
    J_measurement = symmetrize(J_measurement);
    J_post = symmetrize(J_prior + J_measurement);
    P_global = symmetrize(J_post \ eye(nx));
    P_nodes = repmat({P_global}, 1, N);

    info = struct();
    info.iter = iteration;
    info.converged = ADMM.converged;
    info.stop_reason = stop_reason;
    info.primal_residual = ADMM.primal_residual_all;
    info.dual_residual = ADMM.dual_residual_all;
    info.eps_pri = ADMM.eps_pri_all;
    info.eps_dual = ADMM.eps_dual_all;
    info.primal_residual_by_para = ...
        ADMM.primal_residual_by_para(1:iteration);
    info.dual_residual_by_para = ...
        ADMM.dual_residual_by_para(1:iteration);
    info.x_hist = ADMM.all_estimations_every_iter;
    info.c_penalty_final = ADMM.c_penalty;
    info.c_penalty_history = ADMM.c_penalty_all;
    info.fmincon_exitflag = fmincon_exitflag;
    info.fmincon_failure_count = fmincon_failure_count;
    info.fmincon_max_iterations = fmincon_max_iterations;
    info.J_prior = J_prior;
    info.J_measurement = J_measurement;
    info.J_post = J_post;
    info.P_global = P_global;
end

function [mean_common, cov_common] = common_predicted_prior( ...
        x_pred_nodes, P_pred_nodes)
    N = numel(x_pred_nodes);
    nx = numel(x_pred_nodes{1});
    mean_common = mean(cat(2, x_pred_nodes{:}), 2);

    % The node priors describe the same recursively fused state and are
    % correlated. Average their information; never add it N times.
    information_common = zeros(nx, nx);
    for n = 1:N
        Pn = symmetrize(P_pred_nodes{n});
        information_common = information_common + (Pn \ eye(nx));
    end
    information_common = symmetrize(information_common / N);
    cov_common = symmetrize(information_common \ eye(nx));
end

function [range_local, doppler_local, radarpos_local, numNodes_local, idx_set] = ...
    local_measurement_for_admm(n, range_meas, doppler_meas, t, network_topo, mode)

    idx_set = get_measurement_index_set(n, network_topo, mode);
    numNodes_local = numel(idx_set);

    % The recursive simulation updates at every measurement sample, so the
    % local M in ADMM_utils.logLikelihood/MAP is set to 1.  The resulting
    % matrices have size [1 x numNodes_local]. -> Can be set to NUM_CPI
    range_local   = zeros(1, numNodes_local);
    doppler_local = zeros(1, numNodes_local);
    for a = 1:numNodes_local
        j = idx_set(a);
        range_local(1,a)   = range_meas(1, j, t);
        doppler_local(1,a) = doppler_meas(1, j, t);
    end
    radarpos_local = network_topo.radar_pos(idx_set, :);
end

function A = symmetrize(A)
    A = 0.5 * (A + A');
end

function plot_rmse_quick(Results, M_total)
    rmse_da = zeros(M_total, 4);
    rmse_ca = zeros(M_total, 4);
    num_mc = size(Results.estimations_DA_raw, 1);
    for t = 1:M_total
        error_da = zeros(num_mc, 4);
        error_ca = zeros(num_mc, 4);
        for mc = 1:num_mc
            truth = Results.true_params_raw{mc,t}(:).';
            error_da(mc,:) = ...
                Results.estimations_DA_raw{mc,t}(:).' - truth;
            error_ca(mc,:) = ...
                Results.estimations_CA_raw{mc,t}(:).' - truth;
        end
        rmse_da(t,:) = sqrt(mean(error_da.^2, 1));
        rmse_ca(t,:) = sqrt(mean(error_ca.^2, 1));
    end

    names = {'x', 'y', 'v_x', 'v_y'};
    figure('Color', 'white');
    for p = 1:4
        subplot(2, 2, p); hold on; grid on;
        plot(1:M_total, rmse_da(:, p), 'LineWidth', 1.4, ...
             'DisplayName', 'DKF + ADMM');
        plot(1:M_total, rmse_ca(:, p), '--', 'LineWidth', 1.4, ...
             'DisplayName', 'Centralized EKF');
        xlabel('time index');
        ylabel('RMSE');
        title(names{p});
        legend('Location', 'best');
    end
end

function provenance = collect_source_provenance(source_paths)
    source_paths = source_paths(~cellfun(@isempty, source_paths));
    provenance = repmat(struct( ...
        'path', '', 'modified', '', 'bytes', 0, 'contents', ''), ...
        1, numel(source_paths));

    for k = 1:numel(source_paths)
        source_path = source_paths{k};
        file_info = dir(source_path);
        provenance(k).path = source_path;
        provenance(k).modified = file_info.date;
        provenance(k).bytes = file_info.bytes;
        % The working tree may be dirty, so a commit hash alone is not
        % sufficient. Store the exact active source with each experiment.
        provenance(k).contents = fileread(source_path);
    end
end

function save_mat_Log(root_dir, run_name, Log)
    if ~exist(root_dir, "dir")
        mkdir(root_dir);
    end
    ts = datetime("now", "Format", "yyyyMMdd_HHmmss");
    folder = fullfile(root_dir, sprintf("%s_%s", run_name, string(ts)));
    mkdir(folder);
    fn = fullfile(folder, "Log.mat");
    save(fn, "Log");
    fprintf("[Log] saved to %s\n", fn);
end
