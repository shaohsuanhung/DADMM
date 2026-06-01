clc; clear; close all;

ut        = ADMM_utils;
models_ut = models;
fig_ut = make_figs(10);

DEBUG      = true;
PRE_WHITEN = false;   % handled inside dkf_neighbor_update_pw.m
SAVE_LOG   = true;
Log_DIR    = "./data_log";
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
%   dkf_neighbor_update_pw.m
%
% Not required:
%   consensus.m
%   consensus_covariance.m
%% ========================================================================

%% ---------------- P1: Simulation parameters ----------------
NUM_CPI_PER_MEA = 64;
TRACK_TIME      = 1;
dt              = 1e-2;
NUM_TAR         = 1;
M_total         = NUM_CPI_PER_MEA * TRACK_TIME;

% -----------------------
% MC config
% -----------------------
num_monte_carlo = 1;
seed0           = 43;

%% ---------------- ADMM config ----------------
% The ADMM part follows the nonlinear consensus-ADMM style in which each
% node solves a local MLE/MAP problem with fmincon and an edge-wise
% consensus penalty. No consensus.m or consensus_covariance.m is used.
%
% For a recursive EKF-style simulation, MAP is the natural choice because
% the predicted state/covariance at each node provides the prior. MLE is
% kept as an option for debugging or for the first step if desired.
MEASUREMENT_MODE     = "local_neighbor";   % "local_neighbor" or "self_only"
ADMM_OBJECTIVE_TYPE  = "MAP";              % "MAP" or "MLE"

ADMM_MAX_ITER        = 500;
ADMM_TOLERANCE       = 1e-3;
ADMM_C_PENALTY       = [100, 100, 15, 15];
ADMM_VERBOSE_EVERY   = 50;

% Randomly pick one sample index for the ADMM node-convergence snapshot.
% This is selected once so that the plotted snapshot corresponds to one
% concrete recursive EKF time step.
rng(seed0, "twister");
SNAPSHOT_T = randi(M_total);
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
Results.true_params_raw          = cell(num_monte_carlo, M_total);
Results.estimations_CA_raw       = cell(num_monte_carlo, M_total);
Results.estimations_CA_sigma_raw = cell(num_monte_carlo, M_total);
Results.convg_iter_raw           = cell(num_monte_carlo, M_total);
Results.CRLB                     = cell(num_monte_carlo, M_total);

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

[network_topo.adj_matrix, network_topo.degree_matrix, ...
 network_topo.laplacian_matrix, network_topo.inc_matrix, ...
 network_topo.weights_matrix] = ut.calculate_all_graph_matrix( ...
    network_topo.radar_pos, network_topo.com_rad_CR, network_topo.numNodes);

neighbors = ut.get_neighbors(network_topo.adj_matrix, network_topo.numNodes);

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

env.SNR_idx = 50;
env.SNR_lin = 10^(env.SNR_idx/10);

env.range_var   = (3 * env.c^2) / (8 * pi^2 * env.B(1)^2 * env.SNR_lin);
env.doppler_var = (3 * (env.fs(1)^2)) / ...
                  (pi^2 * env.SNR_lin * NUM_CPI_PER_MEA^3);
env.rho = 0.0;

env.Sigma = [env.range_var,                         env.rho*env.range_var*env.doppler_var;
             env.rho*env.range_var*env.doppler_var, env.doppler_var];

env.PRE_WHITEN = PRE_WHITEN;
if env.PRE_WHITEN
    % L*Sigma*L' = I
    env.pre_whit_L    = inv(chol(env.Sigma, 'upper')');
    env.Sigma_filter  = eye(2);
else
    env.pre_whit_L    = eye(2);
    env.Sigma_filter  = env.Sigma;
end

%% ---------------- Target ----------------
target.initial_position = [-30, -30];
target.speed = 20;
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

target.target_state = reshape([squeeze(target.target_position), repmat([target.speed * target.direction(1),target.speed * target.direction(2)],M_total,1)]...
                                ,[1,M_total,4]);

% Same randomized reference trajectory used in run_sim_dkf_neighbor_consensus.m.
pos = ut.gen_ref_trajectory(M_total, [-30 20 -30 25], dt, target.speed);
vx = gradient(pos(:,1), dt); vy = gradient(pos(:,2), dt);
target.target_state = reshape([pos, [vx,vy]],[1,M_total,4]);
target.target_position = reshape(pos, [1, M_total, 2]);

%% ---------------- DKF / CV model ----------------
env.Q = 1e-2 * [dt^4/4, 0,      dt^3/2, 0;
                0,      dt^4/4, 0,      dt^3/2;
                dt^3/2, 0,      dt^2,   0;
                0,      dt^3/2, 0,      dt^2];

P0 = diag([1e6, 1e6, 1e4, 1e4]);
% x0 = [-30; -30; 14.1412; 14.1412];
x0 = squeeze(target.target_state(1,1,:));

%% ---------------- Monte Carlo simulation ----------------
for mc = 1:num_monte_carlo
    rng(seed0 + mc, "twister");
    fprintf("\n[MC %d/%d]\n", mc, num_monte_carlo);

    x_dadmm = cell(1, network_topo.numNodes);
    P_dadmm = cell(1, network_topo.numNodes);
    for n = 1:network_topo.numNodes
        x_dadmm{n} = x0 - [network_topo.radar_pos(n,1),network_topo.radar_pos(n,2),0,0]';
        P_dadmm{n} = P0;
    end

    % Centralized EKF baseline.
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

    [range_true, doppler_true, meas_true] = ut.gt_data_generation_tracking( ...
        range_true, doppler_true, meas_true, target, network_topo, env, ...
        M_total, NUM_TAR);

    [range_meas, doppler_meas, ~] = ut.add_measurement_noise( ...
        range_true, doppler_true, M_total, NUM_TAR, network_topo.numNodes, env);

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
                [x_pred_nodes{n}, P_pred_nodes{n}] = predict_cv( ...
                    x_dadmm{n}, P_dadmm{n}, F, env.Q);
                all_pred{t,n} = x_pred_nodes{n};
            end

            % ------------------------------------------------------------
            % 2) Nonlinear consensus ADMM measurement update
            %
            % Each node solves a local MAP/MLE objective with fmincon:
            %
            %   local negative log-posterior / likelihood
            %   + sum_j Nu_{n,j}'(theta_n - z_{n,j})
            %   + sum_j || (c/2) .* (theta_n - z_{n,j}) ||^2
            %
            % This follows the structure in your reference snippet.
            % ------------------------------------------------------------
            [x_consensus, x_nodes_admm, P_nodes_admm, admm_info] = ...
                consensus_admm_nonlinear_tracking( ...
                    x_pred_nodes, P_pred_nodes, range_meas, doppler_meas, t, ...
                    network_topo, neighbors, env, models_ut, ut, ...
                    ADMM_OBJECTIVE_TYPE, MEASUREMENT_MODE, PRE_WHITEN, ...
                    ADMM_MAX_ITER, ADMM_TOLERANCE, ADMM_C_PENALTY, ...
                    ADMM_VERBOSE_EVERY, DEBUG);

            % Feed the ADMM posterior back to the distributed filters.
            % x_nodes_admm(:,n) is retained so the recursion remains
            % distributed if ADMM stops before exact consensus.
            for n = 1:network_topo.numNodes
                % x_dadmm{n} = x_nodes_admm(:, n);
                x_dadmm{n} = x_consensus; % Prior as consensus result
                P_dadmm{n} = P_nodes_admm{n};
                all_est{t,n} = x_nodes_admm(:, n);
            end

            % For logging, use the mean of node-wise posterior covariances
            % as an approximate covariance summary. The nonlinear fmincon
            % ADMM step itself does not directly return a covariance matrix.
            P_consensus = zeros(4,4);
            for n = 1:network_topo.numNodes
                P_consensus = P_consensus + P_nodes_admm{n};
            end
            P_consensus = symmetrize(P_consensus / network_topo.numNodes);


% ------------------------------------------------------------
            % 4) Centralized EKF baseline using all radar measurements once
            % ------------------------------------------------------------
            [x_ca_pred, P_ca_pred] = predict_cv(x_ca, P_ca, F, env.Q);
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
            Results.true_params_raw{mc, t}          = true_params_k;
            Results.convg_iter_raw{mc, t}           = admm_info.iter;
            Results.primal_residauls_raw{mc, t}     = admm_info.primal_residual;
            Results.dual_residauls_raw{mc, t}       = admm_info.dual_residual;

            Results.estimations_CA_raw{mc, t}       = x_ca(:).';
            Results.estimations_CA_sigma_raw{mc, t} = P_ca;

            if t ~= 1
                Results.CRLB{mc, t} = ut.calculatePCRLB(true_params_k, ...
                    network_topo.radar_pos, network_topo.numNodes, ...
                    NUM_CPI_PER_MEA, env.lambda, env.Sigma_filter, env.Q, ...
                    inv(Results.CRLB{mc, t-1}), F);
            else
                Results.CRLB{mc, t} = ut.calculatePCRLB(true_params_k, ...
                    network_topo.radar_pos, network_topo.numNodes, ...
                    NUM_CPI_PER_MEA, env.lambda, env.Sigma_filter, env.Q, 0, F);
            end

            if DEBUG && (t == 1 || mod(t, 64) == 0)
                fprintf("  ADMM iter=%d, r=%.3e, s=%.3e, est=[%.2f %.2f %.2f %.2f]\n", ...
                    admm_info.iter, admm_info.primal_residual(end), ...
                    admm_info.dual_residual(end), x_consensus);
            end
            if t == SNAPSHOT_T
                    snapshot = admm_info.x_hist;
                    % snapshot = permute(snapshot,[2,1,3]);
                    % snapshot_true = [target.target_position(1,t,1),target.target_position(1,t,2),14.141,14.141];
                    snapshot_true = [target.target_state(1,t,1),target.target_state(1,t,2),target.target_state(1,t,3),target.target_state(1,t,4)];
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
    fig_ut.plot_trajectory_and_network(target.target_position, all_tracking_params_raw,network_topo);
    fig_ut.plot_trajectory_and_network(target.target_position, all_tracking_params_ca_raw,network_topo);
    fig_ut.plot_converge_mse_across_node_withCentrl(snapshot,snapshot_true,network_topo,snapshot_ctrl);
end

%% ---------------- Save log ----------------
if SAVE_LOG
    Log = struct();
    Log.RUN_NAME = RUN_NAME;
    Log.TYPE = TYPE;
    Log.NUM_TAR = NUM_TAR;
    Log.NUM_CPI_PER_MEA = NUM_CPI_PER_MEA;
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
    Log.ADMM_C_PENALTY = ADMM_C_PENALTY;
    Log.Results = Results;
    save_mat_Log(Log_DIR, RUN_NAME, Log);
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
    plot(time_idx, target.target_position(1,:,1), ...
        'DisplayName', 'GT', 'LineStyle', '--', 'Color', 'k');
    title("DKF + ADMM X"); legend('show'); grid on; box on;

    subplot(2,2,2); hold on;
    for n = 1:network_topo.numNodes
        plot(start_idx:tEnd, cellfun(@(x) x(2), all_est(start_idx:tEnd,n)), ...
            'DisplayName', "post." + network_topo.labels{n});
    end
    plot(time_idx, target.target_position(1,:,2), ...
        'DisplayName', 'GT', 'LineStyle', '--', 'Color', 'k');
    title("DKF + ADMM Y"); legend('show'); grid on; box on;

    subplot(2,2,3); hold on;
    for n = 1:network_topo.numNodes
        plot(start_idx:tEnd, cellfun(@(x) x(3), all_est(start_idx:tEnd,n)), ...
            'DisplayName', "post." + network_topo.labels{n});
    end
    yline(target.true_params(3), 'k--', 'GT');
    title("DKF + ADMM Vx"); legend('show'); grid on; box on;

    subplot(2,2,4); hold on;
    for n = 1:network_topo.numNodes
        plot(start_idx:tEnd, cellfun(@(x) x(4), all_est(start_idx:tEnd,n)), ...
            'DisplayName', "post." + network_topo.labels{n});
    end
    yline(target.true_params(4), 'k--', 'GT');
    title("DKF + ADMM Vy"); legend('show'); grid on; box on;
end

% Trajectory and communication-network plot.
fig_ut.plot_trajectory_and_network(target.target_position, ...
                        all_tracking_params_raw, network_topo);
title("Distributed EKF + Consensus ADMM");
% Randomly selected ADMM convergence snapshot.  The snapshot has size
% [4 x numNodes x numADMMIterations], matching make_figs.m.
snapshot = Results.estimations_DA_nodes_hist_raw{1, SNAPSHOT_T};
snapshot_true = Results.true_params_raw{1, SNAPSHOT_T};
snapshot_ctrl = Results.estimations_CA_raw{1, SNAPSHOT_T};

fig_ut.plot_converge_across_node_withCentrl( ...
        snapshot, snapshot_true, network_topo, snapshot_ctrl(:));
    sgtitle(sprintf("ADMM node convergence at random time stamp t = %d", SNAPSHOT_T));
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
            % This follows run_sim_dkf_neighbor_consensus.m:
            % laplacian row nonzeros include the diagonal entry and neighbors.
            idx_set = find(network_topo.laplacian_matrix(n, :) ~= 0).';
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
    measurement_mode, PRE_WHITEN, max_iter, tolerance, c_penalty, ...
    verbose_every, DEBUG)

    N  = network_topo.numNodes;
    nx = numel(x_pred_nodes{1});

    % Initialize ADMM fields in the same format as ADMM_utils.Initialized_ADMM.
    ADMM = ut.Initialized_ADMM(network_topo);
    ADMM.max_iter  = max_iter;
    ADMM.tolerance = tolerance;
    ADMM.c_penalty = c_penalty;

    % Use the EKF predictions as the initial local values.
    ADMM.initial_values = zeros(nx, N);
    for n = 1:N
        ADMM.initial_values(:,n) = x_pred_nodes{n};
        % ADMM.initial_values(:,n) = [0;0;0;0];
    end

    % Initialize z_{n,j} to the average of the two predicted states.  This
    % avoids a very large artificial first dual step from an all-zero z.
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
    primal_residual = inf;
    dual_residual   = inf;

    ADMM.primal_residual_all = zeros(1, max_iter);
    ADMM.dual_residual_all   = zeros(1, max_iter);
    ADMM.all_estimations_every_iter = zeros(nx, N, max_iter);
    ADMM.primal_residual_by_para = cell(1, max_iter);
    ADMM.dual_residual_by_para   = cell(1, max_iter);

    while ~ADMM.converged && iteration < ADMM.max_iter
        iteration = iteration + 1;
        if DEBUG && verbose_every > 0 && mod(iteration, verbose_every) == 0
            fprintf('\rADMM iter %d, Primal: %.3e, Dual: %.3e\n', ...
                iteration, primal_residual, dual_residual);
        end

        % ------------------------------------------------------------
        % Dual update, then local primal solve for each node.
        % This order follows the reference snippet provided by the user.
        % ------------------------------------------------------------
        for n = 1:N
            for j = neighbors{n}
                ADMM.Nu{n}(:,j) = ADMM.Nu_prev{n}(:,j) + ...
                    ADMM.c_penalty(:) .* ...
                    (ADMM.initial_values(:,n) - ADMM.update_z_prev{n}(:,j));
            end
            
            %TODO: Check here the correctness of the measurmeents
            [range_local, doppler_local, radarpos_local, numNodes_local, idx_set] = ...
                local_measurement_for_admm(n, range_meas, doppler_meas, t, ...
                    network_topo, measurement_mode);

            %Prior for MAP: collect predicted priors from the same
            %Local-neighbor set used in the measurement objective
            prior_mean = cell(1, numNodes_local);
            prior_cov  = cell(1, numNodes_local);

            %TODO: Check correctness here 
            for a = 1:numNodes_local
                jj = idx_set(a);
                prior_mean{a} = x_pred_nodes{jj};
                prior_cov{a}  = P_pred_nodes{jj};
            end

            switch string(objective_type)
                case "MLE"
                    if PRE_WHITEN
                        fun = @(p) ut.logLikelihoodWithConsensus(p, ...
                            range_local, doppler_local, radarpos_local, ...
                            numNodes_local, 1, env.lambda, env.Sigma_filter, ...
                            n, neighbors, ADMM.Nu, ADMM.initial_values, ...
                            ADMM.update_z_prev, ADMM.c_penalty, env.pre_whit_L);
                    else
                        fun = @(p) ut.logLikelihoodWithConsensus(p, ...
                            range_local, doppler_local, radarpos_local, ...
                            numNodes_local, 1, env.lambda, env.Sigma_filter, ...
                            n, neighbors, ADMM.Nu, ADMM.initial_values, ...
                            ADMM.update_z_prev, ADMM.c_penalty);
                    end

                case "MAP"
                    if PRE_WHITEN
                        fun = @(p) ut.posteriorWithConsensus(p, ...
                            range_local, doppler_local, prior_mean, prior_cov, ...
                            radarpos_local, numNodes_local, 1, env.lambda, ...
                            env.Sigma_filter, n, neighbors, ADMM.Nu, ...
                            ADMM.initial_values, ADMM.update_z_prev, ...
                            ADMM.c_penalty, env.pre_whit_L);
                    else
                        fun = @(p) ut.posteriorWithConsensus(p, ...
                            range_local, doppler_local, prior_mean, prior_cov, ...
                            radarpos_local, numNodes_local, 1, env.lambda, ...
                            env.Sigma_filter, n, neighbors, ADMM.Nu, ...
                            ADMM.initial_values, ADMM.update_z_prev, ...
                            ADMM.c_penalty);
                    end

                otherwise
                    error('Wrong ADMM_OBJECTIVE_TYPE setting: %s', string(objective_type));
            end

            all_estimations(:,n) = fmincon(fun, ADMM.initial_values(:,n), ...
                [], [], [], [], ADMM.lb, ADMM.ub, [], ADMM.solver);
        end

        ADMM.all_estimations_every_iter(:,:,iteration) = all_estimations;

        % ------------------------------------------------------------
        % z update.
        % ------------------------------------------------------------
        for n = 1:N
            for j = neighbors{n}
                c_inv = 1 ./ ADMM.c_penalty(:);
                ADMM.update_z{n}(:,j) = 0.5 * ( ...
                    c_inv .* (ADMM.Nu{n}(:,j) + ADMM.Nu{j}(:,n)) ...
                    + all_estimations(:,n) + all_estimations(:,j));
            end
        end

        % ------------------------------------------------------------
        % Residuals.
        % ------------------------------------------------------------
        primal_residual = 0;
        dual_residual   = 0;
        prima_by_node = zeros(nx, N);
        dual_by_node  = zeros(nx, N);

        for n = 1:N
            pr = zeros(nx,1);
            dr = zeros(nx,1);
            for j = neighbors{n}
                primal_residual = primal_residual + ...
                    sqrt(norm(all_estimations(:,n) - ADMM.update_z{n}(:,j))^2);
                dual_residual = dual_residual + ...
                    sqrt(norm(ADMM.Nu{n}(:,j) - ADMM.Nu_prev{n}(:,j))^2);
                pr = pr + abs(all_estimations(:,n) - ADMM.update_z{n}(:,j));
                dr = dr + abs(ADMM.Nu{n}(:,j) - ADMM.Nu_prev{n}(:,j));
            end
            prima_by_node(:,n) = pr;
            dual_by_node(:,n)  = dr;
        end

        ADMM.primal_residual_all(iteration) = primal_residual;
        ADMM.dual_residual_all(iteration)   = dual_residual;
        ADMM.primal_residual_by_para{iteration} = prima_by_node;
        ADMM.dual_residual_by_para{iteration}   = dual_by_node;

        % Adaptive penalty update, same style as the provided snippet.
        if mod(iteration, 30) == 0
            if primal_residual < dual_residual
                ADMM.c_penalty = ADMM.tau_incr .* ADMM.c_penalty;
            elseif dual_residual < primal_residual
                ADMM.c_penalty = ADMM.c_penalty .* ((ADMM.tau_decr).^(-1));
            end
        end

        % Shift ADMM buffers.
        ADMM.initial_values = all_estimations;
        ADMM.update_z_prev = ADMM.update_z;
        ADMM.Nu_prev = ADMM.Nu;

        % Stop criterion.  The first condition follows the user's snippet;
        % the max-iteration condition prevents infinite loops.
        if primal_residual < ADMM.tolerance || iteration >= ADMM.max_iter
            ADMM.converged = true;
        end
    end

    %Trim logs.
    ADMM.primal_residual_all = ADMM.primal_residual_all(1:iteration);
    ADMM.dual_residual_all   = ADMM.dual_residual_all(1:iteration);
    ADMM.all_estimations_every_iter = ...
        ADMM.all_estimations_every_iter(:,:,1:iteration);

    x_nodes = all_estimations;
    x_bar   = mean(all_estimations, 2);

    % Approximate node covariance using the same local EKF linearized update
    % as the baseline DKF. This is only for recursive covariance propagation
    % and logging; the ADMM primal estimate itself is from fmincon.
    P_nodes = cell(1, N);
    for n = 1:N %H_{n,k}'*Sigma^{-1}*H_{n,k}
        idx_set = get_measurement_index_set(n, network_topo, measurement_mode);
        y_bar = stack_measurements(idx_set, range_meas, doppler_meas, t);
        [~, P_tmp, ~] = dkf_neighbor_update_pw( ...
                        x_pred_nodes{n}, P_pred_nodes{n}, y_bar, idx_set, ...
                        network_topo, env, models_ut);
        P_nodes{n} = symmetrize(P_tmp);
    end
    for n = 1:N
        for j = neighbors{n} 
            P_nodes{n} = inv(inv(P_nodes{n})+inv(P_nodes{j})); %Sum of information of self + neighbor nodes
        end
    end

    info = struct();
    info.iter = iteration;
    info.primal_residual = ADMM.primal_residual_all;
    info.dual_residual   = ADMM.dual_residual_all;
    info.primal_residual_by_para = ADMM.primal_residual_by_para(1:iteration);
    info.dual_residual_by_para   = ADMM.dual_residual_by_para(1:iteration);
    info.x_hist = ADMM.all_estimations_every_iter;
    info.c_penalty_final = ADMM.c_penalty;
   
end

function [range_local, doppler_local, radarpos_local, numNodes_local, idx_set] = ...
    local_measurement_for_admm(n, range_meas, doppler_meas, t, network_topo, mode)

    idx_set = get_measurement_index_set(n, network_topo, mode);
    numNodes_local = numel(idx_set);

    % The recursive simulation updates at every measurement sample, so the
    % local M in ADMM_utils.logLikelihood/MAP is set to 1.  The resulting
    % matrices have size [1 x numNodes_local].
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
    da = Results.estimations_DA_raw(1, 1:M_total);
    ca = Results.estimations_CA_raw(1, 1:M_total);
    gt = Results.true_params_raw(1, 1:M_total);

    rmse_da = zeros(M_total, 4);
    rmse_ca = zeros(M_total, 4);
    for t = 1:M_total
        rmse_da(t, :) = abs(da{t}(:).' - gt{t}(:).');
        rmse_ca(t, :) = abs(ca{t}(:).' - gt{t}(:).');
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
        ylabel('|error|');
        title(names{p});
        legend('Location', 'best');
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
