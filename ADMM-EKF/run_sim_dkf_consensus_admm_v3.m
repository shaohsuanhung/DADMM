clc; clear; close all;

ut        = ADMM_utils;
models_ut = models;
fig_ut    = make_figs(10);

DEBUG      = true;
PRE_WHITEN = false;   % handled inside dkf_neighbor_update_pw.m
SAVE_LOG   = false;
Log_DIR    = "./data_log";
RUN_NAME   = "dkf_consensus_admm";
TYPE       = "DKF_ADMM";

%% ========================================================================
% Distributed EKF + Consensus ADMM
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
TRACK_TIME      = 10;
dt              = 1e-2;
NUM_TAR         = 1;
M_total         = NUM_CPI_PER_MEA * TRACK_TIME;

% -----------------------
% MC config
% -----------------------
num_monte_carlo = 1;
seed0           = 43;

%% ---------------- ADMM config ----------------
% local_neighbor: node n builds its local EKF quadratic using measurements
%                 from {n and its communication neighbors}; this matches
%                 run_sim_dkf_neighbor_consensus.m.
% self_only     : node n uses only its own radar measurement. This avoids
%                 repeated measurement counting and is closer to centralized
%                 all-radar EKF when ADMM converges.
MEASUREMENT_MODE = "local_neighbor";

% If every node has the same predicted prior and the ADMM objective sums all
% node costs, scaling the prediction information by 1/N avoids counting the
% same common prior N times. Set this to false if you want the exact local
% EKF Hessian from each node without this correction.
SCALE_COMMON_PRIOR = true;

ADMM = struct();
ADMM.max_iter = 100;
ADMM.rho_vec  = [100; 100; 15; 15];  % ADMM penalty for [x,y,vx,vy]
ADMM.abs_tol  = 1e-4;
ADMM.rel_tol  = 1e-3;
ADMM.verbose  = false;

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

neighbors = ut.get_neighbors(network_topo.adj_matrix, network_topo.numNodes); %#ok<NASGU>

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
env.rho = 0.01;

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
target.initial_position = [0, 0];
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

% Same randomized reference trajectory used in run_sim_dkf_neighbor_consensus.m.
pos = ut.gen_ref_trajectory(M_total, [-30 20 -30 25], dt, target.speed);
target.target_position = reshape(pos, [1, M_total, 2]);

%% ---------------- DKF / CV model ----------------
env.Q = 1e-2 * [dt^4/4, 0,      dt^3/2, 0;
                0,      dt^4/4, 0,      dt^3/2;
                dt^3/2, 0,      dt^2,   0;
                0,      dt^3/2, 0,      dt^2];

P0 = diag([1e6, 1e6, 1e4, 1e4]);
x0 = [-30; -30; 14.1412; 14.1412];

%% ---------------- Monte Carlo simulation ----------------
for mc = 1:num_monte_carlo
    rng(seed0 + mc, "twister");
    fprintf("\n[MC %d/%d]\n", mc, num_monte_carlo);

    x_dadmm = cell(1, network_topo.numNodes);
    P_dadmm = cell(1, network_topo.numNodes);
    for n = 1:network_topo.numNodes
        x_dadmm{n} = x0;
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
        M_total, NUM_TAR); %#ok<ASGLU>

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
            % 2) Build local EKF quadratic objectives
            %
            % dkf_neighbor_update_pw gives:
            %   P_post = inv(P_pred^{-1} + sumInfo)
            %   x_post = x_pred + P_post*sumInnov
            %
            % The equivalent local quadratic in theta_n is:
            %   f_n(theta_n) = 1/2 theta_n' A_n theta_n - b_n' theta_n
            % with
            %   A_n = P_pred^{-1} + sumInfo
            %   b_n = A_n*x_pred + sumInnov
            % ------------------------------------------------------------
            Acell       = cell(1, network_topo.numNodes);
            bcell       = cell(1, network_topo.numNodes);
            x_local_ekf = zeros(4, network_topo.numNodes);
            P_local_ekf = zeros(4, 4, network_topo.numNodes);

            for n = 1:network_topo.numNodes
                idx_set = get_measurement_index_set(n, network_topo, MEASUREMENT_MODE);
                y_bar = stack_measurements(idx_set, range_meas, doppler_meas, t);

                [x_tmp, P_tmp, dbg] = dkf_neighbor_update_pw( ...
                    x_pred_nodes{n}, P_pred_nodes{n}, y_bar, idx_set, ...
                    network_topo, env, models_ut);

                [Acell{n}, bcell{n}] = ekf_quadratic_from_dbg( ...
                    x_pred_nodes{n}, P_pred_nodes{n}, dbg, ...
                    SCALE_COMMON_PRIOR, network_topo.numNodes);

                x_local_ekf(:, n)   = x_tmp;
                P_local_ekf(:, :, n) = P_tmp;
            end

            % ------------------------------------------------------------
            % 3) Consensus ADMM over the communication graph
            % ------------------------------------------------------------
            [x_consensus, x_nodes_admm, admm_info] = consensus_admm_quadratic( ...
                Acell, bcell, network_topo.adj_matrix, ADMM, x_local_ekf);

            % Approximate fused covariance associated with the summed
            % quadratic objective. This replaces consensus_covariance.m.
            A_global = zeros(4, 4);
            for n = 1:network_topo.numNodes
                A_global = A_global + Acell{n};
            end
            P_consensus = symmetrize(pinv(A_global));

            % Feed the ADMM posterior back to the distributed filters.
            % x_nodes_admm(:,n) is used so the script still behaves as a
            % distributed recursion even if ADMM stops before exact consensus.
            for n = 1:network_topo.numNodes
                x_dadmm{n} = x_nodes_admm(:, n);
                P_dadmm{n} = P_consensus;
                all_est{t,n} = x_nodes_admm(:, n);
            end

            % ------------------------------------------------------------
            % 4) Centralized EKF baseline using all radar measurements once
            % ------------------------------------------------------------
            [x_ca_pred, P_ca_pred] = predict_cv(x_ca, P_ca, F, env.Q);
            idx_all = (1:network_topo.numNodes).';
            y_all = stack_measurements(idx_all, range_meas, doppler_meas, t);
            [x_ca, P_ca, ~] = dkf_neighbor_update_pw( ...
                x_ca_pred, P_ca_pred, y_all, idx_all, network_topo, env, models_ut);

            % ------------------------------------------------------------
            % 5) Logging
            % ------------------------------------------------------------
            true_params_k = [squeeze(target.target_position(1, t, :))', ...
                             target.true_params(3), target.true_params(4)];

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
    Log.SCALE_COMMON_PRIOR = SCALE_COMMON_PRIOR;
    Log.ADMM = ADMM;
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
try
    fig_ut.plot_trajectory_and_network(target.target_position, ...
        all_tracking_params_raw, network_topo);
    title("Distributed EKF + Consensus ADMM");
catch ME
    warning("Trajectory/network plotting skipped: %s", ME.message);
end

% Randomly selected ADMM convergence snapshot.  The snapshot has size
% [4 x numNodes x numADMMIterations], matching make_figs.m.
try
    snapshot = Results.estimations_DA_nodes_hist_raw{1, SNAPSHOT_T};
    snapshot_true = Results.true_params_raw{1, SNAPSHOT_T};
    snapshot_ctrl = Results.estimations_CA_raw{1, SNAPSHOT_T};

    fig_ut.plot_converge_across_node_withCentrl( ...
        snapshot, snapshot_true, network_topo, snapshot_ctrl(:));
    sgtitle(sprintf("ADMM node convergence at random time stamp t = %d", SNAPSHOT_T));
catch ME
    warning("ADMM convergence plotting skipped: %s", ME.message);
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

function [A, b] = ekf_quadratic_from_dbg(x_pred, P_pred, dbg, scale_common_prior, N)
    nx = numel(x_pred);
    Pinv = pinv(P_pred);
    if scale_common_prior
        Pinv = Pinv / N;
    end

    A = Pinv + dbg.sumInfo;
    b = A * x_pred + dbg.sumInnov;
    A = symmetrize(A + 1e-12*eye(nx));
end

function [x_bar, x_nodes, info] = consensus_admm_quadratic(Acell, bcell, adj_matrix, opts, x_init)
    N  = numel(Acell);
    nx = numel(bcell{1});

    [ei, ej] = find(triu(adj_matrix, 1) > 0);
    E = numel(ei);
    if E == 0
        error("The communication graph has no edges; consensus ADMM cannot run.");
    end

    rho_vec = opts.rho_vec(:);
    if isscalar(rho_vec)
        rho_vec = rho_vec * ones(nx, 1);
    end
    Rho = diag(rho_vec);

    x = x_init;
    if size(x, 1) ~= nx
        x = x.';
    end

    v   = zeros(nx, E);
    u_i = zeros(nx, E);
    u_j = zeros(nx, E);
    for e = 1:E
        v(:, e) = 0.5 * (x(:, ei(e)) + x(:, ej(e)));
    end

    primal_hist = zeros(opts.max_iter, 1);
    dual_hist   = zeros(opts.max_iter, 1);
    x_hist      = zeros(nx, N, opts.max_iter);

    for iter = 1:opts.max_iter
        v_prev = v;

        % x-update: separable over nodes.
        for n = 1:N
            Hn  = Acell{n};
            rhs = bcell{n};

            incident_left  = find(ei == n);
            incident_right = find(ej == n);

            for kk = 1:numel(incident_left)
                e = incident_left(kk);
                Hn  = Hn + Rho;
                rhs = rhs + Rho * (v(:, e) - u_i(:, e));
            end

            for kk = 1:numel(incident_right)
                e = incident_right(kk);
                Hn  = Hn + Rho;
                rhs = rhs + Rho * (v(:, e) - u_j(:, e));
            end

            x(:, n) = Hn \ rhs;
        end

        % Edge consensus-variable update.
        for e = 1:E
            i = ei(e);
            j = ej(e);
            v(:, e) = 0.5 * (x(:, i) + u_i(:, e) + x(:, j) + u_j(:, e));
        end

        % Scaled dual update.
        for e = 1:E
            i = ei(e);
            j = ej(e);
            u_i(:, e) = u_i(:, e) + x(:, i) - v(:, e);
            u_j(:, e) = u_j(:, e) + x(:, j) - v(:, e);
        end

        % Residuals.
        r2 = 0;
        s2 = 0;
        for e = 1:E
            i = ei(e);
            j = ej(e);
            r2 = r2 + norm(x(:, i)-v(:, e))^2 + norm(x(:, j)-v(:, e))^2;
            s2 = s2 + 2 * norm(Rho * (v(:, e)-v_prev(:, e)))^2;
        end
        primal_hist(iter) = sqrt(r2);
        dual_hist(iter)   = sqrt(s2);
        x_hist(:, :, iter) = x;

        eps_pri  = sqrt(2*E*nx)*opts.abs_tol + ...
                   opts.rel_tol * max(norm(x(:)), norm(v(:)));
        eps_dual = sqrt(2*E*nx)*opts.abs_tol + ...
                   opts.rel_tol * norm([u_i(:); u_j(:)]);

        if opts.verbose
            fprintf("    ADMM iter=%3d r=%.3e s=%.3e eps_pri=%.3e eps_dual=%.3e\n", ...
                iter, primal_hist(iter), dual_hist(iter), eps_pri, eps_dual);
        end

        if primal_hist(iter) <= eps_pri && dual_hist(iter) <= eps_dual
            break;
        end
    end

    primal_hist = primal_hist(1:iter);
    dual_hist   = dual_hist(1:iter);
    x_hist      = x_hist(:, :, 1:iter);

    x_nodes = x;
    x_bar   = mean(x, 2);

    info = struct();
    info.iter = iter;
    info.primal_residual = primal_hist;
    info.dual_residual = dual_hist;
    info.edge_i = ei;
    info.edge_j = ej;
    info.x_hist = x_hist;
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
