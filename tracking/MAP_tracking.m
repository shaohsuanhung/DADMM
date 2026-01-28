clc; clear;
ut = ADMM_utils;
fig_ut = make_figs(10);

DEBUG = true;
PRE_WHITEN = false;   % <<<<<< toggle here

%% P1: Initialize ---
num_monte_carlo = 1;
NUM_CPI_PER_MEA = 64;
TRACK_TIME = 10;
time_step = 1e-2;

Results = struct();
Results.primal_residauls = cell(1,TRACK_TIME);
Results.dual_residauls   = cell(1,TRACK_TIME);
Results.estimations      = cell(1,TRACK_TIME);
Results.true_params      = cell(1,TRACK_TIME);
Results.estimations_CA   = cell(1,TRACK_TIME);
Results.convg_iter       = cell(1,TRACK_TIME);

%% Target
NUM_TAR = 1;
target.initial_position = [1000, 1000];
target.speed = 20;
target.angle_degrees = [135];
angle_degrees = target.angle_degrees(num_monte_carlo);

target.direction = [cos(angle_degrees*pi/180), sin(angle_degrees*pi/180)];
target.true_params = [target.initial_position(1), target.initial_position(2), ...
                      target.speed*target.direction(1), target.speed*target.direction(2)];

target.target_position = zeros(NUM_TAR, NUM_CPI_PER_MEA*TRACK_TIME, 2);
for i = 1:NUM_TAR
    target.target_position(i,1,:) = target.initial_position;
end
for i = 1:NUM_TAR
    for t = 2:NUM_CPI_PER_MEA*TRACK_TIME
        target.target_position(i,t,:) = squeeze(target.target_position(i,t-1,:))' + target.speed*target.direction*time_step;
    end
end
% [Try other trajectory]
% t  = linspace(0, 10, TRACK_TIME*NUM_CPI_PER_MEA)';x = t+1000;y = sin(t)+1000;traj = [x, y];
% target.target_position = reshape(traj,[1,TRACK_TIME*NUM_CPI_PER_MEA,2]);


%% Network topology
network_topo.numNodes = 10;
theta = linspace(0,2*pi, network_topo.numNodes+1);
network_topo.theta = theta(1:end-1);
network_topo.com_rad_CR = 3000;
network_topo.radius = 3000;
network_topo.radar_pos = network_topo.radius * [cos(network_topo.theta); sin(network_topo.theta)]';
network_topo.labels = {'N1','N2','N3','N4','N5','N6','N7','N8','N9','N10'};

network_topo.distances_between_radar_nodes = zeros(network_topo.numNodes);
for i = 1:network_topo.numNodes
    for j = 1:network_topo.numNodes
        network_topo.distances_between_radar_nodes(i,j) = norm(network_topo.radar_pos(i,:) - network_topo.radar_pos(j,:));
    end
end

[network_topo.adj_matrix, network_topo.degree_matrix, ...
 network_topo.laplacian_matrix, network.inc_matrix, ...
 network_topo.weights_matrix] = ut.calculate_all_graph_matrix(network_topo.radar_pos, network_topo.com_rad_CR, network_topo.numNodes);

%% ADMM config
options_Dctral = optimoptions('fmincon', 'Display','off', ScaleProblem=true, ...
    OptimalityTolerance=1e-2, FunctionTolerance=1e-6, StepTolerance=1e-6, MaxIterations=100);

ADMM = struct();
ADMM.solver = options_Dctral;
ADMM.converged = false;
ADMM.max_iter = 500;

ADMM.c_penalty = [100,100,15,15];
ADMM.lb = [-inf,-inf,-inf,-inf];
ADMM.ub = [ inf, inf, inf, inf];

ADMM.initial_values = repmat([1000, 1000, 10, 10]', 1, network_topo.numNodes);

ADMM.Nu = cell(NUM_TAR, network_topo.numNodes);
ADMM.Nu_prev = cell(NUM_TAR, network_topo.numNodes);
ADMM.update_z = cell(NUM_TAR, network_topo.numNodes);
ADMM.update_z_prev = cell(NUM_TAR, network_topo.numNodes);

ADMM.primal_residual_all = [];
ADMM.primal_residual_by_para = cell(NUM_TAR, network_topo.numNodes);
ADMM.dual_residual_all = [];
ADMM.dual_residual_by_para = cell(NUM_TAR, network_topo.numNodes);

ADMM.all_estimations_every_iter = [];
ADMM.final_tracking_estimation = cell(NUM_TAR, TRACK_TIME);

ADMM.RANGE_Xs = [];
ADMM.RANGE_Ys = [];
ADMM.DOPPLER_Xs = [];
ADMM.DOPPLER_Ys = [];
ADMM.converg_r = false;
ADMM.converg_d = false;

ADMM.tolerance = 1e-3;

ADMM.tau_incr = [2.01, 2.01, 2.1, 2.1];
ADMM.tau_decr = [2.01, 2.01, 2.1, 2.1];
ADMM.mu = [3,3,10,10];
ADMM.alpha = [0.5, 0.5, 0.5, 0.5];

for n = 1:network_topo.numNodes
    ADMM.Nu{n} = zeros(4, network_topo.numNodes);
    ADMM.Nu_prev{n} = zeros(4, network_topo.numNodes);
    ADMM.update_z{n} = zeros(4, network_topo.numNodes);
    ADMM.update_z_prev{n} = zeros(4, network_topo.numNodes);
end

%% Environment / measurement model
env.c = 3e8;
env.lambda = env.c / 10e9;
env.time_step = time_step;
env.T = env.time_step / 2;

env.B = 10e6 * ones(1,network_topo.numNodes);
env.fs = 2*env.B;

env.SNR_idx = 50;
env.SNR_lin = 10^(env.SNR_idx/10);

env.range_var   = (3 * env.c^2) / (8 * pi^2 * env.B(1)^2 * env.SNR_lin);
env.doppler_var = (3 * (env.fs(1)^2)) / (pi^2 * env.SNR_lin * NUM_CPI_PER_MEA^3);

env.range_sd   = sqrt(env.range_var);
env.doppler_sd = sqrt(env.doppler_var);
env.rho = 0;

env.Sigma = [env.range_var,   env.rho*env.range_sd*env.doppler_sd; ...
             env.rho*env.range_sd*env.doppler_sd, env.doppler_var];

%% [CHANGED] Correct pre-whitening definition
if PRE_WHITEN
    % chol returns upper R such that R'*R = Sigma
    % L = inv(R') -> L*Sigma*L' = I
    env.pre_whit_L = inv(chol(env.Sigma,'upper')');
    env.Sigma_filter = eye(2);
    env.PRE_WHITEN = true;
else
    env.pre_whit_L = eye(2);
    env.Sigma_filter = env.Sigma;
    env.PRE_WHITEN = false;
end

%% Prior
prior_mu = repmat({[1000; 1000; -14; 14]}, 1, network_topo.numNodes);

state_cov_raw = [env.range_var,0,0,0; ...
                 0,env.range_var,0,0; ...
                 0,0,env.doppler_var,0; ...
                 0,0,0,env.doppler_var];

%% [CHANGED] keep prior covariance consistent with PRE_WHITEN branch
if PRE_WHITEN
    prior_sigma = repmat({eye(4)}, 1, network_topo.numNodes);
    state_cov_for_neighbor_prior = eye(4);
else
    prior_sigma = repmat({state_cov_raw}, 1, network_topo.numNodes);
    state_cov_for_neighbor_prior = state_cov_raw;
end

%% P2: Synthetic data generation + tracking
neighbors = ut.get_neighbors(network_topo.adj_matrix, network_topo.numNodes);

for mc = 1:num_monte_carlo
    fprintf('Monte Carlo run: %d\n', mc);

    % -- P2-1 true measurements
    range_true = zeros(NUM_TAR, network_topo.numNodes, NUM_CPI_PER_MEA*TRACK_TIME);
    doppler_true = zeros(NUM_TAR, network_topo.numNodes, NUM_CPI_PER_MEA*TRACK_TIME);
    measurements_true = zeros(NUM_TAR, network_topo.numNodes, 2*NUM_CPI_PER_MEA*TRACK_TIME);

    [range_true, doppler_true, measurements_true] = ut.gt_data_generation_tracking( ...
        range_true, doppler_true, measurements_true, target, network_topo, env, ...
        NUM_CPI_PER_MEA*TRACK_TIME, NUM_TAR);

    % -- P2-2 noisy measurements (this function will pre-whiten if env.PRE_WHITEN=true)
    range_with_error = zeros(NUM_TAR, network_topo.numNodes, NUM_CPI_PER_MEA*TRACK_TIME);
    doppler_with_error = zeros(NUM_TAR, network_topo.numNodes, NUM_CPI_PER_MEA*TRACK_TIME);
    measurements_with_error_all = zeros(NUM_TAR, network_topo.numNodes, 2*NUM_CPI_PER_MEA*TRACK_TIME);

    [range_with_error, doppler_with_error, measurements_with_error_all] = ut.add_measurement_noise( ...
        range_true, doppler_true, NUM_CPI_PER_MEA*TRACK_TIME, NUM_TAR, network_topo.numNodes, env);

    % -- P2-3 parse neighbor measurements
    range_with_error_withNeighbors = cell(NUM_TAR, network_topo.numNodes);
    doppler_with_error_withNeighbors = cell(NUM_TAR, network_topo.numNodes);
    numNodes_withNeighbors = cell(NUM_TAR, network_topo.numNodes);
    radar_positions_withNeighbors = cell(NUM_TAR, network_topo.numNodes);

    range_with_error_cell_window = cell(1, network_topo.numNodes);
    doppler_with_error_cell_window = cell(1, network_topo.numNodes);

    prior_mu_cell = cell(1,network_topo.numNodes);
    prior_sigma_cell = cell(1,network_topo.numNodes);

    [range_with_error_withNeighbors, doppler_with_error_withNeighbors, ...
     numNodes_withNeighbors, radar_positions_withNeighbors, ...
     prior_mu_cell, prior_sigma_cell] = ut.pharse_measurements_tracking( ...
        network_topo.laplacian_matrix, range_with_error, doppler_with_error, ...
        range_with_error_withNeighbors, doppler_with_error_withNeighbors, ...
        NUM_TAR, network_topo, prior_mu, prior_sigma);

    for tar = 1:NUM_TAR
        for k = 1:TRACK_TIME

            % window data for this burst
            for iNode = 1:network_topo.numNodes
                range_with_error_cell_window{iNode} = range_with_error_withNeighbors{tar,iNode}((k-1)*NUM_CPI_PER_MEA+1 : k*NUM_CPI_PER_MEA, :);
                doppler_with_error_cell_window{iNode} = doppler_with_error_withNeighbors{tar,iNode}((k-1)*NUM_CPI_PER_MEA+1 : k*NUM_CPI_PER_MEA, :);
            end

            % init ADMM from previous step
            if k ~= 1
                ADMM.initial_values = repmat(ADMM.final_tracking_estimation{tar,k-1}, 1, network_topo.numNodes);

                estimated_prev = ADMM.final_tracking_estimation{tar,k-1};
                [ADMM.prior_mean, ADMM.prior_cov] = ut.get_neighbors_cell( ...
                    network_topo.laplacian_matrix, NUM_TAR, network_topo.numNodes, estimated_prev, state_cov_for_neighbor_prior);
            end

            iteration = 0;
            all_estimations = zeros(4, network_topo.numNodes);

            while ~ADMM.converged && iteration < ADMM.max_iter
                iteration = iteration + 1;
                if DEBUG
                    fprintf('Time step %d, ADMM iteration: %d\n', k, iteration);
                end

                for n = 1:network_topo.numNodes
                    for j = neighbors{n}
                        ADMM.Nu{n}(:,j) = ADMM.Nu_prev{n}(:,j) + ADMM.c_penalty' .* (ADMM.initial_values(:,n) - ADMM.update_z_prev{n}(:,j));
                    end

                    %% [CHANGED] use env.Sigma_filter (eye(2) if PRE_WHITEN, Sigma otherwise)
                    if k == 1
                        fun = @(params) ut.logLikelihoodWithConsensus( ...
                            params, range_with_error_cell_window{n}, doppler_with_error_cell_window{n}, ...
                            radar_positions_withNeighbors{n}, numNodes_withNeighbors{n}, NUM_CPI_PER_MEA, env.lambda, ...
                            env.Sigma_filter, n, neighbors, ADMM.Nu, ADMM.initial_values, ADMM.update_z_prev, ADMM.c_penalty);
                    else
                        fun = @(params) ut.posteriorWithConsensus( ...
                            params, range_with_error_cell_window{n}, doppler_with_error_cell_window{n}, ...
                            ADMM.prior_mean{n}, ADMM.prior_cov{n}, ...
                            radar_positions_withNeighbors{n}, numNodes_withNeighbors{n}, NUM_CPI_PER_MEA, env.lambda, ...
                            env.Sigma_filter, n, neighbors, ADMM.Nu, ADMM.initial_values, ADMM.update_z_prev, ADMM.c_penalty);
                    end

                    est_n = fmincon(fun, ADMM.initial_values(:,n), [],[],[],[], ADMM.lb, ADMM.ub, [], ADMM.solver);
                    all_estimations(:,n) = est_n;
                end

                ADMM.all_estimations_every_iter(:,:,iteration) = all_estimations;

                for n = 1:network_topo.numNodes
                    for j = neighbors{n}
                        ADMM.update_z{n}(:,j) = 0.5 * ( ((ADMM.c_penalty.^(-1))' .* (ADMM.Nu{n}(:,j) + ADMM.Nu{j}(:,n))) ...
                                                      + all_estimations(:,n) + all_estimations(:,j) );
                    end
                end

                % residuals
                primal_residual = 0;
                dual_residual = 0;
                primal_residual_params = zeros(4,1);
                dual_residual_params = zeros(4,1);

                prima_residual_by_node = zeros(4,network_topo.numNodes);
                dual_residual_by_node  = zeros(4,network_topo.numNodes);

                for n = 1:network_topo.numNodes
                    for j = neighbors{n}
                        primal_residual = primal_residual + norm(all_estimations(:,n) - ADMM.update_z{n}(:,j))^2;
                        primal_residual_params = primal_residual_params + abs(all_estimations(:,n) - ADMM.update_z{n}(:,j));

                        dual_residual = dual_residual + norm(ADMM.Nu{n}(:,j) - ADMM.Nu_prev{n}(:,j))^2;
                        dual_residual_params = dual_residual_params + abs(ADMM.Nu{n}(:,j) - ADMM.Nu_prev{n}(:,j));
                    end
                    prima_residual_by_node(:,n) = primal_residual_params;
                    dual_residual_by_node(:,n)  = dual_residual_params;
                end

                ADMM.primal_residual_all(iteration) = primal_residual;
                ADMM.dual_residual_all(iteration)   = dual_residual;
                ADMM.primal_residual_by_para{iteration} = prima_residual_by_node;
                ADMM.dual_residual_by_para{iteration}   = dual_residual_by_node;

                % update penalty occasionally
                if mod(iteration,30) == 0
                    if primal_residual < dual_residual
                        ADMM.c_penalty = ADMM.tau_incr .* ADMM.c_penalty;
                    elseif dual_residual < primal_residual
                        ADMM.c_penalty = ADMM.c_penalty .* (ADMM.tau_decr.^(-1));
                    end
                end

                % shift
                ADMM.initial_values = all_estimations;
                ADMM.update_z_prev = ADMM.update_z;
                ADMM.Nu_prev = ADMM.Nu;

                % stopping: reuse your helper (kept minimal)
                [all_estimations] = ut.ADMM_stop_criterion( ...
                    ADMM.primal_residual_by_para{iteration}, ADMM.tolerance, ...
                    all_estimations, ADMM.RANGE_Xs, ADMM.RANGE_Ys, ADMM.DOPPLER_Xs, ADMM.DOPPLER_Ys, ...
                    ADMM.converg_r, ADMM.converg_d, ADMM.converged, ...
                    DEBUG, iteration, ADMM.max_iter);

                % hard stop
                if primal_residual < ADMM.tolerance
                    ADMM.converged = true;
                end
            end

            %% [CHANGED] final estimate should NOT be "last node's estimated_params"
            x_hat_global = mean(all_estimations, 2);   % 4x1

            % store
            ADMM.final_tracking_estimation{tar,k} = x_hat_global;

            Results.primal_residauls{k} = ADMM.primal_residual_all;
            Results.dual_residauls{k}   = ADMM.dual_residual_all;
            Results.estimations{k}      = ADMM.all_estimations_every_iter;

            t_idx = k * NUM_CPI_PER_MEA;
            Results.true_params{k} = [squeeze(target.target_position(tar,t_idx,:))', target.true_params(3), target.true_params(4)];
            Results.convg_iter{k}  = iteration;

            % reset ADMM for next k
            ADMM = ut.ADMM_reset(ADMM, NUM_TAR, network_topo);

            if DEBUG
                fprintf('Time step %d completed. x_hat=[%.2f %.2f %.2f %.2f]\n', k, x_hat_global(1), x_hat_global(2), x_hat_global(3), x_hat_global(4));
            end
        end
    end
end

%% Plot
fig_ut.plot_trajectory(target.target_position, ADMM.final_tracking_estimation);
