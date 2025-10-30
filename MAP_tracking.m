clc;clear;close all;
ut = ADMM_utils;
models_ut = models;
fig_ut = make_figs(10);
DEBUG = true;
PRE_WHITEN = true;
% P0, TODO: We can make the setting into config file, then we can sacve up to 150 lines
%% P1: Initialize ---
%-- P1-0: Simulation Scenarios parameters
num_monte_carlo = 1; % Number of monte carlo runs
NUM_CPI_PER_MEA = 64;    % Number of measurements per burst
TRACK_TIME = 4;   % Number of burst times
time_step = 1e-3;    % Time step between two measurements

% [DELETE AFTER DEBUGGING]
all_estimation_from_EKFs = cell(TRACK_TIME*NUM_CPI_PER_MEA,10);
x_pred_from_EKFs = cell(TRACK_TIME*NUM_CPI_PER_MEA,10);
P_pred_from_EKFs = cell(TRACK_TIME*NUM_CPI_PER_MEA,10);
z_corr_from_EKFs = cell(TRACK_TIME*NUM_CPI_PER_MEA,10);
x_corr_from_EKFs = cell(TRACK_TIME*NUM_CPI_PER_MEA,10);
P_corr_from_EKFs = cell(TRACK_TIME*NUM_CPI_PER_MEA,10);
% [DELETE AFTER DEBUGGING]ma

%-- P1-1 Target(s) and moving scenario, deterministic part 
%TODO: Make multiple targets scenario, can make target obj list
NUM_TAR=1;
target.initial_position = [1000, 1000];
target.speed = 20;
target.angle_degrees = [135]; % In deg.
angle_degrees = target.angle_degrees(num_monte_carlo);
target.direction = [cos(angle_degrees * pi / 180), sin(angle_degrees * pi / 180)];%     direction = direction / norm(direction);
target.true_params = [target.initial_position(1), target.initial_position(2), target.speed * target.direction(1), target.speed * target.direction(2)];
target.target_position = zeros(NUM_TAR,NUM_CPI_PER_MEA*TRACK_TIME, 2);
% Initial position for all targets
for i = 1 : NUM_TAR
    target.target_position(i, 1, :) = target.initial_position; 
end
% True target trajectory
for i = 1: NUM_TAR
    for t = 2 : NUM_CPI_PER_MEA*TRACK_TIME
        target.target_position(i, t, :) = squeeze(target.target_position(i, t-1, :))' + target.speed * target.direction * time_step;
    end
end

%-- P1-2 Network topology
%TODO: Make a obj list, for ablation study for different topo.
network_topo.numNodes = 10;
theta = linspace(0,2*pi, network_topo.numNodes+1);
network_topo.theta = theta(1:end-1);
network_topo.com_rad_CR = 3000; % communication radius range
network_topo.radius = 3000;     % spatial placement radius 
network_topo.radar_pos = network_topo.radius * [cos(network_topo.theta); sin(network_topo.theta)]';
network_topo.C_distance = 1;  % Cost per meter
network_topo.C_data = 1;      % Cost per byte
network_topo.distances_between_radar_nodes = zeros(network_topo.numNodes,network_topo.numNodes);
network_topo.labels = {'N1','N2','N3','N4','N5','N6','N7','N8','N9','N10'};
for i = 1:network_topo.numNodes
    for j = 1:network_topo.numNodes
        network_topo.distances_between_radar_nodes(i,j) = norm(network_topo.radar_pos(i,:) - network_topo.radar_pos(j,:));
    end
end

[network_topo.adj_matrix, network_topo.degree_matrix,...
 network_topo.laplacian_matrix, network.inc_matrix, ...
 network_topo.weights_matrix] = ut.calculate_all_graph_matrix(network_topo.radar_pos, network_topo.com_rad_CR, network_topo.numNodes);

target.tartget_state_wrt_node = cell(TRACK_TIME*NUM_CPI_PER_MEA,10);
for i = 1:10
    for t = 1:TRACK_TIME*NUM_CPI_PER_MEA
        % target.tartget_state_wrt_node{t,i} = squeeze(target.target_position(1,t,:)) - network_topo.radar_pos(i,:)';
        target.target_state_wrt_node{t,i} =  [target.target_position(1,t,1), target.target_position(1,t,2), target.true_params(3), target.true_params(4)] - [network_topo.radar_pos(i,1), network_topo.radar_pos(i,2),0,0];
    end
end

%-- P1-3 Algoirthm parameters and solver setup, can change 
% Also effectiveness of the pre-whiterening. 
% Need Solver, -> to show 
% Go to distributed, 
options_Dctral = optimoptions('fmincon', 'Display', 'off', ScaleProblem=true, OptimalityTolerance=1e-2, FunctionTolerance=1e-6, StepTolerance=1e-6, MaxIterations=100);
options_Ctral =  optimoptions('fmincon', 'Display', 'off', ScaleProblem=true, OptimalityTolerance=1e-6, FunctionTolerance=1e-6, StepTolerance=1e-6, MaxIterations=100000);
ADMM = struct();
ADMM.solver = options_Dctral;
ADMM.converged= false;
ADMM.max_iter = 1500; % Maximumconverged ADMM iterations
ADMM.c_penalty = [100,100,15,15];
ADMM.lb = [-inf,-inf,-inf,-inf];
ADMM.ub = [inf,inf, inf, inf];
ADMM.initial_values = repmat([1000, 1000, 20, 20]', 1,network_topo.numNodes);
%
ADMM.prev_r = cell(1,network_topo.numNodes);
ADMM.prev_v = cell(1,network_topo.numNodes);
ADMM.prev_sigma_r = cell(1,network_topo.numNodes);
ADMM.prev_sigma_v = cell(1,network_topo.numNodes);
%
ADMM.Nu = cell(NUM_TAR, network_topo.numNodes);
ADMM.Nu_prev = cell(NUM_TAR, network_topo.numNodes);
ADMM.update_z = cell(NUM_TAR, network_topo.numNodes);
ADMM.update_z_prev = cell(NUM_TAR, network_topo.numNodes);
ADMM.primal_residual_all =[];
ADMM.primal_residual_by_para = cell(NUM_TAR,network_topo.numNodes);
ADMM.dual_residual_all = [];
ADMM.dual_residual_by_para = cell(NUM_TAR,network_topo.numNodes);
ADMM.all_estimations_every_iter = [];
% ADMM.all_estimations_every_iter = cell(NUM_TAR,TRACK_TIME, ADMM.max_iter); % In each cell, there will be a matrix of size (4 x numNodes)
ADMM.final_tracking_estimation = cell(NUM_TAR, TRACK_TIME);
ADMM.RANGE_Xs = [];
ADMM.RANGE_Ys = [];
ADMM.DOPPLER_Xs = [];
ADMM.DOPPLER_Ys = [];
ADMM.converg_r = false;
ADMM.converg_d = false;
ADMM.tolerance = 1e-3; % Convergence tolerance for primal residual
% Define the parameters for adaptive penalty update
% Define more conservative parameters for adaptive penalty update
ADMM.tau_incr = [2.01, 2.01, 2.1, 2.1];  % Smaller increase factor
ADMM.tau_decr = [2.01, 2.01, 2.1, 2.1];  % Smaller decrease factor
ADMM.mu = [3,3,10,10];          % Slightly smaller threshold ratio
ADMM.alpha = [0.5, 0.5, 0.5, 0.5];  % Damping factor
for n = 1:network_topo.numNodes
    ADMM.Nu{n} = zeros(4, network_topo.numNodes);
    ADMM.Nu_prev{n} = zeros(4, network_topo.numNodes);
    ADMM.update_z{n} = zeros(4, network_topo.numNodes);
    ADMM.update_z_prev{n} = zeros(4, network_topo.numNodes);
end


%-- P1-4 EKF & Signal & measurement model setup 
env.c = 3e8; % Speed of light
env.lambda = env.c / 10e9; % Wavelength
env.time_step = time_step; % Time step between two measurements
env.T = env.time_step / 2; % Pulse repetition interval
env.B = 10e6 * ones(1,network_topo.numNodes); % Bandwidth
env.fs = 2*env.B; % Sampling frequency
env.SNR_idx = 50;
env.SNR_lin = 10^(env.SNR_idx/10); % SNR in linear scale
% fprintf('SNR_db: %d dB, SNR_linear: %f\n', env.SNR_idx, env.SNR_lin);
% Noise variance is (2,2) = [rage_var, ro; ro, doppler_var]
env.range_var = (3 * env.c^2) / (8 * pi^2 * env.B(1)^2 * env.SNR_lin) ;
env.doppler_var = (3 * ((env.fs(1))^2)) / (pi^2 * env.SNR_lin * NUM_CPI_PER_MEA^3) ;
env.range_sd = sqrt(env.range_var);
env.doppler_sd = sqrt(env.doppler_var);
env.rho = 0.0;  

env.Sigma = [env.range_var, env.rho * env.range_sd * env.doppler_sd; ...
              env.rho * env.range_sd * env.doppler_sd, env.doppler_var];   
% [DELETE AFTER DEBUGGING] try to approximate the env.Sigma by choskly decomp
if PRE_WHITEN
    env.pre_whit_L = inv(chol(env.Sigma));
    env.Sigma_filter = eye(2);
else
    env.pre_whit_L = eye(2);
    env.Sigma_filter = env.Sigma;
end     


%-- [No need EKF here] EKF setup for each node

%-- P1-5 Save the experiment parameters log in json file
experiment_params_log = struct();
experiment_params_log.network_topo = network_topo;
experiment_params_log.NUM_CPI_PER_MEA = NUM_CPI_PER_MEA;
experiment_params_log.TRACK_TIME = TRACK_TIME;
experiment_params_log.time_step = time_step;
experiment_params_log.num_monte_carlo = num_monte_carlo;
if DEBUG==false
    ut.write_exp_log('./data_log', 'exp_config',experiment_params_log);
end


%% P2: Synthetic data generation
for mc = 1:num_monte_carlo
    fprintf('Monte Carlo run: %d\n', mc);
    for com_rad = 1:length(network_topo.com_rad_CR)
        fprintf('Monte Carlo run: %d\n', mc);
        %-- P2-1: Generate true target measurements for tracking (for each sensor node, so k x M x N)
        range_true = zeros(NUM_TAR,network_topo.numNodes, NUM_CPI_PER_MEA*TRACK_TIME);
        doppler_true = zeros(NUM_TAR, network_topo.numNodes, NUM_CPI_PER_MEA*TRACK_TIME); 
        measurements_true = zeros(NUM_TAR, network_topo.numNodes, 2*NUM_CPI_PER_MEA * TRACK_TIME);
        [range_true, doppler_true, measurements_true] = ut.gt_data_generation_tracking(range_true,doppler_true, measurements_true, target,network_topo,env, NUM_CPI_PER_MEA*TRACK_TIME, NUM_TAR);
        % prewhitening
        % range_true = env.pre_whit_L(1,1) * range_true + env.pre_whit_L(1,2) * doppler_true;
        % doppler_true = env.pre_whit_L(2,1) * range_true + env.pre_whit_L(2,2) * doppler_true;
        measurements_true_all = reshape(measurements_true, [], 1); % flatten

        %-- P2-2: Generate noisy target measurements
        % y_hat = y_gt + noise
        range_with_error = zeros(NUM_TAR, network_topo.numNodes,NUM_CPI_PER_MEA*TRACK_TIME);
        doppler_with_error = zeros(NUM_TAR, network_topo.numNodes, NUM_CPI_PER_MEA*TRACK_TIME); 
        measurements_with_error_all = zeros(NUM_TAR, network_topo.numNodes ,2 * NUM_CPI_PER_MEA*TRACK_TIME);
        [range_with_error, doppler_with_error, measurements_with_error_all] = ut.add_measurement_noise(range_true, doppler_true, NUM_CPI_PER_MEA*TRACK_TIME, NUM_TAR,network_topo.numNodes, env);

        %-- P2-3: Pharse data for algorithm input
        range_with_error_withNeighbors = cell(NUM_TAR, network_topo.numNodes);
        doppler_with_error_withNeighbors = cell(NUM_TAR, network_topo.numNodes);
        numNodes_withNeighbors = cell(NUM_TAR, network_topo.numNodes);
        radar_positions_withNeighbors = cell(NUM_TAR, network_topo.numNodes);

        range_with_error_cell_window = cell(NUM_TAR, network_topo.numNodes);
        doppler_with_error_cell_window = cell(NUM_TAR, network_topo.numNodes);
        numNodes_cell_buffer = cell(NUM_TAR, network_topo.numNodes);
        radar_positions_cell_buffer = cell(NUM_TAR, network_topo.numNodes);

        % Pharse_measurements
        [range_with_error_withNeighbors,doppler_with_error_withNeighbors,...
        numNodes_withNeighbors,radar_positions_withNeighbors] = ut.pharse_measurements_tracking(network_topo.laplacian_matrix, ...
        range_with_error, doppler_with_error, range_with_error_withNeighbors, doppler_with_error_withNeighbors, NUM_TAR,network_topo);
        %% P3: Decentralized optimization for tracking
        % Create a cell array to hold the neighbors of each node
        neighbors = ut.get_neighbors(network_topo.adj_matrix, network_topo.numNodes);
        for tar = 1: NUM_TAR
            for k = 1: TRACK_TIME
                t=0;
                % Measurement chunk
                for i = 1: network_topo.numNodes
                    range_with_error_cell_window{i} = range_with_error_withNeighbors{tar,i}((k-1)*NUM_CPI_PER_MEA + 1 : k*NUM_CPI_PER_MEA,:);
                    doppler_with_error_cell_window{i}  = doppler_with_error_withNeighbors{tar,i}((k-1)*NUM_CPI_PER_MEA + 1 : k*NUM_CPI_PER_MEA,:);
                end
                %-- P3-1: Local filtering (EKF), loop over time and each node 
                if k ~= 1
                    ADMM.initial_values = repmat(ADMM.final_tracking_estimation{tar, k-1}, 1,network_topo.numNodes); % Can initalize as local values?
                    % ADMM.initial_values = repmat([1000,1000,20,20]', 1,network_topo.numNodes);
                    for iter = 1: network_topo.numNodes
                        current_neighbors = find(network_topo.laplacian_matrix(n, :) ~= 0);
                        it = 0;
                        r_neigbors = zeros(1, length(current_neighbors));
                        v_neigbors = zeros(1, length(current_neighbors));
                        sigma_r_neigbors = zeros(1, length(current_neighbors));
                        sigma_v_neigbors = zeros(1, length(current_neighbors));
                        for j = current_neighbors
                            it = k+1;
                            % r_neigbors(:,it) = norm(ADMM.initial_values(1:2, j),2);
                            % v_neigbors(:,it) = norm(ADMM.initial_values(3:4, j),2);
                            % sigma_r_neigbors(:,:,it) = EKF.filter{j}.StateCovariance(1:2,1:2);
                            % sigma_v_neigbors(:,:,it) = EKF.filter{j}.StateCovariance(3:4,3:4);
                            % r_neigbors = mean(range_with_error_cell_window{j});  % local -> global and mean 
                            % v_neigbors = mean(doppler_with_error_cell_window{j});
                            % r_neigbors = norm(ADMM.initial_values(1:2, j),2);
                            % v_neigbors = norm(ADMM.initial_values(1:2, j),2);
                            % sigma_r_neigbors = var(range_with_error_cell_window{j});
                            % sigma_v_neigbors = var(doppler_with_error_cell_window{j});
                            r_neigbors(:,it) = 1000;
                            v_neigbors(:,it) = 20;
                            sigma_r_neigbors = [1, 1, 1];
                            sigma_v_neigbors = [1, 1, 1];
                        end
                        ADMM.prev_r{iter} = r_neigbors;
                        ADMM.prev_v{iter} = v_neigbors;
                        ADMM.prev_sigma_r{iter} = sigma_r_neigbors;
                        ADMM.prev_sigma_v{iter} = sigma_v_neigbors;                    
                    end
                end 
                % %-- P3-2: Distributed consensus optimization (ADMM)
                iteration = 0;
                all_estimations = zeros(4, network_topo.numNodes);
                while ~ADMM.converged && iteration < ADMM.max_iter
                    fprintf('Time step %d, ADMM iteration: %d\n', k, iteration);
                    iteration = iteration + 1;
                    for n = 1:network_topo.numNodes
                        for j = neighbors{n}
                            % Update nu
                            ADMM.Nu{n}(:, j) = ADMM.Nu_prev{n}(:, j) + ADMM.c_penalty' .* (ADMM.initial_values(:, n) - ADMM.update_z_prev{n}(:, j));
                        end
                        % fun = @(params) ut.logLikelihoodWithConsensus(params, range_with_error_withNeighbors{n}, doppler_with_error_withNeighbors{n},...
                        %                                      radar_positions_withNeighbors{n},numNodes_withNeighbors{n}, NUM_CPI_PER_MEA, env.lambda,...
                        %                                      env.Sigma, n, neighbors, ADMM.Nu, ADMM.initial_values,...
                        %                                       ADMM.update_z_prev, ADMM.c_penalty);
                        % estimated_params = fmincon(fun, ADMM.initial_values(:,n),[],[],[],[], ADMM.lb, ADMM.ub, [],ADMM.solver);
                        if k == 1
                            fun = @(params) ut.logLikelihoodWithConsensus(params, range_with_error_cell_window{n},doppler_with_error_cell_window{n},...
                                                             radar_positions_withNeighbors{n},numNodes_withNeighbors{n}, NUM_CPI_PER_MEA, env.lambda,...
                                                             env.Sigma, n, neighbors, ADMM.Nu, ADMM.initial_values,...
                                                              ADMM.update_z_prev, ADMM.c_penalty);
                        else
                            fun = @(params) ut.posteriorWithConsensus(params, range_with_error_cell_window{n},doppler_with_error_cell_window{n},...
                                                             ADMM.prev_r{n}, ADMM.prev_v{n}, ADMM.prev_sigma_r{n}, ADMM.prev_sigma_v{n},...
                                                             radar_positions_withNeighbors{n},numNodes_withNeighbors{n}, NUM_CPI_PER_MEA, env.lambda,...
                                                             env.Sigma, n, neighbors, ADMM.Nu, ADMM.initial_values,...
                                                             ADMM.update_z_prev, ADMM.c_penalty);
                        end

                        % save("variable.mat","range_with_error_cell_window","doppler_with_error_cell_window",...
                        %                                      "radar_positions_withNeighbors","numNodes_withNeighbors", "NUM_CPI_PER_MEA", "env",...
                        %                                       "n", "neighbors", "ADMM");
                        estimated_params = fmincon(fun, ADMM.initial_values(:,n),[],[],[],[], ADMM.lb, ADMM.ub, [], ADMM.solver);
                        all_estimations(:,n) = estimated_params;
                    end
                    ADMM.all_estimations_every_iter(:,:,iteration) = all_estimations;

                    for n = 1: network_topo.numNodes
                        for j = neighbors{n}
                            % Update z
                            ADMM.update_z{n}(:,j) = (1/2) * (((ADMM.c_penalty.^(-1))' .* (ADMM.Nu{n}(:,j) + ADMM.Nu{j}(:,n))) + all_estimations(:,n) + all_estimations(:, j)); 
                        end
                    end
                    
                    % Check convergence for this timestamp
                    primal_residual = 0;
                    dual_residual = 0; 
                    primal_residual_params = zeros(4, 1);
                    dual_residual_params   = zeros(4, 1);
                    prima_residual_by_node = zeros(4,network_topo.numNodes);
                    dual_residual_by_node  = zeros(4,network_topo.numNodes);
                    for n  = 1:network_topo.numNodes
                        for j = neighbors{n}
                            primal_residual = primal_residual + norm(all_estimations(:,n) - ADMM.update_z{n}(:,j))^2;
                            primal_residual_params = primal_residual_params + abs(all_estimations(:,n) - ADMM.update_z{n}(:,j));
                            dual_residual = dual_residual + norm((ADMM.Nu{n}(:,j) - ADMM.Nu_prev{n}(:,j)))^2;
                            dual_residual_params = dual_residual_params + abs(ADMM.Nu{n}(:,j) - ADMM.Nu_prev{n}(:,j));
                        end
                        prima_residual_by_node(:,n) = primal_residual_params;
                        dual_residual_by_node(:,n)  = dual_residual_params;
                    end
                    ADMM.primal_residual_all(iteration)     = primal_residual;
                    ADMM.dual_residual_all(iteration)       = dual_residual;
                    ADMM.primal_residual_by_para{iteration} = prima_residual_by_node;
                    ADMM.dual_residual_by_para{iteration}   = dual_residual_by_node; 

                    % Stop criteria
                    if primal_residual < 10* dual_residual
                        ADMM.c_penalty = ADMM.tau_incr .* ADMM.c_penalty;
                    elseif dual_residual < 10*primal_residual
                        ADMM.c_penalty = ADMM.c_penalty .* ((ADMM.tau_decr).^(-1));
                    end
                    
                    ADMM.initial_values = all_estimations;
                    ADMM.update_z_prev = ADMM.update_z;
                    ADMM.Nu_prev = ADMM.Nu;
                    if norm(ADMM.primal_residual_by_para{iteration}(1:2)) < ADMM.tolerance
                        ADMM.converg_r = true;
                        % Set the store range value of primal residual
                        if isempty(ADMM.RANGE_Xs)
                            if DEBUG
                                disp("[Debug] Range X params converge"+norm(ADMM.primal_residual_by_para{iteration}(1:2))+"<"+ADMM.tolerance);
                            end
                            ADMM.RANGE_Xs = all_estimations(1,:);
                        end
                        if isempty(ADMM.RANGE_Ys)
                            if DEBUG
                                disp("[Debug] Range Y params converge"+norm(ADMM.primal_residual_by_para{iteration}(1:2))+"<"+ADMM.tolerance);
                            end
                            ADMM.RANGE_Ys = all_estimations(2,:);
                        end
                        if not(isempty(ADMM.RANGE_Xs)) && not(isempty(ADMM.RANGE_Ys))
                            % Replace 
                            if DEBUG
                                disp("[Debug] Replace Range estimations "+mean(all_estimations(1,:))+","+mean(all_estimations(2,:))+" with "+mean(ADMM.RANGE_Xs)+","+mean(ADMM.RANGE_Ys)+")");
                            end
                            all_estimations(1,:) = ADMM.RANGE_Xs;
                            all_estimations(2,:) = ADMM.RANGE_Ys;
                        end
                    end
                    if norm(ADMM.primal_residual_by_para{iteration}(3:4)) < ADMM.tolerance
                        ADMM.converg_d = true;
                        % Set the store doppler value of primal residual
                        if isempty(ADMM.DOPPLER_Xs)
                            if DEBUG
                                disp("[Debug] Doppler X params converge"+norm(ADMM.primal_residual_by_para{iteration}(3:4))+"<"+ADMM.tolerance);
                            end
                            ADMM.DOPPLER_Xs = all_estimations(3,:);
                        end
                        if isempty(ADMM.DOPPLER_Ys)
                            ADMM.DOPPLER_Ys = all_estimations(4,:);
                            if DEBUG
                                disp("[Debug] Doppler Y params converge"+norm(ADMM.primal_residual_by_para{iteration}(3:4))+"<"+ADMM.tolerance);
                            end
                        end
                        if not(isempty(ADMM.DOPPLER_Ys)) && not(isempty(ADMM.DOPPLER_Xs))
                            % Replace 
                            if DEBUG
                                disp("[Debug] Replace Doppler estimations "+mean(all_estimations(3,:))+","+mean(all_estimations(4,:))+" with "+mean(ADMM.DOPPLER_Xs)+","+mean(ADMM.DOPPLER_Ys)+")");
                            end
                            all_estimations(3,:) = ADMM.DOPPLER_Xs;
                            all_estimations(4,:) = ADMM.DOPPLER_Ys;
                        end
                    end
                    if (ADMM.converg_r && ADMM.converg_d) || (iteration == ADMM.max_iter)
                        ADMM.converged = true;
                    end        

                end %TODO: Remenber to reset ADMM.iteration = 0; ADMM.converged = false ; after each time step, can write a re-set function in trackingEKF class
                ADMM.final_tracking_estimation{tar, k} = estimated_params;% Also de-whiten
                ADMM = ut.ADMM_reset(ADMM,NUM_TAR,network_topo);
                clear all_estimations;

                % end %TODO: Remenber to reset ADMM.iteration = 0; ADMM.converged = false ; after each time step, can write a re-set function in trackingEKF class
                % Plot EKF estimation result for each node one plot for each parameter


                fprintf('Time step %d completed.\n', k);

                % [Deleted after debugging]

            end 
        end
        %%
        % P4: Performance evaluation & Plotting 
        fig_ut.plot_trajectory(target.target_position,ADMM.final_tracking_estimation);
    end
    % Save the resulted estimation log in json file each monte carlo run
end

