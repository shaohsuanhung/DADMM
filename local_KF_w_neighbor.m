clc;clear;close all;
ut = ADMM_utils;
models_ut = models;
fig_ut = make_figs(10);
DEBUG = true;
PRE_WHITEN = false; % Whether to pre-whiten the measurements
% P0, TODO: We can make the setting into config file (e.g., target, env, ...), then we can sacve up to 150 lines
%% P1: Initialize ---
%-- P1-0: Simulation Scenarios parameters
num_monte_carlo = 1; % Number of monte carlo runs
NUM_CPI_PER_MEA = 64;    % Number of measurements per burst
TRACK_TIME = 8;   % Number of burst times
time_step = 1e-3;    % Time step between two measurements
Results = struct();
Results.primal_residauls = cell(1,TRACK_TIME); % In each cell (time stemp), store the primal residuals (cell: 1 x # iteration of opt.) 
Results.dual_residauls = cell(1,TRACK_TIME);   % In each cell (time stemp), store the dual residuals (cell: 1 x # iteration of opt.)
Results.estimations = cell(1,TRACK_TIME);      % In each cell (time stemp), store the estimation results (cell: 4 x 10 x # iteration of opt.)
Results.true_params = cell(1,TRACK_TIME);      % In each cell (time stemp), store the estimation results (cell: 1 x 4)
Results.estimations_CA = cell(1,TRACK_TIME);   % In each cell (time stemp), store the estimation results from centralized approach (cell: 1 x 4)
Results.convg_iter = cell(1,TRACK_TIME);

% [DELETE AFTER DEBUGGING]
all_estimation_from_EKFs = cell(TRACK_TIME*NUM_CPI_PER_MEA,10);
x_pred_from_EKFs = cell(TRACK_TIME*NUM_CPI_PER_MEA,10);
P_pred_from_EKFs = cell(TRACK_TIME*NUM_CPI_PER_MEA,10);
z_corr_from_EKFs = cell(TRACK_TIME*NUM_CPI_PER_MEA,10);
x_corr_from_EKFs = cell(TRACK_TIME*NUM_CPI_PER_MEA,10);
P_corr_from_EKFs = cell(TRACK_TIME*NUM_CPI_PER_MEA,10);
% [DELETE AFTER DEBUGGING]

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

%-- P1-3 Algoirthm parameters and solver setup
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
% Prior information
ADMM.prev_mean = cell(1,network_topo.numNodes);
ADMM.prev_cov = cell(1,network_topo.numNodes);
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
if PRE_WHITEN
    env.pre_whit_L = inv(chol(env.Sigma));
    env.Sigma_filter = eye(2);
    env.PRE_WHITEN = true;
    emv.Sigma = eye(2);
else
    env.pre_whit_L = eye(2);
    env.Sigma_filter = env.Sigma;
    env.PRE_WHITEN = false;
end     

% Give Prior distribution 
prior_mu = cell(1, network_topo.numNodes);
prior_sigma = cell(1, network_topo.numNodes);
prior_mu = repmat({[1000; 1000; -14; 14]}, 1,network_topo.numNodes);
state_cov = [env.range_var,0,0,0; ...
             0,env.range_var,0,0;...
             0,0,env.doppler_var,0;...
             0,0,0,env.doppler_var];
if PRE_WHITEN
    prior_sigma = repmat({eye(4)}, 1, network_topo.numNodes);

else
    prior_sigma = repmat({state_cov}, 1,network_topo.numNodes);
end

%-- EKF setup for each node
EKF = struct();
EKF.delta_k = env.time_step; % Time interval between two measurements? Can be only for track time.
EKF.cv_motion_model = @(state, dt) models_ut.stateModel(state, dt);
EKF.LocalMeasureModel = @(state, idx, network_topo, env) (models_ut.LocalMeasureModel(state, idx, network_topo, env));
EKF.LocalMeasureModelJacobian = @(state, idx, network_topo, env) (models_ut.LocalMeasureModelJacobian(state, idx, network_topo, env));

EKF.MeasureModel = @(state, idx, network_topo, env) (models_ut.MeasureModel(state, idx, network_topo, env));
EKF.MeasureModelJacobian = @(state, idx, network_topo, env) (models_ut.MeasureModelJacobian(state, idx, network_topo, env));

EKF.system_noise = 1e-2 * [EKF.delta_k^4/4, 0, EKF.delta_k^3/2, 0; 
                          0, EKF.delta_k^4/4, 0, EKF.delta_k^3/2; 
                          EKF.delta_k^3/2, 0, EKF.delta_k^2, 0; 
                          0, EKF.delta_k^3/2, 0, EKF.delta_k^2]; % System noise covariance

% EKF.system_noise = 1e3 * [EKF.delta_k^4/4, 0, EKF.delta_k^3/2, 0; 
%                           0, EKF.delta_k^4/4, 0, EKF.delta_k^3/2; 
%                           EKF.delta_k^3/2, 0, EKF.delta_k^2, 0; 
%                           0, EKF.delta_k^3/2, 0, EKF.delta_k^2]; % System noise covariance
% EKF.system_noise = diag([env.Sigma(1,1), env.Sigma(1,1), env.Sigma(2,2), env.Sigma(2,2)]); 
% EKF.StateCovariance = diag([env.Sigma(1,1), env.Sigma(1,1), env.Sigma(2,2), env.Sigma(2,2)]); % Initial state covariance                     
% EKF.StateCovariance = EKF.system_noise; % Initial state covariance
EKF.StateCovariance = diag([1e3, 1e3, 1e3, 1e3]); % Initial state covariance
% EKF.StateCovariance = eye(4,4);
EKF.initial_tar_guess = [1000,1000,-14.1412,14.1412];

% Global info. 
x_dkf = cell(1, network_topo.numNodes);
P_dkf = cell(1, network_topo.numNodes);

for i = 1:network_topo.numNodes
    initial_guess = EKF.initial_tar_guess - [network_topo.radar_pos(i,1), network_topo.radar_pos(i,2),0,0]; % Relative position to each radar node
    %% Local state intialization
    EKF.filter{i} = trackingEKF(State=initial_guess, ...
                     StateCovariance = EKF.StateCovariance, ...
                     StateTransitionFcn = EKF.cv_motion_model, ...% Check diff between build in function and the self-defined one
                     ProcessNoise = EKF.system_noise, ...
                     MeasurementFcn = EKF.LocalMeasureModel, ...
                     MeasurementJacobianFcn = EKF.LocalMeasureModelJacobian, ...
                     MeasurementNoise = env.Sigma_filter);
    

    x0_local = EKF.filter{i}.State(:); % 4x1 local
    x0_global = models_ut.local2global(network_topo.radar_pos(i,:), x0_local.').'; % -> 4x1
    x_dkf{i} = x0_global;
    P_dkf{i} = EKF.StateCovariance;  % or EKF.filter{i}.StateCovariance if different
end %TOFIGUREOUT: the intial guess here are the relative coordination or global coord of the target?

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
        % Delete after debug, noiseless case:
        % disp("**** Warning, using noiseless measurement for debugging ****");
        % range_with_error = range_true;
        % doppler_with_error = doppler_true;
        %-- P2-3: Pharse data for algorithm input
        range_with_error_withNeighbors = cell(NUM_TAR, network_topo.numNodes);
        doppler_with_error_withNeighbors = cell(NUM_TAR, network_topo.numNodes);
        numNodes_withNeighbors = cell(NUM_TAR, network_topo.numNodes);
        radar_positions_withNeighbors = cell(NUM_TAR, network_topo.numNodes);

        range_with_error_cell_window = cell(NUM_TAR, network_topo.numNodes);
        doppler_with_error_cell_window = cell(NUM_TAR, network_topo.numNodes);
        numNodes_cell_buffer = cell(NUM_TAR, network_topo.numNodes);
        radar_positions_cell_buffer = cell(NUM_TAR, network_topo.numNodes);

        prior_mu_cell = cell(1,network_topo.numNodes);
        prior_sigma_cell = cell(1,network_topo.numNodes);
        % Pharse_measurements
        [range_with_error_withNeighbors,doppler_with_error_withNeighbors,...
        numNodes_withNeighbors,radar_positions_withNeighbors,...
        prior_mu_cell,prior_sigma_cell] = ut.pharse_measurements_tracking(network_topo.laplacian_matrix, ...
        range_with_error, doppler_with_error, range_with_error_withNeighbors, doppler_with_error_withNeighbors, NUM_TAR,network_topo,...
        prior_mu, prior_sigma);
        %% P3: Decentralized optimization for tracking
        % Create a cell array to hold the neighbors of each node
        neighbors = ut.get_neighbors(network_topo.adj_matrix, network_topo.numNodes);
        for tar = 1: NUM_TAR
            for k = 1: TRACK_TIME
                t=0;
                % Measurement chunk (Number of #CPI sample)
                for n = 1: network_topo.numNodes
                    range_with_error_cell_window{n} = range_with_error_withNeighbors{tar,n}((k-1)*NUM_CPI_PER_MEA + 1 : k*NUM_CPI_PER_MEA,:);
                    doppler_with_error_cell_window{n}  = doppler_with_error_withNeighbors{tar,n}((k-1)*NUM_CPI_PER_MEA + 1 : k*NUM_CPI_PER_MEA,:);
                end

                %% TOCORRECT
                for instance = 1:NUM_CPI_PER_MEA
                    % Update EKF with each measurement in the burst (update per CPI)
                    t = (k-1)*NUM_CPI_PER_MEA+instance; % Each time step index
                    disp("Time step: "+t+", Node: "+i);
                    % ---- 1) predict all nodes (global) ----
                    x_pred_nodes = cell(1, network_topo.numNodes);
                    P_pred_nodes = cell(1, network_topo.numNodes);
                    for n = 1:network_topo.numNodes
                        [x_pred_nodes{n}, P_pred_nodes{n}] = dkf_predict_cv(x_dkf{n}, P_dkf{n}, env.time_step, EKF.system_noise);
                    end
                    % ---- 2) correct each node using neighbor measurements (LKF-II) ----
                    for n = 1:network_topo.numNodes
                        idx_set = [n; neighbors{n}(:)];  % (N_n ∪ {n})
                        % Build y_bar = [y_n; y_neighbor1; ...] with GLOBAL measurement (range,doppler)
                        y_bar = zeros(2*numel(idx_set), 1);
                        for a = 1:numel(idx_set)
                            j = idx_set(a);                           
                            y_bar(2*a-1) = range_with_error_cell_window{n}(instance, a);
                            y_bar(2*a)   = doppler_with_error_cell_window{n}(instance,a);
                        end
                        % If you ever turn PRE_WHITEN on:
                        %   - either whiten y_bar here AND ensure MeasureModel returns whitened too
                        %   - or keep both unwhitened
                        % For now PRE_WHITEN=false => do nothing.
                        [x_dkf{n}, P_dkf{n}] = dkf_neighbor_update( ...
                            x_pred_nodes{n}, P_pred_nodes{n}, y_bar, idx_set, network_topo, env, models_ut);
                        x_pred_from_EKFs{t, n} = x_pred_nodes{n}; % TO DLELETE AFTER DEBUGGING
                        P_pred_from_EKFs{t, n} = P_pred_nodes{n}; % TO DLELETE AFTER DEBUGGING
                        all_estimation_from_EKFs{t, n} = x_dkf{n}; % TO DLELETE AFTER DEBUGGING
                    end
                    
                end
                %% Consensus Algorithm 
                global_state = models_ut.local2global(network_topo.radar_pos,ADMM.initial_values');
                % global_state = cell2mat(cellfun(@(x) x(:).', x_dkf, 'UniformOutput', false)); % N x 4
                [estimated_params, x_hist, P, L, eps_used, diff_hist] = consensus(global_state, network_topo.adj_matrix);
                %TODO: Remenber to reset ADMM.iteration = 0; ADMM.converged = false ; after each time step, can write a re-set function in trackingEKF class
                ADMM.final_tracking_estimation{tar, k} = estimated_params;% Also de-whiten
                ADMM = ut.ADMM_reset(ADMM,NUM_TAR,network_topo);
                Results.primal_residauls{k} = diff_hist;
                % Results.dual_residauls{k} = ;
                Results.estimations{k} = x_hist;
                Results.true_params{k} = [squeeze(target.target_position(tar, k*NUM_CPI_PER_MEA, :))', target.true_params(3), target.true_params(4)];
                Results.convg_iter{k} = size(x_hist,3);
                % clear all_estimations;

                % % end %TODO: Remenber to reset ADMM.iteration = 0; ADMM.converged = false ; after each time step, can write a re-set function in trackingEKF class
                % % Plot EKF estimation result for each node one plot for each parameter


                % fprintf('Time step %d completed.\n', k);

                % [Deleted after debugging]

            end 
        end
        %%
        % P4: Performance evaluation & Plotting 
        if DEBUG
            start = 65;
            figure;
            subplot(2,2,1);
            hold on;
            for i = 1:network_topo.numNodes
                plot([start:1:t], cellfun(@(x) x(1), all_estimation_from_EKFs(start:t, i)),'DisplayName', "Corr."+network_topo.labels{i});
                % plot([1:1:t], cellfun(@(x) x(1), x_pred_from_EKFs(1:t,i)), '--','DisplayName', "Pred."+network_topo.labels{i});
                %plot GT
                % plot([start:1:t],  cellfun(@(x) x(1), target.target_state_wrt_node(start:t, i)),'--', 'linewidth', 1.5 ,'DisplayName', "GT_"+network_topo.labels{i});
                plot([start:1:t], cellfun(@(x) x(1), x_pred_from_EKFs(start:t,i)), 'o','DisplayName', "Pred."+network_topo.labels{i});
            end
            % plot([1:1:t], squeeze(target.target_position(tar, 1:t, 1)), 'k--', 'DisplayName', 'True');
            title('EKF Estimation of X position');
            xlabel('Time step');
            ylabel('X position (m)');
            legend('show');
            % ylim([1, 5e3]);
            subplot(2,2,2);
            hold on;        
            for i = 1:network_topo.numNodes
                plot([start:1:t], cellfun(@(x) x(2), all_estimation_from_EKFs(start:t, i)), 'DisplayName', "Corr."+network_topo.labels{i});
                % plot([start:1:t],  cellfun(@(x) x(2), target.target_state_wrt_node(start:t, i)),'--', 'linewidth', 1.5 ,'DisplayName', "GT_"+network_topo.labels{i});
                plot([start:1:t], cellfun(@(x) x(2), x_pred_from_EKFs(start:t,i)), 'o','DisplayName', "Pred. d"+network_topo.labels{i});
            end
            % plot([1:1:t], squeeze(target.target_position(tar, 1:t, 2)), 'k--', 'DisplayName', 'True');
            title('EKF Estimation of Y position');
            xlabel('Time step');
            ylabel('Y position (m)');
            legend('show');
            subplot(2,2,3);
            % ylim([1, 5e3]);
            hold on;        
            for i = 1:network_topo.numNodes
                plot([start:1:t], cellfun(@(x) x(3), all_estimation_from_EKFs(start:t, i)), 'DisplayName', "Corr."+network_topo.labels{i});
                % plot([start:1:t],  cellfun(@(x) x(3), target.target_state_wrt_node(start:t, i)),'--', 'linewidth', 1.5 ,'DisplayName', "GT_"+network_topo.labels{i});
                plot([start:1:t], cellfun(@(x) x(3), x_pred_from_EKFs(start:t,i)), 'o','DisplayName', "Pred"+network_topo.labels{i});
            end         
            % plot([1:1:t], repmat(target.true_params(3), 1, t), 'k--', 'DisplayName', 'True');
            title('EKF Estimation of X velocity');
            xlabel('Time step');
            ylabel('X velocity (m/s)');
            % ylim([-20, 0]);
            legend('show');
            subplot(2,2,4);
            hold on;        
            for i = 1:network_topo.numNodes
                plot([start:1:t], cellfun(@(x) x(4), all_estimation_from_EKFs(start:t, i)), 'DisplayName', "Corr"+network_topo.labels{i});
                % plot([start:1:t],  cellfun(@(x) x(4), target.target_state_wrt_node(start:t, i)),'--', 'linewidth', 1.5 ,'DisplayName', "GT_"+network_topo.labels{i});
                plot([start:1:t], cellfun(@(x) x(4), x_pred_from_EKFs(start:t,i)),'o', 'DisplayName', "Pred"+network_topo.labels{i});
            end         
            % plot([1:1:t], repmat(target.true_params(4), 1, t), 'k--', 'DisplayName', 'True');
            title('EKF Estimation   of Y velocity');
            xlabel('Time step');
            ylabel('Y velocity (m/s)');
            % ylim([0, 20]);
            legend('show');
            %%
        end
        % fig_ut.plot_trajectory(target.target_position,ADMM.final_tracking_estimation);
    end
    % Save the resulted estimation log in json file each monte carlo run
    %-- P1-5 Save the experiment parameters log in json file
    experiment_params_log = struct();
    experiment_params_log.network_topo = network_topo;
    experiment_params_log.NUM_CPI_PER_MEA = NUM_CPI_PER_MEA;
    experiment_params_log.TRACK_TIME = TRACK_TIME;
    experiment_params_log.time_step = time_step;
    experiment_params_log.num_monte_carlo = num_monte_carlo;
    experiment_params_log.env = env;
    experiment_params_log.ADMM = ADMM;
    experiment_params_log.target = target;
    experiment_params_log.results = Results;
    if DEBUG==false
        ut.write_exp_log('./data_log', append('exp_config',num2str(mc)),experiment_params_log);
    end
end

%% -- Plotting
fig_ut.plot_trajectory(target.target_position,ADMM.final_tracking_estimation);



%% Helper function
function [x_post, P_post, dbg] = dkf_neighbor_update( ...
    x_pred, P_pred, y_bar, idx_set, network_topo, env, models_ut)
% DKF/LKF-II neighbor-augmented EKF correction (information-form)
% Implements the revised form:
%   P^{-1}_{n} = P^{-1}_{pred} + sum_j H_j' R_j^{-1} H_j
%   x_{n} = x_pred + P_n * sum_j H_j' R_j^{-1} (y_j - h_j(x_pred))
%
% Inputs
%   x_pred: (4x1) predicted GLOBAL state at node n
%   P_pred: (4x4) predicted covariance
%   y_bar : (2*|idx_set| x 1) stacked measurements [r;fd] for each j in idx_set
%   idx_set: radar indices in (N_n ∪ {n}), e.g., [n, neighbors{n}]
%   network_topo, env: your existing structs
%   models_ut: instance of your models class (or use models static)
%
% Output
%   x_post, P_post: posterior
%   dbg: optional debug info

    m_per_node = 2;
    J = numel(idx_set);
    assert(numel(y_bar) == m_per_node * J, "y_bar length mismatch.");

    % Measurement noise (per-node)
    R = env.Sigma_filter;          % (2x2)
    Rinv = inv(R);

    % Information-form accumulation
    Ppred_inv = inv(P_pred);
    sumInfo   = zeros(4,4);
    sumInnov  = zeros(4,1);

    z_pred_all = zeros(m_per_node*J,1);
    innov_all  = zeros(m_per_node*J,1);

    for a = 1:J
        j = idx_set(a);

        yj = y_bar((a-1)*m_per_node+1 : a*m_per_node);

        % Nonlinear measurement and Jacobian wrt GLOBAL state: (Change to local)
        hj = models_ut.LocalMeasureModel(x_pred, j, network_topo, env);            % (2x1)
        Hj = models_ut.LocalMeasureModelJacobian(x_pred, j, network_topo, env);    % (2x4)

        % innovation
        innov = yj - hj;

        sumInfo  = sumInfo  + Hj' * Rinv * Hj;
        sumInnov = sumInnov + Hj' * Rinv * innov;

        z_pred_all((a-1)*m_per_node+1 : a*m_per_node) = hj;
        innov_all((a-1)*m_per_node+1 : a*m_per_node)  = innov;
    end

    % Posterior covariance / mean
    P_post = inv(Ppred_inv + sumInfo);
    x_post = x_pred + P_post * sumInnov;

    % Debug output
    dbg = struct();
    dbg.z_pred = z_pred_all;
    dbg.innov  = innov_all;
    dbg.sumInfo = sumInfo;
    dbg.sumInnov = sumInnov;
end


function [x_pred, P_pred] = dkf_predict_cv(x_prev, P_prev, dt, Q)
    F = [1 0 dt 0;
         0 1 0 dt;
         0 0 1  0;
         0 0 0  1];
    x_pred = F * x_prev;
    P_pred = F * P_prev * F.' + Q;
end