clc;clear;close all;
ut = ADMM_utils;
models_ut = models;
fig_ut = make_figs(10);
DEBUG = true;
PRE_WHITEN = true; % Whether to pre-whiten the measurements
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
EKF.StateCovariance = EKF.system_noise; % Initial state covariance
% EKF.StateCovariance = diag([1e3, 1e3, 1e3, 1e3]); % Initial state covariance
% EKF.StateCovariance = ones(4,4);
EKF.initial_tar_guess = [1000,1000,-14.1412,14.1412];

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
                for i = 1: network_topo.numNodes
                    range_with_error_cell_window{i} = range_with_error_withNeighbors{tar,i}((k-1)*NUM_CPI_PER_MEA + 1 : k*NUM_CPI_PER_MEA,:);
                    doppler_with_error_cell_window{i}  = doppler_with_error_withNeighbors{tar,i}((k-1)*NUM_CPI_PER_MEA + 1 : k*NUM_CPI_PER_MEA,:);
                end
                %-- P3-1: Local filtering (EKF), loop over time and each node 
                state_cov_nodes = cell(1, network_topo.numNodes);
                for i = 1: network_topo.numNodes
                    current_range_meas = squeeze(range_with_error(tar, i, (k-1)*NUM_CPI_PER_MEA + 1 : k*NUM_CPI_PER_MEA));
                    current_doppler_meas = squeeze(doppler_with_error(tar, i, (k-1)*NUM_CPI_PER_MEA + 1 : k*NUM_CPI_PER_MEA));
                    for instance = 1:NUM_CPI_PER_MEA
                        % Update EKF with each measurement in the burst (update per CPI)
                        t = (k-1)*NUM_CPI_PER_MEA+instance; % Each time step index
                        disp("Time step: "+t+", Node: "+i);
                        %-- Debugging print intermediate values
                        % if i == 2
                        %     models_ut.printEkfIntermediates(EKF.filter{i},env.pre_whit_L*[current_range_meas(instance), current_doppler_meas(instance)]', EKF.delta_k, t , i, network_topo, env);
                        % end
                        %--
                        [xpred, Ppred] = predict(EKF.filter{i}, env.time_step);
                        correct(EKF.filter{i}, env.pre_whit_L*[current_range_meas(instance), current_doppler_meas(instance)]', i ,network_topo, env);
                        all_estimation_from_EKFs{(k-1)*NUM_CPI_PER_MEA+instance, i} = EKF.filter{i}.State; % TO DLELETE AFTER DEBUGGING
                        x_pred_from_EKFs{(k-1)*NUM_CPI_PER_MEA+instance, i} = xpred; % TO DLELETE AFTER DEBUGGING
                        P_pred_from_EKFs{(k-1)*NUM_CPI_PER_MEA+instance, i} = Ppred; % TO DLELETE AFTER DEBUGGING
                    end
                    state_cov_nodes{i} = EKF.filter{i}.StateCovariance;
                    %-- Update EKF with each measurement in one row (update KF per burst)
                    % [xpred, Ppred] = predict(EKF.filter{i}, NUM_CPI_PER_MEA*env.time_step);
                    % correct(EKF.filter{i}, [current_range_meas(end), current_doppler_meas(end)], i ,network_topo,env);
                    % all_estimation_from_EKFs{t, i} = EKF.filter{i}.State; % TO DLELETE AFTER DEBUGGING
                    % x_pred_from_EKFs{k, i} = xpred; % TO DLELETE AFTER DEBUGGING
                    % P_pred_from_EKFs{k, i} = Ppred; % TO DLELETE AFTER DEBUGGING
                    % Perform EKF prediction and update
                    % predict(EKF.filter{i}KF,E.delta_k);
                    % % [EKF.filter{i}.State, EKF.filter{i}.StateCovariance] = correct(EKF.filter{i}, current_measurements(:, i), i ,network_topo,env);
                    % correct(EKF.filter{i}, current_measurements(:, i), i ,network_topo,env);
                    % Store the EKF estimate as the initial value for ADMM
                end
                %--- Local EKF for consensus optimization, handover (1) Initial value, (2) Prior information 
                if k~= 1
                    for i = 1: network_topo.numNodes
                        % global to local coord for initlial value and prior
                        % Try (1) EKF state diff. of all node, (2) previous ADMM estimation same for all nodes
                        ADMM.initial_values(:,i) = EKF.filter{i}.State;
                        % ADMM.initial_values(:, i) = xpred;
                        % ADMM.initial_values = repmat(ADMM.final_tracking_estimation{tar, k-1}, 1,network_topo.numNodes); 
                    end 
                    [ADMM.prior_mean, ADMM.prior_cov]= ut.get_neighbors_cell(network_topo.laplacian_matrix, NUM_TAR, network_topo.numNodes,ADMM.initial_values,state_cov_nodes);
                    % [ADMM.prior_mean, ADMM.prior_cov]= ut.get_neighbors_cell(network_topo.laplacian_matrix, NUM_TAR, network_topo.numNodes,ADMM.initial_values,state_cov);
                end       

                %% TODO: Consensus Algorithm 
                global_state = models_ut.local2global(network_topo.radar_pos,ADMM.initial_values');
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
