clc;clear;close all;
ut = ADMM_utils;
models_ut = models;
DEBUG = true;
% P0, TODO: We can make the setting into config file, then we can sacve up to 100 lines

%% P1: Initialize ---
%-- P1-0: Simulation Scenarios parameters
num_monte_carlo = 1; % Number of monte carlo runs
NUM_CPI_PER_MEA = 64;    % Number of measurements per burst
TRACK_TIME = 1;   % Number of burst times
time_step = 1e-4;    % Time step between two measurements
% [DELETE AFTER DEBUGGING]
all_estimation_from_EKFs = cell(TRACK_TIME*NUM_CPI_PER_MEA,10);

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

[network_topo.adj_matrix, network_topo.degree_matrix, network_topo.laplacian_matrix, network.inc_matrix, network_topo.weights_matrix] = ut.calculate_all_graph_matrix(network_topo.radar_pos, network_topo.com_rad_CR, network_topo.numNodes);


%-- P1-3 Algoirthm parameters and solver setup
options_Dctral = optimoptions('fmincon', 'Display', 'off', ScaleProblem=true, OptimalityTolerance=1e-2, FunctionTolerance=1e-6, StepTolerance=1e-6, MaxIterations=100);
options_Ctral =  optimoptions('fmincon', 'Display', 'off', ScaleProblem=true, OptimalityTolerance=1e-6, FunctionTolerance=1e-6, StepTolerance=1e-6, MaxIterations=100000);
ADMM = struct();
ADMM.solver = options_Dctral;
ADMM.converged = false;
ADMM.max_iter = 100; % Maximum ADMM iterations
ADMM.c_penalty = [100,100,15,15];
ADMM.lb = [-inf,-inf,-inf,-inf];
ADMM.ub = [inf,inf, inf, inf];
ADMM.initial_values = repmat([1000, 1000, 10, 10]', 1,network_topo.numNodes);
ADMM.Nu = cell(NUM_TAR, network_topo.numNodes);
ADMM.Nu_prev = cell(NUM_TAR, network_topo.numNodes);
ADMM.update_z = cell(NUM_TAR, network_topo.numNodes);
ADMM.final_tracking_estimation = cell(NUM_TAR, TRACK_TIME);
ADMM.update_z_prev = cell(NUM_TAR, network_topo.numNodes);
ADMM.primal_residual_all =[];
ADMM.primal_residual_by_para = cell(NUM_TAR,network_topo.numNodes);
ADMM.dual_residual_all = [];
ADMM.dual_residual_by_para = cell(NUM_TAR,network_topo.numNodes);
ADMM.all_estimations_every_iter = [];
ADMM.RANGE_Xs = [];
ADMM.RANGE_Ys = [];
ADMM.DOPPLER_Xs = [];
ADMM.DOPPLER_Ys = [];
ADMM.converg_r = false;
ADMM.converg_d = false;
ADMM.tolerance = 1e-4; % Convergence tolerance for primal residual
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
env.time_step = time_step; % Time step between two measurementsㄎ
env.T = env.time_step / 2; % Pulse repetition interval
env.B = 10e6 * ones(1,network_topo.numNodes); % Bandwidth
env.fs = 2*env.B; % Sampling frequency
env.SNR_idx = 50;
env.SNR_lin = 10^(env.SNR_idx/10); % SNR in linear scale
fprintf('SNR_db: %d dB, SNR_linear: %f\n', env.SNR_idx, env.SNR_lin);
% Noise variance is (2,2) = [rage_var, ro; ro, doppler_var]
env.range_var = (3 * env.c^2) / (8 * pi^2 * env.B(1)^2 * env.SNR_lin) ;
env.doppler_var = (3 * ((env.fs(1))^2)) / (pi^2 * env.SNR_lin * NUM_CPI_PER_MEA^3) ;
env.range_sd = sqrt(env.range_var);
env.doppler_sd = sqrt(env.doppler_var);
env.rho = 0.0;  
env.Sigma = [env.range_var, env.rho * env.range_sd * env.doppler_sd; ...
              env.rho * env.range_sd * env.doppler_sd, env.doppler_var];   

%-- EKF setup for each node
EKF = struct();
EKF.delta_k = NUM_CPI_PER_MEA*env.time_step; % Time interval between two measurements
EKF.cv_motion_model = @(state, dt) models_ut.stateModel(state, dt);
EKF.MeasureModel = @(state, idx, network_topo, env) models_ut.MeasureModel(state, idx, network_topo, env);
EKF.MeasureModelJacobian = @(state, idx, network_topo, env) models_ut.MeasureModelJacobian(state, idx, network_topo, env);
EKF.system_noise = 0.1 * [ EKF.delta_k^4/4, 0, EKF.delta_k^3/2, 0; 
                          0, EKF.delta_k^4/4, 0, EKF.delta_k^3/2; 
                          EKF.delta_k^3/2, 0, EKF.delta_k^2, 0; 
                          0, EKF.delta_k^3/2, 0, EKF.delta_k^2]; % System noise covariance

% EKF.StateCovariance = diag([1e-1, 1e-1, 1e-1, 1e-1]); % Initial state covariance 
EKF.StateCovariance = zeros(4); % Initial state covariance 
EKF.initial_tar_guess = [1000,1000,10,10];
for i = 1:network_topo.numNodes
    initial_guess = EKF.initial_tar_guess - [network_topo.radar_pos(i,1), network_topo.radar_pos(i,2),0,0]; % Relative position to each radar node
    EKF.filter{i} = trackingEKF(State=initial_guess, ...
                     StateCovariance = EKF.StateCovariance,...
                     StateTransitionFcn = EKF.cv_motion_model, ...
                     ProcessNoise = EKF.system_noise, ...
                     MeasurementFcn = EKF.MeasureModel, ...
                     MeasurementJacobianFcn = EKF.MeasureModelJacobian, ...,
                     MeasurementNoise = env.Sigma);
end %TOFIGUREOUT: the intial guess here are the relative coordination or global coord of the target?

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
        range_true = zeros(NUM_TAR,network_topo.numNodes,NUM_CPI_PER_MEA*TRACK_TIME);
        doppler_true = zeros(NUM_TAR, network_topo.numNodes,NUM_CPI_PER_MEA*TRACK_TIME); 
        measurements_true = zeros(NUM_TAR, network_topo.numNodes, 2 * NUM_CPI_PER_MEA*TRACK_TIME);
        [range_true, doppler_true, measurements_true] = ut.gt_data_generation_tracking(range_true,doppler_true, measurements_true, target,network_topo,env, NUM_CPI_PER_MEA*TRACK_TIME, NUM_TAR);
        measurements_true_all = reshape(measurements_true, [], 1);% flatten

        %-- P2-2: Generate noisy target measurements
        % y_hat = y_gt + noise
        range_with_error = zeros(NUM_TAR, network_topo.numNodes,NUM_CPI_PER_MEA*TRACK_TIME);
        doppler_with_error = zeros(NUM_TAR, network_topo.numNodes, NUM_CPI_PER_MEA*TRACK_TIME); 
        measurements_with_error_all = zeros(NUM_TAR, network_topo.numNodes ,2 * NUM_CPI_PER_MEA*TRACK_TIME);

        [range_with_error, doppler_with_error, measurements_with_error_all] = ut.add_measurement_noise(range_true, doppler_true, NUM_CPI_PER_MEA*TRACK_TIME, NUM_TAR,network_topo.numNodes, env);

        % %-- P2-3: Visualize the generated data
        % if DEBUG
        %     idx = 5;
        %     figure;
        %     subplot(2,2,1);
        %     plot([1:1:size(range_true(1,1,:),3)],squeeze(range_true(1,idx,:)), '--'); 
        %     subplot(2,2,2);
        %     plot([1:1:size(range_with_error(1,1,:),3)],squeeze(range_with_error(1,idx,:)), 'r');
        %     title('Range measurement (Node 1)');
        %     legend('True', 'With noise');
        %     subplot(2,2,3);
        %     plot([1:1:size(doppler_true(1,1,:),3)],squeeze(doppler_true(1,idx,:)), '--'); hold on;
        %     subplot(2,2,4);
        %     plot([1:1:size(doppler_with_error(1,1,:),3)],squeeze(doppler_with_error(1,idx,:)),'r');
        %     title('Doppler measurement (Node 1)');
        %     legend('True', 'With noise');
        % end 
        % if DEBUG
        %     % plot target position and node position
        %     figure; hold on;
        %     plot(squeeze(target.target_position(1,:,1)), squeeze(target.target_position(1,:,2)), 'b-o');
        %     plot(network_topo.radar_pos(:,1), network_topo.radar_pos(:,2), 'r*');
        %     for i = 1:network_topo.numNodes
        %         text(network_topo.radar_pos(i,1), network_topo.radar_pos(i,2), network_topo.labels{i}, 'VerticalAlignment','bottom', 'HorizontalAlignment','right');
        %     end
        %     axis equal;
        %     title('Target trajectory and radar node positions');
        % end
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
        [range_with_error_withNeighbors,doppler_with_error_withNeighbors,numNodes_withNeighbors,radar_positions_withNeighbors] = ut.pharse_measurements_tracking(network_topo.laplacian_matrix, ...
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
                    for i = 1: network_topo.numNodes
                        current_range_meas = squeeze(range_with_error_withNeighbors{tar, i}((k-1)*NUM_CPI_PER_MEA + 1 : k*NUM_CPI_PER_MEA));
                        current_doppler_meas = squeeze(doppler_with_error_withNeighbors{tar, i}((k-1)*NUM_CPI_PER_MEA + 1 : k*NUM_CPI_PER_MEA));
                        current_measurements = [current_range_meas'; current_doppler_meas'];
                        for instance = 1:NUM_CPI_PER_MEA
                            % Update EKF with each measurement in the burst
                            t = (k-1)*NUM_CPI_PER_MEA+instance;
                            predict(EKF.filter{i}, env.time_step);
                            correct(EKF.filter{i}, [current_range_meas(instance), current_doppler_meas(instance)], i ,network_topo,env);
                            % all_estimation_from_EKFs{(k-1)*NUM_CPI_PER_MEA+instance, i} = EKF.filter{i}.State; % TO DLELETE AFTER DEBUGGING
                        end
                        % Perform EKF prediction and update
                        % predict(EKF.filter{i}KF,E.delta_k);
                        % % [EKF.filter{i}.State, EKF.filter{i}.StateCovariance] = correct(EKF.filter{i}, current_measurements(:, i), i ,network_topo,env);
                        % correct(EKF.filter{i}, current_measurements(:, i), i ,network_topo,env);
                        % Store the EKF estimate as the initial value for ADMM
                        ADMM.initial_values(:, i) = EKF.filter{i}.State;
                        
                    end
                end
                %-- P3-2: Distributed consensus optimization (ADMM)
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
                        fun = @(params) ut.logLikelihoodWithConsensus(params, range_with_error_cell_window{n},doppler_with_error_cell_window{n},...
                                                             radar_positions_withNeighbors{n},numNodes_withNeighbors{n}, NUM_CPI_PER_MEA, env.lambda,...
                                                             env.Sigma, n, neighbors, ADMM.Nu, ADMM.initial_values,...
                                                              ADMM.update_z_prev, ADMM.c_penalty);

                        % save("variable.mat","range_with_error_cell_window","doppler_with_error_cell_window",...
                        %                                      "radar_positions_withNeighbors","numNodes_withNeighbors", "NUM_CPI_PER_MEA", "env",...
                        %                                       "n", "neighbors", "ADMM");
                        estimated_params = fmincon(fun, ADMM.initial_values(:,n),[],[],[],[], ADMM.lb, ADMM.ub, [], ADMM.solver);
                        
                        % try
                        %     estimated_params = fmincon(fun, ADMM.initial_values(:,n),[],[],[],[], ADMM.lb, ADMM.ub, [], ADMM.solver);
                        % catch
                        %     fprintf('Warning: Optimization failed at node %d, iteration %d. Using previous estimates:\n %d \n', n, iteration, ADMM.initial_values(:,n));
                        % end
                        % [DEBUG]
                        % ut.logLikelihoodWithConsensus([0,0,0,0], range_with_error_cell_window{n},doppler_with_error_cell_window{n},...
                        %                                      radar_positions_withNeighbors{n},numNodes_withNeighbors{n}, NUM_CPI_PER_MEA, env.lambda,...
                        %                                      env.Sigma, n, neighbors, ADMM.Nu, ADMM.initial_values,...
                        %                                       ADMM.update_z_prev, ADMM.c_penalty)
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
                fprintf('Time step %d completed.\n', k);

                % [Deleted after debugging]

            end 
        end

        %% P4: Performance evaluation & Plotting 
    end
end