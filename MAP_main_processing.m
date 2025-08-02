clc; close all; clear;
ut = ADMM_utils;
%TODO: 
% 1. Write a text file to export config. / parameters setup of the run
% 2. Make the algorithm to be parallel computing?
%-- Config. of the run
num_monte_carol = 1;
direction_mc = zeros(num_monte_carol,2);
gt_paras_mc = zeros(num_monte_carol,4);
%-- Network topo
network_topo.numNodes = 10;
theta = linspace(0,2*pi, network_topo.numNodes+1);
network_topo.theta = theta(1:end-1);
network_topo.com_rad_CR = 3000; % communication radius range
network_topo.radius = 3000;     % spatial placement radius 
network_topo.radar_pos = network_topo.radius * [cos(network_topo.theta); sin(network_topo.theta)]';
network_topo.C_distance = 1;  % Cost per meter
network_topo.C_data = 1;      % Cost per byte
network_topo.distances_between_radar_nodes = zeros(network_topo.numNodes,network_topo.numNodes);

for i = 1:network_topo.numNodes
    for j = 1:network_topo.numNodes
        network_topo.distances_between_radar_nodes(i,j) = norm(network_topo.radar_pos(i,:) - network_topo.radar_pos(j,:));
    end
end

knn = 5;
com_rad_CR= 3000; % Communication radius in CD


% Signal and environment parameters
env.c = 3e8;
env.lambda = env.c / 10e9;
env.time_step = 1e-4;
env.T = env.time_step / 2;
env.B = 10e6 * ones(1, network_topo.numNodes); % Bandwidth in Hz
env.fs = 2 * env.B; % Sampling frequency

% M_values = [5, 15, 30];
M_values = 64; % Time_step = 1e-4, means for the time interval of burst is 1e-4. Is this make sense? 

% Range of nodes
% node_range = 20:3:20;
node_range = 10;

options_DA = optimoptions('fmincon', 'Display', 'off', ScaleProblem=true, OptimalityTolerance=1e-6, FunctionTolerance=1e-6, StepTolerance=1e-6, MaxIterations=100);
options_CA = optimoptions('fmincon', 'Display', 'off', ScaleProblem=true, OptimalityTolerance=1e-6, FunctionTolerance=1e-6, StepTolerance=1e-6, MaxIterations=100000);


%-- Start the MC simulation
for mc = 1:num_monte_carol
    %%%%%%%%% Setup synthetic GT data, params %%%%%%%
    disp('Monte Carlo Simulations:');
    disp(mc)

    % Calculate direction vector from angle
    %-- Params of the target
    %TODO: classdef a target
    target.initial_position = [1000, 1000];
    target.speed = 20;
    target.angle_degrees = [135]; % In deg.
    angle_degrees = target.angle_degrees(mc);
    target.direction = [cos(angle_degrees * pi / 180), sin(angle_degrees * pi / 180)];%     direction = direction / norm(direction);
    target.true_params = [target.initial_position(1), target.initial_position(2), target.speed * target.direction(1), target.speed * target.direction(2)];
    

    disp('Direction the target is travelling:');
    disp(target.direction);


    %-- Initialize M (measurements)
    time_vector = 0:env.time_step:M_values * env.time_step; % Adjust time vector for each M, time_step
    
    % Preallocate for plotting
    RMSEs = zeros(length(node_range), 4); % 4 parameters
    
    % Loop over different number of nodes
    numNodes = node_range;
    
    % Compute the distance matrix
    radar_pos_T = network_topo.radar_pos';
      
    disp('Node Range:');
    disp(node_range);
    disp('Number of Measurements:');
    disp(M_values);
    
    % Initialize the adjacency matrix
    adj_matrix = zeros(network_topo.numNodes, network_topo.numNodes);

    for com_rad = 1:length(com_rad_CR)
        fprintf('Communication Radius: %d\n',  com_rad_CR(com_rad));
        communication_radius = com_rad_CR(com_rad);
        % Calculate all matrixs in one function
        [adj_matrix,degree_matrix, laplacian_matrix, inc_matrix, weights_matrix] = ut.calculate_all_graph_matrix(network_topo.radar_pos, communication_radius, network_topo.numNodes);

        % Create a cell array to hold the neighbors of each node
        neighbors = ut.get_neighbors(adj_matrix, network_topo.numNodes);
       
        % Compute the Laplacian matrix
        laplacian_matrix_CR{com_rad}=laplacian_matrix;  
        laplacian_eigen = eig(laplacian_matrix);
        laplacian_eigen_vector = sort(laplacian_eigen, 'descend');
        
        % This can be move out to the loop, and maybe set as env. 
        snr_idx = 50;
        SNR_lin = 10^(snr_idx / 10);
        fprintf('SNR_db: %d dB, SNR_linear: %f\n', snr_idx, SNR_lin);
        %%% Monte carlo simulations
        estimated_param_values = [];
        M = M_values;
        target.target_position = zeros(M, 2);
        target.target_position(1, :) = target.initial_position; % Initial position
        % Target position move over in the M burst time.
        for k = 2:M
            target.target_position(k, :) = target.target_position(k - 1, :) + target.speed * target.direction * env.time_step;
        end
    
        % Matrices for range and Doppler true data (This is a matrix of k x M x N)
        range_true = zeros(M, network_topo.numNodes);
        doppler_true = zeros(M, network_topo.numNodes); 
        measurements_true = zeros(2 * M, network_topo.numNodes);
        [range_true, doppler_true, measurements_true] = ut.gt_data_generation(range_true,doppler_true, measurements_true, target,network_topo,env, M);
        measurements_true_all = reshape(measurements_true, [], 1);% flatten
        
        % Calculate Noise for range and Doppler from signal model
        % TODO: This can also be set into env. params
        range_var = (3 * env.c^2) / (8 * pi^2 * env.B(1)^2 * SNR_lin) ;
        doppler_var = (3 * ((env.fs(1))^2)) / (pi^2 * SNR_lin * M^3) ;
        range_sd = sqrt(range_var);
        doppler_sd = sqrt(doppler_var);
        rho = 0.0;  
        % 2x2 covariance matrix Sigma
        Sigma = [range_var, rho * range_sd * doppler_sd; rho * range_sd * doppler_sd, doppler_var];   
        total_measurements = numNodes * M;    
    
        % Sigma_big is of the size 2NM x 2NM, 2= (range, doppler), 
         % Generate range and Doppler shift's noisy measurements
        Sigma_big = kron(eye(total_measurements), Sigma);
        noise_matrix = mvnrnd(zeros(2 * total_measurements, 1), Sigma_big)';
        range_noise = noise_matrix(1:2:end);
        range_noise_all = reshape(range_noise, M, numNodes);
        doppler_noise = noise_matrix(2:2:end);
        doppler_noise_all = reshape(doppler_noise, M, numNodes);
    
        % y_hat = y_gt + noise
        range_with_error = range_true + range_noise_all;
        doppler_with_error = doppler_true + doppler_noise_all;
        measurements_with_error_all = measurements_true_all + noise_matrix;
        
        range_with_error_CA = range_with_error;
        doppler_with_error_CA = doppler_with_error;
        
        % Prior knowledge of measurements
        % Assume independent mea. of mu and doppler 
        mu_r = mean(range_with_error);
        sigma_r   = var(range_with_error);
        mu_d = mean(doppler_with_error);
        sigma_d = var(doppler_with_error);

        % mu_r = range_true(1,:);
        % sigma_r   = var(range_true);
        % mu_d = doppler_true(1,:);
        % sigma_d = var(range_true);
    

        % y_0 and lower, upper bound
        initial_guess = [1000, 1000, 10, 10];
        lb = [-inf,-inf,-inf,-inf];
        ub = [inf,inf, inf, inf];
        % fun = @(params) ut.logLikelihood(params, range_with_error_CA, doppler_with_error_CA, network_topo.radar_pos, network_topo.numNodes, M, env.lambda, Sigma_big);
        fun = @(params) ut.MAP(params, range_with_error_CA, doppler_with_error_CA, mu_r, mu_d, sigma_r, sigma_d, network_topo.radar_pos, network_topo.numNodes, M, env.lambda, Sigma_big);
        [estimated_params_CA, log_likelihood, exitflag, output] = fmincon(fun, initial_guess,[],[],[],[], lb, ub, [],options_CA);
        estimates_mc_CA(mc, :) = estimated_params_CA;
        
        %%%%% End of Setup synthetic GT, params %%%%%
        %% Start the sensing, assign the measuremnet (range, doppler) to sensor network 
        % Have M brust, for N nodes. per parameters. We have (r, v) as params.
        % This can be replaced by loop over $neighbors var?
        range_with_error_cell = cell(1, numNodes);
        doppler_with_error_cell = cell(1, numNodes);
        numNodes_cell = cell(1, numNodes);
        radar_positions_cell = cell(1, numNodes);
        Sigma_big_1_cell = cell(1, numNodes);
        Sigma_big_2_cell = cell(1, numNodes);
        mu_r_cell = cell(1,network_topo.numNodes);
        mu_d_cell = cell(1,network_topo.numNodes);
        sigma_r_cell = cell(1,network_topo.numNodes);
        sigma_d_cell = cell(1,network_topo.numNodes);
        
        % Pharse_measurements
        [range_with_error_cell,doppler_with_error_cell,numNodes_cell,...
        radar_positions_cell,Sigma_big_1_cell,Sigma_big_2_cell,...
        mu_r_cell,mu_d_cell,sigma_r_cell,sigma_d_cell] = ut.pharse_measurements(laplacian_matrix, ...
        range_with_error, doppler_with_error, mu_r, mu_d, sigma_r, sigma_d,Sigma_big,...
        range_with_error_cell, doppler_with_error_cell, ...
        mu_r_cell, mu_d_cell, sigma_r_cell, sigma_d_cell, network_topo, M);

        %% Distributed consensus algor.
        % %%TODO: Set the distributed algor. config
        % Init.
        converged = false;
        iteration = 0;
        tolerance = 1e-2;
        % tolerance = 1e-2;
        max_iterations = 300;
        c_penalty = [1, 1, 10, 10]; % For SNR 50dB
        % c_penalty = [1e3,1e3, 3e5, 3e5]; % For SNR 50dxB
        % c_penalty = [1e2,1e2,3e3,3e3];
        % initial_values = repmat([1000, 1000, 10, 10]', 1,numNodes);
        initial_values = repmat([1000, 1000, 17, 17]', 1,numNodes);
        Nu = cell(1, numNodes);
        Nu_prev = cell(1, numNodes);
        update_z = cell(1, numNodes);
        update_z_prev = cell(1, numNodes);
        primal_residual_all =[];
        primal_residual_by_para = cell(1,numNodes);
        dual_residual_all = [];
        dual_residual_by_para = cell(1,numNodes);
        all_estimations_every_iter = [];
    
        % Define the parameters for adaptive penalty update
        % Define more conservative parameters for adaptive penalty update
        % tau_incr = [2.01, 2.01, 2.1, 2.1];  % Smaller increase factor
        % tau_decr = [2.01, 2.01, 2.1, 2.1];  % Smaller decrease factor
        % tau_incr = [50, 50, 50, 50];  % Smaller increase factor
        % tau_decr = [50, 50, 50, 50];  % Smaller decrease factor
        tau_incr = [1.1, 1.1, 1.1, 1.1];  % Smaller increase factor
        tau_decr = [1.1, 1.1, 1.1, 1.1];  % Smaller decrease factor
        mu = [3,3,10,10];          % Slightly smaller threshold ratio
        alpha = [0.5, 0.5, 0.5, 0.5];  % Damping factor
    
        for n = 1:network_topo.numNodes
            Nu{n} = zeros(4, network_topo.numNodes);
            Nu_prev{n} = zeros(4, network_topo.numNodes);
            update_z{n} = zeros(4, network_topo.numNodes);
            update_z_prev{n} = zeros(4, network_topo.numNodes);
            % dual_residual_by_para{n} = zeros(4,network_topo.numNodes);
            % primal_residual_by_para{n} = zeros(4,network_topo.numNodes);
        end
        % Start optmization
        while ~converged && iteration < max_iterations
            iteration = iteration+1;
            for n = 1: network_topo.numNodes
                for j = neighbors{n}
                    % Eq. (4.17c)
                    Nu{n}(:,j) = Nu_prev{n}(:,j) + c_penalty' .* (initial_values(:,n) - update_z_prev{n}(:,j));
                end
                % Eq. (4.17a)
                % fun = @(params) ut.logLikelihoodWithConsensus(params, range_with_error_cell{n}, doppler_with_error_cell{n}, radar_positions_cell{n},numNodes_cell{n}, M, env.lambda, Sigma_big_2_cell{n}, n, neighbors, Nu, initial_values, update_z_prev, c_penalty);
                fun = @(params) ut.posteriorWithConsensus(params, range_with_error_cell{n}, doppler_with_error_cell{n}, ...
                                                            mu_r_cell{n}, mu_d_cell{n}, sigma_r_cell{n}, sigma_d_cell{n}, radar_positions_cell{n}, ...
                                                            numNodes_cell{n}, M, env.lambda, Sigma_big_2_cell{n}, ...
                                                            n, neighbors, Nu, initial_values, update_z_prev, c_penalty);
                %TODO: mu_r, mu_d, doppler_r and doppler_d should at here,
                %also have to forward informantion of neighbors. 
                estimated_params = fmincon(fun, initial_values(:,n),[],[],[],[], lb, ub, [],options_DA);
                all_estimations(:,n) = estimated_params;
            end
            all_estimations_every_iter(:,:,iteration) = all_estimations;

            for n = 1: network_topo.numNodes
                for j = neighbors{n}
                    % Eq. (4.17b)
                    update_z{n}(:,j) = (1/2) * (((c_penalty.^(-1))' .* (Nu{n}(:,j) + Nu{j}(:,n))) + all_estimations(:,n) + all_estimations(:, j)); 
                end
            end
            
            
            
            
            % Calculate residual, for monitor convergence
            primal_residual = 0;
            dual_residual = 0; 
            primal_residual_params = zeros(4,10);
            dual_residual_params   = zeros(4,10);
            for n = 1: network_topo.numNodes
                for j = neighbors{n}
                    % Calculate the primal residual, Eq.4.18
                    primal_residual = primal_residual + norm(all_estimations(:,n) - update_z{n}(:,j), 2)^2;
                    % primal_residual = primal_residual + norm(all_estimations(:,n) - update_z{n}(:,j), 2);
                    primal_residual_params = primal_residual_params + abs(all_estimations(:,n) - update_z{n}(:,j));
                    % Calculate the dual residual, Eq.4.19
                    dual_residual = dual_residual + norm(Nu{n}(:,j) - Nu_prev{n}(:,j))^2;
                    % dual_residual = dual_residual + norm(Nu{n}(:,j) - Nu_prev{n}(:,j));
                    dual_residual_params = dual_residual_params + abs(Nu{n}(:,j) - Nu_prev{n}(:,j));
                end
            end
            primal_residual_all(iteration)     = primal_residual;
            dual_residual_all(iteration)       = dual_residual;
            primal_residual_by_para{iteration} = primal_residual_params;
            dual_residual_by_para{iteration}   = dual_residual_params; 

            % Stop criterion
            if primal_residual < tolerance
                break;
            end
    
            if iteration == max_iterations
                break;
            end
            
            % % Every 30 iteration, 
            % if mod(iteration, 30) == 0
            %     % Update the penalty parameter based on the residuals
            %     if primal_residual < 10* dual_residual
            %         c_penalty = tau_incr .* c_penalty;
            %     elseif dual_residual < 10*primal_residual
            %         c_penalty = c_penalty .* ((tau_decr).^(-1));
            %     end
            % end
            % Every iteration, 
            % Update the penalty parameter based on the residuals
            if primal_residual < 10* dual_residual
                c_penalty = tau_incr .* c_penalty;
            elseif dual_residual < 10*primal_residual
                c_penalty = c_penalty .* ((tau_decr).^(-1));
            end
    
            % Update the cur. term to next term (k -> k-1)
            initial_values = all_estimations;
            update_z_prev = update_z;
            Nu_prev = Nu;
        
        end
        
        iteration_CR(com_rad) = iteration;
        primal_residual_CR{com_rad} = primal_residual_all;
        dual_residual_CR{com_rad}  = dual_residual_all;
        all_estimations_every_iter_CR{com_rad} = all_estimations_every_iter;
    end
    true_params_mc(mc,:) = target.true_params;
    direction_mc(mc,:) = target.direction;
    iteration_mc{mc} = iteration_CR;
    primal_residual_mc{mc} = primal_residual_CR;
    dual_residual_mc{mc} = dual_residual_CR;
    all_estimations_every_iter_mc{mc} = all_estimations_every_iter_CR;

end


%% Figures
% True parameters
% set(gcf,'Color','white');
true_params = [target.initial_position(1), target.initial_position(2), target.speed * target.direction(1), target.speed * target.direction(2)];
fig_ut = make_figs(network_topo.numNodes);

%-- Plot covergence of ADMM across all nodes
% fig_ut.plot_converge_across_node(all_estimations_every_iter,true_params,network_topo);
fig_ut.plot_converge_across_node_withCentrl(all_estimations_every_iter,true_params,network_topo,estimates_mc_CA);
% %-- Plot dual & primal residual
% fig_ut.plot_dual_primal_residual(dual_residual_all,primal_residual_CR, all_estimations_every_iter_CR,laplacian_matrix_CR,network_topo);
% %-- 
% node_to_show = 3;
% fig_ut.plot_specific_node_converg(node_to_show,com_rad_CR,laplacian_matrix_CR,all_estimations_every_iter_CR,estimated_params_CA,network_topo,true_params);
% fig_ut.plot_sepcific_node_error_converg(all_estimations_every_iter_mc,estimates_mc_CA,direction_mc,true_params_mc);
% 
% %-- 
% fig_ut.plot_MSE_error(direction_mc,all_estimations_every_iter_mc,true_params_mc);
% 
% %-- Plot measurement errors of all neighhbors
% fig_ut.plot_errors_all_neighbors(com_rad_CR, laplacian_matrix_CR, all_estimations_every_iter_CR, estimated_params_CA,network_topo,true_params);
% 
% %-- 
% fig_ut.plot_MSE_for_all_neightbors(com_rad_CR,laplacian_matrix_CR,estimated_params_CA,all_estimations_every_iter_CR,true_params,network_topo);
% 
% %--
% fig_ut.plot_MSE_error_compare_DA_DS(direction_mc,all_estimations_every_iter_mc,estimates_mc_CA, true_params_mc);

% figure;
% 
% for param = 1:4
%     subplot(2, 2, param);
%     hold on;  % Allows multiple plots on the same axes
% 
%     for i = 1:size(primal_residual_by_para,2)
%         resiaul_by_iter = sum(primal_residual_by_para{i}(param,:));
%         plot(i,resiaul_by_iter,'o');
%     end 
%     set(gca, 'YScale', 'log');
% end