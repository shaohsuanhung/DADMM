%%%%%%%%%%% Author: Srikar Chaganti %%%%%%%%%%%%

% This is a distributed algorithm with ADMM with consensus
% It is working - Here I keep c_penalty constant
% Here the maximumIterations is 1000.

% Primal and Dual residuals are implemented. Stopping criterion is based on
% primal residual.
% Unit Disc Graph
% There is also code if you want to damage a single node (Like make it to send false data, like lying).
% Here we are updating the penalty terms (It is adaptive penalty terms) 
% We can compare with Centralized approach as well
% This is for 30 dB. Change in dB changes initial penalty terms allocation as well.


clc; close all; clear all;

num_monte_carlo = 1; % Number of Monte Carlo simulations
% angles_degrees = [45, 135, 225, 315]; % Angles for each simulation
angles_degrees = [135]; % Angles for each simulation
direction_mc = zeros(num_monte_carlo,2);
true_params_mc = zeros(num_monte_carlo, 4);

for mc = 1:num_monte_carlo
    
    
    disp('Monte carlo simulations: ');
    disp(mc);

    % Parameters of the target
    initial_position = [1000, 1000];
    speed = 20;
    simulation_time = 0.1;
    time_step = 1e-4;
    
%     angle_degrees = 215;  % Change this value to set the direction
    angle_degrees = angles_degrees(mc);
    angle_radians = angle_degrees * pi / 180;  % Convert angle to radians

    % Calculate direction vector from angle
    direction = [cos(angle_radians), sin(angle_radians)];
%     direction = direction / norm(direction);
    true_params = [initial_position(1), initial_position(2), speed * direction(1), speed * direction(2)];
    

    disp('Direction the target is travelling:');
    disp(direction);
    % Signal and environment parameters
    c = 3e8;
    lambda = c / 10e9;
    T = time_step / 2;
    % M_values = [5, 15, 30];
    M_values = 64;

    % Range of nodes
    % node_range = 20:3:20;
    node_range = 10;
    
    C_distance = 1;  % Cost per meter
    C_data = 1;      % Cost per byte
    
    
    
    %% Initialize M 
    
    % Loop over M values
    
    time_vector = 0:time_step:M_values * time_step; % Adjust time vector for each M
    
    % Preallocate for plotting
    RMSEs = zeros(length(node_range), 4); % 4 parameters
    
    
    
    % Loop over different number of nodes
    
    numNodes = node_range;
    
    %         % Circular
    theta = linspace(0, 2*pi, numNodes + 1);
    theta = theta(1:end-1);
    radius = 3000; %(Question: Why 3000 m?)
    radar_positions = radius * [cos(theta); sin(theta)]';
    
    
    % Semicircular
    %         theta = linspace(pi, 0, numNodes + 2); % divide the semicircle into numNodes+2 points
    %         theta = theta(2:end-1); % remove the first and last points (start and end of the semicircle)
    %         radius = 1000; % Radar nodes positions
    %         radar_positions = radius * [cos(theta); -sin(theta)]';
    
    % %         numNodes = node_range(idx);
    % %         theta = linspace(0, 2*pi, numNodes + 1);
    % %         theta = theta(1:end-1);
    % %         radius = 1000;
    % %         radar_positions = radius * [cos(theta); sin(theta)]';
    
    % Setting up the Adjacency Matrix
    % Compute the distance matrix
    radar_pos_T = radar_positions';
    distances_between_radar_nodes = zeros(numNodes, numNodes);
    for i = 1:numNodes
        for j = 1:numNodes
            distances_between_radar_nodes(i, j) = norm(radar_pos_T(:,i) - radar_pos_T(:,j));
        end
    end
    %[Idea] The distances_between_radar_nodes can be upper tri matrix, so
    %can be shrink to numNode*numNode to n^2/2
    
    
    
    %         knn = numNodes:2:numNodes;
     knn = 5:2:5;%[Question] What's the knn variable for? K nearest neighbor?
%      com_rad_CR = [1000, 2000, 3000, 4000, 4500, 5000, 5500, 6000];
%     com_rad_CR = [3000, 4500, 5500, 6000];
     com_rad_CR = 3000;
    
%      com_rad_CR = 500:250:1500;
    
    % Select k-nearest neighbors
    
    
    disp('Node Range:');
    disp(node_range);
    disp('Number of Measurements:');
    disp(M_values);
    
    
    
    % What is options_DA / CA???? -> Decentralize, centralized
    options_DA = optimoptions('fmincon', 'Display', 'off', ScaleProblem=true, OptimalityTolerance=1e-6, FunctionTolerance=1e-6, StepTolerance=1e-6, MaxIterations=100);
    options_CA = optimoptions('fmincon', 'Display', 'off', ScaleProblem=true, OptimalityTolerance=1e-6, FunctionTolerance=1e-6, StepTolerance=1e-6, MaxIterations=100000);
    
    global estimated_param_values;
    global log_likelihood_values;
    log_likelihood_values = [];
    
    
    estimates_mc = [];
    %         for snr_idx = 1:num_SNR
    for com_rad = 1:length(com_rad_CR)
        fprintf('Communication Radius: %d\n',  com_rad_CR(com_rad));
    
        communication_radius = com_rad_CR(com_rad); % Define communication radius
    
        % Initialize the adjacency matrix
        adj_matrix = zeros(numNodes, numNodes);
        
        % Calculate the adjacency matrix
        for i = 1:numNodes
            for j = i+1:numNodes % Since A is symmetric, compute only for j > i
                if norm(radar_positions(i,:) - radar_positions(j,:)) <= communication_radius
                    adj_matrix(i,j) = 1;
                    adj_matrix(j,i) = 1; % Symmetric
                end
            end
        end
        
        % Create a cell array to hold the neighbors of each node
        neighbors = cell(numNodes, 1);
        
        for i = 1:numNodes
            neighbors{i} = find(adj_matrix(i, :) == 1);
        end
        
        % Compute the degree matrix
        degree_matrix = diag(sum(adj_matrix, 2));
        
        % Compute the Laplacian matrix
        laplacian_matrix = degree_matrix - adj_matrix;
        laplacian_matrix_CR{com_rad}=laplacian_matrix;
        
        laplacian_eigen = eig(laplacian_matrix);
        laplacian_eigen_vector = sort(laplacian_eigen, 'descend');
        
        % Calculate incident matrix
        numEdges = sum(adj_matrix(:)) / 2; % Each edge is counted twice in adjacency matrix
        inc_matrix = zeros(numNodes, numEdges);
        
        edge_index = 1;
        for i = 1:numNodes
            for j = i+1:numNodes
                if adj_matrix(i, j) == 1
                    inc_matrix(i, edge_index) = 1;
                    inc_matrix(j, edge_index) = -1;
                    edge_index = edge_index + 1;
                end
            end
        end
        
        %maximum degree weights constants
        alpha = 1 / max(diag(degree_matrix));
        
        % Initialize final weight matrix
        weight_matrix = zeros(numNodes);
        
        % Compute weight matrix
        for i = 1:numNodes
            for j = 1:numNodes
                if i == j
                    weight_matrix(i, j) = 1 - alpha * sum(adj_matrix(i, :));
                elseif adj_matrix(i, j) == 1
                    weight_matrix(i, j) = alpha;
                else
                    weight_matrix(i, j) = 0;
                end
            end
        end
    
        
        
        B = 10e6 * ones(1, numNodes); % Bandwidth in Hz
        fs = 2 * B; % Sampling frequency
    
        snr_idx = 50;
        SNR_lin = 10^(snr_idx / 10);
    %     disp(['SNR_idx: ', num2str(snr_idx)]);
        fprintf('SNR_db: %d dB, SNR_linear: %f\n', snr_idx, SNR_lin);
    
    
        % Monte Carlo simulations
        
        estimated_param_values = [];
    
    
        M = M_values;
    
        target_position = zeros(M, 2);
        target_position(1, :) = initial_position; % Initial position
        % Target position over
        for k = 2:M
            target_position(k, :) = target_position(k - 1, :) + speed * direction * time_step;
        end
    
        % Matrices for range and Doppler true data (This is a matrix of M x N)
        range_true = zeros(M, numNodes);
        doppler_true = zeros(M, numNodes);
        measurements_true = zeros(2 * M, numNodes);
    
        % Calculate range and Doppler true measurements, M x N per param
        for t = 1:M
            for r = 1:numNodes
                % Calculate range data
                range_true(t, r) = norm(target_position(t, :) - radar_positions(r, :));
    
                % Calculate Doppler shift data
                relative_position = radar_positions(r, :) - target_position(t, :);
                doppler_true(t, r) = dot([speed * direction], relative_position) / (norm(relative_position) * lambda);
    
                % Store the true measurements: range followed by Doppler
                measurements_true(2 * t - 1, r) = range_true(t, r);  % Odd index for range
                measurements_true(2 * t, r) = doppler_true(t, r);    % Even index for Doppler
            end
        end
        
        measurements_true_all = reshape(measurements_true, [], 1);
    
        % Calculate Noise for range and Doppler 
        range_var = (3 * c^2) / (8 * pi^2 * B(1)^2 * SNR_lin) ;
        doppler_var = (3 * ((fs(1))^2)) / (pi^2 * SNR_lin * M^3) ;
        range_sd = sqrt(range_var);
        doppler_sd = sqrt(doppler_var);
        rho = 0.0;
    
        % 2x2 covariance matrix Sigma
        Sigma = [range_var, rho * range_sd * doppler_sd; rho * range_sd * doppler_sd, doppler_var];
    
        total_measurements = numNodes * M;
    
        % Sigma_big is of the size 2NM x 2NM, 2=two params, NM=total mea
        Sigma_big = kron(eye(total_measurements), Sigma);
    
        noise_matrix = mvnrnd(zeros(2 * total_measurements, 1), Sigma_big)';
    
        % Generate range and Doppler shift's noisy measurements
        range_noise = noise_matrix(1:2:end);
        range_noise_all = reshape(range_noise, M, numNodes);
        doppler_noise = noise_matrix(2:2:end);
        doppler_noise_all = reshape(doppler_noise, M, numNodes);
    
        range_with_error = range_true + range_noise_all;
        doppler_with_error = doppler_true + doppler_noise_all;
    
       
        measurements_with_error_all = measurements_true_all + noise_matrix;
        
        
    
        range_with_error_cell = cell(1, numNodes);
        doppler_with_error_cell = cell(1, numNodes);
        numNodes_cell = cell(1, numNodes);
        radar_positions_cell = cell(1, numNodes);
        Sigma_big_1_cell = cell(1, numNodes);
        Sigma_big_2_cell = cell(1, numNodes);
        
        range_with_error_CA = range_with_error;
        doppler_with_error_CA = doppler_with_error;
%         range_with_error_CA(:,3) = range_with_error_CA(:,3) - 400;
%         doppler_with_error_CA(:,3) = doppler_with_error_CA(:,3) - 200;
        
        initial_guess = [1000, 1000, 20, 20];
        lb = [-inf,-inf,-inf,-inf];
        ub = [inf,inf, inf, inf];
    
        fun = @(params) logLikelihood(params, range_with_error_CA, doppler_with_error_CA, radar_positions, numNodes, M, lambda, Sigma_big);
        [estimated_params_CA, log_likelihood, exitflag, output] = fmincon(fun, initial_guess,[],[],[],[], lb, ub, [],options_CA);
        estimates_mc_CA(mc, :) = estimated_params_CA;
    
        for n = 1:numNodes
            current_neighbors = find(laplacian_matrix(n, :) ~= 0);
            k = 0;
    %                     Sigma_big_1 = zeros(2*M*length(current_neighbors),1);
            Sigma_big_1 = [];
            Sigma_big_2 = [];
            range_with_error_1 =[];
            doppler_with_error_1 = [];
            radar_positions_1 = [];
    %                     current_neighbors = [current_neighbors, n];
            for j = current_neighbors % current_neightsbos is a array,
                k = k+1; % Assign the in order. (0,1,2,...)
                
                range_with_error_1(:,k) = range_with_error(:,j);
                doppler_with_error_1(:,k) = doppler_with_error(:,j);
                radar_positions_1(k,:) = radar_positions(j,:);
                numNodes_1 = length(current_neighbors);
                
                for i =1:M
                    base_idx = 2 * (M * (j - 1) + (i - 1)) + 1;
                    sigma_r2 = Sigma_big(base_idx, base_idx);
                    Sigma_big_1(((k-1)*M + (i-1))*2 + 1) = sigma_r2; 
                    sigma_fd2 = Sigma_big(base_idx + 1, base_idx + 1);
                    Sigma_big_1(((k-1)*M + (i-1))*2 + 2) = sigma_fd2;
    
                end
                Sigma_big_2 = diag(Sigma_big_1);
    
                % Store the values in cell arrays
                range_with_error_cell{n} = range_with_error_1;
                doppler_with_error_cell{n} = doppler_with_error_1;
                numNodes_cell{n} = numNodes_1;
                radar_positions_cell{n} = radar_positions_1;
                Sigma_big_1_cell{n} = Sigma_big_1;
                Sigma_big_2_cell{n} = Sigma_big_2;
    
            end
            
        end
    
    
    %                 for n = 1:numNodes
    %                     fun = @(params) logLikelihood(params, range_with_error_cell{n}, doppler_with_error_cell{n}, radar_positions_cell{n}, numNodes_cell{n}, M, lambda, Sigma_big_2_cell{n});
    %                     [estimated_params, log_likelihood, exitflag, output] = fmincon(fun, initial_guess,[],[],[],[], lb, ub, [],options);
    %                     
    %                     all_estimations_1_iter(:,n) = estimated_params;
    %                 end
        
    
    
        % Code for distributed consensus
        % Initialization
        converged = false;
        iteration = 0;
        tolerance = 1e-4;
        max_iterations = 1e5;
%     %                 c_penalty = 10^14;
        c_penalty = [10^2, 10^2, 3*5, 3*5]; % For SNR 50dB
%         c_penalty = [10^4, 10^4, 3*10^2, 3*10^2];
        initial_values = repmat([1000, 1000, 10, 10]', 1,numNodes);
        Nu = cell(1, numNodes);
        Nu_prev = cell(1, numNodes);
        update_z = cell(1, numNodes);
        update_z_prev = cell(1, numNodes);
        primal_residual_all =[];
        dual_residual_all = [];
        all_estimations_every_iter = [];
    
        % Define the parameters for adaptive penalty update
        % Define more conservative parameters for adaptive penalty update
        tau_incr = [2.01, 2.01, 2.1, 2.1];  % Smaller increase factor
        tau_decr = [2.01, 2.01, 2.1, 2.1];  % Smaller decrease factor
        mu = [3,3,10,10];          % Slightly smaller threshold ratio
        alpha = [0.5, 0.5, 0.5, 0.5];  % Damping factor
    
    
        for n = 1:numNodes
            Nu{n} = zeros(4, numNodes);
            Nu_prev{n} = zeros(4, numNodes);
            update_z{n} = zeros(4, numNodes);
            update_z_prev{n} = zeros(4, numNodes);
        end
        
    %                 update_z = cell(zeros)
    
    
        while ~converged && iteration < max_iterations
            iteration = iteration+1;
    
    
            if iteration == 1
                range_with_error_cell = cell(1, numNodes);
                doppler_with_error_cell = cell(1, numNodes);
                numNodes_cell = cell(1, numNodes);
                radar_positions_cell = cell(1, numNodes);
                Sigma_big_1_cell = cell(1, numNodes);
                Sigma_big_2_cell = cell(1, numNodes);
%                 range_with_error(:,3) = range_with_error(:,3)- 400;
%                 doppler_with_error(:,3) = doppler_with_error(:,3)-200;
    %                         range_with_error(:,3) = 0;
    %                         doppler_with_error(:,3) = 0;
    
    
                for n = 1:numNodes
                    current_neighbors = find(laplacian_matrix(n, :) ~= 0);
                    k = 0;
    %                     Sigma_big_1 = zeros(2*M*length(current_neighbors),1);
                    Sigma_big_1 = [];
                    Sigma_big_2 = [];
                    range_with_error_1 =[];
                    doppler_with_error_1 = [];
                    radar_positions_1 = [];
    %                     current_neighbors = [current_neighbors, n];
                    for j = current_neighbors
                        k = k+1;
                        
                        range_with_error_1(:,k) = range_with_error(:,j);
                        doppler_with_error_1(:,k) = doppler_with_error(:,j);
                        radar_positions_1(k,:) = radar_positions(j,:);
                        numNodes_1 = length(current_neighbors);
                        
                        for i =1:M
                            base_idx = 2 * (M * (j - 1) + (i - 1)) + 1;
                            sigma_r2 = Sigma_big(base_idx, base_idx);
                            Sigma_big_1(((k-1)*M + (i-1))*2 + 1) = sigma_r2; 
                            sigma_fd2 = Sigma_big(base_idx + 1, base_idx + 1);
                            Sigma_big_1(((k-1)*M + (i-1))*2 + 2) = sigma_fd2;
    
                        end
                        Sigma_big_2 = diag(Sigma_big_1);
    
                        % Store the values in cell arrays
                        range_with_error_cell{n} = range_with_error_1;
                        doppler_with_error_cell{n} = doppler_with_error_1;
                        numNodes_cell{n} = numNodes_1;
                        radar_positions_cell{n} = radar_positions_1;
                        Sigma_big_1_cell{n} = Sigma_big_1;
                        Sigma_big_2_cell{n} = Sigma_big_2;
    
                    end
                    
                end
    %                         c_penalty = [3.5*10^6, 3.5*10^6, 3*10^4, 3*10^4];
            end
    
    %                     
%             disp(['Iteration: ', num2str(iteration)]);
    %                     sum_nodes_primal = 0;
    %                     sum_nodes_dual = 0;
            for n = 1:numNodes
                for j = neighbors{n}
                    Nu{n}(:,j) = Nu_prev{n}(:,j) + c_penalty' .* (initial_values(:,n) - update_z_prev{n}(:,j));
                end

                fun = @(params) logLikelihoodWithConsensus(params, range_with_error_cell{n}, doppler_with_error_cell{n}, radar_positions_cell{n}, numNodes_cell{n}, M, lambda, Sigma_big_2_cell{n}, n, neighbors, Nu, initial_values, update_z_prev, c_penalty);

                estimated_params = fmincon(fun, initial_values(:,n),[],[],[],[], lb, ub, [],options_DA);
                all_estimations(:,n) = estimated_params;


            end
            all_estimations_every_iter(:,:,iteration) = all_estimations;

            for n = 1:numNodes
                for j = neighbors{n}
                    update_z{n}(:,j) = (1/2) * (((c_penalty.^(-1))' .* (Nu{n}(:,j) + Nu{j}(:,n))) + all_estimations(:,n) + all_estimations(:, j)); 
                end
            end


    
            primal_residual = 0;
            dual_residual = 0;
            
            for n = 1:numNodes
                for j = neighbors{n}
                    % Calculate the primal residual
                    primal_residual = primal_residual + norm(all_estimations(:,n) - update_z{n}(:,j), 2)^2;
            
                    % Calculate the dual residual
                    dual_residual = dual_residual + norm(Nu{n}(:,j) - Nu_prev{n}(:,j))^2;
                end
            end
            
            % Scale the dual residual by the penalty constant
    %                     dual_residual = c_penalty * dual_residual;
    
            primal_residual_all(iteration) = primal_residual;
            dual_residual_all(iteration) = dual_residual;
    
    %                     if iteration>3
    %                         prev_iter = iteration-1;
    %                         if abs(primal_residual_all(iteration)- primal_residual_all(prev_iter))<1e-4
    %                             break;
    %                         end
    %                     end
    
            if primal_residual<tolerance
                break;
            end
            
            if iteration == 3000
                break;
            end
    
            % Display residuals for diagnostics
%             disp(['Primal Residual: ', num2str(primal_residual)]);
%             disp(['Dual Residual: ', num2str(dual_residual)]);
%             disp(['Updated Penalty: ', num2str(c_penalty)]);
    
    
            if mod(iteration, 30) == 0
                % Update the penalty parameter based on the residuals
                if primal_residual < 10 * dual_residual
                    c_penalty = tau_incr .* c_penalty;
                elseif dual_residual < 10 * primal_residual
                    c_penalty = c_penalty .* ((tau_decr).^(-1));
                end
            end
                
            % Apply damping to the dual variable updates
    %                     for n = 1:numNodes
    %                         for j = neighbors{n}
    %                             Nu{n}(:,j) = alpha * Nu{n}(:,j) + (1 - alpha) * Nu_prev{n}(:,j);
    %                         end
    %                     end
    
            initial_values = all_estimations;
            update_z_prev = update_z;
            Nu_prev = Nu;
            
        end
        
    
    
    
        
        iteration_CR(com_rad) = iteration;
        primal_residual_CR{com_rad} = primal_residual_all;
    %             primal_residual_CR(:, com_rad) = primal_residual_all;
        dual_residual_CR{com_rad} = dual_residual_all;
    %             dual_residual_CR(:, com_rad) = dual_residual_all;
        all_estimations_every_iter_CR{com_rad} = all_estimations_every_iter;
    %             all_estimations_every_iter_CR(:,:,:,com_rad) = all_estimations_every_iter;
    
    end
    true_params_mc(mc,:) = true_params;
    direction_mc(mc,:) = direction;
    iteration_mc{mc} = iteration_CR;
    primal_residual_mc{mc} = primal_residual_CR;
    dual_residual_mc{mc} = dual_residual_CR;
    all_estimations_every_iter_mc{mc} = all_estimations_every_iter_CR;

end


    


%% Figures 

% % To plot convergence of ADMM across all the nodes
colors = lines(numNodes);  % MATLAB function that provides distinct colors
labels_params = {'Position x ', 'Position y ', 'Velocity x ', 'Velocity y '};

% True parameters
true_params = [initial_position(1), initial_position(2), speed * direction(1), speed * direction(2)];

figure;
for param = 1:4
    subplot(2, 2, param);
    hold on;  % Allows multiple plots on the same axes

    % Plot estimations for each node
    for node = 1:numNodes
        plot(squeeze(all_estimations_every_iter(param, node, :)), 'Color', colors(node, :));
    end

    % Plot true parameter as a dotted line
    h_true = yline(true_params(param), '--k', 'LineWidth', 1.5);

    hold off;
    xlabel('Iterations (k)');
    ylabel(['Parameter ' labels_params{param} 'estimates']);
    title([labels_params{param}]);
    legend_entries = arrayfun(@(x) ['Node ' num2str(x)], 1:numNodes, 'UniformOutput', false);
    legend_entries{end+1} = 'True Parameter';
    legend([legend_entries], 'Location', 'northeastoutside');
end
sgtitle('Parameters of interest (\theta) Estimations');
% % 
% % 
% % Primal Residual Convergence
figure;
semilogy(primal_residual_all);
xlabel("Iterations");
% ylabel('|| \theta(k+1) - \vartheta(k+1)||_2^2');
ylabel('$\sum_{n=1}^{N} \sum_{j \in \mathrm{Neighbors}(n)} ||\theta_m(k+1) - \vartheta_{nj}(k+1)||_2^2$', 'Interpreter', 'latex');
title('Primal Residual (r(k))');
% 
% Dual Residual Convergence
figure;
semilogy(dual_residual_all);
xlabel("Iterations (k)");
% ylabel('|| \nu(k+1) - \nu(k)||_2^2');
ylabel('$\sum_{n=1}^{N} \sum_{j \in \mathrm{Neighbors}(n)} || \nu_{n|j}(k+1) - \nu{n|j}(k)||_2^2$', 'Interpreter', 'latex');
title('Dual Residual (s(k))');

% Primal and Dual Convergence for various node ratio
numColors = length(all_estimations_every_iter_CR);  % Define number of distinct colors needed
colors = hsv(numColors);  % Creates a colormap with numColors distinct colors

display_node = 1;
legendNames = cell(1, length(primal_residual_CR)); 
for i = 1:length(primal_residual_CR)
    legendNames{i} = ['Neighbors: ', num2str(laplacian_matrix_CR{i}(display_node, display_node)), '/', num2str(numNodes)];
end

% Plot Primal Residual Convergence for various node ratios
figure;
for CR = 1:numColors
    semilogy(primal_residual_CR{CR}, 'Color', colors(CR, :));
    hold on;
end
xlabel('Iterations (k)');
ylabel('$\sum_{n=1}^{N} \sum_{j \in \mathrm{Neighbors}(n)} ||\theta_m(k+1) - \vartheta_{nj}(k+1)||_2^2$', 'Interpreter', 'latex');
title('Primal Residual (r(k)) for different node ratios');
legend(legendNames, 'Location', 'best');  % Add legend with node ratios

% Plot Dual Residual Convergence for various communication radius
figure;
for CR = 1:numColors
    semilogy(dual_residual_CR{CR}, 'Color', colors(CR, :));
    hold on;
end
xlabel("Iterations (k)");
ylabel('$\sum_{n=1}^{N} \sum_{j \in \mathrm{Neighbors}(n)} || \nu_{n|j}(k+1) - \nu{n|j}(k)||_2^2$', 'Interpreter', 'latex');
title('Dual Residual (s(k)) for different communication radius');
legend(legendNames, 'Location', 'best');  % Add legend with custom names



% Plot Convergance of theta for various node ratios
display_node = 1;
labels_params = {'Position x ', 'Position y ', 'Velocity x ', 'Velocity y '};
legendNames = cell(1, length(com_rad_CR)); 

% Construct the Node Ratio as a fraction in the format 'numerator/denominator'
for i = 1:length(com_rad_CR)
    legendNames{i} = ['Neighbors: ', num2str(laplacian_matrix_CR{i}(display_node, display_node)), '/', num2str(numNodes)];
end

figure;
for param = 1:4
    subplot(2,2,param);
    hold on;

    % Initialize a cell array to store plot handles
    hPlots = cell(1, length(all_estimations_every_iter_CR) + 2); % +2 for true parameter and centralized approach

    % Plot all estimations with specific color and store handles
    for j = 1:length(all_estimations_every_iter_CR)
        hPlots{j} = plot(squeeze(all_estimations_every_iter_CR{j}(param, display_node, :)), 'Color', colors(j, :));        
    end

    % Plot the centralized approach as a yline
    hPlots{end-1} = yline(estimated_params_CA(param), 'k', 'LineWidth', 1.5, 'DisplayName', 'Centralized Approach');

    % Plot the true parameter line and store the handle
    hPlots{end} = yline(true_params(param), '--k', 'LineWidth', 1.5, 'DisplayName', 'True Parameter');

    % Add the legend
    legend([hPlots{:}], [legendNames, {'Centralized Approach', 'True Parameter'}]);

    hold off;
    xlabel('Iterations (k)');
    ylabel(['Parameter ' labels_params{param} ' estimates']);
    title([labels_params{param}]);

    hold off;


end

sgtitle({'Convergence of parameters of interest ($\theta_n$) at $n^{\mathrm{th}}$ node for different communication radius'}, 'Interpreter', 'latex');
%%
% Plot Error (Estimated(with neighbors ratio) - True Params)
% Plotting Errors (\hat{\theta} - \theta) results
display_node = 1;
colors_size = size(direction_mc, 1);
colors = hsv(colors_size);
labels_params = {'Position x', 'Position y', 'Velocity x', 'Velocity y'};
legendNames = cell(1, size(direction_mc, 1)); 

% Construct the legend names based on the rows of direction_mc
for i = 1:size(direction_mc, 1)
        % Format the direction values to two decimal places
        directionStr = num2str(direction_mc(i, :), '%0.2f ');
        legendNames{i} = ['Direction: ', directionStr];
end

% figure;
% for param = 1:4
%     subplot(2,2,param);
%     hold on;
% 
%     % Initialize a cell array to store plot handles
%     hPlots = cell(1, length(all_estimations_every_iter_mc)+2); % +2 for centralized approach and true parameter error line
% 
%     % Plot estimation errors with specific color and store handles
%     for j = 1:length(all_estimations_every_iter_mc)
%         % Calculate estimation error as estimation minus true parameter
%         errors = squeeze(all_estimations_every_iter_mc{j}{5}(param, display_node, :)) - true_params_mc(j,param);
%         hPlots{j} = semilogy(errors, 'Color', colors(j, :));        
%     end
% 
%     % Plot the centralized approach error as a horizontal line at zero (assuming centralized approach estimates the parameter correctly)
%     hPlots{end-1} = yline(estimates_mc_CA(j,param) - true_params_mc(j,param), 'k', 'LineWidth', 1.5, 'DisplayName', 'Centralized Approach Error');
% 
%     % Plot the true parameter error line (which should be zero) and store the handle
%     hPlots{end} = yline(0, '--k', 'LineWidth', 1.5, 'DisplayName', 'Zero Error');
% 
%     % Add the legend
%     legend([hPlots{:}], [legendNames, {'Centralized Approach Error', 'Zero Error'}]);
% 
%     hold off;
%     xlabel('Iterations (k)');
%     ylabel(['Error for ' labels_params{param}]);
%     title(['Error in ' labels_params{param}]);
% 
% end
% 
% sgtitle({'Error convergence for parameters of interest ($\theta_n$) at $n^{\mathrm{th}}$ node for different communication radius'}, 'Interpreter', 'latex');


% Plot MSE Error (Estimated(with neighbors ratio) - True Params).^2
% Plotting Errors (\hat{\theta} - \theta).^2 results
display_node = 1;
labels_params = {'Position x', 'Position y', 'Velocity x', 'Velocity y'};
legendNames = cell(1, size(direction_mc, 1)); 

% Construct the legend names based on the rows of direction_mc
for i = 1:size(direction_mc, 1)
        % Format the direction values to two decimal places
        directionStr = num2str(direction_mc(i, :), '%0.2f ');
        legendNames{i} = ['Direction: ', directionStr];
end

% figure;
% for param = 1:4
%     subplot(2,2,param);
%     hold on;
% 
%     % Initialize a cell array to store plot handles
%     hPlots = cell(1, length(all_estimations_every_iter_mc)); % +2 for centralized approach and true parameter error line
% 
%     % Plot estimation errors with specific color and store handles
%     for j = 1:length(all_estimations_every_iter_mc)
%         % Calculate estimation error as estimation minus true parameter
%         errors = (squeeze(all_estimations_every_iter_mc{j}{5}(param, display_node, :)) - true_params_mc(j,param)).^2;
%         hPlots{j} = semilogy(errors, 'Color', colors(j, :));        
%     end
%     set(gca, 'YScale', 'log');
% 
%     % Plot the centralized approach error as a horizontal line at zero (assuming centralized approach estimates the parameter correctly)
% %     hPlots{end-1} = yline((estimates_mc_CA(j,param) - true_params_mc(j,param)).^2, 'k', 'LineWidth', 1.5, 'DisplayName', 'Centralized Approach Error');
% %     
% %     % Plot the true parameter error line (which should be zero) and store the handle
% %     hPlots{end} = yline(0, '--k', 'LineWidth', 1.5, 'DisplayName', 'Zero Error');
% %     
%     % Add the legend
% %     legend([hPlots{:}], [legendNames, {'Centralized Approach Error', 'Zero Error'}]);
%     legend([hPlots{:}], legendNames);
% 
% 
%     hold off;
%     xlabel('Iterations (k)');
%     ylabel(['MSE for ' labels_params{param}]);
%     title(['MSE in ' labels_params{param}]);
% 
% end

% sgtitle({'MSE convergence for parameters of interest ($\theta_n$) at $n^{\mathrm{th}}$ node for different communication radius'}, 'Interpreter', 'latex');


%
% Plotting Errors (\hat{\theta} - \theta) results for all neighbors
display_node = 1;
colors = hsv(numColors);
labels_params = {'Position x', 'Position y', 'Velocity x', 'Velocity y'};
legendNames = cell(1, length(com_rad_CR)); 

% Construct the Node Ratio as a fraction in the format 'numerator/denominator'
for i = 1:length(com_rad_CR)
    legendNames{i} = ['Neighbors: ', num2str(laplacian_matrix_CR{i}(display_node, display_node)), '/', num2str(numNodes)];
end

figure;
for param = 1:4
    subplot(2,2,param);
    hold on;

    % Initialize a cell array to store plot handles
    hPlots = cell(1, length(all_estimations_every_iter_CR) + 2); % +2 for centralized approach and true parameter error line

    % Plot estimation errors with specific color and store handles
    for j = 1:length(all_estimations_every_iter_CR)
        % Calculate estimation error as estimation minus true parameter
        errors = squeeze(all_estimations_every_iter_CR{j}(param, display_node, :)) - true_params(param);
        hPlots{j} = semilogy(errors, 'Color', colors(j, :));        
    end

    % Plot the centralized approach error as a horizontal line at zero (assuming centralized approach estimates the parameter correctly)
    hPlots{end-1} = yline(estimated_params_CA(param) - true_params(param), 'k', 'LineWidth', 1.5, 'DisplayName', 'Centralized Approach Error');

    % Plot the true parameter error line (which should be zero) and store the handle
    hPlots{end} = yline(0, '--k', 'LineWidth', 1.5, 'DisplayName', 'Zero Error');

    % Add the legend
    legend([hPlots{:}], [legendNames, {'Centralized Approach Error', 'Zero Error'}]);

    hold off;
    xlabel('Iterations (k)');
    ylabel(['$(', '\hat{\theta}_n', ' - \theta)$' ], 'Interpreter', 'latex');
    title(['For ' labels_params{param}]);

end

sgtitle({'$\left(\hat{\theta}_n - \theta\right)$ for different neighbor nodes'}, 'Interpreter', 'latex');



% For MSE fo all neighbors
display_node = 1;
labels_params = {'Position x', 'Position y', 'Velocity x', 'Velocity y'};
legendNames = cell(1, length(com_rad_CR)); 

% Construct the Node Ratio as a fraction in the format 'numerator/denominator'
for i = 1:length(com_rad_CR)
    legendNames{i} = ['Neighbors: ', num2str(laplacian_matrix_CR{i}(display_node, display_node)), '/', num2str(numNodes)];
end
figure;
for param = 1:4
    subplot(2,2,param);
    hold on;

    % Initialize a cell array to store plot handles
    hPlots = cell(1, length(all_estimations_every_iter_CR) + 2); % +2 for centralized approach and true parameter error line

    % Plot estimation errors with specific color and store handles
    for j = 1:length(all_estimations_every_iter_CR)
        % Calculate estimation error as estimation minus true parameter
        errors = (squeeze(all_estimations_every_iter_CR{j}(param, display_node, :)) - true_params(param)).^2;
        hPlots{j} = semilogy(errors, 'Color', colors(j, :));        
    end

    % Plot the centralized approach error as a horizontal line at zero (assuming centralized approach estimates the parameter correctly)
    hPlots{end-1} = yline((estimated_params_CA(param) - true_params(param)).^2, 'k', 'LineWidth', 1.5, 'DisplayName', 'Centralized Approach Error');

    % Plot the true parameter error line (which should be zero) and store the handle
    hPlots{end} = yline(0, '--k', 'LineWidth', 1.5, 'DisplayName', 'Zero Error');

    % Add the legend
    legend([hPlots{:}], [legendNames, {'Centralized Approach Error', 'Zero Error'}]);

    set(gca, 'YScale', 'log');  % Set the Y-axis to logarithmic scale

    hold off;
    xlabel('Iterations (k)');
    ylabel(['$(', '\hat{\theta}_n', ' - \theta)^2$'], 'Interpreter', 'latex');
    title(['For ' labels_params{param}]);

    % The rest of your code for the inset plots
    % Ensure to set 'YScale' to 'log' for the inset axes if needed
end

sgtitle({'$\left(\hat{\theta}_n - \theta\right)^2$ for Different neighbors ratio'}, 'Interpreter', 'latex');


% Plot MSE error for (\hat{\theta} - \theta).^2 with Decentralized as straight line and distributed as dashed
display_node = 1;
labels_params = {'Position x', 'Position y', 'Velocity x', 'Velocity y'};
legendNames = cell(1, size(direction_mc, 1)); 
colors = hsv(colors_size);
% Construct the legend names based on the rows of direction_mc
for i = 1:size(direction_mc, 1)
        % Format the direction values to two decimal places
        directionStr = num2str(direction_mc(i, :), '%0.2f ');
        legendNames{i} = ['Direction: ', directionStr];
end
% figure;
% legendEntries = {};  % Initialize an empty cell array to hold legend entries
% for param = 1:4
%     subplot(2,2,param);
%     hold on;
% 
%     % Initialize a cell array to store plot handles
%     hPlots = []; % Use a simple array to store handles
% 
%     for j = 1:length(all_estimations_every_iter_mc)
%         % Define colors for this particular direction
%         color = colors(j, :);
% 
%         % Calculate errors for decentralized
%         errors = (squeeze(all_estimations_every_iter_mc{j}{5}(param, display_node, :)) - true_params_mc(j,param)).^2;
%         % Decentralized plot (straight line)
%         hPlotDecentralized = semilogy(errors, 'Color', color, 'LineStyle', '-');
%         hPlots = [hPlots, hPlotDecentralized];  % Append handle
%         legendEntries{end+1} = [legendNames{j}, ' Decentralized'];  % Append legend entry
% 
%         % Calculate centralized estimation error
%         ca_error = (estimates_mc_CA(j,param) - true_params_mc(j,param)).^2;
%         % Centralized plot (dashed line) using the same color
%         hPlotCentralized = semilogy(1:length(errors), repmat(ca_error, 1, length(errors)), 'Color', color, 'LineStyle', '--');
%         hPlots = [hPlots, hPlotCentralized];  % Append handle
%         legendEntries{end+1} = [legendNames{j}, ' Centralized'];  % Append legend entry
%     end
% 
%     set(gca, 'YScale', 'log');
%     legend(hPlots, legendEntries, 'Location', 'bestoutside');
% 
%     hold off;
%     xlabel('Iterations (k)');
%     ylabel(['$(', '\hat{\theta}_n', ' - \theta)^2$'], 'Interpreter', 'latex');
%     title(['For ' labels_params{param}]);
% end
% 
% sgtitle('$\left(\hat{\theta}_n - \theta\right)^2$ for different directions with 19/20 neighbors', 'Interpreter', 'latex');

%% Fisher Information Matrix (FIM) calculation
function FIM = calculateFIM(true_params, radar_positions, numNodes, M, lambda, Sigma)
    x_tar = true_params(1);
    y_tar = true_params(2);
    v_x = true_params(3);
    v_y = true_params(4);

    FIM = zeros(4, 4);
    Sigma_inv = inv(Sigma);  % Using the smaller Sigma meant for single measurements

    for j = 1:numNodes
        x_j = radar_positions(j, 1);
        y_j = radar_positions(j, 2);

        for i = 1:M
            relative_position = [x_j - x_tar, y_j - y_tar];
            norm_rel_pos = norm(relative_position);
            r_model = norm_rel_pos;
            f_d_model = dot([v_x, v_y], relative_position) / (norm_rel_pos * lambda);

            % Partial derivatives of range with respect to parameters
            dr_dx = (x_tar - x_j) / r_model;
            dr_dy = (y_tar - y_j) / r_model;
            dr_dvx = 0;
            dr_dvy = 0;

            % Partial derivatives of Doppler with respect to parameters
            df_dvx = relative_position(1) / (norm_rel_pos * lambda);
            df_dvy = relative_position(2) / (norm_rel_pos * lambda);
            df_dx = -dot([v_x, v_y], relative_position) * (x_tar - x_j) / (norm_rel_pos^3 * lambda);
            df_dy = -dot([v_x, v_y], relative_position) * (y_tar - y_j) / (norm_rel_pos^3 * lambda);

            % Jacobian matrix for the i-th measurement
            J_i = [dr_dx, dr_dy, dr_dvx, dr_dvy; df_dx, df_dy, df_dvx, df_dvy];

            % Update FIM
            FIM = FIM + J_i'* Sigma_inv * J_i;
        end
    end
end




%% Log Likelihood Function for MLE

function log_likelihood = logLikelihood(params, range_with_error, doppler_with_error, radar_positions, numNodes, M, lambda, Sigma_big)

    x_tar = params(1);
    y_tar = params(2);
    v_x = params(3);
    v_y = params(4);
    log_likelihood = 0;

    idx = 1; % Index to access the correct elements in Sigma_big

    for j = 1:numNodes
        x_j = radar_positions(j, 1);
        y_j = radar_positions(j, 2);
        for i = 1:M
            % Range model
            r_model = sqrt((x_j - x_tar).^2 + (y_j - y_tar).^2);
            r_model = max(r_model, 1); % Avoid division by zero for stability

            % Doppler shift model
            relative_position = [x_j - x_tar, y_j - y_tar];
            norm_rel_position = sqrt((x_j - x_tar).^2 + (y_j - y_tar).^2);
            norm_rel_position = max(norm_rel_position, 1);  % Avoid division by zero
            f_d_model = (v_x * (x_j - x_tar) + (v_y * (y_j - y_tar))) / (norm_rel_position * lambda);

            % Extracting current measurements
            r_ij = range_with_error(i,j);
            f_d_ij = doppler_with_error(i,j);

            % Extract the variances from Sigma_big
            sigma_r2 = Sigma_big((2*(idx-1)) + 1, (2*(idx-1))+ 1); % Variance of range
            sigma_fd2 = Sigma_big(2*idx, 2*idx); % Variance of Doppler

            % Accumulate the negative log likelihood

            log_likelihood = log_likelihood +  (1/2*(sigma_fd2*sigma_r2)) * ((f_d_ij - f_d_model).^2 * (sigma_r2) + (r_ij - r_model).^2 * (sigma_fd2));
%             log_likelihood = log_likelihood + (1/2*(sigma_fd2*sigma_r2))* ((f_d_ij -f_d_model).^2 * (sigma_r2) + (r_ij -r_model).^2 * (sigma_fd2));
            idx = idx + 1; % Update index for accessing Sigma_big
        end
    end 
    log_likelihood = log_likelihood + ((1/2)*log(2*pi));
end
% 

%%

% Wrapper function to track negative log-likelihood values
function log_likelihood = log_likelihood_wrapper(params, range_with_error, doppler_with_error, radar_positions, numNodes, M, lambda, Sigma_big, LaplacianMatrix)
    global log_likelihood_values

    log_likelihood = logLikelihood(params, range_with_error, doppler_with_error, radar_positions, numNodes, M, lambda, Sigma_big, LaplacianMatrix);

    % Store the negative log-likelihood value
    log_likelihood_values = [log_likelihood_values; log_likelihood];
end

%% Output function to track the optimization process
function stop = outfun(x, optimValues, state)
    global estimated_param_values

    stop = false;

    switch state
        case 'iter'
            estimated_param_values = [estimated_param_values; x(:)'];
    end
end

%% LogLikelihood with Consensus

function ll_with_consensus = logLikelihoodWithConsensus(params, range_measurements, doppler_measurements, radar_positions, numNodes, M, lambda, Sigma_big, n, neighbors, Nu, initial_values, update_z, c_penalty)
    ll = logLikelihood(params, range_measurements, doppler_measurements, radar_positions, numNodes, M, lambda, Sigma_big);

    x_tar = params(1);
    y_tar = params(2);
    v_x = params(3);
    v_y = params(4);

    sum_L1 = 0;
    sum_L2 = 0;
    for j = neighbors{n}
        sum_L1 = sum_L1 + (Nu{n}(:,j))' * (params - update_z{n}(:,j));
        sum_L2 = sum_L2 + norm((c_penalty/2)' .* (params - update_z{n}(:,j))).^2;
%         sum_L2 = sum_L2 + norm((params - update_z{n}(:,j))).^2;
    end

%     ll_with_consensus = ll + sum_L1 + ((10^(10))/2) * sum_L2;
    ll_with_consensus = ll + sum_L1 + sum_L2;

end

