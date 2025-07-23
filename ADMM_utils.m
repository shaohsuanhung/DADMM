classdef ADMM_utils
    methods (Static)
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

                % Range model
                r_model = sqrt((x_j - x_tar).^2 + (y_j - y_tar).^2);
                r_model = max(r_model, 1); % Avoid division by zero for stability
                
                % Doppler shift model
                relative_position = [x_j - x_tar, y_j - y_tar];
                norm_rel_position = sqrt((x_j - x_tar).^2 + (y_j - y_tar).^2);
                norm_rel_position = max(norm_rel_position, 1);  % Avoid division by zero
                f_d_model = (v_x * (x_j - x_tar) + (v_y * (y_j - y_tar))) / (norm_rel_position * lambda);

                for i = 1:M                    
                    % Extracting current measurements
                    r_ij = range_with_error(i,j); % Per pulse for that sensor 
                    f_d_ij = doppler_with_error(i,j);
        
                    % Extract the variances from Sigma_big
                    sigma_r2 = Sigma_big((2*(idx-1)) + 1, (2*(idx-1))+ 1); % Variance of range
                    sigma_fd2 = Sigma_big(2*idx, 2*idx); % Variance of Doppler
        
                    % Accumulate the negative log likelihood - Eq.4.1
                    log_likelihood = log_likelihood +  (1/2*(sigma_fd2*sigma_r2)) * ((f_d_ij - f_d_model).^2 * (sigma_r2) + (r_ij - r_model).^2 * (sigma_fd2));
        %             log_likelihood = log_likelihood + (1/2*(sigma_fd2*sigma_r2))* ((f_d_ij -f_d_model).^2 * (sigma_r2) + (r_ij -r_model).^2 * (sigma_fd2));
                    idx = idx + 1; % Update index for accessing Sigma_big
                end
            end 
            log_likelihood = log_likelihood + ((1/2)*log(2*pi));% What about 1/2 ln{sigma)
        end

        %% Log Likelihood Function for MAP
        function posterior = MAP(params, range_with_error, doppler_with_error, mu_r, mu_d, sigma_r, sigma_d,radar_positions, numNodes, M, lambda, Sigma_big)
        %%% The MAP, prior, ll should write in a loop that based on how
        %%% many noded you input, then calculate to give flexibility. 

            % Calculate prior
            ut = ADMM_utils;
            prior = ut.prior_distribution(params, mu_r, mu_d, sigma_r, sigma_d, radar_positions, numNodes, M, lambda);

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
                    r_ij = range_with_error(i,j); % Per pulse for that sensor 
                    f_d_ij = doppler_with_error(i,j);
        
                    % Extract the variances from Sigma_big
                    sigma_r2 = Sigma_big((2*(idx-1)) + 1, (2*(idx-1))+ 1); % Variance of range
                    sigma_fd2 = Sigma_big(2*idx, 2*idx); % Variance of Doppler
        
                    % Accumulate the negative log likelihood - Eq.4.1
                    log_likelihood = log_likelihood +  (1/2*(sigma_fd2*sigma_r2)) * ((f_d_ij - f_d_model).^2 * (sigma_r2) + (r_ij - r_model).^2 * (sigma_fd2));
        %             log_likelihood = log_likelihood + (1/2*(sigma_fd2*sigma_r2))* ((f_d_ij -f_d_model).^2 * (sigma_r2) + (r_ij -r_model).^2 * (sigma_fd2));
                    idx = idx + 1; % Update index for accessing Sigma_big
                end
            end 
            posterior = log_likelihood + prior + ((1/2)*log(2*pi));% What about 1/2 ln{sigma)


        end

        %% Log Likelihood Function for MLE
        function prior = prior_distribution(params, mu_r, mu_d,sigma_r,sigma_d, radar_positions, numNodes, M, lambda)
            
            x_tar = params(1);
            y_tar = params(2);
            v_x = params(3);
            v_y = params(4);
            prior = 0;
        
            for j = 1:numNodes
                x_j = radar_positions(j, 1);
                y_j = radar_positions(j, 2);
                mu_rj      =  mu_r(j);
                mu_fdj     =  mu_d(j);
                sigma_fd2 = sigma_r(j);
                sigma_r2  = sigma_d(j);
                for i = 1:M
                    % Range model
                    r_model = sqrt((x_j - x_tar).^2 + (y_j - y_tar).^2);
                    r_model = max(r_model, 1); % Avoid division by zero for stability
                    
                    % Doppler shift model
                    relative_position = [x_j - x_tar, y_j - y_tar];
                    norm_rel_position = sqrt((x_j - x_tar).^2 + (y_j - y_tar).^2);
                    norm_rel_position = max(norm_rel_position, 1);  % Avoid division by zero
                    f_d_model = (v_x * (x_j - x_tar) + (v_y * (y_j - y_tar))) / (norm_rel_position * lambda);

                    % Accumulate the negative log likelihood - Eq.4.1
                    prior = prior +  (1/2*(sigma_fd2*sigma_r2)) * (( mu_fdj- f_d_model).^2 * (sigma_r2) + (mu_rj - r_model).^2 * (sigma_fd2));
                end
            end 
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
            ut = ADMM_utils;
            ll = ut.logLikelihood(params, range_measurements, doppler_measurements, radar_positions, numNodes, M, lambda, Sigma_big);
          
            x_tar = params(1);
            y_tar = params(2);
            v_x = params(3);
            v_y = params(4);
        
            sum_L1 = 0;
            sum_L2 = 0;
            for j = neighbors{n}
                % eq 4.13
                sum_L1 = sum_L1 + (Nu{n}(:,j))' * (params - update_z{n}(:,j));
                sum_L2 = sum_L2 + norm((c_penalty/2)' .* (params - update_z{n}(:,j))).^2;
        %         sum_L2 = sum_L2 + norm((params - update_z{n}(:,j))).^2;
            end
        
        %     ll_with_consensus = ll + sum_L1 + ((10^(10))/2) * sum_L2;
            ll_with_consensus = ll + sum_L1 + sum_L2;
            
        end

        %% LogLikelihood with Consensus
        function map_with_consensus = posteriorWithConsensus(params, range_measurements, doppler_measurements, mu_r, ...
                                                             mu_d, sigma_r, sigma_d, radar_positions, numNodes, ...
                                                             M, lambda, Sigma_big, n, neighbors, Nu, initial_values, update_z, c_penalty)
            ut = ADMM_utils;
            posterior = ut.MAP(params, range_measurements, doppler_measurements, mu_r, mu_d, sigma_r, sigma_d,radar_positions, numNodes, M, lambda, Sigma_big);

            x_tar = params(1);
            y_tar = params(2);
            v_x = params(3);
            v_y = params(4);
        
            sum_L1 = 0;
            sum_L2 = 0;
            for j = neighbors{n}
                % eq 4.13
                sum_L1 = sum_L1 + (Nu{n}(:,j))' * (params - update_z{n}(:,j));
                sum_L2 = sum_L2 + norm((c_penalty/2)' .* (params - update_z{n}(:,j))).^2;
            end
            map_with_consensus = posterior + sum_L1 + sum_L2;
            
        end
    
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
    
        %% Calculate adjcent matrix
        function adj_mtx = calculate_adj_mtx(pos,communication_radius,numNodes)
        % Initialize the adjacency matrix
            adj_mtx = zeros(numNodes,numNodes);  
            % Calculate the adjacency matrix
             for i = 1:numNodes
                for j = i+1:numNodes % Since A is symmetric, compute only for j > i
                   if norm(pos(i,:) - pos(j,:)) <= communication_radius
                      adj_mtx(i,j) = 1;
                      adj_mtx(j,i) = 1; % Symmetric
                   end
               end
            end
        end 

        %% Incident matrx
        function inc_matrix = calculate_inc_mtx(adj_matrix,numNodes)
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
        end

        function weight_matrix = calculate_weight_mtx(degree_matrix, adj_matrix,numNodes)
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
        end 
        
        function [adj_mtx, deg_mtx, lap_mtx, inc_mtx, wei_mtx] = calculate_all_graph_matrix(radar_pos, communication_radius, numNodes)
            ut = ADMM_utils;
            adj_mtx = ut.calculate_adj_mtx(radar_pos,communication_radius,numNodes);
            deg_mtx = diag(sum(adj_mtx,2));
            lap_mtx = deg_mtx - adj_mtx;
            inc_mtx = ut.calculate_inc_mtx(adj_mtx,numNodes);
            wei_mtx = ut.calculate_weight_mtx(deg_mtx,adj_mtx,numNodes);
        end
        function [range_true, doppler_true, measurements_true] = gt_data_generation(r_true,d_true, mea_true,target,network_topo,env,M)
            % Matrices for range and Doppler true data (This is a matrix of M x N) 
            % Range Measurement are stored in relation position way. 
            range_true = zeros(size(r_true));
            doppler_true = zeros(size(d_true));
            measurements_true = zeros(size(mea_true));
    
            % Calculate range and Doppler true measurements
            for t = 1:M % Target at t moment. 
                for r = 1:network_topo.numNodes
                    % Calculate range data
                    range_true(t, r) = norm(target.target_position(t, :) - network_topo.radar_pos(r, :));
    
                    % Calculate Doppler shift data
                    relative_position = network_topo.radar_pos(r, :) - target.target_position(t, :); % [x,y]
                    doppler_true(t, r) = dot([target.speed * target.direction], relative_position) / (norm(relative_position) * env.lambda);
    
                    % Store the true measurements: range followed by Doppler
                    measurements_true(2 * t - 1, r) = range_true(t, r);  % Odd index for range
                    measurements_true(2 * t, r) = doppler_true(t, r);    % Even index for Doppler
                end
            end
        
        end
    end
end