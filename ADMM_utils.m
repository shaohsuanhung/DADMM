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
                    % try
                    %     sigma_r2 = Sigma_big((2*(idx-1)) + 1, (2*(idx-1))+ 1); % Variance of range
                    %     sigma_fd2 = Sigma_big(2*idx, 2*idx); % Variance of Doppler
                    % catch
                    %     % fprintf(Sigma_big);
                    %     sigma_r2 = Sigma_big(1,1);
                    %     sigma_fd2 = Sigma_big(2,2);
                    % end
                    sigma_r2 = Sigma_big(1,1);
                    sigma_fd2 = Sigma_big(2,2);
        
                    % Accumulate the negative log likelihood - Eq.4.1
                    log_likelihood = log_likelihood +  (1/2*(sigma_fd2*sigma_r2)) * ((f_d_ij - f_d_model).^2 * (sigma_r2) + (r_ij - r_model).^2 * (sigma_fd2));
        %             log_likelihood = log_likelihood + (1/2*(sigma_fd2*sigma_r2))* ((f_d_ij -f_d_model).^2 * (sigma_r2) + (r_ij -r_model).^2 * (sigma_fd2));
                    idx = idx + 1; % Update index for accessing Sigma_big
                end
            end 
            log_likelihood = log_likelihood + ((1/2)*log(2*pi));% What about 1/2 ln{sigma)
        end

        %% Log Likelihood Function for MAP
        % function posterior = MAP(params, range_with_error, doppler_with_error, mu_r, mu_d, sigma_r, sigma_d,radar_positions, numNodes, M, lambda, Sigma_big)
        function posterior = MAP(params, range_with_error, doppler_with_error, mu_state, sigma_state,radar_positions, numNodes, M, lambda, Sigma_big)
            %%% The MAP, prior, ll should write in a loop that based on how
            %%% many noded you input, then calculate to give flexibility. 
            % Calculate prior
            ut = ADMM_utils;
            % prior = ut.prior_distribution(params, mu_r, mu_d, sigma_r, sigma_d, radar_positions, numNodes, M, lambda);
            prior = ut.state_prior_distribution(params, mu_state, sigma_state,numNodes);
            % prior = ut.prior_distribution_initial_guess(params, mu_r, mu_d, sigma_r, sigma_d, radar_positions, numNodes, M, lambda);
            % prior = ut.prior_distribution_initial_values(params, mu_r, mu_d, sigma_r, sigma_d, radar_positions, numNodes, M, lambda);

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
                    % try
                    %     sigma_r2 = Sigma_big((2*(idx-1)) + 1, (2*(idx-1))+ 1); % Variance of range
                    %     sigma_fd2 = Sigma_big(2*idx, 2*idx); % Variance of Doppler
                    % catch
                    %     sigma_r2 = Sigma_big(1,1);
                    %     sigma_fd2 = Sigma_big(2,2);
                    % end 
                    sigma_r2 = Sigma_big(1,1);
                    sigma_fd2 = Sigma_big(2,2);
        
                    % Accumulate the negative log likelihood - Eq.4.1
                    log_likelihood = log_likelihood +  (1/2*(sigma_fd2*sigma_r2)) * ((f_d_ij - f_d_model).^2 * (sigma_r2) + (r_ij - r_model).^2 * (sigma_fd2));
                    % log_likelihood = log_likelihood + (1/2*(sigma_fd2*sigma_r2))* ((f_d_ij -f_d_model).^2 * (sigma_r2) + (r_ij -r_model).^2 * (sigma_fd2));
                    idx = idx + 1; % Update index for accessing Sigma_big
                end
            end 
            posterior = log_likelihood + prior + ((1/2)*log(2*pi));% What about 1/2 ln{sigma)
        end
        function prior = prior_distribution(params, mu_r, mu_d,sigma_r,sigma_fd, radar_positions, numNodes, M, lambda)
            % Shape of mu_r, mu_d, sigma_r, sigma_d should be [numNodes x 1]
            x_tar = params(1);
            y_tar = params(2);
            v_x = params(3);
            v_y = params(4);
            prior = 0;
        
            for j = 1:numNodes
                x_j = radar_positions(j, 1);
                y_j = radar_positions(j, 2);
                % Theis should be chnage 
                mu_rj      =  mu_r(j);
                mu_fdj     =  mu_d(j);
                sigma_fd2 = sigma_fd(j);
                sigma_r2  = sigma_r(j);
                %%%%%%%%%%%%%%%%%%%%%%
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
                    prior = prior +  ((1/2)*(sigma_fd2*sigma_r2)) * (( mu_fdj- f_d_model).^2 * (sigma_r2) + (mu_rj - r_model).^2 * (sigma_fd2));
                    % prior = prior +  (1/2*(sigma_fd2*sigma_r2)) * (( mu_fdj).^2 * (sigma_r2) + (mu_rj).^2 * (sigma_fd2));
                    
                end
            end 
        end
        function prior = state_prior_distribution(params, mu,sigma, numNodes)
            % Shape of mu_r, mu_d, sigma_r, sigma_d should be [numNodes x 1]
            prior = 0;
            dim = length(params);
            for j = 1:numNodes
                    prior= prior + (1/sqrt((2*pi)^dim*det((sigma{j}))))* ((params - mu{j})' * inv((sigma{j})) * (params - mu{j}));
            end 
        end
        
        function prior = prior_distribution_initial_values(params, mu_r, mu_d,sigma_r,sigma_d, radar_positions, numNodes, M, lambda)
            x_tar = params(1);
            y_tar = params(2);
            v_x = params(3);
            v_y = params(4);
            prior = 0;
        
            for j = 1:numNodes
                x_j = radar_positions(j, 1);
                y_j = radar_positions(j, 2);
                % Theis should be chnage 
                mu_rj      =  mu_r(j);
                mu_fdj     =  mu_d(j);
                sigma_fd2 = sigma_r(j);
                sigma_r2  = sigma_d(j);
                %%%%%%%%%%%%%%%%%%%%%%
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
                    prior = prior +  (1/2*(sigma_fd2*sigma_r2)) * (( 1414- f_d_model).^2 * (sigma_r2) + (20 - r_model).^2 * (sigma_fd2));
                    % prior = prior +  (1/2*(sigma_fd2*sigma_r2)) * (( mu_fdj).^2 * (sigma_r2) + (mu_rj).^2 * (sigma_fd2));
                    
                end
            end 
        end


        function prior = prior_distribution_initial_guess(params, mu_r, mu_d,sigma_r,sigma_d, radar_positions, numNodes, M, lambda)
            x_tar = params(1);
            y_tar = params(2);
            v_x = params(3);
            v_y = params(4);
            prior = 0;
            
            mu_x = 1000;
            mu_y = 1000;
            mu_vx =-14;
            mu_vy = 10;
            sigma_x = 3.4196e-04;
            sigma_y = 3.4196e-04;
            sigma_vx = 1e+07;
            sigma_vy = 4.6381e+04;
            for j = 1:numNodes
                prior = prior + (1/2)*((x_tar-mu_x)^2/sigma_x+ (y_tar-mu_y)^2/sigma_y+ (v_x-mu_vx)^2/sigma_vx+ (v_y-mu_vy)^2/sigma_vy);
            end 
        end
        %% Output function to track the optimization processs
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
        % function map_with_consensus = posteriorWithConsensus(params, range_measurements, doppler_measurements, prev_r, ...
        %                                                     prev_v, sigma_r, sigma_v, radar_positions, numNodes, ...
        %                                                      M, lambda, Sigma_big, n, neighbors, Nu, initial_values, update_z, c_penalty)
        function map_with_consensus = posteriorWithConsensus(params, range_measurements, doppler_measurements, ...
                                                            prior_mean, prior_sigma, radar_positions, numNodes, ...
                                                             M, lambda, Sigma_big, n, neighbors, Nu, initial_values, update_z, c_penalty)
            ut = ADMM_utils;
            % posterior = ut.MAP(params, range_measurements, doppler_measurements, prev_r, prev_v, sigma_r, sigma_v,radar_positions, numNodes, M, lambda, Sigma_big);
            posterior = ut.MAP(params, range_measurements, doppler_measurements, prior_mean, prior_sigma, radar_positions, numNodes, M, lambda, Sigma_big);
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
            %TODO: To make it to handle the multiple communcation radius cases. 
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
            % Range Measurement are stored in "relation position" way. 
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
        function [range_true, doppler_true, measurements_true] = gt_data_generation_tracking(r_true, d_true, mea_true, target, network_topo, env, M, num_target)
            % Matrices for range and Doppler true data (This is a matrix of M x N) 
            % Range Measurement are stored in "relation position" way. 
            % Output:
            % r_true : [num_target x numNodes x M]
            % d_true : [num_target x numNodes x M]
            % mea_true : [num_target x numNodes x 2M]
            % Input:
            % target.target_position : [num_target x M x 2]


            range_true = zeros(size(r_true));
            doppler_true = zeros(size(d_true));
            measurements_true = zeros(size(mea_true));
    
            % % Calculate range and Doppler true measurements
            % for i = 1:num_target
            %     for t = 1:M % Target at CPI*mea moment. 
            %         for r = 1:network_topo.numNodes
            %             fprintf("Progress: %g/%g \n",t,M);
            %             % Calculate range data
            %             range_true(t, r) = norm(reshape(target.target_position(i,t, :),size(network_topo.radar_pos(r,:))) - network_topo.radar_pos(r, :));
        
            %             % Calculate Doppler shift data
            %             relative_position = network_topo.radar_pos(r, :) - reshape(target.target_position(i,t, :),size(network_topo.radar_pos(r,:))); % [x,y]
            %             doppler_true(t, r) = dot([target.speed * target.direction], relative_position) / (norm(relative_position) * env.lambda);
        
            %             % Store the true measurements: range followed by Doppler
            %             measurements_true(i,2 * t - 1, r) = range_true(t, r);  % Odd index for range
            %             measurements_true(i,2 * t, r) = doppler_true(t, r);    % Even index for Doppler
            %         end
            %     end
            % end
            % rel_pos_t = reshape(network_topo.radar_pos,[1,network_topo.numNodes,1,2]) - reshape(target.target_position,[num_target,1,M,2]);
            % range_true = vecnorm(rel_pos_t,2,4); % [num_target x numNodes x M]
            radar_pos_expand =  repmat(reshape(network_topo.radar_pos,[num_target,size(network_topo.radar_pos,1),1,size(network_topo.radar_pos,2)]),[1,num_target,M,1]);
            target_pos_expand = repmat(reshape(target.target_position,[num_target,1,size(target.target_position,2),size(target.target_position,3)]),[1,network_topo.numNodes,1,1]);
            relative_position = radar_pos_expand - target_pos_expand; % [num_target x numNodes x M x 2]
            %TODO: Keep work on this!
            range_true = vecnorm(relative_position,2,4); % [num_target x numNodes x M]
            % doppler_true = dot([target.speed * target.direction], relative_position,4) / dot(range_true * env.lambda,3); % [num_target x numNodes x M]
            doppler_true = reshape(reshape(relative_position,[],2)*[target.speed * target.direction]',[num_target,network_topo.numNodes,M])./(range_true.* env.lambda);
            measurements_true(:,:, 1:2:end) = range_true; % Odd index for range
            measurements_true(:,:, 2:2:end) = doppler_true; % Even index for Doppler

        end
        function [range_with_error, doppler_with_error, measurements_all_with_error] = add_measurement_noise(range_true, doppler_true, M, NUM_TAR,numNodes, env)
            % Kron function is too expensive when total_measurements is large
            % Use mvnrnd to generate long noise vecotr directly
            % Add noise to the true measurements
            mu_r_d = [0 0]; % Mean for range and Doppler noise
            e_n = mvnrnd(mu_r_d, env.Sigma, NUM_TAR * M * numNodes); % Generate noise for all measurements at once
            % Reshape to [NUM_CPI_PER_MEA*TRACK_TIME , numNodes]
            range_noise_all   = reshape(e_n(:,1), NUM_TAR, numNodes, M);
            doppler_noise_all = reshape(e_n(:,2), NUM_TAR, numNodes, M);
            range_with_error = range_true + range_noise_all;
            doppler_with_error = doppler_true + doppler_noise_all;
            measurements_all_with_error = zeros(NUM_TAR,numNodes, M*2);
            measurements_all_with_error(:,:, 1:2:end) = range_with_error; % Odd index for range
            measurements_all_with_error(:,:, 2:2:end) = doppler_with_error; % Even index for Doppler
        end
        
        function neighbors = get_neighbors(adj_mtx, numNodes)
            % Initialize neighbors cell array
            neighbors = cell(numNodes, 1);
        
            % Fill the neighbors for each node
            for i = 1:numNodes
                neighbors{i} = find(adj_mtx(i, :) > 0); % Find indices of non-zero elements in row i
            end
        end

        % function [range_with_error_cell,doppler_with_error_cell,numNodes_cell,...
        %           radar_positions_cell,Sigma_big_1_cell,Sigma_big_2_cell,...
        %           mu_r_cell,mu_d_cell,sigma_r_cell,sigma_d_cell] = pharse_measurements(laplacian_matrix, ...
        % range_with_error, doppler_with_error, mu_r, mu_d, sigma_r, sigma_d,Sigma_big,...
        % range_with_error_cell, doppler_with_error_cell, ...
        % mu_r_cell, mu_d_cell, sigma_r_cell, sigma_d_cell, network_topo, M)
        function [range_with_error_cell,doppler_with_error_cell,numNodes_cell,...
                  radar_positions_cell,Sigma_big_1_cell,Sigma_big_2_cell, ...
                  state_mu_cell, state_cov_cell] = pharse_measurements(laplacian_matrix, ...
                  range_with_error, doppler_with_error,Sigma_big, ...
                  range_with_error_cell, doppler_with_error_cell, ...
                  state_mu, state_cov, network_topo, M)
            for n = 1: network_topo.numNodes
                current_neighbors = find(laplacian_matrix(n, :) ~= 0);
                k = 0;
                Sigma_big_1 = [];
                Sigma_big_2 = [];
                range_with_error_1 =[];
                doppler_with_error_1 = [];
                radar_positions_1 = [];
                mu_state_neighbor = {};
                sigma_state_neighbor = {};
                for j = current_neighbors
                    k = k+1;
                    % 
                    range_with_error_1(:,k) = range_with_error(:,j);
                    doppler_with_error_1(:,k) = doppler_with_error(:,j);
                    radar_positions_1(k,:) = network_topo.radar_pos(j,:);
                    numNodes_1 = length(current_neighbors);
                    % Get prior parameters from neighbors
                    mu_state_neighbor{k} = state_mu{j};
                    sigma_state_neighbor{k} = state_cov{j};
                    % mu_r_neighbor(:,k) = mu_r(k);
                    % mu_d_neighbor(:,k) = mu_d(k);
                    % sigma_r_neighbor(:,k) = sigma_r(k);
                    % sigma_d_neighbor(:,k) = sigma_d(k);
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
                    state_mu_cell{n} = mu_state_neighbor;
                    state_cov_cell{n} = sigma_state_neighbor;
                    % mu_r_cell{n}        = mu_r(j);
                    % mu_d_cell{n}        = mu_d(j);
                    % sigma_r_cell{n}     = sigma_r(j);
                    % sigma_d_cell{n}     = sigma_d(j);
                    % Prior
                    % mu_d_cell{n} = mu_r_neighbor;
                    % mu_r_cell{n} = mu_d_neighbor;
                    % sigma_r_cell{n} = sigma_r_neighbor;
                    % sigma_d_cell{n} = sigma_d_neighbor; 
                end
            end
        end 
        
        function [range_with_error_cell,doppler_with_error_cell,numNodes_cell,radar_positions_cell] = pharse_measurements_tracking(laplacian_matrix, ...
        range_with_error, doppler_with_error,range_with_error_cell, doppler_with_error_cell,num_tar, network_topo)
        % Remove mu_r, mu_d, sigma_r, sigma_dfrom input
            for i = 1: num_tar
                for n = 1: network_topo.numNodes
                    current_neighbors = find(laplacian_matrix(n, :) ~= 0);
                    k = 0;
                    % Sigma_big_1 = [];
                    % Sigma_big_2 = [];
                    range_with_error_1 =[];
                    doppler_with_error_1 = [];
                    radar_positions_1 = [];
                    for j = current_neighbors
                        k = k+1;
                        % 
                        range_with_error_1(:,k) = range_with_error(i,j,:);
                        doppler_with_error_1(:,k) = doppler_with_error(i,j,:);
                        radar_positions_1(k,:) = network_topo.radar_pos(j,:);
                        numNodes_1 = length(current_neighbors);
                        % mu_r_neighbor(:,k) = mu_r(k);
                        % mu_d_neighbor(:,k) = mu_d(k);
                        % sigma_r_neighbor(:,k) = sigma_r(k);
                        % sigma_d_neighbor(:,k) = sigma_d(k);
                        % for i =1:M
                        %     base_idx = 2 * (M * (j - 1) + (i - 1)) + 1;
                        %     % sigma_r2 = Sigma_big(base_idx, base_idx);
                        %     idx = mod(base_idx,2)+1;
                        %     Sigma_big_1(((k-1)*M + (i-1))*2 + 1) = Sigma(idx,idx); 
                        %     % sigma_fd2 = Sigma_big(base_idx + 1, base_idx + 1);
                        %     Sigma_big_1(((k-1)*M + (i-1))*2 + 2) = Sigma(idx,idx);
                        % end
                        % Sigma_big_2 = diag(Sigma_big_1);
                        % Store the values in cell arrays
                        range_with_error_cell{n} = range_with_error_1;
                        doppler_with_error_cell{n} = doppler_with_error_1;
                        numNodes_cell{n} = numNodes_1;
                        radar_positions_cell{n} = radar_positions_1;
                        % Sigma_big_1_cell{n} = Sigma_big_1;
                        % Sigma_big_2_cell{n} = Sigma_big_2;
                        % mu_r_cell{n}        = mu_r(j);
                        % mu_d_cell{n}        = mu_d(j);
                        % sigma_r_cell{n}     = sigma_r(j);
                        % sigma_d_cell{n}     = sigma_d(j);
                        % Prior
                        % mu_d_cell{n} = mu_r_neighbor;
                        % mu_r_cell{n} = mu_d_neighbor;
                        % sigma_r_cell{n} = sigma_r_neighbor;
                        % sigma_d_cell{n} = sigma_d_neighbor; 
                    end
                end
            end
        end 
        function write_exp_log(write_folder_path, folder_name, data_config)
            %{ 
            Build exp folder, and write log into the corresponding
            With this, we are likeliy reproduce the results
            Write: (1) Data config, (2) Algorithm config (rou, intial value...)
            %}
            checkdate = datestr(datetime);
            file_path = string(write_folder_path+"/"+folder_name+"_"+checkdate);
            mkdir(file_path);
            fid = fopen(file_path'+'/'+'data_config.json','w');
            encodedJSON = jsonencode(data_config,PrettyPrint=true);
            fwrite(fid,encodedJSON); 
            fclose(fid);
        end 

        function decode_data_struct = read_exp_log(file_path)
            %{
            Read the log file, and return the struct
            Input: file path
            Output: struct, reacod hyparapameters, constant setting, and algorith setting for that exp.
            %}
            fid = fopen(file_path);
            raw = fread(fid,inf);
            str = char(raw');
            decode_data_struct = jsondecode(str);
            fclose(fid);
        end

        function ADMM = ADMM_reset(ADMM,NUM_TAR,network_topo)
            ADMM.converged = false;
            ADMM.Nu = cell(NUM_TAR, network_topo.numNodes);
            ADMM.Nu_prev = cell(NUM_TAR, network_topo.numNodes);
            ADMM.update_z = cell(NUM_TAR, network_topo.numNodes);
            ADMM.update_z_prev = cell(NUM_TAR, network_topo.numNodes);
            ADMM.primal_residual_all =[];
            ADMM.primal_residual_by_para = cell(NUM_TAR,network_topo.numNodes);
            ADMM.dual_residual_all = [];
            ADMM.dual_residual_by_para = cell(NUM_TAR,network_topo.numNodes);
            ADMM.all_estimations_every_iter = [];
            ADMM.RANGE_Xs = [];
            ADMM.RANGE_Ys = [];
            ADMM.DOPPLER_Xs = [];
            ADMM.converg_r = false;
            ADMM.converg_d = false;
            for n = 1:network_topo.numNodes
                ADMM.Nu{n} = zeros(4, network_topo.numNodes);
                ADMM.Nu_prev{n} = zeros(4, network_topo.numNodes);
                ADMM.update_z{n} = zeros(4, network_topo.numNodes);
                ADMM.update_z_prev{n} = zeros(4, network_topo.numNodes);
            end

        end
    end
end