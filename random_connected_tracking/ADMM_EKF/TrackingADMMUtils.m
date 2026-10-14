classdef TrackingADMMUtils
    % EKF helpers with a fixed first-order measurement model per update.
    methods (Static)

        function [log_likelihood, gradient] = logLikelihood(params, range_measurements, doppler_measurements, model, Sigma, varargin)
            % Data and model are unwhitened. Optional L transforms the full
            % residual and its Jacobian once; Sigma is then Sigma_filter.
            % State-independent Gaussian normalization constants are omitted.
            delta = params(:) - model.anchor;
            log_likelihood = 0;
            gradient = zeros(4,1);
            if ~isempty(varargin)
                L = varargin{1};
            else
                L = [];
            end
            for j = 1:size(model.value,2)
                H = model.H(:,:,j);
                prediction = model.value(:,j) + H * delta;
                residual = [range_measurements(:,j).'; doppler_measurements(:,j).'] - prediction;
                if ~isempty(L)
                    residual = L * residual;
                    H = L * H;
                end
                weighted_residual = Sigma \ residual;
                log_likelihood = log_likelihood + 0.5 * sum(residual .* weighted_residual, 'all');
                gradient = gradient - H.' * sum(weighted_residual,2);
            end
        end

        function [posterior, gradient] = MAP(params, range_measurements, doppler_measurements, mu_state, sigma_state, model, Sigma, varargin)
            numPriors = numel(mu_state);
            assert(numPriors == numel(sigma_state), ...
                'mu_state and sigma_state must contain the same number of priors.');
            [prior, prior_gradient] = TrackingADMMUtils.state_prior_distribution( ...
                params, mu_state, sigma_state, numPriors);
            [likelihood, likelihood_gradient] = TrackingADMMUtils.logLikelihood( ...
                params, range_measurements, doppler_measurements, model, Sigma, varargin{:});
            posterior = likelihood + prior;
            gradient = likelihood_gradient + prior_gradient;
        end

        function [prior, gradient] = state_prior_distribution(params, mu, sigma, numNodes)
            prior = 0;
            gradient = zeros(4,1);
            for j = 1:numNodes
                delta = params(:) - mu{j}(:);
                weighted_delta = sigma{j} \ delta;
                prior = prior + 0.5 * delta.' * weighted_delta;
                gradient = gradient + weighted_delta;
            end
            % Do not average independent prior terms here. Consensus prior
            % partitioning is handled explicitly by the caller.
        end

        function [value, gradient] = logLikelihoodWithConsensus(params, range_measurements, doppler_measurements, ...
                model, Sigma, n, neighbors, Nu, update_z, c_penalty, varargin)
            [likelihood, likelihood_gradient] = TrackingADMMUtils.logLikelihood( ...
                params, range_measurements, doppler_measurements, model, Sigma, varargin{:});
            [consensus, consensus_gradient] = TrackingADMMUtils.consensus_cost( ...
                params, n, neighbors, Nu, update_z, c_penalty);
            value = likelihood + consensus;
            gradient = likelihood_gradient + consensus_gradient;
        end

        function [value, gradient] = posteriorWithConsensus(params, range_measurements, doppler_measurements, ...
                prior_mean, prior_sigma, model, Sigma, n, neighbors, Nu, update_z, c_penalty, varargin)
            [posterior, posterior_gradient] = TrackingADMMUtils.MAP( ...
                params, range_measurements, doppler_measurements, prior_mean, prior_sigma, model, Sigma, varargin{:});
            [consensus, consensus_gradient] = TrackingADMMUtils.consensus_cost( ...
                params, n, neighbors, Nu, update_z, c_penalty);
            value = posterior + consensus;
            gradient = posterior_gradient + consensus_gradient;
        end

        function [value, gradient] = consensus_cost(params, n, neighbors, Nu, update_z, c_penalty)
            value = 0;
            gradient = zeros(4,1);
            for j = neighbors{n}
                delta = params(:) - update_z{n}(:,j);
                value = value + Nu{n}(:,j).' * delta + ...
                    0.5 * sum(c_penalty(:) .* delta.^2);
                gradient = gradient + Nu{n}(:,j) + c_penalty(:) .* delta;
            end
        end
    

        function [range_true, doppler_true, measurements_true] = gt_data_generation_tracking_ekf(r_true, d_true, mea_true, target, network_topo, env, M, num_target)
            % Matrices for range and Doppler true data (This is a matrix of M x N) 
            % Range Measurement are stored in "relation position" way. 
            % Output: M is the time duration 
            % r_true : [num_target x numNodes x M]
            % d_true : [num_target x numNodes x M]
            % mea_true : [num_target x numNodes x 2M]
            % Input:
            % target.target_position : [num_target x M x 2]
            range_true = zeros(size(r_true));
            doppler_true = zeros(size(d_true));
            measurements_true = zeros(size(mea_true));
            
            % Radar_pos [# node, 2 (x,y)] (1) reshape from expand to 4 [num target(1), time(1), num node, 2]
            % (2) repmat to [num target (remate here), time(remat here), num node, 2]
            radar_pos_expand =  repmat(reshape(network_topo.radar_pos,[1,size(network_topo.radar_pos,1),1,size(network_topo.radar_pos,2)]),[num_target,1,M,1]);
            % target_pos [Num of target, time duration, 2] (1) reshape to [num target,1, time, num node (1),2]
            % (2) repmat to [num target, time (remat here), num doe (remat), 2]
            target_pos_expand = repmat(reshape(target.target_position,[num_target,1,size(target.target_position,2),size(target.target_position,3)]),[1,network_topo.numNodes,1,1]);
            relative_position = -(radar_pos_expand - target_pos_expand); % [num_target x numNodes x M x 2]
            range_true = vecnorm(relative_position,2,4); % [num_target x numNodes x M]
            % Expand [target x time x velocity-component] to match
            % [target x node x time x component]. Avoid squeeze so this is
            % valid for M=1 and for multiple targets.
            target_velocity = reshape(target.target_state(:,:,3:4), ...
                [num_target, 1, M, 2]);
            velocity_expand = repmat(target_velocity, ...
                [1, network_topo.numNodes, 1, 1]);
            radial_numerator = sum(relative_position .* velocity_expand, 4);
            doppler_true = 2 * radial_numerator ./ ...
                (range_true .* env.lambda);
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

        function ADMM = Initialized_ADMM(network_topo)
            options_Dctral = optimoptions('fmincon', ...
                'Display', 'off', ...
                'SpecifyObjectiveGradient', true, ...
                'ScaleProblem', true, ...
                'OptimalityTolerance', 1e-6, ...
                'FunctionTolerance', 1e-8, ...
                'StepTolerance', 1e-8, ...
                'MaxIterations', 300, ...
                'MaxFunctionEvaluations', 5000);
            ADMM = struct();
            ADMM.solver = options_Dctral;
            ADMM.converged= false;
            ADMM.max_iter = 500; % Maximumconverged ADMM iterations
            ADMM.c_penalty = [100,100,15,15];
            ADMM.lb = [-inf,-inf,-inf,-inf];
            ADMM.ub = [inf,inf, inf, inf];
            ADMM.initial_values = repmat([1000, 1000, -14, 14]', 1,network_topo.numNodes);
            % Prior information 
            ADMM.prev_mean = cell(1,network_topo.numNodes);
            ADMM.prev_cov = cell(1,network_topo.numNodes);
            %
            ADMM.Nu = cell(1, network_topo.numNodes);
            ADMM.Nu_prev = cell(1, network_topo.numNodes);
            ADMM.update_z = cell(1, network_topo.numNodes);
            ADMM.update_z_prev = cell(1, network_topo.numNodes);
            ADMM.primal_residual_all =[];
            ADMM.primal_residual_by_para = cell(1,network_topo.numNodes);
            ADMM.dual_residual_all = [];
            ADMM.dual_residual_by_para = cell(1,network_topo.numNodes);
            ADMM.all_estimations_every_iter = [];
            ADMM.final_tracking_estimation = cell(1, 5);
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
        end

    end
end
