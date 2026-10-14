classdef BatchADMMUtils
    % Batch MAP/MLE with a fixed first-order measurement model per batch.
    methods (Static)
        function model = linearize_measurements(anchor, radar_positions, lambda, whitener)
            % h(x) = [range; dot(v,p-g)/(lambda*range)].
            % Build once before ADMM; never linearize at its current iterate.
            if nargin < 4
                whitener = eye(2);
            end
            validateattributes(anchor, {'numeric'}, {'vector','numel',4,'finite'});
            validateattributes(radar_positions, {'numeric'}, {'2d','ncols',2,'finite'});
            validateattributes(lambda, {'numeric'}, {'scalar','positive','finite'});
            validateattributes(whitener, {'numeric'}, {'size',[2,2],'finite'});
            anchor = anchor(:);
            numNodes = size(radar_positions, 1);
            model.anchor = anchor;
            model.value = zeros(2, numNodes);
            model.H = zeros(2, 4, numNodes);
            for j = 1:numNodes
                displacement = anchor(1:2) - radar_positions(j,:).';
                range = norm(displacement);
                if range <= 1e-9
                    error('RadarTracking:SingularLinearization', ...
                        'Linearization point coincides with radar %d; choose another initial state.', j);
                end
                direction = displacement / range;
                velocity = anchor(3:4);
                radial_velocity = direction.' * velocity;
                value = [range; radial_velocity / lambda];
                doppler_position = (velocity - direction * radial_velocity) / (lambda * range);
                H = [direction.', zeros(1,2); doppler_position.', direction.' / lambda];
                % Transform the complete value and Jacobian together.
                model.value(:,j) = whitener * value;
                model.H(:,:,j) = whitener * H;
            end
        end

        function [log_likelihood, gradient] = logLikelihood(params, range_measurements, doppler_measurements, model, Sigma)
            % Gaussian negative log-likelihood, omitting state-independent constants.
            % model.value + H*(x-anchor) retains the affine intercept.
            delta = params(:) - model.anchor;
            log_likelihood = 0;
            gradient = zeros(4,1);
            for j = 1:size(model.value, 2)
                H = model.H(:,:,j);
                prediction = model.value(:,j) + H * delta;
                residual = [range_measurements(:,j).'; doppler_measurements(:,j).'] - prediction;
                weighted_residual = Sigma \ residual;
                log_likelihood = log_likelihood + 0.5 * sum(residual .* weighted_residual, 'all');
                gradient = gradient - H.' * sum(weighted_residual, 2);
            end
        end

        function [posterior, gradient] = MAP(params, range_measurements, doppler_measurements, mu_state, sigma_state, model, Sigma)
            [likelihood, likelihood_gradient] = BatchADMMUtils.logLikelihood( ...
                params, range_measurements, doppler_measurements, model, Sigma);
            [prior, prior_gradient] = BatchADMMUtils.state_prior_distribution( ...
                params(:), mu_state, sigma_state, numel(mu_state));
            posterior = likelihood + prior;
            gradient = likelihood_gradient + prior_gradient;
        end

        function [prior, gradient] = state_prior_distribution(params, mu, sigma, numNodes)
            % Retain the legacy prior coefficient; this is already quadratic.
            % It is not the standard Gaussian negative-log-prior scaling.
            prior = 0;
            gradient = zeros(4,1);
            dim = numel(params);
            for j = 1:numNodes
                delta = params(:) - mu{j}(:);
                weight = 1 / sqrt((2*pi)^dim * det(sigma{j}));
                weighted_delta = sigma{j} \ delta;
                prior = prior + weight * (delta.' * weighted_delta);
                gradient = gradient + 2 * weight * weighted_delta;
            end
            prior = prior / numNodes;
            gradient = gradient / numNodes;
        end

        function [value, gradient] = logLikelihoodWithConsensus(params, range_measurements, doppler_measurements, ...
                model, Sigma, n, neighbors, Nu, update_z, c_penalty)
            [likelihood, likelihood_gradient] = BatchADMMUtils.logLikelihood( ...
                params, range_measurements, doppler_measurements, model, Sigma);
            [consensus, consensus_gradient] = BatchADMMUtils.consensus_cost( ...
                params, n, neighbors, Nu, update_z, c_penalty);
            value = likelihood + consensus;
            gradient = likelihood_gradient + consensus_gradient;
        end

        function [value, gradient] = posteriorWithConsensus(params, range_measurements, doppler_measurements, ...
                prior_mean, prior_sigma, model, Sigma, n, neighbors, Nu, update_z, c_penalty)
            [posterior, posterior_gradient] = BatchADMMUtils.MAP( ...
                params, range_measurements, doppler_measurements, prior_mean, prior_sigma, model, Sigma);
            [consensus, consensus_gradient] = BatchADMMUtils.consensus_cost( ...
                params, n, neighbors, Nu, update_z, c_penalty);
            value = posterior + consensus;
            gradient = posterior_gradient + consensus_gradient;
        end

        function [value, gradient] = consensus_cost(params, n, neighbors, Nu, update_z, c_penalty)
            % Preserve the original augmented cost and update convention.
            % This coefficient is c.^2/4, not the standard ADMM c/2.
            value = 0;
            gradient = zeros(4,1);
            for j = neighbors{n}
                delta = params(:) - update_z{n}(:,j);
                value = value + Nu{n}(:,j).' * delta + sum((c_penalty(:)/2 .* delta).^2);
                gradient = gradient + Nu{n}(:,j) + (c_penalty(:).^2 / 2) .* delta;
            end
        end

        function crlb = calculateBCRLB(true_params, radar_positions, numNodes, M, lambda, Sigma,Q)
            DEBUG = false;
            Q_inv = inv(Q);
            FIM = BatchADMMUtils.calculateFIM(true_params, radar_positions, numNodes, M, lambda, Sigma, Q);
            crlb = inv(FIM + M*numNodes*Q_inv);
            if DEBUG
                disp('Bayesian CRLB:');
                disp(crlb);
            end
        end

        function FIM = calculateFIM(true_params, radar_positions, numNodes, M, lambda, Sigma,varargin)
            x_tar = true_params(1);
            y_tar = true_params(2);
            v_x = true_params(3);
            v_y = true_params(4);
            if numel(varargin) >= 1
                Q = varargin{1};   % Another covariance matrix for the state, which is used for PCRLB calculation
                Q_inv = inv(Q);
            end
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

                    dr_dx = (x_tar - x_j) / r_model;
                    dr_dy = (y_tar - y_j) / r_model;
                    dr_dvx = 0;
                    dr_dvy = 0;

                    df_dvx = relative_position(1) / (norm_rel_pos * lambda);
                    df_dvy = relative_position(2) / (norm_rel_pos * lambda);
                    df_dx = -dot([v_x, v_y], relative_position) * (x_tar - x_j) / (norm_rel_pos^3 * lambda);
                    df_dy = -dot([v_x, v_y], relative_position) * (y_tar - y_j) / (norm_rel_pos^3 * lambda);

                    J_i = [dr_dx, dr_dy, dr_dvx, dr_dvy; df_dx, df_dy, df_dvx, df_dvy];

                    if numel(varargin) == 1
                        pos_cov = J_i * Q * J_i';
                        Sigma_pos = Sigma_inv;
                    else
                        Sigma_pos = Sigma_inv;
                    end
                    FIM = FIM + J_i'* (Sigma_pos) * J_i;
                end
            end
        end

        function [range_true, doppler_true, measurements_true] = gt_data_generation_tracking(r_true, d_true, mea_true, target, network_topo, env, M, num_target)
            range_true = zeros(size(r_true));
            doppler_true = zeros(size(d_true));
            measurements_true = zeros(size(mea_true));
            
            radar_pos_expand =  repmat(reshape(network_topo.radar_pos,[1,size(network_topo.radar_pos,1),1,size(network_topo.radar_pos,2)]),[num_target,1,M,1]);
            target_pos_expand = repmat(reshape(target.target_position,[num_target,1,size(target.target_position,2),size(target.target_position,3)]),[1,network_topo.numNodes,1,1]);
            relative_position = -(radar_pos_expand - target_pos_expand); % [num_target x numNodes x M x 2]
            range_true = vecnorm(relative_position,2,4); % [num_target x numNodes x M]
            % Shared nonlinear truth supplies the velocity at every sample.
            % Preserve the legacy 1/lambda factor and target-minus-radar sign.
            target_velocity = reshape(target.target_state(:,:,3:4), [num_target,1,M,2]);
            doppler_true = sum(relative_position .* target_velocity, 4) ./ ...
                (range_true .* env.lambda);
            measurements_true(:,:, 1:2:end) = range_true; % Odd index for range
            measurements_true(:,:, 2:2:end) = doppler_true; % Even index for Doppler
        end

        function [range_with_error, doppler_with_error, measurements_all_with_error] = add_measurement_noise(range_true, doppler_true, M, NUM_TAR,numNodes, env)
            mu_r_d = [0 0]; % Mean for range and Doppler noise
            e_n = mvnrnd(mu_r_d, env.Sigma, NUM_TAR * M * numNodes); % Generate noise for all measurements at once
            range_noise_all   = reshape(e_n(:,1), NUM_TAR, numNodes, M);
            doppler_noise_all = reshape(e_n(:,2), NUM_TAR, numNodes, M);
            range_with_error = range_true + range_noise_all;
            doppler_with_error = doppler_true + doppler_noise_all;
            if env.PRE_WHITEN
                    L = env.pre_whit_L;
                    r0 = range_with_error;
                    f0 = doppler_with_error;
                    range_with_error   = L(1,1)*r0 + L(1,2)*f0;
                    doppler_with_error = L(2,1)*r0 + L(2,2)*f0;
            end
            measurements_all_with_error = zeros(NUM_TAR,numNodes, M*2);
            measurements_all_with_error(:,:, 1:2:end) = range_with_error; % Odd index for range
            measurements_all_with_error(:,:, 2:2:end) = doppler_with_error; % Even index for Doppler
        end

        function [range_with_error_cell,doppler_with_error_cell,numNodes_cell,radar_positions_cell,...
            state_mu_cell, state_cov_cell] = pharse_measurements_tracking(laplacian_matrix,...
        range_with_error, doppler_with_error,range_with_error_cell, doppler_with_error_cell,num_tar, network_topo,...
        state_mu, state_cov)
            for i = 1: num_tar
                for n = 1: network_topo.numNodes
                    current_neighbors = find(laplacian_matrix(n, :) ~= 0);
                    k = 0;
                    range_with_error_1 =[];
                    doppler_with_error_1 = [];
                    radar_positions_1 = [];
                    mu_state_neighbor = {};
                    sigma_state_neighbor = {};
                    for j = current_neighbors
                        k = k+1;
                        range_with_error_1(:,k) = range_with_error(i,j,:);
                        doppler_with_error_1(:,k) = doppler_with_error(i,j,:);
                        radar_positions_1(k,:) = network_topo.radar_pos(j,:);
                        numNodes_1 = length(current_neighbors);
                        mu_state_neighbor{k} = state_mu{j};
                        sigma_state_neighbor{k} = state_cov{j};
                        range_with_error_cell{n} = range_with_error_1;
                        doppler_with_error_cell{n} = doppler_with_error_1;
                        numNodes_cell{n} = numNodes_1;
                        radar_positions_cell{n} = radar_positions_1;
                        state_mu_cell{n} = mu_state_neighbor;
                        state_cov_cell{n} = sigma_state_neighbor;

                    end
                end
            end
        end

        function [prior_mean, prior_cov] = get_neighbors_cell(laplacian_matrix, num_tar, numNodes, estimated_params, state_cov)
            for i = 1: num_tar
                for n = 1: numNodes
                    current_neighbors = find(laplacian_matrix(n, :) ~= 0);
                    k = 0;
                    mu_state_neighbor = {};
                    sigma_state_neighbor = {};
                    for j = current_neighbors
                        k = k+1;
                        if size(estimated_params,2) == 1 % Check number of estimator params
                            mu_state_neighbor{k} = estimated_params;

                        elseif size(estimated_params,2) == numNodes
                            mu_state_neighbor{k} = estimated_params(:,j);

                        else
                            error('Size of estimated_params is not correct');
                        end

                        if size(state_cov,2) ==  4 % Check number of estimator params
                            sigma_state_neighbor{k} = state_cov;

                        elseif size(state_cov,2) == numNodes
                            sigma_state_neighbor{k} = state_cov{j};

                        else
                            error('Size of estimated_params is not correct');
                        end

                        prior_mean{n} = mu_state_neighbor;
                        prior_cov{n} = sigma_state_neighbor;
                    end
                    
                end
            end
        end

        function ADMM = Initialized_ADMM(network_topo)
            options_Dctral = optimoptions('fmincon', 'Display', 'off', SpecifyObjectiveGradient=true, ScaleProblem=true, OptimalityTolerance=1e-2, FunctionTolerance=1e-6, StepTolerance=1e-6, MaxIterations=100);
            options_Ctral =  optimoptions('fmincon', 'Display', 'off', ScaleProblem=true, OptimalityTolerance=1e-6, FunctionTolerance=1e-6, StepTolerance=1e-6, MaxIterations=100000);
            ADMM = struct();
            ADMM.solver = options_Dctral;
            ADMM.converged= false;
            ADMM.max_iter = 500; % Maximumconverged ADMM iterations
            ADMM.c_penalty = [100,100,15,15];
            ADMM.lb = [-inf,-inf,-inf,-inf];
            ADMM.ub = [inf,inf, inf, inf];
            ADMM.initial_values = repmat([1000, 1000, -14, 14]', 1,network_topo.numNodes);
            ADMM.prev_mean = cell(1,network_topo.numNodes);
            ADMM.prev_cov = cell(1,network_topo.numNodes);
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
            ADMM.DOPPLER_Ys = [];
            ADMM.converg_r = false;
            ADMM.converg_d = false;
            ADMM.c_penalty = [100,100,15,15];
            ADMM.tau_incr = [2.01, 2.01, 2.1, 2.1];  % Smaller increase factor
            ADMM.tau_decr = [2.01, 2.01, 2.1, 2.1];  % Smaller decrease factor
            for n = 1:network_topo.numNodes
                ADMM.Nu{n} = zeros(4, network_topo.numNodes);
                ADMM.Nu_prev{n} = zeros(4, network_topo.numNodes);
                ADMM.update_z{n} = zeros(4, network_topo.numNodes);
                ADMM.update_z_prev{n} = zeros(4, network_topo.numNodes);
            end

        end
    end
end
