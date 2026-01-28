clc; close all; clear;
ut = ADMM_utils;
DEBUG=true; % To see verbose
TYPE="MAP";
PRE_WHITEN=false;   % <<< switch here
SEED = 43;
%-- Config. of the run
num_monte_carlo = 10;
direction_mc = zeros(num_monte_carlo,2);
gt_paras_mc = zeros(num_monte_carlo,4);

%-- Network topo
network_topo.numNodes = 10;
theta = linspace(0,2*pi, network_topo.numNodes+1);
network_topo.theta = theta(1:end-1);
network_topo.com_rad_CR = 3000; % communication radius range
network_topo.radius = 3000;     % spatial placement radius 
network_topo.radar_pos = network_topo.radius * [cos(network_topo.theta); sin(network_topo.theta)]';
network_topo.C_distance = 1;
network_topo.C_data = 1;
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
env.B = 10e6 * ones(1, network_topo.numNodes);
env.fs = 2 * env.B;

M_values = 64;
node_range = 10;

options_DA = optimoptions('fmincon', 'Display', 'off', ScaleProblem=true, ...
    OptimalityTolerance=1e-6, FunctionTolerance=1e-6, StepTolerance=1e-6, MaxIterations=100000);
options_CA = optimoptions('fmincon', 'Display', 'off', ScaleProblem=true, ...
    OptimalityTolerance=1e-6, FunctionTolerance=1e-6, StepTolerance=1e-6, MaxIterations=100000);

%-- Start the MC simulation
for mc = 1:num_monte_carlo
    rng(SEED + mc, "twister");
    fprintf("\n[MC %d/%d]\n", mc, num_monte_carlo);
    %-- Target
    target.initial_position = [1000, 1000];
    target.speed = 20;
    % target.angle_degrees = [135];
    target.angle_degrees = (360-0).*rand(1,1) + 0;
    angle_degrees =  target.angle_degrees;
    target.direction = [cos(angle_degrees * pi / 180), sin(angle_degrees * pi / 180)];
    target.true_params = [target.initial_position(1), target.initial_position(2), ...
                          target.speed * target.direction(1), target.speed * target.direction(2)];

    disp('Direction the target is travelling:'); disp(target.direction);

    % Preallocate
    RMSEs = zeros(length(node_range), 4);

    numNodes = node_range;
    radar_pos_T = network_topo.radar_pos';

    disp('Node Range:'); disp(node_range);
    disp('Number of Measurements:'); disp(M_values);

    adj_matrix = zeros(network_topo.numNodes, network_topo.numNodes);

    for com_rad = 1:length(com_rad_CR)
        fprintf('Communication Radius: %d\n',  com_rad_CR(com_rad));
        communication_radius = com_rad_CR(com_rad);

        [adj_matrix,degree_matrix, laplacian_matrix, inc_matrix, weights_matrix] = ...
            ut.calculate_all_graph_matrix(network_topo.radar_pos, communication_radius, network_topo.numNodes);

        network_topo.adj_matrix = adj_matrix;
        network_topo.degree_matrix = degree_matrix;
        network_topo.laplacian_matrix = laplacian_matrix;
        network_topo.inc_matrix = inc_matrix;
        network_topo.weights_matrix = weights_matrix;

        neighbors = ut.get_neighbors(adj_matrix, network_topo.numNodes);

        laplacian_matrix_CR{com_rad}=laplacian_matrix;
        laplacian_eigen = eig(laplacian_matrix);
        laplacian_eigen_vector = sort(laplacian_eigen, 'descend');

        % ---- SNR + measurement noise Sigma (RAW) ----
        snr_idx = 50;
        SNR_lin = 10^(snr_idx / 10);
        fprintf('SNR_db: %d dB, SNR_linear: %f\n', snr_idx, SNR_lin);

        estimated_param_values = [];
        M = M_values;

        target.target_position = zeros(M, 2);
        target.target_position(1, :) = target.initial_position;
        for k = 2:M
            target.target_position(k, :) = target.target_position(k - 1, :) + target.speed * target.direction * env.time_step;
        end

        range_true = zeros(M, network_topo.numNodes);
        doppler_true = zeros(M, network_topo.numNodes);
        measurements_true = zeros(2 * M, network_topo.numNodes);
        [range_true, doppler_true, measurements_true] = ...
            ut.gt_data_generation(range_true,doppler_true, measurements_true, target,network_topo,env, M);

        measurements_true_all = reshape(measurements_true, [], 1);

        % --- RAW Sigma from signal model (用於產生噪聲) ---
        range_var = (3 * env.c^2) / (8 * pi^2 * env.B(1)^2 * SNR_lin) ;
        doppler_var = (3 * ((env.fs(1))^2)) / (pi^2 * SNR_lin * M^3) ;
        range_sd = sqrt(range_var);
        doppler_sd = sqrt(doppler_var);
        rho = 0;

        Sigma_raw = [range_var, rho * range_sd * doppler_sd; ...
                     rho * range_sd * doppler_sd, doppler_var];

        total_measurements = numNodes * M;

        % --- PRE-WHITEN 設定：L*Sigma_raw*L' = I ---
        if PRE_WHITEN
            pre_whit_L = inv(chol(Sigma_raw,'upper')');  % 同你 tracking 那邊用法一致
            Sigma_filter = eye(2);                       % whiten 後量測雜訊
        else
            pre_whit_L = eye(2);
            Sigma_filter = Sigma_raw;
        end

        % --- 重要：噪聲生成永遠用 Sigma_raw，而不是 Sigma_filter ---
        Sigma_big_noise  = kron(eye(total_measurements), Sigma_raw);
        Sigma_big_filter = kron(eye(total_measurements), Sigma_filter); % 給 likelihood / MAP / ADMM 用

        % Generate noise
        noise_matrix = mvnrnd(zeros(2 * total_measurements, 1), Sigma_big_noise)'; % (2NM x 1)
        range_noise = noise_matrix(1:2:end);
        doppler_noise = noise_matrix(2:2:end);

        range_noise_all = reshape(range_noise, M, numNodes);
        doppler_noise_all = reshape(doppler_noise, M, numNodes);

        % y_hat = y_gt + noise  (RAW domain)
        range_with_error = range_true + range_noise_all;
        doppler_with_error = doppler_true + doppler_noise_all;

        % --- whiten measurements (用暫存，避免連鎖污染) ---
        if PRE_WHITEN
            r0 = range_with_error;
            d0 = doppler_with_error;
            range_with_error   = pre_whit_L(1,1) * r0 + pre_whit_L(1,2) * d0;
            doppler_with_error = pre_whit_L(2,1) * r0 + pre_whit_L(2,2) * d0;
        end

        measurements_with_error_all = measurements_true_all + noise_matrix; %#ok<NASGU>

        range_with_error_CA = range_with_error;
        doppler_with_error_CA = doppler_with_error;

        % ---- Prior ----
        prior_mu = cell(1, numNodes);
        prior_sigma = cell(1, numNodes);

        if TYPE == "MAP"
            prior_mu = repmat({[1000; 1000; -14; 14]}, 1,numNodes);

            if PRE_WHITEN
                prior_sigma = repmat({eye(4)}, 1,numNodes);
            else
                prior_sigma = repmat({[range_var,0,0,0; ...
                                      0,range_var,0,0;...
                                      0,0,doppler_var,0;...
                                      0,0,0,doppler_var]}, 1,numNodes);
            end

        elseif TYPE == "MLE"
            prior_mu = repmat({[0; 0; 0; 0]}, 1,numNodes);
            prior_sigma = repmat({zeros(4)}, 1,numNodes);
        end

        % init / bounds
        initial_guess = [1000, 1000, 10, 10]';
        lb = [-inf,-inf,-inf,-inf];
        ub = [inf,inf, inf, inf];

        % ---- Centralized baseline (CA) ----
        if TYPE == "MLE"
            if PRE_WHITEN
                fun = @(params) ut.logLikelihood(params, range_with_error_CA, doppler_with_error_CA, ...
                    network_topo.radar_pos, network_topo.numNodes, M, env.lambda, ...
                    Sigma_big_filter, pre_whit_L);
            else
                fun = @(params) ut.logLikelihood(params, range_with_error_CA, doppler_with_error_CA, ...
                    network_topo.radar_pos, network_topo.numNodes, M, env.lambda, ...
                    Sigma_big_filter);
            end

        elseif TYPE == "MAP"
            if PRE_WHITEN
                fun = @(params) ut.MAP(params, range_with_error_CA, doppler_with_error_CA, prior_mu, ...
                    prior_sigma, network_topo.radar_pos, network_topo.numNodes, M, ...
                    env.lambda, Sigma_big_filter, pre_whit_L);
            else
                fun = @(params) ut.MAP(params, range_with_error_CA, doppler_with_error_CA, prior_mu, ...
                    prior_sigma, network_topo.radar_pos, network_topo.numNodes, M, ...
                    env.lambda, Sigma_big_filter);
            end
        end

        [estimated_params_CA, log_likelihood, exitflag, output] = ...
            fmincon(fun, initial_guess,[],[],[],[], lb, ub, [],options_CA);
        estimates_mc_CA(mc, :) = estimated_params_CA;

        %% ---- Parse measurements to neighbor-cells ----
        range_with_error_cell = cell(1, numNodes);
        doppler_with_error_cell = cell(1, numNodes);
        numNodes_cell = cell(1, numNodes);
        radar_positions_cell = cell(1, numNodes);
        Sigma_big_1_cell = cell(1, numNodes);
        Sigma_big_2_cell = cell(1, numNodes);
        prior_mu_cell = cell(1,network_topo.numNodes);
        prior_sigma_cell = cell(1,network_topo.numNodes);

        [range_with_error_cell,doppler_with_error_cell,numNodes_cell, ...
         radar_positions_cell,Sigma_big_1_cell,Sigma_big_2_cell, ...
         prior_mu_cell, prior_sigma_cell] = ut.pharse_measurements(laplacian_matrix, ...
            range_with_error, doppler_with_error, Sigma_big_filter, ...
            range_with_error_cell, doppler_with_error_cell, ...
            prior_mu, prior_sigma, network_topo, M);

        %% ---- Distributed ADMM ----
        iteration = 0;
        tolerance = 1e-4;
        max_iterations = 300;

        c_penalty = [100, 100, 15, 15]; % SNR 50dB
        Nu = cell(1, numNodes);
        Nu_prev = cell(1, numNodes);
        update_z = cell(1, numNodes);
        update_z_prev = cell(1, numNodes);
        primal_residual_all =[];
        primal_residual_by_para = cell(1,numNodes);
        dual_residual_all = [];
        dual_residual_by_para = cell(1,numNodes);
        all_estimations_every_iter = [];

        global RANGE_Xs RANGE_Ys DOPPLER_Xs DOPPLER_Ys converg_r converg_d converged;
        RANGE_Xs = []; RANGE_Ys = []; DOPPLER_Xs = []; DOPPLER_Ys = [];
        converged = false; converg_r = false; converg_d = false;

        tau_incr = [2.01, 2.01, 2.1, 2.1];
        tau_decr = [2.01, 2.01, 2.1, 2.1];
        mu = [3,3,10,10];
        alpha = [0.5, 0.5, 0.5, 0.5];

        initial_values = repmat([1000, 1000, 10, 10]', 1,numNodes);

        for n = 1:network_topo.numNodes
            Nu{n} = zeros(4, network_topo.numNodes);
            Nu_prev{n} = zeros(4, network_topo.numNodes);
            update_z{n} = zeros(4, network_topo.numNodes);
            update_z_prev{n} = zeros(4, network_topo.numNodes);
        end

        while ~converged && iteration < max_iterations
            iteration = iteration+1;
            fprintf('ADMM iteration: %d\n', iteration);

            for n = 1: network_topo.numNodes
                for j = neighbors{n}
                    Nu{n}(:,j) = Nu_prev{n}(:,j) + c_penalty' .* (initial_values(:,n) - update_z_prev{n}(:,j));
                end

                if TYPE == "MLE"
                    if PRE_WHITEN
                        fun = @(params) ut.logLikelihoodWithConsensus(params, range_with_error_cell{n}, doppler_with_error_cell{n}, ...
                            radar_positions_cell{n},numNodes_cell{n}, M, env.lambda, ...
                            Sigma_big_2_cell{n}, n, neighbors, Nu, initial_values, update_z_prev, c_penalty, ...
                            pre_whit_L);
                    else
                        fun = @(params) ut.logLikelihoodWithConsensus(params, range_with_error_cell{n}, doppler_with_error_cell{n}, ...
                            radar_positions_cell{n},numNodes_cell{n}, M, env.lambda, ...
                            Sigma_big_2_cell{n}, n, neighbors, Nu, initial_values, update_z_prev, c_penalty);
                    end

                elseif TYPE == "MAP"
                    if PRE_WHITEN
                        fun = @(params) ut.posteriorWithConsensus(params, range_with_error_cell{n}, doppler_with_error_cell{n}, ...
                            prior_mu_cell{n}, prior_sigma_cell{n}, radar_positions_cell{n}, ...
                            numNodes_cell{n}, M, env.lambda, Sigma_big_2_cell{n}, ...
                            n, neighbors, Nu, initial_values, update_z_prev, c_penalty, ...
                            pre_whit_L);
                    else
                        fun = @(params) ut.posteriorWithConsensus(params, range_with_error_cell{n}, doppler_with_error_cell{n}, ...
                            prior_mu_cell{n}, prior_sigma_cell{n}, radar_positions_cell{n}, ...
                            numNodes_cell{n}, M, env.lambda, Sigma_big_2_cell{n}, ...
                            n, neighbors, Nu, initial_values, update_z_prev, c_penalty);
                    end
                end

                estimated_params = fmincon(fun, initial_values(:,n),[],[],[],[], lb, ub, [],options_DA);
                all_estimations(:,n) = estimated_params;
            end

            all_estimations_every_iter(:,:,iteration) = all_estimations;

            for n = 1: network_topo.numNodes
                for j = neighbors{n}
                    update_z{n}(:,j) = (1/2) * (((c_penalty.^(-1))' .* (Nu{n}(:,j) + Nu{j}(:,n))) + ...
                        all_estimations(:,n) + all_estimations(:, j));
                end
            end

            primal_residual = 0;
            dual_residual = 0;
            primal_residual_params = zeros(4, 1);
            dual_residual_params   = zeros(4, 1);
            prima_residual_by_node = zeros(4,network_topo.numNodes);
            dual_residual_by_node  = zeros(4,network_topo.numNodes);

            for n = 1: network_topo.numNodes
                for j = neighbors{n}
                    primal_residual = primal_residual + (norm(all_estimations(:,n) - update_z{n}(:,j), 2)^2);
                    primal_residual_params = primal_residual_params + abs(all_estimations(:,n) - update_z{n}(:,j));

                    dual_residual = dual_residual + (norm(Nu{n}(:,j) - Nu_prev{n}(:,j))^2);
                    dual_residual_params = dual_residual_params + abs(Nu{n}(:,j) - Nu_prev{n}(:,j));
                end
                prima_residual_by_node(:,n) = primal_residual_params;
                dual_residual_by_node(:,n)  = dual_residual_params;
            end

            primal_residual_all(iteration)     = primal_residual;
            dual_residual_all(iteration)       = dual_residual;
            primal_residual_by_para{iteration} = prima_residual_by_node;
            dual_residual_by_para{iteration}   = dual_residual_by_node;

            if primal_residual < tolerance || iteration == max_iterations
                break;
            end

            if mod(iteration, 30) == 0
                if primal_residual < 10* dual_residual
                    c_penalty = tau_incr .* c_penalty;
                elseif dual_residual < 10*primal_residual
                    c_penalty = c_penalty .* ((tau_decr).^(-1));
                end
            end

            initial_values = all_estimations;
            update_z_prev = update_z;
            Nu_prev = Nu;

            [all_estimations]= ut.ADMM_stop_criterion(primal_residual_by_para{iteration}, tolerance, ...
                all_estimations, RANGE_Xs, RANGE_Ys, DOPPLER_Xs, DOPPLER_Ys, converg_r, converg_d, converged, ...
                DEBUG, iteration, max_iterations);
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

%-- save log
experiment_params_log = struct();
experiment_params_log.network_topo = network_topo;
experiment_params_log.NUM_CPI_PER_MEA = M_values;
experiment_params_log.TRACK_TIME = 1;
experiment_params_log.TYPE = TYPE;
experiment_params_log.constant = env;
experiment_params_log.PRE_WHITEN = PRE_WHITEN;
experiment_params_log.primal_residual_mpc = primal_residual_CR;
experiment_params_log.dual_residual_mc  = dual_residual_CR;
experiment_params_log.all_estimations_every_iter_mc = all_estimations_every_iter_CR;
experiment_params_log.true_params_mc = true_params_mc;
experiment_params_log.estimates_mc_CA = estimates_mc_CA;

if DEBUG==false
    ut.write_exp_log('./data_log', 'exp_config_localization',experiment_params_log);
end

%% Figures
true_params = [target.initial_position(1), target.initial_position(2), ...
              target.speed * target.direction(1), target.speed * target.direction(2)];
fig_ut = make_figs(network_topo.numNodes);

fig_ut.plot_converge_across_node_withCentrl(all_estimations_every_iter,true_params,network_topo,estimates_mc_CA);
fig_ut.plot_geometry_and_target(network_topo, target.target_position);
