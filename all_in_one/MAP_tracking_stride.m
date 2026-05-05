clc; clear;close all;
run_tracking_mc()

function run_tracking_mc()
ut = ADMM_utils;
fig_ut = make_figs(10);
DEBUG = true;
PRE_WHITEN = false;              % <--- switch here
TYPE = "MAP";
% -----------------------
% MC config
% -----------------------
num_monte_carlo = 1;
seed0 = 43;
LOG_ENABLE = false;
LOG_DIR = "./data_log/stride";
RUN_NAME = "tracking";
% -----------------------
% Simulation config
% -----------------------
NUM_TAR = 1;
NUM_CPI_PER_MEA = 64;
TRACK_TIME = 30;
time_step = 1e-2;
STRIDE = 32;
NUM_INFERENCE = floor((NUM_CPI_PER_MEA*TRACK_TIME - NUM_CPI_PER_MEA) / STRIDE) + 1;
TRACK_IDX = 1:STRIDE:(NUM_CPI_PER_MEA*TRACK_TIME - STRIDE- 1);
TRACK_IDX = TRACK_IDX+NUM_CPI_PER_MEA-1; % Shift to the last index
TRACK_IDX = TRACK_IDX(TRACK_IDX<=NUM_CPI_PER_MEA*TRACK_TIME);

Results = struct();
Results.primal_residauls = cell(num_monte_carlo, NUM_INFERENCE);
Results.dual_residauls   = cell(num_monte_carlo, NUM_INFERENCE);
Results.estimations_DA   = cell(num_monte_carlo, NUM_INFERENCE);
Results.true_params      = cell(num_monte_carlo, NUM_INFERENCE);
Results.convg_iter       = cell(num_monte_carlo, NUM_INFERENCE);
Results.estimations_CA   = cell(num_monte_carlo, NUM_INFERENCE);
Results.ADMM_setting   = cell(num_monte_carlo, NUM_INFERENCE);
Results.CRLB           = cell(num_monte_carlo, NUM_INFERENCE);
Results.consensus_estimates = cell(num_monte_carlo, NUM_INFERENCE);


%------ TOBE Implement
%------ V2I data formatter
% Need: 1. Object ID, 2. Objecet Type, 3. Position, 4. Speed, 5. Direction,
% 6. Confidence factor, 7. Timestemp
Detection = struct(); % One for a timestamp
Detection.id = 1;     % Always be 1 
Detection.type = 1;   % Always be 1
Detection.position = zeros(2,1); % Write [2x1] 
Detection.speed    = zeros(2,1);   % Write [2x1]
Detection.direction = zeros(2,1); 
Detection.confidence_level = zeros(4,4); % Write [4x4] 
Detection.timestamp = 0.0; %float
% -----------------------
% Network topology
% -----------------------
network_topo.numNodes = 10;
theta = linspace(0,2*pi, network_topo.numNodes+1);
network_topo.theta = theta(1:end-1);
network_topo.com_rad_CR = 30;
network_topo.radius = 30;
network_topo.radar_pos = network_topo.radius * [cos(network_topo.theta); sin(network_topo.theta)]';
network_topo.labels = arrayfun(@(k) sprintf("N%d",k), 1:network_topo.numNodes, 'UniformOutput', false);
network_topo.labels = {'N1','N2','N3','N4','N5','N6','N7','N8','N9','N10'};
[network_topo.adj_matrix, network_topo.degree_matrix, network_topo.laplacian_matrix, network_topo.inc_matrix, network_topo.weights_matrix] = ...
    ut.calculate_all_graph_matrix(network_topo.radar_pos, network_topo.com_rad_CR, network_topo.numNodes);

neighbors = ut.get_neighbors(network_topo.adj_matrix, network_topo.numNodes);

% -----------------------
% Env / signal
% -----------------------
env.c = 3e8;
env.lambda = env.c / 10e9;
env.time_step = time_step;
env.T = env.time_step / 2;
env.B = 10e6 * ones(1,network_topo.numNodes);
env.fs = 2*env.B;

env.SNR_idx = 50;
env.SNR_lin = 10^(env.SNR_idx/10);

env.range_var   = (3 * env.c^2) / (8 * pi^2 * env.B(1)^2 * env.SNR_lin);
env.doppler_var = (3 * (env.fs(1)^2)) / (pi^2 * env.SNR_lin * NUM_CPI_PER_MEA^3);
env.range_sd = sqrt(env.range_var);
env.doppler_sd = sqrt(env.doppler_var);
env.rho = 1e-6;

env.Sigma = [env.range_var, env.rho*env.range_sd*env.doppler_sd; ...
             env.rho*env.range_sd*env.doppler_sd, env.doppler_var];

% We assume the process noise
dt = env.time_step;
env.Q = 1e-2 * [dt^4/4, 0,      dt^3/2, 0;
            0,      dt^4/4, 0,      dt^3/2;
            dt^3/2, 0,      dt^2,   0;
            0,      dt^3/2, 0,      dt^2];

% Pre-whitening (L*Sigma*L' = I)
if PRE_WHITEN
    U = chol(env.Sigma, 'upper');     % U'*U = Sigma
    env.pre_whit_L = inv(U');         % inv(U')*Sigma*inv(U) = I
    env.Sigma_filter = eye(2);
    env.PRE_WHITEN = true;
else
    env.pre_whit_L = eye(2);
    env.Sigma_filter = env.Sigma;
    env.PRE_WHITEN = false;
end

% Prior (for MAP in tracking)
%%----- [Important initial value setting] ------%%
% ADMM base config (kept similar to your script)
prior_mu = repmat({[-30; -30; -14; 14]}, 1, network_topo.numNodes);
ADMM = ut.Initialized_ADMM(network_topo);
ADMM.max_iter = 1000;
ADMM.tolerance = 1e-3;
% ADMM.initial_values = repmat([1000, 1000, 10, 10]', 1, network_topo.numNodes);
ADMM.initial_values = repmat([-30,-30, 10, 10]', 1, network_topo.numNodes);
state_cov = diag([env.range_var, env.range_var, env.doppler_var, env.doppler_var]);
if PRE_WHITEN
    prior_sigma = repmat({eye(4)}, 1, network_topo.numNodes);
else
    prior_sigma = repmat({state_cov}, 1, network_topo.numNodes);
end


options_CA = optimoptions('fmincon', 'Display', 'off', ScaleProblem=true, ...
    OptimalityTolerance=1e-6, FunctionTolerance=1e-6, StepTolerance=1e-6, MaxIterations=50000);
% -----------------------
% MC loop
% -----------------------
for mc = 1:num_monte_carlo
    V2I_msg = cell(size(TRACK_IDX,2));   % One for a MCrun
    rng(seed0 + mc, "twister");
    fprintf("\n[MC %d/%d]\n", mc, num_monte_carlo);

    % -----------------------
    % Target (deterministic trajectory per MC; you can randomize angle per MC)
    % -----------------------
    target.initial_position = [-30, -30];
    target.speed = 20;
    target.angle_degrees = 135;
    % build trajectory
    % angle_degrees = target.angle_degrees; % or randomize by mc
    % target.angle_degrees = (360-0).*rand(1,1) + 0; % Randomize the direction of heading
    angle_degrees =  target.angle_degrees;
    target.direction = [cosd(angle_degrees), sind(angle_degrees)];
    target.true_params = [target.initial_position(1), target.initial_position(2), ...
                          target.speed*target.direction(1), target.speed*target.direction(2)];
    target.target_position = zeros(NUM_TAR, NUM_CPI_PER_MEA*TRACK_TIME, 2);

    for i = 1:NUM_TAR
        target.target_position(i,1,:) = target.initial_position;
    end
    for i = 1:NUM_TAR
        for t = 2:NUM_CPI_PER_MEA*TRACK_TIME
            target.target_position(i,t,:) = squeeze(target.target_position(i,t-1,:))' + target.speed*target.direction*time_step;
        end
    end
    % [Try other trajectory]
    %[Traj 1]
    % t  = linspace(0, 10, TRACK_TIME*NUM_CPI_PER_MEA)';x = 3*t+target.initial_position(1,1);y = 18*sin(t)+target.initial_position(1,2);traj = [x, y];
    % target.target_position = reshape(traj,[1,TRACK_TIME*NUM_CPI_PER_MEA,2]);

    % [Traj 2]
    pos = ut.gen_ref_trajectory(TRACK_TIME*NUM_CPI_PER_MEA,[-30 20 -30 25], dt,target.speed);
    target.target_position = reshape(pos,[1, TRACK_TIME*NUM_CPI_PER_MEA,2]);

    % -----------------------
    % Generate GT measurements (tracking)
    % -----------------------
    range_true = zeros(NUM_TAR, network_topo.numNodes, NUM_CPI_PER_MEA*TRACK_TIME);
    doppler_true = zeros(NUM_TAR, network_topo.numNodes, NUM_CPI_PER_MEA*TRACK_TIME);
    measurements_true = zeros(NUM_TAR, network_topo.numNodes, 2*NUM_CPI_PER_MEA*TRACK_TIME);

    [range_true, doppler_true, measurements_true] = ut.gt_data_generation_tracking( ...
        range_true, doppler_true, measurements_true, target, network_topo, env, NUM_CPI_PER_MEA*TRACK_TIME, NUM_TAR);

    % -----------------------
    % Add noise (IMPORTANT: relies on your ADMM_utils.add_measurement_noise fix)
    % -----------------------
    range_with_error = zeros(size(range_true));
    doppler_with_error = zeros(size(doppler_true));
    measurements_with_error = zeros(size(measurements_true));

    [range_with_error, doppler_with_error, measurements_with_error] = ut.add_measurement_noise( ...
        range_true, doppler_true, NUM_CPI_PER_MEA*TRACK_TIME, NUM_TAR, network_topo.numNodes, env);

    % -----------------------
    % Parse neighbors (tracking version)
    % -----------------------
    range_withNeighbors = cell(NUM_TAR, network_topo.numNodes);
    doppler_withNeighbors = cell(NUM_TAR, network_topo.numNodes);
    numNodes_withNeighbors = cell(NUM_TAR, network_topo.numNodes);
    radarpos_withNeighbors = cell(NUM_TAR, network_topo.numNodes);

    prior_mu_cell = cell(1, network_topo.numNodes);
    prior_sigma_cell = cell(1, network_topo.numNodes);

    [range_withNeighbors, doppler_withNeighbors, numNodes_withNeighbors, radarpos_withNeighbors, ...
        prior_mu_cell, prior_sigma_cell] = ut.pharse_measurements_tracking( ...
        network_topo.laplacian_matrix, range_with_error, doppler_with_error, ...
        range_withNeighbors, doppler_withNeighbors, NUM_TAR, network_topo, prior_mu, prior_sigma);

    % -----------------------
    % Tracking over time steps
    % -----------------------
    for tar = 1:NUM_TAR
        for idx = 1:numel(TRACK_IDX)
            k = TRACK_IDX(idx);
            % windowed chunk
            range_win = cell(1, network_topo.numNodes);
            doppler_win = cell(1, network_topo.numNodes);
            for n = 1:network_topo.numNodes
                range_win{n} = range_withNeighbors{tar,n}(k-NUM_CPI_PER_MEA+1 : k, :);
                doppler_win{n} = doppler_withNeighbors{tar,n}(k-NUM_CPI_PER_MEA+1 : k, :);
            end

            
            % -----------------------
            % Centralized (Need data of the batch from all nodes: CPI x # Nodes)
            % -----------------------
            if idx == 1
                CA_initial_guess = [-30, -30, 10, 10]';

            else
                CA_initial_guess = ADMM.final_tracking_estimation{tar, k-STRIDE};
            end

            lb = [-inf,-inf,-inf,-inf];
            ub = [ inf, inf, inf, inf];
            range_win_CA = squeeze(range_with_error(tar,:,k-NUM_CPI_PER_MEA+1 : k ))';
            doopler_win_CA = squeeze(doppler_with_error(tar,:,k-NUM_CPI_PER_MEA+1 : k))';
            if TYPE == "MLE"
                if PRE_WHITEN
                    funCA = @(p) ut.logLikelihood(p, range_win_CA, doopler_win_CA, network_topo.radar_pos, ...
                        network_topo.numNodes, NUM_CPI_PER_MEA, env.lambda, eye(2), env.pre_whit_L);
                else
                    funCA = @(p) ut.logLikelihood(p, range_win_CA, doopler_win_CA, network_topo.radar_pos, ...
                        network_topo.numNodes, NUM_CPI_PER_MEA, env.lambda, env.Sigma_filter);
                end
            elseif TYPE == "MAP"
                if PRE_WHITEN
                    funCA = @(p) ut.MAP(p, range_win_CA, doopler_win_CA, prior_mu, prior_sigma, network_topo.radar_pos, ...
                        network_topo.numNodes, NUM_CPI_PER_MEA, env.lambda, eye(2), env.pre_whit_L);
                else
                    funCA = @(p) ut.MAP(p, range_win_CA, doopler_win_CA, prior_mu, prior_sigma, network_topo.radar_pos, ...
                        network_topo.numNodes, NUM_CPI_PER_MEA, env.lambda, env.Sigma_filter);
                end
            else
                error('No such TYPE setting.');
            end

            est_CA = fmincon(funCA, CA_initial_guess, [], [], [], [], lb, ub, [], options_CA);
            CA_initial_guess = est_CA;
            % Prior from previous step
            if idx ~= 1
                prev_est = ADMM.final_tracking_estimation{tar, k-STRIDE};
                ADMM.initial_values = repmat(prev_est, 1, network_topo.numNodes);

                [ADMM.prior_mean, ADMM.prior_cov] = ut.get_neighbors_cell( ...
                    network_topo.laplacian_matrix, NUM_TAR, network_topo.numNodes, prev_est, state_cov);

                ADMM.state_cov = [ADMM.prior_cov{1}{1}(1,1),ADMM.prior_cov{1}{1}(1,2),ADMM.prior_cov{1}{1}(1,3),ADMM.prior_cov{1}{1}(1,4);
                                  ADMM.prior_cov{1}{1}(2,1),ADMM.prior_cov{1}{1}(2,2),ADMM.prior_cov{1}{1}(2,3),ADMM.prior_cov{1}{1}(2,4);
                                  ADMM.prior_cov{1}{1}(3,1),ADMM.prior_cov{1}{1}(3,2),env.lambda*ADMM.prior_cov{1}{1}(3,3),ADMM.prior_cov{1}{1}(3,4);
                                  ADMM.prior_cov{1}{1}(4,1),ADMM.prior_cov{1}{1}(4,2),ADMM.prior_cov{1}{1}(4,3),env.lambda*ADMM.prior_cov{1}{1}(4,4)];

            elseif idx == 1
                ADMM.prior_mean = prior_mu_cell;
                ADMM.prior_cov  = prior_sigma_cell;
                ADMM.state_cov = [ADMM.prior_cov{1}{1}(1,1),ADMM.prior_cov{1}{1}(1,2),ADMM.prior_cov{1}{1}(1,3),ADMM.prior_cov{1}{1}(1,4);
                                  ADMM.prior_cov{1}{1}(2,1),ADMM.prior_cov{1}{1}(2,2),ADMM.prior_cov{1}{1}(2,3),ADMM.prior_cov{1}{1}(2,4);  
                                  ADMM.prior_cov{1}{1}(3,1),ADMM.prior_cov{1}{1}(3,2),env.lambda^2*ADMM.prior_cov{1}{1}(3,3),ADMM.prior_cov{1}{1}(3,4);
                                  ADMM.prior_cov{1}{1}(4,1),ADMM.prior_cov{1}{1}(4,2),ADMM.prior_cov{1}{1}(4,3),env.lambda^2*ADMM.prior_cov{1}{1}(4,4)];

            else
                error("Wrong k setting")
            end

            % reset ADMM iteration state each time step (keep your helper)
            ADMM.converged = false;
            ADMM.converg_r = false; ADMM.converg_d = false;

            iteration = 0;
            all_estimations = zeros(4, network_topo.numNodes);

            while ~ADMM.converged && iteration < ADMM.max_iter
                iteration = iteration + 1;
                if mod(iteration,50)== 0
                    fprintf('\rADMM iter %d, Primal: %d, Dual: %d\n', iteration, primal_residual, dual_residual);
                end

                % local solve for each node
                for n = 1:network_topo.numNodes
                    for j = neighbors{n}
                        ADMM.Nu{n}(:,j) = ADMM.Nu_prev{n}(:,j) + ADMM.c_penalty' .* (ADMM.initial_values(:,n) - ADMM.update_z_prev{n}(:,j));
                    end
                    
                    % if ((k == 1) || (TYPE == "MLE")) %Assume prior in k=1
                    if TYPE == "MLE"
                        % first step: use MLE likelihood
                        if PRE_WHITEN
                            fun = @(p) ut.logLikelihoodWithConsensus(p, range_win{n}, doppler_win{n}, ...
                                radarpos_withNeighbors{tar,n}, numNodes_withNeighbors{tar,n}, NUM_CPI_PER_MEA, env.lambda, ...
                                env.Sigma_filter, n, neighbors, ADMM.Nu, ADMM.initial_values, ADMM.update_z_prev, ADMM.c_penalty, env.pre_whit_L);
                        else
                            fun = @(p) ut.logLikelihoodWithConsensus(p, range_win{n}, doppler_win{n}, ...
                                radarpos_withNeighbors{tar,n}, numNodes_withNeighbors{tar,n}, NUM_CPI_PER_MEA, env.lambda, ...
                                env.Sigma_filter, n, neighbors, ADMM.Nu, ADMM.initial_values, ADMM.update_z_prev, ADMM.c_penalty);
                        end
                    elseif (TYPE=="MAP")
                        % later steps: MAP posterior
                        if PRE_WHITEN
                            fun = @(p) ut.posteriorWithConsensus(p, range_win{n}, doppler_win{n}, ...
                                ADMM.prior_mean{n}, ADMM.prior_cov{n}, radarpos_withNeighbors{tar,n}, numNodes_withNeighbors{tar,n}, ...
                                NUM_CPI_PER_MEA, env.lambda, env.Sigma_filter, n, neighbors, ADMM.Nu, ADMM.initial_values, ...
                                ADMM.update_z_prev, ADMM.c_penalty, env.pre_whit_L);
                        else
                            fun = @(p) ut.posteriorWithConsensus(p, range_win{n}, doppler_win{n}, ...
                                ADMM.prior_mean{n}, ADMM.prior_cov{n}, radarpos_withNeighbors{tar,n}, numNodes_withNeighbors{tar,n}, ...
                                NUM_CPI_PER_MEA, env.lambda, env.Sigma_filter, n, neighbors, ADMM.Nu, ADMM.initial_values, ...
                                ADMM.update_z_prev, ADMM.c_penalty);
                        end
                    else
                        error('Wrong TYPE setting.')
                    end

                    all_estimations(:,n) = fmincon(fun, ADMM.initial_values(:,n), [], [], [], [], ADMM.lb, ADMM.ub, [], ADMM.solver);
                end

                ADMM.all_estimations_every_iter(:,:,iteration) = all_estimations;

                % update z
                for n = 1:network_topo.numNodes
                    for j = neighbors{n}
                        ADMM.update_z{n}(:,j) = 0.5 * ( ((ADMM.c_penalty.^(-1))' .* (ADMM.Nu{n}(:,j) + ADMM.Nu{j}(:,n))) ...
                            + all_estimations(:,n) + all_estimations(:,j) );
                    end
                end

                % residuals
                primal_residual = 0;
                dual_residual   = 0;
                prima_by_node = zeros(4, network_topo.numNodes);
                dual_by_node  = zeros(4, network_topo.numNodes);

                for n = 1:network_topo.numNodes
                    pr = zeros(4,1);
                    dr = zeros(4,1);
                    for j = neighbors{n}
                        primal_residual = primal_residual + sqrt(norm(all_estimations(:,n) - ADMM.update_z{n}(:,j))^2);
                        dual_residual   = dual_residual   + sqrt(norm(ADMM.Nu{n}(:,j) - ADMM.Nu_prev{n}(:,j))^2);
                        pr = pr + abs(all_estimations(:,n) - ADMM.update_z{n}(:,j));
                        dr = dr + abs(ADMM.Nu{n}(:,j) - ADMM.Nu_prev{n}(:,j));
                    end
                    prima_by_node(:,n) = pr;
                    dual_by_node(:,n)  = dr;
                end

                ADMM.primal_residual_all(iteration) = primal_residual;
                ADMM.dual_residual_all(iteration)   = dual_residual;
                ADMM.primal_residual_by_para{iteration} = prima_by_node;
                ADMM.dual_residual_by_para{iteration}   = dual_by_node;

                % penalty update (same as your style)
                if mod(iteration,30)==0
                    if primal_residual < dual_residual
                        ADMM.c_penalty = ADMM.tau_incr .* ADMM.c_penalty;
                    elseif dual_residual < primal_residual
                        ADMM.c_penalty = ADMM.c_penalty .* ((ADMM.tau_decr).^(-1));
                    end
                end

                % shift
                ADMM.initial_values = all_estimations;
                ADMM.update_z_prev = ADMM.update_z;
                ADMM.Nu_prev = ADMM.Nu;

                % stop criterion (reuse your util)
                ut.ADMM_stop_criterion(prima_by_node, ADMM.tolerance, all_estimations, ...
                    ADMM.RANGE_Xs, ADMM.RANGE_Ys, ADMM.DOPPLER_Xs, ADMM.DOPPLER_Ys, ...
                    ADMM.converg_r, ADMM.converg_d, ADMM.converged, DEBUG, iteration, ADMM.max_iter);

                if primal_residual < ADMM.tolerance
                    ADMM.converged = true;
                end
            end

            % save per time step, after consensus ADMM. 
            true_params_k = [squeeze(target.target_position(tar, k, :))', target.true_params(3), target.true_params(4)];
            ADMM.final_tracking_estimation{tar,k} = mean(all_estimations,2); % you can also pick node-1 etc, since the diff. of each node shoud be very small.
            
            
            
            Results.primal_residauls{mc,k} = ADMM.primal_residual_all; 
            Results.dual_residauls{mc,k}   = ADMM.dual_residual_all;
            Results.estimations_DA{mc,k}   = ADMM.all_estimations_every_iter;
            Results.true_params{mc,k}      = true_params_k;
            Results.convg_iter{mc,k}       = iteration; 
            Results.estimations_CA{mc,k}   = est_CA(:).';
            Results.consensus_estimates{mc,k} = mean(all_estimations,2);
            Results.node_wise_cov{mc,k} = cov(all_estimations(1:2,:).','omitrows');
            if TYPE == "MAP"
                Results.CRLB{mc,k}         = ut.calculateBCRLB(true_params_k,...
                                                 network_topo.radar_pos, network_topo.numNodes,...
                                                 NUM_CPI_PER_MEA, env.lambda, env.Sigma_filter,env.Q);
            elseif TYPE == "MLE"
                FIM = ut.calculateFIM(true_params_k,...
                                             network_topo.radar_pos, network_topo.numNodes,...
                                             NUM_CPI_PER_MEA, env.lambda, env.Sigma_filter);
                Results.CRLB{mc,k}         = inv(FIM);
            else
                error('No such TYPE setting.');
            end
            % Results.ADMM_setting{mc,k} = ADMM; % save ADMM setting for reference
            %------ Write shapefuture log
            x  = est_CA(1);
            y  = est_CA(2);
            vx = -est_CA(3);
            vy = -est_CA(4);
            dir = rad2deg(atan(vy/vx));
            DetectionLog = make_detection_log(1,1,k*time_step, [x,y], [vx,vy], dir, ADMM.state_cov,[true_params_k(1),true_params_k(2)],[true_params_k(3),true_params_k(4)]);
            V2I_msg{idx} = DetectionLog;
            %------
            % reset ADMM buffers for next time step
            ADMM = ut.ADMM_reset(ADMM, NUM_TAR, network_topo);

            fprintf("Time step %d completed.\n", k);

            
        end

        %------ Save V2I_msg per MC
        save_json_log(LOG_DIR, RUN_NAME + '_V2I_' + string(mc),V2I_msg);
    end

    %% plot (optional, plot random step on convg. plot to make sure results works fine)
    if mc == 1
        % fig_ut.plot_trajectory(target.target_position, ADMM.final_tracking_estimation);
        true_param_k = [squeeze(target.target_position(tar, k, :))', target.true_params(3), target.true_params(4)]
        fig_ut.plot_trajectory_and_network(target.target_position, ADMM.final_tracking_estimation,network_topo);
        % fig_ut.plot_converge_across_node_withCentrl(Results.estimations_DA{mc,k},true_param_k,network_topo, est_CA(:).');
        % fit_ut.plot_geometry_and_target(network_topo, target.target_position);
        fig_ut.plot_converge_mse_across_node_withCentrl(Results.estimations_DA{mc,k},true_param_k,network_topo, est_CA(:).');
    end
end
    % JSON log per MC
    if LOG_ENABLE
        log = struct();
        log.RUN_NAME = RUN_NAME;
        log.NUM_TAR = NUM_TAR;
        log.NUM_CPI_PER_MEA = NUM_CPI_PER_MEA;
        log.track_time = TRACK_TIME; % localization cases

        log.mc = mc;
        log.TYPE = TYPE;
        log.PRE_WHITEN = PRE_WHITEN;
        log.seed = seed0 + mc;

        log.network_topo = network_topo;
        log.constant = env;
        log.target = target;
        log.Results = Results;

        save_json_log(LOG_DIR, RUN_NAME, log);
        % save_mat_log(LOG_DIR, RUN_NAME, log);
        % save("test.mat","log")
    end
fprintf("\nDone.\n");
end

function save_json_log(root_dir, run_name, s)
if ~exist(root_dir, "dir"); mkdir(root_dir); end
ts = datetime("now","Format","yyyyMMdd_HHmmss");
folder = fullfile(root_dir, sprintf("%s_%s", run_name, string(ts)));
mkdir(folder);
fn = fullfile(folder, "log.json");
txt = jsonencode(s, "PrettyPrint", true);
fid = fopen(fn, "w");
fwrite(fid, txt, "char");
fclose(fid);
fprintf("  [log] %s\n", fn);
end

function save_mat_log(root_dir, run_name, log)
if ~exist(root_dir, "dir"); mkdir(root_dir); end
ts = datetime("now","Format","yyyyMMdd_HHmmss");
folder = fullfile(root_dir, sprintf("%s_%s", run_name, string(ts)));
mkdir(folder);
fn = fullfile(folder, "log.mat");
save(fn, "log");
% txt = jsonencode(s, "PrettyPrint", true);
% fid = fopen(fn, "w");
% fwrite(fid, txt, "char");
% fclose(fid);
% fprintf("  [log] %s\n", fn);
end


function pos = gen_ref_trajectory(N, bounds, dt, speed, opts)
%GEN_WALK_TRAJECTORY Realistic randomized walking trajectory segment.
%
%   pos = gen_walk_trajectory(N, bounds, dt, speed)
%   pos = gen_walk_trajectory(N, bounds, dt, speed, opts)
%
% Inputs
%   N      : number of samples (integer >= 2)
%   bounds : [xmin xmax ymin ymax] or [xmin xmax; ymin ymax]
%   dt     : sampling time [s], > 0
%   speed  : walking speed [m/s], >= 0
%   opts   : optional struct
%       .seed            : RNG seed (default [])
%       .randomizeShape  : true/false (default true)
%       .M               : number of waypoints for shape (default 18, >= 6)
%       .shapeJitterY    : normalized shape jitter (default 0.04)
%       .shapeJitterX    : normalized shape jitter (default 0.012)
%       .denseN          : dense samples for curve (default 4000)
%
%       % lateral "human" wandering (meters)
%       .lateralStd      : lateral deviation std [m] (default 0.25)
%       .lateralTau      : correlation time [s] (default 1.5)
%
%       % if desired length > curve length:
%       .longMode        : "clip" or "wrap" (default "clip")
%
% Output
%   pos    : [N x 2] trajectory positions [x,y] in meters

    if nargin < 5 || isempty(opts), opts = struct(); end

    % ---- validate inputs ----
    if ~(isscalar(N) && N == round(N) && N >= 2)
        error("N must be an integer >= 2.");
    end
    if ~(isscalar(dt) && isfinite(dt) && dt > 0)
        error("dt must be a positive scalar (seconds).");
    end
    if ~(isscalar(speed) && isfinite(speed) && speed >= 0)
        error("speed must be a nonnegative scalar (m/s).");
    end

    [xmin, xmax, ymin, ymax] = parse_bounds(bounds);
    if ~(xmax > xmin && ymax > ymin)
        error("bounds must satisfy xmax > xmin and ymax > ymin.");
    end

    % ---- options ----
    seed           = get_opt(opts, "seed", []);
    randomizeShape = get_opt(opts, "randomizeShape", true);
    M              = get_opt(opts, "M", 18);
    shapeJitterY   = get_opt(opts, "shapeJitterY", 0.04);
    shapeJitterX   = get_opt(opts, "shapeJitterX", 0.012);
    denseN         = get_opt(opts, "denseN", 4000);

    lateralStd     = get_opt(opts, "lateralStd", 0.25);
    lateralTau     = get_opt(opts, "lateralTau", 1.5);
    longMode       = string(get_opt(opts, "longMode", "clip"));

    if ~(isscalar(M) && M == round(M) && M >= 6)
        error("opts.M must be an integer >= 6.");
    end

    % ---- RNG ----
    if ~isempty(seed)
        rng(seed);
    else
        rng("shuffle");
    end

    % ---- 1) Build a smooth baseline curve inside bounds ----
    [x_dense, y_dense] = make_base_curve(bounds, randomizeShape, M, shapeJitterX, shapeJitterY, denseN);

    % ---- arclength parameter (meters) ----
    s_dense = [0, cumsum(hypot(diff(x_dense), diff(y_dense)))];
    totalLen = s_dense(end);

    % ---- 2) Decide how long the observed trajectory should be ----
    ds = speed * dt;              % step length per sample
    L  = ds * (N-1);              % expected traveled distance in window

    if speed == 0 || ds == 0
        % stationary: pick a random point along the curve
        s0 = rand() * totalLen;
        x0 = interp1(s_dense, x_dense, s0, "linear");
        y0 = interp1(s_dense, y_dense, s0, "linear");
        pos = repmat([x0, y0], N, 1);
        return;
    end

    % choose a start arclength so that we get a segment of length ~L
    if L <= totalLen
        s0 = rand() * (totalLen - L);
        s_q = s0 + (0:N-1) * ds;
    else
        % requested longer than available curve length
        if longMode == "wrap"
            s0 = rand() * totalLen;
            s_q = s0 + (0:N-1) * ds;
            s_q = mod(s_q, totalLen);
        else
            % "clip": traverse the full curve (speed effectively reduced)
            s_q = linspace(0, totalLen, N);
        end
    end

    % ---- 3) Interpolate base positions at s_q ----
    x_base = interp1(s_dense, x_dense, s_q, "linear");
    y_base = interp1(s_dense, y_dense, s_q, "linear");

    % ---- 4) Add realistic lateral wandering (correlated noise) ----
    % tangent from dense curve -> interpolate at s_q -> normal direction
    tx_dense = gradient(x_dense, s_dense);
    ty_dense = gradient(y_dense, s_dense);

    tx = interp1(s_dense, tx_dense, s_q, "linear");
    ty = interp1(s_dense, ty_dense, s_q, "linear");

    % normal = [-ty, tx] normalized
    nrm = hypot(tx, ty);
    nrm(nrm < 1e-12) = 1;
    nx = -ty ./ nrm;
    ny =  tx ./ nrm;

    % AR(1) correlated lateral offset: rho depends on dt and tau
    tau = max(lateralTau, 1e-3);
    rho = exp(-dt / tau);

    e = zeros(1, N);
    e(1) = randn();
    for k = 2:N
        e(k) = rho * e(k-1) + sqrt(1 - rho^2) * randn();
    end

    % scale to meters; also keep it reasonable relative to bounds
    maxStd = 0.08 * min(xmax - xmin, ymax - ymin);  % safety cap
    sigma = min(lateralStd, maxStd);
    offset = sigma * e;

    x = x_base + offset .* nx;
    y = y_base + offset .* ny;

    % clamp to bounds (simple safety)
    x = min(max(x, xmin), xmax);
    y = min(max(y, ymin), ymax);

    pos = [x(:), y(:)];
end

% ---------------- helpers ----------------
function v = get_opt(opts, name, default)
    if isfield(opts, name) && ~isempty(opts.(name))
        v = opts.(name);
    else
        v = default;
    end
end

function [xmin, xmax, ymin, ymax] = parse_bounds(bounds)
    if isnumeric(bounds) && numel(bounds) == 4
        b = bounds(:).';
        xmin = b(1); xmax = b(2); ymin = b(3); ymax = b(4);
        return;
    end
    if isnumeric(bounds) && isequal(size(bounds), [2, 2])
        xmin = bounds(1,1); xmax = bounds(1,2);
        ymin = bounds(2,1); ymax = bounds(2,2);
        return;
    end
    error("bounds must be [xmin xmax ymin ymax] or [xmin xmax; ymin ymax].");
end

function [x_dense, y_dense] = make_base_curve(bounds, randomizeShape, M, jitterX, jitterY, denseN)
    [xmin, xmax, ymin, ymax] = parse_bounds(bounds);

    % base reference in normalized space (similar to your figure)
    x_base = [0.00, 0.06, 0.14, 0.22, 0.30, 0.36, 0.42, 0.48, 0.52, 0.55, 0.58, 0.62, 0.68, 0.74, 0.82, 0.90, 0.96, 1.00];
    y_base = [0.00, 0.06, 0.13, 0.20, 0.28, 0.33, 0.38, 0.46, 0.52, 0.60, 0.67, 0.73, 0.78, 0.82, 0.90, 0.95, 0.985, 1.00];

    % make waypoints
    x_wp = linspace(0, 1, M);
    if randomizeShape
        x_wp(2:end-1) = x_wp(2:end-1) + jitterX * randn(1, M-2);
        x_wp = sort(x_wp);
        x_wp(1) = 0; x_wp(end) = 1;
    end

    y_wp = interp1(x_base, y_base, x_wp, "pchip");

    if randomizeShape
        w = sin(pi * x_wp).^1.3; % window -> 0 at ends
        y_wp = y_wp + jitterY * w .* randn(size(y_wp));

        % enforce monotone increasing y (like the reference)
        y_wp = cummax(y_wp);
        y_wp = y_wp - y_wp(1);
        if y_wp(end) <= 0
            y_wp = linspace(0, 1, M);
        else
            y_wp = y_wp / y_wp(end);
        end
        y_wp = min(max(y_wp, 0), 1);
    end

    % dense curve
    u = linspace(0, 1, denseN);
    v = interp1(x_wp, y_wp, u, "pchip");

    % scale to bounds
    x_dense = xmin + u * (xmax - xmin);
    y_dense = ymin + v * (ymax - ymin);
end


function DetectionLog= make_detection_log(id, type, time, position, speed, heading, confidence_level, true_position, true_velocity)
    % Placeholder for your shapefuture log writing function.
    % You can implement this to write logs in the format that shapefuture expects.
    % For example, you might want to log the estimated position, true position, timestamp, etc.
    DetectionLog                  = struct();
    DetectionLog.id               = id;
    DetectionLog.type             = type;
    DetectionLog.timestamp        = time;
    DetectionLog.position         = position;
    DetectionLog.speed            = speed;
    DetectionLog.direction        = heading;
    DetectionLog.confidence_level = confidence_level;
    DetectionLog.true_position    = true_position;
    DetectionLog.true_velocity    = true_velocity;
end
% bounds = [0 36 0 40];
% 
% % Case A: short observation window
% N1 = 20; dt1 = 0.1; speed = 1.2;  % T = 1.9 s, distance ~ 2.28 m
% pos_short = gen_ref_trajectory(N1, bounds, dt1, speed, struct("seed", 1));
% 
% % Case B: longer observation window
% N2 = 60; dt2 = 0.2;               % T = 11.8 s, distance ~ 14.16 m
% pos_long  = gen_ref_trajectory(N2, bounds, dt2, speed, struct("seed", 1));
% 
% figure; hold on; grid on; axis equal;
% 
% plot(pos_short(:,1), pos_short(:,2), 'ko-', 'LineWidth', 1.2, 'MarkerSize', 6);
% plot(pos_long(:,1),  pos_long(:,2),  'kx-', 'LineWidth', 1.2, 'MarkerSize', 5);

% 
% bounds = [0 36 0 40];
% pos1 = gen_ref_trajectory(20, bounds, struct("seed", 1, "jitterY", 0.03));
% pos2 = gen_ref_trajectory(20, bounds, struct("seed", 2, "jitterY", 0.05));
% 
% figure; hold on; grid on; axis equal;
% plot(pos1(:,1), pos1(:,2), "ko-", "LineWidth", 1.2, "MarkerSize", 7);
% plot(pos2(:,1), pos2(:,2), "kx-", "LineWidth", 1.2, "MarkerSize", 7);


