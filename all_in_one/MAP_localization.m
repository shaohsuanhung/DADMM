run_localization_mc()

function run_localization_mc()
clc; close all;
clear;

ut = ADMM_utils;
DEBUG = true;              % verbose
TYPE  = "MAP";             % "MAP" or "MLE"
PRE_WHITEN = false;         % <--- switch here

% -----------------------
% MC config
% -----------------------
num_monte_carlo =1;
seed0 = 43;

% Log config
LOG_ENABLE = true;
LOG_DIR = "./data_log/localization/MC100_2";
RUN_NAME = "localization";


Results = struct();              
Results.primal_residauls = cell(num_monte_carlo, 1); % Track time = 1
Results.dual_residauls   = cell(num_monte_carlo, 1);
Results.estimations_DA   = cell(num_monte_carlo, 1);
Results.true_params      = cell(num_monte_carlo, 1);
Results.convg_iter       = cell(num_monte_carlo, 1);
Results.estimations_CA   = cell(num_monte_carlo, 1);                           
Results.ADMM_setting   = cell(num_monte_carlo, 1);
Results.CRLB           = cell(num_monte_carlo, 1);
% -----------------------
% Network topology
% -----------------------
network_topo.numNodes = 10;
theta = linspace(0, 2*pi, network_topo.numNodes+1);
network_topo.theta = theta(1:end-1);
network_topo.com_rad_CR = 3000;
network_topo.radius = 3000;
network_topo.radar_pos = network_topo.radius * [cos(network_topo.theta); sin(network_topo.theta)]';

% Precompute distance matrix (optional)
network_topo.distances_between_radar_nodes = zeros(network_topo.numNodes);
for i = 1:network_topo.numNodes
    for j = 1:network_topo.numNodes
        network_topo.distances_between_radar_nodes(i,j) = norm(network_topo.radar_pos(i,:) - network_topo.radar_pos(j,:));
    end
end

% Optimization options
options_DA = optimoptions('fmincon', 'Display', 'off', ScaleProblem=true, ...
    OptimalityTolerance=1e-6, FunctionTolerance=1e-6, StepTolerance=1e-6, MaxIterations=5000);
options_CA = optimoptions('fmincon', 'Display', 'off', ScaleProblem=true, ...
    OptimalityTolerance=1e-6, FunctionTolerance=1e-6, StepTolerance=1e-6, MaxIterations=50000);

% Measurements per node (burst length)
M = 64;
NUM_CPI_PER_MEA = M;
node_range = network_topo.numNodes;

% Env / signal
env.c = 3e8;
env.lambda = env.c / 10e9;
env.time_step = 1e-4;
env.T = env.time_step / 2;
env.B = 10e6 * ones(1, network_topo.numNodes);
env.fs = 2 * env.B;
dt = env.time_step;

% SNR
snr_idx_list = [50];
% snr_idx_list = [50];
SNR_lin_list = 10.^(snr_idx_list./10);

% storage
true_params_mc  = zeros(num_monte_carlo,4);
estimates_CA_mc = zeros(num_monte_carlo,4);
estimates_DA_mc = cell(num_monte_carlo,1);
primal_hist_mc  = cell(num_monte_carlo,1);
dual_hist_mc    = cell(num_monte_carlo,1);

for SNR_lin = SNR_lin_list
for mc = 1:num_monte_carlo
    env.snr_idx = 10*log10(SNR_lin);
    rng(seed0 + mc, "twister");
    fprintf("\n[MC %d/%d]\n", mc, num_monte_carlo);

    % -----------------------
    % Target
    % -----------------------
    target.initial_position = [1000,1000];
    target.speed = 20;
    % target.angle_degrees = (360-0).*rand(1,1) + 0; % Randomize the direction of heading
    target.angle_degrees = [135];
    angle_degrees =  target.angle_degrees;
    target.direction = [cosd(angle_degrees), sind(angle_degrees)];
    target.true_params = [target.initial_position(1), target.initial_position(2), ...
                          target.speed*target.direction(1), target.speed*target.direction(2)];
    true_params_mc(mc,:) = target.true_params;

    % target trajectory within this localization window (M samples)
    target.target_position = zeros(M,2);
    target.target_position(1,:) = target.initial_position;
    for k = 2:M
        target.target_position(k,:) = target.target_position(k-1,:) + target.speed*target.direction*env.time_step;
    end

    % -----------------------
    % Graph matrices
    % -----------------------
    communication_radius = network_topo.com_rad_CR;
    [adj_matrix, degree_matrix, laplacian_matrix, inc_matrix, weights_matrix] = ...
        ut.calculate_all_graph_matrix(network_topo.radar_pos, communication_radius, network_topo.numNodes);
    network_topo.adj_matrix = adj_matrix;
    network_topo.degree_matrix = degree_matrix;
    network_topo.laplacian_matrix = laplacian_matrix;
    network_topo.inc_matrix = inc_matrix;
    network_topo.weights_matrix = weights_matrix;

    neighbors = ut.get_neighbors(adj_matrix, network_topo.numNodes);

    % -----------------------
    % True measurements
    % -----------------------
    range_true = zeros(M, network_topo.numNodes);
    doppler_true = zeros(M, network_topo.numNodes);
    measurements_true = zeros(2*M, network_topo.numNodes);
    [range_true, doppler_true, measurements_true] = ...
        ut.gt_data_generation(range_true, doppler_true, measurements_true, target, network_topo, env, M);

    % -----------------------
    % Noise covarianceㄈㄣiuuuuuuuuuuuuu
    % -----------------------
    range_var   = (3 * env.c^2) / (8 * pi^2 * env.B(1)^2 * SNR_lin);
    doppler_var = (3 * (env.fs(1)^2)) / (pi^2 * SNR_lin * M^3);
    range_sd = sqrt(range_var);
    doppler_sd = sqrt(doppler_var);
    % rho = 0;
    rho = 1e-6; % [See [6] in Srikar's thesis] The whole scale of the non-diagal element should be something aroud 1e-4,1e-5

    Sigma = [range_var, rho*range_sd*doppler_sd; rho*range_sd*doppler_sd, doppler_var];

    % Pre-whitening config (L * Sigma * L' = I)
    if PRE_WHITEN
        U = chol(Sigma, 'upper');      % U' * U = Sigma
        pre_whit_L = inv(U');          % inv(U') * Sigma * inv(U) = I
        Sigma_eff = eye(2);
        env.Q = diag([1, 1, 1, 1]); % To check
    else
        pre_whit_L = eye(2);
        Sigma_eff = Sigma;
        env.Q = diag([range_var, range_var, doppler_var, doppler_var]);
    end
    env.Sigma_filter  = Sigma_eff;
    % -----------------------
    % Generate noise (localization size is OK to build Sigma_big)
    % -----------------------
    numNodes = node_range;
    total_measurements = numNodes * M;            % number of (r,fd) pairs
    Sigma_big = kron(eye(total_measurements), Sigma);

    % noise vector: (2*total_measurements x 1)
    noise_vec = mvnrnd(zeros(1,2*total_measurements), Sigma_big, 1).';
    range_noise_all   = reshape(noise_vec(1:2:end), M, numNodes);
    doppler_noise_all = reshape(noise_vec(2:2:end), M, numNodes);

    range_with_error   = range_true   + range_noise_all;
    doppler_with_error = doppler_true + doppler_noise_all;

    % If PRE_WHITEN: whiten the DATA, and pass L so MODEL is whitened too
    if PRE_WHITEN
        [range_with_error, doppler_with_error] = whiten_pair_mtx(range_with_error, doppler_with_error, pre_whit_L);
    end

    % -----------------------
    % Prior (MAP) / dummy (MLE)
    % -----------------------
    prior_mu = repmat({[1000; 1000; -14; 14]}, 1, numNodes);
    if TYPE == "MAP"
        if PRE_WHITEN
            prior_sigma = repmat({eye(4)}, 1, numNodes);
        else
            % sigma of v_x, v_y is ill-posed. 
            prior_sigma = repmat({diag([range_var, range_var, doppler_var, doppler_var])}, 1, numNodes);
        end
    else
        prior_sigma = repmat({zeros(4)}, 1, numNodes);
    end

    % -----------------------
    % Centralized
    % -----------------------
    initial_guess = [1000, 1000, 10, 10]';
    lb = [-inf,-inf,-inf,-inf];
    ub = [ inf, inf, inf, inf];

    if TYPE == "MLE"
        if PRE_WHITEN
            funCA = @(p) ut.logLikelihood(p, range_with_error, doppler_with_error, network_topo.radar_pos, ...
                numNodes, M, env.lambda, eye(2), pre_whit_L);
        else
            funCA = @(p) ut.logLikelihood(p, range_with_error, doppler_with_error, network_topo.radar_pos, ...
                numNodes, M, env.lambda, Sigma_eff);
        end
    else
        if PRE_WHITEN
            funCA = @(p) ut.MAP(p, range_with_error, doppler_with_error, prior_mu, prior_sigma, network_topo.radar_pos, ...
                numNodes, M, env.lambda, eye(2), pre_whit_L);
        else
            funCA = @(p) ut.MAP(p, range_with_error, doppler_with_error, prior_mu, prior_sigma, network_topo.radar_pos, ...
                numNodes, M, env.lambda, Sigma_eff);
        end
    end

    est_CA = fmincon(funCA, initial_guess, [], [], [], [], lb, ub, [], options_CA);
    estimates_CA_mc(mc,:) = est_CA(:).';
    
    % -----------------------
    % Parse measurements to neighbor-cells (reuse your existing utility)
    % -----------------------
    range_cell = cell(1, numNodes);
    doppler_cell = cell(1, numNodes);
    numNodes_cell = cell(1, numNodes);
    radar_pos_cell = cell(1, numNodes);
    Sigma_big_1_cell = cell(1, numNodes);
    Sigma_big_2_cell = cell(1, numNodes);
    prior_mu_cell = cell(1, numNodes);
    prior_sigma_cell = cell(1, numNodes);

    % NOTE: Here Sigma_big should match what your pharse_measurements expects.
    % For PRE_WHITEN we pass Sigma_big built with Sigma_eff=I.
    Sigma_big_for_parse = kron(eye(total_measurements), Sigma_eff);

    [range_cell, doppler_cell, numNodes_cell, radar_pos_cell, Sigma_big_1_cell, Sigma_big_2_cell, ...
        prior_mu_cell, prior_sigma_cell] = ut.pharse_measurements( ...
        laplacian_matrix, range_with_error, doppler_with_error, Sigma_big_for_parse, ...
        range_cell, doppler_cell, prior_mu, prior_sigma, network_topo, M);

    % -----------------------
    % ADMM distributed
    % -----------------------
    tolerance = 1e-3;
    max_iterations = 1500;
    c_penalty = [100, 100, 15, 15];
    initial_values = repmat(initial_guess, 1, numNodes);
    all_estimations_every_iter = [];
    Nu = cell(1, numNodes);
    Nu_prev = cell(1, numNodes);
    update_z = cell(1, numNodes);
    update_z_prev = cell(1, numNodes);

    for n = 1:numNodes
        Nu{n} = zeros(4, numNodes);
        Nu_prev{n} = zeros(4, numNodes);
        update_z{n} = zeros(4, numNodes);
        update_z_prev{n} = zeros(4, numNodes);
    end

    primal_hist = [];
    dual_hist   = [];

    iteration = 0;
    converged = false;
    all_estimations = zeros(4, numNodes);
    tau_incr = [2.01, 2.01, 2.1, 2.1];  % Smaller increase factor
    tau_decr = [2.01, 2.01, 2.1, 2.1];  % Smaller decrease factor
    while ~converged && iteration < max_iterations
        iteration = iteration + 1;
        if mod(iteration,100)== 0
            fprintf('\rADMM iter %d, Primal: %d, Dual: %d', iteration,primal_residual, dual_residual);
        end
    

        % (a) local solve
        for n = 1:numNodes
            for j = neighbors{n}
                Nu{n}(:,j) = Nu_prev{n}(:,j) + c_penalty' .* (initial_values(:,n) - update_z_prev{n}(:,j));
            end

            if TYPE == "MLE"
                if PRE_WHITEN
                    fun = @(p) ut.logLikelihoodWithConsensus(p, range_cell{n}, doppler_cell{n}, radar_pos_cell{n}, ...
                        numNodes_cell{n}, M, env.lambda, Sigma_big_2_cell{n}, n, neighbors, Nu, initial_values, update_z_prev, c_penalty, pre_whit_L);
                else
                    fun = @(p) ut.logLikelihoodWithConsensus(p, range_cell{n}, doppler_cell{n}, radar_pos_cell{n}, ...
                        numNodes_cell{n}, M, env.lambda, Sigma_big_2_cell{n}, n, neighbors, Nu, initial_values, update_z_prev, c_penalty);
                end
            else
                if PRE_WHITEN
                    fun = @(p) ut.posteriorWithConsensus(p, range_cell{n}, doppler_cell{n}, prior_mu_cell{n}, prior_sigma_cell{n}, ...
                        radar_pos_cell{n}, numNodes_cell{n}, M, env.lambda, Sigma_big_2_cell{n}, n, neighbors, Nu, initial_values, update_z_prev, c_penalty, pre_whit_L);
                else
                    fun = @(p) ut.posteriorWithConsensus(p, range_cell{n}, doppler_cell{n}, prior_mu_cell{n}, prior_sigma_cell{n}, ...
                        radar_pos_cell{n}, numNodes_cell{n}, M, env.lambda, Sigma_big_2_cell{n}, n, neighbors, Nu, initial_values, update_z_prev, c_penalty);
                end
            end

            all_estimations(:,n) = fmincon(fun, initial_values(:,n), [], [], [], [], lb, ub, [], options_DA);
        end

        % (b) update z
        for n = 1:numNodes
            for j = neighbors{n}
                update_z{n}(:,j) = 0.5 * ( ((c_penalty.^(-1))' .* (Nu{n}(:,j) + Nu{j}(:,n))) ...
                    + all_estimations(:,n) + all_estimations(:,j) );
            end
        end

        % (c) residuals
        primal_residual = 0;
        dual_residual   = 0;
        for n = 1:numNodes
            for j = neighbors{n}
                primal_residual = primal_residual + norm(all_estimations(:,n) - update_z{n}(:,j), 2)^2;
                dual_residual   = dual_residual   + norm(Nu{n}(:,j) - Nu_prev{n}(:,j), 2)^2;
            end
        end
        primal_hist(iteration) = primal_residual; %#ok<AGROW>
        dual_hist(iteration)   = dual_residual;   %#ok<AGROW>

        if primal_residual < tolerance
            converged = true;
        end
        
        % % % Every 30 iteration, 
        if mod(iteration, 30) == 0
           % Update the penalty parameter based on the residuals
           if primal_residual < 10* dual_residual
              c_penalty = tau_incr .* c_penalty;
           elseif dual_residual < 10*primal_residual
              c_penalty = c_penalty .* ((tau_decr).^(-1));
           end
        end

        % shift
        all_estimations_every_iter(:,:,iteration) = all_estimations;
        initial_values = all_estimations;
        update_z_prev  = update_z;
        Nu_prev        = Nu;

        
    end
    % Save Results per MC
    Results.primal_residauls{mc,1} = primal_hist;
    Results.dual_residauls{mc,1}   = dual_hist;
    Results.estimations_DA{mc,1}   = all_estimations;
    Results.true_params{mc,1}      = target.true_params;
    Results.convg_iter{mc,1}       = iteration;
    Results.estimations_CA{mc,1}   = est_CA;
    Results.ADMM_setting{mc,1}     = struct('tolerance', tolerance, 'max_iterations', max_iterations, 'c_penalty', c_penalty);
    if TYPE == "MAP"
           Results.CRLB{mc,1}      = ut.calculateBCRLB(target.true_params,...
                                     network_topo.radar_pos, network_topo.numNodes,...
                                     NUM_CPI_PER_MEA, env.lambda, env.Sigma_filter,env.Q);
    elseif TYPE == "MLE"
           FIM = ut.calculateFIM(target.true_params,...
                                network_topo.radar_pos, network_topo.numNodes,...
                                NUM_CPI_PER_MEA, env.lambda, env.Sigma_filter);

          Results.CRLB{mc,1}         = inv(FIM);
   else
          error('No such TYPE setting.');
   end
    % estimates_DA_mc{mc} = all_estimations;
    % primal_hist_mc{mc}  = primal_hist;
    % dual_hist_mc{mc}    = dual_hist;
end
% -----------------------
% JSON Log per MC
% -----------------------
    if LOG_ENABLE
        Log = struct();
        Log.RUN_NAME = RUN_NAME;
        Log.NUM_TAR = 1;
        Log.NUM_CPI_PER_MEA = M;
        Log.track_time = 1; % localization cases

        Log.mc = mc;
        Log.TYPE = TYPE;
        Log.PRE_WHITEN = PRE_WHITEN;
        Log.seed = seed0 + mc;

        Log.network_topo = network_topo;
        Log.constant = env;
        Log.target = target;
        
        Log.Results = Results;
        % To be correct here in the Results
        % Log.est_CA = est_CA;
        % Log.est_DA_all_nodes = all_estimations;
        % Log.all_estimations_every_iter = all_estimations_every_iter;
        % Log.primal_hist = primal_hist;
        % Log.dual_hist = dual_hist;

        % save_json_log(LOG_DIR, RUN_NAME, Log);
        save_mat_log(LOG_DIR, RUN_NAME, Log);
    end
end
%% -----------------------
% Quick plot (optional), show results of the last mc run 
true_params = [target.initial_position(1), target.initial_position(2), target.speed * target.direction(1), target.speed * target.direction(2)];
fig_ut = make_figs(network_topo.numNodes);
fig_ut.plot_geometry_and_target(network_topo, target.target_position);
% fig_ut.plot_converge_across_node_withCentrl(all_estimations_every_iter,true_params,network_topo,est_CA);
fig_ut.plot_converge_mse_across_node_withCentrl(all_estimations_every_iter,true_params,network_topo,est_CA);
fprintf("\nDone.\n");
end

% ---------------- helper: whiten (matrix form) ----------------
function [rW, dW] = whiten_pair_mtx(r, d, L)
% r,d: (M x N) or any same-size numeric arrays
r0 = r; d0 = d;
rW = L(1,1)*r0 + L(1,2)*d0;
dW = L(2,1)*r0 + L(2,2)*d0;
end

% ---------------- helper: json Log ----------------
function save_json_log(root_dir, run_name, s)
if ~exist(root_dir, "dir"); mkdir(root_dir); end
ts = datetime("now","Format","yyyyMMdd_HHmmss");
folder = fullfile(root_dir, sprintf("%s_%s", run_name, string(ts)));
mkdir(folder);
fn = fullfile(folder, "Log.json");
txt = jsonencode(s, "PrettyPrint", true);
fid = fopen(fn, "w");
fwrite(fid, txt, "char");
fclose(fid);
fprintf("\n[Log] %s\n", fn);
end

function save_mat_log(root_dir, run_name, Log)
if ~exist(root_dir, "dir"); mkdir(root_dir); end
ts = datetime("now","Format","yyyyMMdd_HHmmss");
folder = fullfile(root_dir, sprintf("%s_%s", run_name, string(ts)));
mkdir(folder);
fn = fullfile(folder, "Log.mat");
save(fn, "Log");
% txt = jsonencode(s, "PrettyPrint", true);
% fid = fopen(fn, "w");
% fwrite(fid, txt, "char");
% fclose(fid);
% fprintf("  [Log] %s\n", fn);
end