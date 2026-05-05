clc; clear;close all;
ut = utils;

% -----------------------
% MC config
% -----------------------
num_monte_carlo = 10;
seed0 = 43;
LOG_ENABLE = false;
TYPE = "linear"; % Options: random_walk, sinusoid, linear
LOG_DIR = "./data/"+TYPE;
% -----------------------
% Simulation configuration
% -----------------------
NUM_TAR = 1;
NUM_CPI_PER_MEA = 64;
TRACK_TIME = 5;
time_step = 1e-2;
PRE_WHITEN = false;

% -----------------------
% Target start point
% -----------------------
INIT_POS = [-20, -20];
SPEED    =  20;
RAND_ANGLE = false;
env.RAND_ANGLE = RAND_ANGLE;
if RAND_ANGLE 
    target.angle_degrees = (360-0).*rand(1,1) + 0; % Randomize the direction of heading
else
    ANGLE    = 45; % In deg. 
end
% -----------------------
% Distributed Sensor Network topology
% -----------------------
network_topo.numNodes = 10;
theta = linspace(0,2*pi, network_topo.numNodes+1);
network_topo.theta = theta(1:end-1);
network_topo.com_rad_CR = 20;
network_topo.radius = 20;
network_topo.radar_pos = network_topo.radius * [cos(network_topo.theta); sin(network_topo.theta)]';
network_topo.labels = arrayfun(@(k) sprintf("N%d",k), 1:network_topo.numNodes, 'UniformOutput', false);
network_topo.labels = {'N1','N2','N3','N4','N5','N6','N7','N8','N9','N10'};
[network_topo.adj_matrix, network_topo.degree_matrix, network_topo.laplacian_matrix, network_topo.inc_matrix, network_topo.weights_matrix] = ...
    ut.calculate_all_graph_matrix(network_topo.radar_pos, network_topo.com_rad_CR, network_topo.numNodes);
neighbors = ut.get_neighbors(network_topo.adj_matrix, network_topo.numNodes);

% -----------------------
% Env / signal parameter setting
% -----------------------
env.c = 3e8;
env.lambda = env.c / 10e9;
env.time_step = time_step;
env.Bandwidth = 10e6 * ones(1,network_topo.numNodes);
env.fs = 2*env.Bandwidth;

env.SNR_idx = 50;
env.SNR_lin = 10^(env.SNR_idx/10);

env.range_var   = (3 * env.c^2) / (8 * pi^2 * env.Bandwidth(1)^2 * env.SNR_lin);
env.doppler_var = (3 * (env.fs(1)^2)) / (pi^2 * env.SNR_lin * NUM_CPI_PER_MEA^3);
env.range_sd = sqrt(env.range_var);
env.doppler_sd = sqrt(env.doppler_var);
env.rho = 1e-6;


% Noise setting
env.measurements_noise = [env.range_var, env.rho*env.range_sd*env.doppler_sd; ...
             env.rho*env.range_sd*env.doppler_sd, env.doppler_var];


dt = time_step;
env.process_noise = 1e-2 * [dt^4/4, 0,      dt^3/2, 0;
                    0,      dt^4/4, 0,      dt^3/2;
                    dt^3/2, 0,      dt^2,   0;
                    0,      dt^3/2, 0,      dt^2];


% Pre-whitening (L*Sigma*L' = I)
if PRE_WHITEN
    U = chol(env.Sigma, 'upper');     % U'*U = Sigma
    env.pre_whit_L = inv(U');         % inv(U')*Sigma*inv(U) = I
    env.measurements_noise = eye(2);
    env.PRE_WHITEN = true;
else
    env.pre_whit_L = eye(2);
    env.measurements_noise = env.measurements_noise;
    env.PRE_WHITEN = false;
end

% -----------------------
% MC loop
% -----------------------
for mc = 1:num_monte_carlo
    % Automatically naminig the folder
    RUN_NAME = "radar_tracking_"+ TYPE + "_"+ "_" + mc + "_";
    rng(seed0 + mc, "twister");
    fprintf("\n[MC %d/%d]\n", mc, num_monte_carlo);
    % -----------------------
    % Target (deterministic trajectory per MC; you can randomize angle per MC)
    % -----------------------
    Trajectory = struct();
    % Trajectory.true_params  = cell(NUM_TAR,TRACK_TIME*NUM_CPI_PER_MEA,4); % 4 estimate states
    % Trajectory.measurements = cell(NUM_TAR, network_topo.numNodes, TRACK_TIME*NUM_CPI_PER_MEA,2);  % 2 measurement states

    target.initial_position = INIT_POS;
    target.speed = SPEED;

    if RAND_ANGLE
        % randomize angle
        target.angle_degrees = (360-0).*rand(1,1) + 0; % Randomize the direction of heading
    else
        target.angle_degrees = ANGLE;
    end
    target.direction = [cosd(target.angle_degrees), sind(target.angle_degrees)];
    target.true_params = [target.initial_position(1), target.initial_position(2), ...
                          target.speed*target.direction(1), target.speed*target.direction(2)];


    %%-------------------
    % Here you can select trajectory
    %%-------------------
    if TYPE == "linear"
        target.target_position = zeros(NUM_TAR, NUM_CPI_PER_MEA*TRACK_TIME, 2);
        for i = 1:NUM_TAR
            target.target_position(i,1,:) = target.initial_position;
        end
        for i = 1:NUM_TAR
            for t = 2:NUM_CPI_PER_MEA*TRACK_TIME
                target.target_position(i,t,:) = squeeze(target.target_position(i,t-1,:))' + target.speed*target.direction*time_step;
            end
        end

    elseif TYPE == "sinusoid"
        % [Sine trajectory]
        X_AMP = 1;
        Y_AMP = 10;
        t  = linspace(0, 10, TRACK_TIME*NUM_CPI_PER_MEA)';x = X_AMP*t+target.initial_position(1,1);y = Y_AMP*sin(t)+target.initial_position(1,2);traj = [x, y];
        target.target_position = reshape(traj,[1,TRACK_TIME*NUM_CPI_PER_MEA,2]);
    
    elseif TYPE == "random_walk"
        % [Random walk trajectory]
         pos = ut.gen_ref_trajectory(TRACK_TIME*NUM_CPI_PER_MEA,[-30 20 -30 25], time_step,target.speed);
         target.target_position = reshape(pos,[1, TRACK_TIME*NUM_CPI_PER_MEA,2]);
    
    else
        error("No such TYPE:" + TYPE);
    end
    target.target_state = [squeeze(target.target_position),repmat([target.speed*target.direction(1),target.speed*target.direction(2)],size(squeeze(target.target_position),1),1)];
    % -----------------------
    % Generate GT measurements for agents
    % -----------------------
    range_true = zeros(NUM_TAR, network_topo.numNodes, NUM_CPI_PER_MEA*TRACK_TIME);
    doppler_true = zeros(NUM_TAR, network_topo.numNodes, NUM_CPI_PER_MEA*TRACK_TIME);
    measurements_true = zeros(NUM_TAR, network_topo.numNodes, 2*NUM_CPI_PER_MEA*TRACK_TIME);

    [range_true, doppler_true, measurements_true] = ut.gt_data_generation_tracking( ...
        range_true, doppler_true, measurements_true, target, network_topo, env, NUM_CPI_PER_MEA*TRACK_TIME, NUM_TAR);

    % -----------------------
    % Add noise to measurements
    % -----------------------
    range_with_error = zeros(size(range_true));
    doppler_with_error = zeros(size(doppler_true));
    measurements_with_error = zeros(size(measurements_true));

    [range_with_error, doppler_with_error, measurements_with_error] = ut.add_measurement_noise( ...
        range_true, doppler_true, NUM_CPI_PER_MEA*TRACK_TIME, NUM_TAR, network_topo.numNodes, env);

    % -----------------------
    % Parse neighbors
    % -----------------------
    range_withNeighbors = cell(NUM_TAR, network_topo.numNodes);
    doppler_withNeighbors = cell(NUM_TAR, network_topo.numNodes);
    numNodes_withNeighbors = cell(NUM_TAR, network_topo.numNodes);
    radarpos_withNeighbors = cell(NUM_TAR, network_topo.numNodes);

    prior_mu_cell = cell(1, network_topo.numNodes);
    prior_sigma_cell = cell(1, network_topo.numNodes);

    [range_withNeighbors, doppler_withNeighbors, ...
    numNodes_withNeighbors, radarpos_withNeighbors] = ut.pharse_measurements_tracking( ...
                                                    network_topo.laplacian_matrix, range_with_error, ...
                                                    doppler_with_error,range_withNeighbors, ...
                                                    doppler_withNeighbors, NUM_TAR, network_topo);

    % -----------------------
    % Iterate over time steps
    % -----------------------
    for tar = 1:NUM_TAR
        for k = 1:TRACK_TIME
            % ----------------------
            % Distributed data pharser (Per agent, only aggregate measurements from neighbors)
            %------------------------
            range_win = cell(1, network_topo.numNodes);
            doppler_win = cell(1, network_topo.numNodes);
            for n = 1:network_topo.numNodes
                range_win{n} = range_withNeighbors{tar,n}((k-1)*NUM_CPI_PER_MEA+1 : k*NUM_CPI_PER_MEA, :);
                doppler_win{n} = doppler_withNeighbors{tar,n}((k-1)*NUM_CPI_PER_MEA+1 : k*NUM_CPI_PER_MEA, :);
            end
            % -----------------------
            % Centralized data pharser (Aggregate measurements from all agents)
            % -----------------------
            range_win_CA = squeeze(range_with_error(tar,:,(k-1)*NUM_CPI_PER_MEA+1 : k*NUM_CPI_PER_MEA))';
            doopler_win_CA = squeeze(doppler_with_error(tar,:,(k-1)*NUM_CPI_PER_MEA+1 : k*NUM_CPI_PER_MEA))';

            % save per time step
            % for inst = 1: NUM_CPI_PER_MEA
            %     t = (k-1)*NUM_CPI_PER_MEA + inst;
            %     true_params_k = [squeeze(target.target_position(tar, t, :))', target.true_params(3), target.true_params(4)];
            %     Trajectory.true_params{t,tar}      = true_params_k;
            %     for n = 1:network_topo.numNodes
            %         Trajectory.measurements{t,n}   =  [squeeze(range_with_error)]
            %     end 
            % end
            
            fprintf("Time step %d completed.\n", k);
        end
        Trajectory.true_params{tar}      = target.target_state;
        for n = 1: network_topo.numNodes
            Trajectory.measurements{tar,n}    = [range_with_error(tar,n,:),doppler_with_error(tar,n,:)];
        end
    end

    %% plot (optional, plot random step on convg. plot to make sure results works fine)
    if mc == 1
        % fig_ut.plot_trajectory(target.target_position, ADMM.final_tracking_estimation);
        true_param_k = [squeeze(target.target_position(tar, k*NUM_CPI_PER_MEA, :))', -target.true_params(3), -target.true_params(4)];
        ut.plot_setup(target.target_position,network_topo);
    end

    % Save JSON/MAT Log per MC
    if LOG_ENABLE
        Log = struct();
        Log.RUN_NAME = RUN_NAME;
        Log.NUM_TAR = NUM_TAR;
        Log.NUM_CPI_PER_MEA = NUM_CPI_PER_MEA;
        Log.track_time = TRACK_TIME; % localization cas 
        Log.mc = mc;
        Log.TYPE = TYPE;
        Log.PRE_WHITEN = PRE_WHITEN;
        Log.seed = seed0 + mc;
        Log.network_topo = network_topo;
        Log.constant = env;
        Log.target = target;
        Log.Trajectory = Trajectory;    
        ut.save_json_log(LOG_DIR, RUN_NAME, Log);
        ut.save_mat_log(LOG_DIR, RUN_NAME, Log);
    end
end
fprintf("\nData Generation Done.\n");


