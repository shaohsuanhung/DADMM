clc; clear; close all;

ut       = ADMM_utils;
models_ut = models;
fig_ut   = make_figs(10);

DEBUG       = true;
PRE_WHITEN  = true;   % 你要 pre-whiten：在 dkf_neighbor_update 內一致處理

%% ---------------- P1: Simulation parameters ----------------
num_monte_carlo = 1;
NUM_CPI_PER_MEA = 64;
TRACK_TIME      = 8;
dt              = 1e-3;

NUM_TAR = 1;

%% ---------------- Target ----------------
target.initial_position = [1000, 1000];
target.speed = 20;
target.angle_degrees = 135;
target.direction = [cosd(target.angle_degrees), sind(target.angle_degrees)];
target.true_params = [target.initial_position(1), target.initial_position(2), ...
                      target.speed * target.direction(1), target.speed * target.direction(2)];

target.target_position = zeros(NUM_TAR, NUM_CPI_PER_MEA*TRACK_TIME, 2);
target.target_position(1,1,:) = target.initial_position;

for t = 2:NUM_CPI_PER_MEA*TRACK_TIME
    target.target_position(1,t,:) = squeeze(target.target_position(1,t-1,:))' + target.speed * target.direction * dt;
end

%% ---------------- Network topology ----------------
network_topo.numNodes = 10;
theta = linspace(0,2*pi, network_topo.numNodes+1);
network_topo.theta = theta(1:end-1);
network_topo.radius = 3000;
network_topo.com_rad_CR = 3000;

network_topo.radar_pos = network_topo.radius * [cos(network_topo.theta); sin(network_topo.theta)]';
network_topo.labels = arrayfun(@(i) sprintf("N%d", i), 1:network_topo.numNodes, "UniformOutput", false);

[network_topo.adj_matrix, network_topo.degree_matrix,...
 network_topo.laplacian_matrix, network.inc_matrix, ...
 network_topo.weights_matrix] = ut.calculate_all_graph_matrix( ...
    network_topo.radar_pos, network_topo.com_rad_CR, network_topo.numNodes);

neighbors = ut.get_neighbors(network_topo.adj_matrix, network_topo.numNodes);

%% ---------------- Environment / measurement noise ----------------
env.c = 3e8;
env.lambda = env.c / 10e9;
env.time_step = dt;

env.B = 10e6 * ones(1,network_topo.numNodes);
env.fs = 2*env.B;

env.SNR_idx = 50;
env.SNR_lin = 10^(env.SNR_idx/10);

env.range_var   = (3 * env.c^2) / (8 * pi^2 * env.B(1)^2 * env.SNR_lin);
env.doppler_var = (3 * (env.fs(1)^2)) / (pi^2 * env.SNR_lin * NUM_CPI_PER_MEA^3);

env.Sigma = [env.range_var, 0;
             0,             env.doppler_var];

env.PRE_WHITEN = PRE_WHITEN;
if env.PRE_WHITEN
    % L*Sigma*L' = I
    env.pre_whit_L = inv(chol(env.Sigma,'upper')');
else
    env.pre_whit_L = eye(2);
end

%% ---------------- DKF / CV model ----------------
Q = 1e-2 * [dt^4/4, 0,      dt^3/2, 0;
            0,      dt^4/4, 0,      dt^3/2;
            dt^3/2, 0,      dt^2,   0;
            0,      dt^3/2, 0,      dt^2];

% 初始估計（全域）
x0 = [1000; 1000; -14.1412; 14.1412];

% 重要：P0 不能用 Q（太小會鎖死速度），給速度較大的不確定度
P0 = diag([1e6, 1e6, 1e4, 1e4]);

x_dkf = cell(1, network_topo.numNodes);
P_dkf = cell(1, network_topo.numNodes);

for n = 1:network_topo.numNodes
    x_dkf{n} = x0;
    P_dkf{n} = P0;
end

%% ---------------- Synthetic measurements ----------------
M_total = NUM_CPI_PER_MEA * TRACK_TIME;

range_true   = zeros(NUM_TAR, network_topo.numNodes, M_total);
doppler_true = zeros(NUM_TAR, network_topo.numNodes, M_total);
meas_true    = zeros(NUM_TAR, network_topo.numNodes, 2*M_total);

[range_true, doppler_true, meas_true] = ut.gt_data_generation_tracking( ...
    range_true, doppler_true, meas_true, target, network_topo, env, M_total, NUM_TAR);

% noisy measurements
[range_meas, doppler_meas, ~] = ut.add_measurement_noise( ...
    range_true, doppler_true, M_total, NUM_TAR, network_topo.numNodes, env);

if DEBUG
    disp("**** DEBUG: using noiseless measurements ****");
    range_meas   = range_true;
    doppler_meas = doppler_true;
end

%% ---------------- Run simulation ----------------
all_est = cell(M_total, network_topo.numNodes);
all_pred = cell(M_total, network_topo.numNodes);

for kBurst = 1:TRACK_TIME
    for inst = 1:NUM_CPI_PER_MEA
        t = (kBurst-1)*NUM_CPI_PER_MEA + inst;
        fprintf("t=%d/%d\n", t, M_total);

        % 1) predict all nodes
        x_pred_nodes = cell(1, network_topo.numNodes);
        P_pred_nodes = cell(1, network_topo.numNodes);
        for n = 1:network_topo.numNodes
            [x_pred_nodes{n}, P_pred_nodes{n}] = dkf_predict_cv(x_dkf{n}, P_dkf{n}, dt, Q);
            all_pred{t,n} = x_pred_nodes{n};
        end

        % 2) neighbor-augmented update at each node (LKF-II)
        for n = 1:network_topo.numNodes
            % idx_set MUST match your measurement stacking order:
            % In your parsing function you used find(laplacian(n,:) ~= 0)
            idx_set = find(network_topo.laplacian_matrix(n,:) ~= 0).';
            J = numel(idx_set);

            y_bar = zeros(2*J, 1);
            for a = 1:J
                j = idx_set(a);
                y_bar(2*a-1) = range_meas(1, j, t);
                y_bar(2*a)   = doppler_meas(1, j, t);
            end

            cfgTS = struct('verbose', false, 'check_jacobian', false);
            [x_dkf{n}, P_dkf{n}, dbg] = dkf_neighbor_update( ...
                x_pred_nodes{n}, P_pred_nodes{n}, y_bar, idx_set, network_topo, env, models_ut);

            if DEBUG && (t==1 || mod(t,32)==0) && n==1
                cfgTS.verbose = true;
                cfgTS.check_jacobian = true;
                dkf_troubleshoot_step(cfgTS, x_pred_nodes{n}, P_pred_nodes{n}, y_bar, idx_set, network_topo, env, models_ut, ...
                    'x_post', x_dkf{n}, 'P_post', P_dkf{n});
            end

            all_est{t,n} = x_dkf{n};
        end
    end

    % 3) consensus (可選)
    % 這裡使用 DKF 的全域估計做共識
    global_state = zeros(network_topo.numNodes, 4);
    for n = 1:network_topo.numNodes
        global_state(n,:) = x_dkf{n}.';
    end

    if exist('consensus','file') == 2
        [estimated_params, ~, ~, ~, ~, diff_hist] = consensus(global_state, network_topo.adj_matrix);
        fprintf("Burst %d consensus diff(end)=%.3e\n", kBurst, diff_hist(end));
    else
        estimated_params = mean(global_state, 1);
    end

    fprintf("Burst %d estimate: [%.2f %.2f %.2f %.2f]\n", kBurst, estimated_params);
end

%% ---------------- Plot quick check ----------------
if DEBUG
    start = 1;
    tEnd = M_total;

    figure;
    subplot(2,2,1); hold on;
    for n=1:network_topo.numNodes
        plot(start:tEnd, cellfun(@(x) x(1), all_est(start:tEnd,n)), 'DisplayName', "post."+network_topo.labels{n});
    end
    title("DKF X"); legend('show');

    subplot(2,2,2); hold on;
    for n=1:network_topo.numNodes
        plot(start:tEnd, cellfun(@(x) x(2), all_est(start:tEnd,n)), 'DisplayName', "post."+network_topo.labels{n});
    end
    title("DKF Y"); legend('show');

    subplot(2,2,3); hold on;
    for n=1:network_topo.numNodes
        plot(start:tEnd, cellfun(@(x) x(3), all_est(start:tEnd,n)), 'DisplayName', "post."+network_topo.labels{n});
    end
    yline(target.true_params(3),'k--','GT');
    title("DKF Vx"); legend('show');

    subplot(2,2,4); hold on;
    for n=1:network_topo.numNodes
        plot(start:tEnd, cellfun(@(x) x(4), all_est(start:tEnd,n)), 'DisplayName', "post."+network_topo.labels{n});
    end
    yline(target.true_params(4),'k--','GT');
    title("DKF Vy"); legend('show');
end
