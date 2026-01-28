clc;clear;close all;
ut       = ADMM_utils;
models_ut = models;
fig_ut   = make_figs(10);

DEBUG       = true;
PRE_WHITEN  = true;   % 你要 pre-whiten：在 dkf_neighbor_update 內一致處理

%% ---------------- P1: Simulation parameters ----------------
NUM_CPI_PER_MEA = 64;
TRACK_TIME      = 25;
dt              = 1e-3;
NUM_TAR = 1;


% -----------------------
% MC config
% -----------------------
num_monte_carlo = 1;
seed0 = 43;
LOG_ENABLE = false;
LOG_DIR = "./data_log";
RUN_NAME = "tracking_kf";
TYPE="KF";

%% ---------------- Result log -------------
Results = struct();
Results.primal_residauls = cell(1,TRACK_TIME); % In each cell (time stemp), store the primal residuals (cell: 1 x # iteration of opt.) 
Results.dual_residauls = cell(1,TRACK_TIME);   % In each cell (time stemp), store the dual residuals (cell: 1 x # iteration of opt.)
Results.estimations_DA = cell(1,TRACK_TIME);      % In each cell (time stemp), store the estimation results (cell: 4 x 10 x # iteration of opt.)
Results.true_params = cell(1,TRACK_TIME);      % In each cell (time stemp), store the estimation results (cell: 1 x 4)
Results.estimations_CA = cell(1,TRACK_TIME);   % In each cell (time stemp), store the estimation results from centralized approach (cell: 1 x 4)
Results.convg_iter = cell(1,TRACK_TIME);
all_tracking_params = cell(NUM_TAR, TRACK_TIME);
%% ---------------- Target ----------------
target.initial_position = [20, -20];
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
% [Try other trajectory]
% pos = ut.gen_ref_trajectory(TRACK_TIME*NUM_CPI_PER_MEA,[-30 20 -30 25], dt,target.speed);
% target.target_position = reshape(pos,[1, TRACK_TIME*NUM_CPI_PER_MEA,2]);

%% ---------------- Network topology ----------------
network_topo.numNodes = 10;
theta = linspace(0,2*pi, network_topo.numNodes+1);
network_topo.theta = theta(1:end-1);
network_topo.radius = 30;
network_topo.com_rad_CR = 30;

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
env.rho = 0;
env.Sigma = [env.range_var,                         env.rho*env.range_var*env.doppler_var;
             env.rho*env.range_var*env.doppler_var, env.doppler_var];

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


% 重要：P0 不能用 Q（太小會鎖死速度），給速度較大的不確定度
P0 = diag([1e6, 1e6, 1e4, 1e4]);
% P0 = diag([1e-6, 1e-6, 1e-6, 1e-6]);

x_dkf = cell(1, network_topo.numNodes);
P_dkf = cell(1, network_topo.numNodes);

% global state as initial
% x0 = [20; -20; -14.1412; 14.1412];
x0 = [20;-20;10;10];
% x0 = [0;0;0;0];
for n = 1:network_topo.numNodes
    x_dkf{n} = x0;
    P_dkf{n} = P0;
end

%Local state as initial-> does not work 
% x0 = [1000;1000;10;10];
% % x0 = [0;0;0;0];
% for n = 1:network_topo.numNodes
%     x_dkf{n} = x0 - [network_topo.radar_pos(n,1);network_topo.radar_pos(n,2);0;0];
%     P_dkf{n} = P0;
% end


%% ---------------- Synthetic measurements ----------------

for mc = 1:num_monte_carlo
    rng(seed0 + mc, "twister");
    fprintf("\n[MC %d/%d]\n", mc, num_monte_carlo);

    M_total = NUM_CPI_PER_MEA * TRACK_TIME;

    range_true   = zeros(NUM_TAR, network_topo.numNodes, M_total);
    doppler_true = zeros(NUM_TAR, network_topo.numNodes, M_total);
    meas_true    = zeros(NUM_TAR, network_topo.numNodes, 2*M_total);

    [range_true, doppler_true, meas_true] = ut.gt_data_generation_tracking( ...
        range_true, doppler_true, meas_true, target, network_topo, env, M_total, NUM_TAR);

    % noisy measurements
    [range_meas, doppler_meas, ~] = ut.add_measurement_noise( ...
        range_true, doppler_true, M_total, NUM_TAR, network_topo.numNodes, env);

    % if DEBUG
    %     disp("**** DEBUG: using noiseless measurements ****");
    %     range_meas   = range_true;
    %     doppler_meas = doppler_true;
    % end

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
                [x_dkf{n}, P_dkf{n}, dbg] = dkf_neighbor_update_pw( ...
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
        global_state = models_ut.local2global(network_topo.radar_pos,global_state);

        if exist('consensus','file') == 2
            [estimated_params, estimated_params_hist, ~, ~, ~, diff_hist] = consensus(global_state, network_topo.adj_matrix);
            fprintf("Burst %d consensus diff(end)=%.3e\n", kBurst, diff_hist(end));
            if kBurst == 20
                snapshot= estimated_params_hist;
                snapshot = permute(snapshot,[2,1,3]);
                snapshot_true = [19,-19,-8.75,8.62];
                snapshot_ctrl   = [19,-19,-8.75,8.62];
            end
        else
            estimated_params = mean(global_state, 1);
        end

        fprintf("Burst %d estimate: [%.2f %.2f %.2f %.2f]\n", kBurst, estimated_params);
        all_tracking_params{1,kBurst} = estimated_params;
        % Save log
        Results.estimations_DA{mc,kBurst} = estimated_params;
        Results.true_params{mc,kBurst} = [squeeze(target.target_position(1, kBurst*NUM_CPI_PER_MEA, :))', target.true_params(3), target.true_params(4)];
        Results.convg_iter{mc,kBurst} = size(estimated_params_hist,2);
        Results.primal_residual{mc,kBurst} = diff_hist;
        % results.estimations_CA{mc,k}   = est_CA(:).'; % Tobe implement
        % results.dual_residuals{mc,k}  = %to be implement


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

        % save_json_log(LOG_DIR, RUN_NAME, log);
        save_mat_log(LOG_DIR, RUN_NAME, log);
        % save("test.mat","log")
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
    % yline(target.true_params(1),'k--','GT');
    plot([1:1:TRACK_TIME*NUM_CPI_PER_MEA], target.target_position(1,:,1),'DisplayName','GT','LineStyle','--','Color','k');
    title("DKF X"); legend('show');

    subplot(2,2,2); hold on;
    for n=1:network_topo.numNodes
        plot(start:tEnd, cellfun(@(x) x(2), all_est(start:tEnd,n)), 'DisplayName', "post."+network_topo.labels{n});
    end
    % yline(target.true_params(2),'k--','GT');
    plot([1:1:TRACK_TIME*NUM_CPI_PER_MEA], target.target_position(1,:,2),'DisplayName','GT','LineStyle','--','Color','k');
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


%% -- Plotting
% fig_ut.plot_trajectory(target.target_position,all_tracking_params);
fig_ut.plot_trajectory_and_network(target.target_position, all_tracking_params,network_topo);
fig_ut.plot_converge_across_node_withCentrl(snapshot,snapshot_true,network_topo,snapshot_ctrl);
fig_ut.plot_converge_mse_across_node_withCentrl(snapshot,snapshot_true,network_topo,snapshot_ctrl);
% A = permute(A,[2,1,3]);
% fig_ut.plot_converge_across_node_withCentrl(A,[1000,1000,-8.7,8.6],network_topo,[1000,1000,-8.7,8.62]);

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
