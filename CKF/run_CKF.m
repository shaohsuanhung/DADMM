clc; clear; close all;

ut = ADMM_utils;
models_ut = models;
fig_ut = make_figs(10);

%% ===== Config =====
DEBUG = true;
SAVE_LOG = true;
LOG_DIR = './data_log';
RUN_NAME = "tracking_kf";
TYPE="CKF";

PRE_WHITEN = true;     % <<<< switch here
RNG_SEED = 7;

num_monte_carlo = 1;
NUM_CPI_PER_MEA = 64;
TRACK_TIME = 8;
time_step = 1e-3;

%% ===== Network =====
network_topo.numNodes = 10;
theta = linspace(0,2*pi, network_topo.numNodes+1);
network_topo.theta = theta(1:end-1);
network_topo.com_rad_CR = 3000;
network_topo.radius = 3000;
network_topo.radar_pos = network_topo.radius * [cos(network_topo.theta); sin(network_topo.theta)]';

[network_topo.adj_matrix, network_topo.degree_matrix, ...
 network_topo.laplacian_matrix, network_topo.inc_matrix, ...
 network_topo.weights_matrix] = ut.calculate_all_graph_matrix( ...
    network_topo.radar_pos, network_topo.com_rad_CR, network_topo.numNodes);

%% ===== Target =====
NUM_TAR = 1;
target.initial_position = [1000, 1000];
target.speed = 20;
target.angle_degrees = 135 * ones(1, num_monte_carlo);

%% ===== Environment =====
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
env.rho = 0.0;

env.Sigma = [ env.range_var, env.rho*env.range_sd*env.doppler_sd; ...
              env.rho*env.range_sd*env.doppler_sd, env.doppler_var ];

env.pre_whit_L = inv(chol(env.Sigma,'upper')');   % L*Sigma*L'=I
env.PRE_WHITEN = PRE_WHITEN;

%% ===== Process model (CV) =====
dt = env.time_step;
F = [1 0 dt 0;
     0 1 0 dt;
     0 0 1  0;
     0 0 0  1];

Q = 1e-2 * [dt^4/4, 0, dt^3/2, 0;
            0, dt^4/4, 0, dt^3/2;
            dt^3/2, 0, dt^2, 0;
            0, dt^3/2, 0, dt^2];

%% ===== Results =====
Results = struct();
% Results.x_pred = cell(num_monte_carlo, TRACK_TIME*NUM_CPI_PER_MEA);
Results.estimations_DA = cell(num_monte_carlo, TRACK_TIME*NUM_CPI_PER_MEA); % But this is CA case
Results.P_post = cell(num_monte_carlo, TRACK_TIME*NUM_CPI_PER_MEA);
Results.burst_est = cell(num_monte_carlo, TRACK_TIME);
Results.gt_burst  = cell(num_monte_carlo, TRACK_TIME);

rng(RNG_SEED);

for mc = 1:num_monte_carlo
    fprintf('Monte Carlo run: %d/%d\n', mc, num_monte_carlo);

    angle_degrees = target.angle_degrees(mc);
    target.direction = [cos(angle_degrees*pi/180), sin(angle_degrees*pi/180)];
    target.true_params = [target.initial_position(1), target.initial_position(2), ...
                          target.speed*target.direction(1), target.speed*target.direction(2)];

    % trajectory
    target.target_position = zeros(NUM_TAR, NUM_CPI_PER_MEA*TRACK_TIME, 2);
    target.target_position(1,1,:) = target.initial_position;
    for t = 2:NUM_CPI_PER_MEA*TRACK_TIME
        target.target_position(1,t,:) = squeeze(target.target_position(1,t-1,:))' + target.speed*target.direction*time_step;
    end

    % measurements true + noisy
    range_true = zeros(NUM_TAR,network_topo.numNodes, NUM_CPI_PER_MEA*TRACK_TIME);
    doppler_true = zeros(NUM_TAR,network_topo.numNodes, NUM_CPI_PER_MEA*TRACK_TIME);
    meas_true = zeros(NUM_TAR,network_topo.numNodes, 2*NUM_CPI_PER_MEA*TRACK_TIME);

    [range_true, doppler_true, meas_true] = ut.gt_data_generation_tracking( ...
        range_true, doppler_true, meas_true, target, network_topo, env, NUM_CPI_PER_MEA*TRACK_TIME, NUM_TAR);

    [range_meas, doppler_meas, ~] = ut.add_measurement_noise( ...
        range_true, doppler_true, NUM_CPI_PER_MEA*TRACK_TIME, NUM_TAR, network_topo.numNodes, env);

    % init global state
    x = [1000; 1000; -14.1412; 14.1412];
    P = Q;

    for k = 1:TRACK_TIME
        for instance = 1:NUM_CPI_PER_MEA
            t = (k-1)*NUM_CPI_PER_MEA + instance;

            % predict
            x_pred = F*x;
            P_pred = F*P*F' + Q;

            % build stacked measurement z (2N x 1)
            N = network_topo.numNodes;
            z = zeros(2*N,1);
            for n = 1:N
                z(2*n-1) = range_meas(1,n,t);
                z(2*n)   = doppler_meas(1,n,t);
            end

            % centralized EKF update with optional whitening
            [x, P] = centralized_ekf_update_stack(x_pred, P_pred, z, network_topo, env, models_ut);
            
            % all_tracking_params{1,t} = x;
            Results.x_pred{mc,t} = x_pred;
            Results.x_post{mc,t} = x;
            Results.P_post{mc,t} = P;
        end

        Results.burst_est{mc,k} = x;
        t_idx = k*NUM_CPI_PER_MEA;
        Results.gt_burst{mc,k} = [squeeze(target.target_position(1,t_idx,:))', target.true_params(3), target.true_params(4)];

        if DEBUG
            fprintf('Burst %d done. Central EKF est=[%.2f %.2f %.2f %.2f]\n', k, x(1), x(2), x(3), x(4));
        end
    end

    % Plot target only (adapt if you want to overlay estimate)
    % fig_ut.plot_geometry_and_target(network_topo, squeeze(target.target_position(1,:,:)));
    fig_ut.plot_trajectory_and_network(target.target_position,Results.x_post,network_topo);

    if SAVE_LOG
        logS = struct();
        logS.network_topo = network_topo;
        logS.env = env;
        logS.target = target;
        logS.NUM_CPI_PER_MEA = NUM_CPI_PER_MEA;
        logS.TRACK_TIME = TRACK_TIME;
        logS.time_step = time_step;
        logS.PRE_WHITEN = PRE_WHITEN;
        logS.Results = Results;

        ut.write_exp_log(LOG_DIR, sprintf('Centralized_EKF_mc%d', mc), logS);
    end
end

%% ===== helper: centralized EKF update (stacked) =====
function [x_post, P_post] = centralized_ekf_update_stack(x_pred, P_pred, z, network_topo, env, models_ut)
    N = network_topo.numNodes;
    nx = numel(x_pred);
    mz = 2;

    % Build h(x) and H
    h = zeros(mz*N,1);
    H = zeros(mz*N, nx);

    for n = 1:N
        % Use models_ut.MeasureModel BUT avoid its internal whitening side effects:
        % We call the non-whiten local model via re-implementing "global" measurement:
        [hn, Hn] = global_meas_and_jacobian_no_whiten(x_pred, n, network_topo, env);

        h(2*n-1:2*n) = hn;
        H(2*n-1:2*n, :) = Hn;
    end

    innov = z - h;

    % Whitening in update (if enabled)
    if isfield(env,'PRE_WHITEN') && env.PRE_WHITEN
        L = env.pre_whit_L;              % 2x2
        W = kron(eye(N), L);             % (2N x 2N)
        innov = W * innov;
        H = W * H;
        R = eye(2*N);
    else
        R = kron(eye(N), env.Sigma);
    end

    S = H*P_pred*H' + R;
    K = P_pred*H' / S;

    x_post = x_pred + K*innov;
    P_post = (eye(nx) - K*H)*P_pred;
    P_post = 0.5*(P_post + P_post'); % sym
end

function [z, H] = global_meas_and_jacobian_no_whiten(x, idx, network_topo, env)
    % global measurement: range + doppler relative to radar idx
    xr = x(1); yr = x(2); vx = x(3); vy = x(4);
    x_i = network_topo.radar_pos(idx,1);
    y_i = network_topo.radar_pos(idx,2);

    dx = xr - x_i;
    dy = yr - y_i;
    r  = sqrt(dx^2 + dy^2);

    % Doppler model consistent with your models.m
    v_proj = vx*dx + vy*dy;
    fd = (2/env.lambda) * (v_proj / r);

    z = [r; fd];

    % Jacobian (no whitening)
    drdx = dx/r;
    drdy = dy/r;

    % fd = (2/lambda) * ( (vx*dx + vy*dy)/r )
    % Let g = vx*dx + vy*dy, then fd = c0 * g / r
    c0 = 2/env.lambda;
    g = v_proj;

    dfd_dx = c0 * ( (vx*r - g*(drdx)) / (r^2) );  % derivative w.r.t xr
    dfd_dy = c0 * ( (vy*r - g*(drdy)) / (r^2) );  % derivative w.r.t yr
    dfd_dvx = c0 * (dx / r);
    dfd_dvy = c0 * (dy / r);

    H = [ drdx,    drdy,    0,      0;
          dfd_dx,  dfd_dy,  dfd_dvx, dfd_dvy ];
end
