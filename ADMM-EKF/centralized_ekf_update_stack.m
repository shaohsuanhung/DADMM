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