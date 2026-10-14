function [x_post, P_post] = centralized_ekf_update_stack(x_pred, P_pred, z, network_topo, env, models_ut)
    N = network_topo.numNodes;
    nx = numel(x_pred);
    mz = 2;

    % Build h(x) and H
    h = zeros(mz*N,1);
    H = zeros(mz*N, nx);

    for n = 1:N
        [hn, Hn] = models_ut.MeasureModelAndJacobian( ...
            x_pred, n, network_topo, env);

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
    A = eye(nx) - K*H;
    P_post = A*P_pred*A' + K*R*K';
    P_post = 0.5 * (P_post + P_post');
end
