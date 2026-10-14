function [x_post, P_post, model] = tracking_centralized_ekf_update(x_pred, P_pred, z, network_topo, env, models_ut)
    N = network_topo.numNodes;
    nx = numel(x_pred);
    mz = 2;

    % The returned model remains unwhitened and fixed at the predicted state.
    model = models_ut.linearize_measurements(x_pred, network_topo, env);
    h = model.value(:);
    H = zeros(mz*N, nx);

    for n = 1:N
        H(2*n-1:2*n, :) = model.H(:,:,n);
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
