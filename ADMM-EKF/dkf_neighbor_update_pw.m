function [x_post, P_post, dbg] = dkf_neighbor_update_pw( ...
    x_pred, P_pred, y_bar, idx_set, network_topo, env, models_ut)

    mz = 2;
    J  = numel(idx_set);
    nx = numel(x_pred);
    assert(numel(y_bar) == mz*J);

    % --- Whitening setup ---
    if isfield(env,'PRE_WHITEN') && env.PRE_WHITEN
        L = env.pre_whit_L;   % 2x2, satisfies L*Sigma*L' = I
        Rfilter = eye(mz);    % whitened noise covariance
    else
        L = eye(mz);
        Rfilter = env.Sigma;
    end

    sumInfo   = zeros(nx,nx);
    sumInnov  = zeros(nx,1);

    z_pred_all = zeros(mz*J,1);
    innov_all  = zeros(mz*J,1);
    H_all      = zeros(mz*J,nx);

    for a = 1:J
        j = idx_set(a);

        yj = y_bar((a-1)*mz+1 : a*mz);

        % Canonical unwhitened global model + Jacobian.
        [hj, Hj] = models_ut.MeasureModelAndJacobian( ...
            x_pred, j, network_topo, env);

        % Whiten both measurement and model consistently
        yW = L * yj;
        hW = L * hj;
        HW = L * Hj;

        innov = yW - hW;

        sumInfo  = sumInfo  + HW' * (Rfilter \ HW);
        sumInnov = sumInnov + HW' * (Rfilter \ innov);

        z_pred_all((a-1)*mz+1 : a*mz) = hW;
        innov_all((a-1)*mz+1 : a*mz)  = innov;
        H_all((a-1)*mz+1 : a*mz, :)   = HW;
    end

    R_bar = kron(eye(J), Rfilter);
    S = H_all * P_pred * H_all.' + R_bar;
    K = (P_pred * H_all.') / S;

    x_post = x_pred + K * innov_all;

    A = eye(nx) - K * H_all;
    P_post = A * P_pred * A.' + K * R_bar * K.';
    P_post = 0.5 * (P_post + P_post.');

    dbg = struct();
    dbg.z_pred = z_pred_all;
    dbg.innov  = innov_all;
    dbg.sumInfo  = sumInfo;
    dbg.sumInnov = sumInnov;
end
