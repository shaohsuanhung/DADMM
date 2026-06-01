function [x_post, P_post, dbg] = dkf_neighbor_update_pw( ...
    x_pred, P_pred, y_bar, idx_set, network_topo, env, models_ut)

    mz = 2;
    J  = numel(idx_set);
    nx = numel(x_pred);
    assert(numel(y_bar) == mz*J);

    % --- Whitening setup ---
    if isfield(env,'PRE_WHITEN') && env.PRE_WHITEN
        L = env.pre_whit_L;   % 2x2, satisfies L*Sigma*L' = I
        Rinv = eye(mz);       % because whitened noise covariance is I
    else
        L = eye(mz);
        R = env.Sigma; % 2x2
        Rinv = inv(R);
    end

    Ppred_inv = inv(P_pred);
    sumInfo   = zeros(nx,nx);
    sumInnov  = zeros(nx,1);

    z_pred_all = zeros(mz*J,1);
    innov_all  = zeros(mz*J,1);

    for a = 1:J
        j = idx_set(a);

        yj = y_bar((a-1)*mz+1 : a*mz);

        % Unwhitened model + Jacobian
        hj = models_ut.MeasureModel(x_pred, j, network_topo, env);             % 2x1
        Hj = models_ut.MeasureModelJacobian(x_pred, j, network_topo, env);     % 2xnx

        % Whiten both measurement and model consistently
        yW = L * yj;
        hW = L * hj;
        HW = L * Hj;

        innov = yW - hW;

        sumInfo  = sumInfo  + HW' * Rinv * HW;
        sumInnov = sumInnov + HW' * Rinv * innov;

        z_pred_all((a-1)*mz+1 : a*mz) = hW;
        innov_all((a-1)*mz+1 : a*mz)  = innov;
    end

    P_post = inv(Ppred_inv + sumInfo);
    x_post = x_pred + P_post * sumInnov;

    dbg = struct();
    dbg.z_pred = z_pred_all;
    dbg.innov  = innov_all;
    dbg.sumInfo  = sumInfo;
    dbg.sumInnov = sumInnov;
end