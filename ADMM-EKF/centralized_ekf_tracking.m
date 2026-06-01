
function [x_hist, P_hist, x_pred, P_pred] = centralized_ekf_tracking(range_meas, doppler_meas, network_topo, env, x0, P0, Q)
% CENTRALIZED_EKF_TRACKING
% Centralized EKF using all nodes' (range,doppler) at each time step.
%
% Inputs:
%   range_meas   : (T x N) range measurements (UNWHITEN)
%   doppler_meas : (T x N) doppler measurements (UNWHITEN)
%   network_topo : contains radar_pos (N x 2)
%   env          : contains time_step, Sigma, PRE_WHITEN, pre_whit_L
%   x0, P0       : initial GLOBAL state (4x1) and covariance (4x4)
%   Q            : process noise (4x4)
%
% Output:
%   out.x_hist   : (4 x T) posterior estimates
%   out.P_hist   : (4 x 4 x T) posterior cov
%   out.x_pred   : (4 x T) predicted
%   out.P_pred   : (4 x 4 x T) predicted

    models_ut = models();

    [T, N] = size(range_meas);
    assert(isequal(size(doppler_meas), [T N]), 'doppler_meas size mismatch');

    dt = env.time_step;
    F = [1 0 dt 0;
         0 1 0 dt;
         0 0 1  0;
         0 0 0  1];

    x = x0(:);
    P = P0;

    x_hist = zeros(4,T);
    P_hist = zeros(4,4,T);
    x_pred_hist = zeros(4,T);
    P_pred_hist = zeros(4,4,T);

    for t = 1:T
        % ---- predict ----
        x_pred = F*x;
        P_pred = F*P*F' + Q;

        x_pred_hist(:,t) = x_pred;
        P_pred_hist(:,:,t) = P_pred;

        % ---- sequential update over nodes (stable) ----
        for n = 1:N
            z = [range_meas(t,n); doppler_meas(t,n)];     % 2x1
            h = models_ut.MeasureModel(x_pred, n, network_topo, env);            % 2x1
            H = models_ut.MeasureModelJacobian(x_pred, n, network_topo, env);    % 2x4

            if env.PRE_WHITEN
                L = env.pre_whit_L;
                z = L*z;
                h = L*h;
                H = L*H;
                R = eye(2);
            else
                R = env.Sigma;
            end

            innov = z - h;
            S = H*P_pred*H' + R;

            K = (P_pred*H') / S;      % 4x2
            x_pred = x_pred + K*innov;
            P_pred = (eye(4) - K*H) * P_pred * (eye(4) - K*H)' + K*R*K'; % Joseph
        end

        % ---- commit ----
        x = x_pred;
        P = 0.5*(P_pred + P_pred');

        x_hist(:,t) = x;
        P_hist(:,:,t) = P;
    end

    % out = struct();
    x_hist = x_hist;
    P_hist = P_hist;
    x_pred = x_pred_hist;
    P_pred = P_pred_hist;
end
