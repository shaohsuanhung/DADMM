function [x_post, P_post, dbg] = dkf_neighbor_update( ...
    x_pred, P_pred, y_bar, idx_set, network_topo, env, models_ut)
%DKF_NEIGHBOR_UPDATE Neighbor-augmented EKF update (LKF-II style)
%
% Measurement stacking (lecture p18):
%   y_bar = [y_j]^T,   H_bar = [H_j]^T,   R_bar = blkdiag(R_j)
% for j in (N_n ∪ {n}) using idx_set ordering.
%
% Supports pre-whitening:
%   yW = (I ⊗ L) y,  hW = (I ⊗ L) h,  HW = (I ⊗ L) H
%   with L*Sigma*L' = I.

nx = numel(x_pred);
mz = 2;
J  = numel(idx_set);

assert(numel(y_bar) == mz*J, "y_bar length mismatch.");

% Build stacked h_bar and H_bar
h_bar = zeros(mz*J, 1);
H_bar = zeros(mz*J, nx);

for a = 1:J
    j = idx_set(a);

    hj = models_ut.LocalMeasureModel_dkf(x_pred, j, network_topo, env);            % 2x1 (unwhitened physical)
    Hj = models_ut.LocalMeasureModelJacobian_dkf(x_pred, j, network_topo, env);    % 2x4 (unwhitened)

    h_bar((a-1)*mz+1:a*mz) = hj;
    H_bar((a-1)*mz+1:a*mz, :) = Hj;
end

innov = y_bar - h_bar;

% Whitening handling
if isfield(env,'PRE_WHITEN') && env.PRE_WHITEN
    L = env.pre_whit_L;                    % 2x2
    LW = kron(eye(J), L);                  % (2J)x(2J)
    yW = LW * y_bar;
    hW = LW * h_bar;
    HW = LW * H_bar;

    innovW = yW - hW;

    % R_bar becomes I
    S = HW * P_pred * HW.' + eye(mz*J);
    K = (P_pred * HW.') / S;

    x_post = x_pred + K * innovW;
    P_post = (eye(nx) - K*HW) * P_pred * (eye(nx) - K*HW).' + K*K.'; % Joseph for stability
else
    R = env.Sigma;                         % 2x2
    R_bar = kron(eye(J), R);

    S = H_bar * P_pred * H_bar.' + R_bar;
    K = (P_pred * H_bar.') / S;

    x_post = x_pred + K * innov;
    P_post = (eye(nx) - K*H_bar) * P_pred * (eye(nx) - K*H_bar).' + K*R_bar*K.'; % Joseph
end

% P_post = 0.5*(P_post + P_post.');

dbg = struct();
dbg.h_bar = h_bar;
dbg.H_bar = H_bar;
dbg.innov = innov;
end
