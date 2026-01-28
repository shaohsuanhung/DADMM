function [x_pred, P_pred] = dkf_predict_cv(x_prev, P_prev, dt, Q)
%DKF_PREDICT_CV Constant-velocity prediction
% x_{k|k-1} = F x_{k-1|k-1}
% P_{k|k-1} = F P F' + Q

F = [1 0 dt 0;
     0 1 0 dt;
     0 0 1  0;
     0 0 0  1];

x_pred = F * x_prev;
P_pred = F * P_prev * F.' + Q;
P_pred = 0.5*(P_pred + P_pred.'); % 保對稱
end
