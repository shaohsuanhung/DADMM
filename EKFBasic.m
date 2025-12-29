classdef EKFBasic < handle
    %EKFBasic 
    %
    %   SSM：
    %       x_k = f(x_{k-1}, u_{k-1}) + w,   w ~ N(0, Q)
    %       z_k = h(x_k) + v,                v ~ N(0, R)
    %
    %   Example
    %       ekf = EKFBasic(f_fun, h_fun, F_jac, H_jac, Q, R, x0, P0);
    %       ekf.predict(u_km1);       % (prediction)
    %       ekf.update(z_k);          % (update)
    %
    %   One step example:
    %       ekf.step(z_k, u_km1);

    properties
        f_fun   % x_pred = f_fun(x, u)
        f_mat   % F = f_mat(x, u)   (df/dx)
        h_fun   % z_pred = h_fun(x)
        H_jac   % H = H_jac(x)      (dh/dx)

        Q       % process noise covariance
        R       % measurement noise covariance

        State      % n×1 state estimate
        StateCovariance       % n×n covariance
    end

    methods
        function obj = EKFBasic(f_fun, f_mat, h_fun, H_jac, Q, R, x0, P0)
            obj.f_fun = f_fun;
            obj.f_mat = f_mat;
            obj.h_fun = h_fun;
            obj.H_jac = H_jac;

            obj.Q = Q;
            obj.R = R;

            obj.State = x0;
            obj.StateCovariance= P0;
        end

        % ======================
        %   Prediction 
        % ======================
        function [x_pred, P_pred, F] = predict(obj, dt)
            %PREDICT EKF (process / motion model)
            %
            %   Input:
            %       dt     : time step
            %
            %   Output:
            %       x_pred : predicted state
            %       P_pred : predicted state covariance
            %       F      : dynamics model matrix 
            x_prev = obj.State;
            P_prev = obj.StateCovariance;
            x_pred = obj.f_fun(x_prev, dt);

            %  Jacobian
            F = obj.f_mat;
            P_pred = F * P_prev * F.' + obj.Q;

            obj.State= x_pred;
            obj.StateCovariance = P_pred;

        end

        % ======================
        %   Update
        % ======================
        function [x, P, K, y, z_pred, S, H] = correct(obj, z)
            %UPDATE EKF  (measurement update)
            %
            %   Input:
            %       z : m×1  vector of measurement
            %
            %   Output:
            %       x      : Corrected state estimate
            %       P      : Corrected state covariance
            %       K      : Kalman gain
            %       y      : innovation (z - z_pred)
            %       z_pred : predicted measurement
            %       S      : innovation covariance
            %       H      : Linearized measurement model matrix (under the current state estimate)

            x_pred = obj.State;
            P_pred = obj.StateCovariance;

    
            z_pred = obj.h_fun(x_pred);

            % innovation
            y = z - z_pred;

            % Jacobian
            H = obj.H_jac(x_pred);

            % innovation covariance
            S = H * P_pred * H.' + obj.R;

            % Kalman gain
            K = P_pred * H.' / S;

            % update state
            x = x_pred + K * y;

            I = eye(size(P_pred));
            P = (I - K * H) * P_pred;

            obj.State = x;
            obj.StateCovariance = P;
        end

        % ======================
        %   One step (Prediction + Update)
        % ======================
        function [x, P, K, y, z_pred, S, F, H] = step(obj, z, u)

            if nargin < 3
                u = [];
            end
            [~, ~, F] = obj.predict(u);
            [x, P, K, y, z_pred, S, H] = obj.correct(z);
        end


        function [x, P] = getState(obj)
            x = obj.State;
            P = obj.StateCovariance;
        end

        function reset(obj, x0, P0)
            obj.State = x0;
            obj.StateCovariance = P0;
        end
    end
end
