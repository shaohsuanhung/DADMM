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

        % ---- 方便除錯用的「上一個 step」內部計算結果 ----
        lastF           % 上一次 predict 使用的狀態 Jacobian F
        lastH           % 上一次 update 使用的量測 Jacobian H
        last_x_pred     % 預測後、更新前的狀態 x_pred
        last_P_pred     % 預測後、更新前的協方差 P_pred
        last_z_pred     % 預測量測 z_pred
        last_y          % innovation: y = z - z_pred
        last_S          % innovation covariance S
        last_K          % Kalman gain

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

            % 初始化除錯用的暫存變數
            obj.lastF       = [];
            obj.lastH       = [];
            obj.last_x_pred = [];
            obj.last_P_pred = [];
            obj.last_z_pred = [];
            obj.last_y      = [];
            obj.last_S      = [];
            obj.last_K      = [];
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

             % ---- 紀錄這一步 prediction 的內部資訊 ----
            obj.lastF       = F;
            obj.last_x_pred = x_pred;
            obj.last_P_pred = P_pred;

            % update 部分的東西先清空，避免誤用舊值
            obj.last_z_pred = [];
            obj.last_y      = [];
            obj.last_S      = [];
            obj.last_K      = [];
            obj.lastH       = [];

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


            % ---- 紀錄這一步 update 的內部資訊 ----
            obj.last_z_pred = z_pred;
            obj.last_y      = y;
            obj.last_S      = S;
            obj.last_K      = K;
            obj.lastH       = H;
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

        function printStepInfo(obj)
            %PRINTSTEPINFO 將最近一次 predict / update 的內部計算印出來
            fprintf('================ EKF Step Info ================\n');

            fprintf('Current state x (after update):\n');
            disp(obj.State);
            fprintf('Current covariance P (after update):\n');
            disp(obj.StateCovariance);

            if ~isempty(obj.last_x_pred)
                fprintf('-----------------------------------------------\n');
                fprintf('Prediction (time update):\n');
                fprintf('x_pred (before measurement update):\n');
                disp(obj.last_x_pred);
                fprintf('P_pred (before measurement update):\n');
                disp(obj.last_P_pred);
                fprintf('F (state Jacobian used in prediction):\n');
                disp(obj.lastF);
            else
                fprintf('No prediction info stored yet.\n');
            end

            if ~isempty(obj.last_z_pred)
                fprintf('-----------------------------------------------\n');
                fprintf('Measurement update:\n');
                fprintf('z_pred (predicted measurement):\n');
                disp(obj.last_z_pred);
                fprintf('Innovation y = z - z_pred:\n');
                disp(obj.last_y);
                fprintf('Innovation covariance S:\n');
                disp(obj.last_S);
                fprintf('Kalman gain K:\n');
                disp(obj.last_K);
                fprintf('H (measurement Jacobian):\n');
                disp(obj.lastH);
            else
                fprintf('No measurement update info stored yet.\n');
            end

            fprintf('===============================================\n');
        end
    end
end
