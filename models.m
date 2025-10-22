classdef models
    methods(Static)
        function next_state = stateModel(state, time_step)
            %TODO: Add more motion models such as CA,CT, etc.
            % Constant velocity motion model
            F = [1, 0, time_step, 0; 
                 0, 1, 0, time_step; 
                 0, 0, 1,0; 
                 0, 0, 0,1]; 
            next_state = F * state;
        end
        function z = MeasureModel(state,idx,network_topo,env)
            x = state(1);
            y = state(2);
            v_x = state(3);
            v_y = state(4);

            dist_node_tar = @(x,y,node_pos) sqrt((node_pos(1)-x)^2 + (node_pos(2)-y)^2);
            % r = max(dist_node_tar(x,y,network_topo.radar_pos(idx,:)),1); % Avoid division by zero for stability
            r = max(dist_node_tar(x,y,network_topo.radar_pos(idx,:)),1); % Avoid division by zero for stability
            f =  ((v_x*(network_topo.radar_pos(idx,1)-x))+(v_y*(network_topo.radar_pos(idx,2)-y)))/(env.lambda*r);
            z = [r; f];

        end
        function z = LocalMeasureModel(state,idx,network_topo,env)
            % The state send here is already in local coordinate, so no need to subtract radar position
            x = state(1);
            y = state(2);
            v_x = state(3);
            v_y = state(4);

            dist_node_tar = @(x,y) sqrt((x)^2 + (y)^2);
            % r = max(dist_node_tar(x,y,network_topo.radar_pos(idx,:)),1); % Avoid division by zero for stability
            r = max(dist_node_tar(x,y),1); % Avoid division by zero for stability
            f =  -((v_x*(x))+(v_y*(y)))/(env.lambda*r);
            z = [r; f];

        end
        function z = LocalMeasureModelJacobian(state,idx,network_topo, env)
            x = state(1);
            y = state(2);
            v_x = state(3);
            v_y = state(4);
            dist_node_tar = @(x,y) max(sqrt((x)^2 + (y)^2),1); % Avoid division by zero for stability
            r = dist_node_tar(x,y);

            dr_dx = @(idx,x,y,v_x,v_y, network_topo) (x)/(r);
            dr_dy =  @(idx,x,y,v_x,v_y, network_topo)  (y)/(r);


            df_dx = @(idx,x,y,v_x,v_y, network_topo) ((-v_x*r)+...
                                                    ((v_x*(-x)+v_y*(-y))*...
                                                    (-x)/r))/...
                                                    (env.lambda*r^2);

            df_dy = @(idx,x,y,v_x,v_y, network_topo)  ((-v_y*r)+...
                                                    ((v_x*(-x)+v_y*(-y))*...
                                                    (-y)/r))/...
                                                    (env.lambda*r^2);

            df_dvx = @(idx,x,y,v_x,v_y, network_topo) -(x)/ (env.lambda*r);
            df_dvy = @(idx,x,y,v_x,v_y, network_topo) -(y)/ (env.lambda*r); 


            z = [dr_dx(idx,x,y,v_x,v_y, network_topo), dr_dy(idx,x,y,v_x,v_y, network_topo), 0, 0; 
                 df_dx(idx,x,y,v_x,v_y, network_topo), df_dy(idx,x,y,v_x,v_y, network_topo), ...
                df_dvx(idx,x,y,v_x,v_y, network_topo), df_dvy(idx,x,y,v_x,v_y, network_topo)];
            
            % Show input state
            % disp('Calculating Jacobian Matrix...');
            % fprintf('State3: %s\n', num2str(state(3)));
            % fprintf('State4: %s\n', num2str(state(4)));
            % disp(['Radar Index: ', num2str(idx)]);
            % disp(['Radar Position: ', num2str(network_topo.radar_pos(idx,:))]);
            % disp('Jacobian Matrix:');
            % disp(z);
        end 

        function z = MeasureModelJacobian(state,idx,network_topo, env)
            x = state(1);
            y = state(2);
            v_x = state(3);
            v_y = state(4);
            dist_node_tar = @(x,y,node_pos) max(sqrt((x-node_pos(1))^2 + (y-node_pos(2))^2),1); % Avoid division by zero for stability
            r = dist_node_tar(x,y,network_topo.radar_pos(idx,:));

            dr_dx = @(idx,x,y,v_x,v_y, network_topo) (x-network_topo.radar_pos(idx,1))/(r);
            dr_dy =  @(idx,x,y,v_x,v_y, network_topo)  (y-network_topo.radar_pos(idx,2))/(r);


            df_dx = @(idx,x,y,v_x,v_y, network_topo) ((-v_x*r)+...
                                                    ((v_x*(network_topo.radar_pos(idx,1)-x)+v_y*(network_topo.radar_pos(idx,2)-y))*...
                                                    (network_topo.radar_pos(idx,1)-x)/r))/...
                                                    (env.lambda*r^2);

            df_dy = @(idx,x,y,v_x,v_y, network_topo)  ((-v_y*r)+...
                                                    ((v_x*(network_topo.radar_pos(idx,1)-x)+v_y*(network_topo.radar_pos(idx,2)-y))*...
                                                    (network_topo.radar_pos(idx,2)-y)/r))/...
                                                    (env.lambda*r^2);

            df_dvx = @(idx,x,y,v_x,v_y, network_topo) -(x-network_topo.radar_pos(idx,1))/ (env.lambda*r);
            df_dvy = @(idx,x,y,v_x,v_y, network_topo) -(y-network_topo.radar_pos(idx,2))/ (env.lambda*r); 


            z = [dr_dx(idx,x,y,v_x,v_y, network_topo), dr_dy(idx,x,y,v_x,v_y, network_topo), 0, 0; 
                 df_dx(idx,x,y,v_x,v_y, network_topo), df_dy(idx,x,y,v_x,v_y, network_topo), ...
                df_dvx(idx,x,y,v_x,v_y, network_topo), df_dvy(idx,x,y,v_x,v_y, network_topo)];
            
            % Show input state
            % disp('Calculating Jacobian Matrix...');
            % fprintf('State3: %s\n', num2str(state(3)));
            % fprintf('State4: %s\n', num2str(state(4)));
            % disp(['Radar Index: ', num2str(idx)]);
            % disp(['Radar Position: ', num2str(network_topo.radar_pos(idx,:))]);
            % disp('Jacobian Matrix:');
            % disp(z);
        end 
        function printEkfIntermediates(old_ekf, z, dt, stepTag, i, network_topo, env)
            ekf = clone(old_ekf); % Copy to avoid modifying the original EKF state
            % 安全小常數，避免 r ~ 0 的除零
            epsr = 1e-9;

            % 取出必要函式與雜訊
            f  = ekf.StateTransitionFcn;
            % FJ = ekf.StateTransitionJacobianFcn;
            h  = ekf.MeasurementFcn;
            HJ = ekf.MeasurementJacobianFcn;
            Q  = ekf.ProcessNoise;
            R  = ekf.MeasurementNoise;

            % ---------- 預測（之前） ----------
            x_prior = ekf.State;
            P_prior = ekf.StateCovariance;

            % ---------- 做一次 predict（也會更新 ekf 內部狀態） ----------
            [x_pred, P_pred] = predict(ekf,dt);

            % ---------- 量測預估與雅可比 ----------
            % 用 x_pred 計算量測預估與 H
            z_pred = h(x_pred,i,network_topo,env);         % \hat z = h(x_{k|k-1})
            H = HJ(x_pred,i,network_topo,env);             % H(x_{k|k-1})

            % ---------- 創新與相關量 ----------
            y = z - z_pred;             % innovation
            S = H * P_pred * H.' + R;   % innovation covariance
            K = P_pred * H.' / S;       % Kalman gain

            % ---------- 列印（或可改為寫檔/記錄） ----------
            fprintf('==== Step Tag: %d at %d====\n', stepTag,i);
            fprintf('x_prior:\n'); disp(x_prior.');
            fprintf('P_prior (trace=%.6g):\n', trace(P_prior)); disp(P_prior);
            fprintf('R :\n'); disp(R);
            % 也可顯示 F 方便確認
            F = f(x_prior', dt); 
            fprintf('x_pred:\n'); disp(x_pred.');
            fprintf('F(x_prior):\n'); disp(F); 
            fprintf('P_pred (trace=%.6g):\n', trace(P_pred)); disp(P_pred);

            fprintf('z_meas:\n');  disp(z(:).');
            fprintf('z_pred = h(x_pred):\n'); disp(z_pred(:).');
            fprintf('H(x_pred):\n'); disp(H);
            fprintf('innovation y = z - z_pred:\n'); disp(y(:).');
            fprintf('S = H P H'' + R:\n'); disp(S);
            fprintf('K = P H'' S^{-1}:\n'); disp(K);

            % ---------- 做一次 correct（也會更新 ekf 內部狀態） ----------
            x_corr = correct(ekf, z,i,network_topo,env);
            fprintf('x_corr:\n'); disp(x_corr.');
            fprintf('x_diff:\n'); disp((x_corr-x_pred).');
            fprintf('P_corr (trace=%.6g):\n', trace(ekf.StateCovariance)); disp(ekf.StateCovariance);
            fprintf('============================\n\n');
        end


    end
end