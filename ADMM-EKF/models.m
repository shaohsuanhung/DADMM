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
            % f =  2*((v_x*(network_topo.radar_pos(idx,1)-x))+(v_y*(network_topo.radar_pos(idx,2)-y)))/(env.lambda*r);
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
            f =  ((v_x*(x))+(v_y*(y)))/(env.lambda*r);
            z = env.pre_whit_L*[r; f];
            % z = [r; f];

        end

        function z = LocalMeasureModel_dkf(state,idx,network_topo,env)
            % The state send here is already in local coordinate, so no need to subtract radar position
            x = state(1);
            y = state(2);
            v_x = state(3);
            v_y = state(4);

            dist_node_tar = @(x,y) sqrt((x)^2 + (y)^2);
            % r = max(dist_node_tar(x,y,network_topo.radar_pos(idx,:)),1); % Avoid division by zero for stability
            r = max(dist_node_tar(x,y),1); % Avoid division by zero for stability
            f =  ((v_x*(x))+(v_y*(y)))/(env.lambda*r);
            % z = env.pre_whit_L*[r; f];
            z = [r; f];

        end

        function z = LocalMeasureModel_refactor(state)
            % The state send here is already in local coordinate, so no need to subtract radar position
            x = state(1);
            y = state(2);
            v_x = state(3);
            v_y = state(4);

            dist_node_tar = @(x,y) sqrt((x)^2 + (y)^2);
            % r = max(dist_node_tar(x,y,network_topo.radar_pos(idx,:)),1); % Avoid division by zero for stability
            r = max(dist_node_tar(x,y),1); % Avoid division by zero for stability

            % Different lambda for refactored version
            f =  ((v_x*(x))+(v_y*(y)))/(3e-2*r);
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


            % df_dx = @(idx,x,y,v_x,v_y, network_topo) ((-v_x*r)+...
            %                                         ((v_x*(-x)+v_y*(-y))*...
            %                                         (-x)/r))/...
            %                                         (env.lambda*r^2);

            % df_dy = @(idx,x,y,v_x,v_y, network_topo)  ((-v_y*r)+...
            %                                         ((v_x*(-x)+v_y*(-y))*...
            %                                         (-y)/r))/...
            %                                         (env.lambda*r^2);

            % df_dvx = @(idx,x,y,v_x,v_y, network_topo) -(x)/ (env.lambda*r);
            % df_dvy = @(idx,x,y,v_x,v_y, network_topo) -(y)/ (env.lambda*r); 

            
            df_dx = @(idx,x,y,v_x,v_y, network_topo) (v_x*r^2-(v_x*x+v_y*y)*x)/...
                                                    (env.lambda*r^3);

            df_dy = @(idx,x,y,v_x,v_y, network_topo)  (v_y*r^2-(v_x*x+v_y*y)*y)/...
                                                    (env.lambda*r^3);

            df_dvx = @(idx,x,y,v_x,v_y, network_topo) (x)/ (env.lambda*r);
            df_dvy = @(idx,x,y,v_x,v_y, network_topo) (y)/ (env.lambda*r); 


            z = [dr_dx(idx,x,y,v_x,v_y, network_topo), dr_dy(idx,x,y,v_x,v_y, network_topo), 0, 0; 
                 df_dx(idx,x,y,v_x,v_y, network_topo), df_dy(idx,x,y,v_x,v_y, network_topo), ...
                df_dvx(idx,x,y,v_x,v_y, network_topo), df_dvy(idx,x,y,v_x,v_y, network_topo)];
            
            z = env.pre_whit_L * z;
            % z = [r; f];
            % Show input state
            % disp('Calculating Jacobian Matrix...');
            % fprintf('State3: %s\n', num2str(state(3)));
            % fprintf('State4: %s\n', num2str(state(4)));
            % disp(['Radar Index: ', num2str(idx)]);
            % disp(['Radar Position: ', num2str(network_topo.radar_pos(idx,:))]);
            % disp('Jacobian Matrix:');
            % disp(z);
        end 
        
        function H = LocalMeasureModelJacobian_dkf(state,idx,network_topo, env)
            x = state(1);y = state(2);
            x_i = network_topo.radar_pos(idx,1); y_i = network_topo.radar_pos(idx,2);
            dx = x; dy = y;
            v_x = state(3);
            v_y = state(4);
            dist_node_tar = @(x,y) max(sqrt((x)^2 + (y)^2),1); % Avoid division by zero for stability
            r = dist_node_tar(dx,dy);
                
            dr_dx = @(idx,x,y,v_x,v_y, network_topo) (x)/(r);
            dr_dy =  @(idx,x,y,v_x,v_y, network_topo)  (y)/(r);
            df_dx = @(idx,x,y,v_x,v_y, network_topo) 2*(v_x*r^2-(v_x*x+v_y*y)*x)/...
                                                    (env.lambda*r^3);
            df_dy = @(idx,x,y,v_x,v_y, network_topo)  2*(v_y*r^2-(v_x*x+v_y*y)*y)/...
                                                    (env.lambda*r^3);
            df_dvx = @(idx,x,y,v_x,v_y, network_topo) 2*(x)/ (env.lambda*r);
            df_dvy = @(idx,x,y,v_x,v_y, network_topo) 2*(y)/ (env.lambda*r); 

            H = [dr_dx(idx,dx,dy,v_x,v_y, network_topo), dr_dy(idx,dx,dy,v_x,v_y, network_topo), 0, 0; 
                 df_dx(idx,dx,dy,v_x,v_y, network_topo), df_dy(idx,dx,dy,v_x,v_y, network_topo), ...
                df_dvx(idx,dx,dy,v_x,v_y, network_topo), df_dvy(idx,dx,dy,v_x,v_y, network_topo)];
            
            % z = env.pre_whit_L * z;
            % Show input state
            % disp('Calculating Jacobian Matrix...');
            % fprintf('State3: %s\n', num2str(state(3)));
            % fprintf('State4: %s\n', num2str(state(4)));
            % disp(['Radar Index: ', num2str(idx)]);
            % disp(['Radar Position: ', num2str(network_topo.radar_pos(idx,:))]);
            % disp('Jacobian Matrix:');
            % disp(z);
            
            % global measurement: range + doppler relative to radar idx
            % xr = state(1); yr = state(2); vx = state(3); vy = state(4);
            % x_i = network_topo.radar_pos(idx,1);
            % y_i = network_topo.radar_pos(idx,2);
            % 
            % dx = xr - x_i;
            % dy = yr - y_i;
            % r  = sqrt(dx^2 + dy^2);
            % 
            % Doppler model consistent with your models.m
            % v_proj = vx*dx + vy*dy;
            % fd = (2/env.lambda) * (v_proj / r);
            % 
            % z = [r; fd];
            % 
            % Jacobian (no whitening)
            % drdx = dx/r;
            % drdy = dy/r;
            % 
            % fd = (2/lambda) * ( (vx*dx + vy*dy)/r )
            % Let g = vx*dx + vy*dy, then fd = c0 * g / r
            % c0 = 2/env.lambda;
            % g = v_proj;
            % 
            % dfd_dx = c0 * ( (vx*r - g*(drdx)) / (r^2) );  % derivative w.r.t xr
            % dfd_dy = c0 * ( (vy*r - g*(drdy)) / (r^2) );  % derivative w.r.t yr
            % dfd_dvx = c0 * (dx / r);
            % dfd_dvy = c0 * (dy / r);
            % 
            % H = [ drdx,    drdy,    0,      0;
            %       dfd_dx,  dfd_dy,  dfd_dvx, dfd_dvy ];

        end 

        function z = LocalMeasureModelJacobian_refactor(state)
            x = state(1);
            y = state(2);
            v_x = state(3);
            v_y = state(4);
            dist_node_tar = @(x,y) max(sqrt((x)^2 + (y)^2),1); % Avoid division by zero for stability
            r = dist_node_tar(x,y);

            dr_dx = @(x,y,v_x,v_y ) (x)/(r);
            dr_dy =  @(x,y,v_x,v_y)  (y)/(r);


            df_dx = @(x,y,v_x,v_y) ((-v_x*r)+...
                                                    ((v_x*(-x)+v_y*(-y))*...
                                                    (-x)/r))/...
                                                    (3e-2*r^2);

            df_dy = @(x,y,v_x,v_y)  ((-v_y*r)+...
                                                    ((v_x*(-x)+v_y*(-y))*...
                                                    (-y)/r))/...
                                                    (3e-2*r^2);

            df_dvx = @(x,y,v_x,v_y) -(x)/ (3e-2*r);
            df_dvy = @(x,y,v_x,v_y) -(y)/ (3e-2*r); 


            z = [dr_dx(x,y,v_x,v_y), dr_dy(x,y,v_x,v_y), 0, 0; 
                 df_dx(x,y,v_x,v_y), df_dy(x,y,v_x,v_y), ...
                df_dvx(x,y,v_x,v_y), df_dvy(x,y,v_x,v_y)];
            
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



            FF = [1, 0, 1e-2, 0; 
                 0, 1, 0, 1e-2; 
                 0, 0, 1,0; 
                 0, 0, 0,1]; 
            % ---------- 列印（或可改為寫檔/記錄） ----------
            fprintf('==== Step Tag: %d at %d====\n', stepTag,i);
            fprintf('x_prior:\n'); disp(x_prior.');
            fprintf('P_prior (trace=%.6g):\n', trace(P_prior)); disp(P_prior);
            fprintf('R :\n'); disp(R);
            % 也可顯示 F 方便確認
            F = f(x_prior', dt); 
            fprintf('x_pred:\n'); disp(x_pred.');
            fprintf('F(x_prior):\n'); disp(F); 
            fprintf('Q :\n'); disp(Q);
            fprintf('FF*P_pred*FF^T:\n'); disp(FF*P_prior*FF.');
            fprintf('P_pred (trace=%.6g):\n', trace(P_pred)); disp(P_pred);

            fprintf('z_meas:\n');  disp(z(:).');
            fprintf('z_pred = h(x_pred):\n'); disp(z_pred(:).');
            fprintf('H(x_pred):\n'); disp(H);
            fprintf('innovation y = z - z_pred:\n'); disp(y(:).');
            fprintf(' H P H'':\n'); disp(H * P_pred * H.');
            fprintf('S = H P H'' + R:\n'); disp(S);
            fprintf('K = P H'' S^{-1}:\n'); disp(K);
            fprintf('I-K*H:\n'); disp(eye(size(P_pred)) - K * H);

            % ---------- 做一次 correct（也會更新 ekf 內部狀態） ----------
            x_corr = correct(ekf, z,i,network_topo,env);
            fprintf('x_corr:\n'); disp(x_corr.');
            fprintf('x_diff:\n'); disp((x_corr-x_pred).');
            fprintf('P_corr (trace=%.6g):\n', trace(ekf.StateCovariance)); disp(ekf.StateCovariance);
            fprintf('============================\n\n');
        end
        function global_states = local2global(radar_positions, estimated_states)
            % Range and Doppler CPI measurement from local to global coordinate
            % Regardless of number of nodes
            % Input:x
            %   radar_positions: Nx2 matrix, each row is the (x,y) position of a radar node
            %   estimated_states: Nx4 matrix, each row is the (x,y,vx,vy) state estimated in local coordinate
            % Output:
            %   global_states: Nx4 matrix, each row is the (x,y,vx,vy) state in global coordinate
            num_nodes = size(radar_positions, 1);
            global_states = zeros(size(estimated_states));
            for idx = 1:num_nodes
                radar_pos = radar_positions(idx, :);
                local_state = estimated_states(idx, :);
                global_x = local_state(1) + radar_pos(1);
                global_y = local_state(2) + radar_pos(2);
                global_vx = local_state(3);
                global_vy = local_state(4);
                global_states(idx, :) = [global_x, global_y, global_vx, global_vy];
            end 
        end


        function checkJacobian()
            env.lambda = 3e8/10e9;
            env.pre_whit_L = eye(2);

            state = [1000; -500; -14; 14];  % example local state
            eps = 1e-6;

            % analytic
            H = models.LocalMeasureModelJacobian(state, 1, [], env);

            % numeric
            z0 = models.LocalMeasureModel(state, 1, [], env);
            Hn = zeros(2,4);
            for k=1:4
                s = state;
                s(k) = s(k) + eps;
                zk = models.LocalMeasureModel(s, 1, [], env);
                Hn(:,k) = (zk - z0)/eps;
            end

            disp("Analytic H:"); disp(H);
            disp("Numeric  H:"); disp(Hn);
            disp("Max abs diff:"); disp(max(abs(H(:)-Hn(:))));
        end

    end
end