classdef utils
    methods (Static)
        %% Graph Topology related function: calculate adjcent matrix,  Incident matrx, weight matrix, and laplacian matrix
        function adj_mtx = calculate_adj_mtx(pos,communication_radius,numNodes)
            %TODO: To make it to handle the multiple communcation radius cases. 
            % Initialize the adjacency matrix
            adj_mtx = zeros(numNodes,numNodes);  
            % Calculate the adjacency matrix
             for i = 1:numNodes
                for j = i+1:numNodes % Since A is symmetric, compute only for j > i
                   if norm(pos(i,:) - pos(j,:)) <= communication_radius
                      adj_mtx(i,j) = 1;
                      adj_mtx(j,i) = 1; % Symmetric
                   end
               end
            end
        end 
        function inc_matrix = calculate_inc_mtx(adj_matrix,numNodes)
            numEdges = sum(adj_matrix(:)) / 2; % Each edge is counted twice in adjacency matrix
            inc_matrix = zeros(numNodes, numEdges);
            edge_index = 1;
            for i = 1:numNodes
                for j = i+1:numNodes
                    if adj_matrix(i, j) == 1
                        inc_matrix(i, edge_index) = 1;
                        inc_matrix(j, edge_index) = -1;
                        edge_index = edge_index + 1;
                    end
                end
            end
        end
        function weight_matrix = calculate_weight_mtx(degree_matrix, adj_matrix,numNodes)
            %maximum degree weights constants
            alpha = 1 / max(diag(degree_matrix));
        
            % Initialize final weight matrix
            weight_matrix = zeros(numNodes);
        
            % Compute weight matrix
            for i = 1:numNodes
                for j = 1:numNodes
                    if i == j
                        weight_matrix(i, j) = 1 - alpha * sum(adj_matrix(i, :));
                    elseif adj_matrix(i, j) == 1
                        weight_matrix(i, j) = alpha;
                    else
                        weight_matrix(i, j) = 0;
                    end
                end
            end
        end 
        
        function [adj_mtx, deg_mtx, lap_mtx, inc_mtx, wei_mtx] = calculate_all_graph_matrix(radar_pos, communication_radius, numNodes)
            ut = utils;
            adj_mtx = ut.calculate_adj_mtx(radar_pos,communication_radius,numNodes);
            deg_mtx = diag(sum(adj_mtx,2));
            lap_mtx = deg_mtx - adj_mtx;
            inc_mtx = ut.calculate_inc_mtx(adj_mtx,numNodes);
            wei_mtx = ut.calculate_weight_mtx(deg_mtx,adj_mtx,numNodes);
        end

        function neighbors = get_neighbors(adj_mtx, numNodes)
            % Initialize neighbors cell array
            neighbors = cell(numNodes, 1);
        
            % Fill the neighbors for each node
            for i = 1:numNodes
                neighbors{i} = find(adj_mtx(i, :) > 0); % Find indices of non-zero elements in row i
            end
        end

        function [prior_mean, prior_cov] = get_neighbors_cell(laplacian_matrix, num_tar, numNodes, estimated_params, state_cov)
            for i = 1: num_tar
                for n = 1: numNodes
                    current_neighbors = find(laplacian_matrix(n, :) ~= 0);
                    k = 0;
                    mu_state_neighbor = {};
                    sigma_state_neighbor = {};
                    for j = current_neighbors
                        k = k+1;
                        % Get prior parameters from neighbors
                        if size(estimated_params,2) == 1 % Check number of estimator params
                            mu_state_neighbor{k} = estimated_params;
                            % sigma_state_neighbor{k} = state_cov;

                        elseif size(estimated_params,2) == numNodes
                            mu_state_neighbor{k} = estimated_params(:,j);
                            % sigma_state_neighbor{k} = state_cov{j};

                        else
                            error('Size of estimated_params is not correct');
                        end

                        if size(state_cov,2) ==  4 % Check number of estimator params
                            % mu_state_neighbor{k} = estimated_params;
                            sigma_state_neighbor{k} = state_cov;

                        elseif size(state_cov,2) == numNodes
                            % mu_state_neighbor{k} = estimated_params(:,j);
                            sigma_state_neighbor{k} = state_cov{j};

                        else
                            error('Size of estimated_params is not correct');
                        end

                        prior_mean{n} = mu_state_neighbor;
                        prior_cov{n} = sigma_state_neighbor;
                    end
                    
                end
            end
        end

        %% Ground truth data generation for tracking
        function [range_true, doppler_true, measurements_true] = gt_data_generation_tracking(r_true, d_true, mea_true, target, network_topo, env, M, num_target)
            % Matrices for range and Doppler true data (This is a matrix of M x N) 
            % Range Measurement are stored in "relation position" way. 
            % Output: M is the time duration 
            % r_true : [num_target x numNodes x M]
            % d_true : [num_target x numNodes x M]
            % mea_true : [num_target x numNodes x 2M]
            % Input:
            % target.target_position : [num_target x M x 2]
            range_true = zeros(size(r_true));
            doppler_true = zeros(size(d_true));
            measurements_true = zeros(size(mea_true));
            
            % Radar_pos [# node, 2 (x,y)] (1) reshape from expand to 4 [num target(1), time(1), num node, 2]
            % (2) repmat to [num target (remate here), time(remat here), num node, 2]
            radar_pos_expand =  repmat(reshape(network_topo.radar_pos,[1,size(network_topo.radar_pos,1),1,size(network_topo.radar_pos,2)]),[num_target,1,M,1]);

            % target_pos [Num of target, time duration, 2] (1) reshape to [num target,1, time, num node (1),2]
            % (2) repmat to [num target, time (remat here), num doe (remat), 2]
            target_pos_expand = repmat(reshape(target.target_position,[num_target,1,size(target.target_position,2),size(target.target_position,3)]),[1,network_topo.numNodes,1,1]);
            relative_position = -(radar_pos_expand - target_pos_expand); % [num_target x numNodes x M x 2]
            range_true = vecnorm(relative_position,2,4); % [num_target x numNodes x M]
            doppler_true = reshape(reshape(relative_position,[],2)*[target.speed * target.direction]',[num_target,network_topo.numNodes,M])./(range_true.* env.lambda);
            
            measurements_true(:,:, 1:2:end) = range_true; % Odd index for range
            measurements_true(:,:, 2:2:end) = doppler_true; % Even index for Doppler
        end
        
        %% Add noise to the measurements
        function [range_with_error, doppler_with_error, measurements_all_with_error] = add_measurement_noise(range_true, doppler_true, M, NUM_TAR,numNodes, env)
            % Kron function is too expensive when total_measurements is large
            % Use mvnrnd to generate long noise vecotr directly
            % Add noise to the true measurements
            mu_r_d = [0 0]; % Mean for range and Doppler noise
            e_n = mvnrnd(mu_r_d, env.measurements_noise, NUM_TAR * M * numNodes); % Generate noise for all measurements at once
            % Reshape to [NUM_CPI_PER_MEA*TRACK_TIME , numNodes]
            range_noise_all   = reshape(e_n(:,1), NUM_TAR, numNodes, M);
            doppler_noise_all = reshape(e_n(:,2), NUM_TAR, numNodes, M);
            range_with_error = range_true + range_noise_all;
            doppler_with_error = doppler_true + doppler_noise_all;
           
            if env.PRE_WHITEN
                    L = env.pre_whit_L;
                    r0 = range_with_error;
                    f0 = doppler_with_error;
                    range_with_error   = L(1,1)*r0 + L(1,2)*f0;
                    doppler_with_error = L(2,1)*r0 + L(2,2)*f0;
            end
            measurements_all_with_error = zeros(NUM_TAR,numNodes, M*2);
            measurements_all_with_error(:,:, 1:2:end) = range_with_error; % Odd index for range
            measurements_all_with_error(:,:, 2:2:end) = doppler_with_error; % Even index for Doppler
        end
        
        function [range_with_error_cell,doppler_with_error_cell,numNodes_cell,radar_positions_cell] = pharse_measurements_tracking(laplacian_matrix,...
        range_with_error, doppler_with_error,range_with_error_cell, doppler_with_error_cell,num_tar, network_topo)
            % Remove mu_r, mu_d, sigma_r, sigma_dfrom input
            for i = 1: num_tar
                for n = 1: network_topo.numNodes
                    current_neighbors = find(laplacian_matrix(n, :) ~= 0);
                    k = 0;
                    % Sigma_big_1 = [];
                    % Sigma_big_2 = [];
                    range_with_error_1 =[];
                    doppler_with_error_1 = [];
                    radar_positions_1 = [];
                    mu_state_neighbor = {};
                    sigma_state_neighbor = {};
                    for j = current_neighbors
                        k = k+1;
                        % 
                        range_with_error_1(:,k) = range_with_error(i,j,:);
                        doppler_with_error_1(:,k) = doppler_with_error(i,j,:);
                        radar_positions_1(k,:) = network_topo.radar_pos(j,:);
                        numNodes_1 = length(current_neighbors);
                        % Store the values in cell arrays
                        range_with_error_cell{n} = range_with_error_1;
                        doppler_with_error_cell{n} = doppler_with_error_1;
                        numNodes_cell{n} = numNodes_1;
                        radar_positions_cell{n} = radar_positions_1;
                    end
                end
            end
        end 



        function pos = gen_ref_trajectory(N, bounds, dt, speed, opts)
        % Generate random walk groud truth trajecotry
        %
        %   pos = gen_walk_trajectory(N, bounds, dt, speed)
        %   pos = gen_walk_trajectory(N, bounds, dt, speed, opts)
        %
        % Inputs
        %   N      : number of samples (integer >= 2)
        %   bounds : [xmin xmax ymin ymax] or [xmin xmax; ymin ymax]
        %   dt     : sampling time [s], > 0
        %   speed  : walking speed [m/s], >= 0
        %   opts   : optional struct
        %       .seed            : RNG seed (default [])
        %       .randomizeShape  : true/false (default true)
        %       .M               : number of waypoints for shape (default 18, >= 6)
        %       .shapeJitterY    : normalized shape jitter (default 0.04)
        %       .shapeJitterX    : normalized shape jitter (default 0.012)
        %       .denseN          : dense samples for curve (default 4000)
        %
        %       % lateral "human" wandering (meters)
        %       .lateralStd      : lateral deviation std [m] (default 0.25)
        %       .lateralTau      : correlation time [s] (default 1.5)
        %
        %       % if desired length > curve length:
        %       .longMode        : "clip" or "wrap" (default "clip")
        %
        % Output
        %   pos    : [N x 2] trajectory positions [x,y] in meters
            ut = utils;
            if nargin < 5 || isempty(opts), opts = struct(); end

            % ---- validate inputs ----
            if ~(isscalar(N) && N == round(N) && N >= 2)
                error("N must be an integer >= 2.");
            end
            if ~(isscalar(dt) && isfinite(dt) && dt > 0)
                error("dt must be a positive scalar (seconds).");
            end
            if ~(isscalar(speed) && isfinite(speed) && speed >= 0)
                error("speed must be a nonnegative scalar (m/s).");
            end

            [xmin, xmax, ymin, ymax] = ut.parse_bounds(bounds);
            if ~(xmax > xmin && ymax > ymin)
                error("bounds must satisfy xmax > xmin and ymax > ymin.");
            end

            % ---- options ----
            seed           = ut.get_opt(opts, "seed", []);
            randomizeShape = ut.get_opt(opts, "randomizeShape", true);
            M              = ut.get_opt(opts, "M", 18);
            shapeJitterY   = ut.get_opt(opts, "shapeJitterY", 0.04);
            shapeJitterX   = ut.get_opt(opts, "shapeJitterX", 0.012);
            denseN         = ut.get_opt(opts, "denseN", 4000);

            lateralStd     = ut.get_opt(opts, "lateralStd", 0.25);
            lateralTau     = ut.get_opt(opts, "lateralTau", 1.5);
            longMode       = string(ut.get_opt(opts, "longMode", "clip"));

            if ~(isscalar(M) && M == round(M) && M >= 6)
                error("opts.M must be an integer >= 6.");
            end

            % ---- RNG ----
            if ~isempty(seed)
                rng(seed);
            else
                rng("shuffle");
            end

            % ---- 1) Build a smooth baseline curve inside bounds ----
            [x_dense, y_dense] = ut.make_base_curve(bounds, randomizeShape, M, shapeJitterX, shapeJitterY, denseN);

            % ---- arclength parameter (meters) ----
            s_dense = [0, cumsum(hypot(diff(x_dense), diff(y_dense)))];
            totalLen = s_dense(end);

            % ---- 2) Decide how long the observed trajectory should be ----
            ds = speed * dt;              % step length per sample
            L  = ds * (N-1);              % expected traveled distance in window

            if speed == 0 || ds == 0
                % stationary: pick a random point along the curve
                s0 = rand() * totalLen;
                x0 = interp1(s_dense, x_dense, s0, "linear");
                y0 = interp1(s_dense, y_dense, s0, "linear");
                pos = repmat([x0, y0], N, 1);
                return;
            end

            % choose a start arclength so that we get a segment of length ~L
            if L <= totalLen
                s0 = rand() * (totalLen - L);
                s_q = s0 + (0:N-1) * ds;
            else
                % requested longer than available curve length
                if longMode == "wrap"
                    s0 = rand() * totalLen;
                    s_q = s0 + (0:N-1) * ds;
                    s_q = mod(s_q, totalLen);
                else
                    % "clip": traverse the full curve (speed effectively reduced)
                    s_q = linspace(0, totalLen, N);
                end
            end

            % ---- 3) Interpolate base positions at s_q ----
            x_base = interp1(s_dense, x_dense, s_q, "linear");
            y_base = interp1(s_dense, y_dense, s_q, "linear");

            % ---- 4) Add realistic lateral wandering (correlated noise) ----
            % tangent from dense curve -> interpolate at s_q -> normal direction
            tx_dense = gradient(x_dense, s_dense);
            ty_dense = gradient(y_dense, s_dense);

            tx = interp1(s_dense, tx_dense, s_q, "linear");
            ty = interp1(s_dense, ty_dense, s_q, "linear");

            % normal = [-ty, tx] normalized
            nrm = hypot(tx, ty);
            nrm(nrm < 1e-12) = 1;
            nx = -ty ./ nrm;
            ny =  tx ./ nrm;

            % AR(1) correlated lateral offset: rho depends on dt and tau
            tau = max(lateralTau, 1e-3);
            rho = exp(-dt / tau);

            e = zeros(1, N);
            e(1) = randn();
            for k = 2:N
                e(k) = rho * e(k-1) + sqrt(1 - rho^2) * randn();
            end

            % scale to meters; also keep it reasonable relative to bounds
            maxStd = 0.08 * min(xmax - xmin, ymax - ymin);  % safety cap
            sigma = min(lateralStd, maxStd);
            offset = sigma * e;

            x = x_base + offset .* nx;
            y = y_base + offset .* ny;

            % clamp to bounds (simple safety)
            x = min(max(x, xmin), xmax);
            y = min(max(y, ymin), ymax);

            pos = [x(:), y(:)];
        end

        % ---------------- helpers ----------------
        function v = get_opt(opts, name, default)
            if isfield(opts, name) && ~isempty(opts.(name))
                v = opts.(name);
            else
                v = default;
            end
        end

        function [xmin, xmax, ymin, ymax] = parse_bounds(bounds)
            if isnumeric(bounds) && numel(bounds) == 4
                b = bounds(:).';
                xmin = b(1); xmax = b(2); ymin = b(3); ymax = b(4);
                return;
            end
            if isnumeric(bounds) && isequal(size(bounds), [2, 2])
                xmin = bounds(1,1); xmax = bounds(1,2);
                ymin = bounds(2,1); ymax = bounds(2,2);
                return;
            end
            error("bounds must be [xmin xmax ymin ymax] or [xmin xmax; ymin ymax].");
        end
        function [x_dense, y_dense] = make_base_curve(bounds, randomizeShape, M, jitterX, jitterY, denseN)
            ut = utils;
            [xmin, xmax, ymin, ymax] = ut.parse_bounds(bounds);

            % base reference in normalized space (similar to your figure)
            x_base = [0.00, 0.06, 0.14, 0.22, 0.30, 0.36, 0.42, 0.48, 0.52, 0.55, 0.58, 0.62, 0.68, 0.74, 0.82, 0.90, 0.96, 1.00];
            y_base = [0.00, 0.06, 0.13, 0.20, 0.28, 0.33, 0.38, 0.46, 0.52, 0.60, 0.67, 0.73, 0.78, 0.82, 0.90, 0.95, 0.985, 1.00];

            % make waypoints
            x_wp = linspace(0, 1, M);
            if randomizeShape
                x_wp(2:end-1) = x_wp(2:end-1) + jitterX * randn(1, M-2);
                x_wp = sort(x_wp);
                x_wp(1) = 0; x_wp(end) = 1;
            end

            y_wp = interp1(x_base, y_base, x_wp, "pchip");

            if randomizeShape
                w = sin(pi * x_wp).^1.3; % window -> 0 at ends
                y_wp = y_wp + jitterY * w .* randn(size(y_wp));

                % enforce monotone increasing y (like the reference)
                y_wp = cummax(y_wp);
                y_wp = y_wp - y_wp(1);
                if y_wp(end) <= 0
                    y_wp = linspace(0, 1, M);
                else
                    y_wp = y_wp / y_wp(end);
                end
                y_wp = min(max(y_wp, 0), 1);
            end

            % dense curve
            u = linspace(0, 1, denseN);
            v = interp1(x_wp, y_wp, u, "pchip");

            % scale to bounds
            x_dense = xmin + u * (xmax - xmin);
            y_dense = ymin + v * (ymax - ymin);
        end
        function save_json_log(root_dir, run_name, s)
        if ~exist(root_dir, "dir"); mkdir(root_dir); end
        ts = datetime("now","Format","yyyyMMdd_HHmmss");
        folder = fullfile(root_dir, sprintf("%s_%s", run_name, string(ts)));
        mkdir(folder);
        fn = fullfile(folder, "log.json");
        txt = jsonencode(s, "PrettyPrint", true);
        fid = fopen(fn, "w");
        fwrite(fid, txt, "char");
        fclose(fid);
        fprintf("  [log] %s\n", fn);
        end

        function save_mat_log(root_dir, run_name, log)
        if ~exist(root_dir, "dir"); mkdir(root_dir); end
        ts = datetime("now","Format","yyyyMMdd_HHmmss");
        folder = fullfile(root_dir, sprintf("%s_%s", run_name, string(ts)));
        mkdir(folder);
        fn = fullfile(folder, "log.mat");
        save(fn, "log");
        end

        function plot_setup(true_trajectory,network_topo)
            % Shape of the inputs:
            % true_trajectory: [Num target, track_time, 2]
            fig = figure;
            set(gcf,'Color','white');
            set(gca, 'FontName', 'Times New Roman');
            set(gca,'FontSize',15);
            hold on;
            h1 = plot(true_trajectory(1, :, 1), true_trajectory(1, :, 2), '--ob', 'LineWidth', 1.5, 'DisplayName', 'True Trajectory');
            h1.MarkerSize = 5;
            h1.MarkerIndices = 1:20:length(true_trajectory(1, :, 1));
            plot(network_topo.radar_pos(:,1), network_topo.radar_pos(:,2), 'r.', 'MarkerSize', 50, 'DisplayName', 'Sensor Nodes');
            % Plot communication link
            for n = 1:network_topo.numNodes 
                neighbors_idx = find(network_topo.laplacian_matrix(n,:) == -1).'; 
                % pairs = nchoosek(neighbors_idx,2);
                for j = 1: size(neighbors_idx,1)
                     if (n == 1 & j == 1)
                         plot([network_topo.radar_pos(n,1),network_topo.radar_pos(neighbors_idx(j),1)],...
                         [network_topo.radar_pos(n,2),network_topo.radar_pos(neighbors_idx(j),2)],...
                         '--k','LineWidth',1,'DisplayName','Communication Link');
                     end
                     plot([network_topo.radar_pos(n,1),network_topo.radar_pos(neighbors_idx(j),1)],...
                         [network_topo.radar_pos(n,2),network_topo.radar_pos(neighbors_idx(j),2)],...
                         '--k','LineWidth',1);
                end
            end

            hold off;
            % xlabel('Position x (m)');
            % ylabel('Position y (m)');
            % title('Target Trajectory');
            % legend('Location', 'best');
            set(gca,'xtick',[])
            set(gca,'ytick',[])
            objs = findobj(gca, '-property', 'DisplayName');
            objs = objs(arrayfun(@(h) ~isempty(h.DisplayName), objs));  
            legend(flipud(objs), 'Location', 'best');  
            grid off;box on;ax=gca;ax.LineWidth=1.5;
           
        end
     end
end