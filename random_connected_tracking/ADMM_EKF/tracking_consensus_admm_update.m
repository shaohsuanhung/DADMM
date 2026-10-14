function [x_bar, x_nodes, P_nodes, info] = tracking_consensus_admm_update( ...
    x_pred_nodes, P_pred_nodes, range_meas, doppler_meas, t, ...
    network_topo, neighbors, env, models_ut, ut, objective_type, ...
    measurement_mode, PRE_WHITEN, max_iter, abs_tolerance, ...
    rel_tolerance, c_penalty, rho_update_interval, rho_balance_mu, ...
    rho_scale, verbose_every, DEBUG)

    N  = network_topo.numNodes;
    nx = numel(x_pred_nodes{1});

    if string(measurement_mode) ~= "self_only"
        error('ADMM_EKF:ConsensusObjectivePartition', ...
            ['Graph-wide consensus ADMM requires self_only measurements ', ...
             'so every physical measurement is counted exactly once.']);
    end
    if rho_update_interval < 1 || rho_update_interval ~= floor(rho_update_interval) || ...
            rho_balance_mu <= 1 || rho_scale <= 1
        error('ADMM_EKF:RhoAdaptationConfig', ...
            ['rho_update_interval must be a positive integer; ', ...
             'rho_balance_mu and rho_scale must be greater than one.']);
    end

    ADMM = ut.Initialized_ADMM(network_topo);
    ADMM.max_iter  = max_iter;
    ADMM.tolerance = abs_tolerance;
    ADMM.c_penalty = c_penalty(:).';

    ADMM.initial_values = zeros(nx, N);
    for n = 1:N
        ADMM.initial_values(:,n) = x_pred_nodes{n};
    end

    [prior_mean_common, prior_cov_common] = ...
        common_predicted_prior(x_pred_nodes, P_pred_nodes);

    % Sum_n (1/N)*prior(x) reconstructs one common predicted prior.
    prior_mean = {prior_mean_common};
    prior_cov  = {N * prior_cov_common};

    % One common predicted anchor defines one fixed affine global problem.
    % Every ADMM iterate uses this same model, including the covariance update.
    fixed_model = models_ut.linearize_measurements(prior_mean_common, network_topo, env);
    local_models = cell(1, N);
    if PRE_WHITEN
        whitener = env.pre_whit_L;
    else
        whitener = eye(2);
    end
    range_local = cell(1, N);
    doppler_local = cell(1, N);
    numNodes_local = zeros(1, N);
    for n = 1:N
        [range_local{n}, doppler_local{n}, ~, ...
            numNodes_local(n)] = local_measurement_for_admm( ...
                n, range_meas, doppler_meas, t, network_topo, measurement_mode);
        if numNodes_local(n) ~= 1
            error('ADMM_EKF:LocalMeasurementCount', ...
                'Node %d must own exactly one measurement.', n);
        end
        local_models{n} = struct('anchor', fixed_model.anchor, ...
            'value', fixed_model.value(:,n), 'H', fixed_model.H(:,:,n));
    end

    % Edge-variable initialization.
    for n = 1:N
        for j = neighbors{n}
            z0 = 0.5 * (ADMM.initial_values(:,n) + ADMM.initial_values(:,j));
            ADMM.update_z{n}(:,j)      = z0;
            ADMM.update_z_prev{n}(:,j) = z0;
            ADMM.Nu{n}(:,j)            = zeros(nx,1);
            ADMM.Nu_prev{n}(:,j)       = zeros(nx,1);
        end
    end

    all_estimations = ADMM.initial_values;
    iteration = 0;
    stop_reason = "max_iter";

    ADMM.primal_residual_all = zeros(1, max_iter);
    ADMM.dual_residual_all = zeros(1, max_iter);
    ADMM.eps_pri_all = zeros(1, max_iter);
    ADMM.eps_dual_all = zeros(1, max_iter);
    ADMM.c_penalty_all = zeros(nx, max_iter);
    ADMM.all_estimations_every_iter = zeros(nx, N, max_iter);
    ADMM.primal_residual_by_para = cell(1, max_iter);
    ADMM.dual_residual_by_para = cell(1, max_iter);

    fmincon_exitflag = zeros(N,1);
    fmincon_failure_count = 0;
    fmincon_max_iterations = 0;

    num_directed_edges = sum(cellfun(@numel, neighbors));
    num_primal_entries = nx * num_directed_edges;

    while ~ADMM.converged && iteration < ADMM.max_iter
        iteration = iteration + 1;

        % ------------------------------------------------------------
        % 1) x update using lambda^k and z^k.
        % ------------------------------------------------------------
        for n = 1:N
            switch string(objective_type)
                case "MLE"
                    fun = @(p) ut.logLikelihoodWithConsensus(p, ...
                        range_local{n}, doppler_local{n}, local_models{n}, env.Sigma_filter, ...
                        n, neighbors, ADMM.Nu_prev, ADMM.update_z_prev, ADMM.c_penalty, whitener);
                case "MAP"
                    fun = @(p) ut.posteriorWithConsensus(p, ...
                        range_local{n}, doppler_local{n}, prior_mean, prior_cov, ...
                        local_models{n}, env.Sigma_filter, n, neighbors, ADMM.Nu_prev, ...
                        ADMM.update_z_prev, ADMM.c_penalty, whitener);
                otherwise
                    error('Wrong ADMM_OBJECTIVE_TYPE setting: %s', ...
                        string(objective_type));
            end

            [x_opt, ~, exitflag, solver_output] = fmincon( ...
                fun, ADMM.initial_values(:,n), [], [], [], [], ...
                ADMM.lb(:), ADMM.ub(:), [], ADMM.solver);
            all_estimations(:,n) = x_opt(:);
            fmincon_exitflag(n) = exitflag;
            fmincon_failure_count = fmincon_failure_count + (exitflag <= 0);
            if isfield(solver_output, 'iterations')
                fmincon_max_iterations = max( ...
                    fmincon_max_iterations, solver_output.iterations);
            end
        end

        ADMM.all_estimations_every_iter(:,:,iteration) = all_estimations;

        % ------------------------------------------------------------
        % 2) z update using x^(k+1) and lambda^k.
        % ------------------------------------------------------------
        c_inv = 1 ./ ADMM.c_penalty(:);
        for n = 1:N
            for j = neighbors{n}
                ADMM.update_z{n}(:,j) = 0.5 * ( ...
                    all_estimations(:,n) + all_estimations(:,j) + ...
                    c_inv .* (ADMM.Nu_prev{n}(:,j) + ...
                              ADMM.Nu_prev{j}(:,n)));
            end
        end

        % ------------------------------------------------------------
        % 3) Unscaled dual update using x^(k+1) and z^(k+1).
        % ------------------------------------------------------------
        for n = 1:N
            for j = neighbors{n}
                ADMM.Nu{n}(:,j) = ADMM.Nu_prev{n}(:,j) + ...
                    ADMM.c_penalty(:) .* ...
                    (all_estimations(:,n) - ADMM.update_z{n}(:,j));
            end
        end

        % Standard stacked primal and dual residuals.
        primal_sq = 0;
        dual_sq = 0;
        x_norm_sq = 0;
        z_norm_sq = 0;
        lambda_norm_sq = 0;
        primal_by_node = zeros(nx, N);
        dual_by_node = zeros(nx, N);

        for n = 1:N
            lambda_sum = zeros(nx,1);
            dual_node = zeros(nx,1);
            for j = neighbors{n}
                r_edge = all_estimations(:,n) - ADMM.update_z{n}(:,j);
                s_edge = ADMM.c_penalty(:) .* ...
                    (ADMM.update_z{n}(:,j) - ADMM.update_z_prev{n}(:,j));

                primal_sq = primal_sq + sum(r_edge.^2);
                x_norm_sq = x_norm_sq + sum(all_estimations(:,n).^2);
                z_norm_sq = z_norm_sq + sum(ADMM.update_z{n}(:,j).^2);
                primal_by_node(:,n) = primal_by_node(:,n) + abs(r_edge);
                dual_node = dual_node + s_edge;
                lambda_sum = lambda_sum + ADMM.Nu{n}(:,j);
            end
            % Standard ADMM dual residual is A' * C * B * delta-z.  Its
            % node-n block is the signed sum over incident edge variables,
            % not the root-sum-square of separate edge contributions.
            dual_sq = dual_sq + sum(dual_node.^2);
            dual_by_node(:,n) = abs(dual_node);
            lambda_norm_sq = lambda_norm_sq + sum(lambda_sum.^2);
        end

        primal_residual = sqrt(primal_sq);
        dual_residual = sqrt(dual_sq);
        eps_pri = sqrt(num_primal_entries) * abs_tolerance + ...
            rel_tolerance * max(sqrt(x_norm_sq), sqrt(z_norm_sq));
        eps_dual = sqrt(N * nx) * abs_tolerance + ...
            rel_tolerance * sqrt(lambda_norm_sq);

        ADMM.primal_residual_all(iteration) = primal_residual;
        ADMM.dual_residual_all(iteration) = dual_residual;
        ADMM.eps_pri_all(iteration) = eps_pri;
        ADMM.eps_dual_all(iteration) = eps_dual;
        ADMM.c_penalty_all(:,iteration) = ADMM.c_penalty(:);
        ADMM.primal_residual_by_para{iteration} = primal_by_node;
        ADMM.dual_residual_by_para{iteration} = dual_by_node;

        if DEBUG && verbose_every > 0 && mod(iteration, verbose_every) == 0
            fprintf(['\rADMM iter %d, r %.3e/%.3e, ', ...
                     's %.3e/%.3e\n'], ...
                iteration, primal_residual, eps_pri, dual_residual, eps_dual);
        end

        ADMM.converged = primal_residual <= eps_pri && ...
                         dual_residual <= eps_dual;
        if ADMM.converged
            stop_reason = "converged";
        elseif mod(iteration, rho_update_interval) == 0
            % Residual balancing for an unscaled dual variable.
            if primal_residual > rho_balance_mu * dual_residual
                ADMM.c_penalty = rho_scale * ADMM.c_penalty;
            elseif dual_residual > rho_balance_mu * primal_residual
                ADMM.c_penalty = ADMM.c_penalty / rho_scale;
            end
        end

        ADMM.initial_values = all_estimations;
        ADMM.update_z_prev = ADMM.update_z;
        ADMM.Nu_prev = ADMM.Nu;
    end

    ADMM.primal_residual_all = ADMM.primal_residual_all(1:iteration);
    ADMM.dual_residual_all = ADMM.dual_residual_all(1:iteration);
    ADMM.eps_pri_all = ADMM.eps_pri_all(1:iteration);
    ADMM.eps_dual_all = ADMM.eps_dual_all(1:iteration);
    ADMM.c_penalty_all = ADMM.c_penalty_all(:,1:iteration);
    ADMM.all_estimations_every_iter = ...
        ADMM.all_estimations_every_iter(:,:,1:iteration);

    x_nodes = all_estimations;
    x_bar = mean(all_estimations, 2);

    % Covariance of this fixed affine posterior: use the SAME Jacobians as
    % the objective, one common prior and every physical measurement once.
    % Re-evaluating H at x_bar here would describe a different approximation.
    if string(objective_type) == "MAP"
        J_prior = prior_cov_common \ eye(nx);
    else
        % An MLE state update contains no recursive prior information.
        J_prior = zeros(nx, nx);
    end
    J_measurement = zeros(nx, nx);
    for n = 1:N
        Hn = fixed_model.H(:,:,n);
        J_measurement = J_measurement + Hn' * (env.Sigma \ Hn);
    end
    J_measurement = symmetrize(J_measurement);
    J_post = symmetrize(J_prior + J_measurement);
    P_global = symmetrize(J_post \ eye(nx));
    P_nodes = repmat({P_global}, 1, N);

    info = struct();
    info.iter = iteration;
    info.converged = ADMM.converged;
    info.stop_reason = stop_reason;
    info.primal_residual = ADMM.primal_residual_all;
    info.dual_residual = ADMM.dual_residual_all;
    info.eps_pri = ADMM.eps_pri_all;
    info.eps_dual = ADMM.eps_dual_all;
    info.primal_residual_by_para = ...
        ADMM.primal_residual_by_para(1:iteration);
    info.dual_residual_by_para = ...
        ADMM.dual_residual_by_para(1:iteration);
    info.x_hist = ADMM.all_estimations_every_iter;
    info.c_penalty_final = ADMM.c_penalty;
    info.c_penalty_history = ADMM.c_penalty_all;
    info.fmincon_exitflag = fmincon_exitflag;
    info.fmincon_failure_count = fmincon_failure_count;
    info.fmincon_max_iterations = fmincon_max_iterations;
    info.J_prior = J_prior;
    info.J_measurement = J_measurement;
    info.J_post = J_post;
    info.P_global = P_global;
    info.linearization = fixed_model;
    info.prior_mean_common = prior_mean_common;
    info.prior_cov_common = prior_cov_common;
end

function [mean_common, cov_common] = common_predicted_prior( ...
        x_pred_nodes, P_pred_nodes)
    N = numel(x_pred_nodes);
    nx = numel(x_pred_nodes{1});
    mean_common = mean(cat(2, x_pred_nodes{:}), 2);

    % The node priors describe the same recursively fused state and are
    % correlated. Average their information; never add it N times.
    information_common = zeros(nx, nx);
    for n = 1:N
        Pn = symmetrize(P_pred_nodes{n});
        information_common = information_common + (Pn \ eye(nx));
    end
    information_common = symmetrize(information_common / N);
    cov_common = symmetrize(information_common \ eye(nx));
end

function [range_local, doppler_local, radarpos_local, numNodes_local, idx_set] = ...
    local_measurement_for_admm(n, range_meas, doppler_meas, t, network_topo, mode)

    idx_set = get_measurement_index_set(n, network_topo, mode);
    numNodes_local = numel(idx_set);

    % The recursive simulation updates at every measurement sample, so the
    % local M in ADMM_utils.logLikelihood/MAP is set to 1.  The resulting
    % matrices have size [1 x numNodes_local]. -> Can be set to NUM_CPI
    range_local   = zeros(1, numNodes_local);
    doppler_local = zeros(1, numNodes_local);
    for a = 1:numNodes_local
        j = idx_set(a);
        range_local(1,a)   = range_meas(1, j, t);
        doppler_local(1,a) = doppler_meas(1, j, t);
    end
    radarpos_local = network_topo.radar_pos(idx_set, :);
end

function A = symmetrize(A)
    A = 0.5 * (A + A');
end


function idx_set = get_measurement_index_set(n, network_topo, mode)
    switch string(mode)
        case "local_neighbor"
            % Explicitly include self so an isolated node never loses its
            % own measurement. This mode is for local estimators, not the
            % graph-wide consensus objective used by this script.
            idx_set = unique([n, ...
                find(network_topo.adj_matrix(n, :) > 0)], 'stable').';
        case "self_only"
            idx_set = n;
        otherwise
            error("Unknown MEASUREMENT_MODE: %s", string(mode));
    end
end

