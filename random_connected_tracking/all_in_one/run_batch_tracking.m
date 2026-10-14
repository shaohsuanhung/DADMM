function Results = run_batch_tracking(cfg, network_topo, estimator_type, target)
%RUN_BATCH_TRACKING Batch MAP/MLE with common truth and CV prediction.
% Results = run_batch_tracking(cfg, network_topo, "MAP" or "MLE", target).
% The caller creates the shared target once, outside every Monte Carlo loop.
% Measurements are linearized once per batch, before any ADMM iterations.
% See README.md for retained prior/ADMM formulas and bound limitations.

cfg = fill_defaults(cfg);
TYPE = upper(string(estimator_type));
assert(isscalar(TYPE) && any(TYPE == ["MAP", "MLE"]), ...
    'estimator_type must be MAP or MLE.');
validateattributes(network_topo.numNodes, {'numeric'}, {'scalar','integer','>=',2});
assert(isequal(size(network_topo.radar_pos), [network_topo.numNodes, 2]), ...
    'radar_pos must contain one [x,y] row per node.');
assert(numel(network_topo.neighbors) == network_topo.numNodes, ...
    'network_topo.neighbors must contain one neighbor list per node.');
previous_rng = rng;
rng_cleanup = onCleanup(@() rng(previous_rng));
ut = BatchADMMUtils;
PRE_WHITEN = cfg.PreWhiten;
num_monte_carlo = cfg.NumMonteCarlo;
NUM_TAR = 1;
NUM_CPI_PER_MEA = cfg.SamplesPerBatch;
TRACK_TIME = cfg.NumBatches;
time_step = cfg.TimeStep;
env.SNR_idx = cfg.SNRdB;
num_samples = NUM_CPI_PER_MEA * TRACK_TIME;
assert(isequal(size(target.target_state), [1, num_samples, 4]), ...
    'target.target_state must be 1-by-(SamplesPerBatch*NumBatches)-by-4.');
assert(isequal(size(target.target_position), [1, num_samples, 2]), ...
    'target.target_position must match the supplied target states.');
assert(numel(target.time) == num_samples && ...
    all(abs(diff(target.time(:)) - time_step) < 1e-10), ...
    'The shared truth time grid must match cfg.TimeStep.');
assert(isfield(target, 'trajectory_seed'), ...
    'The shared truth must record trajectory_seed.');

% Configuration, storage, and the original signal/prior parameters.
Results = struct();
Results.primal_residauls = cell(num_monte_carlo, TRACK_TIME);
Results.dual_residauls   = cell(num_monte_carlo, TRACK_TIME);
Results.estimations_DA   = cell(num_monte_carlo, TRACK_TIME);
Results.true_params      = cell(num_monte_carlo, TRACK_TIME);
Results.convg_iter       = cell(num_monte_carlo, TRACK_TIME);
Results.estimations_CA   = cell(num_monte_carlo, TRACK_TIME);
Results.ADMM_setting   = cell(num_monte_carlo, TRACK_TIME);
Results.CRLB           = cell(num_monte_carlo, TRACK_TIME);
Results.consensus_estimates = cell(num_monte_carlo, TRACK_TIME);
Results.solver_exitflags_CA = cell(num_monte_carlo, TRACK_TIME);
Results.solver_exitflags_DA = cell(num_monte_carlo, TRACK_TIME);
Results.converged = false(num_monte_carlo, TRACK_TIME);
Results.targets = cell(num_monte_carlo, 1);
Results.trajectory_seeds = zeros(num_monte_carlo, 1);
Results.measurement_seeds = zeros(num_monte_carlo, 1);
Results.predicted_means_CA = cell(num_monte_carlo, TRACK_TIME);
Results.predicted_means_DA = cell(num_monte_carlo, TRACK_TIME);
Results.prior_covariances_CA = cell(num_monte_carlo, TRACK_TIME);
Results.prior_covariances_DA = cell(num_monte_carlo, TRACK_TIME);
Results.warm_starts_CA = cell(num_monte_carlo, TRACK_TIME);
Results.warm_starts_DA = cell(num_monte_carlo, TRACK_TIME);
Results.linearization_CA = cell(num_monte_carlo, TRACK_TIME);
Results.linearization_DA = cell(num_monte_carlo, TRACK_TIME);
neighbors = network_topo.neighbors;

% Signal/noise settings and MAP priors retained from the original runner.
env.c = 3e8;
env.lambda = env.c / 10e9;
env.time_step = time_step;
env.T = env.time_step / 2;
env.B = 10e6 * ones(1,network_topo.numNodes);
env.fs = 2*env.B;

env.SNR_lin = 10^(env.SNR_idx/10);

env.range_var   = (3 * env.c^2) / (8 * pi^2 * env.B(1)^2 * env.SNR_lin);
env.doppler_var = (3 * (env.fs(1)^2)) / (pi^2 * env.SNR_lin * NUM_CPI_PER_MEA^3);
env.range_sd = sqrt(env.range_var);
env.doppler_sd = sqrt(env.doppler_var);
env.rho = 1e-6;

env.Sigma = [env.range_var, env.rho*env.range_sd*env.doppler_sd; ...
             env.rho*env.range_sd*env.doppler_sd, env.doppler_var];

dt = env.time_step;
env.Q = cfg.ProcessAccelerationVariance * [dt^4/4, 0,      dt^3/2, 0;
            0,      dt^4/4, 0,      dt^3/2;
            dt^3/2, 0,      dt^2,   0;
            0,      dt^3/2, 0,      dt^2];

if PRE_WHITEN
    U = chol(env.Sigma, 'upper');     % U'*U = Sigma
    env.pre_whit_L = U' \ eye(2);         % inv(U')*Sigma*inv(U) = I
    env.Sigma_filter = eye(2);
    env.PRE_WHITEN = true;
else
    env.pre_whit_L = eye(2);
    env.Sigma_filter = env.Sigma;
    env.PRE_WHITEN = false;
end

prior_mu = repmat({cfg.PriorMean(:)}, 1, network_topo.numNodes);
ADMM = ut.Initialized_ADMM(network_topo);
ADMM.max_iter = cfg.MaxIterations;
ADMM.tolerance = cfg.Tolerance;
ADMM.initial_values = repmat(cfg.InitialState(:), 1, network_topo.numNodes);
state_cov = diag([env.range_var, env.range_var, env.doppler_var, env.doppler_var]);
% There is no posterior-covariance recursion in this legacy batch estimator.
% Predict its fixed covariance proxy over the gap between batch endpoints.
dt_batch = NUM_CPI_PER_MEA * time_step;
F_batch = [eye(2), dt_batch * eye(2); zeros(2), eye(2)];
Q_batch = cfg.ProcessAccelerationVariance * ...
    [dt_batch^4/4 * eye(2), dt_batch^3/2 * eye(2); ...
     dt_batch^3/2 * eye(2), dt_batch^2 * eye(2)];
predicted_covariance_proxy = F_batch * state_cov * F_batch' + Q_batch;
% Measurement whitening does not change the state-space prior covariance.
prior_sigma = repmat({state_cov}, 1, network_topo.numNodes);

options_CA = optimoptions('fmincon', 'Display', 'off', SpecifyObjectiveGradient=true, ScaleProblem=true, ...
    OptimalityTolerance=1e-6, FunctionTolerance=1e-6, StepTolerance=1e-6, MaxIterations=50000);
initial_ADMM = ADMM;
% All trials use the supplied truth; only measurement noise changes per trial.
for mc = 1:num_monte_carlo
    ADMM = initial_ADMM; % Every MC trial starts with the same solver state.
    prior_mu_CA = prior_mu;
    prior_sigma_CA = prior_sigma;
    measurement_seed = cfg.Seed + 100000 + mc;
    if cfg.Verbose
        fprintf("\n[%s MC %d/%d]\n", TYPE, mc, num_monte_carlo);
    end

    Results.targets{mc} = target;
    Results.trajectory_seeds(mc) = target.trajectory_seed;
    Results.measurement_seeds(mc) = measurement_seed;
    rng(measurement_seed, "twister");

    range_true = zeros(NUM_TAR, network_topo.numNodes, NUM_CPI_PER_MEA*TRACK_TIME);
    doppler_true = zeros(NUM_TAR, network_topo.numNodes, NUM_CPI_PER_MEA*TRACK_TIME);
    measurements_true = zeros(NUM_TAR, network_topo.numNodes, 2*NUM_CPI_PER_MEA*TRACK_TIME);

    [range_true, doppler_true, ~] = ut.gt_data_generation_tracking( ...
        range_true, doppler_true, measurements_true, target, network_topo, env, NUM_CPI_PER_MEA*TRACK_TIME, NUM_TAR);

    [range_with_error, doppler_with_error, ~] = ut.add_measurement_noise( ...
        range_true, doppler_true, NUM_CPI_PER_MEA*TRACK_TIME, NUM_TAR, network_topo.numNodes, env);

    % Local batches contain the node's measurements and its neighbors'.
    range_withNeighbors = cell(NUM_TAR, network_topo.numNodes);
    doppler_withNeighbors = cell(NUM_TAR, network_topo.numNodes);

    [range_withNeighbors, doppler_withNeighbors, ~, radarpos_withNeighbors, ...
        prior_mu_cell, prior_sigma_cell] = ut.pharse_measurements_tracking( ...
        network_topo.laplacian_matrix, range_with_error, doppler_with_error, ...
        range_withNeighbors, doppler_withNeighbors, NUM_TAR, network_topo, prior_mu, prior_sigma);

    for tar = 1:NUM_TAR
        for k = 1:TRACK_TIME

            range_win = cell(1, network_topo.numNodes);
            doppler_win = cell(1, network_topo.numNodes);
            for n = 1:network_topo.numNodes
                range_win{n} = range_withNeighbors{tar,n}((k-1)*NUM_CPI_PER_MEA+1 : k*NUM_CPI_PER_MEA, :);
                doppler_win{n} = doppler_withNeighbors{tar,n}((k-1)*NUM_CPI_PER_MEA+1 : k*NUM_CPI_PER_MEA, :);
            end

            % Centralized baseline for the same measurement batch.
            if k == 1
                CA_initial_guess = cfg.InitialState(:);
                predicted_CA = cfg.PriorMean(:);
            else
                predicted_CA = F_batch * Results.estimations_CA{mc,k-1}(:);
                CA_initial_guess = predicted_CA;
                prior_mu_CA = repmat({predicted_CA}, 1, network_topo.numNodes);
                prior_sigma_CA = repmat({predicted_covariance_proxy}, 1, network_topo.numNodes);
            end
            Results.predicted_means_CA{mc,k} = predicted_CA;
            Results.prior_covariances_CA{mc,k} = prior_sigma_CA{1};
            Results.warm_starts_CA{mc,k} = CA_initial_guess;
            lb = [-inf,-inf,-inf,-inf];
            ub = [ inf, inf, inf, inf];
            range_win_CA = squeeze(range_with_error(tar,:,(k-1)*NUM_CPI_PER_MEA+1 : k*NUM_CPI_PER_MEA))';
            doopler_win_CA = squeeze(doppler_with_error(tar,:,(k-1)*NUM_CPI_PER_MEA+1 : k*NUM_CPI_PER_MEA))';
            % The affine model is fixed for this entire batch solve.
            model_CA = ut.linearize_measurements(CA_initial_guess, ...
                network_topo.radar_pos, env.lambda, env.pre_whit_L);
            Results.linearization_CA{mc,k} = model_CA;
            if TYPE == "MLE"
                funCA = @(p) ut.logLikelihood(p, range_win_CA, doopler_win_CA, model_CA, env.Sigma_filter);
            else
                funCA = @(p) ut.MAP(p, range_win_CA, doopler_win_CA, ...
                    prior_mu_CA, prior_sigma_CA, model_CA, env.Sigma_filter);
            end

            [est_CA, ~, ca_exitflag] = fmincon(funCA, CA_initial_guess, [], [], [], [], lb, ub, [], options_CA);
            if k ~= 1
                prev_est = ADMM.final_tracking_estimation{tar, k-1};
                predicted_DA = F_batch * prev_est;
                ADMM.initial_values = repmat(predicted_DA, 1, network_topo.numNodes);

                [ADMM.prior_mean, ADMM.prior_cov] = ut.get_neighbors_cell( ...
                    network_topo.laplacian_matrix, NUM_TAR, network_topo.numNodes, ...
                    predicted_DA, predicted_covariance_proxy);

            elseif k == 1
                predicted_DA = cfg.PriorMean(:);
                ADMM.prior_mean = prior_mu_cell;
                ADMM.prior_cov  = prior_sigma_cell;

            else
                error("Wrong k setting")
            end
            Results.predicted_means_DA{mc,k} = repmat(predicted_DA, 1, network_topo.numNodes);
            Results.prior_covariances_DA{mc,k} = ADMM.prior_cov;
            Results.warm_starts_DA{mc,k} = ADMM.initial_values;

            % All nodes share the same anchor; only their radar subsets differ.
            % First batch: InitialState. Later batches: the common CV prediction.
            anchor_DA = ADMM.initial_values(:,1);
            models_DA = cell(1, network_topo.numNodes);
            for n = 1:network_topo.numNodes
                models_DA{n} = ut.linearize_measurements(anchor_DA, ...
                    radarpos_withNeighbors{tar,n}, env.lambda, env.pre_whit_L);
            end
            Results.linearization_DA{mc,k} = models_DA;

            ADMM.converged = false;
            ADMM.converg_r = false; ADMM.converg_d = false;

            iteration = 0;
            all_estimations = zeros(4, network_topo.numNodes);
            local_exitflags = zeros(1, network_topo.numNodes);

            % Original dual/local-solve/edge-update sequence.
            while ~ADMM.converged && iteration < ADMM.max_iter
                iteration = iteration + 1;
                if cfg.Verbose && mod(iteration,50)== 0
                    fprintf('\rADMM iter %d, Primal: %d, Dual: %d\n', iteration,primal_residual, dual_residual);
                end

                for n = 1:network_topo.numNodes
                    for j = neighbors{n}
                        ADMM.Nu{n}(:,j) = ADMM.Nu_prev{n}(:,j) + ADMM.c_penalty' .* (ADMM.initial_values(:,n) - ADMM.update_z_prev{n}(:,j));
                    end
                    if TYPE == "MLE"
                        fun = @(p) ut.logLikelihoodWithConsensus(p, range_win{n}, doppler_win{n}, ...
                            models_DA{n}, env.Sigma_filter, n, neighbors, ADMM.Nu, ...
                            ADMM.update_z_prev, ADMM.c_penalty);
                    else
                        fun = @(p) ut.posteriorWithConsensus(p, range_win{n}, doppler_win{n}, ...
                            ADMM.prior_mean{n}, ADMM.prior_cov{n}, models_DA{n}, ...
                            env.Sigma_filter, n, neighbors, ADMM.Nu, ADMM.update_z_prev, ADMM.c_penalty);
                    end

                    [local_estimate, ~, local_exitflags(n)] = fmincon(fun, ADMM.initial_values(:,n), [], [], [], [], ADMM.lb, ADMM.ub, [], ADMM.solver);
                    all_estimations(:,n) = local_estimate;
                end

                ADMM.all_estimations_every_iter(:,:,iteration) = all_estimations;

                for n = 1:network_topo.numNodes
                    for j = neighbors{n}
                        ADMM.update_z{n}(:,j) = 0.5 * ( ((ADMM.c_penalty.^(-1))' .* (ADMM.Nu{n}(:,j) + ADMM.Nu{j}(:,n))) ...
                            + all_estimations(:,n) + all_estimations(:,j) );
                    end
                end

                primal_residual = 0;
                dual_residual   = 0;
                prima_by_node = zeros(4, network_topo.numNodes);
                dual_by_node  = zeros(4, network_topo.numNodes);

                for n = 1:network_topo.numNodes
                    pr = zeros(4,1);
                    dr = zeros(4,1);
                    for j = neighbors{n}
                        primal_residual = primal_residual + sqrt(norm(all_estimations(:,n) - ADMM.update_z{n}(:,j))^2);
                        dual_residual   = dual_residual   + sqrt(norm(ADMM.Nu{n}(:,j) - ADMM.Nu_prev{n}(:,j))^2);
                        pr = pr + abs(all_estimations(:,n) - ADMM.update_z{n}(:,j));
                        dr = dr + abs(ADMM.Nu{n}(:,j) - ADMM.Nu_prev{n}(:,j));
                    end
                    prima_by_node(:,n) = pr;
                    dual_by_node(:,n)  = dr;
                end

                ADMM.primal_residual_all(iteration) = primal_residual;
                ADMM.dual_residual_all(iteration)   = dual_residual;
                ADMM.primal_residual_by_para{iteration} = prima_by_node;
                ADMM.dual_residual_by_para{iteration}   = dual_by_node;

                if mod(iteration,30)==0
                    if primal_residual < dual_residual
                        ADMM.c_penalty = ADMM.tau_incr .* ADMM.c_penalty;
                    elseif dual_residual < primal_residual
                        ADMM.c_penalty = ADMM.c_penalty .* ((ADMM.tau_decr).^(-1));
                    end
                end

                ADMM.initial_values = all_estimations;
                ADMM.update_z_prev = ADMM.update_z;
                ADMM.Nu_prev = ADMM.Nu;

                % This is the original effective test, not a new ADMM proof.
                if primal_residual < ADMM.tolerance
                    ADMM.converged = true;
                end
            end

            % Store the complete history and distinguish a cap from convergence.
            true_params_k = reshape(target.target_state(tar, k*NUM_CPI_PER_MEA, :), 1, 4);
            ADMM.final_tracking_estimation{tar,k} = mean(all_estimations,2);

            Results.primal_residauls{mc,k} = ADMM.primal_residual_all; 
            Results.dual_residauls{mc,k}   = ADMM.dual_residual_all;
            Results.estimations_DA{mc,k}   = ADMM.all_estimations_every_iter;
            Results.true_params{mc,k}      = true_params_k;
            Results.convg_iter{mc,k}       = iteration;
            Results.converged(mc,k) = ADMM.converged;
            Results.solver_exitflags_CA{mc,k} = ca_exitflag;
            Results.solver_exitflags_DA{mc,k} = local_exitflags;
            Results.ADMM_setting{mc,k} = struct('penalty', ADMM.c_penalty, ...
                'tolerance', ADMM.tolerance, 'max_iter', ADMM.max_iter); 
            Results.estimations_CA{mc,k}   = est_CA(:).';
            Results.consensus_estimates{mc,k} = mean(all_estimations,2);
            Results.node_wise_cov{mc,k} = cov(all_estimations(1:2,:).','omitrows');
            if cfg.ComputeLegacyBound && TYPE == "MAP"
                Results.CRLB{mc,k}         = ut.calculateBCRLB(true_params_k,...
                                                 network_topo.radar_pos, network_topo.numNodes,...
                                                 NUM_CPI_PER_MEA, env.lambda, env.Sigma_filter,env.Q);
            elseif cfg.ComputeLegacyBound && TYPE == "MLE"
                FIM = ut.calculateFIM(true_params_k,...
                                             network_topo.radar_pos, network_topo.numNodes,...
                                             NUM_CPI_PER_MEA, env.lambda, env.Sigma_filter);
                Results.CRLB{mc,k}         = inv(FIM);
            end
            ADMM = ut.ADMM_reset(ADMM, NUM_TAR, network_topo);

            if cfg.Verbose
                fprintf("Batch %d completed (%d iterations).\n", k, iteration);
            end
        end
    end

end
Results.config = cfg;
Results.estimator_type = TYPE;
Results.network_topo = network_topo;
Results.constant = env;
Results.sample_time = target.time(:).';
Results.batch_time = Results.sample_time(NUM_CPI_PER_MEA:NUM_CPI_PER_MEA:end);
Results.truth_velocity_convention = "Per-sample velocity from the shared target.target_state";
Results.measurement_model = "h(anchor)+H(anchor)*(x-anchor); h=[range;dot(v,p-g)/(lambda*range)]";
Results.linearization_policy = "Fixed per batch; InitialState first, then each estimator's CV prediction; common DA anchor across nodes";
Results.likelihood_model = "Gaussian weighted least squares with full 2-by-2 inverse covariance";
Results.F_batch = F_batch;
Results.Q_batch = Q_batch;
Results.state_covariance_proxy = state_cov;
Results.predicted_covariance_proxy = predicted_covariance_proxy;
Results.prior_used = TYPE == "MAP";
Results.prediction_model = "Constant velocity over the interval between batch endpoints; MLE uses prediction for the linearization anchor and warm start, with no prior cost";
Results.covariance_model = "Fixed legacy state_cov proxy predicted by F_batch and Q_batch; no posterior covariance update";
Results.CRLB_status = "Legacy optional diagnostic; not a validated PCRLB";
end

function cfg = fill_defaults(cfg)
defaults = batch_default_config();
fields = fieldnames(defaults);
for i = 1:numel(fields)
    if ~isfield(cfg, fields{i})
        cfg.(fields{i}) = defaults.(fields{i});
    end
end
validateattributes(cfg.NumMonteCarlo, {'numeric'}, {'scalar','integer','positive'});
validateattributes(cfg.NumBatches, {'numeric'}, {'scalar','integer','positive'});
validateattributes(cfg.SamplesPerBatch, {'numeric'}, {'scalar','integer','>=',2});
validateattributes(cfg.MaxIterations, {'numeric'}, {'scalar','integer','positive'});
validateattributes(cfg.TimeStep, {'numeric'}, {'scalar','positive','finite'});
validateattributes(cfg.Tolerance, {'numeric'}, {'scalar','positive','finite'});
validateattributes(cfg.Seed, {'numeric'}, {'scalar','integer','nonnegative','<=',2^32-100001-cfg.NumMonteCarlo});
validateattributes(cfg.SNRdB, {'numeric'}, {'scalar','finite'});
validateattributes(cfg.InitialState, {'numeric'}, {'vector','numel',4,'finite'});
validateattributes(cfg.PriorMean, {'numeric'}, {'vector','numel',4,'finite'});
validateattributes(cfg.ProcessAccelerationVariance, {'numeric'}, {'scalar','nonnegative','finite'});
end
