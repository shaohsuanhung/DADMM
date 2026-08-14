function [Pcrlb, info] = calculate_pcrlb_v5( ...
    mu0, P0, F, Q, radarPositions, lambda, R, ...
    numSteps, numStateSamples, varargin)
%CALCULATE_PCRLB_V5 Bayesian expected-FIM PCRLB for the v5 tracking model.
%
%   [PCRLB, INFO] = CALCULATE_PCRLB_V5(MU0, P0, F, Q, RADARPOSITIONS,
%   LAMBDA, R, NUMSTEPS, NUMSTATESAMPLES) returns PCRLB with size
%   [4 x 4 x NUMSTEPS].  The state expectation in the nonlinear
%   measurement information is approximated with Monte Carlo samples from
%
%       x_1 ~ N(MU0, P0)
%       x_k = F*x_{k-1} + w_k,  w_k ~ N(0,Q),  k >= 2.
%
%   The first measurement is assumed to observe the initial state x_1, so
%   no F/Q prediction is applied before the first measurement update.
%
%   Name-value options:
%       'Seed'              Independent state-MC seed (default 314159)
%       'MinRange'          Error threshold for target/radar coincidence
%                           (default 1e-9)
%       'StoreStateSamples' Store samples in INFO.stateSamples (default
%                           false; this can consume substantial memory)
%
%   R must be the physical, unwhitened 2-by-2 covariance.  The function
%   assumes this covariance is common to all radars and that different
%   radars have mutually independent measurement noise.  The function
%   uses the range/Doppler model
%
%       r   = ||p-p_j||,
%       f_d = (2/lambda) * v'*(p-p_j)/||p-p_j||.
%
%   Q may be positive semidefinite and rank deficient.  Its samples are
%   generated with an eigenvalue square root rather than chol(Q).
%
%   The unregularized Doppler model is singular at a radar position.  With
%   a very broad, full-support Gaussian position prior, rare near-radar
%   samples can dominate the Monte Carlo expected FIM.  Check
%   INFO.minimumSampledRange and sample-count convergence.  Any range
%   regularization must be applied consistently to data generation and the
%   measurement model as well as to this PCRLB calculation.

    parser = inputParser;
    parser.FunctionName = mfilename;
    addParameter(parser, 'Seed', 314159, ...
        @(x) isnumeric(x) && isreal(x) && isscalar(x) && isfinite(x) && ...
             x >= 0 && x <= 2^32-1 && x == floor(x));
    addParameter(parser, 'MinRange', 1e-9, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
    addParameter(parser, 'StoreStateSamples', false, ...
        @(x) islogical_scalar_or_binary_numeric(x));
    parse(parser, varargin{:});

    seed = double(parser.Results.Seed);
    minRange = double(parser.Results.MinRange);
    storeStateSamples = logical(parser.Results.StoreStateSamples);

    if ~isnumeric(mu0) || ~isreal(mu0) || ~isvector(mu0) || ...
            any(~isfinite(mu0(:)))
        error('calculate_pcrlb_v5:PriorMean', ...
            'mu0 must be a finite, real state vector.');
    end
    mu0 = mu0(:);
    nx = numel(mu0);

    if nx ~= 4
        error('calculate_pcrlb_v5:StateDimension', ...
            'The range/Doppler model requires a four-state vector [x;y;vx;vy].');
    end
    validate_square_matrix(P0, nx, 'P0');
    validate_square_matrix(F, nx, 'F');
    validate_square_matrix(Q, nx, 'Q');

    if ~isnumeric(radarPositions) || ~isreal(radarPositions) || ...
            ~ismatrix(radarPositions) || ...
            size(radarPositions, 2) ~= 2 || isempty(radarPositions) || ...
            any(~isfinite(radarPositions(:)))
        error('calculate_pcrlb_v5:RadarPositions', ...
            'radarPositions must be a finite, real N-by-2 matrix.');
    end
    validate_square_matrix(R, 2, 'R');

    if ~isnumeric(lambda) || ~isscalar(lambda) || ~isreal(lambda) || ...
            ~isfinite(lambda) || lambda <= 0
        error('calculate_pcrlb_v5:Lambda', ...
            'lambda must be a finite positive scalar.');
    end
    if ~is_positive_integer(numSteps)
        error('calculate_pcrlb_v5:NumSteps', ...
            'numSteps must be a positive integer.');
    end
    if ~is_positive_integer(numStateSamples)
        error('calculate_pcrlb_v5:NumStateSamples', ...
            'numStateSamples must be a positive integer.');
    end

    numSteps = double(numSteps);
    numStateSamples = double(numStateSamples);

    assert_symmetric(P0, 'P0');
    assert_symmetric(Q, 'Q');
    assert_symmetric(R, 'R');
    P0 = symmetrize_matrix(P0);
    Q = symmetrize_matrix(Q);
    R = symmetrize_matrix(R);

    [L0, p0Flag] = chol(P0, 'lower');
    if p0Flag ~= 0
        error('calculate_pcrlb_v5:P0NotSPD', ...
            'P0 must be symmetric positive definite.');
    end
    [Rchol, rFlag] = chol(R, 'lower');
    if rFlag ~= 0
        error('calculate_pcrlb_v5:RNotSPD', ...
            'R must be symmetric positive definite.');
    end

    Lq = psd_factor(Q, 'Q');
    processNoiseRank = size(Lq, 2);

    % A local stream prevents PCRLB state sampling from changing v5's
    % global RNG sequence for measurement noise and algorithm Monte Carlo.
    stateStream = RandStream('mt19937ar', 'Seed', seed);
    X = repmat(mu0, 1, numStateSamples) + ...
        L0 * randn(stateStream, nx, numStateSamples);

    Pcrlb = zeros(nx, nx, numSteps);
    info = struct();
    info.Jmeas = zeros(nx, nx, numSteps);
    info.Jpost = zeros(nx, nx, numSteps);
    info.Ppred = zeros(nx, nx, numSteps);
    info.stateMean = zeros(nx, numSteps);
    info.minimumSampledRange = inf(1, numSteps);

    if storeStateSamples
        info.stateSamples = zeros(nx, numStateSamples, numSteps);
    else
        info.stateSamples = [];
    end

    for k = 1:numSteps
        Jmeas = zeros(nx, nx);
        minimumSampledRange = inf;
        for sampleIdx = 1:numStateSamples
            [Jsample, sampleMinimumRange] = measurement_information( ...
                X(:, sampleIdx), radarPositions, lambda, Rchol, minRange);
            Jmeas = Jmeas + Jsample;
            minimumSampledRange = min(minimumSampledRange, sampleMinimumRange);
        end
        Jmeas = symmetrize_matrix(Jmeas / numStateSamples);

        if k == 1
            % target_state(:,:,1) in v5 is the initial state.
            Ppred = P0;
        else
            Ppred = F * Pcrlb(:,:,k-1) * F' + Q;
            Ppred = symmetrize_matrix(Ppred);
        end

        Jpred = spd_inverse(Ppred, sprintf('predicted covariance at step %d', k));
        Jpost = symmetrize_matrix(Jpred + Jmeas);
        Ppost = spd_inverse(Jpost, sprintf('posterior information at step %d', k));

        Pcrlb(:,:,k) = symmetrize_matrix(Ppost);
        info.Jmeas(:,:,k) = Jmeas;
        info.Jpost(:,:,k) = Jpost;
        info.Ppred(:,:,k) = Ppred;
        info.stateMean(:,k) = mean(X, 2);
        info.minimumSampledRange(k) = minimumSampledRange;

        if storeStateSamples
            info.stateSamples(:,:,k) = X;
        end

        if k < numSteps
            X = F * X;
            if processNoiseRank > 0
                X = X + Lq * randn( ...
                    stateStream, processNoiseRank, numStateSamples);
            end
        end
    end

    info.kind = 'BayesianExpectedFIM';
    info.version = 1;
    info.seed = seed;
    info.numStateSamples = numStateSamples;
    info.numSteps = numSteps;
    info.firstMeasurementAtInitialState = true;
    info.usesPhysicalMeasurementCovariance = true;
    info.processNoiseRank = processNoiseRank;
    info.priorMean = mu0;
    info.P0 = P0;
    info.F = F;
    info.Q = Q;
    info.R = R;
    info.lambda = lambda;
end

function [J, minimumRange] = measurement_information( ...
    state, radarPositions, lambda, Rchol, minRange)
    position = state(1:2);
    velocity = state(3:4);
    c0 = 2 / lambda;
    J = zeros(4, 4);
    minimumRange = inf;

    for radarIdx = 1:size(radarPositions, 1)
        delta = position - radarPositions(radarIdx,:).';
        range = norm(delta);
        minimumRange = min(minimumRange, range);

        if range <= minRange
            error('calculate_pcrlb_v5:TargetAtRadar', ...
                ['A sampled target is within MinRange of radar %d. ', ...
                 'Use a physically meaningful P0/model or a consistent ', ...
                 'measurement-model regularization.'], radarIdx);
        end

        radialNumerator = velocity' * delta;
        dr_dp = delta' / range;
        dfd_dp = c0 * (velocity' / range - ...
            radialNumerator * delta' / range^3);
        dfd_dv = c0 * delta' / range;

        H = [dr_dp, zeros(1,2); ...
             dfd_dp, dfd_dv];

        weightedH = Rchol' \ (Rchol \ H);
        J = J + H' * weightedH;
    end

    J = symmetrize_matrix(J);
end

function inverseA = spd_inverse(A, matrixName)
    A = symmetrize_matrix(A);
    [L, flag] = chol(A, 'lower');
    if flag ~= 0
        error('calculate_pcrlb_v5:MatrixNotSPD', ...
            '%s must be symmetric positive definite.', matrixName);
    end

    I = eye(size(A));
    inverseA = L' \ (L \ I);
    inverseA = symmetrize_matrix(inverseA);
end

function L = psd_factor(A, matrixName)
    A = symmetrize_matrix(A);
    [V, D] = eig(A);
    eigenvalues = real(diag(D));

    spectralScale = max(abs(eigenvalues));
    if spectralScale == 0
        L = zeros(size(A,1), 0);
        return;
    end
    tolerance = 100 * size(A,1) * eps(spectralScale);

    if any(eigenvalues < -tolerance)
        error('calculate_pcrlb_v5:MatrixNotPSD', ...
            '%s must be positive semidefinite.', matrixName);
    end

    eigenvalues(eigenvalues < 0) = 0;
    keep = eigenvalues > tolerance;
    L = real(V(:,keep) * diag(sqrt(eigenvalues(keep))));
end

function validate_square_matrix(A, expectedSize, matrixName)
    if ~isnumeric(A) || ~isreal(A) || ...
            ~isequal(size(A), [expectedSize, expectedSize]) || ...
            any(~isfinite(A(:)))
        error('calculate_pcrlb_v5:MatrixSize', ...
            '%s must be a finite, real %d-by-%d matrix.', ...
            matrixName, expectedSize, expectedSize);
    end
end

function assert_symmetric(A, matrixName)
    scale = max(1, norm(A, 'fro'));
    if norm(A - A', 'fro') > 1e-10 * scale
        error('calculate_pcrlb_v5:MatrixNotSymmetric', ...
            '%s must be symmetric.', matrixName);
    end
end

function tf = is_positive_integer(value)
    tf = isnumeric(value) && isscalar(value) && isreal(value) && ...
        isfinite(value) && value >= 1 && value == floor(value);
end

function tf = islogical_scalar_or_binary_numeric(value)
    tf = (islogical(value) && isscalar(value)) || ...
         (isnumeric(value) && isreal(value) && isscalar(value) && ...
          isfinite(value) && (value == 0 || value == 1));
end

function A = symmetrize_matrix(A)
    A = 0.5 * (A + A');
end
