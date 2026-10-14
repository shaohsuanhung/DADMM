function [P_local, P_all] = calculate_local_pcrlb_from_information( ...
    P0, F, Q, JmeasByRadar, sensor_sets)
%CALCULATE_LOCAL_PCRLB_FROM_INFORMATION Recur bounds for sensor subsets.
%
%   [P_LOCAL, P_ALL] = CALCULATE_LOCAL_PCRLB_FROM_INFORMATION(
%   P0,F,Q,JMEASBYRADAR,SENSOR_SETS) reuses per-radar expected measurement
%   information produced by calculate_pcrlb_v5.
%
%   JMEASBYRADAR is [NX x NX x NRADAR x T].  SENSOR_SETS is a C-by-N cell
%   array; SENSOR_SETS{c,n} contains the unique one-based radar indices
%   available to node n in configuration c.  Empty sets are allowed and
%   describe a prediction-only local estimator.
%
%   P_LOCAL is [NX x NX x N x T x C], indexed as
%   P_LOCAL(:,:,n,t,c).  P_ALL is [NX x NX x T] and uses every radar.
%   At t=1 the measurement update uses P0 directly.  Prediction with F/Q
%   begins at t=2, matching calculate_pcrlb_v5.

    validateattributes(P0, {'numeric'}, ...
        {'real', 'finite', '2d', 'square', 'nonempty'}, mfilename, 'P0', 1);
    nx = size(P0, 1);
    validate_state_matrix(F, nx, 'F');
    validate_state_matrix(Q, nx, 'Q');

    if ~isnumeric(JmeasByRadar) || ~isreal(JmeasByRadar) || ...
            any(~isfinite(JmeasByRadar(:))) || ...
            size(JmeasByRadar, 1) ~= nx || ...
            size(JmeasByRadar, 2) ~= nx || ...
            size(JmeasByRadar, 3) < 1 || size(JmeasByRadar, 4) < 1
        error('calculate_local_pcrlb_from_information:InformationSize', ...
            ['JmeasByRadar must be a finite, real ', ...
             'NX-by-NX-by-Nradar-by-T array.']);
    end
    if ~iscell(sensor_sets) || isempty(sensor_sets) || ~ismatrix(sensor_sets)
        error('calculate_local_pcrlb_from_information:SensorSetsType', ...
            'sensor_sets must be a nonempty C-by-N cell array.');
    end

    assert_symmetric(P0, 'P0');
    assert_symmetric(Q, 'Q');
    P0 = symmetrize_matrix(P0);
    Q = symmetrize_matrix(Q);
    require_spd(P0, 'P0');
    require_psd(Q, 'Q');

    numRadars = size(JmeasByRadar, 3);
    numSteps = size(JmeasByRadar, 4);
    numConfigurations = size(sensor_sets, 1);
    numNodes = size(sensor_sets, 2);

    normalizedSets = cell(numConfigurations, numNodes);
    for configurationIdx = 1:numConfigurations
        for nodeIdx = 1:numNodes
            indices = sensor_sets{configurationIdx, nodeIdx};
            if isempty(indices)
                normalizedSets{configurationIdx, nodeIdx} = zeros(1, 0);
                continue;
            end
            if ~isnumeric(indices) || ~isreal(indices) || ~isvector(indices) || ...
                    any(~isfinite(indices(:))) || ...
                    any(indices(:) ~= floor(indices(:))) || ...
                    any(indices(:) < 1) || any(indices(:) > numRadars)
                error('calculate_local_pcrlb_from_information:SensorIndex', ...
                    ['sensor_sets{%d,%d} must contain finite integer ', ...
                     'indices between 1 and %d.'], ...
                    configurationIdx, nodeIdx, numRadars);
            end
            indices = indices(:).';
            if numel(unique(indices)) ~= numel(indices)
                error('calculate_local_pcrlb_from_information:DuplicateSensor', ...
                    'sensor_sets{%d,%d} contains duplicate radar indices.', ...
                    configurationIdx, nodeIdx);
            end
            normalizedSets{configurationIdx, nodeIdx} = indices;
        end
    end

    % Symmetrize and validate each independently supplied information block.
    Jradar = zeros(nx, nx, numRadars, numSteps);
    for timeIdx = 1:numSteps
        for radarIdx = 1:numRadars
            information = JmeasByRadar(:,:,radarIdx,timeIdx);
            assert_symmetric(information, sprintf( ...
                'JmeasByRadar(:,:,%d,%d)', radarIdx, timeIdx));
            information = symmetrize_matrix(information);
            require_psd(information, sprintf( ...
                'JmeasByRadar(:,:,%d,%d)', radarIdx, timeIdx));
            Jradar(:,:,radarIdx,timeIdx) = information;
        end
    end

    P_local = zeros(nx, nx, numNodes, numSteps, numConfigurations);
    for configurationIdx = 1:numConfigurations
        for nodeIdx = 1:numNodes
            indices = normalizedSets{configurationIdx, nodeIdx};
            previousPosterior = [];
            for timeIdx = 1:numSteps
                if timeIdx == 1
                    predictedCovariance = P0;
                else
                    predictedCovariance = symmetrize_matrix( ...
                        F * previousPosterior * F' + Q);
                end
                predictedInformation = spd_inverse(predictedCovariance, ...
                    sprintf('local predicted covariance (c=%d,n=%d,t=%d)', ...
                    configurationIdx, nodeIdx, timeIdx));

                localInformation = zeros(nx, nx);
                if ~isempty(indices)
                    localInformation = sum( ...
                        Jradar(:,:,indices,timeIdx), 3);
                end
                posteriorInformation = symmetrize_matrix( ...
                    predictedInformation + localInformation);
                previousPosterior = spd_inverse(posteriorInformation, ...
                    sprintf('local posterior information (c=%d,n=%d,t=%d)', ...
                    configurationIdx, nodeIdx, timeIdx));
                P_local(:,:,nodeIdx,timeIdx,configurationIdx) = ...
                    previousPosterior;
            end
        end
    end

    P_all = zeros(nx, nx, numSteps);
    previousPosterior = [];
    for timeIdx = 1:numSteps
        if timeIdx == 1
            predictedCovariance = P0;
        else
            predictedCovariance = symmetrize_matrix( ...
                F * previousPosterior * F' + Q);
        end
        predictedInformation = spd_inverse(predictedCovariance, ...
            sprintf('all-sensor predicted covariance at t=%d', timeIdx));
        allInformation = sum(Jradar(:,:,:,timeIdx), 3);
        posteriorInformation = symmetrize_matrix( ...
            predictedInformation + allInformation);
        previousPosterior = spd_inverse(posteriorInformation, ...
            sprintf('all-sensor posterior information at t=%d', timeIdx));
        P_all(:,:,timeIdx) = previousPosterior;
    end

end

function validate_state_matrix(A, nx, matrixName)
    if ~isnumeric(A) || ~isreal(A) || ...
            ~isequal(size(A), [nx, nx]) || any(~isfinite(A(:)))
        error('calculate_local_pcrlb_from_information:MatrixSize', ...
            '%s must be a finite, real %d-by-%d matrix.', ...
            matrixName, nx, nx);
    end
end

function assert_symmetric(A, matrixName)
    scale = max(1, norm(A, 'fro'));
    if norm(A - A', 'fro') > 1e-10 * scale
        error('calculate_local_pcrlb_from_information:MatrixNotSymmetric', ...
            '%s must be symmetric.', matrixName);
    end
end

function require_spd(A, matrixName)
    [~, flag] = chol(symmetrize_matrix(A), 'lower');
    if flag ~= 0
        error('calculate_local_pcrlb_from_information:MatrixNotSPD', ...
            '%s must be symmetric positive definite.', matrixName);
    end
end

function require_psd(A, matrixName)
    eigenvalues = eig(symmetrize_matrix(A), 'vector');
    scale = max(1, norm(A, 2));
    tolerance = 100 * size(A, 1) * eps(scale);
    if any(real(eigenvalues) < -tolerance)
        error('calculate_local_pcrlb_from_information:MatrixNotPSD', ...
            '%s must be positive semidefinite.', matrixName);
    end
end

function inverseA = spd_inverse(A, matrixName)
    A = symmetrize_matrix(A);
    [L, flag] = chol(A, 'lower');
    if flag ~= 0
        error('calculate_local_pcrlb_from_information:MatrixNotSPD', ...
            '%s must be symmetric positive definite.', matrixName);
    end
    identity = eye(size(A));
    inverseA = L' \ (L \ identity);
    inverseA = symmetrize_matrix(inverseA);
end

function A = symmetrize_matrix(A)
    A = 0.5 * (A + A');
end
