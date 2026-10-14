function Study = run_local_neighborhood_ekf_study(config)
%RUN_LOCAL_NEIGHBORHOOD_EKF_STUDY Local-neighborhood D-EKF SNR/CR study.
%
%   STUDY = RUN_LOCAL_NEIGHBORHOOD_EKF_STUDY(CONFIG) compares independent
%   closed-neighborhood distributed EKFs (D-EKF) against a centralized EKF
%   (C-EKF). Each distributed node first performs a local measurement update
%   using itself and its communication neighbors. For every center node,
%   the sensors in that closed neighborhood run their own edge-based ADMM
%   nonlinear state-consensus MAP optimization. The covariance is then
%   obtained by average consensus on Gauss-Newton information packets
%   evaluated at the converged MAP state. Different
%   ego neighborhoods are not connected through a second global-consensus
%   layer. Direct stacked-EKF equivalents are retained for validation.
%
%   CONFIG is an optional scalar struct. Supported fields and defaults are:
%
%       SNRdB                  [20 30 40]
%       CommunicationRadii    [30 50 500]
%       NumMonteCarlo         100
%       NumSteps              64
%       Dt                    0.01
%       NumDopplerSamples     64
%       NumPCRLBSamples       2000
%       Seed                  43
%       PCRLBSeed             100043
%       ADMMMaxIter           1000
%       ADMMAbsTolerance      1e-8
%       ADMMRelTolerance      1e-7
%       ADMMRho               1
%       NonlinearSolverMaxIterations  100
%       MakePlots             true
%       SaveResults           true
%       OutputDir             <this folder>/results/local_neighborhood_ekf
%
%   The Monte Carlo experiment is Bayesian-consistent: every realization
%   samples x_1 ~ N(mu0,P0) and process noise from the same Q used by the
%   filters and PCRLB. The underlying truth and standard-normal measurement
%   noise are reused for all SNR and communication-radius configurations,
%   giving paired comparisons. Measurement noise is merely rescaled for
%   each SNR.

    if nargin < 1 || isempty(config)
        config = struct();
    end

    scriptDir = fileparts(mfilename('fullpath'));
    config = apply_defaults_and_validate(config, scriptDir);
    validate_dependency_paths(scriptDir, config.MakePlots);

    modelsUt = models;

    %% Fixed physical and state-space configuration
    nx = 4;
    numNodes = 10;
    numSNR = numel(config.SNRdB);
    numCR = numel(config.CommunicationRadii);
    numSteps = config.NumSteps;
    numMC = config.NumMonteCarlo;

    mu0 = [-30; -30; 35.3553; 35.3553];
    P0 = diag([1, 1, 4, 4]);

    dt = config.Dt;
    F = [1, 0, dt, 0; ...
         0, 1, 0, dt; ...
         0, 0, 1,  0; ...
         0, 0, 0,  1];
    processAccelerationVariance = 1;
    G = [dt^2/2, 0; ...
         0, dt^2/2; ...
         dt, 0; ...
         0, dt];
    Q = processAccelerationVariance * (G * G.');

    physical = struct();
    physical.c = 3e8;
    physical.fc = 10e9;
    physical.lambda = physical.c / physical.fc;
    physical.bandwidth = 10e6;
    physical.samplingFrequency = 2 * physical.bandwidth;
    physical.measurementCorrelation = 0;
    physical.radarRadius = 30;

    theta = linspace(0, 2*pi, numNodes + 1);
    theta = theta(1:end-1);
    radarPositions = physical.radarRadius * [cos(theta); sin(theta)].';

    %% Communication topology and closed-neighborhood sensor sets
    topologies = repmat(struct( ...
        'numNodes', [], 'theta', [], 'radius', [], 'com_rad_CR', [], ...
        'radar_pos', [], 'labels', [], 'adj_matrix', [], ...
        'degree_matrix', [], 'laplacian_matrix', [], 'inc_matrix', [], ...
        'weights_matrix', []), 1, numCR);
    sensorSets = cell(numCR, numNodes);
    degree = zeros(numCR, numNodes);
    neighborhoodSize = zeros(numCR, numNodes);

    for crIdx = 1:numCR
        topology = struct();
        topology.numNodes = numNodes;
        topology.theta = theta;
        topology.radius = physical.radarRadius;
        topology.com_rad_CR = config.CommunicationRadii(crIdx);
        topology.radar_pos = radarPositions;
        topology.labels = arrayfun(@(n) sprintf('N%d', n), 1:numNodes, ...
            'UniformOutput', false);

        [topology.adj_matrix, topology.degree_matrix, ...
         topology.laplacian_matrix, topology.inc_matrix, ...
         topology.weights_matrix] = ADMM_utils.calculate_all_graph_matrix( ...
            radarPositions, topology.com_rad_CR, numNodes);

        degree(crIdx,:) = diag(topology.degree_matrix).';
        for nodeIdx = 1:numNodes
            % Sorting gives a deterministic measurement-stack order. In the
            % fully connected case this is exactly 1:numNodes, which makes
            % the D-EKF/C-EKF equality check particularly strict.
            idxSet = unique([nodeIdx, ...
                find(topology.adj_matrix(nodeIdx,:) > 0)]);
            idxSet = idxSet(:).';

            assert(ismember(nodeIdx, idxSet), ...
                'LocalEKF:MissingSelfMeasurement', ...
                'Node %d does not contain its own measurement.', nodeIdx);
            assert(numel(unique(idxSet)) == numel(idxSet), ...
                'LocalEKF:DuplicateSensorIndex', ...
                'Node %d has duplicate sensor indices.', nodeIdx);
            assert(numel(idxSet) == degree(crIdx,nodeIdx) + 1, ...
                'LocalEKF:NeighborhoodSizeMismatch', ...
                ['Node %d has %d sensors, but degree+1 is %d for ', ...
                 'communication radius %g.'], nodeIdx, numel(idxSet), ...
                degree(crIdx,nodeIdx) + 1, topology.com_rad_CR);

            sensorSets{crIdx,nodeIdx} = idxSet;
            neighborhoodSize(crIdx,nodeIdx) = numel(idxSet);
        end
        topologies(crIdx) = topology;
    end

    %% Paired Bayesian truth and base measurement-noise samples
    simulationStream = RandStream('mt19937ar', 'Seed', config.Seed);
    priorFactor = chol(P0, 'lower');
    truth = zeros(nx, numSteps, numMC);
    for mcIdx = 1:numMC
        truth(:,1,mcIdx) = mu0 + priorFactor * ...
            randn(simulationStream, nx, 1);
        for timeIdx = 2:numSteps
            acceleration = sqrt(processAccelerationVariance) * ...
                randn(simulationStream, 2, 1);
            truth(:,timeIdx,mcIdx) = F * truth(:,timeIdx-1,mcIdx) + ...
                G * acceleration;
        end
    end

    % These standard-normal draws are shared across every SNR and CR. Each
    % SNR-specific Cholesky factor below supplies only the correct scale.
    baseMeasurementNoise = randn( ...
        simulationStream, 2, numNodes, numSteps, numMC);

    modelEnvironment = struct('lambda', physical.lambda);
    trueMeasurements = zeros(2, numNodes, numSteps, numMC);
    geometryTopology = topologies(1);
    for mcIdx = 1:numMC
        for timeIdx = 1:numSteps
            for radarIdx = 1:numNodes
                trueMeasurements(:,radarIdx,timeIdx,mcIdx) = ...
                    modelsUt.MeasureModel( ...
                        truth(:,timeIdx,mcIdx), radarIdx, ...
                        geometryTopology, modelEnvironment);
            end
        end
    end

    %% Allocate node-wise, centralized, and bound results
    estimatesD = zeros(nx, numNodes, numSteps, numMC, numSNR, numCR);
    diagPD = zeros(nx, numNodes, numSteps, numMC, numSNR, numCR);
    estimatesDLocal = zeros(nx, numNodes, numSteps, numMC, numSNR, numCR);
    diagPDLocal = zeros(nx, numNodes, numSteps, numMC, numSNR, numCR);
    estimatesC = zeros(nx, numSteps, numMC, numSNR);
    diagPC = zeros(nx, numSteps, numMC, numSNR);

    pcrlbCovDLocal = zeros(nx, nx, numNodes, numSteps, numSNR, numCR);
    pcrlbCovDConsensus = zeros( ...
        nx, nx, numNodes, numSteps, numSNR, numCR);
    pcrlbCovC = zeros(nx, nx, numSteps, numSNR);
    pcrlbInfo = cell(1, numSNR);
    environments = repmat( ...
        make_environment(config.SNRdB(1), config, physical, Q), ...
        1, numSNR);

    minimumPosteriorEigenvalueD = inf(numSNR, numCR);
    minimumLocalPosteriorEigenvalueD = inf(numSNR, numCR);
    minimumPosteriorEigenvalueC = inf(1, numSNR);
    admmIterations = zeros(numNodes, numSteps, numMC, numSNR, numCR);
    admmPrimalResidual = zeros(numNodes, numSteps, numMC, numSNR, numCR);
    admmDualResidual = zeros(numNodes, numSteps, numMC, numSNR, numCR);
    admmNodeDisagreement = zeros( ...
        numNodes, numSteps, numMC, numSNR, numCR);
    covarianceConsensusIterations = zeros( ...
        numNodes, numSteps, numMC, numSNR, numCR);
    nonlinearSolverFailureCount = zeros( ...
        numNodes, numSteps, numMC, numSNR, numCR);

    %% SNR sweep
    for snrIdx = 1:numSNR
        env = make_environment(config.SNRdB(snrIdx), config, physical, Q);
        environments(snrIdx) = env;
        noiseFactor = chol(env.Sigma, 'lower');

        fprintf(['Local-neighborhood EKF study: SNR %g dB ', ...
                 '(%d/%d)\n'], config.SNRdB(snrIdx), snrIdx, numSNR);

        % One expected-information calculation per SNR. The per-radar
        % information is subsequently reused for every node and CR.
        [pcrlbAllDirect, thisPCRLBInfo] = calculate_pcrlb_v5( ...
            mu0, P0, F, Q, radarPositions, physical.lambda, env.Sigma, ...
            numSteps, config.NumPCRLBSamples, 'Seed', config.PCRLBSeed);
        if ~isfield(thisPCRLBInfo, 'JmeasByRadar')
            error('LocalEKF:MissingPerRadarPCRLBInformation', ...
                ['calculate_pcrlb_v5 must return ', ...
                 'info.JmeasByRadar with size 4-by-4-by-N-by-T.']);
        end

        [thisLocalPCRLB, thisAllPCRLB] = ...
            calculate_local_pcrlb_from_information( ...
                P0, F, Q, thisPCRLBInfo.JmeasByRadar, sensorSets);

        localPCRLBSize = [size(thisLocalPCRLB,1), size(thisLocalPCRLB,2), ...
            size(thisLocalPCRLB,3), size(thisLocalPCRLB,4), ...
            size(thisLocalPCRLB,5)];
        centralPCRLBSize = [size(thisAllPCRLB,1), ...
            size(thisAllPCRLB,2), size(thisAllPCRLB,3)];
        assert(isequal(localPCRLBSize, ...
                [nx, nx, numNodes, numSteps, numCR]), ...
            'LocalEKF:LocalPCRLBSize', ...
            'Unexpected local PCRLB size: %s.', mat2str(localPCRLBSize));
        assert(isequal(centralPCRLBSize, [nx, nx, numSteps]), ...
            'LocalEKF:CentralPCRLBSize', ...
            'Unexpected centralized PCRLB size: %s.', ...
            mat2str(centralPCRLBSize));

        pcrlbAgreement = max(abs( ...
            pcrlbAllDirect(:) - thisAllPCRLB(:)));
        pcrlbTolerance = 1e-9 * max(1, max(abs(pcrlbAllDirect(:))));
        assert(pcrlbAgreement <= pcrlbTolerance, ...
            'LocalEKF:CentralPCRLBMismatch', ...
            ['Central PCRLB reconstructed from per-radar information ', ...
             'differs from calculate_pcrlb_v5 by %.3e.'], pcrlbAgreement);

        pcrlbCovDLocal(:,:,:,:,snrIdx,:) = reshape(thisLocalPCRLB, ...
            [nx, nx, numNodes, numSteps, 1, numCR]);
        % The ego-neighborhood ADMM is a distributed solver for exactly the
        % same local information update, so its matched bound is P_local.
        pcrlbCovDConsensus(:,:,:,:,snrIdx,:) = reshape(thisLocalPCRLB, ...
            [nx, nx, numNodes, numSteps, 1, numCR]);
        pcrlbCovC(:,:,:,snrIdx) = thisAllPCRLB;
        pcrlbInfo{snrIdx} = thisPCRLBInfo;

        % Apply the shared base Gaussian measurement noise at this SNR.
        measurements = trueMeasurements;
        for mcIdx = 1:numMC
            for timeIdx = 1:numSteps
                for radarIdx = 1:numNodes
                    measurements(:,radarIdx,timeIdx,mcIdx) = ...
                        trueMeasurements(:,radarIdx,timeIdx,mcIdx) + ...
                        noiseFactor * baseMeasurementNoise( ...
                            :,radarIdx,timeIdx,mcIdx);
                end
            end
        end

        %% Centralized EKF: computed once because it is independent of CR
        for mcIdx = 1:numMC
            xCentral = mu0;
            PCentral = P0;
            for timeIdx = 1:numSteps
                if timeIdx == 1
                    xPred = xCentral;
                    PPred = PCentral;
                else
                    [xPred, PPred] = predict_cv( ...
                        xCentral, PCentral, F, Q);
                end

                yAll = reshape( ...
                    measurements(:,:,timeIdx,mcIdx), 2*numNodes, 1);
                [xCentral, PCentral] = centralized_ekf_update_stack( ...
                    xPred, PPred, yAll, geometryTopology, env, modelsUt);
                assert_finite_spd_posterior( ...
                    xCentral, PCentral, 'C-EKF', snrIdx, 0, mcIdx, timeIdx);

                estimatesC(:,timeIdx,mcIdx,snrIdx) = xCentral;
                diagPC(:,timeIdx,mcIdx,snrIdx) = diag(PCentral);
                minimumPosteriorEigenvalueC(snrIdx) = min( ...
                    minimumPosteriorEigenvalueC(snrIdx), ...
                    min(eig(0.5 * (PCentral + PCentral.'))));
            end
        end

        %% Independent local-neighborhood D-EKF for every CR
        for crIdx = 1:numCR
            fprintf('  communication radius %g (%d/%d), sensors/node=%s\n', ...
                config.CommunicationRadii(crIdx), crIdx, numCR, ...
                mat2str(neighborhoodSize(crIdx,:)));
            topology = topologies(crIdx);

            for mcIdx = 1:numMC
                xNode = repmat({mu0}, 1, numNodes);
                PNode = repmat({P0}, 1, numNodes);

                for timeIdx = 1:numSteps
                    xNext = cell(1, numNodes);
                    PNext = cell(1, numNodes);

                    for nodeIdx = 1:numNodes
                        if timeIdx == 1
                            xPred = xNode{nodeIdx};
                            PPred = PNode{nodeIdx};
                        else
                            [xPred, PPred] = predict_cv( ...
                                xNode{nodeIdx}, PNode{nodeIdx}, F, Q);
                        end

                        idxSet = sensorSets{crIdx,nodeIdx};
                        yLocal = reshape( ...
                            measurements(:,idxSet,timeIdx,mcIdx), ...
                            2*numel(idxSet), 1);
                        [xDirect, PDirect] = ...
                            dkf_neighbor_update( ...
                                xPred, PPred, yLocal, idxSet, ...
                                topology, env, modelsUt);
                        assert_finite_spd_posterior( ...
                            xDirect, PDirect, 'direct local D-EKF', ...
                            snrIdx, crIdx, mcIdx, timeIdx);

                        estimatesDLocal(:,nodeIdx,timeIdx,mcIdx,snrIdx,crIdx) = ...
                            xDirect;
                        diagPDLocal(:,nodeIdx,timeIdx,mcIdx,snrIdx,crIdx) = ...
                            diag(PDirect);
                        minimumLocalPosteriorEigenvalueD(snrIdx,crIdx) = min( ...
                            minimumLocalPosteriorEigenvalueD(snrIdx,crIdx), ...
                            min(eig(0.5 * ...
                                (PDirect + PDirect.'))));

                        % Run nonlinear state-variable consensus ADMM only
                        % inside this center node's ego neighborhood. Each
                        % participant contributes its nonlinear likelihood
                        % and 1/J of the common predicted prior. Covariance
                        % uses a separate information average consensus at
                        % the converged nonlinear MAP state.
                        yBySensor = measurements(:,idxSet,timeIdx,mcIdx);
                        [xNext{nodeIdx}, PNext{nodeIdx}, consensusInfo] = ...
                            ego_neighborhood_admm_nonlinear_map_update( ...
                                xPred, PPred, yBySensor, idxSet, ...
                                topology, env, modelsUt, config);
                        admmIterations(nodeIdx,timeIdx,mcIdx,snrIdx,crIdx) = ...
                            consensusInfo.iterations;
                        admmPrimalResidual(nodeIdx,timeIdx,mcIdx,snrIdx,crIdx) = ...
                            consensusInfo.primal_residual;
                        admmDualResidual(nodeIdx,timeIdx,mcIdx,snrIdx,crIdx) = ...
                            consensusInfo.dual_residual;
                        admmNodeDisagreement( ...
                            nodeIdx,timeIdx,mcIdx,snrIdx,crIdx) = ...
                            consensusInfo.node_disagreement;
                        covarianceConsensusIterations( ...
                            nodeIdx,timeIdx,mcIdx,snrIdx,crIdx) = ...
                            consensusInfo.covariance_iterations;
                        nonlinearSolverFailureCount( ...
                            nodeIdx,timeIdx,mcIdx,snrIdx,crIdx) = ...
                            consensusInfo.nonlinear_solver_failure_count;

                        assert_finite_spd_posterior( ...
                            xNext{nodeIdx}, PNext{nodeIdx}, ...
                            'ego-neighborhood ADMM D-EKF', snrIdx, crIdx, ...
                            mcIdx, timeIdx);
                        estimatesD(:,nodeIdx,timeIdx,mcIdx,snrIdx,crIdx) = ...
                            xNext{nodeIdx};
                        diagPD(:,nodeIdx,timeIdx,mcIdx,snrIdx,crIdx) = ...
                            diag(PNext{nodeIdx});
                        minimumPosteriorEigenvalueD(snrIdx,crIdx) = min( ...
                            minimumPosteriorEigenvalueD(snrIdx,crIdx), ...
                            min(eig(0.5 * ...
                            (PNext{nodeIdx} + PNext{nodeIdx}.'))));
                    end

                    % Each center node feeds back only its own ego-consensus
                    % posterior. There is deliberately no 10-node global
                    % consensus across the different ego neighborhoods.
                    xNode = xNext;
                    PNode = PNext;
                end
            end
        end
    end

    %% Pooled RMSE and node-matched PCRLB summaries
    rmseD = zeros(nx, numSteps, numSNR, numCR);
    rmseDLocal = zeros(nx, numSteps, numSNR, numCR);
    rmseC = zeros(nx, numSteps, numSNR);
    pcrlbD = zeros(nx, numSteps, numSNR, numCR);
    pcrlbDLocal = zeros(nx, numSteps, numSNR, numCR);
    pcrlbC = zeros(nx, numSteps, numSNR);

    for snrIdx = 1:numSNR
        for timeIdx = 1:numSteps
            truthAtTime = reshape(truth(:,timeIdx,:), [nx, 1, numMC]);
            centralError = reshape( ...
                estimatesC(:,timeIdx,:,snrIdx), [nx, numMC]) - ...
                reshape(truthAtTime, [nx, numMC]);
            rmseC(:,timeIdx,snrIdx) = sqrt(mean(centralError.^2, 2));
            pcrlbC(:,timeIdx,snrIdx) = sqrt(max( ...
                diag(pcrlbCovC(:,:,timeIdx,snrIdx)), 0));

            for crIdx = 1:numCR
                nodeError = reshape( ...
                    estimatesD(:,:,timeIdx,:,snrIdx,crIdx), ...
                    [nx, numNodes, numMC]) - truthAtTime;
                rmseD(:,timeIdx,snrIdx,crIdx) = sqrt( ...
                    mean(reshape(nodeError.^2, nx, []), 2));

                localNodeError = reshape( ...
                    estimatesDLocal(:,:,timeIdx,:,snrIdx,crIdx), ...
                    [nx, numNodes, numMC]) - truthAtTime;
                rmseDLocal(:,timeIdx,snrIdx,crIdx) = sqrt( ...
                    mean(reshape(localNodeError.^2, nx, []), 2));

                localVariance = zeros(nx, numNodes);
                for nodeIdx = 1:numNodes
                    localVariance(:,nodeIdx) = diag( ...
                        pcrlbCovDLocal(:,:,nodeIdx,timeIdx,snrIdx,crIdx));
                end
                pcrlbDLocal(:,timeIdx,snrIdx,crIdx) = sqrt(max( ...
                    mean(localVariance, 2), 0));
                pcrlbD(:,timeIdx,snrIdx,crIdx) = ...
                    pcrlbDLocal(:,timeIdx,snrIdx,crIdx);
            end
        end
    end

    % Reshape explicitly so singleton test configurations retain the public
    % dimensional contract required by the plotting helper.
    rmseD = reshape(rmseD, [nx, numSteps, numSNR, numCR]);
    rmseDLocal = reshape(rmseDLocal, [nx, numSteps, numSNR, numCR]);
    rmseC = reshape(rmseC, [nx, numSteps, numSNR]);
    pcrlbD = reshape(pcrlbD, [nx, numSteps, numSNR, numCR]);
    pcrlbDLocal = reshape(pcrlbDLocal, [nx, numSteps, numSNR, numCR]);
    pcrlbC = reshape(pcrlbC, [nx, numSteps, numSNR]);

    %% Common pre-measurement prior sample for plotting
    % The 64 entries above are measurement posteriors.  Keep them unchanged
    % for numerical analysis, and prepend a separate t=0 prior point only in
    % the plotting summaries.  Consequently the first posterior is plotted
    % at dt instead of being mislabeled as the pre-update t=0 state.
    initialTruth = reshape(truth(:,1,:), [nx, numMC]);
    initialError = mu0 - initialTruth;
    initialPriorRMSE = sqrt(mean(initialError.^2, 2));
    initialPriorStd = sqrt(max(diag(P0), 0));

    commonPriorD = repmat(reshape(initialPriorRMSE, [nx, 1, 1, 1]), ...
        [1, 1, numSNR, numCR]);
    commonPriorC = repmat(reshape(initialPriorRMSE, [nx, 1, 1]), ...
        [1, 1, numSNR]);
    commonBoundD = repmat(reshape(initialPriorStd, [nx, 1, 1, 1]), ...
        [1, 1, numSNR, numCR]);
    commonBoundC = repmat(reshape(initialPriorStd, [nx, 1, 1]), ...
        [1, 1, numSNR]);

    plotRmseD = cat(2, commonPriorD, rmseD);
    plotRmseC = cat(2, commonPriorC, rmseC);
    plotPcrlbD = cat(2, commonBoundD, pcrlbD);
    plotPcrlbC = cat(2, commonBoundC, pcrlbC);
    plotTime = (0:numSteps) * dt;

    %% Validation invariants
    fullNeighborhoodConfigurations = find(all(neighborhoodSize == numNodes, 2));
    fullNeighborhoodStateDifference = nan(1, numCR);
    fullNeighborhoodCovarianceDifference = nan(1, numCR);
    fullNeighborhoodPCRLBDifference = nan(1, numCR);
    for fullIdx = reshape(fullNeighborhoodConfigurations, 1, [])
        centralExpanded = reshape(estimatesC, ...
            [nx, 1, numSteps, numMC, numSNR]);
        stateDifference = estimatesD(:,:,:,:,:,fullIdx) - centralExpanded;
        fullNeighborhoodStateDifference(fullIdx) = ...
            max(abs(stateDifference(:)));

        centralDiagExpanded = reshape(diagPC, ...
            [nx, 1, numSteps, numMC, numSNR]);
        covarianceDifference = diagPD(:,:,:,:,:,fullIdx) - ...
            centralDiagExpanded;
        fullNeighborhoodCovarianceDifference(fullIdx) = ...
            max(abs(covarianceDifference(:)));

        consensusBound = pcrlbCovDConsensus(:,:,:,:,:,fullIdx);
        centralBound = reshape(pcrlbCovC, ...
            [nx, nx, 1, numSteps, numSNR]);
        boundDifference = consensusBound - centralBound;
        fullNeighborhoodPCRLBDifference(fullIdx) = ...
            max(abs(boundDifference(:)));

        % Nonlinear MAP and a one-linearization C-EKF need not agree, even
        % when both use all sensors. Keep their difference as a diagnostic
        % instead of asserting equality between different estimators.
        boundScale = max(1, max(abs(centralBound(:))));
        assert(fullNeighborhoodPCRLBDifference(fullIdx) <= ...
                1e-9 * boundScale, ...
            'LocalEKF:FullNeighborhoodPCRLBMismatch', ...
            ['Full-neighborhood ADMM-consensus D-PCRLB and C-PCRLB ', ...
             'differ by %.3e for communication radius %g.'], ...
            fullNeighborhoodPCRLBDifference(fullIdx), ...
            config.CommunicationRadii(fullIdx));
    end

    %% Return a self-contained study artifact
    Study = struct();
    Study.kind = 'ego_neighborhood_nonlinear_MAP_ADMM_vs_C_EKF';
    Study.version = 5;
    Study.config = config;
    % Legacy posterior-only time vector.  Use plot_time with the *_plot
    % fields when a common pre-measurement prior point is required.
    Study.time = (0:numSteps-1) * dt;
    Study.posterior_time = (1:numSteps) * dt;
    Study.plot_time = plotTime;
    Study.mu0 = mu0;
    Study.P0 = P0;
    Study.F = F;
    Study.G = G;
    Study.Q = Q;
    Study.processAccelerationVariance = processAccelerationVariance;
    Study.physical = physical;
    Study.radarPositions = radarPositions;
    Study.topologies = topologies;
    Study.sensor_sets = sensorSets;
    Study.degree = degree;
    Study.neighborhood_size = neighborhoodSize;
    Study.environments = environments;

    Study.truth = truth;
    % Main D fields contain nonlinear state-consensus MAP results. Different
    % ego centers are not forced to agree. The *_local fields retain the
    % one-linearization stacked-EKF estimates as a comparison baseline; they
    % are not expected to equal nonlinear MAP.
    Study.estimates_D = estimatesD;
    Study.diagP_D = diagPD;
    Study.estimates_D_local = estimatesDLocal;
    Study.diagP_D_local = diagPDLocal;
    Study.estimates_C = estimatesC;
    Study.diagP_C = diagPC;

    Study.rmse_D = rmseD;
    Study.rmse_D_local = rmseDLocal;
    Study.rmse_C = rmseC;
    Study.PCRLB_D_nodes = pcrlbCovDLocal;
    Study.PCRLB_D_nodes_local = pcrlbCovDLocal;
    Study.PCRLB_D_consensus = pcrlbCovDConsensus;
    Study.PCRLB_C = pcrlbCovC;
    Study.pcrlb_D = pcrlbD;
    Study.pcrlb_D_local = pcrlbDLocal;
    Study.pcrlb_C = pcrlbC;
    Study.PCRLB_info = pcrlbInfo;
    Study.ADMM = struct( ...
        'method', 'nonlinear_state_consensus_MAP_per_ego_neighborhood', ...
        'covariance_method', ...
            'average_consensus_on_Gauss_Newton_information', ...
        'iterations', admmIterations, ...
        'primal_residual', admmPrimalResidual, ...
        'dual_residual', admmDualResidual, ...
        'node_disagreement', admmNodeDisagreement, ...
        'covariance_iterations', covarianceConsensusIterations, ...
        'nonlinear_solver_failure_count', nonlinearSolverFailureCount);

    Study.initial_prior = struct( ...
        'mu0', mu0, ...
        'P0', P0, ...
        'empirical_rmse', initialPriorRMSE, ...
        'pcrlb_std', initialPriorStd);
    Study.rmse_D_plot = plotRmseD;
    Study.rmse_C_plot = plotRmseC;
    Study.pcrlb_D_plot = plotPcrlbD;
    Study.pcrlb_C_plot = plotPcrlbC;

    Study.validation = struct();
    Study.validation.minimumPosteriorEigenvalue_D = ...
        minimumPosteriorEigenvalueD;
    Study.validation.minimumLocalPosteriorEigenvalue_D = ...
        minimumLocalPosteriorEigenvalueD;
    Study.validation.minimumPosteriorEigenvalue_C = ...
        minimumPosteriorEigenvalueC;
    Study.validation.fullNeighborhoodConfigurations = ...
        fullNeighborhoodConfigurations(:).';
    Study.validation.fullNeighborhoodStateDifference = ...
        fullNeighborhoodStateDifference;
    Study.validation.fullNeighborhoodCovarianceDifference = ...
        fullNeighborhoodCovarianceDifference;
    Study.validation.fullNeighborhoodPCRLBDifference = ...
        fullNeighborhoodPCRLBDifference;
    Study.validation.centralComputedOncePerSNR = true;
    Study.validation.truthAndBaseNoisePairedAcrossConfigurations = true;
    Study.created_at = char(datetime('now', 'Format', ...
        'yyyy-MM-dd HH:mm:ss.SSS Z'));
    Study.saved_file = '';

    outputDir = char(config.OutputDir);
    if config.MakePlots || config.SaveResults
        if ~exist(outputDir, 'dir')
            mkdir(outputDir);
        end
    end

    if config.MakePlots
        plot_local_neighborhood_ekf_study(Study, outputDir);
    end

    if config.SaveResults
        timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
        resultFile = fullfile(outputDir, ...
            sprintf('local_neighborhood_ekf_study_%s.mat', timestamp));
        Study.saved_file = resultFile;
        save(resultFile, 'Study', '-v7.3');
        fprintf('Saved local-neighborhood EKF study: %s\n', resultFile);
    end
end

function config = apply_defaults_and_validate(config, scriptDir)
    if ~isstruct(config) || ~isscalar(config)
        error('LocalEKF:ConfigType', 'config must be a scalar struct.');
    end

    defaults = struct();
    defaults.SNRdB = [20, 30, 40];
    defaults.CommunicationRadii = [30, 50, 500];
    defaults.NumMonteCarlo = 100;
    defaults.NumSteps = 64;
    defaults.Dt = 0.01;
    defaults.NumDopplerSamples = 64;
    defaults.NumPCRLBSamples = 2000;
    defaults.Seed = 43;
    defaults.PCRLBSeed = 100043;
    defaults.ADMMMaxIter = 1000;
    defaults.ADMMAbsTolerance = 1e-8;
    defaults.ADMMRelTolerance = 1e-7;
    defaults.ADMMRho = 1;
    defaults.NonlinearSolverMaxIterations = 100;
    defaults.MakePlots = true;
    defaults.SaveResults = true;
    defaults.OutputDir = fullfile( ...
        scriptDir, 'results', 'local_neighborhood_ekf');

    suppliedFields = fieldnames(config);
    validFields = fieldnames(defaults);
    unknownFields = setdiff(suppliedFields, validFields);
    if ~isempty(unknownFields)
        error('LocalEKF:UnknownConfigField', ...
            'Unknown config field(s): %s.', strjoin(unknownFields, ', '));
    end
    for fieldIdx = 1:numel(validFields)
        fieldName = validFields{fieldIdx};
        if ~isfield(config, fieldName) || isempty(config.(fieldName))
            config.(fieldName) = defaults.(fieldName);
        end
    end

    validateattributes(config.SNRdB, {'numeric'}, ...
        {'real', 'finite', 'vector', 'nonempty'}, mfilename, 'config.SNRdB');
    validateattributes(config.CommunicationRadii, {'numeric'}, ...
        {'real', 'finite', 'positive', 'vector', 'nonempty'}, ...
        mfilename, 'config.CommunicationRadii');
    validateattributes(config.NumMonteCarlo, {'numeric'}, ...
        {'real', 'finite', 'scalar', 'integer', 'positive'}, ...
        mfilename, 'config.NumMonteCarlo');
    validateattributes(config.NumSteps, {'numeric'}, ...
        {'real', 'finite', 'scalar', 'integer', 'positive'}, ...
        mfilename, 'config.NumSteps');
    validateattributes(config.Dt, {'numeric'}, ...
        {'real', 'finite', 'scalar', 'positive'}, mfilename, 'config.Dt');
    validateattributes(config.NumDopplerSamples, {'numeric'}, ...
        {'real', 'finite', 'scalar', 'integer', 'positive'}, ...
        mfilename, 'config.NumDopplerSamples');
    validateattributes(config.NumPCRLBSamples, {'numeric'}, ...
        {'real', 'finite', 'scalar', 'integer', 'positive'}, ...
        mfilename, 'config.NumPCRLBSamples');
    validateattributes(config.Seed, {'numeric'}, ...
        {'real', 'finite', 'scalar', 'integer', '>=', 0, '<=', 2^32-1}, ...
        mfilename, 'config.Seed');
    validateattributes(config.PCRLBSeed, {'numeric'}, ...
        {'real', 'finite', 'scalar', 'integer', '>=', 0, '<=', 2^32-1}, ...
        mfilename, 'config.PCRLBSeed');
    validateattributes(config.ADMMMaxIter, {'numeric'}, ...
        {'real', 'finite', 'scalar', 'integer', 'positive'}, ...
        mfilename, 'config.ADMMMaxIter');
    validateattributes(config.ADMMAbsTolerance, {'numeric'}, ...
        {'real', 'finite', 'scalar', 'positive'}, ...
        mfilename, 'config.ADMMAbsTolerance');
    validateattributes(config.ADMMRelTolerance, {'numeric'}, ...
        {'real', 'finite', 'scalar', 'positive'}, ...
        mfilename, 'config.ADMMRelTolerance');
    validateattributes(config.ADMMRho, {'numeric'}, ...
        {'real', 'finite', 'scalar', 'positive'}, ...
        mfilename, 'config.ADMMRho');
    validateattributes(config.NonlinearSolverMaxIterations, {'numeric'}, ...
        {'real', 'finite', 'scalar', 'integer', 'positive'}, ...
        mfilename, 'config.NonlinearSolverMaxIterations');
    validateattributes(config.MakePlots, {'logical'}, {'scalar'}, ...
        mfilename, 'config.MakePlots');
    validateattributes(config.SaveResults, {'logical'}, {'scalar'}, ...
        mfilename, 'config.SaveResults');
    if ~(ischar(config.OutputDir) || ...
            (isstring(config.OutputDir) && isscalar(config.OutputDir)))
        error('LocalEKF:OutputDirType', ...
            'config.OutputDir must be a character vector or string scalar.');
    end

    config.SNRdB = reshape(double(config.SNRdB), 1, []);
    config.CommunicationRadii = reshape( ...
        double(config.CommunicationRadii), 1, []);
    config.OutputDir = char(config.OutputDir);
end

function validate_dependency_paths(scriptDir, makePlots)
    dependencies = {'ADMM_utils', 'models', 'dkf_neighbor_update', ...
        'centralized_ekf_update_stack', 'calculate_pcrlb_v5', ...
        'calculate_local_pcrlb_from_information'};
    if makePlots
        dependencies{end+1} = 'plot_local_neighborhood_ekf_study';
    end

    for dependencyIdx = 1:numel(dependencies)
        dependency = dependencies{dependencyIdx};
        expected = fullfile(scriptDir, [dependency, '.m']);
        resolved = which(dependency);
        if ~strcmp(resolved, expected)
            error('LocalEKF:PathShadowing', ...
                ['MATLAB resolved %s to:\n  %s\nExpected:\n  %s\n', ...
                 'Change the current folder to ADMM-EKF, run ', ...
                 'clear classes, and retry.'], ...
                dependency, resolved, expected);
        end
    end
end

function [xPost, PPost, info] = ego_neighborhood_admm_nonlinear_map_update( ...
        xPred, PPred, yBySensor, idxSet, topology, env, modelsUt, config)
%EGO_NEIGHBORHOOD_ADMM_NONLINEAR_MAP_UPDATE Nonlinear state consensus MAP.
% Participant a owns one nonlinear sensor likelihood and 1/J of the common
% predicted prior. Edge-variable ADMM enforces equality of the participant
% state variables. Covariance is a separate average consensus over Laplace/
% Gauss-Newton information packets evaluated at the converged MAP state.

    nx = numel(xPred);
    numParticipants = numel(idxSet);
    if ~isequal(size(yBySensor), [2, numParticipants])
        error('LocalEKF:EgoMeasurementSize', ...
            'yBySensor must be 2-by-numel(idxSet).');
    end

    inducedAdjacency = topology.adj_matrix(idxSet, idxSet);
    inducedAdjacency = double(inducedAdjacency ~= 0);
    inducedAdjacency(1:numParticipants+1:end) = 0;
    if numParticipants > 1 && ...
            numel(unique(conncomp(graph(inducedAdjacency)))) ~= 1
        error('LocalEKF:ADMMDisconnectedEgoGraph', ...
            'The ego-neighborhood induced graph must be connected.');
    end

    PPred = 0.5 * (PPred + PPred.');
    JPrior = PPred \ eye(nx);
    JPrior = 0.5 * (JPrior + JPrior.');
    % State-scaled penalties improve conditioning without changing the MAP
    % optimum. These are the position/velocity scales used by the nonlinear
    % ADMM implementation elsewhere in this project.
    initialRho = config.ADMMRho * [100; 100; 15; 15];
    rho = initialRho;
    state = repmat(xPred(:), 1, numParticipants);
    z = zeros(nx, numParticipants, numParticipants);
    dual = zeros(nx, numParticipants, numParticipants);
    for participantIdx = 1:numParticipants
        for neighborIdx = find(inducedAdjacency(participantIdx,:) > 0)
            z(:,participantIdx,neighborIdx) = xPred(:);
        end
    end

    solver = optimoptions('lsqnonlin', ...
        'Display', 'off', ...
        'Algorithm', 'trust-region-reflective', ...
        'SpecifyObjectiveGradient', true, ...
        'MaxIterations', config.NonlinearSolverMaxIterations, ...
        'MaxFunctionEvaluations', 1000, ...
        'OptimalityTolerance', 1e-8, ...
        'StepTolerance', 1e-10);
    numDirectedEdges = nnz(inducedAdjacency);
    primalResidual = inf;
    dualResidual = inf;
    epsPrimal = inf;
    epsDual = inf;
    converged = false;
    nonlinearSolverFailureCount = 0;
    nonlinearSolverMaxIterations = 0;

    restarted = false;
    for iteration = 1:(2 * config.ADMMMaxIter)
        if iteration == config.ADMMMaxIter + 1
            % A strongly nonlinear high-SNR likelihood can trap different
            % participants in different local modes. Restart once from their
            % common mean with zero duals and a stronger consensus penalty.
            % This changes neither the objective nor the consensus constraint.
            commonSeed = mean(state, 2);
            state = repmat(commonSeed, 1, numParticipants);
            dual(:) = 0;
            rho = 10 * initialRho;
            for participantIdx = 1:numParticipants
                for neighborIdx = find( ...
                        inducedAdjacency(participantIdx,:) > 0)
                    z(:,participantIdx,neighborIdx) = commonSeed;
                end
            end
            restarted = true;
        end
        zPrevious = z;

        % Nonlinear x update: one true range-Doppler likelihood per
        % participant, 1/J of the prior, and its incident ADMM terms.
        for participantIdx = 1:numParticipants
            neighbors = find(inducedAdjacency(participantIdx,:) > 0);
            radarIdx = idxSet(participantIdx);
            objective = @(x) nonlinear_local_map_augmented_residual( ...
                x, xPred, JPrior, yBySensor(:,participantIdx), radarIdx, ...
                topology, env, modelsUt, numParticipants, neighbors, ...
                squeeze(dual(:,participantIdx,:)), ...
                squeeze(z(:,participantIdx,:)), rho);
            [state(:,participantIdx), ~, ~, exitflag, output] = lsqnonlin( ...
                objective, state(:,participantIdx), [], [], solver);
            nonlinearSolverFailureCount = ...
                nonlinearSolverFailureCount + (exitflag <= 0);
            if isfield(output, 'iterations')
                nonlinearSolverMaxIterations = max( ...
                    nonlinearSolverMaxIterations, output.iterations);
            end
        end

        % Shared edge-variable update using lambda^k.
        zNew = z;
        for participantIdx = 1:numParticipants
            for neighborIdx = find(inducedAdjacency(participantIdx,:) > 0)
                zNew(:,participantIdx,neighborIdx) = 0.5 * ( ...
                    state(:,participantIdx) + state(:,neighborIdx) + ...
                    (dual(:,participantIdx,neighborIdx) + ...
                     dual(:,neighborIdx,participantIdx)) ./ rho);
            end
        end

        % Unscaled dual update and residuals in their matching spaces.
        dualNew = dual;
        primalSquared = 0;
        dualSquared = 0;
        xNormSquared = 0;
        zNormSquared = 0;
        dualNormSquared = 0;
        for participantIdx = 1:numParticipants
            dualNodeResidual = zeros(nx,1);
            lambdaNode = zeros(nx,1);
            for neighborIdx = find(inducedAdjacency(participantIdx,:) > 0)
                residual = state(:,participantIdx) - ...
                    zNew(:,participantIdx,neighborIdx);
                dualNew(:,participantIdx,neighborIdx) = ...
                    dual(:,participantIdx,neighborIdx) + rho .* residual;
                primalSquared = primalSquared + sum(residual.^2);
                xNormSquared = xNormSquared + sum( ...
                    state(:,participantIdx).^2);
                zNormSquared = zNormSquared + sum( ...
                    zNew(:,participantIdx,neighborIdx).^2);
                dualNodeResidual = dualNodeResidual + rho .* ( ...
                    zNew(:,participantIdx,neighborIdx) - ...
                    zPrevious(:,participantIdx,neighborIdx));
                lambdaNode = lambdaNode + ...
                    dualNew(:,participantIdx,neighborIdx);
            end
            dualSquared = dualSquared + sum(dualNodeResidual.^2);
            dualNormSquared = dualNormSquared + sum(lambdaNode.^2);
        end
        z = zNew;
        dual = dualNew;

        primalResidual = sqrt(primalSquared);
        dualResidual = sqrt(dualSquared);
        epsPrimal = sqrt(nx * numDirectedEdges) * ...
            config.ADMMAbsTolerance + config.ADMMRelTolerance * max( ...
                sqrt(xNormSquared), sqrt(zNormSquared));
        epsDual = sqrt(nx * numParticipants) * ...
            config.ADMMAbsTolerance + config.ADMMRelTolerance * ...
                sqrt(dualNormSquared);
        converged = primalResidual <= epsPrimal && ...
            dualResidual <= epsDual;
        if converged
            break;
        elseif mod(iteration, 10) == 0
            % Residual balancing for the unscaled dual variable. A fixed
            % rho is especially slow here because range and Doppler give
            % the four state coordinates very different curvature.
            if primalResidual > 10 * dualResidual
                rho = min(2 * rho, 1e8);
            elseif dualResidual > 10 * primalResidual
                rho = max(rho / 2, 1e-8);
            end
        end
    end

    if ~converged
        error('LocalEKF:NonlinearADMMNotConverged', ...
            ['Nonlinear state ADMM did not converge in %d iterations ', ...
             '(r=%.3e/%.3e, s=%.3e/%.3e).'], ...
            2 * config.ADMMMaxIter, primalResidual, epsPrimal, ...
            dualResidual, epsDual);
    end
    xPost = mean(state, 2);
    nodeDisagreement = max(abs(state - xPost), [], 'all');

    % Laplace/Gauss-Newton covariance at the nonlinear MAP solution. Each
    % participant creates one information packet; the existing Gaussian
    % routine performs average consensus on those packets only.
    xPacket = repmat({xPost}, 1, numParticipants);
    PPacket = cell(1, numParticipants);
    for participantIdx = 1:numParticipants
        radarIdx = idxSet(participantIdx);
        [~, HRadar] = modelsUt.MeasureModelAndJacobian( ...
            xPost, radarIdx, topology, env);
        JPacket = JPrior / numParticipants + ...
            HRadar.' * (env.Sigma \ HRadar);
        JPacket = 0.5 * (JPacket + JPacket.');
        PPacket{participantIdx} = JPacket \ eye(nx);
        PPacket{participantIdx} = 0.5 * ( ...
            PPacket{participantIdx} + PPacket{participantIdx}.');
    end
    [~, PGroup, covarianceInfo] = admm_gaussian_average_consensus( ...
        xPacket, PPacket, inducedAdjacency, config);
    PPost = PGroup{1} / numParticipants;
    PPost = 0.5 * (PPost + PPost.');
    info = struct( ...
        'iterations', iteration, ...
        'converged', converged, ...
        'primal_residual', primalResidual, ...
        'dual_residual', dualResidual, ...
        'eps_primal', epsPrimal, ...
        'eps_dual', epsDual, ...
        'node_disagreement', nodeDisagreement, ...
        'neighborhood_size', numParticipants, ...
        'nonlinear_solver_failure_count', nonlinearSolverFailureCount, ...
        'nonlinear_solver_max_iterations', nonlinearSolverMaxIterations, ...
        'rho_final', rho, ...
        'restarted', restarted, ...
        'covariance_iterations', covarianceInfo.iterations, ...
        'covariance_primal_residual', ...
            covarianceInfo.primal_residual, ...
        'covariance_dual_residual', covarianceInfo.dual_residual);
end

function [residual, jacobian] = nonlinear_local_map_augmented_residual( ...
        state, xPred, JPrior, measurement, radarIdx, topology, env, ...
        modelsUt, numParticipants, neighbors, dualForNode, zForNode, rho)
% Exact nonlinear MAP augmented objective written as least squares. Completing
% the square converts each linear-dual plus quadratic-penalty term without
% changing its state minimizer; constants independent of state are omitted.

    state = state(:);
    priorDelta = state - xPred(:);
    [predictedMeasurement, H] = modelsUt.MeasureModelAndJacobian( ...
        state, radarIdx, topology, env);
    innovation = measurement(:) - predictedMeasurement;
    priorFactor = chol(JPrior / numParticipants);
    measurementFactor = chol(env.Sigma \ eye(size(env.Sigma)));
    residual = [priorFactor * priorDelta; ...
        measurementFactor * innovation];
    jacobian = [priorFactor; -measurementFactor * H];

    for neighborIdx = neighbors
        delta = state - zForNode(:,neighborIdx);
        lambda = dualForNode(:,neighborIdx);
        penaltyRoot = sqrt(rho(:));
        residual = [residual; penaltyRoot .* ...
            (delta + lambda ./ rho(:))]; %#ok<AGROW>
        jacobian = [jacobian; diag(penaltyRoot)]; %#ok<AGROW>
    end
end

function [xConsensus, PConsensus, info] = ...
        admm_gaussian_average_consensus(xLocal, PLocal, adjacency, config)
%ADMM_GAUSSIAN_AVERAGE_CONSENSUS Average Gaussian natural parameters.
% Each local posterior N(x_i,P_i) is represented by eta_i=P_i\x_i and
% J_i=P_i\I. Edge-variable ADMM solves the equal-weight average-consensus
% problem for [eta_i;vec(J_i)]. Averaging natural parameters is equivalent
% to an equal-weight logarithmic opinion pool and keeps state/covariance
% internally consistent; directly averaging covariance matrices does not.

    numNodes = numel(xLocal);
    nx = numel(xLocal{1});
    adjacency = double(adjacency ~= 0);
    adjacency(1:numNodes+1:end) = 0;
    if ~isequal(size(adjacency), [numNodes, numNodes]) || ...
            norm(adjacency - adjacency.', 'fro') > 0
        error('LocalEKF:ADMMAdjacency', ...
            'ADMM adjacency must be a symmetric N-by-N matrix.');
    end
    if numNodes > 1 && numel(unique(conncomp(graph(adjacency)))) ~= 1
        error('LocalEKF:ADMMDisconnectedGraph', ...
            'Gaussian ADMM consensus requires a connected graph.');
    end

    natural = zeros(numNodes, nx + nx^2);
    for nodeIdx = 1:numNodes
        Pi = 0.5 * (PLocal{nodeIdx} + PLocal{nodeIdx}.');
        [~, flag] = chol(Pi, 'lower');
        if flag ~= 0
            error('LocalEKF:ADMMInputCovariance', ...
                'Local covariance at node %d is not SPD.', nodeIdx);
        end
        Ji = Pi \ eye(nx);
        Ji = 0.5 * (Ji + Ji.');
        eta = Ji * xLocal{nodeIdx}(:);
        natural(nodeIdx,:) = [eta; Ji(:)].';
    end

    % Component normalization improves ADMM conditioning without changing
    % any coordinate's arithmetic-average solution.
    componentScale = max(max(abs(natural), [], 1), 1);
    localValue = natural ./ componentScale;
    numComponents = size(localValue, 2);
    targetNatural = mean(natural, 1);

    u = localValue;
    z = zeros(numNodes, numNodes, numComponents);
    dual = zeros(numNodes, numNodes, numComponents);
    for nodeIdx = 1:numNodes
        for neighborIdx = find(adjacency(nodeIdx,:) > 0)
            z(nodeIdx,neighborIdx,:) = reshape(0.5 * ( ...
                u(nodeIdx,:) + u(neighborIdx,:)), 1, 1, []);
        end
    end

    rho = config.ADMMRho;
    absTolerance = config.ADMMAbsTolerance;
    relTolerance = config.ADMMRelTolerance;
    maxIterations = config.ADMMMaxIter;
    numDirectedEdges = nnz(adjacency);
    primalResidual = 0;
    dualResidual = 0;
    epsPrimal = 0;
    epsDual = 0;
    converged = false;

    inputDisagreement = max(abs( ...
        localValue - mean(localValue, 1)), [], 'all');
    if inputDisagreement <= absTolerance
        converged = true;
        iteration = 0;
    else
        for iteration = 1:maxIterations
            zPrevious = z;

            % x/u update using the local quadratic objective and edge terms.
            for nodeIdx = 1:numNodes
                neighbors = find(adjacency(nodeIdx,:) > 0);
                numerator = localValue(nodeIdx,:);
                for neighborIdx = neighbors
                    numerator = numerator + rho * reshape( ...
                        z(nodeIdx,neighborIdx,:), 1, []) - reshape( ...
                        dual(nodeIdx,neighborIdx,:), 1, []);
                end
                u(nodeIdx,:) = numerator / (1 + rho * numel(neighbors));
            end

            % Edge-variable update with the previous unscaled dual value.
            zNew = z;
            for nodeIdx = 1:numNodes
                for neighborIdx = find(adjacency(nodeIdx,:) > 0)
                    dualPair = reshape(dual(nodeIdx,neighborIdx,:), 1, []) + ...
                        reshape(dual(neighborIdx,nodeIdx,:), 1, []);
                    zNew(nodeIdx,neighborIdx,:) = reshape(0.5 * ( ...
                        u(nodeIdx,:) + u(neighborIdx,:) + dualPair / rho), ...
                        1, 1, []);
                end
            end

            % Unscaled dual update.
            dualNew = dual;
            primalSquared = 0;
            xNormSquared = 0;
            zNormSquared = 0;
            dualSquared = 0;
            dualNormSquared = 0;
            for nodeIdx = 1:numNodes
                dualNode = zeros(1, numComponents);
                lambdaNode = zeros(1, numComponents);
                for neighborIdx = find(adjacency(nodeIdx,:) > 0)
                    zEdge = reshape(zNew(nodeIdx,neighborIdx,:), 1, []);
                    residualEdge = u(nodeIdx,:) - zEdge;
                    dualNew(nodeIdx,neighborIdx,:) = reshape( ...
                        reshape(dual(nodeIdx,neighborIdx,:), 1, []) + ...
                        rho * residualEdge, 1, 1, []);
                    primalSquared = primalSquared + sum(residualEdge.^2);
                    xNormSquared = xNormSquared + sum(u(nodeIdx,:).^2);
                    zNormSquared = zNormSquared + sum(zEdge.^2);
                    dualNode = dualNode + rho * (zEdge - reshape( ...
                        zPrevious(nodeIdx,neighborIdx,:), 1, []));
                    lambdaNode = lambdaNode + reshape( ...
                        dualNew(nodeIdx,neighborIdx,:), 1, []);
                end
                dualSquared = dualSquared + sum(dualNode.^2);
                dualNormSquared = dualNormSquared + sum(lambdaNode.^2);
            end
            z = zNew;
            dual = dualNew;

            primalResidual = sqrt(primalSquared);
            dualResidual = sqrt(dualSquared);
            epsPrimal = sqrt(numDirectedEdges * numComponents) * ...
                absTolerance + relTolerance * max( ...
                sqrt(xNormSquared), sqrt(zNormSquared));
            epsDual = sqrt(numNodes * numComponents) * absTolerance + ...
                relTolerance * sqrt(dualNormSquared);
            converged = primalResidual <= epsPrimal && ...
                dualResidual <= epsDual;
            if converged
                break;
            end
        end
    end

    if ~converged
        error('LocalEKF:ADMMConsensusNotConverged', ...
            ['Gaussian ADMM consensus did not converge in %d iterations ', ...
             '(r=%.3e/%.3e, s=%.3e/%.3e).'], ...
            maxIterations, primalResidual, epsPrimal, ...
            dualResidual, epsDual);
    end

    % All nodes use the same converged ADMM network value. Averaging the
    % final near-consensus node variables removes only residual roundoff;
    % it does not replace the distributed iterations above.
    commonNatural = mean(u, 1) .* componentScale;
    meanTargetError = max(abs(commonNatural - targetNatural));
    consensusScale = max(1, max(abs(targetNatural)));
    if meanTargetError > 20 * (absTolerance + relTolerance) * consensusScale
        error('LocalEKF:ADMMConsensusTarget', ...
            'ADMM common value differs from the arithmetic target by %.3e.', ...
            meanTargetError);
    end

    etaConsensus = commonNatural(1:nx).';
    JConsensus = reshape(commonNatural(nx+1:end), nx, nx);
    JConsensus = 0.5 * (JConsensus + JConsensus.');
    [~, flag] = chol(JConsensus, 'lower');
    if flag ~= 0
        error('LocalEKF:ADMMConsensusInformation', ...
            'Consensus information matrix is not SPD.');
    end
    PCommon = JConsensus \ eye(nx);
    PCommon = 0.5 * (PCommon + PCommon.');
    xCommon = JConsensus \ etaConsensus;
    xConsensus = repmat(xCommon, 1, numNodes);
    PConsensus = repmat({PCommon}, 1, numNodes);

    normalizedDisagreement = max(abs(u - mean(u, 1)), [], 'all');
    info = struct( ...
        'iterations', iteration, ...
        'converged', converged, ...
        'primal_residual', primalResidual, ...
        'dual_residual', dualResidual, ...
        'eps_primal', epsPrimal, ...
        'eps_dual', epsDual, ...
        'normalized_node_disagreement', normalizedDisagreement, ...
        'mean_target_error', meanTargetError);
end

function env = make_environment(snrDB, config, physical, Q)
    env = struct();
    env.c = physical.c;
    env.lambda = physical.lambda;
    env.time_step = config.Dt;
    env.B = physical.bandwidth * ones(1, 10);
    env.fs = physical.samplingFrequency * ones(1, 10);
    env.SNR_idx = snrDB;
    env.SNR_lin = 10^(snrDB/10);
    env.range_var = (3 * env.c^2) / ...
        (8 * pi^2 * physical.bandwidth^2 * env.SNR_lin);
    env.doppler_var = (3 * physical.samplingFrequency^2) / ...
        (pi^2 * env.SNR_lin * config.NumDopplerSamples^3);
    env.rho = physical.measurementCorrelation;
    crossCovariance = env.rho * sqrt(env.range_var * env.doppler_var);
    env.Sigma = [env.range_var, crossCovariance; ...
                 crossCovariance, env.doppler_var];
    env.PRE_WHITEN = false;
    env.pre_whit_L = eye(2);
    env.Sigma_filter = env.Sigma;
    env.Q = Q;
end

function [xPred, PPred] = predict_cv(x, P, F, Q)
    xPred = F * x;
    PPred = F * P * F.' + Q;
    PPred = 0.5 * (PPred + PPred.');
end

function assert_finite_spd_posterior( ...
        state, covariance, method, snrIdx, crIdx, mcIdx, timeIdx)
    if any(~isfinite(state(:))) || any(~isfinite(covariance(:)))
        error('LocalEKF:NonfinitePosterior', ...
            ['%s returned a nonfinite posterior at SNR index %d, ', ...
             'CR index %d, MC %d, time %d.'], ...
            method, snrIdx, crIdx, mcIdx, timeIdx);
    end

    covariance = 0.5 * (covariance + covariance.');
    [~, cholFlag] = chol(covariance, 'lower');
    if cholFlag ~= 0
        error('LocalEKF:PosteriorNotSPD', ...
            ['%s covariance is not SPD at SNR index %d, CR index %d, ', ...
             'MC %d, time %d (minimum eigenvalue %.3e).'], ...
            method, snrIdx, crIdx, mcIdx, timeIdx, ...
            min(eig(covariance)));
    end
end
