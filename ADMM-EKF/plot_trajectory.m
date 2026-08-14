clear; clc; close all;

matFile = '/Users/shaohung/Desktop/Desktop - K2D4GJQXL6/PhD/paper/DADMM/ADMM-EKF/data_log/testing_CRLB/dkf_30db/Log.mat';
cntrlmatFile = '/Users/shaohung/Desktop/Desktop - K2D4GJQXL6/PhD/paper/DADMM/ADMM-EKF/data_log/testing_CRLB/dkf_50db/Log.mat';
% matFile = '/Users/shaohung/Desktop/Desktop - K2D4GJQXL6/PhD/paper/DADMM/all_in_one/data_log/rMSE_SNR/MAP_50db_tracking/Log.mat';
% cntrlmatFile = '/Users/shaohung/Desktop/Desktop - K2D4GJQXL6/PhD/paper/DADMM/all_in_one/data_log/rMSE_SNR/MAP_50db_tracking/Log.mat';


S = load(matFile, 'Log');
SC = load(cntrlmatFile,'Log');
Log = S.Log;
LogSC = SC.Log;

network_topo = Log.network_topo;
numMc = size(Log.Results.estimations_DA_raw, 1);
numTime = size(Log.Results.estimations_DA_raw, 2);
mcIdx = randi(numMc);
fprintf('Plotting Monte Carlo run %d of %d\n', mcIdx, numMc);
trueTraj = squeeze(LogSC.target.target_position(1, 1:numTime, :));  % [time x 2]
% trueTraj = stackStateCells(Log.Results.true_params_raw(mcIdx, 1:numTime));
% trueTraj  = stackStateCells(Log.Results.true_params(mcIdx, 1:12));
distState = stackStateCells(Log.Results.estimations_DA_raw(mcIdx, 1:numTime));
% centState = stackStateCells(Log.Results.estimations_CA_raw(mcIdx, 1:numTime));
centState = stackStateCells(LogSC.Results.estimations_CA_raw(size(LogSC.Results.estimations_CA, 1),1:size(LogSC.Results.estimations_CA, 2)));


fig = figure;
set(fig, 'Color', 'white');

ax = axes(fig);
hold(ax, 'on');
set(ax, 'FontName', 'Times New Roman');
set(ax, 'FontSize', 25);
ax.LineWidth = 1.5;

% Plot communication links
linkHandle = gobjects(0);
for n = 1:network_topo.numNodes
    neighborsIdx = find(network_topo.laplacian_matrix(n, :) < 0);

    for jj = 1:numel(neighborsIdx)
        m = neighborsIdx(jj);

        % Avoid drawing each undirected link twice
        if m <= n
            continue;
        end

        h = plot( ...
            [network_topo.radar_pos(n, 1), network_topo.radar_pos(m, 1)], ...
            [network_topo.radar_pos(n, 2), network_topo.radar_pos(m, 2)], ...
            '-k', 'LineWidth', 1.5);

        if isempty(linkHandle)
            linkHandle = h;
            set(linkHandle, 'DisplayName', 'Communication link');
        end
    end
end

% Plot sensor nodes
sensorHandle = plot( ...
    network_topo.radar_pos(:, 1), ...
    network_topo.radar_pos(:, 2), ...
    'r.', ...
    'MarkerSize', 40, ...
    'DisplayName', 'Sensor Nodes');

% Plot true, distributed, and centralized trajectories
gtHandle = plot( ...
    trueTraj(:, 1), trueTraj(:, 2), ...
    '-.k', ...
    'LineWidth', 2, ...
    'MarkerSize', 20, ...
    'DisplayName', 'True Trajectory');

distHandle = plot( ...
    distState(:, 1), distState(:, 2), ...
    '--or', ...
    'LineWidth', 2, ...
    'MarkerSize', 10, ...
    'DisplayName', 'D-EKF');

centHandle = plot( ...
    centState(:, 1), centState(:, 2), ...
    '-sb', ...
    'LineWidth', 2, ...
    'MarkerSize', 10, ...
    'DisplayName', 'C-EKF');

% Reduce marker density
gtHandle.MarkerIndices = markerIndices(size(trueTraj, 1), 20);
distHandle.MarkerIndices = markerIndices(size(distState, 1), 20);
centHandle.MarkerIndices = markerIndices(size(centState, 1), 20);

xlabel('Position x (m)');
ylabel('Position y (m)');

grid on;
box on;
axis equal;

legendHandles = [gtHandle, distHandle, centHandle, sensorHandle];
if ~isempty(linkHandle)
    legendHandles = [legendHandles, linkHandle];
end
legend(legendHandles, 'Location', 'bestoutside');

% Optional export
% exportgraphics(fig, sprintf('trajectory_mc_%d.pdf', mcIdx), 'ContentType', 'vector');

function X = stackStateCells(C)
    C = C(:);
    C = C(~cellfun(@isempty, C));

    firstState = C{1};

    if isrow(firstState)
        X = cell2mat(C);          % [time x stateDim]
    elseif iscolumn(firstState)
        X = cell2mat(C.').';      % [time x stateDim]
    else
        error('Each estimator cell must contain a state vector.');
    end
end

function idx = markerIndices(numPoints, maxMarkers)
    step = max(1, floor(numPoints / maxMarkers));
    idx = 1:step:numPoints;
end