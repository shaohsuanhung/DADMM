clear; clc; close all;

%% File settings
matFile = "./0708_tracking/MAP_50db/log.mat";   % Replace with the full path when needed

% Select a Monte Carlo run manually, or leave empty for a random run.
mcIdx = [];

% Plot sensor nodes and communication links.
% For this dataset, the sensor network covers a much larger area than the
% short target trajectory, so false gives a clearer trajectory plot.
plotNetwork = true;

%% Load log structure
S = load(matFile);

% The uploaded file stores the structure as "log". This fallback also
% supports files that use "Log".
if isfield(S, 'log')
    Log = S.log;
elseif isfield(S, 'Log')
    Log = S.Log;
else
    error('The MAT-file must contain a structure named log or Log.');
end

estimations_DA = Log.Results.estimations_DA;
estimations_CA = Log.Results.estimations_CA;

numMc   = size(estimations_DA, 1);
numTime = size(estimations_DA, 2);

if isempty(mcIdx)
    mcIdx = randi(numMc);
end

validateattributes(mcIdx, {'numeric'}, ...
    {'scalar', 'integer', '>=', 1, '<=', numMc});

fprintf('Plotting Monte Carlo run %d of %d\n', mcIdx, numMc);

%% Extract centralized estimates: [time x stateDimension]
centState = stackStateCells(estimations_CA(mcIdx, 1:numTime));

%% Extract distributed estimates
% Each DA cell in this file has size:
%   [stateDimension x numberOfNodes x ADMM iterations]
%
% We take the final ADMM iteration and average the converged estimates over
% all nodes to obtain one distributed state estimate per time instant.
distState = nan(numTime, size(centState, 2));

for k = 1:numTime
    Xda = estimations_DA{mcIdx, k};

    if isempty(Xda)
        continue;
    end

    if isvector(Xda)
        % Already one state vector.
        distState(k, :) = Xda(:).';

    elseif ndims(Xda) == 2
        % [stateDimension x numberOfNodes]
        distState(k, :) = mean(Xda, 2, 'omitnan').';

    elseif ndims(Xda) == 3
        % [stateDimension x numberOfNodes x ADMM iterations]
        finalNodeStates = Xda(:, :, end);
        distState(k, :) = mean(finalNodeStates, 2, 'omitnan').';

    else
        error('Unexpected DA estimate dimensions at time index %d.', k);
    end
end

%% Extract the true position at the estimation instants
% In this file, target_position contains one sample per CPI and there are
% NUM_CPI_PER_MEA CPI samples for every state estimate.
if isfield(Log.target, 'true_position')
    truePositionFine = squeeze(Log.target.true_position);
elseif isfield(Log.target, 'target_position')
    truePositionFine = squeeze(Log.target.target_position);
else
    error('No true_position or target_position field exists in Log.target.');
end

% Convert to [numberOfSamples x 2] if necessary.
if size(truePositionFine, 2) ~= 2 && size(truePositionFine, 1) == 2
    truePositionFine = truePositionFine.';
end

if size(truePositionFine, 2) < 2
    error('The true-position array must contain x and y coordinates.');
end

if isfield(Log, 'NUM_CPI_PER_MEA') && ...
        size(truePositionFine, 1) >= Log.NUM_CPI_PER_MEA * numTime

    trueIdx = Log.NUM_CPI_PER_MEA : ...
              Log.NUM_CPI_PER_MEA : ...
              Log.NUM_CPI_PER_MEA * numTime;
    trueTraj = truePositionFine(trueIdx, 1:2);

elseif size(truePositionFine, 1) == numTime
    trueTraj = truePositionFine(:, 1:2);

elseif isfield(Log.Results, 'true_params')
    % Fallback when the fine true trajectory cannot be aligned directly.
    trueState = stackStateCells(Log.Results.true_params(mcIdx, 1:numTime));
    trueTraj = trueState(:, 1:2);

else
    error(['Cannot align the true trajectory with the estimator output. ', ...
           'Check the true-position sampling interval.']);
end

%% Keep only time instants available in all three trajectories
numPlot = min([size(trueTraj, 1), size(distState, 1), size(centState, 1)]);
trueTraj = trueTraj(1:numPlot, :);
distState = distState(1:numPlot, :);
centState = centState(1:numPlot, :);

%% Plot trajectories
fig = figure('Color', 'white');
ax = axes(fig);

hold(ax, 'on');

set(ax, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 22, ...
    'LineWidth', 1.5);

% Optional sensor-network plot
sensorHandle = gobjects(0);
linkHandle   = gobjects(0);

if plotNetwork && isfield(Log, 'network_topo')
    network_topo = Log.network_topo;

    if isfield(network_topo, 'laplacian_matrix') && ...
            isfield(network_topo, 'radar_pos')

        for n = 1:network_topo.numNodes
            neighborsIdx = find(network_topo.laplacian_matrix(n, :) < 0);

            for jj = 1:numel(neighborsIdx)
                m = neighborsIdx(jj);

                % Draw each undirected link only once.
                if m <= n
                    continue;
                end

                h = plot(ax, ...
                    [network_topo.radar_pos(n, 1), network_topo.radar_pos(m, 1)], ...
                    [network_topo.radar_pos(n, 2), network_topo.radar_pos(m, 2)], ...
                    '-k', ...
                    'LineWidth', 1.2, ...
                    'HandleVisibility', 'off');

                if isempty(linkHandle)
                    linkHandle = h;
                    set(linkHandle, ...
                        'DisplayName', 'Communication link', ...
                        'HandleVisibility', 'on');
                end
            end
        end

        sensorHandle = plot(ax, ...
            network_topo.radar_pos(:, 1), ...
            network_topo.radar_pos(:, 2), ...
            'r.', ...
            'MarkerSize', 35, ...
            'DisplayName', 'Sensor nodes');
    end
end

% Ground truth
trueHandle = plot(ax, ...
    trueTraj(:, 1), trueTraj(:, 2), ...
    '-.k', ...
    'LineWidth', 2.5, ...
    'DisplayName', 'True trajectory');

% Distributed estimate
daHandle = plot(ax, ...
    distState(:, 1), distState(:, 2), ...
    '--or', ...
    'LineWidth', 2, ...
    'MarkerSize', 8, ...
    'DisplayName', 'D-MAP');

% Centralized estimate
caHandle = plot(ax, ...
    centState(:, 1), centState(:, 2), ...
    '-sb', ...
    'LineWidth', 2, ...
    'MarkerSize', 8, ...
    'DisplayName', 'C-MAP');

% xlim([-40, 20]);
% ylim([-20 22]);
% Mark trajectory start and end
% plot(ax, trueTraj(1, 1), trueTraj(1, 2), ...
%     'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 9, ...
%     'HandleVisibility', 'off');
% plot(ax, trueTraj(end, 1), trueTraj(end, 2), ...
%     'k>', 'MarkerFaceColor', 'k', 'MarkerSize', 10, ...
%     'HandleVisibility', 'off');

% Reduce marker density for long trajectories.
trueHandle.MarkerIndices = markerIndices(size(trueTraj, 1), 20);
daHandle.MarkerIndices   = markerIndices(size(distState, 1), 20);
caHandle.MarkerIndices   = markerIndices(size(centState, 1), 20);

xlabel(ax, 'Position x (m)');
ylabel(ax, 'Position y (m)');

% title(ax, sprintf('Trajectory comparison, Monte Carlo run %d', mcIdx));

grid(ax, 'on');
box(ax, 'on');
axis(ax, 'equal');

legendHandles = [trueHandle, daHandle, caHandle];
if ~isempty(sensorHandle)
    legendHandles(end + 1) = sensorHandle;
end
if ~isempty(linkHandle)
    legendHandles(end + 1) = linkHandle;
end

legend(ax, legendHandles, 'Location', 'best');

put_inset_box(ax,fig,trueTraj,distState, centState,[-3.5, -3.40],[-0.3,-0.25],[0.6, 0.4, 0.15, 0.15],'right');
put_inset_box(ax,fig,trueTraj,distState, centState,[-14.2, -14.16],[-16.14,-16.13],[0.2, 0.25, 0.15, 0.15],'left');

% % 3. Build Inset plot
% % Position 參數為 [左下角X, 左下角Y, 寬度, 高度]，範圍 0 到 1
% axInset = axes('Position', [0.6, 0.35, 0.2, 0.2]); 
% box on; % 加上外框
% 
% % Ground truth
% trueHandle = plot(axInset, ...
%     trueTraj(:, 1), trueTraj(:, 2), ...
%     '-.k', ...
%     'LineWidth', 2.5, ...
%     'DisplayName', 'True trajectory');
% hold on 
% % Distributed estimate
% daHandle = plot( axInset,...
%     distState(:, 1), distState(:, 2), ...
%     '--or', ...
%     'LineWidth', 2, ...
%     'MarkerSize', 8, ...
%     'DisplayName', 'D-MAP');
% 
% % Centralized estimate
% caHandle = plot(axInset, ...
%     centState(:, 1), centState(:, 2), ...
%     '-sb', ...
%     'LineWidth', 2, ...
%     'MarkerSize', 8, ...
%     'DisplayName', 'C-MAP');
% 
% % Choose fewer index to visualized
% gtstep = max(1, floor(length(trueTraj(:,2))/100));
% trueHandle.MarkerIndices = 1:gtstep:length(trueTraj(:,2));
% 
% predstep = max(1, floor(length(daHandle(1,:))/20));
% daHandle.MarkerIndices = 1:predstep:length(centState(1,:));
% 
% predstep_ca = max(1, floor(length(centState(1,:))/20));
% caHandle.MarkerIndices = 1:predstep_ca:length(centState(1,:));
% 
% % 4. 設定放大區域
% xZoom = [-6, -4];
% yZoom = [-5, -1]
% xlim(xZoom); % 設定想觀察的細部 X 軸範圍
% ylim([yZoom]); % 設定想觀察的細部 Y 軸範圍
% % title('Zoomed View');
% 
% 
% 
% % Main-axes corners in data coordinates
% p1Data = [xZoom(2), yZoom(2)];  % upper-right corner
% p2Data = [xZoom(2), yZoom(1)];  % lower-right corner
% 
% % Convert them to normalized figure coordinates
% axMain = ax;
% p1Fig = data2fig(axMain, p1Data(1), p1Data(2));
% p2Fig = data2fig(axMain, p2Data(1), p2Data(2));
% 
% % Inset position is already in normalized figure coordinates
% insetPos = axInset.Position;
% 
% % Left-side corners of the inset
% insetUpperLeft = [insetPos(1), insetPos(2) + insetPos(4)];
% insetLowerLeft = [insetPos(1), insetPos(2)];
% 
% %% Region to zoom
% % Draw the zoom-area rectangle on the main axes
% rectangle(axMain, ...
%     'Position', [xZoom(1), yZoom(1), ...
%                  diff(xZoom), diff(yZoom)], ...
%     'LineStyle', '--', ...
%     'LineWidth', 1.2);
% % Draw connecting lines
% annotation(fig, 'line', ...
%     [p1Fig(1), insetUpperLeft(1)], ...
%     [p1Fig(2), insetUpperLeft(2)], ...
%     'LineWidth', 1);
% 
% annotation(fig, 'line', ...
%     [p2Fig(1), insetLowerLeft(1)], ...
%     [p2Fig(2), insetLowerLeft(2)], ...
%     'LineWidth', 1);
% 
% % Optional vector export:
% % exportgraphics(fig, sprintf('trajectory_mc_%d.pdf', mcIdx), ...
% %     'ContentType', 'vector');

%% Local functions
function X = stackStateCells(C)
    C = C(:);
    validCell = ~cellfun(@isempty, C);
    C = C(validCell);

    if isempty(C)
        X = zeros(0, 0);
        return;
    end

    stateDimension = numel(C{1});
    X = nan(numel(C), stateDimension);

    for ii = 1:numel(C)
        state = C{ii};

        if ~isvector(state)
            error('Cell %d does not contain a state vector.', ii);
        end

        if numel(state) ~= stateDimension
            error('The estimator state dimensions are inconsistent.');
        end

        X(ii, :) = state(:).';
    end
end

function idx = markerIndices(numPoints, maxMarkers)
    if numPoints <= maxMarkers
        idx = 1:numPoints;
    else
        idx = unique(round(linspace(1, numPoints, maxMarkers)));
    end
end


function pFig = data2fig(ax, xData, yData)
%DATA2FIG Convert axes data coordinates to normalized figure coordinates.

    % Preserve the original units
    originalUnits = ax.Units;
    ax.Units = 'normalized';

    axPos = ax.Position;
    xLimits = ax.XLim;
    yLimits = ax.YLim;

    % Convert data coordinates to relative axes coordinates
    xRelative = (xData - xLimits(1)) / diff(xLimits);
    yRelative = (yData - yLimits(1)) / diff(yLimits);

    % Account for reversed axes
    if strcmp(ax.XDir, 'reverse')
        xRelative = 1 - xRelative;
    end

    if strcmp(ax.YDir, 'reverse')
        yRelative = 1 - yRelative;
    end

    % Convert relative axes coordinates to normalized figure coordinates
    xFig = axPos(1) + xRelative * axPos(3);
    yFig = axPos(2) + yRelative * axPos(4);

    pFig = [xFig, yFig];

    % Restore original units
    ax.Units = originalUnits;
end


function put_inset_box(ax,fig, trueTraj, distState, centState, xZoom, yZoom, boxPos, corner2connect)
% 3. Build Inset plot
% Position 參數為 [左下角X, 左下角Y, 寬度, 高度]，範圍 0 到 1
% boxPos = [0.6, 0.35, 0.2, 0.2]
axis([-40 25 -22 22]);
axInset = axes('Position', boxPos); 
box on; % 加上外框

% Ground truth
trueHandle = plot(axInset, ...
    trueTraj(:, 1), trueTraj(:, 2), ...
    '-.k', ...
    'LineWidth', 2.5, ...
    'DisplayName', 'True trajectory');
hold on 
% Distributed estimate
daHandle = plot( axInset,...
    distState(:, 1), distState(:, 2), ...
    '--or', ...
    'LineWidth', 2, ...
    'MarkerSize', 8, ...
    'DisplayName', 'D-MAP');

% Centralized estimate
caHandle = plot(axInset, ...
    centState(:, 1), centState(:, 2), ...
    '-sb', ...
    'LineWidth', 2, ...
    'MarkerSize', 8, ...
    'DisplayName', 'C-MAP');

% Choose fewer index to visualized
gtstep = max(1, floor(length(trueTraj(:,2))/100));
trueHandle.MarkerIndices = 1:gtstep:length(trueTraj(:,2));

predstep = max(1, floor(length(daHandle(1,:))/20));
daHandle.MarkerIndices = 1:predstep:length(centState(1,:));

predstep_ca = max(1, floor(length(centState(1,:))/20));
caHandle.MarkerIndices = 1:predstep_ca:length(centState(1,:));

% 4. 設定放大區域
% xZoom = [-6, -4];
% yZoom = [-5, -1]
xlim(xZoom); % 設定想觀察的細部 X 軸範圍
ylim([yZoom]); % 設定想觀察的細部 Y 軸範圍
% title('Zoomed View');



% Main-axes corners in data coordinates
p1Data = [xZoom(2), yZoom(2)];  % upper-right corner
p2Data = [xZoom(2), yZoom(1)];  % lower-right corner

p3Data = [xZoom(1), yZoom(2)];  % upper-leftcorner
p4Data = [xZoom(1), yZoom(1)];  % lower-left corner
% Convert them to normalized figure coordinates
axMain = ax;


p1Fig = data2fig(axMain, p1Data(1), p1Data(2));
p2Fig = data2fig(axMain, p2Data(1), p2Data(2));
p3Fig = data2fig(axMain, p3Data(1), p3Data(2));
p4Fig = data2fig(axMain, p4Data(1), p4Data(2));

% Inset position is already in normalized figure coordinates
insetPos = axInset.Position;

% Left-side corners of the inset
insetUpperLeft = [insetPos(1), insetPos(2) + insetPos(4)];
insetLowerLeft = [insetPos(1), insetPos(2)];
% right-side corners of the inset
insetUpperRight = [insetPos(1)+insetPos(3), insetPos(2) + insetPos(4)];
insetLowerRight = [insetPos(1)+insetPos(3), insetPos(2)];


%% Region to zoom
% Draw the zoom-area rectangle on the main axes
rectangle(axMain, ...
    'Position', [xZoom(1), yZoom(1), ...
                 diff(xZoom), diff(yZoom)], ...
    'LineStyle', '--', ...
    'LineWidth', 1.2);
% Draw connecting lines

if corner2connect == "right"
    annotation(fig, 'line', ...
        [p1Fig(1), insetUpperLeft(1)], ...
        [p1Fig(2), insetUpperLeft(2)], ...
        'LineWidth', 1);
    
    annotation(fig, 'line', ...
        [p2Fig(1), insetLowerLeft(1)], ...
        [p2Fig(2), insetLowerLeft(2)], ...
        'LineWidth', 1);

elseif corner2connect == "left"
    annotation(fig, 'line', ...
    [p3Fig(1), insetUpperRight(1)], ...
    [p3Fig(2), insetUpperRight(2)], ...
    'LineWidth', 1);

    annotation(fig, 'line', ...
        [p4Fig(1), insetLowerRight(1)], ...
        [p4Fig(2), insetLowerRight(2)], ...
        'LineWidth', 1);
end
% Optional vector export:
% exportgraphics(fig, sprintf('trajectory_mc_%d.pdf', mcIdx), ...
%     'ContentType', 'vector');

end