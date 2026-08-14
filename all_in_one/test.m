clear; clc; close all;

%% Example data
x = linspace(0, 10, 1000);
y = sin(x) + 0.05*randn(size(x));

fig = figure;
axMain = axes(fig);

plot(axMain, x, y, 'LineWidth', 1.5);
xlabel(axMain, 'x');
ylabel(axMain, 'y');
grid(axMain, 'on');
hold(axMain, 'on');

%% Region to zoom
xZoom = [4.0, 5.0];
yZoom = [-1.2, -0.5];

% Draw the zoom-area rectangle on the main axes
rectangle(axMain, ...
    'Position', [xZoom(1), yZoom(1), ...
                 diff(xZoom), diff(yZoom)], ...
    'LineStyle', '--', ...
    'LineWidth', 1.2);

%% Create inset axes
axInset = axes(fig, ...
    'Position', [0.58, 0.58, 0.28, 0.28]);

plot(axInset, x, y, 'LineWidth', 1.2);
xlim(axInset, xZoom);
ylim(axInset, yZoom);
grid(axInset, 'on');
box(axInset, 'on');

%% Connect zoom region to inset
% Main-axes corners in data coordinates
p1Data = [xZoom(2), yZoom(2)];  % upper-right corner
p2Data = [xZoom(2), yZoom(1)];  % lower-right corner

% Convert them to normalized figure coordinates
p1Fig = data2fig(axMain, p1Data(1), p1Data(2));
p2Fig = data2fig(axMain, p2Data(1), p2Data(2));

% Inset position is already in normalized figure coordinates
insetPos = axInset.Position;

% Left-side corners of the inset
insetUpperLeft = [insetPos(1), insetPos(2) + insetPos(4)];
insetLowerLeft = [insetPos(1), insetPos(2)];

% Draw connecting lines
annotation(fig, 'line', ...
    [p1Fig(1), insetUpperLeft(1)], ...
    [p1Fig(2), insetUpperLeft(2)], ...
    'LineWidth', 1);

annotation(fig, 'line', ...
    [p2Fig(1), insetLowerLeft(1)], ...
    [p2Fig(2), insetLowerLeft(2)], ...
    'LineWidth', 1);



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