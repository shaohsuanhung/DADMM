function trajectory = sigmoidTrajectory(startPoint, endPoint, numPoints, amplitude, steepness)
%SIGMOIDTRAJECTORY Generate an S-shaped 2D target trajectory.
%
% trajectory = sigmoidTrajectory(startPoint, endPoint)
% trajectory = sigmoidTrajectory(startPoint, endPoint, numPoints, ...
%                                amplitude, steepness)
%
% Inputs:
%   startPoint : [x_start, y_start]
%   endPoint   : [x_end, y_end]
%   numPoints  : Number of trajectory samples, default = 200
%   amplitude  : Maximum lateral deviation from the straight line.
%                Default = 0.15 times the start-to-end distance
%   steepness  : Controls how sharply the target turns, default = 10
%
% Output:
%   trajectory : numPoints-by-2 matrix containing [x, y]
%
% Example:
%   p = sigmoidTrajectory([-30, -30], [30, 30], 300, 10, 12);
%   plot(p(:,1), p(:,2), 'LineWidth', 2);
%   axis equal;
%   grid on;

    arguments
        startPoint (1,2) double
        endPoint   (1,2) double
        numPoints  (1,1) double {mustBeInteger, mustBeGreaterThan(numPoints,1)} = 200
        amplitude  (1,1) double = NaN
        steepness  (1,1) double {mustBePositive} = 10
    end

    displacement = endPoint - startPoint;
    distance = norm(displacement);

    if distance == 0
        error('The start point and end point must be different.');
    end

    if isnan(amplitude)
        amplitude = 0.15 * distance;
    end

    % Normalized trajectory parameter
    u = linspace(0, 1, numPoints)';

    % Unit vector from start point to end point
    direction = displacement / distance;

    % Perpendicular unit vector
    perpendicular = [-direction(2), direction(1)];

    % Raw sigmoid curve
    sigmoid = 2 ./ (1 + exp(-steepness * (u - 0.5))) - 1;

    % Remove the line between the sigmoid endpoints.
    % This ensures lateralOffset(1) = lateralOffset(end) = 0.
    sigmoidStart = sigmoid(1);
    sigmoidEnd = sigmoid(end);

    endpointLine = (1 - u) * sigmoidStart + u * sigmoidEnd;
    correctedSigmoid = sigmoid - endpointLine;

    % Normalize so amplitude represents the maximum lateral deviation
    maxValue = max(abs(correctedSigmoid));

    if maxValue > eps
        correctedSigmoid = correctedSigmoid / maxValue;
    end

    lateralOffset = amplitude * correctedSigmoid;

    % Straight-line component
    trajectory = startPoint ...
               + u * displacement ...
               + lateralOffset * perpendicular;

    % Enforce endpoints exactly
    trajectory(1,:) = startPoint;
    trajectory(end,:) = endPoint;
end