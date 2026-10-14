function radar_pos = make_radar_positions(cfg, num_nodes)
%MAKE_RADAR_POSITIONS Place radars in a disk or on its circumference.
% cfg.Mode is 'random_disk' or 'circle'. Radius, Center and Seed specify
% the disk and a private random stream. The caller's RNG is unchanged.

validateattributes(num_nodes, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', 'positive'});
validateattributes(cfg.Radius, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'positive'});
validateattributes(cfg.Center, {'numeric'}, ...
    {'vector', 'real', 'finite', 'numel', 2});
validateattributes(cfg.Seed, {'numeric'}, ...
    {'scalar', 'real', 'finite', 'integer', '>=', 0, '<=', 2^32-1});
if ~(ischar(cfg.Mode) && isrow(cfg.Mode)) && ...
        ~(isstring(cfg.Mode) && isscalar(cfg.Mode))
    error('RadarTracking:RadarPositionMode', ...
        'Mode must be random_disk or circle.');
end

radius = double(cfg.Radius);
center = reshape(double(cfg.Center), 1, 2);
num_nodes = double(num_nodes);
switch string(cfg.Mode)
    case "random_disk"
        stream = RandStream('mt19937ar', 'Seed', double(cfg.Seed));
        % Squared radius is uniform, so equal areas have equal probability.
        radial_distance = radius * sqrt(rand(stream, num_nodes, 1));
        angle = 2*pi * rand(stream, num_nodes, 1);
    case "circle"
        radial_distance = radius * ones(num_nodes, 1);
        angle = (0:num_nodes-1).' * (2*pi/num_nodes);
    otherwise
        error('RadarTracking:RadarPositionMode', ...
            'Mode must be random_disk or circle.');
end

radar_pos = center + radial_distance .* [cos(angle), sin(angle)];
end
