function check_radar_positions()
%CHECK_RADAR_POSITIONS Check reproducibility, geometry and uniform disk area.
project_root = fileparts(mfilename('fullpath'));
previous_path = path;
cleanup = onCleanup(@() path(previous_path));
addpath(fullfile(project_root, 'common'));

before_rng = rng;
cfg = struct('Mode', 'random_disk', 'Radius', 20, ...
    'Center', [3 -7], 'Seed', 20261013);
num_nodes = 257;
positions = make_radar_positions(cfg, num_nodes);
repeated = make_radar_positions(cfg, num_nodes);
assert(isequal(size(positions), [num_nodes, 2]));
assert(isequal(positions, repeated), ...
    'The same seed must reproduce radar positions exactly.');

changed = cfg;
changed.Seed = cfg.Seed + 1;
assert(~isequal(positions, make_radar_positions(changed, num_nodes)), ...
    'Changing the seed must change a random layout.');
radial_distance = vecnorm(positions - cfg.Center, 2, 2);
assert(all(radial_distance <= cfg.Radius * (1 + 1e-12)), ...
    'Every random radar must lie inside the configured disk.');

changed = cfg;
changed.Center = [-11; 19]; % Column-vector centers are also accepted.
changed.Radius = 3 * cfg.Radius;
transformed = make_radar_positions(changed, num_nodes);
expected = changed.Center.' + 3 * (positions - cfg.Center);
assert(norm(transformed - expected, 'fro') < 1e-10, ...
    'Center and radius must translate and scale the same seeded layout.');

circle_cfg = struct('Mode', 'circle', 'Radius', 9, ...
    'Center', [-3; 4], 'Seed', uint32(77));
num_circle_nodes = 12;
circle = make_radar_positions(circle_cfg, num_circle_nodes);
angles = (0:num_circle_nodes-1).' * (2*pi/num_circle_nodes);
expected_circle = circle_cfg.Center.' + ...
    circle_cfg.Radius * [cos(angles), sin(angles)];
assert(norm(circle - expected_circle, 'fro') < 1e-12, ...
    'Circle mode must start at angle zero and use equal angular spacing.');
circle_cfg.Seed = uint32(78);
assert(isequal(circle, make_radar_positions(circle_cfg, num_circle_nodes)), ...
    'Circle positions must not depend on the seed.');
single = make_radar_positions(circle_cfg, 1);
assert(isequal(single, circle_cfg.Center.' + [circle_cfg.Radius, 0]));

% E[r^2/R^2] = 1/2 for uniform disk area, but 1/3 for uniform radius.
% A fixed seed makes this distribution check deterministic and reproducible.
moment_cfg = struct('Mode', 'random_disk', 'Radius', 7, ...
    'Center', [11 -5], 'Seed', 90210);
many_positions = make_radar_positions(moment_cfg, 50000);
normalized = (many_positions - moment_cfg.Center) / moment_cfg.Radius;
radial_second_moment = mean(sum(normalized.^2, 2));
assert(abs(radial_second_moment - 0.5) < 0.01, ...
    'Disk samples must be uniform in area, not uniform in radius.');
assert(all(abs(mean(normalized, 1)) < 0.02), ...
    'Disk samples should not favor an angular direction.');
assert(isequal(before_rng, rng), ...
    'Radar placement must leave the caller RNG unchanged.');

fprintf(['Radar position checks passed (repeatability; seed sensitivity; ', ...
    'bounds; translation/scaling; circle; RNG isolation; disk moment %.5f).\n'], ...
    radial_second_moment);
end
