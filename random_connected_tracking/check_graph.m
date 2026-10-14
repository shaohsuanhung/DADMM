function check_graph()
%CHECK_GRAPH Verify topology invariants and reproducibility without toolboxes.
project_root = fileparts(mfilename('fullpath'));
previous_path = path;
cleanup = onCleanup(@() path(previous_path));
addpath(fullfile(project_root, 'common'));
cfg = struct('NumNodes', 10, 'EdgeProbability', 0.35, ...
    'Seed', 20261013, 'MaxAttempts', 10000);
radar = struct('Mode', "random_disk", 'Radius', 30, ...
    'Center', [0, 0], 'Seed', 20261014);
before_rng = rng;
for num_nodes = [2 7 10 16]
    cfg.NumNodes = num_nodes;
    topology = make_connected_random_graph(cfg, radar);
    changed_radar = radar;
    changed_radar.Seed = radar.Seed + 1;
    changed_radar.Radius = 20;
    repeated = make_connected_random_graph(cfg, changed_radar);
    A = topology.adj_matrix;
    assert(isequal(A, A.') && ~any(diag(A)), 'Graph must be simple and undirected.');
    assert(all(conncomp(graph(A)) == 1), 'Graph must be connected.');
    assert(isequal(A, repeated.adj_matrix), 'Moving radars must not change communication edges.');
    assert(~isequal(topology.radar_pos, repeated.radar_pos));
    assert(norm(topology.inc_matrix * topology.inc_matrix.' - topology.laplacian_matrix, 'fro') < 1e-12);
    assert(topology.algebraic_connectivity > 0, 'Connected graph must have positive lambda2.');
    assert(isequal(before_rng, rng), 'Graph generation must leave simulation RNG unchanged.');
end
cfg.NumNodes = 10;
cfg.EdgeProbability = 1;
topology = make_connected_random_graph(cfg, radar);
assert(topology.numEdges == 45 && topology.generation_attempts == 1);
changed_graph = cfg;
changed_graph.Seed = cfg.Seed + 1;
changed_graph.EdgeProbability = 0.35;
repeated = make_connected_random_graph(changed_graph, radar);
assert(isequal(topology.radar_pos, repeated.radar_pos), 'Changing edges must not move radars.');
integer_center = radar;
integer_center.Center = int32([5, -7]);
translated = make_connected_random_graph(cfg, integer_center);
assert(norm(translated.radar_pos - topology.radar_pos - [5, -7], 'fro') < 1e-12);
cfg.EdgeProbability = realmin;
cfg.MaxAttempts = 2;
try
    make_connected_random_graph(cfg, radar);
    error('RadarTracking:ExpectedFailure', 'Disconnected graph must fail explicitly.');
catch exception
    assert(strcmp(exception.identifier, 'RadarTracking:DisconnectedGraph'));
end
fprintf('Graph checks passed (N=2,7,10,16; repeatability; RNG isolation; failure path).\n');
end
