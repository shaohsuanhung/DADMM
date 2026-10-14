function check_ring_graph()
%CHECK_RING_GRAPH Check cycle geometry and legacy random-graph compatibility.
project_root = fileparts(mfilename('fullpath'));
previous_path = path;
cleanup = onCleanup(@() path(previous_path));
addpath(fullfile(project_root, 'common'));

before_rng = rng;
cfg = struct('Mode', 'ring', 'NumNodes', 10, ...
    'EdgeProbability', 0.35, 'Seed', 20261013, 'MaxAttempts', 10000);
radar = struct('Mode', 'circle', 'Radius', 30, ...
    'Center', [3, -7], 'Seed', 20261014);

for num_nodes = [3, 4, 10, 16]
    cfg.NumNodes = num_nodes;
    topology = make_connected_random_graph(cfg, radar);
    A = topology.adj_matrix;
    expected = diag(ones(num_nodes - 1, 1), 1);
    expected(1, num_nodes) = 1;
    expected = expected + expected.';
    assert(isequal(A, expected), ...
        'Ring edges must follow node indices, including the last-to-first edge.');
    assert(isequal(A, A.') && ~any(diag(A)) && all(A(:) == 0 | A(:) == 1), ...
        'A ring must be a simple, undirected graph.');
    assert(all(conncomp(graph(A)) == 1), 'A ring must be connected.');
    assert(topology.numNodes == num_nodes && topology.numEdges == num_nodes);
    assert(nnz(triu(A, 1)) == num_nodes && all(sum(A, 2) == 2), ...
        'Every ring node must have degree two and a ring must have N edges.');
    assert(A(1, num_nodes) == 1 && A(num_nodes, 1) == 1, ...
        'The ring must include its wrap-around edge.');
    assert(isequal(topology.degree_matrix, 2 * eye(num_nodes)));
    assert(isequal(topology.laplacian_matrix, 2 * eye(num_nodes) - A));
    assert(norm(topology.inc_matrix * topology.inc_matrix.' - ...
        topology.laplacian_matrix, 'fro') < 1e-12, ...
        'Incidence and Laplacian matrices must describe the same ring.');
    expected_lambda2 = 2 - 2 * cos(2 * pi / num_nodes);
    assert(abs(topology.algebraic_connectivity - expected_lambda2) < 1e-11, ...
        'A cycle must have algebraic connectivity 2 - 2*cos(2*pi/N).');

    % Circle placement makes adjacent indices the two nearest radars.
    for node = 1:num_nodes
        distances = vecnorm(topology.radar_pos - topology.radar_pos(node, :), 2, 2);
        distances(node) = Inf;
        [~, ordered] = sort(distances);
        actual_neighbors = topology.neighbors{node};
        assert(isequal(sort(actual_neighbors(:)), sort(ordered(1:2))), ...
            'On a circle, ring neighbors must be the two nearest radars.');
    end
    assert(strcmp(topology.graph_mode, 'ring'));
    assert((ischar(topology.graph_model) || isstring(topology.graph_model)) && ...
        strlength(string(topology.graph_model)) > 0, ...
        'Ring metadata must describe its graph model.');
    assert(isnan(topology.graph_seed) && isnan(topology.edge_probability) && ...
        topology.generation_attempts == 0, ...
        'Deterministic ring metadata must not imply random graph sampling.');

    % Ring generation does not require or validate random-only parameters.
    minimal = struct('Mode', "ring", 'NumNodes', num_nodes);
    assert(isequaln(topology, make_connected_random_graph(minimal, radar)), ...
        'A ring must work when all random-only settings are absent.');
    ignored = cfg;
    ignored.EdgeProbability = NaN;
    ignored.Seed = -1;
    ignored.MaxAttempts = 0;
    assert(isequaln(topology, make_connected_random_graph(ignored, radar)), ...
        'Random-only settings must have no effect in ring mode.');

    moved_radar = radar;
    moved_radar.Mode = 'random_disk';
    moved_radar.Radius = 7;
    moved_radar.Center = [-11, 19];
    moved_radar.Seed = radar.Seed + 1;
    moved = make_connected_random_graph(cfg, moved_radar);
    assert(~isequal(topology.radar_pos, moved.radar_pos));
    assert(isequal(A, moved.adj_matrix) && ...
        isequal(topology.edge_list, moved.edge_list) && ...
        isequal(topology.inc_matrix, moved.inc_matrix), ...
        'Changing radar positions must not change index-based ring edges.');
    assert(isequal(before_rng, rng), 'Ring generation must preserve the caller RNG.');
end

% Preserve the historical default and original seeded rejection sampler.
legacy = struct('NumNodes', 10, 'EdgeProbability', 0.35, ...
    'Seed', 20261013, 'MaxAttempts', 10000);
random_cfg = legacy;
random_cfg.Mode = 'random_connected';
disk_radar = radar;
disk_radar.Mode = 'random_disk';
for graph_seed = [20261013, 17]
    legacy.Seed = graph_seed;
    random_cfg.Seed = graph_seed;
    default_random = make_connected_random_graph(legacy, disk_radar);
    explicit_random = make_connected_random_graph(random_cfg, disk_radar);
    assert(isequaln(default_random, explicit_random), ...
        'An omitted graph Mode must preserve explicit random_connected behavior.');
    [legacy_adjacency, legacy_attempts] = original_random_draw(legacy);
    assert(isequal(explicit_random.adj_matrix, legacy_adjacency) && ...
        explicit_random.generation_attempts == legacy_attempts, ...
        'Seeded random graphs must preserve the original rejection-sampling sequence.');
    assert(strcmp(explicit_random.graph_mode, 'random_connected'));
    circle_random = make_connected_random_graph(random_cfg, radar);
    assert(~isequal(explicit_random.radar_pos, circle_random.radar_pos));
    assert(isequal(explicit_random.adj_matrix, circle_random.adj_matrix) && ...
        isequal(explicit_random.edge_list, circle_random.edge_list) && ...
        isequal(explicit_random.inc_matrix, circle_random.inc_matrix) && ...
        explicit_random.generation_attempts == circle_random.generation_attempts, ...
        'Switching to circle placement must preserve seeded random communication edges.');
end

too_small = cfg;
too_small.NumNodes = 2;
assert_error(@() make_connected_random_graph(too_small, radar), ...
    'RadarTracking:RingNodeCount');
invalid_mode = cfg;
invalid_mode.Mode = 'nearest_neighbors';
assert_error(@() make_connected_random_graph(invalid_mode, radar), ...
    'RadarTracking:GraphMode');
assert(isequal(before_rng, rng), ...
    'Graph generation, compatibility checks, and error paths must preserve the caller RNG.');
fprintf(['Ring graph checks passed (N=3,4,10,16; circle neighbors; ', ...
    'incidence/Laplacian; spectrum; metadata; RNG isolation; ', ...
    'legacy random compatibility; invalid inputs).\n']);
end

function [adjacency, attempt] = original_random_draw(cfg)
% Frozen pre-mode sampling algorithm, using its own independent stream.
stream = RandStream('mt19937ar', 'Seed', cfg.Seed);
for attempt = 1:cfg.MaxAttempts
    upper = triu(rand(stream, cfg.NumNodes) < cfg.EdgeProbability, 1);
    adjacency = double(upper | upper.');
    if all(conncomp(graph(adjacency)) == 1)
        return;
    end
end
error('RadarTracking:TestFixtureDisconnected', ...
    'The legacy random-graph fixture did not produce a connected graph.');
end

function assert_error(action, expected_identifier)
try
    action();
catch exception
    assert(strcmp(exception.identifier, expected_identifier), ...
        'Expected %s, received %s.', expected_identifier, exception.identifier);
    return;
end
error('RadarTracking:ExpectedFailure', ...
    'Expected operation to fail with %s.', expected_identifier);
end
