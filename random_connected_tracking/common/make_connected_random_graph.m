function network_topo = make_connected_random_graph(cfg, radar_cfg)
%MAKE_CONNECTED_RANDOM_GRAPH Build a connected random graph or an index ring.
% cfg.Mode is "random_connected" (default) or "ring"; name retained for callers.
% Radar positions come from a separate deployment configuration and seed.
% On the evenly spaced circle, ring edges join each radar's two nearest nodes.

validateattributes(cfg.NumNodes, {'numeric'}, {'scalar', 'integer', '>=', 2});
num_nodes = cfg.NumNodes;
mode = "random_connected"; % Old saved configs have no Mode field.
if isfield(cfg, 'Mode')
    mode = string(cfg.Mode);
end
if ~isscalar(mode) || ~any(mode == ["random_connected", "ring"])
    error('RadarTracking:GraphMode', 'Graph Mode must be random_connected or ring.');
end

radar_pos = make_radar_positions(radar_cfg, num_nodes);
switch mode
    case "random_connected"
        validateattributes(cfg.EdgeProbability, {'numeric'}, {'scalar', '>', 0, '<=', 1});
        validateattributes(cfg.Seed, {'numeric'}, {'scalar', 'integer', '>=', 0, '<=', 2^32-1});
        validateattributes(cfg.MaxAttempts, {'numeric'}, {'scalar', 'integer', 'positive'});
        stream = RandStream('mt19937ar', 'Seed', cfg.Seed);
        for attempt = 1:cfg.MaxAttempts
            upper = triu(rand(stream, num_nodes) < cfg.EdgeProbability, 1);
            adjacency = double(upper | upper.');
            if all(conncomp(graph(adjacency)) == 1)
                break;
            end
        end
        if ~all(conncomp(graph(adjacency)) == 1)
            error('RadarTracking:DisconnectedGraph', ...
                'No connected graph in %d draws. Increase EdgeProbability or MaxAttempts.', ...
                cfg.MaxAttempts);
        end
        graph_model = 'Connected Erdos-Renyi G(N,p), rejection sampled';
        graph_seed = cfg.Seed;
        edge_probability = cfg.EdgeProbability;

    case "ring"
        if num_nodes < 3
            error('RadarTracking:RingNodeCount', ...
                'A ring with two distinct neighbors per node requires at least 3 nodes.');
        end
        % Node order closes the cycle: 1--2--...--N--1.
        upper = diag(ones(num_nodes-1, 1), 1);
        upper(1, num_nodes) = 1;
        adjacency = upper + upper.';
        graph_model = 'Undirected ring in node-index order (degree 2)';
        graph_seed = NaN;       % Random graph settings are unused in ring mode.
        edge_probability = NaN;
        attempt = 0;
end

degree = sum(adjacency, 2);
laplacian = diag(degree) - adjacency;
edge_count = nnz(upper);
incidence = zeros(num_nodes, edge_count);
edge_list = zeros(edge_count, 2);
edge_index = 0;
for i = 1:num_nodes
    for j = i+1:num_nodes
        if adjacency(i, j)
            edge_index = edge_index + 1;
            edge_list(edge_index, :) = [i, j];
            incidence(i, edge_index) = 1;
            incidence(j, edge_index) = -1;
        end
    end
end

eigenvalues = sort(eig(laplacian));
network_topo.numNodes = num_nodes;
offset = radar_pos - double(radar_cfg.Center(:).');
network_topo.theta = atan2(offset(:, 2), offset(:, 1)).';
network_topo.radius = radar_cfg.Radius;
network_topo.radar_pos = radar_pos;
network_topo.radar_config = radar_cfg;
network_topo.labels = arrayfun(@(n) sprintf('N%d', n), 1:num_nodes, ...
    'UniformOutput', false);
network_topo.adj_matrix = adjacency;
network_topo.degree_matrix = diag(degree);
network_topo.laplacian_matrix = laplacian;
network_topo.inc_matrix = incidence;
% Same maximum-degree weights as the original utilities (metadata only).
network_topo.weights_matrix = eye(num_nodes) - laplacian / max(degree);
network_topo.neighbors = arrayfun(@(n) find(adjacency(n, :)), ...
    1:num_nodes, 'UniformOutput', false);
network_topo.graph_mode = mode;
network_topo.graph_model = graph_model;
network_topo.graph_seed = graph_seed;
network_topo.edge_probability = edge_probability;
network_topo.generation_attempts = attempt;
network_topo.edge_list = edge_list;
network_topo.numEdges = edge_count;
network_topo.algebraic_connectivity = eigenvalues(2);
end
