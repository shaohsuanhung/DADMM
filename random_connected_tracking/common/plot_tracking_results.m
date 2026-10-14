function plot_tracking_results(Run, visible)
%PLOT_TRACKING_RESULTS Manuscript style; trajectories use MC1, RMSE all MCs.
% Plot saved values without smoothing, rerunning estimators or changing MATs.
style = tracking_plot_config();
if isfield(Run.Config, 'Plot')
    style = tracking_plot_config(Run.Config.Plot);
end
R = Run.Results;
network_topo = Run.network_topo;
if Run.Method == "EKF"
    distributed_cells = R.estimations_DA_raw;
    central_cells = R.estimations_CA_raw;
    truth_cells = R.true_params_raw;
    primal = R.primal_residauls_raw;
    dual = R.dual_residauls_raw;
    iterations = cell2mat(R.convg_iter_raw);
    target = R.target;
else
    distributed_cells = R.consensus_estimates;
    central_cells = R.estimations_CA;
    truth_cells = R.true_params;
    primal = R.primal_residauls;
    dual = R.dual_residauls;
    iterations = cell2mat(R.convg_iter);
    target = R.targets{1};
end
[num_mc, num_steps] = size(truth_cells);
truth = zeros(4, num_steps, num_mc);
distributed = truth;
central = truth;
for mc = 1:num_mc
    for k = 1:num_steps
        truth(:, k, mc) = truth_cells{mc, k}(:);
        distributed(:, k, mc) = distributed_cells{mc, k}(:);
        central(:, k, mc) = central_cells{mc, k}(:);
    end
end
[time, time_label] = tracking_time(Run, target, num_steps);
rmse_distributed = zeros(2, num_steps);
rmse_central = rmse_distributed;
for component = 1:2
    idx = (2*component-1):(2*component);
    rmse_distributed(component, :) = sqrt(mean(sum( ...
        (distributed(idx, :, :) - truth(idx, :, :)).^2, 1), 3));
    rmse_central(component, :) = sqrt(mean(sum( ...
        (central(idx, :, :) - truth(idx, :, :)).^2, 1), 3));
end
root_pcrlb = [];
if Run.Method == "EKF" && style.ShowPCRLB && ...
        isfield(R, 'PCRLB_Bayes') && ~isempty(R.PCRLB_Bayes)
    B = R.PCRLB_Bayes;
    assert(size(B, 1) == 4 && size(B, 2) == 4 && size(B, 3) == num_steps, ...
        'RadarTracking:PCRLBSize', 'Saved PCRLB must have one 4-by-4 matrix per update.');
    root_pcrlb = [reshape(sqrt(B(1,1,:) + B(2,2,:)), 1, []); ...
                  reshape(sqrt(B(3,3,:) + B(4,4,:)), 1, [])];
end

% Standalone figures for the manuscript, with the original trajectory styles.
trajectory_fig = new_figure(style.FigureSize, visible, style);
draw_tracking(axes(trajectory_fig), network_topo, truth(:,:,1), ...
    distributed(:,:,1), central(:,:,1), Run.Method, style);
save_figure(trajectory_fig, Run.OutputFolder, 'trajectory', visible, style, true);

rmse_fig = new_figure(style.RMSEFigureSize, visible, style);
tiledlayout(rmse_fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
rmse_axes = gobjects(1, 2);
for component = 1:2
    rmse_axes(component) = nexttile;
    draw_rmse(rmse_axes(component), component, time, time_label, rmse_distributed, ...
        rmse_central, root_pcrlb, Run.Method, style);
end
% Fig. 6 uses a shared legend below the panels, clear of every data curve.
delete(legend(rmse_axes(1)));
shared_legend = legend(rmse_axes(2));
shared_legend.Layout.Tile = 'south';
shared_legend.NumColumns = 2 + ~isempty(root_pcrlb);
save_figure(rmse_fig, Run.OutputFolder, 'rmse', visible, style, true);

% Keep the existing combined overview and diagnostic output names.
overview = new_figure(style.OverviewFigureSize, visible, style);
tiledlayout(overview, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile;
draw_network(ax, network_topo, style);
axis(ax, 'equal'); grid(ax, 'on');
xlabel(ax, 'Position x (m)'); ylabel(ax, 'Position y (m)');
title(ax, sprintf('N=%d, edges=%d, \\lambda_2=%.3f', network_topo.numNodes, ...
    network_topo.numEdges, network_topo.algebraic_connectivity));
draw_tracking(nexttile, network_topo, truth(:,:,1), distributed(:,:,1), ...
    central(:,:,1), Run.Method, style);
title('Tracking estimates (MC1)');
for component = 1:2
    draw_rmse(nexttile, component, time, time_label, rmse_distributed, ...
        rmse_central, root_pcrlb, Run.Method, style);
    title(sprintf('%d Monte Carlo trial(s)', num_mc));
end
sgtitle(sprintf('%s | %s | stopping rule met at %d/%d steps', ...
    Run.Method, Run.Config.Profile, Run.ConvergedSteps, Run.TotalSteps), ...
    'Interpreter', 'none', 'FontName', style.FontName, 'FontWeight', 'normal');
save_figure(overview, Run.OutputFolder, 'overview', visible, style, false);

convergence = new_figure(style.WideFigureSize, visible, style);
tiledlayout(convergence, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile;
semilogy(ax, max(primal{1,end}(:), realmin), 'k-', 'LineWidth', style.LineWidth);
hold(ax, 'on');
semilogy(ax, max(dual{1,end}(:), realmin), 'r--', 'LineWidth', style.LineWidth);
grid(ax, 'on'); xlabel(ax, 'ADMM iteration'); ylabel(ax, 'Residual norm');
title(ax, 'MC1, final tracking step');
legend(ax, 'Primal', 'Dual', 'Location', 'best');
ax = nexttile;
h = plot(ax, time, iterations.', 'o-', 'LineWidth', style.LineWidth);
set(h, 'MarkerIndices', marker_indices(num_steps, style.NumMarkers), ...
    'MarkerSize', style.MarkerSize);
grid(ax, 'on'); xlabel(ax, time_label); ylabel(ax, 'ADMM iterations');
title(ax, 'Per trial');
save_figure(convergence, Run.OutputFolder, 'convergence', visible, style, false);

% Full reference and actual simulated prefix remain useful for short runs.
if isfield(target, 'reference')
    reference = target.reference;
    sample_positions = reshape(target.target_position, [], 2);
    layout_fig = new_figure(style.FigureSize, visible, style);
    ax = axes(layout_fig);
    [sensor, link] = draw_network(ax, network_topo, style);
    full_line = plot(ax, reference.states(:,1), reference.states(:,2), ...
        'k-.', 'LineWidth', style.LineWidth);
    prefix_line = plot(ax, sample_positions(:,1), sample_positions(:,2), ...
        'r-', 'LineWidth', style.LineWidth);
    axis(ax, 'equal'); grid(ax, 'on');
    fit_positions(ax, [network_topo.radar_pos; reference.states(:,1:2)]);
    xlabel(ax, 'Position x (m)'); ylabel(ax, 'Position y (m)');
    legend(ax, [sensor, link, full_line, prefix_line], ...
        'Sensor nodes', 'Communication link', 'Full reference', 'Simulated prefix', ...
        'Location', 'best');
    title(ax, sprintf('%s\nN=%d, edges=%d | %s', ...
        radar_layout_description(Run), network_topo.numNodes, ...
        network_topo.numEdges, graph_description(network_topo)), 'Interpreter', 'none');
    save_figure(layout_fig, Run.OutputFolder, 'radar_layout', visible, style, false);

    truth_fig = new_figure(style.WideFigureSize, visible, style);
    tiledlayout(truth_fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    ax = nexttile;
    plot(ax, reference.states(:,1), reference.states(:,2), 'k-.', ...
        sample_positions(:,1), sample_positions(:,2), 'r-', 'LineWidth', style.LineWidth);
    axis(ax, 'equal'); grid(ax, 'on');
    fit_positions(ax, reference.states(:,1:2));
    xlabel(ax, 'Position x (m)'); ylabel(ax, 'Position y (m)');
    legend(ax, 'Full reference', 'Simulated prefix', 'Location', 'best');
    title(ax, 'Ground truth');
    ax = nexttile;
    plot(ax, reference.time, reference.states(:,3), 'b-', ...
        reference.time, reference.states(:,4), 'r--', 'LineWidth', style.LineWidth);
    xline(ax, target.time(end), 'k--', 'Run end', 'HandleVisibility', 'off');
    grid(ax, 'on'); xlabel(ax, 'Time (s)'); ylabel(ax, 'True velocity (m/s)');
    legend(ax, 'v_x', 'v_y', 'Location', 'best');
    title(ax, 'Truth velocity');
    save_figure(truth_fig, Run.OutputFolder, 'truth_trajectory', visible, style, false);
end
end

function fig = new_figure(dimensions, visible, style)
visibility = 'off';
if visible
    visibility = 'on';
end
fig = figure('Visible', visibility, 'Color', 'w', ...
    'Units', 'centimeters', 'Position', [2, 2, dimensions], ...
    'DefaultAxesFontName', style.FontName, 'DefaultAxesFontSize', style.FontSize, ...
    'DefaultTextFontName', style.FontName, 'DefaultLegendFontName', style.FontName, ...
    'DefaultLegendFontSize', style.LegendFontSize, ...
    'DefaultAxesLabelFontSizeMultiplier', style.LabelFontSize / style.FontSize);
end

function [sensor, link] = draw_network(ax, network, style)
% NaN-separated segments draw each undirected communication link once.
[i, j] = find(triu(network.adj_matrix, 1));
p = network.radar_pos;
x = [p(i,1), p(j,1), nan(numel(i),1)].';
y = [p(i,2), p(j,2), nan(numel(i),1)].';
link = plot(ax, x(:), y(:), 'k-', 'LineWidth', style.LinkLineWidth);
hold(ax, 'on');
sensor = plot(ax, p(:,1), p(:,2), 'r.', 'MarkerSize', style.SensorMarkerSize);
end

function draw_tracking(ax, network, truth, distributed, central, method, style)
[sensor, link] = draw_network(ax, network, style);
indices = marker_indices(size(truth,2), style.NumMarkers);
gt = plot(ax, truth(1,:), truth(2,:), 'k-.', 'LineWidth', style.LineWidth);
da = plot(ax, distributed(1,:), distributed(2,:), 'ro--', ...
    'LineWidth', style.LineWidth, 'MarkerSize', style.MarkerSize, 'MarkerIndices', indices);
ca = plot(ax, central(1,:), central(2,:), 'bs-', ...
    'LineWidth', style.LineWidth, 'MarkerSize', style.MarkerSize, 'MarkerIndices', indices);
axis(ax, 'equal'); grid(ax, 'on');
fit_positions(ax, [network.radar_pos; truth(1:2,:).'; ...
    distributed(1:2,:).'; central(1:2,:).']);
xlabel(ax, 'Position x (m)'); ylabel(ax, 'Position y (m)');
legend(ax, [gt, da, ca, sensor, link], 'True trajectory', ['D-' char(method)], ...
    ['C-' char(method)], 'Sensor nodes', 'Communication link', 'Location', 'best');
end

function draw_rmse(ax, component, time, time_label, distributed, central, bound, method, style)
indices = marker_indices(numel(time), style.NumMarkers);
hold(ax, 'on');
plot(ax, time, display_values(central(component,:), style), 'o-', ...
    'Color', style.CentralColor, 'LineWidth', style.LineWidth, ...
    'MarkerSize', style.MarkerSize, 'MarkerIndices', indices, 'DisplayName', ['C-' char(method)]);
plot(ax, time, display_values(distributed(component,:), style), 'd--', ...
    'Color', style.DistributedColor, 'LineWidth', style.LineWidth, ...
    'MarkerSize', style.MarkerSize, 'MarkerIndices', indices, 'DisplayName', ['D-' char(method)]);
if ~isempty(bound)
    plot(ax, time, display_values(bound(component,:), style), 'k--', ...
        'LineWidth', style.LineWidth, 'DisplayName', 'Root PCRLB (assumed CV)');
end
set(ax, 'YScale', style.RMSEYScale, 'YMinorGrid', 'on');
if strcmpi(style.RMSEYScale, 'log')
    values = [distributed(component,:), central(component,:)];
    if ~isempty(bound), values = [values, bound(component,:)]; end
    values = values(isfinite(values) & values > 0);
    if ~isempty(values)
        powers = [floor(log10(min(values))), ceil(log10(max(values)))];
        if powers(1) == powers(2), powers = powers + [-1, 1]; end
        ylim(ax, 10.^powers);
        yticks(ax, 10.^(powers(1):powers(2)));
    end
end
grid(ax, 'on'); xlabel(ax, time_label);
if component == 1
    ylabel(ax, 'Position RMSE (m)');
else
    ylabel(ax, 'Velocity RMSE (m/s)');
end
legend(ax, 'show', 'Location', 'best');
end

function values = display_values(values, style)
% Zero cannot appear on a logarithmic axis; never replace it with a fake floor.
if strcmpi(style.RMSEYScale, 'log')
    values(values <= 0) = NaN;
end
end

function indices = marker_indices(count, requested)
indices = unique(round(linspace(1, count, min(count, requested))));
end

function [time, label] = tracking_time(Run, target, count)
label = 'Time (s)';
if Run.Method == "EKF" && isfield(target, 'time') && numel(target.time) == count
    time = target.time(:).';
elseif isfield(Run.Results, 'batch_time') && numel(Run.Results.batch_time) == count
    time = Run.Results.batch_time(:).';
else
    time = 1:count;
    label = 'Tracking step';
end
end

function fit_positions(ax, positions)
lower = min(positions, [], 1);
upper = max(positions, [], 1);
center = (lower + upper) / 2;
span = 1.16 * max(upper - lower, 1);
% Keep equal physical scales while filling the available axes rectangle.
% Long vertical trajectories otherwise leave a narrow strip under the legend.
pixels = getpixelposition(ax);
aspect = pixels(3) / pixels(4);
span(1) = max(span(1), span(2) * aspect);
span(2) = max(span(2), span(1) / aspect);
xlim(ax, center(1) + [-0.5, 0.5] * span(1));
ylim(ax, center(2) + [-0.5, 0.5] * span(2));
end

function description = graph_description(network_topo)
% Older saved topologies contain no mode and always used the random graph.
if isfield(network_topo, 'graph_mode') && network_topo.graph_mode == "ring"
    description = 'Ring (degree 2)';
else
    description = sprintf('Random graph seed=%d', network_topo.graph_seed);
end
end

function description = radar_layout_description(Run)
% Describe saved metadata only; current defaults may use another layout.
description = 'Radar layout: saved positions (legacy run)';
if isfield(Run.network_topo, 'radar_config')
    radar = Run.network_topo.radar_config;
elseif isfield(Run.Config, 'Radar')
    radar = Run.Config.Radar;
else
    return;
end
description = 'Radar layout: saved positions';
if isfield(radar, 'Mode')
    description = sprintf('Radar layout: %s', radar.Mode);
end
if isfield(radar, 'Seed')
    description = sprintf('%s | position seed=%d', description, radar.Seed);
end
end

function save_figure(fig, output_folder, name, visible, style, paper_figure)
cleanup = onCleanup(@() close_if_hidden(fig, visible));
font_size = style.FontSize;
label_size = style.LabelFontSize;
legend_size = style.LegendFontSize;
if ~paper_figure
    font_size = style.DiagnosticFontSize;
    label_size = font_size + 2;
    legend_size = max(font_size - 2, 8);
end
for ax = findall(fig, 'Type', 'axes').'
    set(ax, 'FontName', style.FontName, 'FontSize', font_size, ...
        'LineWidth', style.AxesLineWidth, 'Box', 'on', 'TickDir', 'in', ...
        'GridColor', [0.6, 0.6, 0.6], 'GridAlpha', 0.18, ...
        'MinorGridColor', [0.65, 0.65, 0.65], 'MinorGridAlpha', 0.10, ...
        'MinorGridLineStyle', ':');
    set([ax.XLabel, ax.YLabel], 'FontName', style.FontName, 'FontSize', label_size);
    set(ax.Title, 'FontName', style.FontName, 'FontSize', font_size, 'FontWeight', 'normal');
end
set(findall(fig, 'Type', 'legend'), 'FontName', style.FontName, ...
    'FontSize', legend_size, 'Box', 'on', 'EdgeColor', 'k');
if ~isempty(output_folder)
    drawnow;
    exportgraphics(fig, fullfile(output_folder, [name '.png']), 'Resolution', style.Resolution);
    if paper_figure && style.ExportPDF
        exportgraphics(fig, fullfile(output_folder, [name '.pdf']), 'ContentType', 'vector');
    end
end
end

function close_if_hidden(fig, visible)
if ~visible && isgraphics(fig)
    close(fig);
end
end
