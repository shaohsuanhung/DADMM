function fig = plot_dekf_3_7_cekf_cpcrlb(studyInput, outputDir)
%PLOT_DEKF_3_7_CEKF_CPCRLB Compare local D-EKF and centralized references.
%   FIG = PLOT_DEKF_3_7_CEKF_CPCRLB(STUDYINPUT, OUTPUTDIR) plots a 2-by-3
%   grid. Rows show combined position and velocity RMSE; columns show
%   20/30/40 dB. Every tile contains exactly four curves:
%       D-EKF-3 + ADMM, D-EKF-7 + ADMM, C-EKF, and C-PCRLB.
%
%   STUDYINPUT may be the Study struct returned by
%   RUN_LOCAL_NEIGHBORHOOD_EKF_STUDY or a MAT-file containing Study.

    Study = select_plot_data(load_study(studyInput));
    validate_study(Study);

    if ~(ischar(outputDir) || (isstring(outputDir) && isscalar(outputDir)))
        error('DEKFPlot:OutputDir', ...
            'outputDir must be a character vector or scalar string.');
    end
    outputDir = char(outputDir);
    if ~isfolder(outputDir)
        mkdir(outputDir);
    end

    snrValues = [20, 30, 40];
    radiusValues = [30, 50];
    [foundSnr, snrIndices] = ismember(snrValues, Study.config.SNRdB);
    [foundRadius, radiusIndices] = ismember( ...
        radiusValues, Study.config.CommunicationRadii);
    assert(all(foundSnr) && all(foundRadius), ...
        'DEKFPlot:MissingConfiguration', ...
        'Study must contain SNR [20 30 40] and CR [30 50].');

    time = Study.time(:).';
    markerIndices = marker_indices(numel(time), 28);
    stateGroups = {[1, 2], [3, 4]};
    rowLabels = {'Position RMSE', 'Velocity RMSE'};
    commonYLimits = common_y_limits( ...
        Study, stateGroups, snrIndices, radiusIndices);

    blue = [0, 0.4470, 0.7410];
    orange = [0.8500, 0.3250, 0.0980];

    fig = figure('Color', 'w', 'Visible', 'off', ...
        'Name', 'D-EKF-3/7 nodes, C-EKF, and C-PCRLB', ...
        'NumberTitle', 'off', 'Position', [100, 100, 1280, 720]);
    layout = tiledlayout(fig, 2, 3, ...
        'TileSpacing', 'compact', 'Padding', 'compact');

    legendHandles = gobjects(4, 1);
    for rowIdx = 1:2
        for snrDisplayIdx = 1:3
            snrIdx = snrIndices(snrDisplayIdx);
            ax = nexttile(layout, (rowIdx - 1) * 3 + snrDisplayIdx);
            hold(ax, 'on');

            d3 = combine_metric(Study.rmse_D, stateGroups{rowIdx}, ...
                snrIdx, radiusIndices(1));
            d7 = combine_metric(Study.rmse_D, stateGroups{rowIdx}, ...
                snrIdx, radiusIndices(2));
            central = combine_metric(Study.rmse_C, ...
                stateGroups{rowIdx}, snrIdx);
            centralBound = combine_metric(Study.pcrlb_C, ...
                stateGroups{rowIdx}, snrIdx);

            h1 = plot(ax, time, positive(d3), '-', ...
                'Color', blue, 'LineWidth', 1.5, 'Marker', 's', ...
                'MarkerIndices', markerIndices, 'MarkerSize', 5.2, ...
                'DisplayName', 'D-EKF-3 + ADMM');
            h2 = plot(ax, time, positive(d7), '-', ...
                'Color', orange, 'LineWidth', 1.5, 'Marker', 'o', ...
                'MarkerIndices', markerIndices, 'MarkerSize', 5.2, ...
                'DisplayName', 'D-EKF-7 + ADMM');
            h3 = plot(ax, time, positive(central), '-', ...
                'Color', 'k', 'LineWidth', 1.65, 'Marker', 'd', ...
                'MarkerIndices', markerIndices, 'MarkerSize', 5.0, ...
                'DisplayName', 'C-EKF');
            h4 = plot(ax, time, positive(centralBound), '--', ...
                'Color', 'k', 'LineWidth', 1.55, ...
                'DisplayName', 'C-PCRLB');

            if rowIdx == 1 && snrDisplayIdx == 1
                legendHandles = [h1; h2; h3; h4];
            end

            if rowIdx == 1
                title(ax, sprintf('%g dB', snrValues(snrDisplayIdx)), ...
                    'FontWeight', 'normal');
                ax.XTickLabel = [];
            else
                xlabel(ax, 'time (s)');
            end
            if snrDisplayIdx == 1
                ylabel(ax, rowLabels{rowIdx});
            end
            if numel(time) > 1
                xlim(ax, [time(1), time(end)]);
            end
            ax.YScale = 'log';
            ylim(ax, commonYLimits(rowIdx, :));
            ax.YTick = 10 .^ (floor(log10(commonYLimits(rowIdx, 1))): ...
                ceil(log10(commonYLimits(rowIdx, 2))));
            ax.YMinorGrid = 'on';
            ax.FontName = 'Times New Roman';
            ax.FontSize = 12;
            ax.YAxis.Exponent = 0;
            ax.Toolbar.Visible = 'off';
            grid(ax, 'on');
            box(ax, 'on');
            ytickformat(ax, '%.3g');
        end
    end

    legendHandle = legend(legendHandles, ...
        {'D-EKF-3 + ADMM', 'D-EKF-7 + ADMM', 'C-EKF', 'C-PCRLB'}, ...
        'Orientation', 'horizontal');
    legendHandle.Layout.Tile = 'south';
    legendHandle.NumColumns = 4;
    drawnow;
    stem = fullfile(outputDir, 'dekf_3_7_cekf_cpcrlb_by_snr');
    exportgraphics(fig, [stem, '.png'], 'Resolution', 300);
    exportgraphics(fig, [stem, '.pdf'], 'ContentType', 'vector');
    savefig(fig, [stem, '.fig']);
end

function Study = select_plot_data(Study)
    plotFields = {'plot_time', 'rmse_D_plot', 'rmse_C_plot', ...
        'pcrlb_C_plot'};
    if all(isfield(Study, plotFields))
        Study.time = Study.plot_time;
        Study.rmse_D = Study.rmse_D_plot;
        Study.rmse_C = Study.rmse_C_plot;
        Study.pcrlb_C = Study.pcrlb_C_plot;
    end
end

function Study = load_study(studyInput)
    if isstruct(studyInput) && isscalar(studyInput)
        Study = studyInput;
        return;
    end
    if ~(ischar(studyInput) || ...
            (isstring(studyInput) && isscalar(studyInput)))
        error('DEKFPlot:StudyInput', ...
            'studyInput must be a scalar Study struct or MAT-file path.');
    end
    loaded = load(char(studyInput), 'Study');
    if ~isfield(loaded, 'Study') || ~isscalar(loaded.Study)
        error('DEKFPlot:MissingStudy', ...
            'The MAT file must contain a scalar Study struct.');
    end
    Study = loaded.Study;
end

function validate_study(Study)
    fields = {'config', 'time', 'rmse_D', 'rmse_C', 'pcrlb_C'};
    assert(all(isfield(Study, fields)), 'DEKFPlot:MissingField', ...
        'Study is missing one or more required fields.');
    assert(all(isfinite(Study.rmse_D), 'all') && ...
        all(isfinite(Study.rmse_C), 'all') && ...
        all(isfinite(Study.pcrlb_C), 'all'), ...
        'DEKFPlot:NonfiniteMetric', 'Metrics must be finite.');
    assert(all(Study.rmse_D >= 0, 'all') && ...
        all(Study.rmse_C >= 0, 'all') && ...
        all(Study.pcrlb_C >= 0, 'all'), ...
        'DEKFPlot:NegativeMetric', 'Metrics must be nonnegative.');
end

function metric = combine_metric(metricArray, stateIndices, snrIdx, varargin)
    if isempty(varargin)
        components = reshape(metricArray(stateIndices, :, snrIdx), 2, []);
    else
        components = reshape( ...
            metricArray(stateIndices, :, snrIdx, varargin{1}), 2, []);
    end
    metric = sqrt(sum(components.^2, 1));
end

function limits = common_y_limits(Study, stateGroups, snrIndices, radiusIndices)
    limits = zeros(2, 2);
    for rowIdx = 1:2
        values = [];
        for snrIdx = snrIndices
            values = append_positive(values, combine_metric( ...
                Study.rmse_D, stateGroups{rowIdx}, snrIdx, radiusIndices(1)));
            values = append_positive(values, combine_metric( ...
                Study.rmse_D, stateGroups{rowIdx}, snrIdx, radiusIndices(2)));
            values = append_positive(values, combine_metric( ...
                Study.rmse_C, stateGroups{rowIdx}, snrIdx));
            values = append_positive(values, combine_metric( ...
                Study.pcrlb_C, stateGroups{rowIdx}, snrIdx));
        end
        lower = 10 ^ floor(log10(min(values)));
        upper = 10 ^ ceil(log10(max(values)));
        if lower == upper
            lower = lower / 10;
            upper = upper * 10;
        end
        limits(rowIdx, :) = [lower, upper];
    end
end

function values = append_positive(values, newValues)
    values = [values, newValues(isfinite(newValues) & newValues > 0)];
end

function values = positive(values)
    values(values <= 0) = nan;
end

function indices = marker_indices(numPoints, maxMarkers)
    step = max(1, ceil(numPoints / maxMarkers));
    indices = 1:step:numPoints;
    if indices(end) ~= numPoints
        indices(end + 1) = numPoints;
    end
end
