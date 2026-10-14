function figureHandle = plot_snr_crlb_rmse_study(studyInput, outputDir)
%PLOT_SNR_CRLB_RMSE_STUDY Plot the local-neighborhood SNR/CR study.
%   This Study-compatible implementation preserves the active layout,
%   metric aggregation, typography, log axes, and export formats from:
%   /Users/shaohung/Documents/Codex/2026-07-11/i/outputs/
%   plot_snr_crlb_rmse.m
%
%   FIG = PLOT_SNR_CRLB_RMSE_STUDY(STUDY, OUTPUTDIR) accepts the Study
%   structure returned by RUN_LOCAL_NEIGHBORHOOD_EKF_STUDY. It creates a
%   2-by-3 tiled figure whose rows are position/velocity error and whose
%   columns are 20/30/40 dB. Each tile contains D-EKF RMSE and local PCRLB
%   curves for communication radii 30, 50, and 500, plus black C-EKF RMSE
%   and centralized-PCRLB references.
%
%   STUDYINPUT may instead be a MAT-file path. The MAT file must contain a
%   scalar variable named Study; legacy Log/Results files are not accepted.
%
%   Position and velocity curves are computed from the component metrics:
%
%       position = sqrt(metric_x.^2  + metric_y.^2)
%       velocity = sqrt(metric_vx.^2 + metric_vy.^2)
%
%   The function writes the following files to OUTPUTDIR:
%
%       pos_vel_rmse_pcrlb_by_snr_cr.png   (300 dpi)
%       pos_vel_rmse_pcrlb_by_snr_cr.pdf   (vector)
%       pos_vel_rmse_pcrlb_by_snr_cr.fig
%
%   The figure is created invisibly for compatibility with batch/headless
%   MATLAB sessions. Set FIG.Visible = 'on' after the call to display it.

    Study = select_plot_data(load_study(studyInput));
    [time, snrValues, snrIndices, communicationRadii, radiusIndices] = ...
        validate_study(Study);
    outputDirectory = validate_output_directory(outputDir);

    rowLabels = {'Position RMSE', 'Velocity RMSE'};
    stateIndices = {[1, 2], [3, 4]};
    radiusColors = lines(numel(communicationRadii));
    radiusMarkers = {'s', 'o', '^'};
    commonYLimits = compute_common_y_limits( ...
        Study, stateIndices, snrIndices, radiusIndices);
    markerIndices = get_marker_indices(numel(time), 28);

    figureHandle = figure( ...
        'Color', 'w', ...
        'Visible', 'off', ...
        'Name', 'Position and velocity RMSE/PCRLB by SNR and CR', ...
        'NumberTitle', 'off', ...
        'Position', [100, 100, 1280, 720]);
    layout = tiledlayout(figureHandle, 2, 3, ...
        'TileSpacing', 'compact', 'Padding', 'compact');

    legendHandles = gobjects(2 * numel(communicationRadii) + 2, 1);
    legendLabels = cell(1, numel(legendHandles));

    for rowIndex = 1:2
        for displayedSNRIndex = 1:numel(snrValues)
            snrIndex = snrIndices(displayedSNRIndex);
            axisHandle = nexttile(layout, ...
                (rowIndex - 1) * numel(snrValues) + displayedSNRIndex);
            hold(axisHandle, 'on');

            lineIndex = 0;
            for displayedRadiusIndex = 1:numel(communicationRadii)
                radiusIndex = radiusIndices(displayedRadiusIndex);
                distributedRMSE = combine_components( ...
                    Study.rmse_D, stateIndices{rowIndex}, snrIndex, radiusIndex);
                distributedPCRLB = combine_components( ...
                    Study.pcrlb_D, stateIndices{rowIndex}, snrIndex, radiusIndex);

                lineIndex = lineIndex + 1;
                currentRMSE = plot(axisHandle, time, ...
                    positive_for_log(distributedRMSE), ...
                    'LineStyle', '-', ...
                    'LineWidth', 1.45, ...
                    'Marker', radiusMarkers{displayedRadiusIndex}, ...
                    'MarkerIndices', markerIndices, ...
                    'MarkerSize', 5.2, ...
                    'Color', radiusColors(displayedRadiusIndex, :), ...
                    'DisplayName', sprintf('D+ADMM RMSE, CR %g', ...
                        communicationRadii(displayedRadiusIndex)));

                lineIndex = lineIndex + 1;
                currentPCRLB = plot(axisHandle, time, ...
                    positive_for_log(distributedPCRLB), ...
                    'LineStyle', '--', ...
                    'LineWidth', 1.35, ...
                    'Color', radiusColors(displayedRadiusIndex, :), ...
                    'DisplayName', sprintf('D+ADMM PCRLB, CR %g', ...
                        communicationRadii(displayedRadiusIndex)));

                if rowIndex == 1 && displayedSNRIndex == 1
                    legendHandles(lineIndex - 1) = currentRMSE;
                    legendHandles(lineIndex) = currentPCRLB;
                    legendLabels{lineIndex - 1} = sprintf('D+ADMM RMSE (CR %g)', ...
                        communicationRadii(displayedRadiusIndex));
                    legendLabels{lineIndex} = sprintf('D+ADMM PCRLB (CR %g)', ...
                        communicationRadii(displayedRadiusIndex));
                end
            end

            centralizedRMSE = combine_components( ...
                Study.rmse_C, stateIndices{rowIndex}, snrIndex);
            centralizedPCRLB = combine_components( ...
                Study.pcrlb_C, stateIndices{rowIndex}, snrIndex);

            lineIndex = lineIndex + 1;
            currentCentralizedRMSE = plot(axisHandle, time, ...
                positive_for_log(centralizedRMSE), ...
                'LineStyle', '-', ...
                'LineWidth', 1.65, ...
                'Marker', 'd', ...
                'MarkerIndices', markerIndices, ...
                'MarkerSize', 5.0, ...
                'Color', 'k', ...
                'DisplayName', 'C-EKF RMSE');
            lineIndex = lineIndex + 1;
            currentCentralizedPCRLB = plot(axisHandle, time, ...
                positive_for_log(centralizedPCRLB), ...
                'LineStyle', '--', ...
                'LineWidth', 1.55, ...
                'Color', 'k', ...
                'DisplayName', 'C-EKF PCRLB');

            if rowIndex == 1 && displayedSNRIndex == 1
                legendHandles(lineIndex - 1) = currentCentralizedRMSE;
                legendHandles(lineIndex) = currentCentralizedPCRLB;
                legendLabels{lineIndex - 1} = 'C RMSE';
                legendLabels{lineIndex} = 'C PCRLB';
            end

            if rowIndex == 1
                title(axisHandle, sprintf('%g dB', ...
                    snrValues(displayedSNRIndex)), 'FontWeight', 'normal');
                axisHandle.XTickLabel = [];
            else
                xlabel(axisHandle, 'time (s)');
            end
            if displayedSNRIndex == 1
                ylabel(axisHandle, rowLabels{rowIndex});
            end
            if numel(time) > 1
                xlim(axisHandle, [time(1), time(end)]);
            end

            grid(axisHandle, 'on');
            box(axisHandle, 'on');
            axisHandle.FontName = 'Times New Roman';
            axisHandle.FontSize = 12;
            axisHandle.Toolbar.Visible = 'off';
            apply_log_y_axis(axisHandle, commonYLimits(rowIndex, :));
        end
    end

    legendHandle = legend(legendHandles, legendLabels, ...
        'Orientation', 'horizontal');
    legendHandle.Layout.Tile = 'south';
    legendHandle.NumColumns = 4;

    drawnow;
    outputStem = fullfile(outputDirectory, ...
        'pos_vel_rmse_pcrlb_by_snr_cr');
    exportgraphics(figureHandle, [outputStem, '.png'], 'Resolution', 300);
    exportgraphics(figureHandle, [outputStem, '.pdf'], ...
        'ContentType', 'vector');
    savefig(figureHandle, [outputStem, '.fig']);
end

function Study = select_plot_data(Study)
    plotFields = {'plot_time', 'rmse_D_plot', 'rmse_C_plot', ...
        'pcrlb_D_plot', 'pcrlb_C_plot'};
    if all(isfield(Study, plotFields))
        Study.time = Study.plot_time;
        Study.rmse_D = Study.rmse_D_plot;
        Study.rmse_C = Study.rmse_C_plot;
        Study.pcrlb_D = Study.pcrlb_D_plot;
        Study.pcrlb_C = Study.pcrlb_C_plot;
    end
end

function Study = load_study(studyInput)
    if isstruct(studyInput)
        if ~isscalar(studyInput)
            error('SNRCRStudy:InvalidStudy', ...
                'The Study input must be a scalar structure.');
        end
        Study = studyInput;
        return;
    end

    validString = isstring(studyInput) && isscalar(studyInput) && ...
        strlength(studyInput) > 0;
    validCharacterVector = ischar(studyInput) && isrow(studyInput) && ...
        ~isempty(studyInput);
    if ~(validString || validCharacterVector)
        error('SNRCRStudy:InvalidInput', ...
            'studyInput must be a scalar Study structure or a MAT-file path.');
    end

    matFile = char(studyInput);
    if exist(matFile, 'file') ~= 2
        error('SNRCRStudy:MissingMATFile', ...
            'Study MAT file not found: %s', matFile);
    end

    loaded = load(matFile, 'Study');
    if ~isfield(loaded, 'Study') || ~isstruct(loaded.Study) || ...
            ~isscalar(loaded.Study)
        error('SNRCRStudy:MissingStudyVariable', ...
            'MAT file must contain one scalar structure named Study: %s', ...
            matFile);
    end
    Study = loaded.Study;
end

function [time, snrValues, snrIndices, communicationRadii, radiusIndices] = ...
        validate_study(Study)
    require_fields(Study, ...
        {'config', 'time', 'rmse_D', 'rmse_C', 'pcrlb_D', 'pcrlb_C'}, ...
        'Study');
    if ~isstruct(Study.config) || ~isscalar(Study.config)
        error('SNRCRStudy:InvalidConfig', ...
            'Study.config must be a scalar structure.');
    end
    require_fields(Study.config, {'SNRdB', 'CommunicationRadii'}, ...
        'Study.config');

    validateattributes(Study.config.SNRdB, {'numeric'}, ...
        {'real', 'finite', 'vector', 'nonempty'}, mfilename, ...
        'Study.config.SNRdB');
    validateattributes(Study.config.CommunicationRadii, {'numeric'}, ...
        {'real', 'finite', 'positive', 'vector', 'nonempty'}, mfilename, ...
        'Study.config.CommunicationRadii');
    validateattributes(Study.time, {'numeric'}, ...
        {'real', 'finite', 'vector', 'nonempty'}, mfilename, 'Study.time');

    configuredSNR = double(Study.config.SNRdB(:).');
    configuredRadii = double(Study.config.CommunicationRadii(:).');
    time = double(Study.time(:).');
    requestedSNR = [20, 30, 40];
    requestedRadii = [30, 50, 500];

    [hasSNR, snrIndices] = ismember(requestedSNR, configuredSNR);
    [hasRadii, radiusIndices] = ismember(requestedRadii, configuredRadii);
    if numel(configuredSNR) ~= 3 || ~all(hasSNR)
        error('SNRCRStudy:RequiredSNR', ...
            'Study.config.SNRdB must contain exactly [20 30 40].');
    end
    if numel(configuredRadii) ~= 3 || ~all(hasRadii)
        error('SNRCRStudy:RequiredRadii', ...
            ['Study.config.CommunicationRadii must contain exactly ', ...
             '[30 50 500].']);
    end
    if numel(time) > 1 && any(diff(time) <= 0)
        error('SNRCRStudy:InvalidTime', ...
            'Study.time must be strictly increasing.');
    end

    numTime = numel(time);
    numSNR = numel(configuredSNR);
    numRadii = numel(configuredRadii);
    validate_metric(Study.rmse_D, [4, numTime, numSNR, numRadii], ...
        'Study.rmse_D');
    validate_metric(Study.rmse_C, [4, numTime, numSNR], ...
        'Study.rmse_C');
    validate_metric(Study.pcrlb_D, [4, numTime, numSNR, numRadii], ...
        'Study.pcrlb_D');
    validate_metric(Study.pcrlb_C, [4, numTime, numSNR], ...
        'Study.pcrlb_C');

    snrValues = requestedSNR;
    communicationRadii = requestedRadii;
end

function validate_metric(metric, expectedSize, metricName)
    if ~isnumeric(metric) || ~isreal(metric) || ...
            any(~isfinite(metric(:))) || any(metric(:) < 0) || ...
            ~isequal(size(metric), expectedSize)
        error('SNRCRStudy:InvalidMetric', ...
            '%s must be a finite, real, nonnegative array of size %s.', ...
            metricName, mat2str(expectedSize));
    end
end

function require_fields(structure, requiredFields, structureName)
    missingFields = requiredFields(~isfield(structure, requiredFields));
    if ~isempty(missingFields)
        error('SNRCRStudy:MissingField', ...
            '%s is missing required field(s): %s.', structureName, ...
            strjoin(missingFields, ', '));
    end
end

function outputDirectory = validate_output_directory(outputDir)
    validString = isstring(outputDir) && isscalar(outputDir) && ...
        strlength(outputDir) > 0;
    validCharacterVector = ischar(outputDir) && isrow(outputDir) && ...
        ~isempty(outputDir);
    if ~(validString || validCharacterVector)
        error('SNRCRStudy:InvalidOutputDirectory', ...
            'outputDir must be a nonempty character vector or string scalar.');
    end

    outputDirectory = char(outputDir);
    if isfile(outputDirectory)
        error('SNRCRStudy:OutputPathIsFile', ...
            'outputDir points to an existing file: %s', outputDirectory);
    end
    if ~isfolder(outputDirectory)
        [created, message] = mkdir(outputDirectory);
        if ~created
            error('SNRCRStudy:CreateOutputDirectory', ...
                'Could not create outputDir "%s": %s', ...
                outputDirectory, message);
        end
    end
end

function metric = combine_components(metricArray, componentIndices, ...
        snrIndex, varargin)
    if isempty(varargin)
        componentMetric = reshape( ...
            metricArray(componentIndices, :, snrIndex), ...
            numel(componentIndices), []);
    else
        radiusIndex = varargin{1};
        componentMetric = reshape( ...
            metricArray(componentIndices, :, snrIndex, radiusIndex), ...
            numel(componentIndices), []);
    end
    metric = sqrt(sum(componentMetric.^2, 1));
end

function commonYLimits = compute_common_y_limits(Study, stateIndices, ...
        snrIndices, radiusIndices)
    commonYLimits = nan(2, 2);
    metricNames = {'position', 'velocity'};
    for rowIndex = 1:2
        positiveValues = [];
        for snrIndex = snrIndices
            for radiusIndex = radiusIndices
                distributedRMSE = combine_components( ...
                    Study.rmse_D, stateIndices{rowIndex}, snrIndex, radiusIndex);
                distributedPCRLB = combine_components( ...
                    Study.pcrlb_D, stateIndices{rowIndex}, snrIndex, radiusIndex);
                positiveValues = append_positive_values( ...
                    positiveValues, distributedRMSE);
                positiveValues = append_positive_values( ...
                    positiveValues, distributedPCRLB);
            end
            centralizedRMSE = combine_components( ...
                Study.rmse_C, stateIndices{rowIndex}, snrIndex);
            centralizedPCRLB = combine_components( ...
                Study.pcrlb_C, stateIndices{rowIndex}, snrIndex);
            positiveValues = append_positive_values( ...
                positiveValues, centralizedRMSE);
            positiveValues = append_positive_values( ...
                positiveValues, centralizedPCRLB);
        end

        if isempty(positiveValues)
            error('SNRCRStudy:NoPositiveMetric', ...
                'The %s metrics contain no positive values for a log axis.', ...
                metricNames{rowIndex});
        end

        lowerLimit = 10 ^ floor(log10(min(positiveValues)));
        upperLimit = 10 ^ ceil(log10(max(positiveValues)));
        if lowerLimit == upperLimit
            lowerLimit = lowerLimit / 10;
            upperLimit = upperLimit * 10;
        end
        commonYLimits(rowIndex, :) = [lowerLimit, upperLimit];
    end
end

function positiveValues = append_positive_values(positiveValues, values)
    values = values(isfinite(values) & values > 0);
    positiveValues = [positiveValues, values];
end

function values = positive_for_log(values)
    values(values <= 0) = nan;
end

function apply_log_y_axis(axisHandle, limits)
    axisHandle.YScale = 'log';
    axisHandle.YAxis.Exponent = 0;
    axisHandle.YLim = limits;
    axisHandle.YTick = 10 .^ ...
        (floor(log10(limits(1))):ceil(log10(limits(2))));
    axisHandle.YMinorGrid = 'on';
    ytickformat(axisHandle, '%.3g');
end

function markerIndices = get_marker_indices(numPoints, maxMarkers)
    if numPoints < 1
        markerIndices = [];
        return;
    end
    step = max(1, ceil(numPoints / maxMarkers));
    markerIndices = 1:step:numPoints;
    if markerIndices(end) ~= numPoints
        markerIndices(end + 1) = numPoints;
    end
end
