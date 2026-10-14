function figures = plot_local_neighborhood_ekf_study(Study, output_dir)
%PLOT_LOCAL_NEIGHBORHOOD_EKF_STUDY Plot and save the local-EKF SNR/CR study.
%
%   FIGURES = PLOT_LOCAL_NEIGHBORHOOD_EKF_STUDY(STUDY, OUTPUT_DIR) creates
%   one 3-by-3 tiled figure for each state. Rows correspond to SNR values
%   and columns correspond to communication radii. Every tile compares
%   D-EKF RMSE, C-EKF RMSE, the local-neighborhood PCRLB, and the
%   centralized PCRLB over time. Figures are created invisibly so the
%   function also works in MATLAB batch/headless sessions.
%
%   Required STUDY fields:
%       Study.config.SNRdB                 [1 x 3]
%       Study.config.CommunicationRadii    [1 x 3]
%       Study.time                         [1 x T]
%       Study.rmse_D                       [4 x T x 3 x 3]
%       Study.rmse_C                       [4 x T x 3]
%       Study.pcrlb_D                      [4 x T x 3 x 3]
%       Study.pcrlb_C                      [4 x T x 3]
%
%   The returned FIGURES is a 4-by-1 graphics-handle array ordered as
%   [x, y, vx, vy]. Each figure is saved to OUTPUT_DIR as both PNG and FIG.

    [snrValues, communicationRadii, time] = validate_study(Study);
    outputDirectory = validate_output_directory(output_dir);

    stateLabels = {'x', 'y', 'v_x', 'v_y'};
    stateFileNames = {'x', 'y', 'vx', 'vy'};
    distributedColor = [0.0000, 0.4470, 0.7410];
    centralizedColor = [0.8500, 0.3250, 0.0980];

    numStates = 4;
    numSNR = numel(snrValues);
    numRadii = numel(communicationRadii);
    figures = gobjects(numStates, 1);

    for stateIndex = 1:numStates
        currentFigure = figure( ...
            'Color', 'w', ...
            'Visible', 'off', ...
            'Name', sprintf('Local-neighborhood EKF study: %s', ...
                stateLabels{stateIndex}), ...
            'NumberTitle', 'off', ...
            'Position', [100, 100, 1500, 1000]);
        figures(stateIndex) = currentFigure;

        layout = tiledlayout(currentFigure, numSNR, numRadii, ...
            'TileSpacing', 'compact', 'Padding', 'compact');
        title(layout, sprintf('Local-neighborhood D-EKF vs C-EKF: %s state', ...
            stateLabels{stateIndex}), 'Interpreter', 'tex', ...
            'FontWeight', 'bold');

        legendHandles = gobjects(4, 1);
        for snrIndex = 1:numSNR
            for radiusIndex = 1:numRadii
                axisHandle = nexttile(layout, ...
                    (snrIndex - 1) * numRadii + radiusIndex);
                hold(axisHandle, 'on');
                grid(axisHandle, 'on');
                box(axisHandle, 'on');

                distributedRMSE = reshape( ...
                    Study.rmse_D(stateIndex, :, snrIndex, radiusIndex), ...
                    1, []);
                centralizedRMSE = reshape( ...
                    Study.rmse_C(stateIndex, :, snrIndex), 1, []);
                distributedPCRLB = reshape( ...
                    Study.pcrlb_D(stateIndex, :, snrIndex, radiusIndex), ...
                    1, []);
                centralizedPCRLB = reshape( ...
                    Study.pcrlb_C(stateIndex, :, snrIndex), 1, []);

                lineHandles = gobjects(4, 1);
                lineHandles(1) = plot(axisHandle, time, distributedRMSE, ...
                    '-', 'Color', distributedColor, 'LineWidth', 1.8, ...
                    'DisplayName', 'D-EKF + ADMM');
                lineHandles(2) = plot(axisHandle, time, centralizedRMSE, ...
                    '-', 'Color', centralizedColor, 'LineWidth', 1.8, ...
                    'DisplayName', 'C-EKF');
                lineHandles(3) = plot(axisHandle, time, distributedPCRLB, ...
                    '--', 'Color', distributedColor, 'LineWidth', 1.8, ...
                    'DisplayName', 'D-EKF + ADMM PCRLB');
                lineHandles(4) = plot(axisHandle, time, centralizedPCRLB, ...
                    '--', 'Color', centralizedColor, 'LineWidth', 1.8, ...
                    'DisplayName', 'C-EKF PCRLB');

                if snrIndex == 1 && radiusIndex == 1
                    legendHandles = lineHandles;
                end

                title(axisHandle, sprintf('SNR = %g dB, CR = %g', ...
                    snrValues(snrIndex), communicationRadii(radiusIndex)));
                if snrIndex == numSNR
                    xlabel(axisHandle, 'Time (s)');
                end
                if radiusIndex == 1
                    ylabel(axisHandle, sprintf('RMSE(%s)', ...
                        stateLabels{stateIndex}), 'Interpreter', 'tex');
                end
                if numel(time) > 1
                    xlim(axisHandle, [time(1), time(end)]);
                end
                axisHandle.FontName = 'Times New Roman';
                axisHandle.FontSize = 11;
            end
        end

        legendHandle = legend(legendHandles, ...
            {'D-EKF + ADMM', 'C-EKF', ...
             'D-EKF + ADMM PCRLB', 'C-EKF PCRLB'}, ...
            'Orientation', 'horizontal');
        legendHandle.Layout.Tile = 'south';
        legendHandle.NumColumns = 4;

        drawnow;
        fileStem = sprintf('local_neighborhood_ekf_%s', ...
            stateFileNames{stateIndex});
        exportgraphics(currentFigure, ...
            fullfile(outputDirectory, [fileStem, '.png']), ...
            'Resolution', 300);
        savefig(currentFigure, ...
            fullfile(outputDirectory, [fileStem, '.fig']));
    end
end

function [snrValues, communicationRadii, time] = validate_study(Study)
    if ~isstruct(Study) || ~isscalar(Study)
        error('LocalEKFStudy:InvalidStudy', ...
            'Study must be a scalar structure.');
    end
    require_fields(Study, ...
        {'config', 'time', 'rmse_D', 'rmse_C', 'pcrlb_D', 'pcrlb_C'}, ...
        'Study');
    if ~isstruct(Study.config) || ~isscalar(Study.config)
        error('LocalEKFStudy:InvalidConfig', ...
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

    snrValues = double(Study.config.SNRdB(:).');
    communicationRadii = ...
        double(Study.config.CommunicationRadii(:).');
    time = double(Study.time(:).');

    if numel(snrValues) ~= 3 || numel(communicationRadii) ~= 3
        error('LocalEKFStudy:ExpectedThreeByThree', ...
            ['The requested layout requires exactly three SNR values and ', ...
             'three communication radii.']);
    end
    if numel(unique(snrValues)) ~= numel(snrValues) || ...
            numel(unique(communicationRadii)) ~= numel(communicationRadii)
        error('LocalEKFStudy:DuplicateConfiguration', ...
            'SNR values and communication radii must each be unique.');
    end
    if numel(time) > 1 && any(diff(time) <= 0)
        error('LocalEKFStudy:InvalidTime', ...
            'Study.time must be strictly increasing.');
    end

    numTime = numel(time);
    numSNR = numel(snrValues);
    numRadii = numel(communicationRadii);
    validate_metric(Study.rmse_D, [4, numTime, numSNR, numRadii], ...
        'Study.rmse_D');
    validate_metric(Study.rmse_C, [4, numTime, numSNR], ...
        'Study.rmse_C');
    validate_metric(Study.pcrlb_D, [4, numTime, numSNR, numRadii], ...
        'Study.pcrlb_D');
    validate_metric(Study.pcrlb_C, [4, numTime, numSNR], ...
        'Study.pcrlb_C');
end

function validate_metric(metric, expectedSize, metricName)
    if ~isnumeric(metric) || ~isreal(metric) || ...
            any(~isfinite(metric(:))) || any(metric(:) < 0) || ...
            ~isequal(size(metric), expectedSize)
        error('LocalEKFStudy:InvalidMetric', ...
            '%s must be a finite, real, nonnegative numeric array of size %s.', ...
            metricName, mat2str(expectedSize));
    end
end

function require_fields(structure, requiredFields, structureName)
    missingFields = requiredFields(~isfield(structure, requiredFields));
    if ~isempty(missingFields)
        error('LocalEKFStudy:MissingField', ...
            '%s is missing required field(s): %s.', structureName, ...
            strjoin(missingFields, ', '));
    end
end

function outputDirectory = validate_output_directory(output_dir)
    validString = isstring(output_dir) && isscalar(output_dir) && ...
        strlength(output_dir) > 0;
    validCharacterVector = ischar(output_dir) && isrow(output_dir) && ...
        ~isempty(output_dir);
    if ~(validString || validCharacterVector)
        error('LocalEKFStudy:InvalidOutputDirectory', ...
            'output_dir must be a nonempty character vector or string scalar.');
    end

    outputDirectory = char(output_dir);
    if isfile(outputDirectory)
        error('LocalEKFStudy:OutputPathIsFile', ...
            'output_dir points to an existing file: %s', outputDirectory);
    end
    if ~isfolder(outputDirectory)
        [created, message] = mkdir(outputDirectory);
        if ~created
            error('LocalEKFStudy:CreateOutputDirectory', ...
                'Could not create output_dir "%s": %s', ...
                outputDirectory, message);
        end
    end
end
