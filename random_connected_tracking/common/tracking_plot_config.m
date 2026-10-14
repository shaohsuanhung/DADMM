function style = tracking_plot_config(overrides)
%TRACKING_PLOT_CONFIG Manuscript-style figures; sizes are in centimeters.
% Trajectory conventions follow the original MAP/EKF plot_trajectory scripts.
style.FontName = 'Times New Roman';
style.FontSize = 20;
style.LabelFontSize = 24;
style.LegendFontSize = 16;
style.DiagnosticFontSize = 16;
style.AxesLineWidth = 1.5;
style.LineWidth = 2;
style.MarkerSize = 8;
style.NumMarkers = 20;
style.SensorMarkerSize = 22;
style.LinkLineWidth = 1.2;
style.FigureSize = [20, 14.5];
style.RMSEFigureSize = [20, 23];
style.OverviewFigureSize = [36, 25];
style.WideFigureSize = [32, 13];
style.Resolution = 300;
style.ExportPDF = true;         % Vector trajectory.pdf and rmse.pdf.
style.ShowPCRLB = true;         % Only when the saved EKF result contains it.
style.RMSEYScale = 'log';
style.CentralColor = [0, 0.4470, 0.7410];
style.DistributedColor = [0.8500, 0.3250, 0.0980];

if nargin == 0
    return;
end
names = fieldnames(overrides);
for k = 1:numel(names)
    assert(isfield(style, names{k}), 'RadarTracking:UnknownPlotSetting', ...
        'Unknown plotting setting: %s', names{k});
    style.(names{k}) = overrides.(names{k});
end
end
