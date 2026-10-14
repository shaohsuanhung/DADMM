function Run = replot_tracking_results(results_file, visible, plot_options)
%REPLOT_TRACKING_RESULTS Redraw a saved run without running the estimator.
% Run = replot_tracking_results('results/EKF_.../results.mat');
% A result folder is also accepted. Pass true to keep figures open.
% PNGs and manuscript PDFs are regenerated beside the unchanged MAT file.
% Optional plot_options overrides the saved style, e.g. struct('FontSize', 18).
if nargin < 2
    visible = false;
end
if isfolder(results_file)
    results_file = fullfile(results_file, 'results.mat');
end
assert(isfile(results_file), 'RadarTracking:MissingResults', ...
    'Cannot find saved results: %s', results_file);
[~, info] = fileattrib(results_file);
saved = load(info.Name, 'Run');
assert(isfield(saved, 'Run'), 'RadarTracking:MissingRun', ...
    'The MAT file must contain the saved Run structure.');
Run = saved.Run;
% Use the supplied file location even if the result folder has been moved.
Run.OutputFolder = fileparts(info.Name);
project_root = fileparts(mfilename('fullpath'));
previous_path = path;
path_cleanup = onCleanup(@() path(previous_path));
addpath(fullfile(project_root, 'common'));
if nargin >= 3
    if isfield(Run.Config, 'Plot')
        style = tracking_plot_config(Run.Config.Plot);
    else
        style = tracking_plot_config();
    end
    names = fieldnames(plot_options);
    for k = 1:numel(names)
        style.(names{k}) = plot_options.(names{k});
    end
    Run.Config.Plot = tracking_plot_config(style);
end
plot_tracking_results(Run, visible);
fprintf('Replotted saved %s run: %s\n', Run.Method, Run.OutputFolder);
end
