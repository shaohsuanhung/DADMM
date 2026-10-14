function metrics = summarize_ekf_convergence(Run)
%SUMMARIZE_EKF_CONVERGENCE Summarize a saved EKF run; no simulation is run.
% Returns one table row per MC trial and measurement epoch, and writes CSV,
% a text summary, and a diagnostic figure into the existing output folder.
assert(string(Run.Method) == "EKF", 'This diagnostic requires an EKF Run.');
assert(isfolder(Run.OutputFolder), 'Run.OutputFolder must be an existing folder.');
R = Run.Results;
cfg = R.Config;
[num_mc, num_epochs] = size(R.true_params_raw);
time = R.time(:);
assert(num_epochs > 0 && numel(time) == num_epochs, 'Truth and time logs must match.');
names = {'MC', 'Epoch', 'Time', 'Iterations', 'Converged', ...
    'PrimalResidual', 'EpsPrimal', 'PrimalRatio', ...
    'DualResidual', 'EpsDual', 'DualRatio', ...
    'NodePositionDisagreement', 'NodeVelocityDisagreement', ...
    'DistributedPositionError', 'DistributedVelocityError', ...
    'CentralPositionError', 'CentralVelocityError', 'LocalSolverFailures'};
values = zeros(num_mc * num_epochs, numel(names));
for mc = 1:num_mc
    for k = 1:num_epochs
        truth = R.true_params_raw{mc,k}(:);
        nodes = R.estimations_DA_nodes_raw{mc,k};
        state_mean = mean(nodes, 2);
        central = R.estimations_CA_raw{mc,k}(:);
        deviation = nodes - state_mean;
        primal = R.primal_residauls_raw{mc,k}(end);
        eps_primal = R.admm_eps_pri_raw{mc,k}(end);
        dual = R.dual_residauls_raw{mc,k}(end);
        eps_dual = R.admm_eps_dual_raw{mc,k}(end);
        row = (mc-1) * num_epochs + k;
        values(row,:) = [mc, k, time(k), R.convg_iter_raw{mc,k}, ...
            R.admm_converged_raw{mc,k}, primal, eps_primal, primal/eps_primal, ...
            dual, eps_dual, dual/eps_dual, ...
            sqrt(mean(sum(deviation(1:2,:).^2, 1))), ...
            sqrt(mean(sum(deviation(3:4,:).^2, 1))), ...
            norm(state_mean(1:2)-truth(1:2)), norm(state_mean(3:4)-truth(3:4)), ...
            norm(central(1:2)-truth(1:2)), norm(central(3:4)-truth(3:4)), ...
            R.fmincon_failure_count_raw{mc,k}];
    end
end
metrics = array2table(values, 'VariableNames', names);
metrics.Converged = logical(metrics.Converged);
writetable(metrics, fullfile(Run.OutputFolder, 'convergence_metrics.csv'));

% These four scalars pool squared 2-D errors across both time and MC trials.
window_rmse = sqrt(mean(values(:,14:17).^2, 1));
[max_iterations, slowest_row] = max(metrics.Iterations);
slow_mc = metrics.MC(slowest_row);
slow_epoch = metrics.Epoch(slowest_row);
ratio_pass = metrics.PrimalRatio <= 1 & metrics.DualRatio <= 1;
filename = fullfile(Run.OutputFolder, 'convergence_summary.txt');
fid = fopen(filename, 'w');
assert(fid ~= -1, 'Cannot create convergence_summary.txt.');
file_cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'EKF convergence diagnostics from saved numerical results\n');
fprintf(fid, 'MC trials: %d; epochs per trial: %d; total updates: %d\n', ...
    num_mc, num_epochs, height(metrics));
fprintf(fid, 'Observed interval: %.6f to %.6f s; duration: %.6f s\n', ...
    time(1), time(end), time(end)-time(1));
fprintf(fid, 'Full reference duration: %.6f s\n', Run.Config.Trajectory.Duration);
fprintf(fid, 'Logged stopping rule met: %d/%d; not met: %d\n', ...
    nnz(metrics.Converged), height(metrics), nnz(~metrics.Converged));
fprintf(fid, 'Both final residual ratios <= 1: %d/%d; flag mismatches: %d\n', ...
    nnz(ratio_pass), height(metrics), nnz(ratio_pass ~= metrics.Converged));
fprintf(fid, 'Iterations min/median/max: %g / %g / %g; at cap: %d\n', ...
    min(metrics.Iterations), median(metrics.Iterations), max_iterations, ...
    nnz(metrics.Iterations >= cfg.ADMMMaxIterations));
fprintf(fid, 'Max final normalized residual: primal %.9g; dual %.9g\n', ...
    max(metrics.PrimalRatio), max(metrics.DualRatio));
fprintf(fid, 'Slowest update: MC %d, epoch %d, time %.6f s\n', ...
    slow_mc, slow_epoch, time(slow_epoch));
fprintf(fid, 'Local fmincon exitflag <= 0 count (all local solves): %d\n', ...
    sum(metrics.LocalSolverFailures));
fprintf(fid, 'Updates with local solver failures: %d\n', nnz(metrics.LocalSolverFailures));
fprintf(fid, 'Finite logged metrics: %d/%d numeric values\n', ...
    nnz(isfinite(values)), numel(values));
fprintf(fid, '\nFinite-window RMSE, pooled over all MC trials AND all sampled times:\n');
fprintf(fid, 'Network mean: position %.9g m; velocity %.9g m/s\n', window_rmse(1:2));
fprintf(fid, 'Central EKF: position %.9g m; velocity %.9g m/s\n', window_rmse(3:4));
fprintf(fid, 'These scalars are not a time-wise Monte Carlo RMSE curve.\n');
fprintf(fid, 'Max node RMS disagreement: position %.9g m; velocity %.9g m/s\n', ...
    max(metrics.NodePositionDisagreement), max(metrics.NodeVelocityDisagreement));
fprintf(fid, '\nADMM cap: %d; abs tolerance: %.9g; rel tolerance: %.9g\n', ...
    cfg.ADMMMaxIterations, cfg.ADMMAbsTolerance, cfg.ADMMRelTolerance);
fprintf(fid, 'Initial penalty: %s; process acceleration variance: %.9g\n', ...
    mat2str(cfg.ADMMPenalty), cfg.ProcessAccelerationVariance);
fprintf(fid, 'TimeStep: %.9g s; Q: %s\n', cfg.TimeStep, mat2str(R.env.Q, 9));
fprintf(fid, '\nConvergence means the implemented ADMM residual test passed.\n');
if isfield(R, 'MeasurementModel') && string(R.MeasurementModel) == "first_order_affine"
    fprintf(fid, 'Model: first_order_affine; covariance uses the same fixed Jacobian.\n');
    fprintf(fid, 'Central EKF uses its own recursive prior/anchor; compare a same-prior frozen quadratic for ADMM optimality.\n');
else
    fprintf(fid, 'The central EKF is a different optimizer; its tracking error is a baseline, not an ADMM optimality gap.\n');
end
fprintf(fid, 'Nonlinear truth and CV prediction can produce tracking error even after ADMM converges.\n');

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1150 800]);
figure_cleanup = onCleanup(@() close(fig));
tiledlayout(2, 2, 'TileSpacing', 'compact');
ratio_floor = 1e-12; % Display floor only; CSV retains the unmodified values.
nexttile;
hp = semilogy(time, max(reshape(metrics.PrimalRatio, num_epochs, num_mc), ratio_floor), ...
    '-', 'Color', [0 0.447 0.741]); hold on;
hd = semilogy(time, max(reshape(metrics.DualRatio, num_epochs, num_mc), ratio_floor), ...
    '--', 'Color', [0.85 0.325 0.098]);
ht = yline(1, 'k:'); grid on;
all_ratios = [metrics.PrimalRatio; metrics.DualRatio; 1];
ylim([max(min(all_ratios)*0.8, ratio_floor), max(all_ratios)*1.2]);
xlabel('Time (s)'); ylabel('Final residual / threshold');
title(sprintf('All updates: %d/%d passed', nnz(metrics.Converged), height(metrics)));
legend([hp(1), hd(1), ht], {'Primal', 'Dual', 'Threshold'}, 'Location', 'best');

nexttile;
plot(time, reshape(metrics.Iterations, num_epochs, num_mc), '.-'); hold on;
grid on; ylim([0, max(1, max_iterations)*1.1]);
xlabel('Time (s)'); ylabel('ADMM iterations');
title(sprintf('One trace per MC trial | cap = %d', cfg.ADMMMaxIterations));

nexttile;
primal_history = R.primal_residauls_raw{slow_mc,slow_epoch}(:) ./ ...
    R.admm_eps_pri_raw{slow_mc,slow_epoch}(:);
dual_history = R.dual_residauls_raw{slow_mc,slow_epoch}(:) ./ ...
    R.admm_eps_dual_raw{slow_mc,slow_epoch}(:);
semilogy(max(primal_history, ratio_floor), '-', 'LineWidth', 1.2); hold on;
semilogy(max(dual_history, ratio_floor), '--', 'LineWidth', 1.2);
yline(1, 'k:'); grid on;
xlabel('ADMM iteration'); ylabel('Residual / threshold');
title(sprintf('Slowest: MC %d, epoch %d, t=%.3f s', slow_mc, slow_epoch, time(slow_epoch)));
legend('Primal', 'Dual', 'Threshold', 'Location', 'best');

nexttile;
distributed_curve = sqrt(mean(reshape(metrics.DistributedPositionError, num_epochs, num_mc).^2, 2));
central_curve = sqrt(mean(reshape(metrics.CentralPositionError, num_epochs, num_mc).^2, 2));
plot(time, distributed_curve, '-', time, central_curve, '--', 'LineWidth', 1.2);
grid on; xlabel('Time (s)'); ylabel('Position RMSE across MC trials (m)');
title('Tracking accuracy');
legend('Network mean', 'Central EKF baseline', 'Location', 'best');
sgtitle('EKF convergence diagnostics | residual display floor = 10^{-12}');
drawnow;
exportgraphics(fig, fullfile(Run.OutputFolder, 'convergence_diagnostics.png'), 'Resolution', 160);
end
