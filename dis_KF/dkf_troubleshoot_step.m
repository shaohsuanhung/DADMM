function report = dkf_troubleshoot_step(cfg, x_pred, P_pred, y_bar, idx_set, network_topo, env, models_ut, varargin)
%DKF_TROUBLESHOOT_STEP One-step diagnostics for DKF neighbor update.

p = inputParser;
addParameter(p, 'x_post', []);
addParameter(p, 'P_post', []);
parse(p, varargin{:});
x_post = p.Results.x_post;
P_post = p.Results.P_post;

cfg = local_default_cfg(cfg);

nx = numel(x_pred);
mz = 2;
J  = numel(idx_set);

report = struct();
report.ok = true;
report.warnings = strings(0,1);

% sym / SPD checks
P_pred = 0.5*(P_pred + P_pred.');
report.P_pred_minEig = min(eig(P_pred));
report.P_pred_cond = cond(P_pred);
if report.P_pred_minEig <= cfg.spd_tol
    report = addwarn(report, sprintf("P_pred not SPD: minEig=%.3e", report.P_pred_minEig));
end

if env.PRE_WHITEN
    R_bar = eye(mz*J);
else
    R_bar = kron(eye(J), env.Sigma);
end

% build stacked prediction
h_bar = zeros(mz*J,1);
H_bar = zeros(mz*J,nx);
Hv_norm = zeros(J,1);

for a=1:J
    j = idx_set(a);
    hj = models_ut.MeasureModel(x_pred, j, network_topo, env);
    Hj = models_ut.MeasureModelJacobian(x_pred, j, network_topo, env);

    h_bar((a-1)*mz+1:a*mz) = hj;
    H_bar((a-1)*mz+1:a*mz,:) = Hj;

    Hv_norm(a) = norm(Hj(2,3:4)); % doppler row sensitivity to velocity
end

% whiten if enabled
if env.PRE_WHITEN
    L = env.pre_whit_L;
    LW = kron(eye(J), L);
    y_use = LW*y_bar;
    h_use = LW*h_bar;
    H_use = LW*H_bar;
else
    y_use = y_bar;
    h_use = h_bar;
    H_use = H_bar;
end

innov = y_use - h_use;
S = H_use * P_pred * H_use.' + R_bar;
S = 0.5*(S + S.');

report.innov_norm = norm(innov);
report.S_cond = cond(S);

% NIS
report.NIS = innov.' * (S \ innov);
if report.NIS > cfg.nis_warn
    report = addwarn(report, sprintf("Large NIS=%.2f (warn>%.2f): ordering/whiten/Jacobian issue likely.", report.NIS, cfg.nis_warn));
end

report.doppler_vel_sensitivity_mean = mean(Hv_norm);
report.doppler_vel_sensitivity_min  = min(Hv_norm);
if report.doppler_vel_sensitivity_mean < cfg.vel_sens_warn
    report = addwarn(report, sprintf("Weak doppler->velocity sensitivity: mean=%.3e", report.doppler_vel_sensitivity_mean));
end

% optional FD Jacobian check
if cfg.check_jacobian
    maxErr = 0;
    eps0 = cfg.fd_eps;
    for a=1:J
        j = idx_set(a);
        H_ana = models_ut.MeasureModelJacobian(x_pred, j, network_topo, env);
        z0 = models_ut.MeasureModel(x_pred, j, network_topo, env);
        H_fd = zeros(mz,nx);
        for k=1:nx
            xx = x_pred; xx(k) = xx(k) + eps0;
            zk = models_ut.MeasureModel(xx, j, network_topo, env);
            H_fd(:,k) = (zk - z0)/eps0;
        end
        maxErr = max(maxErr, max(abs(H_fd(:) - H_ana(:))));
    end
    report.jac_fd_maxAbsErr = maxErr;
    if maxErr > cfg.jac_tol
        report = addwarn(report, sprintf("Jacobian FD mismatch: maxAbsErr=%.3e (>%.1e)", maxErr, cfg.jac_tol));
    end
end

% posterior checks (optional)
if ~isempty(x_post) && ~isempty(P_post)
    dx = x_post(:) - x_pred(:);
    report.dx_norm = norm(dx);
    if report.dx_norm > cfg.dx_warn
        report = addwarn(report, sprintf("Large correction step ||dx||=%.3e", report.dx_norm));
    end
    report.trace_P_pred = trace(P_pred);
    report.trace_P_post = trace(P_post);
end

if cfg.verbose
    fprintf("\n[DKF TS] J=%d | NIS=%.2f | doppler vel-sens mean=%.2e | Ppred minEig=%.2e cond=%.2e\n", ...
        J, report.NIS, report.doppler_vel_sensitivity_mean, report.P_pred_minEig, report.P_pred_cond);
    if isfield(report,'jac_fd_maxAbsErr')
        fprintf("  Jacobian FD maxAbsErr=%.2e\n", report.jac_fd_maxAbsErr);
    end
    if ~isempty(report.warnings)
        fprintf("  WARNINGS(%d):\n", numel(report.warnings));
        for i=1:numel(report.warnings)
            fprintf("   - %s\n", report.warnings(i));
        end
    end
end
end

% -------- helpers --------
function cfg = local_default_cfg(cfg)
if nargin<1 || isempty(cfg); cfg = struct(); end
if ~isfield(cfg,'verbose'); cfg.verbose = true; end
if ~isfield(cfg,'check_jacobian'); cfg.check_jacobian = false; end
if ~isfield(cfg,'fd_eps'); cfg.fd_eps = 1e-6; end
if ~isfield(cfg,'jac_tol'); cfg.jac_tol = 1e-3; end
if ~isfield(cfg,'nis_warn'); cfg.nis_warn = 25; end
if ~isfield(cfg,'vel_sens_warn'); cfg.vel_sens_warn = 1e-4; end
if ~isfield(cfg,'dx_warn'); cfg.dx_warn = 1e3; end
if ~isfield(cfg,'spd_tol'); cfg.spd_tol = 1e-12; end
end

function report = addwarn(report, msg)
report.warnings(end+1,1) = string(msg);
end
