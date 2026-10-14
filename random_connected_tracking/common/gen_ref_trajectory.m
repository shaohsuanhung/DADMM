function pos = gen_ref_trajectory(N, bounds, dt, speed, opts)
%GEN_REF_TRAJECTORY Seeded reference curve from the original ADMM-EKF utility.
% pos is N-by-2. bounds is [xmin xmax ymin ymax] or [xmin xmax; ymin ymax].
% The original waypoint jitter, pchip interpolation, arc-length sampling,
% AR(1) lateral wandering and clipping arithmetic are retained.
% Options (original defaults except deterministic seed):
%   seed=42, randomizeShape=true, M=18, denseN=4000,
%   shapeJitterX=0.012, shapeJitterY=0.012,
%   lateralStd=0.25, lateralTau=1.5, longMode="clip" (or "wrap").
% Caller RNG state is restored, including on an early return or error.

    if nargin < 5 || isempty(opts), opts = struct(); end

    validateattributes(N, {'numeric'}, {'scalar','real','finite','integer','>=',2});
    validateattributes(dt, {'numeric'}, {'scalar','real','finite','positive'});
    validateattributes(speed, {'numeric'}, {'scalar','real','finite','nonnegative'});
    if ~isstruct(opts) || ~isscalar(opts)
        error('ReferenceTrajectory:Options', 'opts must be a scalar struct.');
    end
    validateattributes(bounds, {'numeric'}, {'real','finite'});

    [xmin, xmax, ymin, ymax] = parse_bounds(bounds);
    if ~(xmax > xmin && ymax > ymin)
        error("bounds must satisfy xmax > xmin and ymax > ymin.");
    end

    % ---- options ----
    seed           = get_opt(opts, "seed", 42);
    randomizeShape = get_opt(opts, "randomizeShape", true);
    M              = get_opt(opts, "M", 18);
    shapeJitterY   = get_opt(opts, "shapeJitterY", 0.012);
    shapeJitterX   = get_opt(opts, "shapeJitterX", 0.012);
    denseN         = get_opt(opts, "denseN", 4000);

    lateralStd     = get_opt(opts, "lateralStd", 0.25);
    lateralTau     = get_opt(opts, "lateralTau", 1.5);
    longMode       = string(get_opt(opts, "longMode", "clip"));

    validateattributes(seed, {'numeric'}, ...
        {'scalar','real','finite','integer','>=',0,'<=',2^32-1});
    validateattributes(M, {'numeric'}, {'scalar','real','finite','integer','>=',6});
    validateattributes(denseN, {'numeric'}, {'scalar','real','finite','integer','>=',2});
    validateattributes(shapeJitterX, {'numeric'}, {'scalar','real','finite','nonnegative'});
    validateattributes(shapeJitterY, {'numeric'}, {'scalar','real','finite','nonnegative'});
    validateattributes(lateralStd, {'numeric'}, {'scalar','real','finite','nonnegative'});
    validateattributes(lateralTau, {'numeric'}, {'scalar','real','finite'});
    if ~(isscalar(randomizeShape) && ...
            (islogical(randomizeShape) || isnumeric(randomizeShape)) && ...
            isreal(randomizeShape) && any(randomizeShape == [0, 1]))
        error('ReferenceTrajectory:RandomizeShape', 'opts.randomizeShape must be true or false.');
    end
    if ~isscalar(longMode) || ~any(longMode == ["clip", "wrap"])
        error('ReferenceTrajectory:LongMode', 'opts.longMode must be "clip" or "wrap".');
    end

    caller_rng = rng;
    rng_cleanup = onCleanup(@() rng(caller_rng)); %#ok<NASGU>
    rng(seed, 'twister');

    % ---- 1) Build a smooth baseline curve inside bounds ----
    [x_dense, y_dense] = make_base_curve(bounds, randomizeShape, M, shapeJitterX, shapeJitterY, denseN);

    % ---- arclength parameter (meters) ----
    s_dense = [0, cumsum(hypot(diff(x_dense), diff(y_dense)))];
    totalLen = s_dense(end);

    % ---- 2) Decide how long the observed trajectory should be ----
    ds = speed * dt;              % step length per sample
    L  = ds * (N-1);              % expected traveled distance in window

    if speed == 0 || ds == 0
        % stationary: pick a random point along the curve
        s0 = rand() * totalLen;
        x0 = interp1(s_dense, x_dense, s0, "linear");
        y0 = interp1(s_dense, y_dense, s0, "linear");
        pos = repmat([x0, y0], N, 1);
        return;
    end

    % choose a start arclength so that we get a segment of length ~L
    if L <= totalLen
        s0 = rand() * (totalLen - L);
        s_q = s0 + (0:N-1) * ds;
    else
        % requested longer than available curve length
        if longMode == "wrap"
            s0 = rand() * totalLen;
            s_q = s0 + (0:N-1) * ds;
            s_q = mod(s_q, totalLen);
        else
            % "clip": traverse the full curve (speed effectively reduced)
            s_q = linspace(0, totalLen, N);
        end
    end

    % ---- 3) Interpolate base positions at s_q ----
    x_base = interp1(s_dense, x_dense, s_q, "linear");
    y_base = interp1(s_dense, y_dense, s_q, "linear");

    % ---- 4) Add realistic lateral wandering (correlated noise) ----
    % tangent from dense curve -> interpolate at s_q -> normal direction
    tx_dense = gradient(x_dense, s_dense);
    ty_dense = gradient(y_dense, s_dense);

    tx = interp1(s_dense, tx_dense, s_q, "linear");
    ty = interp1(s_dense, ty_dense, s_q, "linear");

    % normal = [-ty, tx] normalized
    nrm = hypot(tx, ty);
    nrm(nrm < 1e-12) = 1;
    nx = -ty ./ nrm;
    ny =  tx ./ nrm;

    % AR(1) correlated lateral offset: rho depends on dt and tau
    tau = max(lateralTau, 1e-3);
    rho = exp(-dt / tau);

    e = zeros(1, N);
    e(1) = randn();
    for k = 2:N
        e(k) = rho * e(k-1) + sqrt(1 - rho^2) * randn();
    end

    % scale to meters; also keep it reasonable relative to bounds
    maxStd = 0.08 * min(xmax - xmin, ymax - ymin);  % safety cap
    sigma = min(lateralStd, maxStd);
    offset = sigma * e;

    x = x_base + offset .* nx;
    y = y_base + offset .* ny;

    % clamp to bounds (simple safety)
    x = min(max(x, xmin), xmax);
    y = min(max(y, ymin), ymax);

    pos = [x(:), y(:)];

% ---------------- helpers ----------------
function v = get_opt(opts, name, default)
    if isfield(opts, name) && ~isempty(opts.(name))
        v = opts.(name);
    else
        v = default;
    end
end

function [xmin, xmax, ymin, ymax] = parse_bounds(bounds)
    % A 2-by-2 matrix also has four elements, so handle its layout first.
    if isequal(size(bounds), [2, 2])
        xmin = bounds(1,1); xmax = bounds(1,2);
        ymin = bounds(2,1); ymax = bounds(2,2);
        return;
    end
    if isvector(bounds) && numel(bounds) == 4
        b = bounds(:).';
        xmin = b(1); xmax = b(2); ymin = b(3); ymax = b(4);
        return;
    end
    error("bounds must be [xmin xmax ymin ymax] or [xmin xmax; ymin ymax].");
end

function [x_dense, y_dense] = make_base_curve(bounds, randomizeShape, M, jitterX, jitterY, denseN)
    [xmin, xmax, ymin, ymax] = parse_bounds(bounds);

    % base reference in normalized space (similar to your figure)
    x_base = [0.00, 0.06, 0.14, 0.22, 0.30, 0.36, 0.42, 0.48, 0.52, 0.55, 0.58, 0.62, 0.68, 0.74, 0.82, 0.90, 0.96, 1.00];
    y_base = [0.00, 0.06, 0.13, 0.20, 0.28, 0.33, 0.38, 0.46, 0.52, 0.60, 0.67, 0.73, 0.78, 0.82, 0.90, 0.95, 0.985, 1.00];

    % make waypoints
    x_wp = linspace(0, 1, M);
    if randomizeShape
        x_wp(2:end-1) = x_wp(2:end-1) + jitterX * randn(1, M-2);
        x_wp = sort(x_wp);
        x_wp(1) = 0; x_wp(end) = 1;
    end

    y_wp = interp1(x_base, y_base, x_wp, "pchip");

    if randomizeShape
        w = sin(pi * x_wp).^1.3; % window -> 0 at ends
        y_wp = y_wp + jitterY * w .* randn(size(y_wp));

        % enforce monotone increasing y (like the reference)
        y_wp = cummax(y_wp);
        y_wp = y_wp - y_wp(1);
        if y_wp(end) <= 0
            y_wp = linspace(0, 1, M);
        else
            y_wp = y_wp / y_wp(end);
        end
        y_wp = min(max(y_wp, 0), 1);
    end

    % dense curve
    u = linspace(0, 1, denseN);
    v = interp1(x_wp, y_wp, u, "pchip");

    % scale to bounds
    x_dense = xmin + u * (xmax - xmin);
    y_dense = ymin + v * (ymax - ymin);
end
end
