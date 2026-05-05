function [P_cons, J_cons, J_hist, W, L, eps_used, diff_hist] = consensus_covariance(P_in, adj_matrix, varargin)
%CONSENSUS_COVARIANCE  Consensus for covariance matrices (via information matrices).
%
% This function "mimics" typical consensus.m style:
%   - iterative weighted averaging over graph
%   - returns histories + convergence diagnostics
%
% Recommended approach:
%   - Convert covariance P_i to information J_i = inv(P_i)
%   - Run matrix consensus on J_i (element-wise linear averaging but as matrices)
%   - Convert back: P_cons_i = inv(J_cons_i)
%
% Inputs
%   P_in        : covariance per node.
%                Allowed formats:
%                  (a) cell array: P_in{i} is (d x d)
%                  (b) 3D array  : P_in(:,:,i) is (d x d)
%   adj_matrix  : (N x N) adjacency matrix (0/1), assumed undirected
%
% Optional name-value
%   'max_iter'  : default 200
%   'eps'       : consensus stepsize. If empty -> use Metropolis weights (recommended) and ignore eps
%   'tol'       : default 1e-6
%   'use_metropolis' : default true
%   'spd_floor' : eigenvalue floor for SPD projection, default 1e-12
%   'verbose'   : default false
%
% Outputs
%   P_cons    : consensus covariance per node (cell, 1xN)
%   J_cons    : consensus information per node (cell, 1xN)
%   J_hist    : history of J (cell, max_iter+1), each is cell(1,N) of (dxd)
%   W         : weight matrix used (N x N)
%   L         : graph Laplacian (N x N)
%   eps_used  : eps actually used (empty if metropolis weights)
%   diff_hist : (max_iter x 1) max Frobenius difference between nodes' J at each iter
%
% Example
%   [P_cons, J_cons, J_hist, W, L, eps_used, diff_hist] = consensus_covariance(P_nodes, adj);
%
% Note:
%   - To compare with your state consensus, you can do:
%       x_i fused with consensus.m
%       P_i fused with consensus_covariance
%
% Shao-style / minimal dependencies.

% ------------------ parse inputs ------------------
p = inputParser;
addParameter(p, 'max_iter', 200);
addParameter(p, 'eps', []);
addParameter(p, 'tol', 1e-6);
addParameter(p, 'use_metropolis', true);
addParameter(p, 'spd_floor', 1e-12);
addParameter(p, 'verbose', false);
parse(p, varargin{:});

max_iter = p.Results.max_iter;
eps_in   = p.Results.eps;
tol      = p.Results.tol;
use_metropolis = p.Results.use_metropolis;
spd_floor = p.Results.spd_floor;
verbose  = p.Results.verbose;

% ------------------ normalize input format ------------------
[P_cell, d, N] = normalize_cov_input(P_in);

% ------------------ build graph matrices ------------------
A = double(adj_matrix ~= 0);
A = A - diag(diag(A));           % clear diagonal
A = max(A, A');                  % symmetrize

deg = sum(A,2);
L = diag(deg) - A;

% Weight matrix W (Metropolis recommended)
if use_metropolis
    W = metropolis_weights(A);
    eps_used = [];
else
    % Laplacian consensus: J^{k+1} = (I - eps*L) J^k
    if isempty(eps_in)
        % safe default bound for stability: eps < 1/max_degree
        eps_in = 0.9 / max(deg);
    end
    eps_used = eps_in;
    W = eye(N) - eps_in * L;
end

% ------------------ convert to information matrices ------------------
J_cell = cell(1,N);
for i = 1:N
    Pi = 0.5*(P_cell{i} + P_cell{i}.');  % sym
    Pi = project_spd(Pi, spd_floor);
    % Use chol-based inverse for stability
    J_cell{i} = inv_spd(Pi);
end

% Store history (optional but requested "模仿 consensus.m")
J_hist = cell(max_iter+1, 1);
J_hist{1} = J_cell;

diff_hist = zeros(max_iter, 1);

% Precompute weighted neighbors for speed
nbrs = cell(1,N);
wts  = cell(1,N);
for i = 1:N
    idx = find(W(i,:) ~= 0);
    nbrs{i} = idx;
    wts{i}  = W(i,idx);
end

% ------------------ iterate consensus on J ------------------
for k = 1:max_iter
    J_next = cell(1,N);

    % weighted averaging: J_i^{k+1} = sum_j w_ij J_j^k
    for i = 1:N
        idx = nbrs{i};
        wi  = wts{i};

        Ji = zeros(d,d);
        for t = 1:numel(idx)
            j = idx(t);
            Ji = Ji + wi(t) * J_cell{j};
        end

        % sym + SPD projection (important for numerical stability)
        Ji = 0.5*(Ji + Ji.');
        Ji = project_spd(Ji, spd_floor);

        J_next{i} = Ji;
    end

    % diff diagnostic: max ||J_i - J_ref||_F
    ref = J_next{1};
    mx = 0;
    for i = 1:N
        mx = max(mx, norm(J_next{i} - ref, 'fro'));
    end
    diff_hist(k) = mx;

    J_cell = J_next;
    J_hist{k+1} = J_cell;

    if verbose
        fprintf('[consensus_covariance] iter=%d, diff=%.3e\n', k, mx);
    end

    if mx < tol
        diff_hist = diff_hist(1:k);
        J_hist = J_hist(1:k+1);
        break;
    end
end

% ------------------ convert back to covariance ------------------
J_cons = J_cell;
P_cons = cell(1,N);
for i = 1:N
    Ji = 0.5*(J_cons{i} + J_cons{i}.');
    Ji = project_spd(Ji, spd_floor);
    P_cons{i} = inv_spd(Ji);
end

end

% =====================================================================
% Helpers
% =====================================================================

function [P_cell, d, N] = normalize_cov_input(P_in)
    if iscell(P_in)
        N = numel(P_in);
        assert(N >= 1, 'P_in cell is empty.');
        d = size(P_in{1},1);
        P_cell = cell(1,N);
        for i = 1:N
            Pi = P_in{i};
            assert(isequal(size(Pi), [d d]), 'P_in{%d} size mismatch.', i);
            P_cell{i} = Pi;
        end
    else
        % 3D array: d x d x N
        assert(ndims(P_in) == 3, 'P_in must be cell or dxdxN array.');
        d = size(P_in,1);
        assert(size(P_in,2) == d, 'P_in must be square in first two dims.');
        N = size(P_in,3);
        P_cell = cell(1,N);
        for i = 1:N
            P_cell{i} = P_in(:,:,i);
        end
    end
end

function W = metropolis_weights(A)
    % Metropolis-Hastings weights for undirected graph
    N = size(A,1);
    deg = sum(A,2);
    W = zeros(N);
    for i = 1:N
        for j = 1:N
            if i ~= j && A(i,j) ~= 0
                W(i,j) = 1 / (1 + max(deg(i), deg(j)));
            end
        end
    end
    % diagonal to make rows sum to 1
    for i = 1:N
        W(i,i) = 1 - sum(W(i,:));
    end
end

function A_spd = project_spd(A, floorval)
    % Symmetrize + eigenvalue flooring
    A = 0.5*(A + A.');
    [V,D] = eig(A);
    d = diag(D);
    d = max(real(d), floorval);
    A_spd = V * diag(d) * V';
    A_spd = 0.5*(A_spd + A_spd.');
end

function Xinv = inv_spd(X)
    % Stable inverse for SPD matrix using chol
    X = 0.5*(X + X.');
    [R,p] = chol(X);
    if p ~= 0
        % fallback to pseudo-inverse if not SPD (should be rare after projection)
        Xinv = pinv(X);
        return;
    end
    % inv(X) = inv(R) * inv(R)'
    Ri = R \ eye(size(R));
    Xinv = Ri * Ri';
end