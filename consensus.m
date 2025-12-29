% Consensus algorithm try out
% clc;clear;close all;
% data = [-2000.08688105232	-1419.45117288024	-771.572368219108	1740.82434585811	3417.59330879102	3991.34298092154	3421.49243740948	2462.72210240796	20.4786521774650	-1470.75682428627
% 1021.34255350041	-783.618692796857	-1679.08511856733	-2015.68424635879	-766.167561231156	1012.86810895446	2768.50742556249	3539.04280156682	3860.90607452168	2751.38345549746
% -1.56448869139050	14.5273043084207	-1171.54822046546	-289.362730198092	-16.6875438620006	-12.3473086507043	-10.4472001237986	765.674159105423	-98.5119165611232	-68.5123771311562
% 38.8534452683665	-38.6403915844441	514.933502241864	-226.060385217697	2.78239598609237	6.50247245791244	9.08974282629487	-515.389205827116	14.5976481087643	-15.0543864397544]';
% 
% A = [0	1	0	0	0	0	0	0	0	1
% 1	0	1	0	0	0	0	0	0	0
% 0	1	0	1	0	0	0	0	0	0
% 0	0	1	0	1	0	0	0	0	0
% 0	0	0	1	0	1	0	0	0	0
% 0	0	0	0	1	0	1	0	0	0
% 0	0	0	0	0	1	0	1	0	0
% 0	0	0	0	0	0	1	0	1	0
% 0	0	0	0	0	0	0	1	0	1
% 1	0	0	0	0	0	0	0	1	0];
% 
% 
% [x, x_hist, P, L, eps_used] = consensus(data, A, 'Verbose', true,'Tol', 1e-6);


function [x, x_hist, P, L, eps_used] = consensus(x0, A, varargin)
%CONSENSUS_LAPLACIAN  Discrete-time consensus: x_{k+1} = P x_k, P = I - eps*L.
%
% Inputs:
%   x0 : (n x d) initial node values
%   A  : (n x n) adjacency matrix, A_ij >= 0
%        For undirected graph, use symmetric A (A = A').
%
% Optional name-value pairs:
%   'Epsilon'   : step size eps (default: 0.9 / dmax)
%   'MaxIter'   : maximum iterations (default: 5000)
%   'Tol'       : stopping tolerance on ||x_{k+1} - x_k||_inf (default: 1e-8)
%   'StoreHist' : true/false to store history (default: true)
%   'Verbose'   : true/false (default: false)
%
% Outputs:
%   x        : final consensus state
%   x_hist   : (n x (T+1)) history (if StoreHist=false, returns [])
%   P, L     : Perron/mixing matrix and Laplacian
%   eps_used : epsilon actually used

% ---- parse inputs ----
p = inputParser;
p.addParameter('Epsilon', [], @(v) isempty(v) || (isscalar(v) && v >= 0));
p.addParameter('MaxIter', 5000, @(v) isscalar(v) && v > 0);
p.addParameter('Tol', 1e-5, @(v) isscalar(v) && v > 0);
p.addParameter('StoreHist', true, @(v) islogical(v) && isscalar(v));
p.addParameter('Verbose', false, @(v) islogical(v) && isscalar(v));
p.parse(varargin{:});
opts = p.Results;

% x0 = x0(:);
n = size(x0,1);
m = size(x0,2);

if ~ismatrix(A) || size(A,1) ~= n || size(A,2) ~= n
    error('A must be an (n x n) matrix matching length(x0).');
end
if any(A(:) < 0)
    error('Adjacency matrix A must be nonnegative.');
end

% ---- build Laplacian ----
d = sum(A, 2);          % degree vector
D = diag(d);
L = D - A;

% ---- choose epsilon ----
dmax = max(d);
if isempty(opts.Epsilon)
    if dmax == 0
        warning('All degrees are zero (no edges). Consensus cannot mix nodes.');
        eps_used = 0;
    else
        eps_used = 0.9 / dmax;  % ensures P_ii = 1 - eps*d_i >= 0 and typically P_ii > 0
    end
else
    eps_used = opts.Epsilon;
end

% ---- build Perron/mixing matrix ----
P = eye(n) - eps_used * L;

% (Optional) basic checks
row_sums = sum(P,2);
if norm(row_sums - 1, inf) > 1e-10
    warning('Row sums of P are not ~1. Check your A/L/epsilon settings.');
end
if any(P(:) < -1e-12)
    warning('P has negative entries. If you need Perron/Markov properties, reduce epsilon.');
end

% ---- iterate ----
x = x0;

if opts.StoreHist
    x_hist = zeros(n,m,opts.MaxIter + 1);
    x_hist(:,:,1) = x0;
else
    x_hist = [];
end

for k = 1:opts.MaxIter
    x_next = P * x;
    diff_inf = norm(x_next - x, inf);

    x = x_next;

    if opts.StoreHist
        x_hist(:,:,k+1) = x;
    end

    if opts.Verbose && (mod(k,100) == 0 || diff_inf < opts.Tol)
        fprintf('Iter %d: ||x_{k+1}-x_{k}||_inf = %.3e\n', k, diff_inf);
    end

    if diff_inf < opts.Tol
        if opts.StoreHist
            x_hist = x_hist(:,1:k+1);
        end
        return;
    end
end

if opts.StoreHist
    x_hist = x_hist(:,1:opts.MaxIter+1);
end

end



