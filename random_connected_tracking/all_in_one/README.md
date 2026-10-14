# Batch MAP/MLE with first-order measurements

This folder adapts the original `all_in_one/MAP_tracking.m` into a callable runner. Original source folders are unchanged. MAP and MLE now use a **fixed affine measurement model per batch**. They share the selected truth (`gen_ref_trajectory` or `generateNonlinearTrajectory`) with EKF and keep constant-velocity prediction between batches.

```matlab
cfg = batch_default_config();
Results = run_batch_tracking(cfg, network_topo, "MAP", target);
% Use "MLE" for the measurement-only branch.
```

## Measurement model

For state `x = [p;v]`, radar position `g`, `d = p-g`, `r = norm(d)`, and `u = d/r`, the physical batch measurement is

```text
h(x) = [r; u'*v/lambda]
H(x) = [u',                         zeros(1,2);
        (v-u*(u'*v))'/(lambda*r),    u'/lambda]
```

`BatchADMMUtils.linearize_measurements(anchor, radar_positions, lambda, whitener)` computes `model.anchor`, `model.value = h(anchor)`, and `model.H` once. All four objectives (`logLikelihood`, `MAP`, `logLikelihoodWithConsensus`, `posteriorWithConsensus`) use

```text
h_linear(x) = model.value + model.H * (x-model.anchor)
```

The affine intercept is included. No objective recomputes range, Doppler, or a Jacobian at an optimizer candidate. All pulses from a radar use this same four-state batch model, retaining the original within-batch static-state approximation. The true trajectory and simulated measurements remain nonlinear.

First-batch anchors are `cfg.InitialState`. Later centralized anchors are `F_batch` times the previous centralized estimate; all distributed nodes share `F_batch` times the previous network mean. These models are built **outside** the ADMM loop and remain fixed until the next batch. CA and DA predictions can differ after the first batch, so their linearized objectives can also differ. There is no outer iterated relinearization loop.

The likelihood is now `0.5 * sum(residual' * (Sigma \ residual))`, including correlation in the 2-by-2 range/Doppler covariance. Constants independent of the state are omitted. In addition to linearization, this corrects the legacy variance-product weighting and aligns the Doppler sign with the data generator: **target minus radar**, with the existing batch scale **1/lambda**. EKF's separate scale `2/lambda` is unchanged. The old 1 m range floor was removed; an anchor within `1e-9` m of a radar raises a clear error because the physical derivative is singular there.

With `PreWhiten=true`, the same matrix transforms observations, `h(anchor)`, and `H(anchor)`; the likelihood then uses identity covariance. The state prior covariance is unchanged by measurement whitening. This also removes the old sequential-overwrite error in the MAP whitening calculation.

All four objectives return `[value, gradient]`, and `fmincon` uses the analytic gradient. The entry structure, cells, node loops and solver are retained. This reduces nonlinear function evaluations; it does **not** establish that ADMM requires fewer iterations. The objective helper signatures now take a mandatory fixed `model` instead of radar/count/wavelength arguments; the runner supplies it automatically.

## Inputs, prediction and results

The runner reads `numNodes`, `radar_pos`, `laplacian_matrix`, and `neighbors`. Neighbor lists exclude self; measurement parsing includes self through the Laplacian diagonal. Local batches therefore contain own and neighbor measurements.

`target.target_state` must be `1 x K x 4`, `target.target_position` must be `1 x K x 2`, and `target.time` has `K = SamplesPerBatch*NumBatches` equally spaced samples at `cfg.TimeStep`. The outer runner creates the target once; every Monte Carlo trial shares it. Each saved batch truth is the state at its final sample. Noise uses `Seed + 100000 + mc`; caller RNG state is restored.

`batch_default_config.m` retains 1 trial, 30 batches, 64 samples per batch, 0.01 s per sample, and at most 1000 ADMM iterations. A shorter profile changes the Doppler variance when it changes `SamplesPerBatch`; the validation entry therefore keeps 64 samples for its tracking checks.

`F_batch = [I,dt_batch*I;0,I]`, where `dt_batch = SamplesPerBatch*TimeStep`. MAP uses prediction as the prior mean, linearization anchor and warm start. MLE uses it as the anchor and warm start, without a prior cost. The first MAP prior mean remains `cfg.PriorMean`, which can differ from `cfg.InitialState`.

The fixed `state_cov = diag([range_var,range_var,doppler_var,doppler_var])` remains a legacy proxy. Later priors use `F_batch*state_cov*F_batch' + Q_batch`; there is no posterior-covariance recursion. `ProcessAccelerationVariance` retains its original value `1e-2`.

Results preserve the original cells, including `primal_residauls` and `dual_residauls`. `estimations_DA{mc,k}` stores state-by-node-by-iteration history; `consensus_estimates` is its final node mean. `estimations_CA` is the centralized estimate. `converged` reports the implemented primal-only stopping test; solver flags describe the final local solves.

`linearization_CA{mc,k}` saves the full centralized model. `linearization_DA{mc,k}{n}` saves each node's fixed model, including its common anchor. `warm_starts_CA/DA`, `predicted_means_CA/DA`, prior covariance metadata and `F_batch` make the anchor policy inspectable. Models are in whitened coordinates when whitening is enabled; `Results.constant.pre_whit_L` records that transformation.

## Retained limitations

- The original prior coefficient is `1/sqrt((2*pi)^4*det(P))` times the quadratic, averaged over the supplied priors. It is already quadratic and was retained; it is **not** standard Gaussian negative-log-prior scaling.
- Own-plus-neighbor measurements repeat sensors with degree-dependent multiplicity. Their sum is not generally the centralized once-per-sensor objective.
- The original consensus penalty is `sum((c/2 .* delta).^2)`, whereas standard ADMM would use `0.5*delta'*diag(c)*delta`. The old dual update, penalty adaptation, residual definitions and primal-only stopping rule are retained. Passing that stopping rule does not certify stationarity or convergence of standard ADMM.
- Linearization is a local approximation. Poor anchors, large motion within a batch, or an inaccurate CV prediction can degrade tracking even when local quadratic solves succeed.
- `node_wise_cov` is dispersion across nodes, not posterior covariance. The legacy FIM/bound helper remains nonlinear, unaudited and disabled by default (`ComputeLegacyBound=false`); it is not a validated bound for the new affine estimator.
- Asynchronous ADMM is not implemented. The separate EKF entry has its own first-order model with Doppler scale 2/lambda; these batch objectives retain 1/lambda.

Run `check_batch_linearization` from the project root for independent Jacobian, Taylor-error, correlated-noise, whitening, four-objective gradient and quadratic-minimum checks. `run_batch_linearization_test` additionally runs both whitening branches, checks CV anchors, saves bounded MAP/MLE tracking runs, and benchmarks a frozen local solve. Its runtime comparison uses the same corrected measurement sign/weights on both sides, not the old inconsistent likelihood.

Requires MATLAB, Optimization Toolbox (`fmincon`) and Statistics and Machine Learning Toolbox (`mvnrnd`).
