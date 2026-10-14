# Circle / ring topology selector — 2026-10-13

Added `cfg.Graph.Mode="ring"` alongside `"random_connected"`. `run_example.m` now selects `Radar.Mode="circle"` and `Graph.Mode="ring"`. With N=10, edges form 1–2–...–10–1: every node has degree two and connects to its two nearest neighbors on the evenly spaced circle. The shared generator supplies the same topology to MAP, MLE and EKF. It keeps the original incidence/Laplacian/edge-list formats.

The generic `tracking_config` default and old configurations with no Graph.Mode continue to use connected random G(N,p). Its original seeded rejection-sampling operations remain unchanged. Ring edges depend on node index, not distance; random positions plus ring do not imply geometric nearest neighbors. Ring needs at least three nodes and ignores graph Seed, EdgeProbability and MaxAttempts. Plot titles and summaries now identify the selected graph model; old saved topologies without graph_mode keep their random-graph labels.

Validation status: **MATLAB R2025b runtime validation passed on the retry at approximately 22:33 on 2026-10-13.** The previous attempt had stopped during startup with Licensing Error 15 (-15.2); the adjacent startup-error file is retained as historical context.

- `check_graph` passed: existing connected random graphs, reproducibility, RNG isolation and failure handling.
- `check_ring_graph` passed for N=3/4/10/16: exact circle nearest neighbors, degree two, N edges, connectivity, incidence/Laplacian consistency, cycle spectrum, metadata, legacy random sampling, ignored random-only ring options and invalid-input errors.
- The actual `run_example.m` was checked without running its full profile: circle + ring, N=10, 10 edges, every degree two. Lambda2 is approximately 0.381966.
- MAP/MLE/EKF smoke runs all completed; their distributed estimates were finite, RNG state was restored, saved MAT results matched returned results, and graph summaries/plots identified the ring correctly. No recorded local solver failures (batch logging covers the final iteration; EKF covers all iterations).
- Plotting an older saved topology without `graph_mode` passed. The new ring radar-layout image was visually inspected.

| Method | Updates | ADMM cap | Stop criterion met | Tracking compute time |
| --- | ---: | ---: | ---: | ---: |
| MAP | 1 | 5 | 0/1 | 3.16 s |
| MLE | 1 | 5 | 0/1 | 1.00 s |
| EKF | 2 | 5 | 0/2 | 1.27 s |

These short runs verify execution and integration, not convergence or tracking accuracy. They used the default reference truth from `tracking_config("smoke")`; the user-facing nonlinear selection was read to check topology settings, not run as a full experiment. Results are in `results/ring_topology_check_20261013/`, including `validation.mat`, `matlab_validation.log`, the three timestamped smoke folders and `legacy_replot/`.

To repeat the topology checks from the project folder:

```matlab
check_graph;
check_ring_graph;
```

`results/ring_topology_check_20261013/run_validation.m` additionally exercises the actual run_example settings, all three methods with small smoke caps, saving, plotting and legacy result plotting. The adjacent startup error record documents the earlier blocked attempt; the retry passed. No convergence claims are made.

Estimator formulas, process-noise settings, truth trajectory, measurement model and existing numerical result files were not changed by this update.

---

# Trajectory selector update — 2026-10-13

The shared truth now supports `Type="reference"` (unchanged default) and `Type="nonlinear"`. `run_example.m` offers `TRAJECTORY="original"`, `"smooth"`, `"another"`, or `"nonlinear"`. Nonlinear restores the original v5 sinusoidal speed/heading example with optional endpoint correction; CV predictions and the affine measurement objectives are unchanged.

The adapted nonlinear generator retains Euler position integration, smoothed optional noise and the analytic cubic-smoothstep velocity correction. It uses samples at `0:dt:Duration` and restores the caller RNG. This intentionally fixes the original utility's first sample at `dt`, which shifted the specified start when endpoint correction was enabled. Nonlinear truth is not clipped to the reference Bounds. The default 50 m/s nominal speed can produce a much wider path than the reference curve.

Validated with MATLAB R2025b:

- `check_trajectory` passed unchanged.
- `check_trajectory_switch` passed: exact default reference numerics; old config without Type; nonlinear start/end points; independent speed/heading, Euler and endpoint-correction formulas; deterministic and noisy prefixes; seeded repeatability/RNG isolation; both legacy Doppler scales; invalid-input guards.
- All four selectors in the actual `run_example.m` script were evaluated and checked without launching a full simulation.
- Short nonlinear MAP, MLE and EKF runs each completed two updates with finite states. Their complete reference states were identical; CV prediction recursion, result save/load, RNG restoration and exported plots passed.
- No recorded local solver failures. Batch logging covers the final ADMM iteration; EKF covers all iterations.

| Method | ADMM cap | Stopping rule met | Tracking compute time |
| --- | ---: | ---: | ---: |
| MAP | 3 | 0/2 | 1.53 s |
| MLE | 3 | 0/2 | 0.71 s |
| EKF | 5 | 0/2 | 1.00 s |

These are integration checks, not convergence or full-trajectory accuracy evidence. All three runs deliberately use smoke caps. Defaults, graph, radar deployment, noise, tolerances and estimator formulas were not changed.

Artifacts: `results/trajectory_switch_check_20261013_200138/` includes the MATLAB log, validation.mat, four-path preview, and saved runs. Tests ran from the writable staging checkout; MAT files retain those original execution paths. Tested source files and results were copied to the active DADMM folder after source-hash checks.

---

# EKF first-order measurement correction — 2026-10-13

The EKF entry now solves a **fixed affine MAP/MLE measurement objective** inside ADMM. The common DA predicted mean is the shared anchor for every node in a measurement update. The model `h(anchor)+H(anchor)*(x-anchor)` is built before the ADMM loop and held fixed. `TrackingModels.MeasureModelAndJacobian` remains the physical nonlinear function used to construct that model, the truth generator and the optional assumed-CV bound remain nonlinear, and CV dynamics are unchanged.

`TrackingADMMUtils` supplies analytic gradients for all four objectives. It retains the EKF convention `+2/lambda`, raw observations, full covariance and correct complete-residual whitening. Its Gaussian prior and `0.5*delta'*diag(c)*delta` penalty are preserved. Each physical measurement occurs once; each node contributes 1/N of the common prior. Centralized EKF was already a first-order update and now shares the model builder. DA posterior covariance uses the **same frozen Jacobians as the objective**, replacing the former re-evaluation at the estimated state.

New results record `MeasurementModel="first_order_affine"`, both DA/CA models, and the DA common prior. The frozen diagnostic now compares the saved affine objective against an independent direct normal-equations solution; it explicitly rejects historical nonlinear runs before replay or writing. Historical MAT files remain untouched and can still be replotted.

Validation ran in MATLAB R2025b in the workspace staging copy. Source and results were published after active-file hash checks; saved paths record the original execution location. Reproduce with `Report = run_ekf_linearization_test();` or run `check_ekf_linearization` for formulas only.

Results: [ekf_linearization_check_20261013_185248_506](ekf_linearization_check_20261013_185248_506/).

## Formula and integration checks

- Physical Jacobian relative finite-difference error: **1.96e-10**. Maximum gradient error across the four raw/whitened objectives: **5.8e-11**.
- True measurement generator exactly matches the model value at the anchor, including the **2/lambda** factor. Taylor remainder shrinks by factors **3.9815 / 3.9906** when perturbations halve.
- Correlated-noise whitening, coefficient checks and fixed Hessian checks passed.
- Sum of self-only likelihoods and 1/N prior shares matches the global MAP cost/gradient with relative error **1.15e-16**.
- Centralized EKF equals an independent same-prior/same-anchor affine MAP solution: state relative difference **2.83e-15**, covariance relative difference **5.84e-17**. Joseph/raw/whitened covariance checks passed.
- MAP/MLE and whitening on/off all execute; saved common anchors match the predicted means; the caller RNG is restored. Over all recursive updates, covariance matches the frozen objective's information inverse within **2.77e-15** relative error.

## Recursive run and optimization accuracy

Ten random-disk radars, existing connected graph (12 edges, lambda2=0.407004424263), SNR40, 64 Doppler noise samples, Q=0, **16 updates x 2 trials**, dt=0.01: **0 through 0.15 s**. This is an implementation check, not the full 30-second trajectory. The user's defaults remain `ADMMAbsTolerance=ADMMRelTolerance=1e-3`, cap2000, with the existing penalty/adaptation.

| Diagnostic | Result |
| --- | --- |
| Both primal and dual tests passed | 32 / 32 |
| ADMM iterations min / median / max | 26 / 53 / 124 |
| Local solver failures across all iterations | 0 |
| Tracking computation time (excludes plotting/diagnostics) | 51.471 s |
| Pooled network-mean position / velocity RMSE | 0.04187 m / 1.67015 m/s |

Passing the residual test does **not** imply a negligible state error relative to the exact quadratic optimum. At MC1/update1, the default tolerances stop after 113 iterations with position gap **0.000579 m**, velocity gap **0.76950 m/s**, and relative objective gap **0.01718** to the same-prior/same-anchor affine MAP reference. MC2/update1 has velocity gap **1.44809 m/s**. These are optimization gaps, not errors to the true trajectory.

A separate MC1/update1 replay at abs/rel **1e-5**, cap3000, passes both residual tests after **1153** iterations with no solver failure: position gap **1.78e-6 m**, velocity gap **0.008912 m/s**, final primal/threshold **0.028604** and dual/threshold **0.998267**. This stricter setting is confined to the validation script; the user's default configuration was not changed.

The saved report contains unit checks, per-update covariance/optimization gaps, residual histories, frozen comparison and strict replay. The diagnostic plot was visually inspected. No matched before/after speed experiment was run, so these timings do not establish a linearization speedup. The earlier nonlinear convergence experiment below describes a different estimator version.

---

# Batch first-order linearization — 2026-10-13

`BatchADMMUtils` now uses fixed affine measurement models in all four MAP/MLE objective functions. Each model is created before its batch solve; all DA nodes share an anchor. The first anchor is `InitialState`, followed by the estimator's CV prediction. Analytic gradients are supplied to `fmincon`.

Alongside linearization, measurement likelihood corrections align the Doppler sign with the generator (+1/lambda), use the full inverse covariance, transform both model components correctly when whitening, and replace the old 1 m range floor with a zero-range anchor error. Measurement whitening no longer changes the state prior covariance. Legacy prior coefficients, own-plus-neighbor allocation, ADMM penalties/updates and primal-only stopping remain explicitly documented in `all_in_one/README.md`. The separate EKF implementation and its current tolerance settings were not edited.

All numerical validation ran in **MATLAB R2025b**. Source was tested in the workspace staging copy, then published after checking that active files had not changed. Saved MAT files retain the original execution/output paths. Results are also copied beside this document:

[batch_linearization_check_20261013_183455_979](batch_linearization_check_20261013_183455_979/)

Reproduce with `Report = run_batch_linearization_test();` from the project root. For the formulas only, run `check_batch_linearization`.

## Mathematical checks

- Physical Jacobian finite-difference relative error: approximately **2.0e-10**.
- Maximum finite-difference gradient relative error across all four objectives: **9.5e-11**.
- Physical / whitened likelihood and gradient relative difference: **2.5e-16**, including correlated noise.
- Halving perturbations reduces first-order Taylor error by factors **3.9825** and **3.9911**.
- Anchor exactness, a range below 1 m, rejection of singular anchors, retained prior/penalty coefficients and a fixed Hessian all passed.
- The frozen affine local `fmincon` solution differs from independently assembled normal equations by **1.49e-12** in state norm, with relative stationarity **3.02e-14** and relative objective gap **7.63e-16**.
- MAP and MLE both execute with whitening on/off. Two-batch integration checks verify common DA anchors, subsequent CV prediction, finite states, RNG restoration, unchanged physical prior covariance, and matching raw/whitened first-batch centralized solutions.

## Bounded tracking check

Current 10-node random disk deployment, connected graph with 12 edges and lambda2=0.407004424263; one Monte Carlo trial, three batches, **64 samples per batch**, SNR50, dt=0.01, ADMM cap300, original primal tolerance1e-3. This covers samples from 0 through **1.91 s**, with batch endpoints 0.63, 1.27 and 1.91 s.

| Method | ADMM iterations per batch | Primal stops | Tracking compute time |
| --- | --- | --- | --- |
| MAP | 183 / 212 / 196 | 3 / 3 | 20.689 s |
| MLE | 182 / 213 / 211 | 3 / 3 | 21.418 s |

All final local solver flags were positive. The saved plots were generated successfully and the MAP overview was visually inspected. Each method's MAT file preserves the fixed models and full residual histories. **These are passes of the retained primal-only rule, not proof of standard ADMM convergence or accurate tracking.** There was no matched nonlinear tracking baseline in this experiment.

## Frozen local-solve benchmark

One fixed local problem, three repetitions after warmup, both models using the same corrected sign/covariance, analytic gradients, prior, penalty and solver tolerances. The affine and nonlinear costs agree in value and gradient at the anchor but are different approximations away from it.

| Local measurement model | Median fmincon iterations | Function evaluations | Median solve time |
| --- | --- | --- | --- |
| Nonlinear | 21 | 31 | 15.324 ms |
| Fixed affine | 13 | 21 | 21.212 ms |

The affine solve used fewer optimizer iterations/evaluations, but **this short timing test did not show a wall-clock speedup**. It does not compare ADMM rounds or communication. Raw trials, solver settings and inputs are saved in `validation.mat`; `local_solver_benchmark.csv` and `tracking_checks.csv` contain the summaries. A direct quadratic solve is not introduced by this change.

---

# Longer convergence experiment — 2026-10-13

**The current random-layout EKF entry passed its ADMM stopping test in all 256 updates.** This is an empirical result for this fixed deployment and finite observation window. The entry retains a nonlinear consensus MAP correction with a centralized EKF tracking baseline; MAP/MLE batch methods were not rerun in this longer experiment.

Run: `EKF_convergence_long_20261013_182045_631`, produced in MATLAB R2025b by `run_convergence_test.m`.

- Same 10-node random disk deployment (radius 30 m, position seed 20261014), connected graph (seed 20261013, 12 edges, lambda2=0.407004424263), gen_ref_trajectory truth and CV model.
- 128 updates per trial, two measurement-noise trials (seeds 44 and 45), dt=0.01: observed times **0 through 1.27 seconds**. The full 30-second reference was not tracked.
- ADMM cap increased to 5000; absolute and relative tolerances remained 1e-5. Initial penalties [100 100 15 15], rho adaptation, Q=0, SNR=40 dB, Doppler sample count=64 and estimator formulas were unchanged.
- The original initialization remains centered on the first true state; this is not an initial-acquisition test.

| Diagnostic | Result |
| --- | --- |
| Both primal and dual residual tests passed | 256 / 256 |
| Iterations: minimum / median / maximum | 127 / 278.5 / 739 |
| Updates reaching the 5000-iteration cap | 0 |
| Maximum final primal / threshold | 0.999962384 |
| Maximum final dual / threshold | 0.999957131 |
| Nonpositive local fmincon exit flags | 0 |
| Maximum node position RMS disagreement | 0.00151898616 m |
| Maximum node velocity RMS disagreement | 0.00103678294 m/s |
| Simulation runtime, excluding plots/export | 2819.74 s (about 47 minutes) |

The existing original-profile cap of 2000 already exceeds the largest observed requirement (739) in this test; the full 5000 iterations were never needed. The earlier cap-5 smoke test was too short to judge convergence.

Finite-window RMSE, pooling squared 2-D errors across both trials and all sampled times:

| Estimator | Position | Velocity |
| --- | --- | --- |
| Mean of distributed node estimates | 0.056496956 m | 2.09523857 m/s |
| Centralized EKF baseline | 0.0564198296 m | 2.09528329 m/s |

The similar errors do not make the centralized EKF an identical optimization problem. ADMM stopping and tracking accuracy are separate: velocity errors remain even after consensus converges. Q=0 and the time-varying truth velocity, as well as the measurement-noise model, are relevant to subsequent tracking-quality experiments; this run did not isolate their individual effects.

The saved `convergence_metrics.csv` contains every trial/update; `convergence_summary.txt` reports the exact statistics; `convergence_diagnostics.png` includes all final normalized residuals and the slowest update (MC2, epoch5, 739 iterations). All 4608 numeric table entries were finite, and the recomputed stopping flags matched the logged flags. Both diagnostic and tracking figures were visually inspected.

An additional frozen-problem comparison used the completed two-update pilot in `EKF_convergence_pilot_20261013_173103_898`. Its first scan reused the exact same prior and measurements for a 100-iteration replay and a centralized numerical nonlinear MAP reference:

| First-scan solve | Iterations | Primal / threshold | Dual / threshold | Relative objective gap to numerical MAP reference |
| --- | --- | --- | --- | --- |
| ADMM, cap 100 | 100 | 3.6173 | 30.061 | 3.3247e-4 |
| ADMM, converged | 559 | 0.998882 | 0.945946 | 7.5273e-7 |

The centralized reference's objective gradient norm was 4.04e-8. Its constant-free objective matched the existing MAP utility; it is a local numerical reference, not a proof of global optimality. Pilot and longer-run noise arrays are generated for different sample counts, so the controlled cap comparison explicitly reuses the pilot's saved arrays rather than comparing different runs' first scans.

Reproduction: run `run_convergence_test.m`. Summarize saved EKF results with `summarize_ekf_convergence(Run)`; run the frozen first-scan diagnostic with `check_frozen_map_solution(Run)`. These new helpers do not change estimator or trajectory formulas.

Final source comparison found a concurrent change in `ADMM_EKF/ekf_default_config.m`: both default tolerances are now 1e-3. That current edit was preserved. The completed Run's captured configuration and diagnostics confirm both tolerances were 1e-5 throughout this experiment. The reproduction entry now explicitly pins 1e-5 so rerunning it does not silently inherit the relaxed defaults. All other inspected estimator, truth and layout source/config files matched the test-start copies.

---

# Earlier random radar position validation — 2026-10-13

The active default is a shared 10-node layout sampled uniformly in disk area, centered at [0,0] with radius 30 m and position seed 20261014. The connected G(N,p) graph keeps seed 20261013, 12 edges and lambda2 approximately 0.407. Position and graph random streams are independent. The user's `run_example.m` selections (`EKF`, `original`) were preserved; validation explicitly used a smaller smoke configuration.

Validated in MATLAB R2025b from the installed DADMM folder:

- Position checks: exact seeded repeatability, seed sensitivity, disk bounds, center/radius transformations, evenly spaced circle mode and caller RNG isolation. A fixed 50,000-point sample gave mean(r^2/R^2)=0.49652, consistent with area-uniform sampling (expected 0.5).
- Graph checks: N=2/7/10/16, connectivity and Laplacian/incidence invariants, unchanged edges when positions change, unchanged positions when graph seed/probability change, complete graph and bounded failure. Integer-typed deployment centers are also checked.
- Three-method integration: the same 10-node coordinates, graph and truth reference across MAP/MLE/EKF; two Monte Carlo trials and two tracking steps per method; finite estimates, CV recursion, save/load equality, radar coordinate CSV roundtrip and all four exported figures. The optional assumed-CV PCRLB reference branch was exercised.
- The default layout's minimum distance to the sampled full reference trajectory was 2.195639 m. This is a diagnostic of this seed, not a placement constraint or a guarantee for other seeds.
- Eight original source hashes and all 13 estimator/trajectory source files remained unchanged. Actual simulation and plotting used MATLAB; Python was used only for file synchronization and source hash checks.

| Method | Stopping rule met | Simulation time | Result folder |
| --- | --- | --- | --- |
| MAP | 0/4 steps | 2.45 s | `MAP_smoke_20261013_172447_552` |
| MLE | 0/4 steps | 1.01 s | `MLE_smoke_20261013_172459_883` |
| EKF | 0/4 steps | 1.72 s | `EKF_smoke_20261013_172506_993` |

Smoke iteration caps were 3 for batch ADMM and 5 for EKF ADMM; these checks do not establish convergence or tracking accuracy. No full original-profile simulation was run. Results include `radar_positions.csv` (x,y in meters, row n is node n), `radar_layout.png`, the existing three plots and MAT/config metadata. Plots were regenerated from saved numerical results after adding full-reference margins.

The historical sections below used circle deployments and their earlier trajectory versions. Their numerical results do not describe this new default layout.

---

# Historical: gen_ref_trajectory correction — 2026-10-13

The active shared truth uses the user's requested `ADMM-EKF/ADMM_utils.m::gen_ref_trajectory`, extracted to `common/gen_ref_trajectory.m`. It replaces the previously selected sinusoidal speed/heading trajectory. Original estimator files and CV prediction settings were unchanged by this correction.

Validated with MATLAB R2025b:

- Exact position-array equality against the original function for seeds 42 and 7, sample counts 64 and 3001, original vector bounds `[-30 20 -30 25]`, dt=0.01 and nominal speed=50.
- Shared-reference checks: bounded curved path, varying velocity, full-reference `gradient` velocity, identical position/velocity prefixes for 1/2/8/64 samples, seed repeatability, caller RNG restoration, and vector/2-by-2 bounds equivalence. Both measurement generators read the per-sample truth velocity.
- Three-method smoke test: irregular 7-node graph, two Monte Carlo trials and two steps, finite estimates, CV prediction recursion, shared reference, save/load equality, figure export and optional assumed-CV PCRLB reference label.
- Installed-folder quick runs: 10 nodes, identical complete truth reference across all methods, finite estimates, successful MAT reload and three exported figures per method. Each run retained the 100-iteration ADMM cap.
- All eight recorded original source hashes remained unchanged, including the original `ADMM-EKF/ADMM_utils.m`. Installed source matched the reviewed staging version.

| Method | Stopping rule met | Simulation time | Result folder |
| --- | --- | --- | --- |
| MAP | 0/3 steps | 11.64 s | `MAP_quick_20261013_170722_850` |
| MLE | 0/3 steps | 10.48 s | `MLE_quick_20261013_170819_064` |
| EKF | 0/8 steps | 27.52 s | `EKF_quick_20261013_170854_257` |

These are short execution checks; no step met its stopping criterion at the quick-run cap. Full 30-second estimator runs were not executed. `truth_trajectory.png` shows both the complete reference and the prefix used by the run. The 30-second reference and original `longMode="clip"` reduce effective speed when the nominal distance exceeds the curve length. Gradient velocity includes the original lateral wandering.

---

# Historical validation: superseded trajectory selection

The section below records the earlier `generateNonlinearTrajectory` version and its saved outputs. Those old trajectory tests and figures do not describe the active generator.

# Nonlinear truth / Constant Velocity update — 2026-10-13

The current version shares the adapted original ADMM-EKF nonlinear trajectory across all methods and measurement-noise trials. The full reference lasts 30 seconds, independently of quick/smoke execution length. Samples at overlapping times match when trajectory configuration and sampling interval match. All Doppler generators now read the per-sample truth velocity; their previously documented Doppler scaling/sign differences remain.

Validated with MATLAB R2025b:

- `check_trajectory`: exact start/end points, noncollinear full path, varying true velocity, fixed prefixes for 1/2/8/64 samples, seeded stochastic repeatability, RNG isolation, source Euler integration and analytic endpoint velocity correction, and per-sample Doppler in both generators.
- `smoke_test`: all three estimators, 7-node irregular graph, two MC trials and two updates each, same complete truth reference, direct checks of CV predictions against previous estimates, finite states, save/load equality, and all three exported figures. Optional assumed-CV PCRLB was exercised and its nonmatching-reference label checked.
- Installed-folder quick runs: 10 nodes, finite estimates, identical reference trajectory across MAP/MLE/EKF, and PCRLB disabled by default. The iteration cap remains 100; no quick-run step met its stopping criterion.
- Eight original source hashes, including `ADMM-EKF/generateNonlinearTrajectory.m`, remained unchanged.

| Method | Stopping rule met | Simulation time | New result folder |
| --- | --- | --- | --- |
| MAP | 0/3 steps | 11.09 s | `MAP_quick_20261013_165225_948` |
| MLE | 0/3 steps | 8.56 s | `MLE_quick_20261013_165242_244` |
| EKF | 0/8 steps | 27.46 s | `EKF_quick_20261013_165313_733` |

The new truth/CV tests do not establish tracking accuracy or convergence. The full 30-second estimator runs were not executed. D-MAP uses CV prior-mean prediction and the existing fixed covariance proxy; D-MLE uses CV only for initialization; D-EKF uses CV mean/covariance recursion. The old likelihood, Doppler convention, and neighborhood-weighting limitations are documented in README.

---

# Earlier topology-only validation (historical; before this update)

# Validation — 2026-10-13

Executed with MATLAB R2025b, Optimization Toolbox, and Statistics and Machine Learning Toolbox.

## Graph checks

`check_graph` passed for N = 2, 7, 10, and 16: symmetry, no self-loops, connectivity, positive lambda2, incidence/Laplacian consistency, repeatable edges, and preservation of the caller RNG. The complete-graph case and bounded failure for an extremely sparse graph also passed.

## Integration checks

Three 10-node smoke runs passed with finite estimates and successful figure export. The subsequent `smoke_test` passed for all three methods on a 7-node graph with degree vector `[2 2 2 2 5 4 1]`, two Monte Carlo trials and two tracking steps. It checked finite outputs, result save/load equality, RNG restoration, graph reuse across methods, and presence of both figures. The EKF test also exercised the optional Bayesian PCRLB branch with 20 state samples.

The smoke iteration limits were deliberately small; all 4 updates per method reached the cap rather than satisfying their stopping rule. These are execution checks, not convergence evidence.

## Runs from the final installed folder

The final `DADMM/random_connected_tracking` folder was tested directly with each method's `quick` configuration. The graph has N=10, 12 edges, seed 20261013, p=0.35 and lambda2 approximately 0.407. Adjacency is identical across all three methods. Saved MAT contents were reloaded and matched the returned Results.

| Method | Tracking steps | ADMM cap per step | Steps satisfying stopping rule | Simulation time |
| --- | ---: | ---: | ---: | ---: |
| MAP | 3 | 100 | 0/3 | 9.74 s |
| MLE | 3 | 100 | 0/3 | 7.63 s |
| EKF entry | 8 | 100 | 0/8 | 25.85 s |

Times exclude plotting and file export. All stored estimates were finite. Logged local fmincon nonpositive exit-flag counts were zero; the batch logger inspects only the final local solves per step, while the EKF logger counts all local solves. Neither fact establishes ADMM convergence or global optimality.

Saved runs:

- `MAP_quick_20261013_163119_709/`
- `MLE_quick_20261013_163133_478/`
- `EKF_quick_20261013_163202_119/`

Each contains `results.mat`, `summary.txt`, adjacency/edge CSV files, `overview.png`, and `convergence.png`. These are short diagnostic runs, not paper-ready performance results. Full `original` runs and Monte Carlo convergence studies were not run.

## Source preservation

All seven original source hashes in `SOURCE_PROVENANCE.txt` were unchanged after installation. The installed MATLAB files matched the reviewed staging files. Numerical source comparisons checked that the extracted batch helper expressions and global ADMM solver/prior/covariance expressions retained their original arithmetic, apart from renamed helpers and removed debug output. The README identifies the deliberate runnable configuration and reproducibility changes, as well as inherited model limitations.
