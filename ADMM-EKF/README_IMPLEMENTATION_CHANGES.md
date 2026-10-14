# ADMM-EKF 與 local-neighborhood D-EKF 實作變更說明

本文件說明這次檢查後所做的程式修改、修改原因，以及目前實作與原先版本的差異。操作方式與 MAT 欄位讀取範例另見 `README_local_neighborhood_ekf.md`。

> 重要：目前 codebase 中保留了兩種不同的 estimator。它們回答不同的研究問題，不能把名稱或結果混在一起比較。
>
> 1. `run_sim_dkf_consensus_admm_v5.m`：全網路的 nonlinear consensus ADMM/MAP estimator。
> 2. `run_local_neighborhood_ekf_study.m`：固定 10-node circular network；每個中心 node 分別在自己的 3/7/10-node closed neighborhood 內執行一個獨立 ADMM EKF optimization。

本文的行號以目前版本為準；後續修改檔案後可能位移。

## 1. 最重要的 estimator 定義差異

### 1.1 原本的 consensus ADMM 並不是 local-neighborhood estimator

原先的想法是在 node `n` 的 local objective 中傳入：

```matlab
range_local{n}
doppler_local{n}
```

即使這些變數包含 node `n` 的 closed-neighborhood measurements，只要後面仍在一張 connected graph 上執行 consensus ADMM，最後求解的仍是：

```text
minimize_x  sum_n f_n(x)
```

共識約束會透過多跳通訊把所有 local objectives 聚合成一個全網解。因此，「每個 local function 收到鄰居量測」不等於「每個 node 是獨立的 local-neighborhood estimator」。

若每個 measurement `j` 被所有包含 `j` 的 closed neighborhoods 重複加入，則全域 objective 會變成：

```text
prior(x) + sum_j multiplicity_j * likelihood_j(x)
```

在目前 10-node 正規圓形 topology 中，CR30、CR50、CR500 的每個 measurement 分別會被重複 3、7、10 次。這只是把同一個 physical measurement 當成多份獨立資料，並不是真的得到新量測；在正規 graph 上甚至可能只把整個 objective 等比例放大，而不改變 exact minimizer。

### 1.2 新增 per-ego-neighborhood ADMM D-EKF

`run_local_neighborhood_ekf_study.m:1` 新增了一個獨立的實驗流程。對每個 node `n`：

```text
C_n = {n} union neighbors(n)
```

每個中心 node 使用自己的 predicted mean/covariance，而 `C_n` 內的 participants 分散持有各自的 sensor measurement：

```text
center n prior (mu_n^-,P_n^-) is partitioned over C_n
each participant j in C_n contributes measurement j once
        -> edge ADMM only on the induced graph of C_n
        -> one ego-consensus posterior (mu_n^+,P_n^+)
        -> feedback only to center node n
```

這個流程具有以下特性：

- 整張 physical network 始終有 10 nodes。
- CR30/CR50/CR500 只決定每個 ego problem 含 3/7/10 participants。
- 十個中心 nodes 各自執行自己的 ADMM problem；不同 ego problems 不再串成全網 consensus。
- `estimates_D` 保存 ego ADMM 解；`estimates_D_local` 保存數學上等價的直接 stacked-EKF 解作 validation。
- 每個 ego posterior 只回饋給其中心 node，因此 CR30/50 時不同中心 nodes 可以有不同 state/covariance。

目前圓形 10-radar geometry 的結果為：

| Communication radius | Degree | 每個 node 使用的 sensors |
|---:|---:|---:|
| CR30 | 2 | 3 |
| CR50 | 6 | 7 |
| CR500 | 9 | 10 |

sensor sets 建立於 `run_local_neighborhood_ekf_study.m` 的 topology 區塊；每個 set 的 induced adjacency 交給 `ego_neighborhood_admm_ekf_update`，內部再使用 edge ADMM。

### 1.3 理論預期也因此不同

| 實驗 | 增加 CR 的主要作用 | exact convergence 後的理論預期 |
|---|---|---|
| Consensus ADMM，正確 self-only objective | 增加 communication edges | 改變收斂速度；connected graph 的最終全域解應相同 |
| Per-ego-neighborhood ADMM D-EKF | 每個中心 node 的 ADMM problem 含 3/7/10 sensors | 每個 ego estimator 的 information 隨 neighborhood 增大；CR500 的十個 ego problems 都等於 all-sensor C-EKF |

因此，要驗證「3 sensors、7 sensors、10 sensors 的 D-EKF」時，應執行 `run_local_neighborhood_ekf_study.m`，而不是只修改 `run_sim_dkf_consensus_admm_v5.m` 的 `MEASUREMENT_MODE`。

### 1.4 新 study 中 ADMM consensus 的精確定義

對中心 node `n`，令 `m=|C_n|`。每個 participant `j in C_n` 建立一份 information packet：

```text
J_packet,j   = (1/m) J_prior,n + H_j' R^-1 H_j
eta_packet,j = (1/m) eta_prior,n
               + H_j' R^-1 (y_j-h_j+H_j mu_n^-)
```

只在 `C_n` 的 induced graph 上，edge-based ADMM 對
`[eta_packet,j; vec(J_packet,j)]` 求 average consensus。將 average information
乘回 `m` 後得到：

```text
J_post,n   = J_prior,n + sum_{j in C_n} H_j' R^-1 H_j
eta_post,n = eta_prior,n
             + sum_{j in C_n} H_j'R^-1(y_j-h_j+H_j mu_n^-)
```

最後由 `P_n^+=J_post,n^-1`、`mu_n^+=P_n^+ eta_post,n` 回復 ego posterior。
這與直接使用相同 `C_n` measurement stack 的 EKF update 相同，但 optimization
由該 ego graph 上的 ADMM 分散完成。

關鍵是十個 ego problems 彼此獨立：CR30 的 node 1 consensus 結果不會再傳入
node 5 的 ego problem。如此 3-node neighborhood 確實無法取得其 neighborhood
以外的 measurement information；7-node 與 10-node 的資訊量才會依序增加。

## 2. Measurement model 與 likelihood 修正

### 2.1 統一 range/Doppler convention

新增單一 canonical implementation：

```matlab
[z, H] = models.MeasureModelAndJacobian(state, radarIdx, network_topo, env)
```

位置在 `models.m:17-53`。目前所有 active EKF paths 使用 global state：

```text
state = [x; y; vx; vy]
delta = target_position - radar_position
r     = norm(delta)
fd    = (2/lambda) * velocity' * delta / r
```

原先不同 functions 中曾同時存在以下不一致：

- target-minus-radar 與 radar-minus-target 的 Doppler sign 不同。
- 某些 model 缺少 monostatic radar 的 factor `2`。
- local-coordinate model 被傳入 global state。
- 某些 Jacobian 沒有反映實際 radar position，使不同 sensors 可能得到相同或錯誤的 `H`。

現在 `models.MeasureModel` 與 `models.MeasureModelJacobian` 都委派給同一個 canonical function，避免 generator、D-EKF、C-EKF 與 PCRLB 使用不同 measurement convention。

### 2.2 修正 Gaussian negative log-likelihood 的尺度

原先 MLE/MAP likelihood 使用過類似：

```matlab
(1/2 * (sigma_fd2 * sigma_r2)) * residual_term
```

MATLAB 會把它解釋成乘上 `0.5*sigma_fd2*sigma_r2`，不是除以 covariance。這會使 measurement likelihood 與 EKF covariance update 使用完全不同的 noise model，而且錯誤倍率會隨 SNR 大幅變動。

目前 `ADMM_utils.m:36-47` 與 `ADMM_utils.m:95-102` 都改為完整 innovation 的 Mahalanobis cost：

```matlab
innovation = y - h(x);
nll = nll + 0.5 * innovation.' * (Sigma \ innovation);
```

這個寫法也正確支援 range/Doppler correlated covariance，而不是假設只能使用 diagonal `R`。

### 2.3 修正 pre-whitening

原先 whitening path 只轉換 model，measurement 沒有一起轉換；而且曾先覆寫 `r_model`，再拿覆寫後的值計算 Doppler。

現在一律先形成 physical residual，再對完整 residual whitening：

```matlab
innovation = [range_measured - range_model;
              doppler_measured - doppler_model];
innovationW = L * innovation;
```

EKF paths 則一致地轉換 `y`、`h` 與 `H`。相關位置：

- `ADMM_utils.m:39-47`、`97-102`
- `dkf_neighbor_update.m:35-50`
- `dkf_neighbor_update_pw.m:9-57`
- `centralized_ekf_update_stack.m:20-37`

## 3. EKF state/covariance update 修正

### 3.1 D-EKF 現在保留 radar geometry

`dkf_neighbor_update.m:23-31` 原先使用 local-coordinate helper，但 caller 傳入的是 global `x_pred`。現在每個 sensor `j` 都呼叫：

```matlab
[hj, Hj] = models_ut.MeasureModelAndJacobian( ...
    x_pred, j, network_topo, env);
```

因此每個 radar 的位置、range direction 與 Doppler geometry 都會反映在 `h_j` 和 `H_j` 中。

### 3.2 covariance 使用 Joseph form

D-EKF、pre-whitened D-EKF 與 C-EKF 均改為：

```matlab
A = I - K*H;
P_post = A*P_pred*A' + K*R*K';
P_post = 0.5*(P_post + P_post');
```

位置：

- `dkf_neighbor_update.m:45-62`
- `dkf_neighbor_update_pw.m:49-57`
- `centralized_ekf_update_stack.m:31-37`

相較於簡化式 `(I-KH)P`，Joseph form 對浮點誤差較穩定，較能維持 covariance 對稱與 positive semidefinite。

### 3.3 移除顯式 matrix inverse

active update paths 使用 MATLAB linear solve `/` 或 `\`，不再以 `inv(...)` 建立 Kalman gain/information update。這可提高數值穩定性，也避免不必要的 inverse。

## 4. 原本 consensus ADMM 程式的修正

這一節只適用於 `run_sim_dkf_consensus_admm_v5.m`，不等於新 local-neighborhood study。

### 4.1 objective partition 改為 self-only

目前 `run_sim_dkf_consensus_admm_v5.m:85-92` 使用：

```matlab
MEASUREMENT_MODE = "self_only";
```

並在 `run_sim_dkf_consensus_admm_v5.m:181-192` 檢查每個 physical measurement 在所有 local objectives 的總和中剛好出現一次。這避免 closed-neighborhood overlap 造成重複計數。

這個設定的意義是「正確分割一個全域 posterior」，不是「每個 node 只做 local estimation」。

### 4.2 state coordinate 與 Monte Carlo initialization

原先曾將初始 global state 減去各 radar position，但後續 MAP likelihood 又把 parameter 當 global state；目前 `x_dadmm{n}=x0`，所有 state paths 統一為 global coordinates（`run_sim_dkf_consensus_admm_v5.m:323-328`）。

另外，不再於 Monte Carlo loop 中覆寫共同的 `x0`。每次 trial 使用相同 estimator definition，避免 MC1 與 MC2 以後使用不同初始條件。

### 4.3 正確的 ADMM update order

目前 `run_sim_dkf_consensus_admm_v5.m:753-834` 使用：

```text
1. x update with lambda^k and z^k
2. z update with x^(k+1) and lambda^k
3. unscaled dual update with x^(k+1) and z^(k+1)
```

原先的 dual-before-x 順序與 objective 中使用的 iteration variables 不一致。

### 4.4 augmented penalty 尺度

原先 penalty 類似：

```matlab
norm((rho/2) .* (x-z))^2
```

這實際產生 `rho^2/4` 的權重，與 z/dual update 所用的 `rho` 不一致。現在改為：

```matlab
0.5 * sum(rho(:) .* (x-z).^2)
```

因此 x、z 與 dual updates 使用同一個 augmented-Lagrangian 定義。

### 4.5 primal/dual residual 與停止條件

目前 residual 計算位於 `run_sim_dkf_consensus_admm_v5.m:836-890`：

- primal residual 由 `x_n-z_nj` 組成。
- dual residual 使用 node-space 的 signed incident-edge sum，即 `A' C B Delta-z`。
- absolute 與 relative thresholds 分別記錄為 `eps_pri`、`eps_dual`。
- 只有 `r <= eps_pri` 且 `s <= eps_dual` 同時成立才標記 converged。

原先只看 primal residual，complete graph/local objectives 相同時可能第一輪就得到零 primal residual，雖然 `z` 尚未 stationary，卻被誤判收斂。

### 4.6 adaptive rho 方向

目前 `run_sim_dkf_consensus_admm_v5.m:893-899` 的 residual balancing 為：

```text
if primal > mu * dual: rho increases
if dual   > mu * primal: rho decreases
```

這與 unscaled dual formulation 一致；原先的增減方向相反。

### 4.7 solver 與 convergence diagnostics

目前額外記錄：

- fmincon exit flags
- cumulative failure count
- maximum inner iterations
- stop reason
- primal/dual residual histories
- primal/dual thresholds
- rho history 與 final rho

相關 `Results` fields 在 `run_sim_dkf_consensus_admm_v5.m:119-143`，寫入位置在 `459-477`，ADMM `info` 在 `937-958`。

### 4.8 consensus covariance 的意義

原先 state update 與 `P_nodes/P_dadmm` 曾使用不同 information sets：state 可能是重複 neighborhood objectives 的 ADMM 解，covariance 卻是 one-hop EKF update；因此 state RMSE 與 `sqrt(diag(P))` 不一致。

目前 consensus runner 在 `run_sim_dkf_consensus_admm_v5.m:918-935` 依照同一個 global MAP posterior 建立：

```text
J_post = J_common_prior + sum over physical sensors once of H_j' R^-1 H_j
P_global = inverse(J_post)
```

再將同一個 `P_global` 指派給所有 consensus nodes。`P_nodes` 與遞迴用的 `P_dadmm` 都另外存入 log，方便逐 node 驗證。

這個 common covariance 只適用於 nonlinear global MAP consensus runner。新 study 的每個中心 node 各有一個 ego ADMM posterior；CR30/50 時十個中心 nodes 的 `P_n^+` 通常不同，只有 CR500 時才全部等於 centralized covariance。

## 5. 時間索引、初始 prior 與第一個繪圖點

### 5.1 measurement timing

measurement index 1 對應 initial state。因此 D-EKF、C-EKF 與 PCRLB 在第一筆 measurement update 前不執行 `F/Q` prediction；prediction 從第二筆 measurement 開始。

新 study 中的相關位置：

- C-EKF：`run_local_neighborhood_ekf_study.m:253-263`
- D-EKF：`run_local_neighborhood_ekf_study.m:291-301`
- local PCRLB：`calculate_local_pcrlb_from_information.m:97-117`
- centralized PCRLB：`calculate_local_pcrlb_from_information.m:124-140`

這修正了原先第一個 measurement 尚未更新便先把 state 推進一個 `dt` 的時間錯位。

### 5.2 新增共同的更新前 t=0 點

64 個原始 samples 都是 measurement posterior。為了讓圖上的起始點具有一致意義，現在另外儲存更新前的 prior：

```matlab
Study.initial_prior.mu0
Study.initial_prior.P0
Study.initial_prior.empirical_rmse
Study.initial_prior.pcrlb_std
```

位置在 `run_local_neighborhood_ekf_study.m:359-391` 與 `467-477`。

plot-only arrays 有 65 點：

```matlab
Study.plot_time
Study.rmse_D_plot
Study.rmse_C_plot
Study.pcrlb_D_plot
Study.pcrlb_C_plot
```

第一點的定義為：

- D3、D7、D10 與 C-EKF：同一個 `mu0` 對 sampled initial truths 的 empirical prior RMSE。
- D-PCRLB 與 C-PCRLB：`sqrt(diag(P0))`。
- 原本第 1 筆 measurement posterior 改畫在 `t=dt`，不再錯標成更新前 `t=0`。

原本的 64-point `rmse_D/rmse_C/pcrlb_D/pcrlb_C` 保留不變，供 posterior-only 數值分析使用。

## 6. Monte Carlo 與 noise experiment 的差異

新 local study 使用 Bayesian-consistent paired experiment（`run_local_neighborhood_ekf_study.m:130-174`）：

1. 每個 MC 的 initial truth 由 `x_1 ~ N(mu0,P0)` 抽樣。
2. truth dynamics 使用與 filter/PCRLB 相同的 `F` 與 `Q`。
3. 先產生共同的 standard-normal measurement noise。
4. 同一 SNR、MC、time 與 radar 的 measurement realization 在所有 CR 間完全共用。
5. 不同 SNR 只用對應的 Cholesky factor 縮放同一組 base noise。
6. C-EKF 每個 SNR/MC 只計算一次，因為它與 communication radius 無關。

這使 CR30/50/500 成為 paired comparison；差異來自 sensor sets，不是不同 random trial。

相較之下，先前固定一條 deterministic truth 卻拿 Bayesian PCRLB 比較，兩者不屬於同一 statistical ensemble。新 study 的 truth、filter prior、process model 與 PCRLB assumptions 已對齊。

## 7. PCRLB 的修改

### 7.1 保存每個 radar 的 expected information

`calculate_pcrlb_v5.m:133-176` 新增：

```matlab
info.JmeasByRadar   % [4 x 4 x Nradar x T]
```

而 total measurement information 保證等於：

```matlab
info.Jmeas = sum(info.JmeasByRadar, 3);
```

這讓 local 與 centralized bounds 使用相同的 state samples、model 與 noise assumptions。

### 7.2 新增 local PCRLB recursion

`calculate_local_pcrlb_from_information.m` 是新檔案。它對每個 CR、node 與 time，只加總該 node sensor set 中的 information：

```text
J_n,t^post = inverse(P_n,t^pred) + sum_{j in C_n} J_j,t^measurement
P_n,t^bound = inverse(J_n,t^post)
```

輸出：

```matlab
P_local  % [4 x 4 x node x time x CR]
P_all    % [4 x 4 x time]
```

function 內加入 sensor index、duplicate、symmetry、SPD/PSD 與 Cholesky solve 檢查。

### 7.3 圖上的 ego-ADMM PCRLB

ADMM 是每個 ego local EKF 的 distributed solver，所以相符的 bound 是每個中心 node 自己的 local PCRLB：

```text
P_bound,n(t) = inverse(P_pred,n(t)^-1
                       + sum_{j in C_n} J_meas,j(t))
```

`Study.pcrlb_D` 是先對十個 ego variances 平均再開平方，與 node-pooled RMSE 對齊。完整 node bounds 位於 `Study.PCRLB_D_nodes`；`PCRLB_D_consensus` 是同一 bound 的明確命名 alias。

## 8. RMSE 定義修正

### 8.1 D-EKF

主 `Study.rmse_D` 使用十個中心 nodes 各自的 ego-ADMM state，並以 node-pooled squared error 計算：

```text
RMSE_D(s,t) = sqrt(
    mean over MC and node of
    (xhat_n(s,t,MC) - xtrue(s,t,MC))^2
)
```

實作在 `run_local_neighborhood_ekf_study.m:335-357`。

直接 stacked local EKF 的 validation metric 另外保存在 `Study.rmse_D_local`；它應與 ego-ADMM RMSE 一致至 solver tolerance。原先某些 plotting/analysis paths 是先算 node estimate 平均值：

```text
mean_n(xhat_n)
```

再計算 RMSE。這會讓 nodes 的正負誤差互相抵消，衡量的是 network-average estimate，而不是「一個典型 local node 的 estimation error」。兩種 metric 都可以有用途，但不能拿後者來驗證 local estimator 隨 sensor count 增加的效能。

### 8.2 C-EKF

C-EKF RMSE 只對 MC 平均：

```text
RMSE_C(s,t) = sqrt(mean_MC((xhat_C(s,t,MC)-xtrue(s,t,MC))^2))
```

### 8.3 position/velocity 圖的 aggregation

目前 2-row figures 顯示：

```text
position metric = sqrt(metric_x^2  + metric_y^2)
velocity metric = sqrt(metric_vx^2 + metric_vy^2)
```

這是 vector-component aggregation，不是再次對資料取 RMSE。

## 9. 新增與修改的檔案

### 9.1 新增檔案

| 檔案 | 用途 |
|---|---|
| `run_local_neighborhood_ekf_study.m` | per-node local-neighborhood EKF、Gaussian ADMM consensus、paired MC、C-EKF、metrics、validation 與 MAT 儲存 |
| `calculate_local_pcrlb_from_information.m` | 由 per-radar information 建立 node/CR-matched local PCRLB |
| `plot_local_neighborhood_ekf_study.m` | 每個 state 一張 3x3 SNR/CR figure |
| `plot_snr_crlb_rmse_study.m` | 直接讀 `Study` 的 position/velocity RMSE/PCRLB 圖 |
| `plot_dekf_3_7_cekf_cpcrlb.m` | 只畫 D-EKF-3、D-EKF-7、C-EKF 與 C-PCRLB |
| `README_local_neighborhood_ekf.md` | 執行、讀 MAT 與出圖使用手冊 |
| `README_IMPLEMENTATION_CHANGES.md` | 本文件；記錄舊/新實作差異與原因 |

### 9.2 修改檔案

| 檔案 | 主要修改 |
|---|---|
| `ADMM_utils.m` | likelihood scaling、Doppler convention、whitening、MAP prior count、ADMM penalty、graph edge cases、generator consistency |
| `models.m` | 新增 canonical global measurement model/Jacobian，舊 global entry points 委派至同一實作 |
| `dkf_neighbor_update.m` | 使用正確 global radar geometry、Joseph covariance、symmetrization |
| `dkf_neighbor_update_pw.m` | whitening/model 一致、移除 explicit inverse、Joseph covariance |
| `centralized_ekf_update_stack.m` | 共用 canonical model，使用 Joseph covariance 與 symmetrization |
| `calculate_pcrlb_v5.m` | 回傳 per-radar expected information，以便建立 local bounds |
| `run_sim_dkf_consensus_admm_v5.m` | objective partition、初始化/時間、ADMM order/penalty/residual/stop/rho、covariance matching、diagnostic logging |
| `make_figs.m` | convergence plot 的 x-axis 改由實際 iteration 數決定，不再固定 50/100 |

## 10. Plotting 與外部參考 script 的差異

外部參考檔：

```text
/Users/shaohung/Documents/Codex/2026-07-11/i/outputs/plot_snr_crlb_rmse.m
```

沒有直接修改。原因是它原本：

- 是帶有 `clear` 與 hard-coded paths 的 script。
- 預期舊式 `Log.Results` cell schema。
- 對 node matrix 先取 node mean 再算 RMSE。
- 只支援一條 PCRLB，不支援每個 CR 的 local PCRLB 與 centralized PCRLB。

因此專案內新增 `plot_snr_crlb_rmse_study.m`，保留參考圖的 position/velocity layout、log y-axis、字型與 PNG/PDF/FIG 輸出，但直接讀取新 `Study` 中已正確計算的 numeric metrics。

依後續需求，`plot_dekf_3_7_cekf_cpcrlb.m` 只顯示：

- D-EKF-3 sensors
- D-EKF-7 sensors
- C-EKF
- C-PCRLB

兩個 Study-compatible plot functions 都優先使用 65-point `*_plot` fields，且已移除整張 figure 的總標題。每個 SNR 欄位的小標題仍保留，方便辨識 20/30/40 dB。

## 11. 目前 MAT schema 的重要差異

新 study 不再使用大量 `[MC x time]` nested cells，而使用固定維度 numeric arrays：

| Field | Dimensions | 意義 |
|---|---:|---|
| `truth` | `4 x T x MC` | Bayesian truth trajectories |
| `estimates_D` | `4 x node x T x MC x SNR x CR` | 每個 local node posterior state |
| `diagP_D` | `4 x node x T x MC x SNR x CR` | 每個 local node posterior covariance diagonal |
| `estimates_D_local` | `4 x node x T x MC x SNR x CR` | direct stacked local EKF validation state |
| `diagP_D_local` | `4 x node x T x MC x SNR x CR` | direct stacked local EKF validation covariance diagonal |
| `estimates_C` | `4 x T x MC x SNR` | centralized posterior state |
| `diagP_C` | `4 x T x MC x SNR` | centralized posterior covariance diagonal |
| `PCRLB_D_nodes` | `4 x 4 x node x T x SNR x CR` | 每個 local node 的完整 bound covariance |
| `PCRLB_D_consensus` | `4 x 4 x node x T x SNR x CR` | ego-ADMM matched bound covariance |
| `PCRLB_C` | `4 x 4 x T x SNR` | centralized bound covariance |
| `rmse_D` | `4 x T x SNR x CR` | 64-point posterior node-pooled RMSE |
| `rmse_C` | `4 x T x SNR` | 64-point posterior centralized RMSE |
| `pcrlb_D` | `4 x T x SNR x CR` | 已開平方的 local bound summary |
| `pcrlb_C` | `4 x T x SNR` | 已開平方的 centralized bound |
| `*_plot` | 同 metric，但 time 為 `T+1` | prepend 共同更新前 prior 點 |

注意：`pcrlb_D`、`pcrlb_C` 與其 `_plot` 版本已是 standard-deviation scale，不可再次呼叫 `sqrt`。

## 12. 已執行的驗證

### 12.1 Measurement/Jacobian/update tests

- canonical analytic Jacobian 與 finite difference 的最大誤差約 `1.37e-7`。
- unwhitened D-EKF、pre-whitened D-EKF 與同 sensor stack 的 centralized update 在測試中一致至 numerical precision。
- correlated measurement covariance 的 whitened/unwhitened result 一致至約 `7.1e-15`。
- 測試 posterior covariance 對稱且 minimum eigenvalue 為正。

### 12.2 Local-neighborhood study invariants

- CR30/50/500 的 neighborhood sizes 為 3/7/10。
- CR500 時每個 D-EKF node 與 C-EKF 使用完全相同、排序相同的 10-sensor measurement stack。
- 最新 v4 正式結果中 CR500 的 state difference 為 `3.07e-12`。
- 儲存的 covariance diagonal difference 為 `5.77e-15`。
- full-neighborhood ego-ADMM PCRLB 與 centralized PCRLB difference 為 `0`。
- ego ADMM 與直接 stacked local EKF 的最大 state/diag(P) 差為 `2.24e-6` / `1.86e-7`。
- CR30 不同中心 node 的最大 state spread 為 `4.72`，證明沒有再被第二層全網 consensus 壓成同一 estimate。
- 所有 EKF posterior 與 PCRLB covariance 通過 finite、symmetry 與 positive-definite checks。
- local PCRLB 隨 3 -> 7 -> 10 sensors 呈 non-increasing，完整 covariance difference 通過 PSD ordering check。

### 12.3 共同 prior plot point

- `plot_time` 與所有 `*_plot` metrics 都有 65 點。
- D3、D7、D10 與 C 的第 1 點完全相同。
- D/C PCRLB 的第 1 點都等於 `sqrt(diag(P0))`。
- 四曲線 figure 已移除 figure-level title。

目前正式的 v4 ego-neighborhood ADMM artifact：

```text
results/local_neighborhood_ekf/
local_neighborhood_ekf_study_20260814_194607.mat
```

## 13. 如何選擇要執行哪一支程式

### 研究 communication graph 對 consensus 收斂速度的影響

使用：

```matlab
run_sim_dkf_consensus_admm_v5.m
```

應觀察：

- ADMM iteration count
- primal/dual residual histories
- convergence rate
- fmincon status
- converged global state/objective 是否跨 connected topologies 一致

不要期待 exact-converged RMSE 隨 degree 必然下降，因為所有 connected graphs 解的是同一個 global posterior。

### 研究 3/7/10-node ego-neighborhood ADMM 的效能

使用：

```matlab
Study = run_local_neighborhood_ekf_study(config);
```

應觀察：

- `Study.rmse_D` 或 `Study.rmse_D_plot`
- `Study.pcrlb_D` 或 `Study.pcrlb_D_plot`
- node-specific `Study.estimates_D`
- node-specific `Study.diagP_D`
- `Study.PCRLB_D_nodes`
- `Study.ADMM.iterations/primal_residual/dual_residual`
- direct stacked-EKF validation 使用 `Study.rmse_D_local` 與 `Study.pcrlb_D_local`

這才是 D-EKF-3、D-EKF-7 與 D-EKF-10 的比較。

## 14. 解讀結果時仍需注意

1. **有限 Monte Carlo 不保證逐點單調。** 理論上增加獨立 measurement information 會讓 local covariance/PCRLB 非增，但 10 MC 的 realized RMSE 可能在部分 time points 交叉。
2. **nodes 不是獨立 MC samples。** 不同 nodes 共用且重疊 measurements。主要 RMSE 可以對 nodes 聚合平方誤差，但 confidence interval 不能把每個 node 當成完全獨立 trial。
3. **EKF 與 PCRLB 不必完全貼合。** PCRLB 是 model-based lower bound；非線性、linearization error 與有限樣本會造成 RMSE 與 bound 間距。
4. **CR500 equality 是 implementation invariant。** 因為每個 node 與 C-EKF 使用完全相同的 prior、measurement ordering、model 與 update function，所以 state 與 covariance diagonal 應一致；若未來不再為零，應先視為 regression。
5. **不要將 local node covariances 做算術平均後稱為 network-mean covariance。** nodes 共享 measurements，cross-covariances 未知；算術平均沒有描述 `mean(x_n)` covariance 所需的 cross terms。
6. **不要恢復 overlapping neighbor priors。** 相鄰 nodes 的 recursive priors 已因共享歷史資料而相關，直接將它們視為獨立 Gaussian 相乘會造成 data incest 與過度自信。

## 15. 總結

這次修改不只是調整 line 827 傳入哪些 measurements，而是將兩種原本混在一起的概念拆開：

- consensus ADMM runner 現在使用一致的 global objective、likelihood、ADMM residual 與 covariance definition。
- local-neighborhood study 對十個中心 nodes 各跑一個獨立 3/7/10-node ego ADMM，沒有第二層全網 consensus。
- C-EKF、D-EKF、local PCRLB 與 centralized PCRLB 現在使用同一 measurement convention、timing、prior/process ensemble 與 paired noise realizations。
- RMSE 與 PCRLB 的 aggregation 已和 estimator 定義對齊。
- plot-only time series 額外加入共同更新前 `mu0/P0`，因此所有曲線在 `t=0` 具有相同且明確的統計意義。
