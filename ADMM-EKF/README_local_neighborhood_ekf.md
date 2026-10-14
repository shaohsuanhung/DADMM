# Local-neighborhood D-EKF study

若要了解本次修改了哪些數學與程式邏輯，以及它們和原先 ADMM-EKF 實作的差異，請先閱讀 `README_IMPLEMENTATION_CHANGES.md`。

這份說明對應 `run_local_neighborhood_ekf_study.m` 的實驗資料，包含：

- 10 個 radar nodes。
- CR 30、50、500，每個 node 分別使用 3、7、10 個 sensors。
- SNR 20、30、40 dB。
- D-EKF、C-EKF、local PCRLB 與 centralized PCRLB。
- 固定保留 10-node circular network；每個中心 node 只在自己的 3/7/10-node closed neighborhood 內執行一個獨立 edge-based ADMM。
- 不同中心 node 的 ego-neighborhood ADMM 不會再串成全網 consensus。
- `Study.estimates_D_local/diagP_D_local` 保存直接 stacked-EKF 參考值；主要 `Study.estimates_D/diagP_D` 保存數學上等價的 ego-ADMM 解。

## 1. 執行完整實驗

在 MATLAB 中切換到 `ADMM-EKF`：

```matlab
cd('/Users/shaohung/Desktop/Desktop - K2D4GJQXL6/PhD/paper/DADMM/ADMM-EKF')
clear classes

cfg = struct( ...
    'NumMonteCarlo', 10, ...
    'NumSteps', 64, ...
    'SNRdB', [20 30 40], ...
    'CommunicationRadii', [30 50 500], ...
    'NumPCRLBSamples', 2000, ...
    'MakePlots', false, ...
    'SaveResults', true);

Study = run_local_neighborhood_ekf_study(cfg);
matFile = Study.saved_file
```

ADMM Gaussian consensus 的可選設定及預設值：

```matlab
cfg.ADMMMaxIter = 1000;
cfg.ADMMAbsTolerance = 1e-8;
cfg.ADMMRelTolerance = 1e-7;
cfg.ADMMRho = 1;
```

每個 time/MC/SNR/CR 的 diagnostics 位於 `Study.ADMM.iterations`、
`primal_residual`、`dual_residual` 與 `mean_target_error`。

結果預設儲存在：

```text
ADMM-EKF/results/local_neighborhood_ekf/
```

檔名包含 timestamp，因此重新執行不會覆寫舊的 MAT。

目前正式的 10-MC、64-step、Study v4 per-ego-neighborhood ADMM 結果：

```text
results/local_neighborhood_ekf/local_neighborhood_ekf_study_20260814_194607.mat
```

## 2. 載入指定 MAT

```matlab
matFile = fullfile(pwd, 'results', 'local_neighborhood_ekf', ...
    'local_neighborhood_ekf_study_YYYYMMDD_HHMMSS.mat');

loaded = load(matFile, 'Study');
Study = loaded.Study;
```

自動尋找最新結果：

```matlab
resultDir = fullfile(pwd, 'results', 'local_neighborhood_ekf');
files = dir(fullfile(resultDir, 'local_neighborhood_ekf_study_*.mat'));
assert(~isempty(files), 'No study MAT files were found.');
[~, newest] = max([files.datenum]);
matFile = fullfile(files(newest).folder, files(newest).name);

loaded = load(matFile, 'Study');
Study = loaded.Study;
```

## 3. 主要資料欄位與維度

State 順序固定為：

```text
[x, y, vx, vy]
```

令：

- `N = 10`：nodes 數量。
- `T = 64`：time steps。
- `M`：Monte Carlo 次數。
- `S = 3`：SNR 數量。
- `C = 3`：communication radius 數量。

| 欄位 | 維度 | 索引順序 |
|---|---:|---|
| `Study.truth` | `4 × T × M` | state, time, MC |
| `Study.estimates_D` | `4 × N × T × M × S × C` | state, node, time, MC, SNR, CR |
| `Study.diagP_D` | `4 × N × T × M × S × C` | state, node, time, MC, SNR, CR |
| `Study.estimates_D_local` | `4 × N × T × M × S × C` | ADMM 前的 local state |
| `Study.diagP_D_local` | `4 × N × T × M × S × C` | ADMM 前的 local covariance diagonal |
| `Study.estimates_C` | `4 × T × M × S` | state, time, MC, SNR |
| `Study.diagP_C` | `4 × T × M × S` | state, time, MC, SNR |
| `Study.rmse_D` | `4 × T × S × C` | state, time, SNR, CR |
| `Study.rmse_C` | `4 × T × S` | state, time, SNR |
| `Study.PCRLB_D_nodes` | `4 × 4 × N × T × S × C` | covariance, node, time, SNR, CR |
| `Study.PCRLB_D_consensus` | `4 × 4 × N × T × S × C` | 每個 ego-ADMM estimator 的 matched bound |
| `Study.PCRLB_C` | `4 × 4 × T × S` | covariance, time, SNR |
| `Study.pcrlb_D` | `4 × T × S × C` | state, time, SNR, CR |
| `Study.pcrlb_C` | `4 × T × S` | state, time, SNR |
| `Study.plot_time` | `1 × (T+1)` | common prior followed by posteriors |
| `Study.rmse_D_plot` | `4 × (T+1) × S × C` | common prior + D posterior RMSE |
| `Study.rmse_C_plot` | `4 × (T+1) × S` | common prior + C posterior RMSE |
| `Study.pcrlb_D_plot` | `4 × (T+1) × S × C` | `P0` bound + local posterior bounds |
| `Study.pcrlb_C_plot` | `4 × (T+1) × S` | `P0` bound + central posterior bounds |
| `Study.ADMM.iterations` | `N × T × M × S × C` | 每個中心 node 的 ego ADMM iterations |

`Study.pcrlb_D` 與 `Study.pcrlb_C` 已經是 covariance diagonal 的平方根，單位與 RMSE 相同，不可再次 `sqrt`。

`Study.time` 與沒有 `_plot` suffix 的 metrics 保留原始 64 個 measurement posteriors。正式繪圖應使用 `Study.plot_time` 與 `_plot` fields：第一點是更新前的共同 `mu0/P0`，原本第一筆 posterior 位於 `t=Dt`。更新前資料另存於：

```matlab
Study.initial_prior.mu0
Study.initial_prior.P0
Study.initial_prior.empirical_rmse
Study.initial_prior.pcrlb_std
```

## 4. 找出 SNR 與 CR index

不要假設 index；從設定值查找：

```matlab
snrDb = 30;
communicationRadius = 50;

snrIdx = find(Study.config.SNRdB == snrDb, 1);
crIdx = find(Study.config.CommunicationRadii == communicationRadius, 1);

assert(~isempty(snrIdx), 'Requested SNR was not found.');
assert(~isempty(crIdx), 'Requested CR was not found.');
```

讀取某個 state 的 time history：

```matlab
stateIdx = 1;  % 1=x, 2=y, 3=vx, 4=vy

dRmse = reshape(Study.rmse_D_plot(stateIdx,:,snrIdx,crIdx), 1, []);
cRmse = reshape(Study.rmse_C_plot(stateIdx,:,snrIdx), 1, []);
dPcrlb = reshape(Study.pcrlb_D_plot(stateIdx,:,snrIdx,crIdx), 1, []);
cPcrlb = reshape(Study.pcrlb_C_plot(stateIdx,:,snrIdx), 1, []);
time = Study.plot_time;
```

## 5. 使用指定版型出圖

原始參考 script：

```text
/Users/shaohung/Documents/Codex/2026-07-11/i/outputs/plot_snr_crlb_rmse.m
```

該檔案原本只讀取 hard-coded 舊式 `Log.mat`，並會先平均 nodes 再計算 RMSE；這不等於本實驗所需的 node-pooled RMSE。專案內的相容版本保留其 active figure 的 position/velocity 聚合、Times New Roman、log y-axis 與 PNG/PDF/FIG 輸出格式，但直接讀取正確的 `Study` metrics：

```matlab
outputDir = fullfile(pwd, 'results', 'local_neighborhood_ekf');
fig = plot_snr_crlb_rmse_study(matFile, outputDir);
close(fig);
```

也可以直接傳入已載入的 struct：

```matlab
fig = plot_snr_crlb_rmse_study(Study, outputDir);
```

輸出檔案：

```text
pos_vel_rmse_pcrlb_by_snr_cr.png
pos_vel_rmse_pcrlb_by_snr_cr.pdf
pos_vel_rmse_pcrlb_by_snr_cr.fig
```

只畫 D-EKF-3 nodes、D-EKF-7 nodes、C-EKF 與 C-PCRLB：

```matlab
fig = plot_dekf_3_7_cekf_cpcrlb(matFile, outputDir);
close(fig);
```

對應輸出是 `dekf_3_7_cekf_cpcrlb_by_snr.{png,pdf,fig}`。這兩個 plotting functions 都會優先使用 65-point `_plot` fields，並且不顯示最上方的總標題。

圖中的 position 與 velocity metrics 定義為：

```matlab
positionMetric = sqrt(metricX.^2 + metricY.^2);
velocityMetric = sqrt(metricVx.^2 + metricVy.^2);
```

## 6. 自己畫單一 state

```matlab
figure;
semilogy(time, dRmse, '-', 'LineWidth', 1.4); hold on;
semilogy(time, cRmse, '-', 'LineWidth', 1.4);
semilogy(time, dPcrlb, '--', 'LineWidth', 1.3);
semilogy(time, cPcrlb, '--', 'LineWidth', 1.3);
grid on;
xlabel('time (s)');
ylabel('RMSE / PCRLB');
legend('D-EKF RMSE', 'C-EKF RMSE', ...
    'D-PCRLB', 'C-PCRLB', 'Location', 'best');
```

## 7. RMSE 與 ADMM consensus

對每個中心 node `n`，其 closed neighborhood `C_n` 分別含 3/7/10 個 nodes。
每個 participant 貢獻自己的 measurement information 與 `1/|C_n|` predicted-prior
information，再只在 `C_n` 的 induced graph 上執行 ADMM。主欄位
`Study.rmse_D` 聚合十個不同 ego estimators 的結果：

```text
RMSE_D(s,t) = sqrt(mean over MC and node of error(s,node,t,MC)^2)
```

`Study.rmse_D_local` 是直接 stacked local EKF 的 validation reference，理論上應與
ego ADMM 的 `Study.rmse_D` 一致至 solver tolerance。

C-EKF RMSE 則只對 MC 平均：

```text
RMSE_C(s,t) = sqrt(mean over MC of error(s,t,MC)^2)
```

主 `Study.pcrlb_D` 對應各中心 node 的 3/7/10-sensor ego posterior，先對 node
variance 平均再開平方；`Study.pcrlb_D_local` 是相同 bound 的相容欄位。

## 8. 驗證 neighborhood 與 CR500 equality

```matlab
assert(isequal(Study.neighborhood_size, ...
    repmat([3; 7; 10], 1, 10)));

cr500Idx = find(Study.config.CommunicationRadii == 500, 1);
M = Study.config.NumMonteCarlo;
T = Study.config.NumSteps;
S = numel(Study.config.SNRdB);

centralState = reshape(Study.estimates_C, [4, 1, T, M, S]);
centralDiagP = reshape(Study.diagP_C, [4, 1, T, M, S]);

stateDifference = max(abs( ...
    Study.estimates_D(:,:,:,:,:,cr500Idx) - centralState), [], 'all');
covarianceDifference = max(abs( ...
    Study.diagP_D(:,:,:,:,:,cr500Idx) - centralDiagP), [], 'all');

fprintf('CR500 state difference: %.17g\n', stateDifference);
fprintf('CR500 diag(P) difference: %.17g\n', covarianceDifference);
```

兩個結果都應該是 `0`。

## 9. 解讀 Monte Carlo 曲線

增加 sensors 會使 expected information 增加，因此 local PCRLB 應隨 CR 增加而非增。有限 MC 的 empirical RMSE 不保證每一個 time sample 都嚴格單調；應同時檢查：

- time-pooled RMSE。
- 整條曲線的整體趨勢。
- PCRLB covariance 的 PSD ordering。
- 較大的 MC 次數與 confidence interval。

10 MC 適合快速比較趨勢；若要做論文中的統計結論，建議使用至少 100 MC。
