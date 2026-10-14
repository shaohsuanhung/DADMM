# Random connected radar tracking

這個資料夾是原有 MATLAB simulation 的獨立整理版。D-MAP／D-MLE 來自 `all_in_one/MAP_tracking.m`；D-EKF 入口採用你指定的 `ADMM-EKF/run_sim_dkf_consensus_admm_v5.m` 全網路版本。原始資料夾不需要加入 MATLAB path，也沒有被修改。

## 先執行

將 MATLAB Current Folder 切到本資料夾，開啟 **`run_example.m`**，選擇方法後按 Run：

```matlab
METHOD = "MAP";       % "MAP", "MLE", "EKF"
PROFILE = "quick";    % "quick", "original", "smoke"
TRAJECTORY = "original"; % "original", "smooth", "another", "nonlinear"
```

`run_example.m` 現在使用 `cfg.Radar.Mode="circle"` 與 `cfg.Graph.Mode="ring"`，恢復圓周上每個 node 連接左右最近兩個鄰居的設定。兩個 Mode 可以分別調整；`tracking_config` 的通用預設仍保留 random disk 與 connected random graph。

也可以在 Command Window 執行：

```matlab
cfg = tracking_config("quick");
cfg.Graph.NumNodes = 10;
cfg.Graph.EdgeProbability = 0.35;
cfg.Graph.Seed = 20261013;
Run = run_tracking("MAP", cfg);
```

執行需要 Optimization Toolbox (`fmincon`) 和 Statistics and Machine Learning Toolbox (`mvnrnd`)。已用 MATLAB R2025b 測試。入口會暫時加入本資料夾的三個子目錄，結束時還原 path 和 RNG；不需要 `addpath(genpath(...))` 或 `clear classes`。

## 程式結構

```text
run_example.m                   開始執行、修改常用參數
tracking_config.m               共用圖設定及執行規模
run_tracking.m                  建圖 → 選擇方法 → 儲存 → 繪圖
common/
  make_connected_random_graph.m 建立 connected random graph／ring 及共用矩陣格式
  make_radar_positions.m        產生共用的隨機／圓周雷達位置
  trajectory_config.m          三個方法共用的非直線真實軌跡設定
  gen_ref_trajectory.m         原 ADMM-EKF 的平緩參考曲線
  generateNonlinearTrajectory.m 原 ADMM-EKF 的正弦變速／轉向曲線
  make_tracking_truth.m        位置、逐時速度與完整參考軌跡
  plot_tracking_results.m       網路、軌跡、RMSE、殘差圖
all_in_one/
  batch_default_config.m        D-MAP／D-MLE 的參數
  run_batch_tracking.m          原本的 batch tracking 主流程
  BatchADMMUtils.m              固定一階線性化量測目標函數／解析梯度
ADMM_EKF/
  ekf_default_config.m          全網路 tracker 的參數
  run_ekf_tracking.m            prediction → ADMM correction → C-EKF
  tracking_consensus_admm_update.m  單一 tracking step 的 ADMM
  TrackingADMMUtils.m           EKF 固定一階量測目標函數／解析梯度
  TrackingModels.m              物理模型／Jacobian 與固定一階模型
  tracking_centralized_ekf_update.m
  tracking_pcrlb.m
results/                        每次執行建立新的子資料夾
check_graph.m                   不跑 estimator 的圖檢查
check_ring_graph.m              ring 最近兩鄰居、degree、連通性及舊 random 相容性
check_radar_positions.m         位置分布、重現性及 RNG 隔離檢查
check_trajectory.m              共同真值、速度、端點與 Doppler 檢查
check_trajectory_switch.m       兩種 generator、速度公式、prefix 與 RNG 檢查
check_batch_linearization.m     線性化、Jacobian、whitening 與梯度檢查
run_batch_linearization_test.m  MAP／MLE 整合檢查及局部求解 benchmark
check_ekf_linearization.m       EKF 一階模型、prior 分攤與 covariance 檢查
run_ekf_linearization_test.m    EKF 遞迴測試與固定二次型解比對
smoke_test.m                    三個方法的小規模整合測試
```

保留原來的 struct、cell、for-loop 與 `fmincon` 結構。兩套工具類別使用不同名稱，避免舊的多個 `ADMM_utils` 相互覆蓋。

## 通訊 topology 如何設定

恢復原本的圓周位置與相鄰連線，在 `run_example.m` 設定：

```matlab
cfg.Radar.Mode = "circle";
cfg.Graph.Mode = "ring";
cfg.Graph.NumNodes = 10;
```

連線為 `1–2–3–...–10–1`；每個 node 的 degree 都是 2，總共 10 條無向 edge。例如 node 1 的鄰居為 node 2 和 node 10。在等間距圓周上，這就是兩個最近的 node。Ring 至少需要 3 個 nodes。

Ring 的 `EdgeProbability`、graph `Seed` 與 `MaxAttempts` 不使用；儲存的 `graph_seed`／`edge_probability` 為 NaN（不適用），`generation_attempts=0`。它按 node index 連成一圈；若同時選 `Radar.Mode="random_disk"`，仍是相同的 index ring，而不是空間最近鄰圖。

切回 connected random graph：

```matlab
cfg.Graph.Mode = "random_connected";
cfg.Graph.EdgeProbability = 0.35;
cfg.Graph.Seed = 20261013;
% cfg.Radar.Mode = "random_disk"; % 可另行切換位置；也可保留 circle。
```

`tracking_config` 預設使用 10 個節點、無向、無 self-loop、無權重的 **Erdős–Rényi G(N,p)，條件為 connected**。每個候選圖獨立抽樣；不連通便重抽，超過 `MaxAttempts` 明確報錯。沒有額外強制補邊，因此不會把生成模型悄悄改成「spanning tree 加隨機邊」。舊設定沒有 `Graph.Mode` 時，也沿用這個模式。

- `cfg.Graph.NumNodes`：節點數，至少 2。
- `cfg.Graph.EdgeProbability`：抽取候選邊的機率，範圍 `(0,1]`。因為只接受連通圖，稀疏設定的最終邊密度會受此條件影響。
- `cfg.Graph.Seed`：重現同一張圖；修改它才換圖。
- `cfg.Graph.MaxAttempts`：預設 10000，避免在不合理的稀疏設定下無限重抽。

**通訊連線與雷達位置分開設定**。雷達位置的通用預設在半徑 30 m 的圓形區域內隨機產生，設定見下節。`random_connected` 的通訊邊不由距離決定，也沒有加入最大通訊距離限制。相同設定下，三個方法共用同樣的座標與 adjacency；SNR、量測公式及取樣長度仍須另外統一才能公平比較。

每次執行只建一次圖，所有 Monte Carlo trial 和 tracking step 共用該圖。Random graph 使用獨立的 `RandStream`，ring 不使用亂數，都不會消耗量測／軌跡的亂數。儲存內容包含 graph mode/model、adjacency、edge list、degree、Laplacian、incidence matrix、seed、實際抽樣次數和 `algebraic_connectivity = lambda2(L)`。這裡只記錄 lambda2，尚未加入其對收斂速度的實驗或理論分析。

## Random radar positions 如何設定

在 `run_example.m` 修改以下欄位即可：

```matlab
cfg.Radar.Mode = "random_disk"; % 圓形區域內均勻抽樣；"circle" 為等間距圓周
cfg.Radar.Radius = 30;          % 區域半徑 [m]
cfg.Radar.Center = [0, 0];      % 區域中心 [x,y]，單位 m
cfg.Radar.Seed = 20261014;      % 換 seed 就換位置，保留相同通訊邊
```

`random_disk` 使用 `r = Radius*sqrt(U)` 與均勻角度，讓節點在圓形**面積**內均勻分布。所有節點獨立抽樣，沒有額外加入最小間距或挑選較佳幾何。位置使用自己的 `RandStream`，不消耗量測、軌跡或 graph 的亂數。

每次 run 的位置固定，所有 Monte Carlo trial 及 tracking step 都共用。D-MAP、D-MLE、D-EKF 在相同 `cfg.Radar` 與節點數下使用相同部署。只改 `cfg.Radar.Seed` 可以比較位置的影響；只改 `cfg.Graph.Seed` 可以比較連線的影響。這次的共用 `cfg.Radar.Radius` 取代舊的 `BatchRadarRadius`／`EKFRadarRadius`，不再依方法自動切換半徑。若要重現先前圓周位置，設 `Mode="circle"`、`Center=[0,0]`，並指定原 batch 半徑 20 或 EKF 半徑 30。

`results.mat` 保存 `Run.network_topo.radar_pos`（每列對應一個節點的 `[x,y]`）和完整設定，另輸出 `radar_positions.csv` 與 `radar_layout.png`。位置改變不會改動 `gen_ref_trajectory` 或 CV prediction。

## 切換 truth trajectory，Constant Velocity estimation

在 `run_example.m` 只改 `TRAJECTORY` 即可切換；D-MAP、D-MLE、D-EKF 都讀取同一個 truth generator，估計端仍使用 CV。

| `TRAJECTORY` | 真實軌跡 |
| --- | --- |
| `"original"` | 原 `gen_ref_trajectory`：reference waypoints、`pchip`、弧長取樣與側向擾動 |
| `"smooth"` | 同一 reference 類型，關閉 waypoint 隨機擾動與側向擾動；仍是曲線 |
| `"another"` | 原 reference 類型，seed 改成 43，可自行換成 44、45 等 |
| `"nonlinear"` | 先前的 `generateNonlinearTrajectory`：正弦變速、轉向與可選的終點修正 |

Command Window 可直接選擇 nonlinear：

```matlab
cfg = tracking_config("quick");
cfg.Trajectory.Type = "nonlinear";
cfg.Trajectory.Nonlinear.TurnAmplitudeDeg = 50;
Run = run_tracking("EKF", cfg);
```

預設 `cfg.Trajectory.Type="reference"`，沿用原 `ADMM-EKF/ADMM_utils.m` 的 **`gen_ref_trajectory`**，限制在指定 bounds 內。沒有 `Type` 欄位的舊設定也會使用 reference。

```matlab
cfg.Trajectory.Duration = 30;           % 固定 reference 長度 [s]
cfg.Trajectory.Bounds = [-30,20,-30,25]; % [xmin xmax ymin ymax]
cfg.Trajectory.Speed = 50;              % 沿用原 v5 的 nominal speed
cfg.Trajectory.Seed = 42;
cfg.Trajectory.ShapeJitterY = 0.012;    % ADMM-EKF 原實作的值
cfg.Trajectory.LateralStd = 0.25;       % 原側向擾動 [m]；設 0 得到平滑 baseline
cfg.Trajectory.LongMode = "clip";
```

原始曲線的數值計算方式保留；抽出的函式只整理成可獨立呼叫的檔案、固定 seed、還原 caller RNG，並修正文件列出的 2×2 bounds 格式解析。相同 `N`、vector bounds、`dt`、speed 和 seed/options 的輸出已與原函式比對。

`gen_ref_trajectory` 會根據總樣本數選取曲線片段。因此三個方法先以固定 `Duration` 建立同一條完整 reference，再取所需前段；`quick`／`smoke` 不會另外抽取不同片段。所有 MC trials 共用真值，只改量測雜訊。共同 reference 要使用相同 trajectory config 和 `TimeStep`；各方法預設取樣數仍不同。

`LongMode="clip"` 沿用原函式：如果 `Speed*Duration` 超過曲線長度，會把整條曲線走一次，**實際速度降低**，不會循環繞行。`Speed` 因此不是所有設定下都保證達到的速度。位置生成後，依原 v5 註解中的做法，用完整路徑的 `gradient(x,dt)` 和 `gradient(y,dt)` 得到逐時速度，再一起取 prefix。量測生成和 truth logging 都讀取這組逐時速度；側向擾動也會反映在速度中。

`Type="nonlinear"` 的設定放在 `cfg.Trajectory.Nonlinear`，與 reference 的 `Bounds`、`Speed`、`LateralStd` 分開：

```matlab
cfg.Trajectory.Nonlinear.StartPosition = [-30, -30];
cfg.Trajectory.Nonlinear.EndPosition = [20, 20];
cfg.Trajectory.Nonlinear.InitialSpeed = 50;       % 原 v5 範例值 [m/s]
cfg.Trajectory.Nonlinear.InitialHeadingDeg = 45;
cfg.Trajectory.Nonlinear.SpeedAmplitude = 0.4;    % nominal speed 的相對振幅
cfg.Trajectory.Nonlinear.SpeedFrequency = 0.15;   % Hz
cfg.Trajectory.Nonlinear.TurnAmplitudeDeg = 50;
cfg.Trajectory.Nonlinear.TurnFrequency = 0.1;     % Hz
cfg.Trajectory.Nonlinear.EnforceEndpoint = true;
```

這條軌跡保留原函式的 Euler 位置積分與 cubic smoothstep 終點修正，速度使用原本 speed/heading 分量加上終點修正的解析導數。時間統一為 `0:dt:Duration`，因此起點和指定終點準確對應首末樣本；完整 reference 生成後才取 prefix，並還原 caller RNG。`EnforceEndpoint=false` 可關閉終點修正。它不會裁切到 reference 的 `Bounds`，而且終點修正會改變實際速度。

預設沿用原範例 `UseRandomness=true`、兩個 noise std 為 0，因此 nonlinear 模式只改 `Seed` 不會改變路徑。若要加入隨機變化，設定 `SpeedNoiseStd`（m/s）或 `HeadingNoiseStdDeg` 大於 0；平滑窗口由 `SmoothNoiseWindow`（樣本數）控制。所有方法在相同設定、`TimeStep` 下共用完整 reference。

`Trajectory.Duration` 是完整 reference 的秒數。EKF 實際樣本數為 `NumCPIPerMeasurement * TrackTime`，實際追蹤至 `(樣本數-1)*TimeStep` 秒；`TrackTime` 在目前程式中不是直接的秒數。

估計端沿用上一版的 CV transition `F = [I, dt*I; 0, I]`：

- D-EKF／C-EKF：CV mean/covariance prediction，`P_pred = F*P*F' + Q`。
- D-MAP／C-MAP：跨 batch 的 prior mean 使用 CV。Batch 間隔為 `SamplesPerBatch*TimeStep`；prior covariance 仍由舊版固定 proxy 經 `F*P_proxy*F'+Q` 取得，沒有新增 posterior covariance recursion。
- D-MLE／C-MLE：CV 用於線性化點與最佳化初始值，likelihood 不加入 dynamic prior。

`cfg.EKF.ProcessAccelerationVariance` 與 `cfg.Batch.ProcessAccelerationVariance` 分別控制估計端的 CV process noise，可獨立於 truth trajectory 調整。

EKF 的 CV-PCRLB 在 `original`、`quick`、`smoke` 三種 profile 均預設開啟，使用 EKF 相同的 `F`、`Q`、初始 prior 與量測 covariance，預設以 2000 個 state samples 計算 expected measurement information。可設定 `cfg.EKF.CalculatePCRLB = false` 關閉。PCRLB helper 的生成模型是 Gaussian CV，因此保留 **assumed-CV reference** 標示；對指定的 nonlinear truth，它是 CV 模型的參考下界。

## 修改實驗參數

```matlab
cfg = tracking_config("original");
cfg.Batch.NumMonteCarlo = 10;
cfg.Batch.NumBatches = 30;
cfg.Batch.SamplesPerBatch = 64;
cfg.Batch.SNRdB = 40;
cfg.Batch.MaxIterations = 1000;

cfg.EKF.NumMonteCarlo = 10;
cfg.EKF.NumCPIPerMeasurement = 64; % 與 TrackTime 相乘得到 tracking epochs
cfg.EKF.TrackTime = 1;
cfg.EKF.NumDopplerSamples = 64;   % 決定 Doppler noise variance，獨立於 epochs
cfg.EKF.SNRdB = 40;
cfg.EKF.ADMMMaxIterations = 2000;
cfg.EKF.CalculatePCRLB = true;   % 預設開啟 assumed-CV reference
cfg.EKF.PCRLBNumStateSamples = 2000;

cfg.MakePlots = true;
cfg.FigureVisible = true;        % false 時只輸出圖片，不保留視窗
Run = run_tracking("EKF", cfg);
```

`original` 保留來源的時間、MC 與 iteration 上限；計算量較大。`quick` 縮短資料與 iteration，供初步探索。`smoke` 使用極小上限，只驗證能執行。Batch 的 `SamplesPerBatch` 也會改變原公式中的 Doppler variance，因此縮短 profile 的數值不能代表原實驗。到達 iteration 上限不代表收斂。

若要使用相同資料比較 MAP 與 MLE，重用相同的 `cfg.Batch` 和圖設定即可；兩者有相同的已記錄 trajectory／measurement seeds。

## 結果怎麼讀

每次執行建立 `results/METHOD_PROFILE_TIMESTAMP/`，包括：

- `results.mat`：變數 `Run`，包含 `Results`、完整 config、network、MATLAB 版本、執行時間和停止狀態。
- `summary.txt`：圖參數、lambda2、達到停止條件的步數、local solver 狀態。Batch 只記錄各步最後一次 local solve 的 exit flags；EKF 另外累計所有 local solve 的非正 exit flags。
- `adjacency.csv`、`edges.csv`：可直接重建圖；edge list 使用 MATLAB 的 1-based node indices。
- `radar_positions.csv`：無表頭的 N×2 座標矩陣，兩欄為 x、y（m）；第 n 列對應節點 n。
- `radar_layout.png`：雷達位置、通訊邊、完整參考軌跡與本次取用的前段。
- `trajectory.png`、`trajectory.pdf`：manuscript 樣式的 MC1 tracking 軌跡、sensor nodes 與 communication links。
- `rmse.png`、`rmse.pdf`：position／velocity RMSE 的兩列圖；EKF 有保存 PCRLB 時自動加入 assumed-CV root PCRLB。
- `overview.png`：網路、MC1 軌跡與所有 MC 的 RMSE。
- `convergence.png`：MC1 最後一個 tracking step 的殘差，以及各步 iteration 數。
- `truth_trajectory.png`：完整非直線 reference、實際取用的前段及逐時真實速度。

```matlab
saved = load(fullfile(Run.OutputFolder, 'results.mat'));
Results = saved.Run.Results;
```

EKF 啟用 PCRLB 後，`Results.PCRLB_Bayes` 為 `4 × 4 × K` 的每步 CV-PCRLB covariance（state 順序為 `[x, y, vx, vy]`）；`Results.CRLB{mc,t}` 保存同一份第 t 步矩陣。`Results.PCRLB_info` 記錄採樣數、seed、information matrices 與模型參數，`Results.PCRLB_status` 記錄 assumed-CV reference 標示。繪圖時會以黑色虛線加入 root PCRLB（assumed CV）：position 為 `sqrt(B(1,1,t)+B(2,2,t))`，velocity 為 `sqrt(B(3,3,t)+B(4,4,t))`。既有的 MAT 檔不會因開啟設定而自動補入 PCRLB。

若模擬已完成但繪圖中途報錯，可直接重用儲存的 MAT 檔，無需再次執行 estimator：

```matlab
Run = replot_tracking_results('results/EKF_original_20261013_173424_628/results.mat');
% 第二個參數設 true 可保留 MATLAB figure 視窗。
```

此 helper 在 MAT 檔所在資料夾重新輸出 PNG 與兩張 manuscript PDF，不改寫 `results.mat`。Plotter 支援舊版沒有 `Config.Radar` 的資料，使用儲存的雷達座標；未記錄的部署方式／position seed 不會以目前 defaults 補上。

RMSE 圖的分散式曲線使用**節點估計平均值**，不是各節點誤差平方的平均。原來的 `primal_residauls`／`dual_residauls` 拼字保留在 Results，方便對照舊程式；MAT 檔的頂層改為 `Run`，舊 plotting script 不能保證直接讀取。

## Manuscript 繪圖設定

共用設定集中在 `common/tracking_plot_config.m`，新實驗可透過 `cfg.Plot` 覆寫。軌跡圖沿用 manuscript Fig. 3–4 與原 `plot_trajectory.m` 的 Times New Roman、黑色 dash-dot truth、紅色虛線圓圈 D-estimator、藍色實線方框 C-estimator、紅色 sensor nodes 和黑色 communication links。每條估計曲線約顯示 20 個 marker；保留全部線段，沒有平滑或下採樣數據。座標範圍依本次軌跡及 radar positions 自動設定。

RMSE 配色／marker 沿用 Fig. 5 的 C-estimator 藍色實線圓圈與 D-estimator 橘色虛線菱形，縱軸使用 logarithmic scale。EKF 橫軸使用 measurement time；MAP/MLE 使用儲存的 batch-end time，舊檔缺少時間時回退為 tracking step。單次實驗輸出兩列 position／velocity RMSE，沒有合成額外 SNR 或鄰居數實驗。RMSE 仍是既有的「節點平均估計」誤差，未改成 node-wise RMSE。

預設輸出 300-dpi PNG，`trajectory.pdf` 與 `rmse.pdf` 為 vector PDF。`FigureSize` 等尺寸單位是 cm，字級／線寬單位是 point；放入 manuscript 時可依欄寬縮放。

```matlab
cfg = tracking_config("original");
cfg.Plot.FontSize = 20;
cfg.Plot.LabelFontSize = 24;
cfg.Plot.LegendFontSize = 16;
cfg.Plot.LineWidth = 2;
cfg.Plot.MarkerSize = 8;
cfg.Plot.NumMarkers = 20;
cfg.Plot.FigureSize = [20, 14.5];
cfg.Plot.RMSEYScale = 'log';      % 'linear' 可切回線性縱軸
cfg.Plot.ShowPCRLB = true;       % 只讀取已儲存的 bound，不重新計算
cfg.Plot.ExportPDF = true;
cfg.Plot.Resolution = 300;
```

重畫舊結果也可覆寫設定，不必重跑 estimator：

```matlab
plot_options = struct('FontSize', 18, 'NumMarkers', 15);
Run = replot_tracking_results('results/EKF_.../results.mat', false, plot_options);
```

## 保留的演算法假設與必要整理

D-MAP／D-MLE 現在使用每個 batch 固定的 **first-order affine measurement model**：`h(anchor) + H(anchor)*(x-anchor)`。首批 anchor 為 `cfg.Batch.InitialState`，之後採用各方法的 CV prediction；所有 DA 節點共用同一 anchor，ADMM 迭代內不重新線性化。`fmincon` 使用解析梯度。非線性真實軌跡與量測生成維持原設定；尚未加入 asynchronous ADMM。

**D-MAP／D-MLE**：量測 likelihood 同時修正為完整 covariance 的 `0.5*r'*(Sigma\r)`，Doppler 符號與 generator 一致，尺度保留 `1/lambda`。Whitening 同時轉換量測、模型值與 Jacobian，且不改動 state prior covariance。原本 prior 的係數、ADMM penalty／更新／停止條件仍保留，並未因此成為標準 Gaussian MAP／ADMM。Own-plus-neighbor objective 在不規則圖上重複計入量測，CA／DA 後續 anchors 也可能不同，兩者差異不能直接當成 ADMM optimization error。詳細公式與限制見 [all_in_one/README.md](all_in_one/README.md)。未驗證的舊 bound 預設不算（`ComputeLegacyBound=false`）。

**D-EKF 入口**：ADMM correction 現在也使用 **first-order affine measurement model**。每個 measurement update 開始時，以既有 common predicted mean 作所有節點共用的 anchor，建立 `h(anchor)+H(anchor)*(x-anchor)`；ADMM 迭代內固定。Centralized EKF 本來就以自身 prediction 做一階更新，現在與 DA 共用模型 builder。EKF 保留正號 `2/lambda` Doppler，與 batch 的 `1/lambda` 不同；完整 residual 與 Jacobian 只白化一次。狀態 prior、CV model、雷達位置和真實 nonlinear measurement generation 維持設定。

DA 的四個 objective 使用解析梯度；保留 `fmincon`、原本正確的 `0.5*delta'*diag(c)*delta` penalty、dual 更新與 primal/dual 停止條件。每個 radar measurement 以 `self_only` 計入一次，共同 prior 由各節點分攤 `1/N`。Posterior covariance 改用**與 objective 相同的固定 Jacobian**：`P_post = (P_common^-1 + sum(H_n'*Sigma^-1*H_n))^-1`；不再於求解後的狀態重新線性化。MLE 分支不加入 prior information。

`Results.MeasurementModel="first_order_affine"` 明確標記新結果；`linearization_DA_raw/CA_raw{mc,t}` 保存 physical raw `anchor/value/H`，`prior_mean_common_raw` 和 `prior_cov_common_raw` 保存當次 DA common prior。之後 CA 與 DA 的遞迴 prediction 可能稍有差異，所以應以**相同 saved prior／anchor 的二次型解**評估 ADMM optimization error。Common prior 的平均和 posterior information 的加總仍是全網路參考計算，尚未實作額外鄰居交換或 communication-round 計數。

因此，這份版本可用來調整網路與繼續修改程式；論文中的三方法公平比較、CRLB 解讀、收斂宣稱仍需後續統一模型與實驗條件。

## EKF 收斂與線性化測試

開啟 `run_convergence_test.m` 可用**目前一階模型**執行較長設定：128 個 measurement updates、2 個 Monte Carlo trials，每點最多 5000 次 ADMM iteration。它讀取 `tracking_config("original")`；若要測其他部署，可直接在此 entry 修改 `cfg.Radar` 或 `cfg.Graph`。`run_example.m` 裡的個別覆寫不會自動帶入。

此 entry 明確指定 `ADMMAbsTolerance=1e-5`、`ADMMRelTolerance=1e-5`，保留 CV model、Q 和 penalty 設定；一般 `ekf_default_config.m` 仍使用你設定的 `1e-3`。先前儲存的長測試使用 nonlinear correction，其歷史數字不代表現在的一階版本。`dt=0.01` 下，128 點涵蓋 **0–1.27 秒**；完整 30 秒參考路徑需要 3001 點。`NumDopplerSamples=64` 控制 Doppler 雜訊，不因 tracking 點數改變。

執行完成後，`summarize_ekf_convergence(Run)` 另外輸出：

- `convergence_metrics.csv`：每個 trial／update 的 iteration、停止狀態、primal／dual 殘差與門檻比值、節點間差異、位置／速度誤差及 local solver failure count。
- `convergence_diagnostics.png`：所有更新點的殘差比值、iteration 數、最慢更新點的迭代歷史，以及 tracking position RMSE。
- `convergence_summary.txt`：通過數、iteration min／median／max、實際模擬時間、模型設定和整個有限時間窗的誤差摘要。

**primal 和 dual 的「殘差／門檻」都 ≤ 1 才算通過。** 最多迭代上限、fmincon 成功旗標、節點達成一致與追蹤真值的誤差，是不同的指標。有限次數值測試不構成所有部署或整段 tracking trajectory 的收斂保證。

若要隔離 optimizer 的效果，可對**新的一階 EKF/MAP 結果**執行 `check_frozen_map_solution(Run)`。它固定 MC1 第一個更新點保存的量測、common prior 和一階模型，比較原結果、100 次 iteration 的 replay，以及同一 affine MAP objective 透過 normal equations 得到的解；輸出 `frozen_map_comparison.csv/.mat`。此 reference 與遞迴 C-EKF baseline 分開記錄。工具會在 replay／寫檔之前拒絕歷史 nonlinear 結果，避免用新 objective 錯誤解讀舊實驗；舊結果仍可使用 `replot_tracking_results` 繪圖。

`Report = run_ekf_linearization_test();` 執行公式檢查、MAP/MLE 與 whitening 分支、多步 CV 遞迴，以及嚴格門檻下第一個更新點的二次型 reference 比較。測試設定為 16 updates × 2 MC，保留 64 個 Doppler noise samples，涵蓋 0–0.15 s；這是實作驗證，不是完整 30 秒軌跡的效能結論。

## 檢查

```matlab
check_graph;  % 連通性、對稱性、L=B*B'、重現性及 RNG 隔離
check_ring_graph; % circle 上的最近兩鄰居、ring 矩陣與 random 相容性
check_radar_positions; % 均勻面積分布、圓周模式、seed 和範圍
check_trajectory; % 共同真值、端點、逐時 Doppler 和 prefix 重現性
check_trajectory_switch; % reference / nonlinear 與原公式、prefix、RNG 比對
check_batch_linearization; % 一階近似、四個 batch objective 梯度與 whitening
check_ekf_linearization;   % EKF 四個 objective、prior 分攤、central state/covariance 等價
% Report = run_batch_linearization_test(); % 儲存 MAP／MLE 測試及 local-solve benchmark
smoke_test;   % 三個方法、CV prediction、共同 reference、儲存／繪圖
```

交付驗證見 [results/VALIDATION.md](results/VALIDATION.md)。來源檔案指紋見 [SOURCE_PROVENANCE.txt](SOURCE_PROVENANCE.txt)。
