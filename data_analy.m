clc;clear;close all;

function plot_state_mse_from_json(files,mode)
    % colors_size = size(files, 1);
    % colors = hsv(colors_size);
    legendNames = cell(1, size(files, 1)); 
    labels_params = cell({'X','Y','V_x','V_y'});
    % Construct the legend names based on the rows of direction_mc
    for i = 1:size(files, 1)
            % Format the direction values to two decimal places
            [folder, ~, ~]= fileparts(files{i});
            [~, folder_name] = fileparts(folder);
            legendNames{i} = ['Run-', folder_name];
    end
    figure('Color','w');
    
    for i = 1:numel(files)
        S = jsondecode(fileread(files{i}));
        
        true_params_mc = S.true_params_mc';
        switch lower(mode)
            case 'one'
            display_node = 1; % Change this to select different node for display
            all_estimations_every_iter_mc = cell({{squeeze(S.all_estimations_every_iter_mc)}});
            case 'mean'
            all_estimations_every_iter_mc = cell({{squeeze(mean(squeeze(S.all_estimations_every_iter_mc),2))}});
        end
        param_names = {'x','y','vx','vy'};
        param_labels = {'Position X Error','Position Y Error','Velocity X Error','Velocity Y Error'};
        for param = 1:4
            subplot(2,2,param);
            hold on; grid on;
            set(gca,'FontName','Times New Roman','FontSize',14,'LineWidth',1.0);
            % Initialize a cell array to store plot handles
            hPlots = cell(1, length(all_estimations_every_iter_mc)); % +2 for centralized approach and true parameter error line

            % Plot estimation errors with specific color and store handles
            for j = 1:length(all_estimations_every_iter_mc)
                % Calculate estimation error as estimation minjus true parameter
                switch lower(mode)
                    case 'one'
                    errors = (squeeze(all_estimations_every_iter_mc{j}{:}(param, display_node, :)) - true_params_mc(j,param)).^2;
                    case 'mean'
                    errors = (squeeze(all_estimations_every_iter_mc{j}{:}(param, :)) - true_params_mc(j,param)).^2;
                end
                % hPlots{j} = semilogy(errors, 'Color', colors(i, :));
                hPlots{j} = semilogy(errors,'LineWidth',1.5,'DisplayName',legendNames{i});
            end
            set(gca, 'YScale','log');
            hold off; 
            xlabel('Iterations','FontSize',16);
            legend('FontSize',11)
            % legend([hPlots{:}], legendNames{i});
            % legend(legendNames{:});
            ylabel(['MSE ' labels_params{param}],'FontSize',16);
            % title(['MSE in ' obj.labels_params{param}]);
        end
    end
    sgtitle('MSE by States','fontsize',20,'FontName','Times New Roman', 'FontWeight','bold');
    
end
function plot_mse_from_json(files, varargin)
% PLOT_MSE_FROM_JSON
% 用途：
%   - (預設) 以 all_estimations_every_iter_mc + ref 計算並繪製 MSE 隨迭代曲線
%   - (選項) 直接繪製 primal/dual 殘差隨迭代曲線
%
% 語法：
%   plot_mse_from_json(files, 'Ref', [x;y;vx;vy], 'Labels', {...}, 'Mode','mse', ...)
%   plot_mse_from_json(files, 'Mode','residual','ResidualType','primal', ...)
%
% 參數：
%   files        : 字串或字串cell陣列，JSON 檔路徑
%
% 名稱參數（可選）：
%   'Ref'        : 1x4 或 4x1 參考向量（用於 Mode='mse'），預設 [] 表示嘗試自資料推斷
%   'Mode'       : 'mse' (預設) 或 'residual'
%   'ResidualType': 'primal' 或 'dual'（Mode='residual' 時生效）
%   'Labels'     : 曲線標籤 cell，數量需與 files 相同
%   'YScale'     : 'linear' | 'semilog' | 'loglog'，預設 'semilog'
%   'IterRange'  : [iStart iEnd]，迭代截取範圍（1-based）
%   'MovingAvg'  : 平滑視窗長度（正整數；0=不平滑），預設 0
%   'LW'         : 線寬，預設 1.8
%   'MS'         : Marker size，預設 4
%
% 輸出：
%   圖形直接畫在目前 figure。也會將每條曲線的 y 值回傳於 base workspace（變數名：last_curves）
%
% 備註：
%   - JSON 應含下列之一：
%       (a) all_estimations_every_iter_mc  → 用 Ref 算 MSE@iter
%       (b) primal_residual_mc / dual_residual_mc → 直接畫殘差
%
% 作者：你最稱職的小助手 🍵

% ---------- 解析參數 ----------
p = inputParser;
addRequired(p, 'files', @(x) ischar(x) || isstring(x) || iscellstr(x) || isstring(x));
addParameter(p, 'Ref', [], @(x) isempty(x) || (isnumeric(x) && numel(x)==4));
addParameter(p, 'Mode', 'mse', @(x) any(strcmpi(x, {'mse','residual'})));
addParameter(p, 'ResidualType', '', @(x) any(strcmpi(x, {'primal','dual'})));
addParameter(p, 'Labels', [], @(x) iscell(x) || isempty(x));
addParameter(p, 'YScale', 'semilog', @(x) any(strcmpi(x, {'linear','semilog','loglog'})));
addParameter(p, 'IterRange', [], @(x) isempty(x) || (isnumeric(x) && numel(x)==2));
addParameter(p, 'MovingAvg', 0, @(x) isnumeric(x) && isscalar(x) && x>=0);
addParameter(p, 'LW', 1.8, @isscalar);
addParameter(p, 'MS', 4, @isscalar);
parse(p, files, varargin{:});
opt = p.Results;

% 正規化 files
if ischar(files) || isstring(files)
    files = {char(files)};
else
    files = cellstr(files);
end
nF = numel(files);

if isempty(opt.Labels), opt.Labels = arrayfun(@(i) sprintf('Run %d', i), 1:nF, 'uni',0); end
assert(numel(opt.Labels)==nF, 'Labels 數量需與 files 相同');

% ---------- 繪圖設定 ----------
figure('Color','w'); hold on; grid on;
set(gca,'FontName','Times New Roman','FontSize',22,'LineWidth',1.0);
xlabel('Iteration','FontSize',20);
switch lower(opt.Mode)
    case 'mse',      ylabel('MSE','FontSize',20);
    case 'residual', ylabel('Residual','FontSize',20);
end

switch lower(opt.YScale)
    case 'linear'
        set(gca,'YScale','linear');
    case 'semilog'
        set(gca,'YScale','log');
    case 'loglog'
        set(gca,'XScale','log'); set(gca,'YScale','log');
end

curves = cell(1,nF);

% ---------- 主迴圈：讀檔並產生曲線 ----------
for i = 1:nF
    S = jsondecode(fileread(files{i}));

    switch lower(opt.Mode)
        case 'residual'
            switch lower(opt.ResidualType)
                case 'primal'
                    % 取第一組 primal 殘差序列
                    y = sqrt(first_numeric_row(S, 'primal_residual_mc'));
                case 'dual'
                    y = sqrt(first_numeric_row(S, 'dual_residual_mc'));
            end

        case 'mse'
            % 用 all_estimations_every_iter_mc + Ref 計算 MSE@iter
            assert(isfield(S,'all_estimations_every_iter_mc'), ...
                'JSON 缺少 all_estimations_every_iter_mc，無法計算 MSE（請改用 Mode=''residual''）');

            % 取得 [node × state × iter] 或其他巢狀結構 → 均攤到 (state,iter) 維度
            A = S.all_estimations_every_iter_mc;  % 巢狀 cell
            % 將巢狀陣列展開成 cell: 每個 cell 是 1×T 的「某一 state@某節點」的序列
            % The original shape of all_estimations_every_iter_mc is 4 x num_nodes x num_iters of ADMM (for one localization)
            % Here we  first calculate the mean estimation cross node, so become 4 x num_iters 
            % A = squeeze(mean(squeeze(A),2));
            
            A = squeeze(A);
            mean_A = squeeze(mean(squeeze(A),2));
            seqs = mat2cell(A,ones(size(A,1),1),size(A,2),size(A,3));
            mean_seqs = mat2cell(mean_A, ones(size(mean_A,1),1),size(mean_A,2)); % each cell is 1 x T
            % 決定參考 ref
            % 優先順序：使用者給的 Ref > JSON 的 true_params > 退而求其次用首迭代均值
            ref = opt.Ref;

            % 決定群組數（每 4 條序列視為一組：x,y,vx,vy）
            nSeq = numel(mean_seqs);
            nNode = size(seqs{1},2);
            T = numel(mean_seqs{1});
            assert(mod(nSeq,4)==0, '序列數非 4 的倍數，請依實際資料調整配對規則');
            nGroup = nSeq/4;

            if isempty(ref)
                [ref_mat, has_group_refs] = extract_true_params(S, nGroup);  % ← 新增的工具函式
                if has_group_refs
                    % 每個群組都有一組 4×1 的參考
                    % 直接使用 ref_mat(:, g)
                elseif ~isempty(ref_mat)
                    % 單一 4×1 參考，所有群組共用
                    ref = ref_mat(:);
                else
                    % 最後的保險：用首迭代均值當粗略參考
                    firsts = cellfun(@(v) v(1), seqs);
                    ref_scalar = mean(firsts,'omitnan');
                    ref = repmat(ref_scalar,4,1);
                end
            else
                ref = ref(:); % 強制成 4×1
            end

            % 計算每次迭代的 MSE（跨所有群組與 4 維狀態平均）
            sqerr_sum = zeros(1,T);
            for g = 1:nGroup
                if exist('ref_mat','var') && ~isempty(ref_mat) && size(ref_mat,2)==nGroup
                    ref_g = ref_mat(:,g);
                else
                    ref_g = ref;  % 單一參考（4×1）
                end
                for d = 1:4
                    idx = (g-1)*4 + d;
                    err = seqs{idx} - ref_g(d);
                    % sqerr_sum = sqerr_sum + (err.^2);
                    sqerr_sum = sqerr_sum + (1/nNode)*squeeze(sum(err.^2,2))';
                end
            end
            y = sqerr_sum / (nGroup*4);
    end

    % 迭代截取
    if ~isempty(opt.IterRange)
        i1 = max(1, opt.IterRange(1));
        i2 = min(numel(y), opt.IterRange(2));
        y = y(i1:i2);
        x = i1:i2;
    else
        x = 1:numel(y);
    end

    % 平滑
    if opt.MovingAvg > 0
        w = opt.MovingAvg;
        y = movmean(y, w, 'omitnan');
    end

    % 畫線
    plt = plot(x, y, 'LineWidth', opt.LW, 'Marker', '.', 'MarkerSize', opt.MS, 'DisplayName', opt.Labels{i});
    curves{i} = y; %#ok<AGROW>


    switch lower(opt.Mode)
    case 'mse'
        % Help me to write a more elegant code to plot the centralized MSE line in the for loop, with changing color and legend 
        ctrl_est = S.estimates_mc_CA;
        ctrl_mse = ctrl_est - S.true_params_mc;
        ctrl_mse = mean(ctrl_mse.^2, 'all');
        yline(ctrl_mse,'LineWidth',1.5,'Color',plt.Color,'LineStyle','--','DisplayName',append(S.TYPE,' (Centr.)'));
        % legend([opt.Labels,append(S.TYPE,'(Centralized)')], 'Location','best', 'Interpreter','latex', 'Box','on');

    otherwise
        % legend(opt.Labels, 'Location','best', 'Interpreter','latex', 'Box','on');
    end
end

legend('Location','best', 'Interpreter','latex', 'Box','on');
title(sprintf('Metric vs Iteration (%s %s)',lower(opt.ResidualType), lower(opt.Mode)),'FontSize',20);

% 把數據丟到 base 方便你存取
assignin('base','last_curves',curves);

end

% --------- 小工具：抓第一列數值序列（如 primal_residual_mc{1}） ----------
function y = first_numeric_row(S, fieldname)
    assert(isfield(S, fieldname), 'JSON 缺少欄位 %s', fieldname);
    C = S.(fieldname);
    % 結構通常是 { [1×T double] ; [1×T double] ; ... } 的 cell 巢狀
    % 取第一個向量
    y = [];
    if iscell(C)
        v = C{1};
        if iscell(v), v = v{1}; end
        y = double(v(:)).';  % row
    else
        % 直接是數值陣列
        y = double(C(1,:));
    end
    assert(~isempty(y), '%s 內容無法解析為數列', fieldname);
end

% --------- 小工具：把巢狀估計序列展開成 {1×T double, ...} ----------
% --------- 取代舊的 flatten_iter_sequences(A) ----------
function seqs = flatten_iter_sequences(A)
    % 將巢狀 cell/數值陣列展平成 {1×T double, ...}
    A = num2cell(A);
    seqs = {};
    function visit(x)
        if iscell(x)
            for ii = 1:numel(x)
                visit(x{ii});
            end
        elseif isnumeric(x)
            if isvector(x) && numel(x) >= 2
                % 1×T 或 T×1 視為一條序列
                seqs{end+1} = double(x(:)).'; %#ok<AGROW>
            elseif ismatrix(x) && all(size(x) >= 2)
                % 若拿到 2D 矩陣（例如 10×T），將每一列當作一條序列
                [r,c] = size(x);
                if c >= 2
                    for rr = 1:r
                        seqs{end+1} = double(x(rr,:)); %#ok<AGROW>
                    end
                elseif r >= 2
                    % 或者每一行
                    for cc = 1:c
                        seqs{end+1} = double(x(:,cc)).'; %#ok<AGROW>
                    end
                end
            end
        end
    end
    visit(A);

    % 對齊長度（取最小 T），避免有些序列較短
    if ~isempty(seqs)
        lens = cellfun(@numel, seqs);
        T = min(lens);
        seqs = cellfun(@(v) v(1:T), seqs, 'uni', 0);
    end
end


function [ref_mat, has_group_refs] = extract_true_params(S, nGroup)
% 從 JSON 結構 S 中彈性解析 true_params
% 輸出：
%   ref_mat        : 若是「一組」參考，為 4×1；若是「每群組一組」，為 4×nGroup
%   has_group_refs : 若 true，表示 ref_mat 提供每群組（節點）各自的 4×1 參考
    ref_mat = [];
    has_group_refs = false;

    if ~isfield(S,'true_params_mc'), return; end
    TP = S.true_params_mc;

    % Case A: 直接是數值向量/矩陣
    if isnumeric(TP)
        if numel(TP)==4
            ref_mat = reshape(double(TP),[4,1]);
            return;
        end
        % 允許 4×n 或 n×4（取 4×n）
        [r,c] = size(TP);
        if r==4 && c>=1
            ref_mat = double(TP);
            has_group_refs = (c==nGroup);
            return;
        elseif c==4 && r>=1
            ref_mat = double(TP.').';
            % 這行其實等同於轉為 4×r；更直白：
            ref_mat = double(TP.').';  %#ok<NASGU>
            ref_mat = double(TP.').';  % 安全起見保持 4×r
            ref_mat = double(TP.');    % ← 轉成 4×r
            has_group_refs = (size(ref_mat,2)==nGroup);
            return;
        end
    end

    % Case B: cell：{4×1} 或 {nGroup×1，每個都是 4×1}
    if iscell(TP)
        % 扁平
        flat = {};
        stack = {TP};
        while ~isempty(stack)
            x = stack{end}; stack(end) = [];
            if iscell(x)
                stack = [stack, x]; %#ok<AGROW>
            else
                flat{end+1} = x; %#ok<AGROW>
            end
        end
        % 收集所有 4 元數值向量
        vecs = {};
        for i=1:numel(flat)
            v = flat{i};
            if isnumeric(v) && numel(v)==4
                vecs{end+1} = reshape(double(v),[4,1]); %#ok<AGROW>
            end
        end
        if ~isempty(vecs)
            if numel(vecs)==1
                ref_mat = vecs{1};
                return;
            else
                % 多組參考 → 盡量湊成 4×nGroup
                M = min(numel(vecs), nGroup);
                ref_mat = zeros(4,M);
                for k=1:M, ref_mat(:,k) = vecs{k}; end
                has_group_refs = (M==nGroup);
                return;
            end
        end
    end

    % Case C: struct：有欄位 x,y,vx,vy（不分大小寫）
    if isstruct(TP)
        f = fieldnames(TP);
        % 嘗試讀單一組
        keys = lower(f);
        need = {'x','y','vx','vy'};
        if all(ismember(need, keys))
            ref_mat = [TP.x; TP.y; TP.vx; TP.vy];
            ref_mat = double(ref_mat(:));
            ref_mat = reshape(ref_mat,[4,1]);
            return;
        end
        % 多組的情況（例如 TP(g).x ...）
        try
            ref_mat = zeros(4,nGroup);
            for g = 1:nGroup
                rg = [TP(g).x; TP(g).y; TP(g).vx; TP(g).vy];
                ref_mat(:,g) = double(rg(:));
            end
            has_group_refs = true;
            return;
        catch
            % 忽略，回傳空
        end
    end
end



%------ Main Script Example ------
% files = {
%     './data_log/MLE/data_config.json'
%     './data_log/MAP/data_config.json'
%     './data_log/MLE_w/data_config.json'
%     './data_log/MAP_w/data_config.json'
% };
files = {
    './data_log/MLE/data_config.json'
    './data_log/MAP/data_config.json'
};
% ref = [1000; 1000; -14.141; 14.141]; % 例如狀態真值（請換成你的 4×1）
% plot_mse_from_json(files, 'Ref', ref, ...
%     'Mode','residual', 'Labels',{'MLE','MAP'}, ...
%     'YScale','semilog', 'MovingAvg',3, 'LW',2.0);
% plot_mse_from_json(files, 'Ref', ref,...
%     'Mode','mse', 'Labels',{'MLE','MAP'}, ...
%     'YScale','semilog', 'MovingAvg',3, 'LW',2.0);

% Overall MSE
plot_mse_from_json(files,...
    'Mode','mse', 'Labels',{'MLE (Decentr.)','MAP (Decentr.)'}, ...
    'YScale','semilog', 'MovingAvg',3, 'LW',2.0);

% % Primal residual
% plot_mse_from_json(files, ...
%     'Mode','residual', 'ResidualType','primal',...
%     'Labels',{'MLE','MAP'}, ...
%     'YScale','semilog', 'MovingAvg',3, 'LW',2.0);
% 
% % Dual residual
% plot_mse_from_json(files, ...
%     'Mode','residual', 'ResidualType','dual',...
%     'Labels',{'MLE','MAP'}, ...
%     'YScale','semilog', 'MovingAvg',3, 'LW',2.0);
% 
% % State MSE
% plot_state_mse_from_json(files, 'mean');
