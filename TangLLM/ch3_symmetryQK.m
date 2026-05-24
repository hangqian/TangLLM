% 示例：用QK打破X*X'对称性

% 导入数据
load('TrainedModel.mat','Parameters','vocabulary')

% 查词表
Poems = "大漠孤烟直";
[~, ind] = ismember(char(Poems), vocabulary);

% 查嵌入表
TokenEmbedding = Parameters.TokenEmbedding;
X = TokenEmbedding(ind,:);

% QK线性变换
Q = X * Parameters.Wq(:,:,end);
K = X * Parameters.Wk(:,:,end);

% 点积相关性
d = size(Q,2);
S = Q*K'/sqrt(d);
XX = X*X'/sqrt(d);


% --------- 颜色范围（让对比更公平）---------
% S 和 A 用“以0为中心”的对称色轴
sMax = max(abs(S(:)));


% --------- 画图 ---------
fig = figure('Color','w','Position',[100 100 1150 380]);
t = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');

% (1) XX：用单色系即可（强调“对称”）
labels = num2cell(char(Poems));
nexttile(t,1);
plotHeat(XX, labels, 'X X^T / \surd d  (对称)', 'parula', []);
axis square;

% (2) S：发散色，中心0
nexttile(t,2);
plotHeat(S, labels, 'Q K^T / \surd d  (非对称)', 'redblue', [-sMax sMax]);
axis square;


% ==========================
function plotHeat(M, labels, ttl, cmapName, climRange)
% 辅助：画单张热力图 + 标数值 + colorbar

    imagesc(M);
    ax = gca;
    ax.XTick = 1:numel(labels);
    ax.YTick = 1:numel(labels);
    ax.XTickLabel = labels;
    ax.YTickLabel = labels;
    ax.TickLength = [0 0];
    ax.FontSize = 12;
    ax.LineWidth = 1;

    title(ttl, 'FontSize', 13);

    % 网格线（出版更清晰）
    hold on;
    n = size(M,1);
    for k = 0.5:1:n+0.5
        plot([0.5 n+0.5],[k k], 'Color',[1 1 1]*0.9, 'LineWidth',1);
        plot([k k],[0.5 n+0.5], 'Color',[1 1 1]*0.9, 'LineWidth',1);
    end
    hold off;

    % 颜色映射
    if strcmpi(cmapName,'redblue')
        colormap(redblueCmap(256));
    else
        colormap(cmapName);
    end
    colorbar;

    % 统一色轴范围（可选）
    if ~isempty(climRange)
        caxis(climRange);
    end

    % 在格子里写数值（保留3位小数）
    addCellText(M);
end

% ==========================
function addCellText(M)
% 在热力图格子中央写数值：颜色根据背景深浅自动切换

    ax = gca;
    clim = ax.CLim;
    n = size(M,1);
    for i = 1:n
        for j = 1:n
            v = M(i,j);
            % 简单按归一化亮度选文字色
            if (v - clim(1)) / (clim(2)-clim(1) + eps) > 0.65
                tc = 'w';
            else
                tc = 'k';
            end
            text(j, i, sprintf('%.3f', v), ...
                'HorizontalAlignment','center', ...
                'VerticalAlignment','middle', ...
                'FontSize', 9, ...
                'Color', tc);
        end
    end
end

% ==========================
function cmap = redblueCmap(m)
% 自定义红蓝发散色图（中心接近白）
    if nargin<1; m = 256; end
    r = [(0:m-1)'/(m-1), zeros(m,1), flipud((0:m-1)'/(m-1))];
    % r 是 [x,0,1-x] 风格，略显紫；改成更“红-白-蓝”
    x = linspace(0,1,m)';
    cmap = [x, 1-abs(2*x-1), flipud(x)];   % 蓝↔白↔红 的一种简单构造
end