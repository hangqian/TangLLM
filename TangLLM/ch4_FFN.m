% 前馈网络中的稀疏激活

% 导入数据
load('TrainedModel.mat','Parameters','vocabulary')

% 查词表
% Poems = "大漠孤烟直";
Poems = "欲穷千里目";
[~, ind] = ismember(char(Poems), vocabulary);

% 查嵌入表
TokenEmbedding = Parameters.TokenEmbedding;
X = TokenEmbedding(ind,:);

% 多头注意力
Y = attention(X,Parameters.Wq(:,:,1),Parameters.Wk(:,:,1),Parameters.Wv(:,:,1),Parameters.Wo(:,:,1),8);

% 前馈网络
[Result3,Result2,Result1] =  feedforward(Y,Parameters.W1(:,:,1), Parameters.b1(:,:,1), Parameters.W2(:,:,1), Parameters.b2(:,:,1));


% =====================
% 前馈网络 稀疏激活可视化
% =====================

% 提取第 5 个 Token（"目"）在最后一层 FFN 的中间状态
token_idx = 5; 
token_char = chars(token_idx);

% Result1: 升维后的未激活状态 (与 W1 的 4d 个 Keys 进行点乘的得分)
scores_pre = Result1(token_idx, :); 

% Result2: 激活后的状态 (经过 GELU 阈值过滤后的得分)
scores_post = Result2(token_idx, :); 
num_neurons = length(scores_post);
x_axis = 1:num_neurons;

% 创建高分辨率的宽屏绘图窗口
fig = figure('Name', sprintf('FFN 内部微观旅程："%s" 字的升维与过滤', token_char), ...
       'Position', [100, 80, 1100, 900], 'Color', 'w');

% 定义现代配色方案
color_pre   = [0.3010, 0.7450, 0.9330]; % 柔和蓝
color_post  = [0.4660, 0.6740, 0.1880]; % 活力绿
color_focus = [0.8500, 0.3250, 0.0980]; % 醒目橙红

% ---------------------------------------------------------
% 图 1. 升维阶段：与 W1 (Keys) 的匹配得分分布
% ---------------------------------------------------------
ax1 = subplot(3, 1, 1);
plot(x_axis, scores_pre, 'Color', color_pre, 'LineWidth', 1.2);
hold on;
yline(0, '-', 'Color', [0.8 0.2 0.2 0.6], 'LineWidth', 1.5); % 半透明红色基准线
title(sprintf('第一阶段：升维匹配，点乘得分'), ...
    'FontSize', 12, 'FontWeight', 'bold');
ylabel('匹配得分', 'FontSize', 10);
set(gca, 'TickDir', 'out', 'box', 'off');
grid on; grid minor;

% ---------------------------------------------------------
% 图 2. 激活阶段：GELU 阈值过滤后的稀疏状态
% ---------------------------------------------------------
ax2 = subplot(3, 1, 2);
% 使用 area 绘制面积图，增强"尖峰"的实体感
area(x_axis, scores_post, 'FaceColor', color_post, 'FaceAlpha', 0.2, ...
     'EdgeColor', color_post, 'LineWidth', 1.2);
title('第二阶段：激活过滤，稀疏模式', ...
    'FontSize', 12, 'FontWeight', 'bold');
ylabel('激活后得分', 'FontSize', 10);
set(gca, 'TickDir', 'out', 'box', 'off');
grid on; grid minor;

% ---------------------------------------------------------
% 图 3. 聚焦阶段：找出被最高度“点亮”的 Top 20 核心神经元
% ---------------------------------------------------------
ax3 = subplot(3, 1, 3);
[top_scores, top_indices] = maxk(scores_post, 20); % 找出得分最高的前 20 个

% 绘制火柴杆图 (Stem)
stem(top_indices, top_scores, 'filled', ...
    'Color', color_focus, 'MarkerFaceColor', color_focus, ...
    'MarkerEdgeColor', 'w', 'LineWidth', 1.5, 'MarkerSize', 8);
hold on;

title('第三阶段：降维压缩，核心触发', ...
    'FontSize', 12, 'FontWeight', 'bold');
xlabel('神经元编号', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('最高激活权重', 'FontSize', 10);
set(gca, 'TickDir', 'out', 'box', 'off');
grid on; grid minor;

% 在图上标注 Top 5 神经元的编号
for i = 1:5 
    text(top_indices(i), top_scores(i) + max(top_scores)*0.1, ...
        sprintf('#%d', top_indices(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', ...
        'Color', color_focus * 0.8, 'BackgroundColor', [1 1 1 0.7], 'Margin', 1);
end

% ---------------------------------------------------------
% 关键动作：链接所有 X 轴并强制统一对齐范围
% ---------------------------------------------------------
linkaxes([ax1, ax2, ax3], 'x');
xlim(ax1, [1, num_neurons]); % 只需设置 ax1，其他会自动同步

% ---------------------------------------------------------
% 打印终端统计信息
% ---------------------------------------------------------
fprintf('\n=== 第 %d 个 Token ("%s") 在 FFN 最后一层的微观统计 ===\n', token_idx, token_char);
fprintf('总模式检测器数量 (4d): %d\n', num_neurons);

% 统计静默比例 (由于 GELU 不完全等于 0，取极小值阈值)
silenced_ratio = mean(scores_post < 0.01) * 100;
fprintf('被激活函数完全/几乎静默（权重 < 0.01）的神经元比例: %.2f%%\n', silenced_ratio);


function [X3,X2,X1] = feedforward(X0, W1, b1, W2, b2)
% 前馈网络模块

% 线性层，神经元数量倍增
X1 = X0 * W1 + b1;

% GELU/RELU激活层
% X2 = max(0,X1);
X2 = X1 .* normcdf(X1);

% 线性层，神经元数量缩回
X3 = X2 * W2 + b2;

end



