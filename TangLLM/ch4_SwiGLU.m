% SwiGLU前馈网络中的稀疏激活


% 读取模型
load('TrainedModel.mat','Parameters','Specifications','vocabulary')
TokenEmbedding = Parameters.TokenEmbedding;
Poems = "蓬门今始为君开";
chars = char(Poems);
[~, ind] = ismember(chars, vocabulary);
X = TokenEmbedding(ind,:); 

N = size(X,1);
d = Specifications.d;
numHeads = Specifications.numHeads;
numLayers = Specifications.numLayers;
for L = 1:numLayers

    % 归一化    
    X0 = RMSNorm(X, Parameters.scale(L,:));

    % 注意力机制
    XAttention = attention(X0,Parameters.Wq(:,:,L),Parameters.Wk(:,:,L),Parameters.Wv(:,:,L),Parameters.Wo(:,:,L),numHeads);   

    % 残差网络：保留原输入信号，注意力模块提供增量
    X1 = X + XAttention;

    % 下面进入传统的前馈神经网络   
    X2 = RMSNorm(X1, Parameters.scale(numLayers+L,:));

    % 残差形式的前馈网络
    [Result3,Result2,Result1,G0,G] = SwiGLU(X2,Parameters.W0(:,:,L), Parameters.b0(:,:,L),Parameters.W1(:,:,L), Parameters.b1(:,:,L), Parameters.W2(:,:,L), Parameters.b2(:,:,L));    
    X3 = X1 + Result3;

    disp(mean(G(7,:)<0.01))

    % 本层输出更新为下一层的输入！
    X = X3;
end

% ===============================
% 揭秘 SwiGLU：动态草拟与门控可视化
% ===============================

% 聚焦于句末最具信息量的字："开"
token_idx = 7; 
token_char = chars(token_idx);

% 提取核心中间变量
% 1. 动态草稿 X1 (W0 的输出，代表不计成本生成的全量候选知识)
drafts = Result1(token_idx, :); 

% 2. 门控确信度 G (SiLU 激活后的权重，代表过滤器的放行标准)
gates = G(token_idx, :); 

% 3. 幸存知识 X2 (X1 与 G 逐元素相乘，代表最终真正生效的特征)
survivors = Result2(token_idx, :); 

num_neurons = length(drafts);
x_axis = 1:num_neurons;

% 创建绘图窗口
fig = figure('Name', sprintf('SwiGLU 微观机制："%s" 字的草拟与门控', token_char), ...
       'Position', [100, 80, 1100, 900], 'Color', 'w');

color_draft = [0.4940, 0.1840, 0.5560]; % 紫色 (代表丰富的草稿)
color_gate  = [0.9290, 0.6940, 0.1250]; % 金色 (代表守门人的权力)
color_surv  = [0.8500, 0.3250, 0.0980]; % 橙红 (代表最终突围的特征)

% ---------------------------------------------------------
% 图 1. 动态草拟阶段 (X1) —— 稠密且充满细节的草稿库
% ---------------------------------------------------------
ax1 = subplot(3, 1, 1);
plot(x_axis, drafts, 'Color', color_draft, 'LineWidth', 1.2);
hold on; yline(0, 'k:', 'LineWidth', 1);
title(sprintf('第一阶段：动态草拟 (X1) —— W0 为 "%s" 字算出的全量候选知识', token_char), ...
    'FontSize', 12, 'FontWeight', 'bold');
ylabel('草稿强度', 'FontSize', 10);
set(gca, 'TickDir', 'out', 'box', 'off');
grid on; grid minor;

% ---------------------------------------------------------
% 图 2. 门控过滤阶段 (G) —— 极其稀疏的激活权重 (SiLU输出)
% ---------------------------------------------------------
ax2 = subplot(3, 1, 2);
area(x_axis, gates, 'FaceColor', color_gate, 'FaceAlpha', 0.4, ...
     'EdgeColor', color_gate, 'LineWidth', 1.2);
title('第二阶段：门控审核 (G) —— 守门人严苛，绝大部分被压制在 0 附近', ...
    'FontSize', 12, 'FontWeight', 'bold');
ylabel('门控放行权重', 'FontSize', 10);
set(gca, 'TickDir', 'out', 'box', 'off');
grid on; grid minor;

% ---------------------------------------------------------
% 图 3. 幸存突围阶段 (X2) —— 最终保留的知识切片
% ---------------------------------------------------------
ax3 = subplot(3, 1, 3);
[top_scores, top_indices] = maxk(abs(survivors), 20); % 找出绝对值最大的前20个

% 使用火柴杆图展示极其稀疏的最终幸存者
stem(x_axis, survivors, 'Marker', 'none', 'Color', [color_surv 0.3], 'LineWidth', 1);
hold on;
stem(top_indices, survivors(top_indices), 'filled', ...
    'Color', color_surv, 'MarkerFaceColor', color_surv, ...
    'MarkerEdgeColor', 'w', 'LineWidth', 1.5, 'MarkerSize', 8);
yline(0, 'k-', 'LineWidth', 1);

title('第三阶段：幸存突围 (X2 = X1 .* G) —— 被无情销毁后，真正改变词义的少数定海神针', ...
    'FontSize', 12, 'FontWeight', 'bold');
xlabel('神经元编号', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('最终融合特征强度', 'FontSize', 10);
set(gca, 'TickDir', 'out', 'box', 'off');
grid on; grid minor;

% 标注 Top 5 幸存特征
for i = 1:5 
    idx = top_indices(i);
    val = survivors(idx);
    offset = sign(val) * max(abs(survivors))*0.1; % 根据正负决定标签位置
    text(idx, val + offset, sprintf('#%d', idx), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', ...
        'Color', color_surv * 0.8, 'BackgroundColor', [1 1 1 0.7], 'Margin', 1);
end

% 链接 X 轴，方便对比查阅
linkaxes([ax1, ax2, ax3], 'x');
xlim(ax1, [1, num_neurons]);

% 终端打印“杀伤率”统计
fprintf('\n=== SwiGLU 对 "%s" 字的统计 ===\n', token_char);
fprintf('总特征维度数: %d\n', num_neurons);
% 定义门控接近 0 为被“销毁”
destroyed_ratio = mean(gates < 0.05) * 100;
fprintf('被门控无情销毁 (权重 < 0.05) 的草稿比例: %.2f%%\n', destroyed_ratio);
fprintf('即便第一张图(X1)波涛汹涌，最终真正奏效的只有第三张图的几个尖峰！\n');


function [X3,X2,X1,G0,G] = SwiGLU(X0, W0, b0, W1, b1, W2, b2)
% 门控SwiGLU网络

% 值
X1 = X0 * W0 + b0;

% 门: Swish 激活函数
G0 = X0 * W1 + b1;
G = G0 .* logitCDF(G0);

% 门控乘法
X2 = X1 .* G;

% 线性层
X3 = X2 * W2 + b2;

end

function y = logitCDF(x)
y = 1 ./ (1+exp(-x));
end


