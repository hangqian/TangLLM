% 交叉熵损失可视化：贾岛“推敲”之境
clear; clc; close all;

% 1. 定义词表与位置
vocab = {'归', '行', '推', '眠', '知', '敲', '吟', '寻'};
num_words = length(vocab);
idx_tui = 3; % '推'的索引
idx_qiao = 6; % '敲'的索引

% 2. 假设模型输出的预测概率分布 Q (已过Softmax)
% 赋予'推'和'敲'较高的概率，其余分配给其他词
Q = [0.1, 0.08, 0.15, 0.05, 0.03, 0.2, 0.08, 0.06]; 
% 确保概率和为1
Q = Q / sum(Q); 

% 3. 定义两种假设下的真实分布 P (One-hot)
P_tui = zeros(1, num_words); P_tui(idx_tui) = 1;
P_qiao = zeros(1, num_words); P_qiao(idx_qiao) = 1;

% 4. 计算交叉熵 Loss = -sum(P .* log(Q))
% 数值上等价于负对数似然 NLL = -log(Q(true_idx))
loss_tui = -log(Q(idx_tui));
loss_qiao = -log(Q(idx_qiao));

% 打印数值验证
fprintf('--- 数值计算验证 ---\n');
fprintf('假设真实为【推】: 交叉熵公式计算=%.4f, 负对数似然=%.4f\n', -sum(P_tui .* log(Q)), loss_tui);
fprintf('假设真实为【敲】: 交叉熵公式计算=%.4f, 负对数似然=%.4f\n', -sum(P_qiao .* log(Q)), loss_qiao);

% 5. 可视化设置
figure('Name', '推敲之境：交叉熵损失可视化', 'Position', [100, 100, 1000, 700], 'Color', 'w');
x_curve = linspace(0.01, 1, 100);
y_curve = -log(x_curve);
color_true = [0.2 0.6 0.8]; % 蓝色表示真实
color_pred = [0.9 0.5 0.1]; % 橙色表示预测

%% 左列：假设真实标签为“推”
% 图 1: 概率分布对比 (推)
subplot(2, 2, 1);
b1 = bar(1:num_words, [P_tui; Q]', 'grouped');
b1(1).FaceColor = color_true; b1(2).FaceColor = color_pred;
set(gca, 'XTick', 1:num_words, 'XTickLabel', vocab, 'FontName', 'Microsoft YaHei', 'FontSize', 11);
title('概率分布对比 (真实标签：推)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('概率');
legend({'真实分布 P', '预测分布 Q'}, 'Location', 'northeast');
grid on; set(gca, 'GridAlpha', 0.15);

% 图 3: 对数损失曲线 (推)
subplot(2, 2, 3);
plot(x_curve, y_curve, 'k-', 'LineWidth', 1.5); hold on;
plot(Q(idx_tui), loss_tui, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
plot([Q(idx_tui) Q(idx_tui)], [0 loss_tui], 'r--', 'LineWidth', 1);
plot([0 Q(idx_tui)], [loss_tui loss_tui], 'r--', 'LineWidth', 1);
title(sprintf('交叉熵曲线 (损失 = %.3f)', loss_tui), 'FontSize', 12, 'FontWeight', 'bold');
xlabel('预测概率'); ylabel('交叉熵损失');
xlim([0 1]); ylim([0 5]);
grid on; set(gca, 'GridAlpha', 0.15);

%% 右列：假设真实标签为“敲”
% 图 2: 概率分布对比 (敲)
subplot(2, 2, 2);
b2 = bar(1:num_words, [P_qiao; Q]', 'grouped');
b2(1).FaceColor = color_true; b2(2).FaceColor = color_pred;
set(gca, 'XTick', 1:num_words, 'XTickLabel', vocab, 'FontName', 'Microsoft YaHei', 'FontSize', 11);
title('概率分布对比 (真实标签：敲)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('概率');
legend({'真实分布 P', '预测分布 Q'}, 'Location', 'northwest');
grid on; set(gca, 'GridAlpha', 0.15);

% 图 4: 对数损失曲线 (敲)
subplot(2, 2, 4);
plot(x_curve, y_curve, 'k-', 'LineWidth', 1.5); hold on;
plot(Q(idx_qiao), loss_qiao, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
plot([Q(idx_qiao) Q(idx_qiao)], [0 loss_qiao], 'r--', 'LineWidth', 1);
plot([0 Q(idx_qiao)], [loss_qiao loss_qiao], 'r--', 'LineWidth', 1);
title(sprintf('交叉熵曲线 (损失 = %.3f)', loss_qiao), 'FontSize', 12, 'FontWeight', 'bold');
xlabel('预测概率'); ylabel('交叉熵损失');
xlim([0 1]); ylim([0 5]);
grid on; set(gca, 'GridAlpha', 0.15);

% 整体标题
% sgtitle('大语言模型输出层：从“推敲”理解交叉熵与极大似然', 'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Microsoft YaHei');


% 从词元到嵌入
load('TrainedModel.mat','Parameters','Specifications','vocabulary')
TokenEmbedding = Parameters.TokenEmbedding;
Poems = "鸟宿池边树，僧";
chars = char(Poems);
[~, ind] = ismember(chars, vocabulary);
X = TokenEmbedding(ind,:); 

% 向前传播
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
    Result3 = SwiGLU(X2,Parameters.W0(:,:,L), Parameters.b0(:,:,L),Parameters.W1(:,:,L), Parameters.b1(:,:,L), Parameters.W2(:,:,L), Parameters.b2(:,:,L));    
    X3 = X1 + Result3;   

    % 本层输出更新为下一层的输入！
    X = X3;
end

% 进入输出层之前再做一次归一化，防止残差网络方差膨胀
X3Norm = RMSNorm(X3, Parameters.scale(2*numLayers+1,:));

% 输出层（神经元数量扩展到三千词表量）
% 对词表中每个词评分（预测词与词表中各词的相关性）
Score = X3Norm * Parameters.W + Parameters.b;

% 多元逻辑斯蒂模型: Softmax概率分布
logProb = Score - max(Score, [], 2);
ProbRaw = exp(logProb);
Prob = ProbRaw ./ sum(ProbRaw, 2);

% 最后一行是对下一个字的预测概率
prob = Prob(end,:);

% 最高概率的十个预测词
[p,ind] = maxk(prob,10);
words = vocabulary(ind);
disp(array2table(p','rowNames',string(words')));

% 似然函数
[~, TargetLabel] = ismember(char("宿池边树，僧敲"), vocabulary);
[N,v] = size(Prob);
linearIdx = sub2ind([N, v], (1:N), TargetLabel); 
targetProb = Prob(linearIdx); 
loss = sum(-log(targetProb)+1e-9)/N;
