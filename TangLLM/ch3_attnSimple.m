% 简化版注意力的权重矩阵热力图

load('TrainedModel.mat','Parameters','vocabulary')
TokenEmbedding = Parameters.TokenEmbedding;
Poems = "欲得周郎顾，时时误拂弦。";
chars = char(Poems);
[~, ind] = ismember(chars, vocabulary);
X = TokenEmbedding(ind,:);

% 简化版注意力
[Y,W] = attnSimple(X);

% 权重矩阵热力图
figure(1)
imagesc(W);
axis image; colormap hot; colorbar;

labels = cellstr(chars(:));  % 每个字变成一个cell元素
set(gca, 'XTick', 1:numel(labels), 'XTickLabel', labels, ...
    'YTick', 1:numel(labels), 'YTickLabel', labels, ...
    'FontSize', 13);
xlabel('索引键（被关注的词）','FontSize',13);
ylabel('查询者','FontSize',13);


function [Y,W] = attnSimple(X)
% 简化的注意力模块

% 未归一化的权重分数
S = X*X';

% SoftMax形式的权重
expS = exp(S - max(S, [], 2));
W = expS./sum(expS,2);

% 对上下文做加权平均
Y = W*X;
end