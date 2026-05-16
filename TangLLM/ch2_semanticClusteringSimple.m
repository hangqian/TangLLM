% 在二维平面上展示高维词嵌入向量的语义聚类（简化版）

% 导入模型参数
load('TrainedModel.mat','Parameters','vocabulary')
TokenEmbedding = Parameters.TokenEmbedding;

% 利用 SVD分解计算主成分
X = TokenEmbedding - mean(TokenEmbedding, 1);
[~, ~, V] = svd(X, 'econ');
X2d = X * V(:, 1:2);

% 星空图
figure;
scatter(X2d(:,1), X2d(:,2), 5, [0.8 0.8 0.8], 'filled');
hold on;

% 选取若干有代表性的字标注
keywords = ['春','夏','秋','冬','东','南','西','北',...
    '红','黄','青','绿','白','黑','一','二','三','万'];
for k = 1:length(keywords)
    idx = find(vocabulary == keywords(k));
    text(X2d(idx,1), X2d(idx,2), keywords(k), 'FontSize', 10);
end
