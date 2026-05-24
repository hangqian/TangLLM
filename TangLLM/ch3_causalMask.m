% 验证因果掩码下的注意力绝不偷看后文

% 导入数据
load('TrainedModel.mat','Parameters','vocabulary')

% 查词表
Poems = "不畏浮云遮望眼";
[~, ind] = ismember(char(Poems), vocabulary);

% 查嵌入表
TokenEmbedding = Parameters.TokenEmbedding;
X = TokenEmbedding(ind,:);

% 权重矩阵
layer = 3;
Wq = Parameters.Wq(:,:,layer);
Wk = Parameters.Wk(:,:,layer);
Wv = Parameters.Wv(:,:,layer);

% 用因果掩码一次算出注意力
Y = attnCausal(X,Wq,Wk,Wv);

% 只用第一个字计算注意力
X1 = X(1,:);
Y1 = X1 * Wv;
disp(norm(Y1-Y(1,:)))

% 只用前两个字计算注意力
X12 = X(1:2,:);
Y12 = attnCausal(X12,Wq,Wk,Wv);
Y2 = Y12(2,:);
disp(norm(Y2-Y(2,:)))

% 循环语句逐字验证
for i = 1:7
    Y1i = attnCausal(X(1:i,:),Wq,Wk,Wv);
    Yi = Y1i(end,:);
    disp(norm(Yi-Y(i,:)))
end

% 画张热力图
[Y,W] = attnCausal(X,Wq,Wk,Wv);
imagesc(W);
axis image; colormap hot; colorbar;


function [Y,W] = attnCausal(X,Wq,Wk,Wv)
% 因果注意力模块

% 词嵌入线性变换，分身QKV三剑客
Q = X * Wq;
K = X * Wk;
V = X * Wv;

% 未归一化的权重分数
d = size(Q,2);
S = Q*K'/sqrt(d);

% 因果注意力
N = size(Q,1);
nonCausal = triu(true(N), 1);
S(nonCausal) = -Inf;

% SoftMax形式的权重
expS = exp(S - max(S, [], 2));
W = expS./sum(expS,2);

% 对上下文做加权平均
Y = W*V;
end