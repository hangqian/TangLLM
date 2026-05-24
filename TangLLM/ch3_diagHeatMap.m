% 非黑即白的注意力热力图

rng('Default')

% 嵌入矩阵为正态随机数
n = 12;
d = 1024;
head = 8;
X = randn(n,d);

% 忽略权重矩阵
Wq = eye(d);
Wk = eye(d);
Wv = eye(d);
Wo = eye(d);

% 画张热力图
[~,W1] = attnHead(X,Wq,Wk,Wv,Wo,head);
figure(1)
for m = 1:head
    subplot(2,head/2,m);imagesc(W1(:,:,m));axis image; colormap hot;
end

% 随机线性变换矩阵
Wq = randn(d);
Wk = randn(d);
Wv = randn(d);
Wo = randn(d);


% 画张热力图
[~,W2] = attnHead(X,Wq,Wk,Wv,Wo,head);
figure(2)
for m = 1:head
    subplot(2,head/2,m);imagesc(W2(:,:,m));axis image; colormap hot;
end



function [Y,Weight] = attnHead(X,Wq,Wk,Wv,Wo,head)
% 多头注意力模块

% 词嵌入线性变换，分身QKV三剑客
Q = X * Wq;
K = X * Wk;
V = X * Wv;

% 多头注意力
[N,d] = size(Q);
dk = d/head;
Query = reshape(Q, [N, dk, head]);
Key = reshape(K, [N, dk, head]);
Value = reshape(V, [N, dk, head]);

% 上下文中各词之间的相关性
S = pagemtimes(Query, 'none', Key, 'transpose') ./ sqrt(dk);

% 下三角因果链
nonCausal = triu(true(N), 1);
S(repmat(nonCausal, [1, 1, head])) = -Inf;

% 注意力权重矩阵
expS = exp(S - max(S, [], 2));
Weight = expS ./ sum(expS, 2);

% 对上下文中各词做加权平均
Ym = reshape(pagemtimes(Weight, 'none', Value, 'none'), [N, d]);  

% 多头融合: 线性投影层将所有头的信息混合
Y = Ym * Wo; 
end

