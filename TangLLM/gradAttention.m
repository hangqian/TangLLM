function [dX,dWq,dWk,dWv,dWo] = gradAttention(X,dY,Wq,Wk,Wv,Wo,numHeads)
% 注意力模块的导数反向传播
%
% X:  输入嵌入
% dY: 后方传来的导数dL/dY
% Wq,Wk,Wv,Wo,numHeads: 模型参数与配置
%
% dX: 输出导数dL/dX
% dWq,dWk,dWv,dWo: 参数的导数

% QKV三剑客: 词嵌入的线性变换
Q = X * Wq;
K = X * Wk;
V = X * Wv;

% 融入位置信息，RoPE 前向旋转
Q = RoPE(Q,numHeads,'forward');
K = RoPE(K,numHeads,'forward');

% 多头注意力：词嵌入向量被瓜分为几块，各头并行处理其中一块
[N,d] = size(Q);
dk = d/numHeads;
Query = reshape(Q, [N, dk, numHeads]);
Key = reshape(K, [N, dk, numHeads]);
Value = reshape(V, [N, dk, numHeads]);

% 注意力分数: 上下文各词相关性
S = pagemtimes(Query, 'none', Key, 'transpose') ./ sqrt(dk);

% 因果掩码: 下三角因果链
nonCausal = triu(true(N), 1);
S(repmat(nonCausal, [1, 1, numHeads])) = -Inf;

% 注意力权重: SoftMax
expS = exp(S - max(S, [], 2));
W = expS ./ sum(expS, 2);

% 语义融合: 上下文嵌入加权平均
Y3D = pagemtimes(W, 'none', Value, 'none');
Ym = reshape(Y3D, [N, d]);

% 多头融合: 线性投影层将所有头的信息混合
% Y = Ym * Wo;

% 反向传播：对 Y = Ym * Wo 求导
dWo = Ym' * dY;
dYm = dY * Wo';
dYm3D = reshape(dYm,[N, dk, numHeads]);

% 反向传播：对 Ym = W*V 求导
dV3D = pagemtimes(W, 'transpose', dYm3D, 'none');
dW = pagemtimes(dYm3D, 'none', Value, 'transpose');

% 反向传播：SoftMax步骤求导
dS = W.*(dW - sum(W.*dW,2));

% 因果掩码的导数: 未来信息的梯度截断为0
dS(repmat(nonCausal, [1, 1, numHeads])) = 0;

% 反向传播：对 S=Q*K 求导
dQ3D = pagemtimes(dS, 'none', Key, 'none')./ sqrt(dk);
dK3D = pagemtimes(dS, 'transpose', Query, 'none')./ sqrt(dk);

% 重塑为二维矩阵
dQ = reshape(dQ3D, [N, d]);
dK = reshape(dK3D, [N, d]);
dV = reshape(dV3D, [N, d]);

% 反向传播：RoPE 反向旋转
dQ = RoPE(dQ,numHeads,'backward');
dK = RoPE(dK,numHeads,'backward');

% 反向传播：对 Q = X * Wq;K = X * Wk;V = X * Wv; 求导
dWq = X' * dQ;
dWk = X' * dK;
dWv = X' * dV;
dX = dQ*Wq' + dK*Wk' + dV*Wv';
end