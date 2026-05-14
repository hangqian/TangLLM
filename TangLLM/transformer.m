function [loss,Gradients] = transformer(BatchData,Parameters,Specifications)
% 以注意力机制为核心的解码器

% 网络输入与真实标签
InputLabel  = BatchData(1:end-1);
TargetLabel = BatchData(2:end);

% 把词元转换为嵌入向量
X = Parameters.TokenEmbedding(InputLabel, :);

% 读取模型配置
% N = Specifications.N;
N = numel(InputLabel); % 通常是 Specifications.N - 1
d = Specifications.d;
numHeads = Specifications.numHeads;
numLayers = Specifications.numLayers;
vocabularySize = Specifications.vocabularySize;
EOS = Specifications.EOS;
WeightTying = Specifications.WeightTying;
isSwiGLU = Specifications.isSwiGLU;
MoE = Specifications.MoE;

% 读取参数
Wq = Parameters.Wq;
Wk = Parameters.Wk;
Wv = Parameters.Wv;
Wo = Parameters.Wo;
W1 = Parameters.W1;
W2 = Parameters.W2;
b1 = Parameters.b1;
b2 = Parameters.b2;
if isSwiGLU
    W0 = Parameters.W0;
    b0 = Parameters.b0;
end
if MoE > 1
    WGate = Parameters.WGate;
end
scale = Parameters.scale;
if WeightTying
    W = Parameters.TokenEmbedding';
else
    W = Parameters.W;
end
b = Parameters.b;
Gradients = Parameters;


% 向前传播 X -> X0 -> X1 -> X2 -> X3
% 残差捷径 X    ->    X1    ->    X3

CacheX = repmat(X,[1,1,numLayers]);
CacheX0 = CacheX;
CacheX1 = CacheX;
CacheX2 = CacheX;
for L = 1:numLayers

    % RMS归一化（注意力前）
    X0 = RMSNorm(X, scale(L,:));

    % 残差形式的注意力机制
    X1 = X + attention(X0,Wq(:,:,L),Wk(:,:,L),Wv(:,:,L),Wo(:,:,L),numHeads,InputLabel,EOS);

    % RMS归一化（前馈网络前）
    X2 = RMSNorm(X1, scale(numLayers+L,:));

    % 残差形式的前馈网络
    if MoE == 1
        if isSwiGLU
            X3 = X1 + SwiGLU(X2,W0(:,:,L), b0(:,:,L),W1(:,:,L), b1(:,:,L), W2(:,:,L), b2(:,:,L));
        else
            X3 = X1 + feedforward(X2,W1(:,:,L), b1(:,:,L), W2(:,:,L), b2(:,:,L));
        end
    else
        W0Use = reshape(W0(:,:,L,:),[d,4*d,MoE]);
        b0Use = reshape(b0(:,:,L,:),[1,4*d,MoE]);
        W1Use = reshape(W1(:,:,L,:),[d,4*d,MoE]);
        b1Use = reshape(b1(:,:,L,:),[1,4*d,MoE]);
        W2Use = reshape(W2(:,:,L,:),[4*d,d,MoE]);
        b2Use = reshape(b2(:,:,L,:),[1,d,MoE]);
        X3 = X1 + MoESwiGLU(X2, WGate(:,:,L), W0Use, b0Use, W1Use, b1Use, W2Use, b2Use);
    end

    % 准备进入下一轮循环
    CacheX(:,:,L) = X;
    CacheX0(:,:,L) = X0;
    CacheX1(:,:,L) = X1;
    CacheX2(:,:,L) = X2;    
    X = X3;
end

% 进入输出层之前再做一次归一化，防止残差网络方差膨胀
X3Norm = RMSNorm(X3, scale(2*numLayers+1,:));

% 输出层及其导数
if nargout < 2
    loss = gradOutputLayer(X3Norm,TargetLabel,W,b,InputLabel,EOS);
    return
else
    [loss,dX3Norm, dW, db] = gradOutputLayer(X3Norm,TargetLabel,W,b,InputLabel,EOS);
end
if ~WeightTying
    Gradients.W = dW;
end
Gradients.b = db;

% 最后那次归一化的导数
[dX3, dscaleOut] = gradRMSNorm(X3, dX3Norm, scale(2*numLayers+1,:));
Gradients.scale(2*numLayers+1,:) = dscaleOut;

% 导数反向传播 dX3 -> dX2 -> dX1 -> dX0 >- dX

for L = numLayers:-1:1

    % 前馈网络的导数
    if MoE == 1
        if isSwiGLU
            [dX2,dW0,db0,dW1,db1,dW2,db2] = gradSwiGLU(CacheX2(:,:,L),dX3,W0(:,:,L),b0(:,:,L),W1(:,:,L),b1(:,:,L),W2(:,:,L));
        else
            [dX2,dW1,db1,dW2,db2] = gradFeedForward(CacheX2(:,:,L),dX3,W1(:,:,L),b1(:,:,L),W2(:,:,L));
        end
    else
        W0Use = reshape(W0(:,:,L,:),[d,4*d,MoE]);
        b0Use = reshape(b0(:,:,L,:),[1,4*d,MoE]);
        W1Use = reshape(W1(:,:,L,:),[d,4*d,MoE]);
        b1Use = reshape(b1(:,:,L,:),[1,4*d,MoE]);
        W2Use = reshape(W2(:,:,L,:),[4*d,d,MoE]);
        b2Use = reshape(b2(:,:,L,:),[1,d,MoE]);
        [dX2, dWGate, dW0, db0, dW1, db1, dW2, db2] = gradMoESwiGLU(CacheX2(:,:,L), dX3, WGate(:,:,L), W0Use, b0Use, W1Use, b1Use, W2Use, b2Use);
    end

    % 残差网络: RMS归一化(FFN前)的导数 + 捷径（输出层）的导数
    [dX1_norm, dscaleFFN] = gradRMSNorm(CacheX1(:,:,L), dX2, scale(numLayers+L,:));
    Gradients.scale(numLayers+L,:) = dscaleFFN;
    dX1 = dX1_norm + dX3;

    % 注意力机制的导数
    [dX0,dWq,dWk,dWv,dWo] = gradAttention(CacheX0(:,:,L),dX1,Wq(:,:,L),Wk(:,:,L),Wv(:,:,L),Wo(:,:,L),numHeads,InputLabel,EOS);

    % 残差网络: RMS归一化(注意力前)的导数 + 捷径的导数
    [dX_norm, dscaleAttn] = gradRMSNorm(CacheX(:,:,L), dX0, scale(L,:));
    Gradients.scale(L,:) = dscaleAttn;
    dX = dX_norm + dX1;

    % 准备进入下一轮循环
    dX3 = dX;
    Gradients.Wq(:,:,L) = dWq;
    Gradients.Wk(:,:,L) = dWk;
    Gradients.Wv(:,:,L) = dWv;
    Gradients.Wo(:,:,L) = dWo;
    Gradients.W1(:,:,L,:) = dW1;
    Gradients.W2(:,:,L,:) = dW2;
    Gradients.b1(:,:,L,:) = db1;
    Gradients.b2(:,:,L,:) = db2;
    if isSwiGLU
        Gradients.W0(:,:,L,:) = dW0;
        Gradients.b0(:,:,L,:) = db0;
    end
    if MoE > 1
        Gradients.WGate(:,:,L) = dWGate;
    end
end

% 反向传播嵌入的导数
% OneHotRoute = sparse(InputLabel, (1:N)', 1, vocabularySize, N);
% dTokenEmbedding = OneHotRoute * dX;
dTokenEmbedding = zeros(vocabularySize, d);
if numel(unique(InputLabel)) == N
    dTokenEmbedding(InputLabel, :) = dX;
else
    for k = 1:N
        dTokenEmbedding(InputLabel(k), :) = dTokenEmbedding(InputLabel(k), :) + dX(k, :);
    end
end

% 嵌入权重共享时加上输出层W的导数
if WeightTying    
    Gradients.TokenEmbedding = dTokenEmbedding + dW';
else
    Gradients.TokenEmbedding = dTokenEmbedding;
end

end


%-------------------------------------------------------------------------
% 局部函数: 增加了EOS用以序列拼接和注意力遮挡
function Y = attention(X,Wq,Wk,Wv,Wo,numHeads,InputLabel,EOS)
% 多头注意力+因果掩码+旋转位置编码

% QKV三剑客: 词嵌入的线性变换
Q = X * Wq;
K = X * Wk;
V = X * Wv;

% 融入位置信息: RoPE 前向旋转
Q = RoPE(Q,numHeads);
K = RoPE(K,numHeads);

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

% 识别本段语料句末EOS标记
% 凡是在分隔符之后的字，不允许看分隔符及以前的字
% positions = find(InputLabel == EOS);
% mask = nonCausal;
% for k = 1:numel(positions)
%    pos = positions(k);
%    mask(pos+1:end, 1:pos) = true;
% end
isEOS = (InputLabel == EOS);      
groups = cumsum([0; isEOS(1:end-1)]);   
maskEOS = groups ~= groups'; 
S(repmat(maskEOS, [1, 1, numHeads])) = -Inf;

% 注意力权重: SoftMax
expS = exp(S - max(S, [], 2));
Weight = expS ./ sum(expS, 2);

% 语义融合: 上下文嵌入加权平均
Ym = reshape(pagemtimes(Weight, 'none', Value, 'none'), [N, d]);

% 多头融合: 线性投影层将所有头的信息混合
Y = Ym * Wo;

end


%-------------------------------------------------------------------------
% 局部函数: 增加了EOS用以序列拼接和注意力遮挡
function [dX,dWq,dWk,dWv,dWo] = gradAttention(X,dY,Wq,Wk,Wv,Wo,numHeads,InputLabel,EOS)
% 注意力模块的导数反向传播
%
% X:  输入嵌入
% dY: 后方传来的导数dL/dY
% Wq,Wk,Wv,Wo,numHeads: 模型参数与配置
% InputLabel,EOS: 用以遮挡注意力
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
nonCausal3D = repmat(nonCausal, [1, 1, numHeads]);
S(nonCausal3D) = -Inf;

% 识别本段语料句末EOS标记
% 凡是在分隔符之后的字，不允许看分隔符及以前的字
% positions = find(InputLabel == EOS);
% mask = nonCausal;
% for k = 1:numel(positions)
%    pos = positions(k);
%    mask(pos+1:end, 1:pos) = true;
% end
isEOS = (InputLabel == EOS);      
groups = cumsum([0; isEOS(1:end-1)]);   
maskEOS = groups ~= groups'; 
maskEOS3D = repmat(maskEOS, [1, 1, numHeads]);
S(maskEOS3D) = -Inf;

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
dS(nonCausal3D) = 0;
dS(maskEOS3D) = 0;

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


%-------------------------------------------------------------------------
% 局部函数: 增加了EOS，其似然函数贡献需删除
function [loss,dX, dW, db] = gradOutputLayer(X,TargetLabel,W,b,InputLabel,EOS)
% 输出层的导数反向传播
%
% X:           n*d 输入嵌入(模型预测的下一个词嵌入）
% Targetlabel: n*1 真实标签
% W,b:         模型参数
% InputLabel,EOS: 用以遮挡EOS对似然函数的贡献
%
% loss:        负对数似然函数值（交叉熵损失）
% dX:          输出导数dL/dX（供下一阶段导数反向传播）
% dW,db:       参数的导数

% 对词表中每个词评分（预测词与词表中各词的相关性）
Score = X * W + b;

% 多元逻辑斯蒂模型(亦即SoftMax)
logProb = Score - max(Score, [], 2);
ProbRaw = exp(logProb);
Prob = ProbRaw ./ sum(ProbRaw, 2);

% 提取目标概率
[N,v] = size(Prob);
linearIdx = sub2ind([N, v], (1:N)', TargetLabel); 
targetProb = Prob(linearIdx); 

% 删除当前为EOS时预测对似然函数的贡献
ignoreMask = (InputLabel == EOS);
valid_N = max(sum(~ignoreMask), 1); 

% 计算负似然函数值(亦即交叉熵损失）
loss = sum(-log(targetProb) .* (~ignoreMask)) / valid_N;

% 4. 导数反向传播
if nargout > 1
    % 生成供反向传播使用的逻辑矩阵 Y
    Y = false(N, v);
    Y(linearIdx) = true;
    
    % 负似然函数对评分S求导：预测概率-真实标签
    dS = (Prob - Y) / valid_N;
    
    % 将 EOS 所在行的梯度清零
    dS(ignoreMask, :) = 0;

    % 根据链式法则和线性函数S=X*W+b，得到关于X,W,b的导数
    dW = X' * dS;
    db = sum(dS, 1);
    dX = dS * W';
end
end
