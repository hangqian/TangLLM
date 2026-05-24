
% 导入数据
load('TrainedModel.mat','Parameters','vocabulary')
TokenEmbedding = Parameters.TokenEmbedding;
Wq = Parameters.Wq(:,:,1);
Wk = Parameters.Wk(:,:,1);
Wv = Parameters.Wv(:,:,1);

% 正向读诗
Poems = "夫忆妻兮父忆儿";
[~, ind] = ismember(char(Poems), vocabulary);
X = TokenEmbedding(ind,:);

% 反向读诗
PoemsRev = "儿忆父兮妻忆夫";
[~, indRev] = ismember(char(PoemsRev), vocabulary);
XRev = TokenEmbedding(indRev,:);

% 未加编码时的注意力
[Y,W] = attnSingle(X, Wq, Wk, Wv);
[YRev,WRev] = attnSingle(XRev, Wq, Wk, Wv);

% 对比"夫"字语义
gap1 = norm(Y(1,:)-YRev(end,:));
fprintf('无位置编码时，正反词序的首尾输出误差: %e\n', gap1); 

% 对比“词袋”
gapBag = norm(sortrows(Y) - sortrows(YRev));
fprintf('无位置编码时，正反词序的词袋差异: %e\n', gapBag); 

% 对比Y与YRev本身
gapSeq = norm(Y-YRev(end:-1:1,:));
fprintf('无位置编码时，正反词序的序列差异: %e\n', gapSeq); 
disp(' ')

% 热力图
subplot(1,2,1);imagesc(W);   colormap sky; title('夫忆妻兮父忆儿')
subplot(1,2,2);imagesc(WRev);colormap sky; title('儿忆父兮妻忆夫')

%-----------------------------------------------------
% 添加位置编码后的注意力
Y = attnRoPE(X, Wq, Wk, Wv);
YRev = attnRoPE(XRev, Wq, Wk, Wv);

% 对比"夫"字语义
gap1 = norm(Y(1,:)-YRev(end,:));
fprintf('加位置编码后，正反词序的首尾输出误差: %4.3f\n', gap1); 

% 对比“词袋”
gapBag = norm(sortrows(Y) - sortrows(YRev));
fprintf('加位置编码后，正反词序的词袋差异: %4.3f\n', gapBag); 

% 对比Y与YRev本身
gapSeq = norm(Y-YRev(end:-1:1,:));
fprintf('加位置编码后，正反词序的序列差异: %e\n', gapSeq); 
disp(' ')

%---------------------------------------
% 添加因果掩码（没有位置编码）后的注意力
Y = attnCausal(X, Wq, Wk, Wv);
YRev = attnCausal(XRev, Wq, Wk, Wv);

% 对比"夫"字语义
gap1 = norm(Y(1,:)-YRev(end,:));
fprintf('有因果掩码时，正反词序的首尾输出误差: %4.3f\n', gap1); 

% 对比“词袋”
gapBag = norm(sortrows(Y) - sortrows(YRev));
fprintf('有因果掩码时，正反词序的词袋差异: %4.3f\n', gapBag); 

% 对比Y与YRev本身
gapSeq = norm(Y-YRev(end:-1:1,:));
fprintf('有因果掩码时，正反词序的序列差异: %e\n', gapSeq); 



function [Y,W] = attnSingle(X,Wq,Wk,Wv)
% 单头注意力模块

% 词嵌入向量做线性变换，分身为QKV三剑客
Q = X * Wq;
K = X * Wk;
V = X * Wv;

% 未归一化的权重分数
d = size(Q,2);
S = Q*K'/sqrt(d);

% SoftMax形式的权重
expS = exp(S - max(S, [], 2));
W = expS./sum(expS,2);

% 对上下文做加权平均
Y = W*V;
end

function Y = attnCausal(X,Wq,Wk,Wv)
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

function [Y,W] = attnRoPE(X,Wq,Wk,Wv)
% 单头注意力模块

% 词嵌入向量做线性变换，分身为QKV三剑客
Q = X * Wq;
K = X * Wk;
V = X * Wv;

% 加入位置编码
Q = RoPE2(Q);
K = RoPE2(K);

% 未归一化的权重分数
d = size(Q,2);
S = Q*K'/sqrt(d);

% SoftMax形式的权重
expS = exp(S - max(S, [], 2));
W = expS./sum(expS,2);

% 对上下文做加权平均
Y = W*V;
end

function Y = RoPE2(X,inverse)
% 旋转位置编码（复数版）
%
% X:       n*d 输入嵌入
% inverse: 反向旋转(求导时用），默认为0
% Y:       n*d 输出嵌入

if nargin < 2
    inverse = false;
end

% n 代表上下文长度，d 代表嵌入维数
[n,d] = size(X);

% 生成角速度
theta = 10000.^(-2 * ((0:(d/2-1)) / d));

% 将相邻的实数配对成复数 (x1 + i*x2), (x3 + i*x4)...
Xc = complex(X(:, 1:2:end), X(:, 2:2:end));

% 利用欧拉公式乘以 e^(i * t * theta)，一步完成所有高维平面的旋转
if ~inverse
    Yc = Xc .* exp(1i * (1:n)' * theta);
else
    Yc = Xc .* exp(-1i * (1:n)' * theta);
end

% 拆分实部和虚部，交错拼装回实数向量
Y = zeros(n, d);
Y(:, 1:2:end) = real(Yc);
Y(:, 2:2:end) = imag(Yc);
end