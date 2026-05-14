function modelInference(Prompt,temperature,trainedModel,reproducibility,heatMap)
% 用训练好的唐诗大模型推断
% Prompt:         提示词 (字符串）
% temperature:    创造力（正数标量）
% trainedModel:   训练好的模型文件名
% reproducibility 可复现（逻辑变量）

if nargin < 1
    Prompt = input("请输入一句诗：","s");
end

% 推理阶段的创造力程度
if nargin < 2
    temperature = 0.8;
end

% 训练好的唐诗大模型文件名
if nargin < 3 || isempty(trainedModel)
    trainedModel = 'TrainedModel.mat';
end

% 可复现性
if nargin < 4
    reproducibility = true;
end
if reproducibility
    rng('Default')
end

% 
if nargin < 5
    heatMap = false;
end

% 导入训练好的大模型参数与配置
load(trainedModel, 'Parameters', 'Specifications', 'vocabulary');
numLayers = Specifications.numLayers;
EOS = Specifications.EOS;
if ~isfield(Parameters,"W")
    Parameters.W =  Parameters.TokenEmbedding';
end
if ~isfield(Parameters,"scale")
    Parameters.scale = ones(2*numLayers+1, Specifications.d);
elseif isvector(Parameters.scale)
    Parameters.scale = ones(2*numLayers+1, Specifications.d);
end
if ~isfield(Specifications,"WeightTying")
    Specifications.WeightTying = false;
end
if ~isfield(Specifications,"isSwiGLU")
    Specifications.isSwiGLU = false;
end
if ~isfield(Specifications,"MoE")
    Specifications.MoE = 1;
end

% 利用大模型推理
Prompt = string(Prompt);
numPrompt = numel(Prompt);
for m = 1:numPrompt

    % 本句提示词
    PromptChar = char(Prompt(m));

    % 标点符号规范化
    PromptChar = strrep(PromptChar, ',', '，');
    numTokens = length(PromptChar);
    if ~contains(PromptChar,'，') && (numTokens==5||numTokens==7)
        PromptChar = [PromptChar,'，']; %#ok<AGROW>
        numTokens = numTokens + 1;
    end 

    % 提示词转换为词元ID   
    [~, InputID] = ismember(PromptChar, vocabulary);

    % 词汇表以外的词 (ID == 0) 用Unicode 编码最接近的词替代
    unknown = find(InputID == 0);
    if ~isempty(unknown)        
        Unicode = double(vocabulary);
        for k = 1:length(unknown)
            pos = unknown(k);
            txt = PromptChar(pos);
            [~, nearestId] = min(abs(Unicode - double(txt)));
            InputID(pos) = nearestId;
            fprintf('遇到了生僻字 "%s"，根据部首字形，我猜它大概是 "%s" 的意思...\n', txt, vocabulary(nearestId));
        end
    end

    % 初始化 
    CurrentID = InputID;
    ResponseID = [];
    CacheK = cell(numLayers, 1);
    CacheV = cell(numLayers, 1);
    
    % 每次只传最新的单个字进网络(KV 缓存)
    for i = 1:numTokens*3

        % 大模型推断，预测下一个词元的概率分布
        [Score, CacheK, CacheV] = transformerInference(CurrentID, Parameters, Specifications, CacheK, CacheV);
        lastScore = Score(end, :) / temperature;        
        lastScore = lastScore - max(lastScore);
        prob = exp(lastScore) / sum(exp(lastScore));

        % 依概率抽样
        nextId = find(rand < cumsum(prob), 1, 'first');
        if nextId == EOS
            break
        else
            CurrentID = nextId;
            ResponseID = [ResponseID, nextId]; %#ok<AGROW>
        end
    end
    fprintf('【上句】: %s\n', join(vocabulary(InputID), ""));
    fprintf('【下句】: %s\n', join(vocabulary(ResponseID), ""));
end

% 注意力热力图
if heatMap
    AllID = [InputID,ResponseID];
    [~, ~, ~, AllWeights] = transformerInference(AllID, Parameters, Specifications, cell(numLayers, 1), cell(numLayers, 1));
    txt = vocabulary(AllID);
    plot_attention_maps(txt, AllWeights);
end

end


%-------------------------------------------------------------------------
function [Score, CacheK, CacheV, AllWeights] = transformerInference(InputIds, Parameters, Specifications, CacheK, CacheV)

% 模型维数与配置
N = length(InputIds);
d = Specifications.d;
numHeads = Specifications.numHeads;
numLayers = Specifications.numLayers;
dk = d / numHeads;
WeightTying = Specifications.WeightTying;
isSwiGLU = Specifications.isSwiGLU;
MoE = Specifications.MoE;

if isempty(CacheK{1})
    pastLength = 0;
else
    pastLength = size(CacheK{1}, 1);
end

requestWeights = nargout > 3;
if requestWeights
    AllWeights = cell(numLayers, 1);
end

% 把词元转换为嵌入向量
X = Parameters.TokenEmbedding(InputIds, :);

for L = 1:numLayers

    % RMS归一化（注意力前）
    X0 = RMSNorm(X, Parameters.scale(L,:));

    % 分身为QKV三剑客
    Q = X0 * Parameters.Wq(:,:,L);
    KNew = X0 * Parameters.Wk(:,:,L);
    VNew = X0 * Parameters.Wv(:,:,L);

    % 应用 RoPE：仅对当前新输入的特征在其绝对位置上进行旋转
    for i = 1:N
        pos = pastLength + i; % 计算绝对时间戳
        Q(i, :) = RoPE_complex(Q(i, :), pos, d,numHeads);
        KNew(i, :) = RoPE_complex(KNew(i, :), pos, d,numHeads);
    end

    % 利用缓存和增量，把K和V补充完整
    K = [CacheK{L}; KNew];
    V = [CacheV{L}; VNew];
    CacheK{L} = K;
    CacheV{L} = V;

    % 多头注意力：词嵌入向量被瓜分为几块，各头并行处理其中一块
    NBig = pastLength + N;
    Query = reshape(Q, [N, dk, numHeads]);
    Key = reshape(K, [NBig, dk, numHeads]);
    Value = reshape(V, [NBig, dk, numHeads]);

    % 注意力分数: 上下文各词相关性
    QK = pagemtimes(Query, 'none', Key, 'transpose') ./ sqrt(dk);

    % 因果掩码
    if pastLength == 0
        nonCausal = triu(true(N), 1);
        QK(repmat(nonCausal, [1, 1, numHeads])) = -Inf;
    end    

    % 注意力权重: SoftMax
    expQK = exp(QK - max(QK, [], 2));
    Weight = expQK ./ sum(expQK, 2);

    if requestWeights
        AllWeights{L} = Weight;
    end

    % 语义融合: 上下文嵌入加权平均
    WeightedAverage = reshape(pagemtimes(Weight, 'none', Value, 'none'), [N, d]);

    % 多头融合: 线性投影层将所有头的信息混合
    XAttention = WeightedAverage * Parameters.Wo(:,:,L);

    % 残差网络：保留原输入信号，注意力模块提供增量
    X1 = X + XAttention;

    % RMS归一化（前馈网络前）
    X2 = RMSNorm(X1, Parameters.scale(numLayers+L,:));

    % 残差形式的前馈网络
    if MoE == 1
        if isSwiGLU
            X3 = X1 + SwiGLU(X2,Parameters.W0(:,:,L), Parameters.b0(:,:,L),Parameters.W1(:,:,L), Parameters.b1(:,:,L), Parameters.W2(:,:,L), Parameters.b2(:,:,L));
        else
            X3 = X1 + feedforward(X2,Parameters.W1(:,:,L), Parameters.b1(:,:,L), Parameters.W2(:,:,L), Parameters.b2(:,:,L));
        end
    else
        W0Use = reshape(Parameters.W0(:,:,L,:),[d,4*d,MoE]);
        b0Use = reshape(Parameters.b0(:,:,L,:),[1,4*d,MoE]);
        W1Use = reshape(Parameters.W1(:,:,L,:),[d,4*d,MoE]);
        b1Use = reshape(Parameters.b1(:,:,L,:),[1,4*d,MoE]);
        W2Use = reshape(Parameters.W2(:,:,L,:),[4*d,d,MoE]);
        b2Use = reshape(Parameters.b2(:,:,L,:),[1,d,MoE]);
        X3 = X1 + MoESwiGLU(X2, Parameters.WGate(:,:,L), W0Use, b0Use, W1Use, b1Use, W2Use, b2Use);
    end

    % 本层输出更新为下一层的输入！
    X = X3;
end

% 进入输出层之前再做一次归一化，防止残差网络方差膨胀
X3Norm = RMSNorm(X3, Parameters.scale(2*numLayers+1,:));

% 输出层（神经元数量扩展到三千词表量）
% 对词表中每个词评分（预测词与词表中各词的相关性）
if WeightTying
    W = Parameters.TokenEmbedding';
else
    W = Parameters.W;
end
Score = X3Norm * W + Parameters.b;
end


%-------------------------------------------------------------------------
function y = RoPE_complex(x, position, d, numHeads)
    dk = d / numHeads;
    % 频率衰减必须基于单头维度 dk
    theta = 10000.^(-2*((1:dk/2)-1)/dk); 
    
    % 将 theta 复制 numHeads 次，使得每个头经历相同的频段
    theta = repmat(theta, 1, numHeads); 
    
    % 按列维度切片提取特征
    x_complex = complex(x(:, 1:2:end), x(:, 2:2:end));
    
    % 广播机制进行相位旋转
    y_complex = x_complex .* exp(1i * position * theta);
    y = x;
    y(:, 1:2:end) = real(y_complex);
    y(:, 2:2:end) = imag(y_complex);
end


% 注意力热力图渲染
function plot_attention_maps(PromptChars, AllWeights)
numLayers = length(AllWeights);
numHeads = size(AllWeights{1},3);
N = length(PromptChars);
figure(1)
x_labels = num2cell(PromptChars);
y_labels = num2cell(PromptChars);
plot_idx = 1;
for L = 1:numLayers
    W_layer = AllWeights{L}; % 提取当前层的权重矩阵 [N, N, numHeads]
    for h = 1:numHeads
        subplot(numLayers, numHeads, plot_idx);
        % 画热力图
        imagesc(W_layer(:, :, h));
        colormap(gca, 'hot');
        clim([0, 1]); % 概率范围固定在 0 到 1

        % 设置坐标轴标签 (只在边缘显示文字，避免画面过于杂乱)
        set(gca, 'XTick', 1:N, 'XTickLabel', x_labels, 'YTick', 1:N, 'YTickLabel', y_labels);
        set(gca, 'FontName', 'Microsoft YaHei', 'FontSize', 10);
        title(sprintf('层%d - 头%d', L, h), 'FontSize', 12, 'FontWeight', 'bold');
        if h ~= 1, set(gca, 'YTickLabel', []); end
        if L ~= numLayers, set(gca, 'XTickLabel', []); end
        plot_idx = plot_idx + 1;
    end
end
sgtitle('注意力概率分布解析', 'FontSize', 16, 'FontWeight', 'bold');
end

