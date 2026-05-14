% 训练唐诗模型

clear
rng('Default')
tic;

% 训练数据：全唐诗
Poems = readlines("TangPoetry.txt");

% 构建词汇表
vocabulary = unique(char(join(Poems, "")));

% 上下文结束符，用于遮挡注意力
EOS = length(vocabulary) + 1;
vocabularySize = length(vocabulary) + 1;
fprintf('当前EOS的ID   %d\n', EOS);

% 文本转换为整数 ID
% 将所有诗句用分隔符char(0)串联，并在文本最末尾也补上一个分隔符
% 由于 char(0) 不在词汇表中，它会被映射为 0. 将 0 批量替换为 EOS 标记
joinedStr = join(Poems, char(0)) + char(0);
[~, TokenID] = ismember(char(joinedStr), vocabulary);
TokenID(TokenID == 0) = EOS;
TokenID = TokenID(:);
numDataPoints = numel(TokenID);

% 模型维数与配置
contextLength = 48;       % 语料上下文长度(N)    
embeddingDimension = 128; % 词嵌入向量的长度(d)
numHeads = 8;             % 注意力模块的头数
numLayers = 3;            % 注意力模块的层数  
numEpochs = 50000;        % 训练期数
batchSize = 32;           % 训练时批量读取诗句的数量
learningRate0 = 0.005;    % 初始学习率
warmupSteps = 100;        % 前期慢慢加热
weightDecay = 0.05;       % AdamW 权重衰减惩罚力度
gradientBound = 1.0;      % 可用于学习的导数取值范围
std0 = 0.02;              % 初始参数的标准差
WeightTying = false+0;    % 输入和输出嵌入权重是否共享
isSwiGLU = true+0;        % SwiGLU前馈网络
MoE = 1;                  % 混合专家数
if MoE > 1
    isSwiGLU = true;
end

N = contextLength;        % 约定一个简称 N
d = embeddingDimension;   % 约定一个简称 d
Specifications.N = N;
Specifications.d = d;
Specifications.numHeads = numHeads;
Specifications.numLayers = numLayers;
Specifications.vocabularySize = vocabularySize;
Specifications.EOS = EOS;
Specifications.WeightTying = WeightTying;
Specifications.isSwiGLU = isSwiGLU;
Specifications.MoE = MoE;
fprintf('上下文长度   %d\n', N);
fprintf('词嵌入维数   %d\n', d);
fprintf('注意力层数   %d\n', numLayers);
fprintf('小批数据数   %d\n', batchSize);

% 初始化模型参数
if exist('TrainedModel.mat','file') == 2
    disp('导入预训练的模型参数')
    load('TrainedModel.mat','Parameters')
    if ~WeightTying && ~isfield(Parameters,"W")
        Parameters.W =  Parameters.TokenEmbedding';
    end
    if ~isfield(Parameters,"scale")
        Parameters.scale = ones(2*numLayers+1, d);
    elseif isvector(Parameters.scale)
        Parameters.scale = ones(2*numLayers+1, d);
    end
    if isSwiGLU && ~isfield(Parameters,"W0")
        Parameters.W0 = std0 .* randn(d,4*d,numLayers,MoE);
    end
    if isSwiGLU && ~isfield(Parameters,"b0")
        Parameters.b0 = std0 .* randn(1,4*d,numLayers,MoE);
    end
    if MoE > 1
        if ~isfield(Parameters,"WGate")
            Parameters.WGate = std0 .* randn(d,MoE,numLayers);
        end
        if ndims(Parameters.W1) == 3
            Parameters.W0 = repmat(Parameters.W0,[1,1,1,MoE]);
            Parameters.W1 = repmat(Parameters.W1,[1,1,1,MoE]);
            Parameters.W2 = repmat(Parameters.W2,[1,1,1,MoE]);
            Parameters.b0 = repmat(Parameters.b0,[1,1,1,MoE]);
            Parameters.b1 = repmat(Parameters.b1,[1,1,1,MoE]);
            Parameters.b2 = repmat(Parameters.b2,[1,1,1,MoE]);
        end
    end
else
    % 随机初始化
    % 初始化词嵌入向量，随机初始化表明没有汉语的先验知识
    % 嵌入也视作待估参数，与其它模型参数一起估计
    Parameters.TokenEmbedding = std0 .* randn(vocabularySize, d);
    Parameters.Wq = std0 .* randn(d,d,numLayers);
    Parameters.Wk = std0 .* randn(d,d,numLayers);
    Parameters.Wv = std0 .* randn(d,d,numLayers);
    Parameters.Wo = std0 .* randn(d,d,numLayers);
    Parameters.W1 = std0 .* randn(d,4*d,numLayers,MoE);
    Parameters.W2 = std0 .* randn(4*d,d,numLayers,MoE);
    Parameters.b1 = std0 .* randn(1,4*d,numLayers,MoE);
    Parameters.b2 = std0 .* randn(1,d,numLayers,MoE);
    if isSwiGLU
        Parameters.W0 = std0 .* randn(d,4*d,numLayers,MoE);
        Parameters.b0 = std0 .* randn(1,4*d,numLayers,MoE);
    end
    if MoE > 1
        Parameters.WGate = std0 .* randn(d,MoE,numLayers);
    end
    Parameters.scale = ones(2*numLayers+1, d);
    if ~WeightTying
        Parameters.W =  std0 .* randn(d,vocabularySize);
    end
    Parameters.b =  std0 .* randn(1,vocabularySize);
end

VarNames = fieldnames(Parameters);
numFields = length(VarNames);
numParamsEmbedding = vocabularySize*d;
numParamsAttention = 4*d*d*numLayers;
if isSwiGLU
    numParamsForward = 3*4*d*d*numLayers + 4*d*numLayers + d*numLayers + d*(2*numLayers+1);
else
    numParamsForward = 2*4*d*d*numLayers + 4*d*numLayers + d*numLayers + d*(2*numLayers+1);
end
if MoE > 1
    numParamsForward = MoE * numParamsForward + d*MoE*numLayers;
end
if WeightTying
    numParamsOutput = vocabularySize;
else
    numParamsOutput = vocabularySize*d + vocabularySize;
end
numParams = numParamsEmbedding + numParamsAttention + numParamsForward + numParamsOutput;
fprintf('模型参数总数   %d\n', numParams);
fprintf('嵌入矩阵参数   %d\n', numParamsEmbedding);
fprintf('注意力层参数   %d\n', numParamsAttention);
fprintf('前馈网络参数   %d\n', numParamsForward);
fprintf('输出层参数     %d\n', numParamsOutput);

% Adam 优化器的初始化
beta1 = 0.9; 
beta2 = 0.999;
Momentum = Parameters; 
Volatitlity = Parameters; 
for f = 1:numFields
    Momentum.(VarNames{f})(:) = 0; 
    Volatitlity.(VarNames{f})(:) = 0; 
end

checkGradient = 1+0;

% 随机梯度下降，数值最优化迭代
for epoch = 1:numEpochs

    % 初始化累加器
    AccumulatedGradients = Parameters;
    for f = 1:numFields
        AccumulatedGradients.(VarNames{f})(:) = 0;
    end
    lossSum = 0;
    
    for b = 1:batchSize

        % 随机抽取一批数据用于本期训练
        select = randi(numDataPoints-N+1);
        BatchData = TokenID(select:select+N-1); 

        % 以注意力机制为核心的网络前向与反向传播
        [loss, Gradients] = transformer(BatchData,Parameters,Specifications);
        lossSum = lossSum + loss;

        % 校验解析导数的正确性
        if checkGradient && (epoch == 1 || epoch == numEpochs) && b==1
            fun = @(Parameters) transformer(BatchData,Parameters,Specifications);
            validateGradients(fun, Parameters, Gradients, VarNames);
        end
        
        % 累加梯度
        for f = 1:numFields
            name = VarNames{f};
            AccumulatedGradients.(name) = AccumulatedGradients.(name) + Gradients.(name);
        end
    end
    
    % 平均损失和平均梯度
    lossAvg = lossSum / batchSize;
    for f = 1:numFields
        name = VarNames{f};
        AccumulatedGradients.(name) = AccumulatedGradients.(name) / batchSize;
    end

    % 学习率先在预热期增加，然后随时间推移而递减
    if epoch <= warmupSteps
        % 线性预热
        learningRate = learningRate0 * (epoch / warmupSteps);
    else
        % 余弦退火衰减
        progress = (epoch - warmupSteps) / (numEpochs - warmupSteps);
        learningRate = learningRate0 * 0.5 * (1 + cos(pi * progress));
    end
    
    % Adam 优化器利用导数信息更新参数
    for f = 1:numFields

        % 当前变量的梯度
        name = VarNames{f};
        gt = AccumulatedGradients.(name);

        % 限速以防梯度爆炸
        gt = max(min(gt, gradientBound), -gradientBound); 

        % 指数平滑历史梯度的动量及波动性
        Momentum.(name) = beta1 * Momentum.(name) + (1 - beta1) * gt;
        Volatitlity.(name) = beta2 * Volatitlity.(name) + (1 - beta2) * (gt.^2);

        % 初始值校正
        mHat = Momentum.(name) / (1 - beta1^epoch);
        vHat = Volatitlity.(name) / (1 - beta2^epoch);

        % AdamW 更新参数
        Parameters.(name) = Parameters.(name) - learningRate * (mHat ./ (sqrt(vHat) + 1e-8)) - learningRate * weightDecay * Parameters.(name);
    end
    
    % 定期汇报学习成果
    if mod(epoch, 20) == 0
        fprintf('训练期数 %5d / %d | 负对数似然: %.4f | 学习率: %.6f\n', epoch, numEpochs, lossAvg, learningRate);
    end

    % 定期保存到 mat 文件，以供模型推断
    if mod(epoch, 1000) == 0
        save('TrainedModel.mat', 'Parameters', 'Specifications', 'vocabulary','epoch');
        fprintf('\n模型已保存至 TrainedModel. 负对数似然:  %.4f\n', lossAvg);
    end

end
toc;
