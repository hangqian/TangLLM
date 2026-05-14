function [dX0, dWGate, dW0, db0, dW1, db1, dW2, db2] = gradMoESwiGLU(X0, dX3, WGate, W0, b0, W1, b1, W2, b2)
% MoE SwiGLU 层的导数反向传播

% 模型维数
numExperts = size(WGate, 2);
[N, d] = size(X0);

% 路由器打分
logP = X0 * WGate; % [N, num_experts]

% 计算 Softmax 概率
logP = logP - max(logP, [], 2);
ProbRaw = exp(logP);
Prob = ProbRaw ./ sum(ProbRaw, 2);

% maxk 函数返回前2名的概率值和对应的专家编号
topK = 2;
[TopProb, TopIndices] = maxk(Prob, topK, 2);

% 将前2名专家的概率重新归一化，使其加和为 1
SumTopK = sum(TopProb, 2);
Weights = TopProb ./ SumTopK;

% 向量化掩码分发: 让每个专家分别认领他专业领域里的词元
% 第一志愿
X3A = zeros(N, d);
for e = 1:numExperts
    % 本专业领域里的词元
    select = (TopIndices(:, 1) == e);

    % 门控SwiGLU网络
    X3A(select,:) = SwiGLU(X0(select,:), W0(:,:,e), b0(:,:,e), W1(:,:,e), b1(:,:,e), W2(:,:,e), b2(:,:,e));
end

% 第二志愿
X3B = zeros(N, d);
for e = 1:numExperts
    % 本专业领域里的词元
    select = (TopIndices(:, 2) == e);

    % 门控SwiGLU网络
    X3B(select,:) = SwiGLU(X0(select,:), W0(:,:,e), b0(:,:,e), W1(:,:,e), b1(:,:,e), W2(:,:,e), b2(:,:,e));
end

% 综合两个专家意见
% X3 = Weights(:,1) .* X3A + Weights(:,2) .* X3B;

% 导数反向传播
% 对 X3 = Weights(:,1) .* X3A + Weights(:,2) .* X3B 求导
% Weights(:,i)向第二维度广播，需把导数加总
dX3A = dX3 .* Weights(:, 1);
dX3B = dX3 .* Weights(:, 2);
dWeights = zeros(N, 2);
dWeights(:, 1) = sum(dX3 .* X3A, 2);
dWeights(:, 2) = sum(dX3 .* X3B, 2);

% 调用 gradSwiGLU 分别计算各个专家的导数
% 由于两个专家所用的输入变量完全相同，导数需加总
% 又由于X0既影响专家，又影响路由器，这两条路径的导数也需加总
dX0_experts = zeros(N, d);
dW0 = zeros(d, 4*d, numExperts); db0 = zeros(1, 4*d, numExperts);
dW1 = zeros(d, 4*d, numExperts); db1 = zeros(1, 4*d, numExperts);
dW2 = zeros(4*d, d, numExperts); db2 = zeros(1, d, numExperts);

% 第一志愿梯度的累加
for e = 1:numExperts
    select = (TopIndices(:, 1) == e);
    if any(select)        
        [dX0_e, dW0_e, db0_e, dW1_e, db1_e, dW2_e, db2_e] = gradSwiGLU(X0(select, :), dX3A(select, :),W0(:, :, e), b0(:, :, e), W1(:, :, e), b1(:, :, e), W2(:, :, e));        
        dX0_experts(select, :) = dX0_experts(select, :) + dX0_e;
        dW0(:,:,e) = dW0(:,:,e) + dW0_e; db0(:,:,e) = db0(:,:,e) + db0_e;
        dW1(:,:,e) = dW1(:,:,e) + dW1_e; db1(:,:,e) = db1(:,:,e) + db1_e;
        dW2(:,:,e) = dW2(:,:,e) + dW2_e; db2(:,:,e) = db2(:,:,e) + db2_e;
    end
end

% 第二志愿梯度的累加
for e = 1:numExperts
    select = (TopIndices(:, 2) == e);
    if any(select)
        [dX0_e, dW0_e, db0_e, dW1_e, db1_e, dW2_e, db2_e] = gradSwiGLU(X0(select, :), dX3B(select, :), W0(:, :, e), b0(:, :, e), W1(:, :, e), b1(:, :, e), W2(:, :, e));        
        dX0_experts(select, :) = dX0_experts(select, :) + dX0_e;
        dW0(:,:,e) = dW0(:,:,e) + dW0_e; db0(:,:,e) = db0(:,:,e) + db0_e;
        dW1(:,:,e) = dW1(:,:,e) + dW1_e; db1(:,:,e) = db1(:,:,e) + db1_e;
        dW2(:,:,e) = dW2(:,:,e) + dW2_e; db2(:,:,e) = db2(:,:,e) + db2_e;
    end
end

% Top-K 归一化求导： Weights = TopProb ./ SumTopK; 
% 根据商的求导法则推导出的化简公式：
dTopProb = zeros(N, 2);
dTopProb(:, 1) = (dWeights(:, 1) - dWeights(:, 2)) .* TopProb(:, 2) ./ (SumTopK.^2 + 1e-8);
dTopProb(:, 2) = (dWeights(:, 2) - dWeights(:, 1)) .* TopProb(:, 1) ./ (SumTopK.^2 + 1e-8);

% 将 TopK 的梯度原路返回给全部 numExperts 个概率 (这就是 maxk 的反向传播)
dProb = zeros(N, numExperts);
linearIdx1 = sub2ind([N, numExperts], (1:N)', TopIndices(:, 1));
dProb(linearIdx1) = dTopProb(:, 1);

linearIdx2 = sub2ind([N, numExperts], (1:N)', TopIndices(:, 2));
dProb(linearIdx2) = dTopProb(:, 2);

% Softmax 导数标准公式
dlogP = Prob .* (dProb - sum(Prob .* dProb, 2));

% 路由器的导数
dWGate = X0' * dlogP;
dX0_gate = dlogP * WGate'; 

% 两条路径 (路由器 + 专家) 的梯度加总
dX0 = dX0_gate + dX0_experts;

end