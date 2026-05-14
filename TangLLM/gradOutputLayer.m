function [loss,dX, dW, db] = gradOutputLayer(X,TargetLabel,W,b)
% 输出层的导数反向传播
%
% X:           n*d 输入嵌入(模型预测的下一个词嵌入）
% Targetlabel: n*1 真实标签
% W,b:         模型参数
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

% 计算负似然函数值(亦即交叉熵损失）
[N,v] = size(Prob);
Y = false(N, v);
Y(sub2ind([N, v], (1:N)', TargetLabel)) = true;
loss = sum(-log(Prob(Y)+1e-9)) / N;

% 导数反向传播
if nargout > 1
    % 负似然函数对评分S求导：预测概率-真实标签
    dS = (Prob - Y) / N;

    % 根据链式法则和线性函数S=X*W+b，得到关于X,W,b的导数
    dW = X' * dS;
    db = sum(dS, 1);
    dX = dS * W';
end

end
