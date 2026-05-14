function X3 = feedforward(X0, W1, b1, W2, b2)
% 前馈网络模块

% 线性层，神经元数量倍增
X1 = X0 * W1 + b1;

% GELU激活层
% X2 = 0.5 .* X1 .* (1 + tanh(sqrt(2/pi) .* (X1 + 0.044715 .* X1.^3)));
X2 = X1 .* normcdf(X1);

% 线性层，神经元数量缩回
X3 = X2 * W2 + b2;

end
