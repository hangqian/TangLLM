function [dX0,dW1,db1,dW2,db2] = gradFeedForward(X0,dX3,W1,b1,W2)
% 前馈网络的导数反向传播
%
% X0:  输入嵌入
% dX3: 后方传来的导数dL/dX3
% W1, b1, W2, b2: 模型参数
%
% dX0: 输出导数dL/dX0
% dW1,db1,dW2,db2: 参数的导数

% 向前传播 X0 -> X1 -> X2 -> X3
% 第一线性层，神经元数量倍增
X1 = X0 * W1 + b1;        

% 非线性激活层(GeLU)
X2 = X1 .* normcdf(X1);     

% 第二线性层，神经元数量缩回
% X3 = X2 * W2 + b2;

% 导数向后传播 dX3 -> dX2 -> dX1 -> dX0
% 第二线性层的梯度
dW2 = X2' * dX3;
db2 = sum(dX3, 1);
dX2 = dX3 * W2';

% 非线性激活层 GELU 导数： Phi(x) + x * phi(x)
dGELU = normcdf(X1) + X1 .* normpdf(X1);
dX1 = dX2 .* dGELU;

% 第一线性层的梯度
dW1 = X0' * dX1;
db1 = sum(dX1, 1);
dX0 = dX1 * W1';

end
