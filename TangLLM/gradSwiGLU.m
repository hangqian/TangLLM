function [dX0,dW0,db0,dW1,db1,dW2,db2] = gradSwiGLU(X0,dX3,W0, b0, W1, b1, W2)
% 门控SwiGLU网络的导数反向传播
%
% X0:  输入嵌入
% dX3: 后方传来的导数dL/dX3
% W0, b0, W1, b1, W2, b2: 模型参数
%
% dX0: 输出导数dL/dX0
% dW0,db1,dW1,db1,dW2,db2: 参数的导数

% 向前传播 X0 -> G  -> X2 -> X3
%         X0 -> X1 -> X2 -> X3

% 值
X1 = X0 * W0 + b0;

% 门: Swish 激活函数
G0 = X0 * W1 + b1;
sigma = logitCDF(G0);
G = G0 .* sigma;

% 门控乘法
X2 = X1 .* G;

% 线性层
% X3 = X2 * W2 + b2;

% 导数向后传播 dX3 -> dX2 -> dX1 (dG) -> dX0
% 第二线性层的梯度
dW2 = X2' * dX3;
db2 = sum(dX3, 1);
dX2 = dX3 * W2';

% 门控乘法 X2 = X1 .* G 导数
dX1 = dX2 .* G;
dG = X1 .* dX2;

% 门 G = G0 * logitCDF(G0) 导数
dG0 = dG .* (sigma + G.*(1-sigma));

% G0 = X0 * W1 + b1 导数
dW1 = X0' * dG0;
db1 = sum(dG0, 1);
dX0RouteOne = dG0 * W1';

% 值 X1 = X0 * W0 + b0 导数
dW0 = X0' * dX1;
db0 = sum(dX1, 1);
dX0RouteTwo = dX1 * W0';

% 两条路径的效应加总
dX0 = dX0RouteOne + dX0RouteTwo;

end

function y = logitCDF(x)
y = 1 ./ (1+exp(-x));
end
