function X3 = SwiGLU(X0, W0, b0, W1, b1, W2, b2)
% 门控SwiGLU网络

% 值
X1 = X0 * W0 + b0;

% 门: Swish 激活函数
G0 = X0 * W1 + b1;
G = G0 .* logitCDF(G0);

% 门控乘法
X2 = X1 .* G;

% 线性层
X3 = X2 * W2 + b2;

end

function y = logitCDF(x)
y = 1 ./ (1+exp(-x));
end
