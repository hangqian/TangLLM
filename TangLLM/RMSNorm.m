function Y = RMSNorm(X, scale)
% RMSNorm 前向传播
% X: [N, d]
% scale: [1, d]

epsilon = 1e-6; 
RMS = sqrt(mean(X.^2, 2)) + epsilon;
Y = (X ./ RMS) .* scale;
end