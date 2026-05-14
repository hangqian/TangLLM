function [dX, dscale] = gradRMSNorm(X, dY, scale)
% RMSNorm 反向传播
epsilon = 1e-6;
RMS = sqrt(mean(X.^2, 2)) + epsilon; % [N, 1]

% 缓存归一化后的 X
X_norm = X ./ RMS; % [N, d]

% 求缩放比例参数的导数
dscale = sum(dY .* X_norm, 1); % [1, d]

% 求 X 的导数
dY_norm = dY .* scale; % [N, d]
dX = (dY_norm - X_norm .* mean(dY_norm .* X_norm, 2)) ./ RMS;
end