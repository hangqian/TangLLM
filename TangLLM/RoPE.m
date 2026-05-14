function Y = RoPE(X, numHeads, direction, positions)
% 旋转位置编码 (按头子维度旋转的实数版本)
%
% X:         n*d 输入嵌入
% numHeads:  注意力头数，默认为 1
% direction: 'forward' 正向旋转;  'backward' 反向旋转
% positions: 绝对位置索引向量（可选）。若不传，则默认从 1 到 n。

if nargin < 2 || isempty(numHeads); numHeads = 1; end
if nargin < 3 || isempty(direction); direction = 'forward'; end

% 矩阵维度
[n, d] = size(X);

% 兼容推理阶段：如果传入了绝对位置，就用绝对位置；否则默认 1:n
if nargin < 4 || isempty(positions)
    positions = (1:n)';
else
    % 确保是列向量
    positions = positions(:); 
end

dk = d / numHeads; 
assert(mod(d, numHeads) == 0, '总维度 d 必须能被 numHeads 整除');
assert(mod(dk, 2) == 0, '子维度 dk 必须是偶数，RoPE 需要两两配对');

% 生成单个头的角速度 (长度为 dk/2)
theta_head = 10000.^(-2 * ((0:(dk/2-1)) / dk));
theta = repmat(theta_head, 1, numHeads);

% 旋转角度
angles = positions * theta;

cos_val = cos(angles);
sin_val = sin(angles);

% 反向旋转是转置(逆矩阵)
if strcmpi(direction,'backward')
    sin_val = -sin_val;
end

% 构造正交矩阵
Y = zeros(n, d, 'like', X);
X_even = X(:, 1:2:end);
X_odd  = X(:, 2:2:end);

Y(:, 1:2:end) = X_even .* cos_val - X_odd .* sin_val;
Y(:, 2:2:end) = X_even .* sin_val + X_odd .* cos_val;
end
