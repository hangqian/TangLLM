% 残差连接的"不废旧章"特色

rng('Default')

% "残"的嵌入
load('TrainedModel.mat','Parameters','vocabulary')
TokenEmbedding = Parameters.TokenEmbedding;
[~, ind] = ismember('残', vocabulary);
x = Parameters.TokenEmbedding(ind,:)';

% 激活函数
relu = @(z) max(0, z);
gelu = @(z) 0.5*z.*(1 + tanh(sqrt(2/pi)*(z + 0.044715*z.^3)));

% 运行两组实验
[cosPlainReLU, cosResidReLU] = MonteCarlo(x, relu);
[cosPlainGeLU, cosResidGeLU] = MonteCarlo(x, gelu);

% 直方图
figure('Color','w');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
edges = linspace(-1, 1, 80); 

% (1) ReLU
nexttile;
histogram(cosPlainReLU, edges, 'Normalization','probability', ...
    'FaceAlpha',0.55, 'EdgeColor','none');
hold on;
histogram(cosResidReLU, edges, 'Normalization','probability', ...
    'FaceAlpha',0.55, 'EdgeColor','none');
grid on;
xlabel('cos(x, f(x))');
ylabel('概率');
% legend('直接连接: \phi(Wx)', '残差连接: x + \phi(Wx)', 'Location','best');
title('ReLU激活函数');

% (2) GeLU
nexttile;
histogram(cosPlainGeLU, edges, 'Normalization','probability', ...
    'FaceAlpha',0.55, 'EdgeColor','none');
hold on;
histogram(cosResidGeLU, edges, 'Normalization','probability', ...
    'FaceAlpha',0.55, 'EdgeColor','none');
grid on;
xlabel('cos(x, f(x))');
legend('直接连接: \phi(Wx)', '残差连接: x + \phi(Wx)', 'Location','best');
title('GeLU激活函数');

% 打印统计量 
fprintf('\n===== ReLU =====\n');
fprintf('Plain: mean=%.4f, median=%.4f, std=%.4f\n', mean(cosPlainReLU), median(cosPlainReLU), std(cosPlainReLU));
fprintf('Res  : mean=%.4f, median=%.4f, std=%.4f\n', mean(cosResidReLU),   median(cosResidReLU),   std(cosResidReLU));

fprintf('\n===== GeLU =====\n');
fprintf('Plain: mean=%.4f, median=%.4f, std=%.4f\n', mean(cosPlainGeLU), median(cosPlainGeLU), std(cosPlainGeLU));
fprintf('Res  : mean=%.4f, median=%.4f, std=%.4f\n', mean(cosResidGeLU),   median(cosResidGeLU),   std(cosResidGeLU));

% -------------------- 子函数：单次批量模拟 --------------------
function [cosPlain, cosResid] = MonteCarlo(x, activation)
N = 10000;
d = length(x);
cosPlain = zeros(N,1);
cosResid   = zeros(N,1);
for t = 1:N
    % 随机线性变换权重
    W = randn(d) / sqrt(d);

    % 对比有无残差连接的前馈网络
    yPlain = activation(W*x);
    yResid = x + activation(W*x);

    % 余弦相似度
    cosPlain(t) = (x' * yPlain) / (norm(x)*norm(yPlain));
    cosResid(t)   = (x' * yResid)   / (norm(x)*norm(yResid));
end
end
