% 没有归一化时的深层网络

rng('Default')


% 嵌入矩阵
load('TrainedModel.mat','Parameters','vocabulary');
TokenEmbedding = Parameters.TokenEmbedding;
Poems = readlines("Tang300.txt");
joinedStr = join(Poems);
[~, TokenID] = ismember(char(joinedStr), vocabulary);
TokenID = TokenID(TokenID~=0);
X = TokenEmbedding(TokenID,:);

% 三个增益系数
g_list   = [1, 1.838, 2];

% ---------------- 创建同一画布并排三图 ----------------
figure(1);
for k = 1:3
    g = g_list(k);

    % 蒙特卡罗模拟
    [mu_bar, sd_bar] = MonteCarlo(X, g, false);
   
    switch k
        case 1
            subplot(2,3,1);plot(1:L, mu_bar, '-o', 'LineWidth', 1.1, 'MarkerSize', 4); ylabel('均值') ;title(sprintf('W方差小 g=%.3g',g));
            subplot(2,3,4);plot(1:L, sd_bar, '-s', 'LineWidth', 1.1, 'MarkerSize', 4); ylabel('标准差') 
        case 2
            subplot(2,3,2);plot(1:L, mu_bar, '-o', 'LineWidth', 1.1, 'MarkerSize', 4); title(sprintf('W方差中 g=%.3g',g)); 
            subplot(2,3,5);plot(1:L, sd_bar, '-s', 'LineWidth', 1.1, 'MarkerSize', 4); 
        case 3
            subplot(2,3,3);plot(1:L, mu_bar, '-o', 'LineWidth', 1.1, 'MarkerSize', 4);   title(sprintf('W方差大 g=%.3g',g));
            subplot(2,3,6);plot(1:L, sd_bar, '-s', 'LineWidth', 1.1, 'MarkerSize', 4); 
    end
    xlabel('层数');
    
end

figure(2);
for k = 1:3
    g = g_list(k);

    % 蒙特卡罗模拟
    [mu_bar, sd_bar] = MonteCarlo(X, g, true);
   
    switch k
        case 1
            subplot(2,3,1);plot(1:L, mu_bar, '-o', 'LineWidth', 1.1, 'MarkerSize', 4); ylabel('均值') ;title(sprintf('W方差小 g=%.3g',g));
            subplot(2,3,4);plot(1:L, sd_bar, '-s', 'LineWidth', 1.1, 'MarkerSize', 4); ylabel('标准差') 
        case 2
            subplot(2,3,2);plot(1:L, mu_bar, '-o', 'LineWidth', 1.1, 'MarkerSize', 4); title(sprintf('W方差中 g=%.3g',g)); 
            subplot(2,3,5);plot(1:L, sd_bar, '-s', 'LineWidth', 1.1, 'MarkerSize', 4); 
        case 3
            subplot(2,3,3);plot(1:L, mu_bar, '-o', 'LineWidth', 1.1, 'MarkerSize', 4);   title(sprintf('W方差大 g=%.3g',g));
            subplot(2,3,6);plot(1:L, sd_bar, '-s', 'LineWidth', 1.1, 'MarkerSize', 4); 
    end
    xlabel('层数');
    
end

%% ---------------- 子函数：NoNorm 深层传播模拟 ----------------
function [muBar, sdBar] = MonteCarlo(X, g, normalize)
d = size(X,2);
L = 36;

% 激活函数：GeLU
gelu = @(z) 0.5*z.*(1 + tanh(sqrt(2/pi)*(z + 0.044715*z.^3)));

% 深层网络传播
muBar = zeros(L,1);
sdBar = zeros(L,1);
for l = 1:L
    % 线性变换的权重矩阵：正态随机数 N(0, g^2/d)
    W = (g / sqrt(d)) * randn(d);

    % 层归一化
    if normalize
        X = normalization(X);
    end

    % 线性变换和非线性激活，无归一化
    X = gelu(X * W);

    % 各层神经元的均值与方差
    [muBar(l), sdBar(l)] = rowStats(X);
end
end

%% ---------------- 子函数：每行统计再取平均 ----------------
function [mu_avg, sd_avg] = rowStats(H)
    row_mu = mean(H, 2);
    row_sd = std(H, 0, 2);
    mu_avg = mean(row_mu);
    sd_avg = mean(row_sd);
end


function Y = normalization(X)
% 归一化
mu = mean(X,2);
sigma = std(X,1,2)+1e-6;
Y = (X - mu)./sigma;
end

