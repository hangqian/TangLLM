% 动手实践：两词对决——Softmax、对数几率与交叉熵
clear; clc; close all;

% --------- Step 1：分差 Δ 扫描 ----------
Delta = linspace(-6, 6, 500)';  % Δ = s1 - s2

% 为了突出“只与分差有关”，我们设 s2=0, s1=Δ（相当于平移不影响概率）
s1 = Delta;
s2 = zeros(size(Delta));

% 二分类 Softmax：p1 = exp(s1)/(exp(s1)+exp(s2))
% 用数值稳定写法：先减去每行最大值（护身符）
Score = [s1, s2];                          % N×2
logProb = Score - max(Score, [], 2);       % N×2
ProbRaw = exp(logProb);                    % N×2
Prob = ProbRaw ./ sum(ProbRaw, 2);         % N×2
p1 = Prob(:,1);

% --------- Step 2：交叉熵损失（真实词 = 1） ----------
loss = -log(p1 + 1e-12);  % 加小量防止 log(0)

% --------- 作图：两张核心曲线 ----------
figure('Color','w');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% (a) 概率曲线：p1 vs Δ
nexttile;
plot(Delta, p1, 'LineWidth', 1.5); grid on;
xlabel('\Delta = s_1 - s_2');
ylabel('p_1');
title('逻辑斯蒂分布S曲线');
yline(0.5,'--'); xline(0,'--');

% (b) 损失曲线：-log p1 vs Δ
nexttile;
plot(Delta, loss, 'LineWidth', 1.5); grid on;
xlabel('\Delta = s_1 - s_2');
ylabel('loss = -log(p_1)');
title('交叉熵惩罚');
xline(0,'--');

% --------- Step 3：数值稳定性演示 ----------
% 理论：Softmax 只看相对差异，(s1,s2) 同时加常数 C，概率不变
C = 800; % 故意取大，触发 exp 溢出风险（你可改更大/更小）
ScoreShift = Score + C;

% 直接算（可能 Inf 或 NaN）
Prob_bad = exp(ScoreShift) ./ sum(exp(ScoreShift),2);

% 稳定算（护身符：减去每行最大值）
logProbShift = ScoreShift - max(ScoreShift,[],2);
Prob_good = exp(logProbShift) ./ sum(exp(logProbShift),2);

fprintf('【数值稳定性演示】\n');
fprintf('直接计算是否出现 NaN/Inf：%d\n', any(~isfinite(Prob_bad(:))));
fprintf('稳定计算是否全部有限：%d\n', all(isfinite(Prob_good(:))));
fprintf('平移前后最大概率差（稳定算法）：%.3e\n', max(abs(Prob_good(:,1) - Prob(:,1))));


%% 三词混战：第三个词如何改变概率分布

Delta = linspace(-6, 6, 500)';   % 推 vs 敲 的分差

c_list = [-2, 0, 2];             % 第三个词的得分（弱/中/强竞争者）

figure('Color','w');
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

for k = 1:length(c_list)
    c = c_list(k);

    s1 = Delta;
    s2 = zeros(size(Delta));
    s3 = c * ones(size(Delta));

    Score = [s1, s2, s3];

    % Softmax（稳定版）
    logProb = Score - max(Score, [], 2);
    ProbRaw = exp(logProb);
    Prob = ProbRaw ./ sum(ProbRaw, 2);

    p1 = Prob(:,1);   % 推
    p2 = Prob(:,2);   % 敲
    p3 = Prob(:,3);   % 干扰项

    nexttile;
    plot(Delta, p1, 'LineWidth',1.5); hold on;
    plot(Delta, p2, 'LineWidth',1.5);
    plot(Delta, p3, '--', 'LineWidth',1.3);

    grid on;
    xlabel('\Delta = s_1 - s_2');    
    title(sprintf('s_3 = %.1f', c));

    if k == 1        
        ylabel('Probability');
    end
    if k == 2
        legend('p_1','p_2','p_3','Location','northwest');
    end   
end