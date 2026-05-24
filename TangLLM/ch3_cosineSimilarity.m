% 教学示例：从高维词嵌入到二维平面的保角投影
% 核心数学：施密特正交化 (Gram-Schmidt Process)

clear; clc; close all;

%% 1. 读取模型与词向量
load('TrainedModel.mat','Parameters','vocabulary');
TokenEmbedding = Parameters.TokenEmbedding;

Poems = "远看山有色近听水无声";
chars = char(Poems);
[~, ind] = ismember(chars, vocabulary);
X = TokenEmbedding(ind,:);  % 获取诗句中每个字的高维向量

n = size(X,1);

%% 2. 计算余弦相似性矩阵
% 向量点积除以各自的模长
norms = sqrt(sum(X.^2, 2));
C = (X * X') ./ (norms * norms');   

%% 3. 选取三对典型词元用于演示
% 为了代码简洁，这里我们直接挑出刚才算出来的极值对索引
C_no_diag = C; 
C_no_diag(logical(eye(n))) = NaN; % 排除自己和自己的相似度(对角线)

[~, posIdx]  = max(C_no_diag(:), [], 'omitnan');       % 余弦最大（夹角最小）
[~, weakIdx] = min(abs(C_no_diag(:)), [], 'omitnan');  % 余弦最接近0（正交/垂直）
[~, negIdx]  = min(C_no_diag(:), [], 'omitnan');       % 余弦最小（反向）

[iPos, jPos]   = ind2sub([n,n], posIdx);
[iWeak, jWeak] = ind2sub([n,n], weakIdx);
[iNeg, jNeg]   = ind2sub([n,n], negIdx);

pairs = [iPos, jPos; iWeak, jWeak; iNeg, jNeg];
titles = {"夹角最小 (正相关)", "接近垂直 (弱相关)", "夹角最大 (负相关)"};

%% 4. 降维投影与可视化
fig = figure('Color','w','Position',[100,100,1200,400]);
tiledlayout(1, 3, 'TileSpacing', 'compact');

for k = 1:3
    wordA = X(pairs(k,1), :);
    wordB = X(pairs(k,2), :);
    
    % 【核心步骤】：将两个高维向量投影到它们自身的二维平面
    [vecA_2D, vecB_2D] = project2D(wordA, wordB);
    
    % 绘图
    nexttile;
    drawVectors(vecA_2D, vecB_2D, chars(pairs(k,1)), chars(pairs(k,2)), titles{k});
end


%% ============================================================
% 【施密特正交化】
%% ============================================================
function [A2D, B2D] = project2D(A, B)
    % 本函数利用“施密特正交化”，在 A 和 B 张成的高维空间切面上，
    % 建立一个全新的二维直角坐标系 (X轴 和 Y轴)。
    
    % ----------------------------------------------------
    % 第一步：建立 X 轴 (以 A 的方向作为基准)
    % ----------------------------------------------------
    % 取 A 的单位向量作为 X 轴正方向 (基向量 e1)
    normA = norm(A);
    x = A / normA;  
    
    % ----------------------------------------------------
    % 第二步：寻找 Y 轴 (必须与 X 轴垂直)
    % ----------------------------------------------------
    % 1. 先算出 B 在 X 轴方向上的“影子” (平行分量)
    %    公式：(B点乘X轴单位向量) * X轴单位向量
    BParallel = dot(B, x) * x;
    
    % 2. 剔除平行分量，剩下的就是垂直于 X 轴的“正交分量”
    BPerp = B - BParallel;
    
    % 3. 将这个垂直分量归一化，作为 Y 轴正方向 (基向量 e2)
    y = BPerp / norm(BPerp);
    
    % ----------------------------------------------------
    % 第三步：计算两个向量在新坐标系下的二维坐标
    % ----------------------------------------------------
    % A 本就在 X 轴上，所以它的 X 坐标就是自己的长度，Y 坐标为 0
    A2D = [normA, 0];
    
    % B 的二维坐标，就是它分别在 X 轴和 Y 轴方向上的投影长度
    B2D = [dot(B, x), dot(B, y)];
end


%% ============================================================
% 辅助绘图函数 
%% ============================================================
function drawVectors(v1, v2, label1, label2, ttl)
    hold on; axis equal; grid on; box on;
    
    % 1. 动态确定坐标轴范围
    maxL = 1.2 * max([norm(v1), norm(v2)]);
    axis([-maxL, maxL, -maxL, maxL]);
    plot([-maxL maxL], [0 0], 'k-', 'Color', [0.8 0.8 0.8]); % X轴
    plot([0 0], [-maxL maxL], 'k-', 'Color', [0.8 0.8 0.8]); % Y轴
    
    % 2. 画向量箭头
    quiver(0, 0, v1(1), v1(2), 0, 'LineWidth', 2.5, 'Color', '#0072BD', 'MaxHeadSize', 0.2);
    quiver(0, 0, v2(1), v2(2), 0, 'LineWidth', 2.5, 'Color', '#D95319', 'MaxHeadSize', 0.2);
    
    % 3. 标注词元文字
    text(v1(1)*1.06, v1(2)*1.06, label1, 'FontSize', 14, 'Color', '#0072BD', 'FontWeight', 'bold');
    text(v2(1)*1.06, v2(2)*1.06, label2, 'FontSize', 14, 'Color', '#D95319', 'FontWeight', 'bold');
    
    % =========================================================
    % 4. 计算并绘制夹角弧线与度数
    % =========================================================
    % 计算余弦值，并强制限制在[-1, 1]区间，防止因浮点误差导致 acos 产生复数
    cos_val = dot(v1, v2) / (norm(v1) * norm(v2));
    cos_val = max(-1, min(1, cos_val));
    theta_deg = rad2deg(acos(cos_val)); % 转换为角度
    
    % 获取两个向量相对于 X轴 的极角 (范围 [-pi, pi])
    phi1 = atan2(v1(2), v1(1));
    phi2 = atan2(v2(2), v2(1));
    
    % 计算角度差，并通过 mod 操作确保画的是“劣弧”（即小于等于 180° 的那个夹角）
    dphi = mod(phi2 - phi1 + pi, 2*pi) - pi;
    
    % 设定圆弧的半径 (取较短向量长度的 35%)
    arc_R = 0.35 * min(norm(v1), norm(v2));
    
    % 生成圆弧上的点并绘制
    t = linspace(phi1, phi1 + dphi, 50);
    plot(arc_R * cos(t), arc_R * sin(t), 'Color', '#77AC30', 'LineWidth', 2);
    
    % 在圆弧外侧一点居中写上夹角度数
    phi_mid = phi1 + dphi / 2;
    text(arc_R * 1.35 * cos(phi_mid), arc_R * 1.35 * sin(phi_mid), ...
        sprintf('\\theta = %.1f^\\circ', theta_deg), ...
        'Color', '#77AC30', 'FontSize', 12, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center');
    % =========================================================

    % 5. 设置标题与清爽外观
    title(sprintf('%s\ncos = %.3f', ttl, cos_val), 'FontSize', 12);
    set(gca, 'XTick', [], 'YTick', []); % 隐藏刻度
end
