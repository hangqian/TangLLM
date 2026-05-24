% 置换等变性

% 导入数据
load('TrainedModel.mat','Parameters','vocabulary')
TokenEmbedding = Parameters.TokenEmbedding;
Wq = Parameters.Wq(:,:,1);
Wk = Parameters.Wk(:,:,1);
Wv = Parameters.Wv(:,:,1);

% 查词表
[~, ind] = ismember(char("不知心恨谁"), vocabulary);

% 查嵌入表
X = TokenEmbedding(ind,:);

% 把"不知心恨谁" 置换为 "谁知心不恨"
P1 = [0 0 0 0 1; 0 1 0 0 0; 0 0 1 0 0; 1 0 0 0 0; 0 0 0 1 0];

% 把"不知心恨谁" 置换为 "知心不恨谁"
P2 = [0 1 0 0 0; 0 0 1 0 0; 1 0 0 0 0; 0 0 0 1 0;0 0 0 0 1];

% 把"不知心恨谁" 置换为 "谁不知心恨"
P3 = [0 0 0 0 1; 1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 0 0 0 1 0];

% 置换等变性
f = @(X) attnSingle(X,Wq,Wk,Wv);
gap1 = f(P1*X) - P1*f(X); 
gap2 = f(P2*X) - P2*f(X); 
gap3 = f(P3*X) - P3*f(X); 
disp(norm(gap1))
disp(norm(gap2))
disp(norm(gap3))


% 绘制置换矩阵示意图：展示P如何对调"不知心恨谁"的字词

chars = {'不','知','心','恨','谁'};
n = 5;

P1 = [0 0 0 0 1; 0 1 0 0 0; 0 0 1 0 0; 1 0 0 0 0; 0 0 0 1 0];
P2 = [0 1 0 0 0; 0 0 1 0 0; 1 0 0 0 0; 0 0 0 1 0; 0 0 0 0 1];
P3 = [0 0 0 0 1; 1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 0 0 0 1 0];

results = {{'谁','知','心','不','恨'}, ...
           {'知','心','不','恨','谁'}, ...
           {'谁','不','知','心','恨'}};

titles = {'P_1','P_2','P_3'};

figure('Position',[100 100 1100 420],'Color','w');

for k = 1:3
    ax = subplot(1,3,k);
    P = eval(['P' num2str(k)]);

    % 绘制矩阵网格
    hold on;
    matTop = 5.5;   % 矩阵顶部y坐标
    matBot = 0.5;   % 矩阵底部y坐标
    cellSz = 1;

    % 画方格和填色
    for i = 1:n
        for j = 1:n
            x0 = j - 0.5;
            y0 = matTop - i + 0.5 - 0.5;
            if P(i,j) == 1
                fill([x0 x0+cellSz x0+cellSz x0], ...
                     [y0 y0 y0+cellSz y0+cellSz], ...
                     [0.2 0.4 0.8], 'EdgeColor', [0.5 0.5 0.5], 'LineWidth', 0.5);
                text(x0+0.5, y0+0.5, '1', 'HorizontalAlignment','center', ...
                     'VerticalAlignment','middle', 'Color','w', 'FontSize', 13, 'FontWeight','bold');
            else
                fill([x0 x0+cellSz x0+cellSz x0], ...
                     [y0 y0 y0+cellSz y0+cellSz], ...
                     [0.95 0.95 0.97], 'EdgeColor', [0.7 0.7 0.7], 'LineWidth', 0.5);
            end
        end
    end

    % 原诗字（矩阵上方，对应列）
    for j = 1:n
        text(j, matTop + 0.7, chars{j}, 'HorizontalAlignment','center', ...
             'FontSize', 15, 'FontName','SimSun', 'Color', [0.1 0.1 0.1]);
    end

    % 结果字（矩阵左侧，对应行）
    res = results{k};
    for i = 1:n
        text(-0.3, matTop - i + 0.5, res{i}, 'HorizontalAlignment','center', ...
             'FontSize', 15, 'FontName','SimSun', 'Color', [0.8 0.15 0.15]);
    end

    % 画连线：从上方原字到对应的1所在位置
    for i = 1:n
        j = find(P(i,:) == 1);
        % 从列顶(原字)画箭头到该格
        xFrom = j;
        yFrom = matTop + 0.35;
        xTo = j;
        yTo = matTop - i + 0.5 + 0.5;
        % 从该格画箭头到行左(结果字)
        xFrom2 = 0.35;
        yFrom2 = matTop - i + 0.5;
        xTo2 = xFrom2;
        yTo2 = yFrom2;
    end

    %{
    % 标题
    text(3, matTop + 1.6, titles{k}, 'HorizontalAlignment','center', ...
          'FontSize', 16, 'FontWeight','bold');

    % 在矩阵下方写出结果诗句
    resStr = strjoin(res, '');
    text(3, -0.8, ['"' resStr '"'], 'HorizontalAlignment','center', ...
         'FontSize', 14, 'FontName','SimSun', 'Color', [0.8 0.15 0.15]);
    %}

    axis equal;
    axis([-1 6.2 -1.5 matTop + 2.2]);
    axis off;
    hold off;
end

% 总标题
% sgtitle('置换矩阵将 "不知心恨谁" 重排为三句歪诗', 'FontSize', 16, 'FontName','SimSun');

% 保存
exportgraphics(gcf, 'ch3_permutation.png', 'Resolution', 200);
fprintf('图片已保存: ch3_permutation.png\n');


function Y = attnSingle(X,Wq,Wk,Wv)
% 单头注意力模块

% 词嵌入向量做线性变换，分身为QKV三剑客
Q = X * Wq;
K = X * Wk;
V = X * Wv;

% 未归一化的权重分数
d = size(Q,2);
S = Q*K'/sqrt(d);

% SoftMax形式的权重
expS = exp(S - max(S, [], 2));
W = expS./sum(expS,2);

% 对上下文做加权平均
Y = W*V;
end

