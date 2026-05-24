% 示例：查询词"光"对"床前明月光"的三种相似度指标对比
% 点积、协方差、余弦相似度

load('TrainedModel.mat','Parameters','vocabulary')
TokenEmbedding = Parameters.TokenEmbedding;
Poems = "床前明月光";
[~, ind] = ismember(char(Poems), vocabulary);
X = TokenEmbedding(ind,:);
n = size(X,1);
d = size(X,2);

labels = {'床','前','明','月','光'};
q = X(end,:);  % 查询词"光"

% 1. 点积: X_i * X_j'
dotProd = X * q';

% 2. 协方差: Cov(X_i, X_j) = E(X_i .* X_j) - E(X_i)*E(X_j)
covVals = zeros(n,1);
meanQ = mean(q);
for i = 1:n
    covVals(i) = mean(X(i,:) .* q) - mean(X(i,:)) * meanQ;
end

% 3. 余弦相似度: cos(X_i, X_j) = (X_i * X_j') / (||X_i|| * ||X_j||)
normX = sqrt(sum(X.^2, 2));
normQ = norm(q);
cosVals = dotProd ./ (normX * normQ);

% 绘制三组柱状图
figure('Position',[200 200 1500 500]);

% --- 子图1：点积 ---
subplot(1,3,1);
b1 = bar(dotProd, 'FaceColor', [0.2 0.6 0.9]);
set(gca,'XTickLabel',labels,'FontSize',12);
% ylabel('点积','FontSize',13);
title('点积','FontSize',14);
ylim([-0.5 2.2]);
xtips = b1.XEndPoints; ytips = b1.YEndPoints;
txt = compose('%.3f', dotProd');
for k = 1:n
    if dotProd(k) >= 0
        text(xtips(k), ytips(k)+0.05, txt{k}, 'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom', 'FontSize',10);
    else
        text(xtips(k), ytips(k)-0.05, txt{k}, 'HorizontalAlignment','center', ...
            'VerticalAlignment','top', 'FontSize',10);
    end
end

% --- 子图2：协方差 ---
subplot(1,3,2);
b2 = bar(covVals*1000, 'FaceColor', [0.2 0.6 0.9]);  % 放大1000倍便于显示
set(gca,'XTickLabel',labels,'FontSize',12);
% ylabel('协方差 (×10^{-3})','FontSize',13);
title('协方差','FontSize',14);
ylim([-1 17]);
xtips = b2.XEndPoints; ytips = b2.YEndPoints;
covDisp = covVals*1000;
txt = compose('%.2f', covDisp');
for k = 1:n
    if covDisp(k) >= 0
        text(xtips(k), ytips(k)+0.2, txt{k}, 'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom', 'FontSize',10);
    else
        text(xtips(k), ytips(k)-0.2, txt{k}, 'HorizontalAlignment','center', ...
            'VerticalAlignment','top', 'FontSize',10);
    end
end

% --- 子图3：余弦相似度 ---
subplot(1,3,3);
b3 = bar(cosVals, 'FaceColor', [0.2 0.6 0.9]);
set(gca,'XTickLabel',labels,'FontSize',12);
% ylabel('余弦相似度','FontSize',13);
title('余弦相似度','FontSize',14);
ylim([-0.2 1.1]);
xtips = b3.XEndPoints; ytips = b3.YEndPoints;
txt = compose('%.3f', cosVals');
for k = 1:n
    if cosVals(k) >= 0
        text(xtips(k), ytips(k)+0.02, txt{k}, 'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom', 'FontSize',10);
    else
        text(xtips(k), ytips(k)-0.02, txt{k}, 'HorizontalAlignment','center', ...
            'VerticalAlignment','top', 'FontSize',10);
    end
end

% 保存
print('ch3_queryLight_comparison', '-dpng', '-r150');

% 打印数值对比表
fprintf('\n查询词"光"对"床前明月光"各词的三种相似度指标：\n');
fprintf('%-4s  %8s  %10s  %8s\n', '词元', '点积', '协方差', '余弦');
fprintf('%-4s  %8s  %10s  %8s\n', '----', '------', '--------', '------');
for i = 1:n
    fprintf('%-4s  %8.3f  %10.4f  %8.3f\n', labels{i}, dotProd(i), covVals(i), cosVals(i));
end

% 展示比例关系
fprintf('\n各指标的比例关系（以"月"为基准=1）：\n');
fprintf('%-4s  %8s  %10s  %8s\n', '词元', '点积比', '协方差比', '余弦比');
fprintf('%-4s  %8s  %10s  %8s\n', '----', '------', '--------', '------');
for i = 1:n
    fprintf('%-4s  %8.3f  %10.3f  %8.3f\n', labels{i}, ...
        dotProd(i)/dotProd(4), covVals(i)/covVals(4), cosVals(i)/cosVals(4));
end
