% 全唐诗字频统计与词表截断
%  输入：Corpus.txt（已清洗的纯诗句文本）
%  输出：字频分布图 + 高频3000字词表


%% 1. 读取语料，提取全部汉字
raw = fileread('Corpus.txt');
corpus = string(raw);

% 仅保留汉字（Unicode 范围 4E00-9FFF），去除标点和空白
chars = char(corpus);
isHanzi = chars >= 0x4E00 & chars <= 0x9FFF;
hanzi = chars(isHanzi);

fprintf('语料汉字总数：%d\n', numel(hanzi));

%% 2. 统计字频，按频次降序排列
[uniqueChars, ~, idx] = unique(hanzi);
freq = accumarray(idx, 1);
[freq, order] = sort(freq, 'descend');
uniqueChars = uniqueChars(order);

fprintf('不重复汉字数：%d\n', numel(uniqueChars));

%% 3. 截取高频 3000 字
topK = 3000;
topChars = uniqueChars(1:topK);
coverage = cumsum(freq) / sum(freq);
coverageAtK = coverage(topK);

fprintf('前 %d 字覆盖率：%.2f%%\n', topK, 100*coverageAtK);
fprintf('高频字前20：%s\n', topChars(1:20));

%% 4. 绘制字频分布与词表截断图
fig = figure('Color','w', 'Position',[100 100 1200 480]);
tiledlayout(1, 2, 'TileSpacing','compact', 'Padding','compact');

% --- 左图：字频分布（双对数坐标） ---
nexttile;
rank = 1:numel(freq);
loglog(rank, freq, 'Color',[0.15 0.40 0.75], 'LineWidth',1.5);
hold on;
xline(topK, '--', sprintf('高频 %d', topK), ...
    'Color',[0.85 0.20 0.20], 'LineWidth',1.5, ...
    'LabelOrientation','horizontal', 'LabelVerticalAlignment','middle');

% 标注最高频的几个字
for i = 1:7
    text(rank(i)*1.2, freq(i)*0.8, string(uniqueChars(i)), 'FontSize',10);
end
grid on;
xlabel('字频排序名次');
ylabel('出现次数');
title('字频分布（按频次降序）');

% --- 右图：累计覆盖率 ---
nexttile;
plot(rank, coverage, 'Color',[0.15 0.40 0.75], 'LineWidth',2);
hold on; grid on;
xline(topK, '--', sprintf('高频 %d', topK), ...
    'Color',[0.85 0.20 0.20], 'LineWidth',1.5, ...
    'LabelOrientation','horizontal', 'LabelVerticalAlignment','middle');
yline(coverageAtK, ':', sprintf('覆盖率 %.2f%%', 100*coverageAtK), ...
    'Color',[0.20 0.55 0.25], 'LineWidth',1.5, ...
    'LabelHorizontalAlignment','left');
plot(topK, coverageAtK, 'o', 'MarkerSize',8, ...
    'MarkerFaceColor',[0.85 0.20 0.20], 'MarkerEdgeColor','none');
xlabel('词表大小（按字频排序）');
ylabel('累计覆盖率');
title('词表截断与累计覆盖率');
ylim([0 1.02]);

% exportgraphics(fig, '全唐诗字频分布与词表截断示意图.png', 'Resolution',300);
% fprintf('图已保存。\n');
