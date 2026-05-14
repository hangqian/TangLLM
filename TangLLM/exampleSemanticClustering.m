% 在二维平面上展示高维词嵌入向量的语义聚类
close all

% 导入模型参数
load('TrainedModel.mat','Parameters','vocabulary')
TokenEmbedding = Parameters.TokenEmbedding;

% 初始分布是随机数
rng(12345)
std0 = 0.02; 
TokenEmbedding0 = std0 .* randn(size(TokenEmbedding));

% 绘制聚类图
plot_semantic_clustering(TokenEmbedding0, vocabulary);
plot_semantic_clustering(TokenEmbedding, vocabulary);


% =========================================================================
% 词嵌入空间语义聚类可视化探测器
% =========================================================================
function plot_semantic_clustering(TokenEmbedding, vocabulary)
    % 1. 利用 SVD 手搭 PCA 降维，摆脱对任何工具箱的依赖，将高维压缩到 2 维
    X = TokenEmbedding;
    X = X - mean(X, 1); % 中心化
    [~, ~, V] = svd(X, 'econ');
    X_2d = X * V(:, 1:2); % 取前两个主成分投影

    % 2. 定义几个典型特征词族
    word_groups = {
        ["春", "夏", "秋", "冬"], ...                 % 四季
        ["东", "南", "西", "北"], ...                 % 方位
        ["风", "雨", "雪", "霜", "云"], ...           % 气象
        ["红", "黄", "绿", "白", "青", "黑", "紫"], ... % 颜色
        ["一", "二", "三", "四", "五", "六", "千", "万"] % 数字
    };

    colors = lines(length(word_groups)); % 生成不同的颜色

    % 3. 画图
    % figure('Name', title_str, 'Position', [100, 100, 900, 700], 'Color', 'w');
    figure('Name', "语义聚类", 'Position', [100, 100, 900, 700], 'Color', 'w')
    hold on;

    % 先把所有的字（3000多个）用极浅的灰色画成背景星空
    scatter(X_2d(:, 1), X_2d(:, 2), 15, [0.85 0.85 0.85], 'filled', 'MarkerFaceAlpha', 0.6);

    % 计算 X 轴的极差（最大值减最小值），取其 1.5% 作为文字偏移量
    x_offset = (max(X_2d(:, 1)) - min(X_2d(:, 1))) * 0.015;

    % 遍历每个词族，高亮显示
    for g = 1:length(word_groups)
        group_words = word_groups{g};
        
        % 提取当前词族在字典中的索引
        [~, idx] = ismember(char(join(group_words,"")), vocabulary);
        
        % 过滤掉可能不在字典里的字
        valid_mask = idx > 0;
        idx = idx(valid_mask);
        words = group_words(valid_mask);
        
        if isempty(idx), continue; end
        
        % 提取这些字在 2D 空间的坐标
        x_coords = X_2d(idx, 1);
        y_coords = X_2d(idx, 2);
        
        % 画出高亮散点
        scatter(x_coords, y_coords, 80, colors(g,:), 'filled', 'MarkerEdgeColor', 'k');

        % 标上汉字 (使用自适应偏移量 x_offset)
        for i = 1:length(idx)
            text(x_coords(i) + x_offset, y_coords(i), char(words(i)), ...
                'FontName', 'Microsoft YaHei', 'FontSize', 14, 'FontWeight', 'bold', ...
                'Color', colors(g,:));
        end        
        
    end

    % title("语义聚类", 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('主成分 1 (第一最大方差方向)'); ylabel('主成分 2 (第二最大方差方向)');
    grid on; hold off;
end
