% 全唐诗字频统计与词表截断（简化版）

% 读取语料
Poetry = readlines("Corpus.txt");

% 将所有诗句拼接为一个长字符串，再拆为单字序列
allChars = char(join(Poetry, ""));

% 统计每个字的出现次数
[uniqueChars, ~, idx] = unique(allChars);
charCounts = accumarray(idx, 1);

% 按频次从高到低排序
[sortedCounts, sortIdx] = sort(charCounts, 'descend');
sortedChars = uniqueChars(sortIdx);

% 绘制字频分布图（双对数坐标）
figure;
loglog(1:length(sortedCounts), sortedCounts, '.');
xlabel('字频排序名次');
ylabel('出现次数');
title('《全唐诗》字频分布');
