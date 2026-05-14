% 《全唐诗》数据清理
%  仅保留包含三千常用字的诗句，构建五言与七言混合的大语言模型语料库

fprintf('正在读取文本文件...\n');
rawText = fileread('Corpus.txt'); 

% 按行分割文本
lines = splitlines(rawText);
numLines = length(lines);

% 预分配一个大的字符串数组存储有效的诗句 (10字或14字)
validCouplets = strings(numLines, 1);
coupletCount = 0;

fprintf('正在清洗数据并提取【五言】与【七言】诗句...\n');
for i = 1:numLines
    str = strtrim(lines{i});
    
    % 1. 过滤空行和包含标题/卷号的行
    if strlength(str) == 0 || contains(str, '卷') || contains(str, '【')
        continue;
    end
    
    % 2. 清理噪音数据
    str = regexprep(str, '--.*', ''); % 去掉联句作者署名
    str = regexprep(str, '[a-zA-Z]', ''); % 去掉意外的英文字母
    str = regexprep(str, '[知古主斋]$', ''); % 去掉行尾的单字标记
    str = regexprep(str, '□', '');  % 去除缺失字占位符
    str = regexprep(str, '\s', ''); % \s 会匹配并去除所有半角空格、全角空格和制表符
    
    % 3. 按句号拆分成独立的长句
    sentences = split(str, '。');
    
    for j = 1:length(sentences)
        sen = strtrim(sentences{j});
        if strlength(sen) == 0
            continue;
        end

        % 4. 寻找逗号，拆分上下半句
        parts = split(sen, '，');
        
        if length(parts) == 2
            half1 = strtrim(parts{1});
            half2 = strtrim(parts{2});
            len1 = strlength(half1);
            len2 = strlength(half2);
            
            % 5. 提取五言 (5+5) 或 七言 (7+7)
            if (len1 == 5 && len2 == 5) || (len1 == 7 && len2 == 7)
                coupletCount = coupletCount + 1;
                
                % 把逗号和句号拼进去，赋予模型明确的语法边界
                validCouplets(coupletCount) = strcat(half1, "，", half2, "。");
            end
        end
    end
end

% 截断未使用的预分配空间
validCouplets = validCouplets(1:coupletCount);
fprintf('提取完成！共找到 %d 句格式工整的唐诗。\n\n', coupletCount);

% =========================================================================
% 统计字频，寻找 Top 3000 常用字
% =========================================================================
fprintf('正在统计字频并截取前 3000 个高频字...\n');

allChars = char(join(validCouplets, ""));
[uniqueChars, ~, idx] = unique(allChars);
charCounts = accumarray(idx, 1); 

[~, sortIdx] = sort(charCounts, 'descend');
topK = min(3000, length(uniqueChars));
topChars = uniqueChars(sortIdx(1:topK));
fprintf('高频字集（前500）\n');
disp(topChars(1:500))

% 构建快速查找表：isTopChar(码点) = true 表示该字属于高频字集
isTopChar = false(1, max(double(allChars)));
isTopChar(double(topChars)) = true;
fprintf('高频字集构建完毕，包含 %d 个汉字。\n', topK);

% =========================================================================
% 严格过滤：剔除包含生僻字的诗句
% =========================================================================
fprintf('正在过滤包含生僻字的诗句...\n');

keepIdx = false(coupletCount, 1);
for i = 1:coupletCount
    codes = double(char(validCouplets(i)));
    keepIdx(i) = all(isTopChar(codes));
end

% 提取最终的纯净字符串向量
PoetryStrings = validCouplets(keepIdx);
keepCount = length(PoetryStrings);

fprintf('过滤完成！保留了 %d 句完全由高频字组成的纯净诗句。\n', keepCount);

% 仅保存这个纯净的字符串向量
save('TangPoetry.mat', 'PoetryStrings');
fprintf('数据已保存至 TangPoetry.mat\n');