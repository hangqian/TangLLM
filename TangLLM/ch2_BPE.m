% BPE分词教学演示代码
% 算法核心思想：
% 1. 统计语料中所有相邻符号对出现次数
% 2. 找到出现次数最多的一对 (a b)，且其间无句读
% 3. 把所有 (a b)合并为新符号ab
% 仅供教学演示，非通用代码，未考虑重叠符号对(a a a)等复杂情形

% 示例语料
lines = ["夜雨寄北"
    "君问归期未有期"
    "巴山夜雨涨秋池"
    "何当共剪西窗烛"
    "却话巴山夜雨时"];

%{
% 另一组示例语料
lines = ["越人语天姥，云霞明灭或可睹"
    "天姥连天向天横，势拔五岳掩赤城"
    "我欲因之梦吴越，一夜飞度镜湖月"
    "湖月照我影，送我至剡溪"
    "谢公宿处今尚在，渌水荡漾清猿啼"
    "脚著谢公屐，身登青云梯"];
%}

% 初始单字作词元
token = string(char(join(lines, '，'))');

% BPE迭代合并词元
numIter = 10;
CombinedWords = strings(numIter,1);
for n = 1:numIter
    % 相邻两个词组成词元对
    tokenPair = token(1:end-1) + token(2:end);
    
    % 凡遇句读不得合并，独自关进小黑屋
    tokenPair(contains(tokenPair, '，')) = missing;
    validPairs = rmmissing(tokenPair);    
        
    % 统计最高频词元对
    [words, ~, idx] = unique(validPairs);
    highFrequency = words(mode(idx));
    select = find(tokenPair == highFrequency);
    
    % 若无重复，则告终止
    if isscalar(select)
        break
    end
    
    % 合并最高频词元对
    select(diff(select) == 1) = [];
    token(select) = highFrequency;
    token(select+1) = [];
    CombinedWords(n) = highFrequency;
end

disp('BPE合并的字词')
disp(CombinedWords(1:n-1))
disp('BPE切分出的词元')
disp(token')
