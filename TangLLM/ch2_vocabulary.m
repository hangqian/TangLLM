% 构建唐诗大模型的词表
% 并把诗句转换为词元编号

% 读取唐诗文本
Poetry = readlines("TangPoetry.txt");

% 五言诗部分
numWords = strlength(Poetry);
Poems = Poetry(numWords==12);

% 构建词汇表
vocabulary = unique(char(join(Poems, "")));

% 系统默认编码顺序（基于康熙字典的部首和笔画排列）
fprintf('词表中第一个字符是句号  %s\n',vocabulary(1))
fprintf('词表中第三千个字是逗号  %s\n',vocabulary(3000))
fprintf('前十个词元：')
disp(vocabulary(1:10))
fprintf('后十个词元：')
disp(vocabulary(end-9:end))
disp(' ')

% 词元编号：把每个汉字映射为整数号码
charMatrix = char(Poems);
[~, DataMatrix] = ismember(charMatrix, vocabulary);

% 《全唐诗》前两句诗及其词元编号
fprintf('第一句诗：%s\n词元编号\n',Poems(1));
disp(DataMatrix(1,:))
fprintf('第二句诗：%s\n词元编号\n',Poems(2));
disp(DataMatrix(2,:))

% 句末标记
EOS = length(vocabulary) + 1;
numObs = size(DataMatrix, 1);
DataMatrix = [DataMatrix, repmat(EOS, numObs, 1)];
