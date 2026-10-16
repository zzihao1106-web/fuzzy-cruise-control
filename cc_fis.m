function fis = cc_fis(saveDir)
% 构建定速巡航用模糊推理系统
%
%   输入1  e   车速误差 = 目标车速 - 实际车速  （归一化到 [-1, 1]）
%   输入2  ec  误差变化率 de/dt                （归一化到 [-1, 1]）
%   输出   dA  油门开度增量                     （归一化到 [-1, 1]）
%
% 语言变量 5 档：NB 负大、NS 负小、ZO 零、PS 正小、PB 正大
% 规则 5 x 5 = 25 条

if nargin < 1 || isempty(saveDir)
    saveDir = fullfile(fileparts(mfilename('fullpath')), 'fis');
end
if ~exist(saveDir, 'dir'); mkdir(saveDir); end

lbl = {'NB','NS','ZO','PS','PB'};
% 5 个均匀分布的三角隶属函数，论域 [-1, 1]
mf  = [-1   -1   -0.5;
       -1   -0.5  0  ;
       -0.5  0    0.5;
        0    0.5  1  ;
        0.5  1    1  ];

fis = mamfis('Name', 'cruise_fuzzy', ...
             'AndMethod', 'min', 'OrMethod', 'max', ...
             'ImplicationMethod', 'min', 'AggregationMethod', 'max', ...
             'DefuzzificationMethod', 'centroid');

fis = addInput(fis,  [-1 1], 'Name', 'e');
fis = addInput(fis,  [-1 1], 'Name', 'ec');
fis = addOutput(fis, [-1 1], 'Name', 'dA');

for i = 1:5
    fis = addMF(fis, 'e',  'trimf', mf(i,:), 'Name', lbl{i});
    fis = addMF(fis, 'ec', 'trimf', mf(i,:), 'Name', lbl{i});
    fis = addMF(fis, 'dA', 'trimf', mf(i,:), 'Name', lbl{i});
end

% 规则表（行 = e，列 = ec）：车速越低、而且还在掉，油门就加得越狠
%                                              ec
%                                    NB   NS   ZO   PS   PB
tbl = [1 1 1 2 3;   % e = NB  车速远高于目标
       1 1 2 3 4;   % e = NS
       1 2 3 4 5;   % e = ZO
       2 3 4 5 5;   % e = PS
       3 4 5 5 5];  % e = PB  车速远低于目标

rules = zeros(25, 5);
n = 0;
for i = 1:5
    for j = 1:5
        n = n + 1;
        rules(n,:) = [i, j, tbl(i,j), 1, 1];   % [e  ec  输出  权重  连接方式(1=AND)]
    end
end
fis = addRule(fis, rules);

try
    writeFIS(fis, fullfile(saveDir, 'cruise_fuzzy'));
catch ME
    warning('writeFIS 落盘失败：%s', ME.message);
end
end
