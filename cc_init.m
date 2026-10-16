function cc_init()
% 模型预加载（由 CruiseFuzzy.slx 的 PreLoadFcn 回调自动调用）
% 把参数、模糊推理系统和工况常数写入基础工作区。
% 注意：必须 assignin 到 base，Simulink 的参数才能解析到。

here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
addpath(here);

p   = cc_params();
fis = cc_fis(fullfile(here, 'fis'));

assignin('base', 'p', p);
assignin('base', 'cruise_fuzzy', fis);

% 坡度常数：默认 0（平路）。想跑上坡，把模型里 grade 方块的值改成 p.grade
if evalin('base', 'exist(''grade'',''var'')') == 0
    assignin('base', 'grade', 0);
end
end
