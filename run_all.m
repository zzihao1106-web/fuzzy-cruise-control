function run_all()
% 一键复现全部结果：模糊推理系统 -> Simulink 模型 -> 仿真与配图
here = fileparts(mfilename('fullpath'));
addpath(here);
set(0, 'DefaultAxesFontName', 'Microsoft YaHei');
set(0, 'DefaultTextFontName', 'Microsoft YaHei');

fprintf('\n########## 1/3  构建模糊推理系统 ##########\n');
cc_fis(fullfile(here, 'fis'));

fprintf('\n########## 2/3  生成 Simulink 模型 ##########\n');
cc_build();

fprintf('\n########## 3/3  仿真并出图 ##########\n');
cc_run();

fprintf('\n全部完成，结果见 %s\n', fullfile(here, 'results'));
end
