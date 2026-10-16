function cc_build()
% 构建定速巡航模糊控制的 Simulink 模型 CruiseFuzzy.slx
%
% 顶层只有四样东西：
%   设定车速（常数）──┬─→ 模糊控制器 ─→ 油门 ─→ 车辆 ─→ 车速 ─┬─→ 示波器
%   坡道（常数）──────┘                              ↑        └─→ 记录
%                                                    └── 反馈 ──┘
%
% 跑两次得到两组结果：
%   第一次：把 grade 方框的值改成 0      → 平路
%   第二次：把 grade 方框的值改成 p.grade → 上坡

here = fileparts(mfilename('fullpath'));
mdl  = 'CruiseFuzzy';
addpath(here);

%% ---------- 解析 Fuzzy Logic Controller 块的真实库路径 ----------
load_system('fuzblock');
bl = find_system('fuzblock', 'SearchDepth', 1, 'Type', 'Block');
kk = find(contains(bl,'Fuzzy Logic') & contains(bl,'Controller') & ~contains(bl,'Ruleviewer'), 1);
FLC = bl{kk};

%% ---------- 新建模型 ----------
if bdIsLoaded(mdl), close_system(mdl, 0); end
f = fullfile(here, [mdl '.slx']);
if exist(f, 'file'), delete(f); end
new_system(mdl);
set_param(mdl, 'Solver', 'ode4', 'FixedStep', '1e-3', 'StopTime', '30');
set_param(mdl, 'PreLoadFcn', 'cc_init;');

%% ==================================================================
%  顶层：只有两个常数、两个子系统、一个示波器、一个记录
%% ==================================================================
add_block('simulink/Sources/Constant', [mdl '/v_set'], ...
    'Position', [60 130 150 150], 'Value', 'p.v_ref');
add_block('simulink/Sources/Constant', [mdl '/grade'], ...
    'Position', [60 300 150 320], 'Value', 'grade');

add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/Controller'], ...
    'Position', [260 100 440 240]);
add_block('simulink/Ports & Subsystems/Subsystem', [mdl '/Vehicle'], ...
    'Position', [520 90 720 280]);

add_block('simulink/Sinks/Scope', [mdl '/Scope_v'], ...
    'Position', [800 110 840 190], 'NumInputPorts', '2', 'Open', 'off');
add_block('simulink/Sinks/To Workspace', [mdl '/simout'], ...
    'Position', [800 250 880 290], 'VariableName', 'simout', ...
    'SaveFormat', 'Structure With Time', 'SampleTime', '-1');

% 顶层连线放在两个子系统配置完成之后（见文件末尾）

%% ==================================================================
%  Controller 子系统：模糊控制器（按采样周期 Ts 离散工作）
%  In1 = 设定车速   In2 = 实际车速   Out1 = 油门开度
%% ==================================================================
sub = [mdl '/Controller'];
clean_subsystem(sub, 2, 1);

add_block('simulink/Math Operations/Sum', [sub '/e'], ...
    'Position', [100 90 120 130], 'Inputs', '+-');
add_block('simulink/Discrete/Zero-Order Hold', [sub '/e_h'], ...
    'Position', [150 50 190 90], 'SampleTime', 'p.Ts');
add_block('simulink/Discrete/Unit Delay', [sub '/e_d'], ...
    'Position', [150 130 190 170], 'SampleTime', 'p.Ts');
add_block('simulink/Math Operations/Sum', [sub '/de'], ...
    'Position', [220 90 240 130], 'Inputs', '+-');
add_block('simulink/Math Operations/Gain', [sub '/g1Ts'], ...
    'Position', [270 90 310 114], 'Gain', ['1/' 'p.Ts']);

add_block('simulink/Math Operations/Gain', [sub '/gKe'], ...
    'Position', [340 30 380 54], 'Gain', 'p.Ke');
add_block('simulink/Discontinuities/Saturation', [sub '/sen'], ...
    'Position', [410 30 450 54], 'UpperLimit', '1', 'LowerLimit', '-1');
add_block('simulink/Math Operations/Gain', [sub '/gKec'], ...
    'Position', [340 130 380 154], 'Gain', 'p.Kec');
add_block('simulink/Discontinuities/Saturation', [sub '/secn'], ...
    'Position', [410 130 450 154], 'UpperLimit', '1', 'LowerLimit', '-1');

add_block('simulink/Signal Routing/Mux', [sub '/Mux'], ...
    'Position', [480 40 485 164], 'Inputs', '2');
add_block(FLC, [sub '/FLC'], 'Position', [510 70 610 120]);
set_param([sub '/FLC'], 'FIS', 'cruise_fuzzy', 'SimulateUsing', 'Interpreted execution');
add_block('simulink/Math Operations/Gain', [sub '/gKu'], ...
    'Position', [640 80 680 104], 'Gain', 'p.Ku');

add_block('simulink/Math Operations/Sum', [sub '/al_sum'], ...
    'Position', [710 80 730 140], 'Inputs', '++');
add_block('simulink/Discontinuities/Saturation', [sub '/sal'], ...
    'Position', [760 90 800 130], 'UpperLimit', '1', 'LowerLimit', '0');
add_block('simulink/Discrete/Unit Delay', [sub '/alpha_d'], ...
    'Position', [700 200 740 240], 'SampleTime', 'p.Ts', 'InitialCondition', 'p.alpha0');

add_line(sub, 'In1/1',   'e/1',      'autorouting','on');
add_line(sub, 'In2/1',   'e/2',      'autorouting','on');
add_line(sub, 'e/1',     'e_h/1',    'autorouting','on');
add_line(sub, 'e/1',     'e_d/1',    'autorouting','on');
add_line(sub, 'e_h/1',   'de/1',     'autorouting','on');
add_line(sub, 'e_d/1',   'de/2',     'autorouting','on');
add_line(sub, 'de/1',    'g1Ts/1',   'autorouting','on');
add_line(sub, 'g1Ts/1',  'gKec/1',   'autorouting','on');
add_line(sub, 'gKec/1',  'secn/1',   'autorouting','on');
add_line(sub, 'secn/1',  'Mux/2',    'autorouting','on');
add_line(sub, 'e/1',     'gKe/1',    'autorouting','on');
add_line(sub, 'gKe/1',   'sen/1',    'autorouting','on');
add_line(sub, 'sen/1',   'Mux/1',    'autorouting','on');
add_line(sub, 'Mux/1',   'FLC/1',    'autorouting','on');
add_line(sub, 'FLC/1',   'gKu/1',    'autorouting','on');
add_line(sub, 'gKu/1',   'al_sum/2', 'autorouting','on');
add_line(sub, 'alpha_d/1','al_sum/1','autorouting','on');
add_line(sub, 'al_sum/1','sal/1',    'autorouting','on');
add_line(sub, 'sal/1',   'Out1/1',   'autorouting','on');
add_line(sub, 'sal/1',   'alpha_d/1','autorouting','on');

%% ==================================================================
%  Vehicle 子系统：整车纵向动力学
%  In1 = 油门开度   In2 = 坡度   Out1 = 车速
%% ==================================================================
sub = [mdl '/Vehicle'];
clean_subsystem(sub, 2, 1);

add_block('simulink/Math Operations/Gain', [sub '/gFmax'], ...
    'Position', [100 50 140 74], 'Gain', 'p.Fmax');
add_block('simulink/Math Operations/Sum', [sub '/lag'], ...
    'Position', [180 45 200 105], 'Inputs', '+-');
add_block('simulink/Math Operations/Gain', [sub '/gTau'], ...
    'Position', [230 60 270 84], 'Gain', ['1/' 'p.tau']);
add_block('simulink/Continuous/Integrator', [sub '/Fd'], ...
    'Position', [310 55 340 95], 'InitialCondition', 'p.Fd0');

add_block('simulink/Sources/Constant', [sub '/Ff'], ...
    'Position', [30 240 70 260], 'Value', 'p.m*p.g*p.f');
add_block('simulink/Math Operations/Gain', [sub '/gAero'], ...
    'Position', [30 288 70 312], 'Gain', '0.5*p.rho*p.Cd*p.A');
add_block('simulink/Math Operations/Product', [sub '/v2'], ...
    'Position', [180 212 210 243], 'Inputs', '**');
add_block('simulink/Sources/Constant', [sub '/Fg'], ...
    'Position', [30 340 70 360], 'Value', 'p.m*p.g');
add_block('simulink/Math Operations/Product', [sub '/slopeF'], ...
    'Position', [120 332 150 363], 'Inputs', '**');

add_block('simulink/Math Operations/Sum', [sub '/Fr'], ...
    'Position', [290 230 310 310], 'Inputs', '+++');
add_block('simulink/Math Operations/Sum', [sub '/net'], ...
    'Position', [360 90 380 150], 'Inputs', '+-');
add_block('simulink/Math Operations/Gain', [sub '/gInvM'], ...
    'Position', [420 100 460 124], 'Gain', ['1/' 'p.m']);
add_block('simulink/Continuous/Integrator', [sub '/v'], ...
    'Position', [500 90 530 130], 'InitialCondition', 'p.v_ref');

add_line(sub, 'In1/1',  'gFmax/1',  'autorouting','on');
add_line(sub, 'gFmax/1','lag/1',    'autorouting','on');
add_line(sub, 'Fd/1',   'lag/2',    'autorouting','on');
add_line(sub, 'lag/1',  'gTau/1',   'autorouting','on');
add_line(sub, 'gTau/1', 'Fd/1',     'autorouting','on');
add_line(sub, 'Fd/1',   'net/1',    'autorouting','on');
add_line(sub, 'v/1',    'v2/1',     'autorouting','on');
add_line(sub, 'v/1',    'v2/2',     'autorouting','on');
add_line(sub, 'v2/1',   'gAero/1',  'autorouting','on');
add_line(sub, 'gAero/1','Fr/2',     'autorouting','on');
add_line(sub, 'Ff/1',   'Fr/1',     'autorouting','on');
add_line(sub, 'Fg/1',   'slopeF/1', 'autorouting','on');
add_line(sub, 'In2/1',  'slopeF/2', 'autorouting','on');
add_line(sub, 'slopeF/1','Fr/3',    'autorouting','on');
add_line(sub, 'Fr/1',   'net/2',    'autorouting','on');
add_line(sub, 'net/1',  'gInvM/1',  'autorouting','on');
add_line(sub, 'gInvM/1','v/1',      'autorouting','on');
add_line(sub, 'v/1',    'Out1/1',   'autorouting','on');

%% ==================================================================
%  顶层连线（要等两个子系统的端口都建好）
%% ==================================================================
add_line(mdl, 'v_set/1',      'Controller/1', 'autorouting','on');
add_line(mdl, 'v_set/1',      'Scope_v/1',    'autorouting','on');
add_line(mdl, 'grade/1',      'Vehicle/2',    'autorouting','on');
add_line(mdl, 'Controller/1', 'Vehicle/1',    'autorouting','on');
add_line(mdl, 'Vehicle/1',    'Controller/2', 'autorouting','on');
add_line(mdl, 'Vehicle/1',    'Scope_v/2',    'autorouting','on');
add_line(mdl, 'Vehicle/1',    'simout/1',     'autorouting','on');

%% ---------- 保存 ----------
save_system(mdl, f);
fprintf('模型已保存：%s\n', f);
end

% ======================================================================
function clean_subsystem(sub, nIn, nOut)
% 清空子系统内的默认内容，并加入指定数量的输入/输出端口
lines = find_system(sub, 'SearchDepth', 1, 'FindAll','on', 'Type','line');
for i = 1:numel(lines)
    try, delete_line(lines(i)); catch, end
end
blks = find_system(sub, 'SearchDepth', 1, 'Type','Block');
for i = 1:numel(blks)
    if ~strcmp(blks{i}, sub)
        try, delete_block(blks{i}); catch, end
    end
end
for i = 1:nIn
    add_block('simulink/Ports & Subsystems/In1', sprintf('%s/In%d', sub, i), ...
        'Position', [30 60+90*(i-1) 60 74+90*(i-1)]);
end
for i = 1:nOut
    add_block('simulink/Ports & Subsystems/Out1', sprintf('%s/Out%d', sub, i), ...
        'Position', [860 40+70*(i-1) 890 54+70*(i-1)]);
end
end
