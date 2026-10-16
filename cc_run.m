function cc_run()
% 主结果脚本：跑两次仿真、校核、出图
%   第一次：平路（坡度 = 0）
%   第二次：上坡（坡度 = p.grade）
here = fileparts(mfilename('fullpath'));
resDir = fullfile(here, 'results');
if ~exist(resDir, 'dir'), mkdir(resDir); end
addpath(here);
set(0, 'DefaultAxesFontName', 'Microsoft YaHei');
set(0, 'DefaultTextFontName', 'Microsoft YaHei');

cc_init;
p   = cc_params();
mdl = 'CruiseFuzzy';
if ~bdIsLoaded(mdl), load_system(fullfile(here, [mdl '.slx'])); end

colF = [0.00 0.45 0.74];      % 平路
colS = [0.85 0.33 0.10];      % 上坡

%% ---------- 1. 两次运行（纯 MATLAB 参考解）----------
rFlat  = cc_sim(0);
rSlope = cc_sim(p.grade);

vref_kmh = p.v_ref*3.6;
fprintf('=== 第一次：平路（坡度 = 0）===\n');
fprintf('  车速 %.1f km/h，油门 %.2f%%，驱动力 %.0f N\n', ...
    rFlat.v(end)*3.6, rFlat.alpha(end)*100, rFlat.Fd(end));

fprintf('\n=== 第二次：上坡（坡度 = %.0f%%）===\n', p.grade*100);
dip = (p.v_ref - min(rSlope.v))*3.6;
fprintf('  车速最低 %.2f km/h（掉 %.2f km/h），最后回到 %.2f km/h\n', ...
    min(rSlope.v)*3.6, dip, rSlope.v(end)*3.6);
fprintf('  油门 %.2f%% -> %.2f%%，驱动力 %.0f N -> %.0f N\n', ...
    p.alpha0*100, rSlope.alpha(end)*100, p.Fd0, rSlope.Fd(end));
fprintf('  需要多出的力 = %.0f N\n\n', p.m*p.g*p.grade);

%% ---------- 2. Simulink 模型校核 ----------
fprintf('=== Simulink 模型校核 ===\n');
for gr = [0 p.grade]
    assignin('base', 'grade', gr);
    so = sim(mdl, 'StopTime', num2str(p.Tend));
    tS = so.simout.time;  vS = so.simout.signals.values;
    r  = cc_sim(gr);
    vR = interp1(r.t, r.v, tS, 'linear', 'extrap');
    fprintf('  坡度 %.0f%%：max|v_simulink - v_matlab| = %.4f m/s\n', ...
        gr*100, max(abs(vS - vR)));
end
fprintf('\n');

%% ---------- 3. 幻灯片配图 ----------
% 图A：模糊推理系统总览（隶属函数 + 规则表 + 规则曲面）
fis = cc_fis();
lam = linspace(-1,1,400);
mfs = [-1 -1 -0.5; -1 -0.5 0; -0.5 0 0.5; 0 0.5 1; 0.5 1 1];
lbl = {'NB','NS','ZO','PS','PB'};
cc5 = [0.00 0.45 0.74; 0.85 0.33 0.10; 0.93 0.69 0.13; 0.49 0.18 0.56; 0.47 0.67 0.19];
tbl = [1 1 1 2 3; 1 1 2 3 4; 1 2 3 4 5; 2 3 4 5 5; 3 4 5 5 5];

f = figure('Position',[60 60 1620 460], 'Color','w');
subplot(1,3,1); hold on; grid on;
for i = 1:5, plot(lam, trimf(lam, mfs(i,:)), 'LineWidth',2.6, 'Color', cc5(i,:)); end
xlabel('归一化输入或输出'); ylabel('隶属度');
title('(a) 5 个语言变量：NB NS ZO PS PB', 'FontWeight','bold');
legend(lbl, 'Location','northoutside', 'Orientation','horizontal', 'FontSize',12);
xlim([-1.02 1.02]); ylim([0 1.05]);
subplot(1,3,2);
imagesc(tbl); axis square;
cmap = [0.80 0.18 0.18; 0.93 0.58 0.20; 0.95 0.95 0.95; 0.42 0.72 0.42; 0.10 0.42 0.72];
colormap(cmap);
for i = 1:5
    for j = 1:5
        col = 'k'; if tbl(i,j)==1 || tbl(i,j)==5, col = 'w'; end
        text(j, i, lbl{tbl(i,j)}, 'HorizontalAlignment','center', ...
             'FontSize',15, 'FontWeight','bold', 'Color',col);
    end
end
set(gca,'XTick',1:5,'XTickLabel',lbl,'YTick',1:5,'YTickLabel',lbl,'FontSize',14);
xlabel('误差变化率  ec'); ylabel('车速误差  e');
title('(b) 5 x 5 = 25 条规则', 'FontWeight','bold');
subplot(1,3,3);
[X1, X2] = meshgrid(linspace(-1,1,41));
V = evalfis(fis, [X1(:), X2(:)]);
s = surf(X1, X2, reshape(V,size(X1)), 'EdgeColor','none');
s.FaceColor = [0.25 0.55 0.80];  s.FaceLighting = 'gouraud';
s.AmbientStrength = 0.55;  s.DiffuseStrength = 0.65;
camlight('headlight'); view(-35,30); grid on;
xlabel('e'); ylabel('ec'); zlabel('油门增量 \Delta\alpha');
title('(c) 规则曲面', 'FontWeight','bold');
exportgraphics(f, fullfile(resDir,'slide_figA.png'), 'Resolution', 100); close(f);

% 图B：两次运行的结果
f = figure('Position',[80 80 1500 520], 'Color','w');
subplot(1,2,1); hold on; grid on;
h1 = plot([0 p.Tend], [vref_kmh vref_kmh], '--', 'Color',[0.45 0.45 0.45], 'LineWidth',1.6);
h2 = plot(rFlat.t,  rFlat.v*3.6,  'Color',colF, 'LineWidth',2.8);
h3 = plot(rSlope.t, rSlope.v*3.6, 'Color',colS, 'LineWidth',2.8);
set(h1, 'DisplayName', '设定车速');
set(h2, 'DisplayName', '第一次：平路');
set(h3, 'DisplayName', ['第二次：' sprintf('%.0f%%',p.grade*100) ' 上坡']);
xlabel('时间 t / s'); ylabel('车速 v / (km/h)');
title('(a) 车速：两次运行都稳在设定值', 'FontWeight','bold');
legend('show', 'Location','southeast','FontSize',13);
xlim([0 p.Tend]);
ylim([floor(min(rSlope.v)*3.6 - 0.8), vref_kmh + 1.2]);
subplot(1,2,2); hold on; grid on;
g1 = plot(rFlat.t,  rFlat.alpha*100,  'Color',colF, 'LineWidth',2.8);
g2 = plot(rSlope.t, rSlope.alpha*100, 'Color',colS, 'LineWidth',2.8);
set(g1, 'DisplayName', '第一次：平路');
set(g2, 'DisplayName', ['第二次：' sprintf('%.0f%%',p.grade*100) ' 上坡']);
xlabel('时间 t / s'); ylabel('油门开度 \alpha / %');
title('(b) 油门开度：上坡时自动加大', 'FontWeight','bold');
legend('show', 'Location','east','FontSize',13);
xlim([0 p.Tend]); ylim([0 max(rSlope.alpha)*100+8]);
exportgraphics(f, fullfile(resDir,'slide_figB.png'), 'Resolution', 100); close(f);

%% ---------- 4. Simulink 模型框图 ----------
targets = {mdl, [mdl '/Controller'], [mdl '/Vehicle']};
suffix  = {'top','controller','vehicle'};
for i = 1:numel(targets)
    try
        open_system(targets{i});
        set_param(targets{i}, 'ZoomFactor', 'Fit to view');
        print(['-s' targets{i}], '-dpng', '-r160', fullfile(resDir, ['model_' suffix{i} '.png']));
    catch ME
        fprintf('框图导出失败 %s : %s\n', targets{i}, ME.message);
    end
end

save(fullfile(resDir,'results_summary.mat'), 'rFlat','rSlope','p','-v7.3');
fprintf('图已写入 %s\n', resDir);
fprintf('DONE\n');
end
