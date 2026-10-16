function res = cc_sim(grade)
% 定速巡航 —— 跑一次仿真（纯 MATLAB 参考解）
%
%   grade : 坡道坡度，0 = 平路，0.05 = 5% 上坡
%
% 两次运行就得到两组结果：
%   cc_sim(0)            平路
%   cc_sim(p.grade)      上坡

if nargin < 1, grade = 0; end
p  = cc_params();
dt = p.dt;
t  = (0:dt:p.Tend)';
N  = numel(t);

% 模糊推理系统缓存为网格插值表（等价于 evalfis，但快很多）
persistent G
if isempty(G)
    f = cc_fis();
    [X1, X2] = meshgrid(linspace(-1,1,41), linspace(-1,1,41));
    V = evalfis(f, [X1(:), X2(:)]);
    G = {linspace(-1,1,41), linspace(-1,1,41), reshape(V,41,41)};
end

% 初始状态：平路巡航平衡点（油门先给平路的值，上坡时控制器要自己补）
v      = p.v_ref;
Fd     = p.Fd0;
al     = p.alpha0;
e_prev = 0;
iNext  = 1;

R = zeros(N, 5);
for k = 1:N
    % ---- 行驶阻力 = 滚动 + 空气 + 坡道 ----
    vSafe = max(v, 0);
    Fr = p.m*p.g*p.f + 0.5*p.rho*p.Cd*p.A*vSafe^2 + p.m*p.g*grade;

    % ---- 模糊控制器：每 Ts 更新一次 ----
    if k >= iNext
        iNext = iNext + round(p.Ts/dt);
        e_k = p.v_ref - v;                      % 车速误差
        ec  = (e_k - e_prev) / p.Ts;            % 误差变化率
        e_prev = e_k;
        en  = min(max(p.Ke  * e_k, -1), 1);     % 归一化到 [-1,1]
        ecn = min(max(p.Kec * ec,  -1), 1);
        dA  = interp2(G{1}, G{2}, G{3}, en, ecn, 'linear', 0);
        al  = min(max(al + p.Ku*dA, 0), 1);     % 油门开度累加
    end

    % ---- 记录 ----
    R(k,:) = [v, al, Fd, Fr, p.v_ref - v];

    if k == N, break; end

    % ---- 前向欧拉积分 ----
    Fd = Fd + ((p.Fmax*al - Fd) / p.tau) * dt;
    v  = v  + ((Fd - Fr) / p.m) * dt;
end

res.t     = t;
res.v     = R(:,1);
res.alpha = R(:,2);
res.Fd    = R(:,3);
res.Fr    = R(:,4);
res.e     = R(:,5);
res.v_ref = p.v_ref;
res.grade = grade;
end
