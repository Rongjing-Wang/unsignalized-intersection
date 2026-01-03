clc; clear; close all;

% ====== 1) 读取数据 ======
file_path = 'C:\Users\Lenovo\Desktop\DRL statistic\reward.xlsx';  % 改成你的路径
data = xlsread(file_path);  % 也可用 readmatrix(file_path)

y1 = data(2:end,1);   % CoMADDPG-GAT
y2 = data(2:end,2);   % CoMADDPG
y3 = data(2:end,3);   % MADDPG

% ====== 2) 限制显示步数（和你原图一致）======
max_x = 2.5e4;
N = min([length(y1), length(y2), length(y3), max_x]);
y1 = y1(1:N);  y2 = y2(1:N);  y3 = y3(1:N);

%====== 3) 抑制抖动：去尖刺 -> 低通滤波 -> 分箱均值 ======
% 3.1 轻度去尖刺（中值滤波，窗口需为奇数；可调 41/61/81）
y1 = medfilt1(y1, 41);
y2 = medfilt1(y2, 41);
y3 = medfilt1(y3, 41);

% 3.2 零相位 Butterworth 低通（避免相位滞后）
%     截止频率 0.02（相对 Nyquist），越小越平滑（可在 0.01~0.05 之间调）
[b,a] = butter(4, 0.02);     % 4 阶低通
s1 = filtfilt(b, a, y1);
s2 = filtfilt(b, a, y2);
s3 = filtfilt(b, a, y3);

% 3.3 分箱重采样（每 bin_size 个点做均值与标准差；越大越平滑）
bin_size = 200;              % 建议 200~800 之间调
M   = floor(N / bin_size);
cut = M * bin_size;

reshape_mean = @(y) mean(reshape(y(1:cut), bin_size, M), 1);
reshape_std  = @(y)  std(reshape(y(1:cut), bin_size, M), 0, 1);

m1 = reshape_mean(s1);   e1 = reshape_std(s1);
m2 = reshape_mean(s2);   e2 = reshape_std(s2);
m3 = reshape_mean(s3);   e3 = reshape_std(s3);

x  = (1:M) * bin_size;   % 分箱中心对应的 Step（横轴）

% ====== 4) 绘图：均值线 + 1σ 阴影带 ======
figure('Color','w'); hold on;

c1 = [255 127 14]/255;   % 蓝
c2 = [0 128 0]/255;      % 绿
c3 = [31 119 180]/255;   % 橙


% 阴影带（不进图例）
fill([x, fliplr(x)], [m1-e1, fliplr(m1+e1)], c1, 'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility','off');
fill([x, fliplr(x)], [m2-e2, fliplr(m2+e2)], c2, 'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility','off');
fill([x, fliplr(x)], [m3-e3, fliplr(m3+e3)], c3, 'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility','off');

% 均值曲线
plot(x, m1, 'Color', c1, 'LineWidth', 2, 'DisplayName', 'our method');
plot(x, m2, 'Color', c2, 'LineWidth', 2, 'DisplayName', 'CoMADDPG-GE');
plot(x, m3, 'Color', c3, 'LineWidth', 2, 'DisplayName', 'CoMADDPG');

% 论文风格外观
xlabel('Step'); ylabel('Value');
legend('show', 'Location','northeast');
box on;
%grid on;
ax = gca;
ax.GridColor = [0.5 0.5 0.5];
ax.GridAlpha = 0.5;
ax.GridLineStyle = ':';
ax.FontName = 'Times New Roman';
ax.FontSize = 12;
ax.TickLength = [0 0];
xlim([1, N]);  % 横轴限制到原始步数

%（可选）导出图片
% exportgraphics(gcf, 'reward_smooth.png', 'Resolution', 300);
