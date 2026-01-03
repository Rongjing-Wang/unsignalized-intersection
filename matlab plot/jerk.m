clc; clear; close all;

% ====== 1) 读取数据 ======
file_path = "C:\Users\Lenovo\Desktop\DRL statistic\jerk.xlsx";
data = xlsread(file_path);

% 获取原始数据 (注意对应关系)
column1 = data(2:end, 1); % CoMADDPG
column2 = data(2:end, 2); % Our Method
column3 = data(2:end, 3); % CoMADDPG-GE

% ====== 2) 数据平滑 ======
% 保持你偏好的 lowess 强平滑逻辑
smooth_column1 = smooth(column1, 400, 'lowess');
smooth_column2 = smooth(column2, 400, 'lowess');
smooth_column3 = smooth(column3, 400, 'lowess');

% ====== 3) 绘制图像 ======
figure('Color', 'w', 'Position', [200, 200, 850, 600]); 
hold on;

% --- 严格同步 Actor/Critic 的配色方案 ---
c1 = [0.85, 0.33, 0.10]; % 橙红 (Our Method)
c2 = [0.47, 0.67, 0.19]; % 橄榄绿 (CoMADDPG-GE)
c3 = [0.00, 0.45, 0.74]; % 深蓝 (CoMADDPG)

% 绘制三条曲线 (确保颜色与算法名称匹配)
plot(smooth_column2, 'Color', c1, 'LineWidth', 2.5, 'DisplayName', 'Our method');
plot(smooth_column3, 'Color', c2, 'LineWidth', 2.5, 'DisplayName', 'CoMADDPG-GE');
plot(smooth_column1, 'Color', c3, 'LineWidth', 2.5, 'DisplayName', 'CoMADDPG');

% ====== 4) 论文风格美化 ======
% 坐标轴标签设置
xlabel('Step', 'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
ylabel('Jerk', 'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Times New Roman');

% 图例设置：白色背景、黑色边框
lgd = legend('show', 'Location', 'northeast');
set(lgd, 'Box', 'on', 'EdgeColor', [0 0 0], 'LineWidth', 0.8, 'Color', [1 1 1], ...
    'FontName', 'Times New Roman', 'FontSize', 12);

% 坐标轴细节设置
ax = gca;
ax.FontName = 'Times New Roman';
ax.FontSize = 14;
ax.LineWidth = 1.3;

% 设置网格线 (根据之前的样式保持一致)
ax.GridColor = [0.5, 0.5, 0.5];
ax.GridAlpha = 0.5;
ax.GridLineStyle = ':';
% grid on; % 如果需要网格背景可开启

% 边框与刻度
box on; % 开启全封闭方框
set(gca, 'TickDir', 'in', 'TickLength', [.015 .015]); % 刻度向内，比完全去掉刻度更专业

hold off;