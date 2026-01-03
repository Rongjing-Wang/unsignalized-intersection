clc
clear
% 读取Excel文件
file_path = "C:\Users\Lenovo\Desktop\DRL statistic\collisions - update.xlsx";
data = xlsread(file_path);

% 获取从第二行到最后一行的数据
column1 = data(2:end, 1);
column2 = data(2:end, 2);
column3 = data(2:end, 3);

% 使用更强的平滑来减少噪声
smooth_column1 = smooth(column1, 400, 'lowess');
smooth_column2 = smooth(column2, 400, 'lowess');
smooth_column3 = smooth(column3, 400, 'lowess');

% 绘制图像
figure;
hold on;
% 绘制三条曲线
plot(smooth_column1, 'Color', [255/255, 127/255, 14/255], 'LineWidth', 2, 'DisplayName', 'CoMADDPG-GAT-PVE');
plot(smooth_column3, 'Color', [0/255, 128/255, 0/255], 'LineWidth', 2, 'DisplayName', 'CoMADDPG');
plot(smooth_column2, 'Color', [31/255, 119/255, 180/255], 'LineWidth', 2, 'DisplayName', 'MADDPG');

% 添加坐标轴标签
xlabel('Step', 'FontSize', 12, 'Interpreter', 'latex');
ylabel('Total number of vehicle collisions', 'FontSize', 12, 'Interpreter', 'latex');

% 显示图例
legend('show', 'Location', 'northwest', 'FontSize', 10, 'Interpreter', 'latex');

% 设置图框样式为论文常用风格
box on; % 添加外部黑色框线
%grid on; % 确保网格打开
ax = gca;
ax.GridColor = [0.5, 0.5, 0.5]; % 设置网格线颜色为更明显的灰色，可根据需要调节
ax.GridAlpha = 0.5; % 设置网格线的透明度，0为完全透明，1为完全不透明
ax.GridLineStyle = ':'; % 虚线风格


% 设置背景颜色为白色
set(gcf,'Color','w');

% 调整坐标轴字体与尺寸（更贴近论文风格）
set(gca,'FontName','Times New Roman','FontSize',12);

set(gca, 'TickLength', [0 0]);  % 去掉所有刻度标记的小竖线


hold off;
