%% Plot all MMC 2D fluence maps vs IMMC
clear all;
close all;

%% Load benchmark results
fprintf('Loading benchmark results...\n');
load('mmc_benchmark_results.mat', 'results', 'immc');

%% Parameters
voxel_size = 0.05;  % Voxel size in mm
y_slice = 15 / voxel_size;

% Plot limits (mm)
x_plot_min = 2;
x_plot_max = 12;
z_plot_min = 24;
z_plot_max = 34;

% Circle parameters
theta = linspace(0, 2*pi, 400);
r_mm = 10;
cx_mm = 15;
cy_mm = 20;
x_circle = cx_mm + r_mm*cos(theta);
y_circle = cy_mm + r_mm*sin(theta);

% Color limits
cmin = 4;
cmax = 7;

%% Extract IMMC slice
immc_slice = squeeze(immc.flux_data(y_slice, :, :));
log_slice_immc = log10(immc_slice');
x_coords = (1:size(immc_slice, 1)) * voxel_size;
z_coords = (1:size(immc_slice, 2)) * voxel_size;

%% Determine subplot layout (IMMC + all MMC)
num_tests = length(results.sphere_n);
num_plots = num_tests + 1;  % +1 for IMMC
ncols = ceil(sqrt(num_plots));
nrows = ceil(num_plots / ncols);

%% Create figure
figure('Position', [50, 50, 1800, 1000]);

% Panel 1: IMMC reference
subplot(nrows, ncols, 1);
imagesc(x_coords, z_coords, log_slice_immc);
set(gca, 'YDir', 'normal');
hold on;
plot(x_circle, y_circle, 'k--', 'LineWidth', 1.5);
hold off;
axis equal;
xlim([x_plot_min, x_plot_max]);
ylim([z_plot_min, z_plot_max]);
colormap(gca, parula);
caxis([cmin, cmax]);
title('IMMC (reference)', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('X (mm)', 'FontSize', 9);
ylabel('Z (mm)', 'FontSize', 9);
set(gca, 'FontSize', 8);

% Panels 2+: MMC results
for i = 1:num_tests
    subplot(nrows, ncols, i + 1);
    
    mmc_slice = squeeze(results.flux_data{i}(y_slice, :, :));
    log_slice_mmc = log10(mmc_slice');
    
    imagesc(x_coords, z_coords, log_slice_mmc);
    set(gca, 'YDir', 'normal');
    hold on;
    plot(x_circle, y_circle, 'k--', 'LineWidth', 1.5);
    hold off;
    
    axis equal;
    xlim([x_plot_min, x_plot_max]);
    ylim([z_plot_min, z_plot_max]);
    colormap(gca, parula);
    caxis([cmin, cmax]);
    
    title(sprintf('MMC %.1fk tri (n=%d)', results.num_triangles_sphere(i)/1000, results.sphere_n(i)), 'FontSize', 10);
    
    if i + 1 > (nrows-1)*ncols
        xlabel('X (mm)', 'FontSize', 9);
    end
    if mod(i, ncols) == 0
        ylabel('Z (mm)', 'FontSize', 9);
    end
    
    set(gca, 'FontSize', 8);
end

% Add colorbar
cb = colorbar('Position', [0.93, 0.15, 0.02, 0.7]);
cb.Label.String = 'Log_{10}(Fluence)';
cb.Label.FontSize = 12;
cb.FontSize = 10;

sgtitle('2D Fluence Maps: IMMC vs MMC at varying mesh densities', 'FontSize', 14);

savefig('all_fluence_maps.fig');
saveas(gcf, 'all_fluence_maps.png');
print(gcf, 'all_fluence_maps.pdf', '-dpdf', '-painters');
fprintf('Saved: all_fluence_maps.fig/.png/.pdf\n');
