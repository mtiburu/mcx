%% MMC Mesh Density Benchmark with IMMC Comparison
% This script tests different sphere mesh densities and records timing data
% by capturing and parsing console output from mmclab.
% Also compares against faster IMMC approach.
% Uses mmcaddsrc to properly embed source in MMC mesh only.
% Uses sphere() + convhulln() for sphere mesh generation.

clear all;
close all;

%% ========== FIRST: Run IMMC baseline ==========
fprintf('Running IMMC Baseline Test\n');
fprintf('==========================\n\n');

% Generate mesh using IMMC approach
immc_mesh_start = tic;
[node_box, face_box, elem_box] = meshabox([0, 0, 0], [30, 30, 40], 10000);
newnode = [15, 15, 20];
[node_immc, elem_immc] = meshrefine(node_box, elem_box, newnode);
immc_mesh_time_ms = toc(immc_mesh_start) * 1000;

immc.num_elements = size(elem_immc, 1);
immc.num_nodes = size(node_immc, 1);
immc.mesh_preprocess_time_ms = immc_mesh_time_ms;

fprintf('  IMMC Mesh: %d elements, %d nodes\n', immc.num_elements, immc.num_nodes);
fprintf('  Mesh preprocessing time: %.3f ms\n', immc_mesh_time_ms);

% Configure IMMC simulation
cfg_immc.srctype = 'disk';
cfg_immc.srcdir = [0, 0, -1];
cfg_immc.srcparam1 = 1.125;
cfg_immc.srcpos = [15, 10, 40];
cfg_immc.elem = elem_immc;
cfg_immc.elem(:,5) = 1;
cfg_immc.node = node_immc;
cfg_immc.nphoton = 2e8;
cfg_immc.tstart = 0;
cfg_immc.tend = 1e-10;
cfg_immc.tstep = 1e-10;
cfg_immc.outputtype = 'flux';
cfg_immc.isreflect = 1;
cfg_immc.prop = [0 0 1 1;
                 0.0001 0 1 1;
                 0.0001 0 1 1.33];
cfg_immc.steps = [0.05 0.05 0.05];
cfg_immc.noderoi = zeros(size(node_immc,1),1);
% Find node closest to sphere center for noderoi
dists = sqrt(sum((node_immc - [15,15,20]).^2, 2));
[~, center_node] = min(dists);
cfg_immc.noderoi(center_node) = 10;
cfg_immc.gpuid = 1;
cfg_immc.compute = 'cuda';

% Capture IMMC console output
diary_file = 'immc_output_temp.txt';
if exist(diary_file, 'file')
    delete(diary_file);
end

diary(diary_file);
immc_output = mmclab(cfg_immc);
diary off;

% Parse IMMC console output
console_output = fileread(diary_file);

kernel_match = regexp(console_output, 'kernel complete:\s*(\d+)\s*ms', 'tokens');
if ~isempty(kernel_match)
    immc.kernel_time_ms = str2double(kernel_match{1}{1});
end

transfer_match = regexp(console_output, 'transfer complete:\s*(\d+)\s*ms', 'tokens');
if ~isempty(transfer_match)
    immc.transfer_time_ms = str2double(transfer_match{1}{1});
end

speed_match = regexp(console_output, 'simulation speed:\s*([\d.]+)\s*photon/ms', 'tokens');
if ~isempty(speed_match)
    immc.photon_speed = str2double(speed_match{1}{1});
end

raytet_match = regexp(console_output, '\(ray-tet\s*(\d+)\)', 'tokens');
if ~isempty(raytet_match)
    immc.ray_tet_intersections = str2double(raytet_match{1}{1});
end

delete(diary_file);

% Save IMMC flux data
immc.flux_data = immc_output.data;

fprintf('  GPU Kernel time: %.0f ms\n', immc.kernel_time_ms);
fprintf('  Transfer complete time: %.0f ms\n', immc.transfer_time_ms);
fprintf('  Photon speed: %.2f photon/ms\n', immc.photon_speed);
fprintf('  Ray-tet intersections: %.2e\n\n', immc.ray_tet_intersections);

clear node_box face_box elem_box node_immc elem_immc immc_output cfg_immc;

%% ========== SECOND: Run MMC density sweep ==========
fprintf('Starting MMC Mesh Density Benchmark\n');
fprintf('===================================\n\n');

% Define sphere resolution parameters (n for sphere(n))
% Triangle count ≈ 2*n^2, so n≈22 gives ~1000 triangles, n≈387 gives ~300k
sphere_n_values = round(logspace(log10(22), log10(387), 6));

% Pause duration between runs (seconds) to avoid thermal throttling
PAUSE_BETWEEN_RUNS = 1;

% Initialize results storage
num_tests = length(sphere_n_values);
results = struct();
results.sphere_n = sphere_n_values;
results.num_triangles_sphere = zeros(1, num_tests);
results.num_triangles_total = zeros(1, num_tests);
results.num_nodes = zeros(1, num_tests);
results.num_elements = zeros(1, num_tests);
results.mesh_preprocess_time_ms = zeros(1, num_tests);
results.kernel_time_ms = zeros(1, num_tests);
results.transfer_time_ms = zeros(1, num_tests);
results.total_mmclab_time_ms = zeros(1, num_tests);
results.photon_speed = zeros(1, num_tests);
results.ray_tet_intersections = zeros(1, num_tests);
results.output_data_size_bytes = zeros(1, num_tests);

% Cell array to store flux data for each test
results.flux_data = cell(1, num_tests);

% Fixed simulation parameters
box_min = [0, 0, 0];
box_max = [30, 30, 40];
sphere_center = [15, 15, 20];
sphere_radius = 10;

% Source definition (used for mmcaddsrc)
srcdef.srctype = 'disk';
srcdef.srcpos = [15, 10, 40];
srcdef.srcdir = [0, 0, -1];
srcdef.srcparam1 = 1.125;

% Simulation parameters
sim_cfg.nphoton = 2e8;
sim_cfg.tstart = 0;
sim_cfg.tend = 1e-10;
sim_cfg.tstep = 1e-10;
sim_cfg.outputtype = 'flux';
sim_cfg.isreflect = 1;
sim_cfg.prop = [0 0 1 1;
                0.0001 0 1 1;                
                0.0001 0 1 1.33;
                ];
sim_cfg.steps = [0.05 0.05 0.05];
sim_cfg.gpuid = 1;
sim_cfg.compute = 'cuda';

fprintf('Pause between runs: %d seconds\n\n', PAUSE_BETWEEN_RUNS);

% Wait before starting MMC tests
fprintf('Waiting %d seconds before MMC tests...\n\n', PAUSE_BETWEEN_RUNS);
pause(PAUSE_BETWEEN_RUNS);

for i = 1:num_tests
    n = sphere_n_values(i);
    fprintf('Test %d/%d: Sphere resolution n = %d\n', i, num_tests, n);
    
    %% Mesh Generation Phase
    mesh_start_time = tic;
    
    [node_box, face_box, ~] = meshabox(box_min, box_max, 100);
    
    % Generate sphere surface using sphere() + convhulln()
    [xi, yi, zi] = sphere(n);
    % Scale and translate to sphere_center and sphere_radius
    xi = xi * sphere_radius + sphere_center(1);
    yi = yi * sphere_radius + sphere_center(2);
    zi = zi * sphere_radius + sphere_center(3);
    
    % Create node list
    node_sph = [xi(:), yi(:), zi(:)];
    
    % Remove duplicate points (especially at poles)
    [node_sph, ~, ic] = unique(node_sph, 'rows', 'stable');
    
    % Regenerate triangulation from unique points
    face_sph = convhulln(node_sph);
    
    results.num_triangles_sphere(i) = size(face_sph, 1);
    
    [node, face] = mergesurf(node_sph, face_sph, node_box, face_box);
    results.num_triangles_total(i) = size(face, 1);
    
    % Use -YY flag for mesh generation
    regions = [sphere_center; 2, 2, 2];
    [node, elem, face] = surf2mesh(node, face, box_min, box_max, 1, 100, ...
                                   regions, [], 0, 'tetgen1.5', '-YY');
    
    % Check for degenerate tetrahedra (zero or negative volume)
    v1 = node(elem(:,2),:) - node(elem(:,1),:);
    v2 = node(elem(:,3),:) - node(elem(:,1),:);
    v3 = node(elem(:,4),:) - node(elem(:,1),:);
    tet_volumes = abs(dot(v1, cross(v2, v3, 2), 2)) / 6;
    
    degenerate_count = sum(tet_volumes < eps);
    if degenerate_count > 0
        fprintf('  WARNING: %d degenerate tetrahedra detected!\n', degenerate_count);
        % Remove degenerate elements
        valid_elems = tet_volumes >= eps;
        elem = elem(valid_elems, :);
        fprintf('  Removed %d degenerate elements, %d remaining\n', degenerate_count, size(elem, 1));
    end
    
    % Relabel elements based on centroid distance to sphere center
    elem(:,5) = 1;  % Initialize all elements as material 1 (box)
    
    % Calculate centroid of each element
    centroids = zeros(size(elem, 1), 3);
    for j = 1:size(elem, 1)
        node_indices = elem(j, 1:4);
        centroids(j, :) = mean(node(node_indices, :), 1);
    end
    
    % For all elements with centroid < sphere_radius from sphere_center, label as 2
    distances = sqrt(sum((centroids - sphere_center).^2, 2));
    elem(distances < sphere_radius, 5) = 2;
    
    % Store element properties before mmcaddsrc
    elemprop = elem(:,5);
    
    % Add source to mesh using mmcaddsrc
    [node, elem] = mmcaddsrc(node, elem, ...
        mmcsrcdomain(srcdef, [min(node); max(node)]));
    elemprop = elem(:,5);
    elem = elem(:,1:4);
    
    mesh_preprocess_time = toc(mesh_start_time);
    results.mesh_preprocess_time_ms(i) = mesh_preprocess_time * 1000;
    results.num_nodes(i) = size(node, 1);
    results.num_elements(i) = size(elem, 1);
    
    fprintf('  Mesh: %d sphere triangles, %d elements\n', ...
            results.num_triangles_sphere(i), results.num_elements(i));
    fprintf('  Mesh preprocessing time: %.3f ms\n', results.mesh_preprocess_time_ms(i));
    
    %% Simulation Phase - Capture console output
    cfg = sim_cfg;
    cfg.srctype = srcdef.srctype;
    cfg.srcdir = srcdef.srcdir;
    cfg.srcparam1 = srcdef.srcparam1;
    cfg.srcpos = srcdef.srcpos;
    cfg.node = node;
    cfg.elem = elem;
    cfg.elemprop = elemprop;
    
    % Capture console output using diary
    diary_file = sprintf('mmc_output_temp_%d.txt', i);
    if exist(diary_file, 'file')
        delete(diary_file);
    end
    
    sim_start_time = tic;
    diary(diary_file);
    mmc_output = mmclab(cfg);
    diary off;
    total_mmclab_time = toc(sim_start_time);
    results.total_mmclab_time_ms(i) = total_mmclab_time * 1000;
    
    % Parse the captured console output
    console_output = fileread(diary_file);
    
    kernel_match = regexp(console_output, 'kernel complete:\s*(\d+)\s*ms', 'tokens');
    if ~isempty(kernel_match)
        results.kernel_time_ms(i) = str2double(kernel_match{1}{1});
    end
    
    transfer_match = regexp(console_output, 'transfer complete:\s*(\d+)\s*ms', 'tokens');
    if ~isempty(transfer_match)
        results.transfer_time_ms(i) = str2double(transfer_match{1}{1});
    end
    
    speed_match = regexp(console_output, 'simulation speed:\s*([\d.]+)\s*photon/ms', 'tokens');
    if ~isempty(speed_match)
        results.photon_speed(i) = str2double(speed_match{1}{1});
    end
    
    raytet_match = regexp(console_output, '\(ray-tet\s*(\d+)\)', 'tokens');
    if ~isempty(raytet_match)
        results.ray_tet_intersections(i) = str2double(raytet_match{1}{1});
    end
    
    delete(diary_file);
    
    % Save flux data
    results.flux_data{i} = mmc_output.data;
    
    % Calculate output data size
    data_info = whos('mmc_output');
    results.output_data_size_bytes(i) = data_info.bytes;
    
    fprintf('  GPU Kernel time: %.0f ms\n', results.kernel_time_ms(i));
    fprintf('  Transfer complete time: %.0f ms\n', results.transfer_time_ms(i));
    fprintf('  Photon speed: %.2f photon/ms\n', results.photon_speed(i));
    fprintf('  Ray-tet intersections: %.2e\n', results.ray_tet_intersections(i));
    fprintf('  Output data size: %.2f MB\n', results.output_data_size_bytes(i)/1e6);
    
    clear mmc_output node elem elemprop face node_sph face_sph node_box face_box cfg centroids distances xi yi zi;
    
    if i < num_tests
        fprintf('  Waiting %d seconds before next run...\n\n', PAUSE_BETWEEN_RUNS);
        pause(PAUSE_BETWEEN_RUNS);
    else
        fprintf('\n');
    end
end

%% Save results (including all flux data)
fprintf('Saving results (this may take a moment due to flux data size)...\n');
save('mmc_benchmark_results.mat', 'results', 'immc', '-v7.3');

%% Display summary table
fprintf('\n\nBenchmark Summary\n');
fprintf('=================\n\n');

fprintf('IMMC Baseline:\n');
fprintf('  Elements: %d, Kernel: %.0f ms, Speed: %.2f photon/ms\n\n', ...
        immc.num_elements, immc.kernel_time_ms, immc.photon_speed);

T = table(results.sphere_n', ...
          results.num_triangles_sphere', ...
          results.num_elements', ...
          results.mesh_preprocess_time_ms', ...
          results.kernel_time_ms', ...
          results.transfer_time_ms', ...
          results.photon_speed', ...
          results.ray_tet_intersections', ...
          results.output_data_size_bytes'/1e6, ...
          'VariableNames', {'Sphere_N', 'Sphere_Tris', 'Elements', ...
                           'Mesh_ms', 'Kernel_ms', 'Transfer_ms', ...
                           'Speed_ph_ms', 'RayTet', 'Data_MB'});
disp(T);

%% Generate plots with IMMC comparison
figure('Position', [100, 100, 1400, 500]);

% Create triangle count labels for x-axis
tri_labels = arrayfun(@(x) sprintf('%d', x), results.num_triangles_sphere, 'UniformOutput', false);

% Plot 1: Mesh Preprocessing Time
subplot(1, 3, 1);
bar(1:num_tests, results.mesh_preprocess_time_ms, 'FaceColor', [0.3 0.5 0.8]);
hold on;
yline(immc.mesh_preprocess_time_ms, 'g--', 'LineWidth', 2);
hold off;
xlabel('Sphere Triangle Count');
ylabel('Time (ms)');
title('Mesh Preprocessing Time');
set(gca, 'XTick', 1:num_tests, 'XTickLabel', tri_labels);
xtickangle(45);
legend('MMC', sprintf('IMMC (%.1f ms)', immc.mesh_preprocess_time_ms), 'Location', 'northwest');
grid on;

% Plot 2: GPU Kernel Time
subplot(1, 3, 2);
bar(1:num_tests, results.kernel_time_ms, 'FaceColor', [0.8 0.3 0.3]);
hold on;
yline(immc.kernel_time_ms, 'g--', 'LineWidth', 2);
hold off;
xlabel('Sphere Triangle Count');
ylabel('Time (ms)');
title('GPU Kernel Time');
set(gca, 'XTick', 1:num_tests, 'XTickLabel', tri_labels);
xtickangle(45);
legend('MMC', sprintf('IMMC (%.0f ms)', immc.kernel_time_ms), 'Location', 'northwest');
grid on;

% Plot 3: Photon Speed
subplot(1, 3, 3);
bar(1:num_tests, results.photon_speed, 'FaceColor', [0.4 0.7 0.4]);
hold on;
yline(immc.photon_speed, 'g--', 'LineWidth', 2);
hold off;
xlabel('Sphere Triangle Count');
ylabel('Speed (photon/ms)');
title('Simulation Speed');
set(gca, 'XTick', 1:num_tests, 'XTickLabel', tri_labels);
xtickangle(45);
legend('MMC', sprintf('IMMC (%.0f ph/ms)', immc.photon_speed), 'Location', 'southwest');
grid on;

sgtitle('MMC vs IMMC Performance Comparison');

savefig('mmc_immc_comparison.fig');
saveas(gcf, 'mmc_immc_comparison.png');

%% Additional comparison figure
figure('Position', [100, 100, 800, 600]);

% Stacked time breakdown with IMMC reference
subplot(2, 1, 1);
bar_data = [results.mesh_preprocess_time_ms; 
            results.kernel_time_ms]';
bar(1:num_tests, bar_data, 'stacked');
hold on;
yline(immc.mesh_preprocess_time_ms + immc.kernel_time_ms, 'g--', 'LineWidth', 2);
hold off;
xlabel('Sphere Triangle Count');
ylabel('Time (ms)');
title('Total Time Breakdown (Mesh + Kernel)');
set(gca, 'XTick', 1:num_tests, 'XTickLabel', tri_labels);
xtickangle(45);
legend('Mesh Prep', 'GPU Kernel', sprintf('IMMC Total (%.0f ms)', ...
       immc.mesh_preprocess_time_ms + immc.kernel_time_ms), 'Location', 'northwest');
grid on;

% Ray-tet intersections comparison
subplot(2, 1, 2);
bar(1:num_tests, results.ray_tet_intersections, 'FaceColor', [0.6 0.4 0.7]);
hold on;
yline(immc.ray_tet_intersections, 'g--', 'LineWidth', 2);
hold off;
xlabel('Sphere Triangle Count');
ylabel('Ray-Tet Intersections');
title('Ray-Tet Workload');
set(gca, 'XTick', 1:num_tests, 'XTickLabel', tri_labels);
xtickangle(45);
legend('MMC', sprintf('IMMC (%.2e)', immc.ray_tet_intersections), 'Location', 'northwest');
grid on;

sgtitle('MMC vs IMMC Detailed Comparison');

savefig('mmc_immc_detailed.fig');
saveas(gcf, 'mmc_immc_detailed.png');

%% Export to CSV (timing data only, not flux)
writetable(T, 'mmc_benchmark_results.csv');

% Also save IMMC results
immc_table = table(immc.num_elements, immc.mesh_preprocess_time_ms, ...
                   immc.kernel_time_ms, immc.transfer_time_ms, ...
                   immc.photon_speed, immc.ray_tet_intersections, ...
                   'VariableNames', {'Elements', 'Mesh_ms', 'Kernel_ms', ...
                                     'Transfer_ms', 'Speed_ph_ms', 'RayTet'});
writetable(immc_table, 'immc_baseline_results.csv');

fprintf('\nBenchmark complete!\n');
fprintf('Results saved to:\n');
fprintf('  - mmc_benchmark_results.mat (includes all flux data)\n');
fprintf('  - mmc_benchmark_results.csv\n');
fprintf('  - immc_baseline_results.csv\n');
fprintf('  - mmc_immc_comparison.fig/.png\n');
fprintf('  - mmc_immc_detailed.fig/.png\n');

%% Display how to access saved data
fprintf('\nTo access saved flux data later:\n');
fprintf('  load(''mmc_benchmark_results.mat'');\n');
fprintf('  mcxplotvol(log10(results.flux_data{1}));  %% Plot first MMC result\n');
fprintf('  mcxplotvol(log10(immc.flux_data));        %% Plot IMMC result\n');
