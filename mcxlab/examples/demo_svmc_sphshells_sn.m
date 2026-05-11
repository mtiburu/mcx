%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCXLAB - Monte Carlo eXtreme for MATLAB/Octave by Qianqian Fang
%
% Three-way comparison of voxelized photon transport on the concentric
% sphere-shell phantom from demo_svmc_sphshells.m:
%   (1) conventional VMC                        cfg.issvmc = 0
%   (2) marching-cubes SVMC ("SVMC (MC mode)")  cfg.issvmc = 1
%   (3) SurfaceNets-based SVMC ("SN-SVMC")      cfg.issvmc = 2
%
% Where demo_svmc_sphshells.m draws SVMC vs VMC and SN vs VMC in two
% separate panels, this script overlays all three methods on a single
% mid-plane contour map, then plots a 1D fluence profile along the
% source axis so the agreement between SN-SVMC, MC-SVMC, and VMC can be
% read off quantitatively.
%
% This file is part of Monte Carlo eXtreme (MCX) URL:http://mcx.sf.net
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear cfg cfg_mcx cfg_svmc cfg_sn;

%% common MC setup
cfg.nphoton = 1e8;
cfg.seed = randi([1 2^31 - 1], 1, 1); % random seed

% pencil beam light source
cfg.srcpos = [30.5 30.5 0];
cfg.srcdir = [0 0 1];
cfg.issrcfrom0 = 1;

% optical properties (tissue-like multi-layered media)
cfg.prop = [0.0   0.0   1.0  1.0    % background (air, void)
            0.02  7.0   0.89 1.37   % scalp/skull
            0.004 0.009 0.89 1.37   % CSF
            0.02  9.0   0.89 1.37   % gray matter
            0.05  0.0   1.0  1.37]; % non-scattering inclusion

% time-domain simulation parameters
cfg.tstart = 0;
cfg.tend = 5e-9;
cfg.tstep = 5e-10;

% enable boundary reflection/refraction
cfg.isreflect = 1;

% spatial resolution
cfg.unitinmm = 1;

% output fluence
cfg.outputtype = 'fluence';

% GPU settings
cfg.gpuid = 1;
cfg.autopilot = 1;

%% prepare VMC input volume (voxel-center sampling at 0.5..dim-0.5)
dim = 60;
[xi, yi, zi] = ndgrid(0.5:(dim - 0.5), 0.5:(dim - 0.5), 0.5:(dim - 0.5));
dist = (xi - 30.5).^2 + (yi - 30.5).^2 + (zi - 30.5).^2;
mcxvol = ones(size(xi));
mcxvol(dist < 625) = 2;
mcxvol(dist < 529) = 3;
mcxvol(dist < 100) = 4;

cfg_mcx = cfg;
cfg_mcx.vol = uint8(mcxvol);

%% prepare SVMC input volume (voxel-center sampling at 0.5..dim-0.5)
%   Both SVMC variants take the same labeled volume; the preprocessor
%   inside mcx decides whether to build the boundary mesh via marching
%   cubes (issvmc=1) or SurfaceNets (issvmc=2).
%
%   Note: we use the SAME ndgrid as the VMC volume above. The original
%   demo_svmc_sphshells.m used ndgrid(1:dim) which placed the phantom
%   center at MCX 30, half a voxel away from the source at MCX 30.5.
%   That source/phantom misalignment shows up as a left/right asymmetry
%   in the SN-SVMC contours (left side lower than right by ~1 mm of
%   extra travel through the phantom). Using ndgrid(0.5:dim-0.5) puts
%   the phantom center at MCX 30.5, exactly at the source, so all three
%   methods see a symmetric geometry.
[xi, yi, zi] = ndgrid(0.5:(dim - 0.5), 0.5:(dim - 0.5), 0.5:(dim - 0.5));
dist = (xi - 30.5).^2 + (yi - 30.5).^2 + (zi - 30.5).^2;
svmcvol = ones(size(xi));
svmcvol(dist < 625) = 2;
svmcvol(dist < 529) = 3;
svmcvol(dist < 100) = 4;

cfg_svmc = cfg;
cfg_svmc.vol = uint8(svmcvol);
cfg_svmc.issvmc = 1;   % marching-cubes SVMC

cfg_sn = cfg;
cfg_sn.vol = uint8(svmcvol);
cfg_sn.issvmc = 2;     % SurfaceNets SVMC

%% run simulations
addpath ../;
output_vmc  = mcxlab(cfg_mcx);
output_svmc = mcxlab(cfg_svmc);
output_sn   = mcxlab(cfg_sn);

%% time-integrate to CW fluence
phi_vmc  = sum(output_vmc.data,  4);
phi_svmc = sum(output_svmc.data, 4);
phi_sn   = sum(output_sn.data,   4);

%% Figure 1: mid-plane (x = 31) contour overlay of all three methods
figure;
clines = -10:0.5:10;

slice_vmc  = log10(abs(squeeze(phi_vmc(31, :, :))'));
slice_svmc = log10(abs(squeeze(phi_svmc(31, :, :))'));
slice_sn   = log10(abs(squeeze(phi_sn(31, :, :))'));

% draw VMC as filled contour underneath, then overlay SVMC + SN as line contours
contourf(slice_vmc, clines, 'linestyle', 'none', 'DisplayName', 'VMC (fill)');
hold on;
contour(slice_svmc, clines, 'linestyle', '--', ...
        'linecolor', 'w', 'linewidth', 1.8, 'DisplayName', 'SVMC (MC)');
contour(slice_sn, clines, 'linestyle', '-', ...
        'linecolor', 'k', 'linewidth', 1.8, 'DisplayName', 'SN-SVMC');
colorbar('EastOutside');

% media boundaries (radii 10, 23, 25 around (31,31))
radii = [10, 23, 25];
for r = radii
    [xcirc, ycirc] = cylinder([r, r], 200);
    xcirc = xcirc(1, :) + 31;
    ycirc = ycirc(1, :) + 31;
    plot(xcirc, ycirc, '--', 'linewidth', 1.5, 'color', [.4 .4 .4], 'HandleVisibility', 'off');
end

axis equal;
xlabel('y (mm)');
ylabel('z (mm)');
title('VMC vs MC-SVMC vs SN-SVMC: log_{10} fluence on mid-plane (x=31)');
lg = legend('Location', 'NorthEast');
set(lg, 'Color', [0.5 0.5 0.5]);
set(lg, 'Box', 'on');
set(gca, 'FontSize', 14);

%% Figure 2: 1D line profile along the source axis (x=31, y=31)
%   This is the quantitative comparison: deviations between the three
%   methods show up immediately as separated curves on the log axis.
figure;
zaxis = 1:dim;
profile_vmc  = squeeze(phi_vmc(31,  31, :));
profile_svmc = squeeze(phi_svmc(31, 31, :));
profile_sn   = squeeze(phi_sn(31,   31, :));

semilogy(zaxis, profile_vmc,  '-',  'linewidth', 2, 'DisplayName', 'VMC');
hold on;
semilogy(zaxis, profile_svmc, '--', 'linewidth', 2, 'DisplayName', 'SVMC (MC)');
semilogy(zaxis, profile_sn,   ':',  'linewidth', 2.5, 'DisplayName', 'SN-SVMC');

% mark the layer boundaries the beam crosses along z
for r = radii
    xline = 30.5 - r;
    if xline >= 1 && xline <= dim
        plot([xline xline], ylim, ':', 'color', [.6 .6 .6], 'HandleVisibility', 'off');
    end
    xline = 30.5 + r;
    if xline >= 1 && xline <= dim
        plot([xline xline], ylim, ':', 'color', [.6 .6 .6], 'HandleVisibility', 'off');
    end
end

xlabel('z (mm)');
ylabel('CW fluence (a.u.)');
title('On-axis fluence profile (x=31, y=31)');
legend('Location', 'NorthEast');
grid on;
set(gca, 'FontSize', 14);

%% Figure 3: relative difference of each SVMC variant against VMC
figure;
ref = profile_vmc;
ref(ref == 0) = NaN;     % avoid divide-by-zero outside the simulation domain
rel_svmc = (profile_svmc - ref) ./ ref;
rel_sn   = (profile_sn   - ref) ./ ref;

plot(zaxis, 100 * rel_svmc, '--', 'linewidth', 2, 'DisplayName', 'SVMC (MC) - VMC');
hold on;
plot(zaxis, 100 * rel_sn,   '-',  'linewidth', 2, 'DisplayName', 'SN-SVMC - VMC');
plot(xlim, [0 0], 'k:', 'HandleVisibility', 'off');

xlabel('z (mm)');
ylabel('relative difference (%)');
title('Per-z relative difference vs VMC, on the source axis');
legend('Location', 'NorthEast');
grid on;
set(gca, 'FontSize', 14);
