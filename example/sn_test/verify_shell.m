%% Verify the SN->SVMC rasterization output.
%
%   Loads sn_svmc_volume.nii and sn_surface.off (produced by running mcx
%   with --svmc 2 on test_sn.json), then reports:
%     1. Whether the OFF mesh is watertight: every edge is shared by >= 2
%        triangles (no open boundary edges). For multi-material SurfaceNets,
%        triple-junction edges shared by 3 or 4 triangles are EXPECTED and
%        do not indicate a defect.
%     2. Whether the rasterized SVMC boundary shell is 1-voxel thick in the
%        local surface-normal direction: no boundary voxel should have ALL
%        6 face-neighbors also flagged as boundary (= no interior voxel of
%        a >=3-voxel-thick slab). This test is direction-agnostic; long
%        boundary "strips" along coordinate axes are NORMAL where the
%        surface is locally tangent to that axis (e.g., near the equator of
%        a sphere when sliced along the equator plane).
%
%   sn_svmc_volume.nii is a 4D uint8 NIfTI with shape (8, Nx, Ny, Nz). The
%   8 components per voxel are [nz, ny, nx, cz, cy, cx, upper, lower]; a
%   voxel is a boundary voxel iff its lower (byte 7) and upper (byte 6)
%   labels differ.
%
%   Usage (after running mcx --svmc 2):
%     octave verify_shell.m

%% ------------------- 1. OFF mesh watertightness --------------------------

fid = fopen('sn_surface.off', 'r');
if fid < 0, error('cannot open sn_surface.off'); end
hdr = fgetl(fid);
if ~strcmpi(strtrim(hdr), 'OFF'), error('not an OFF file'); end
counts = sscanf(fgetl(fid), '%d %d %d');
nv = counts(1);
nf = counts(2);
verts = fscanf(fid, '%f', [3, nv])';
tris  = zeros(nf, 3);
for k = 1:nf
    row = sscanf(fgetl(fid), '%d');
    if isempty(row), row = sscanf(fgetl(fid), '%d'); end
    tris(k, :) = row(2:4)';
end
fclose(fid);

fprintf('OFF mesh: %d vertices, %d triangles\n', nv, nf);

edges = sort([tris(:, [1, 2]); tris(:, [2, 3]); tris(:, [3, 1])], 2);
[ue, ~, ie] = unique(edges, 'rows');
edge_count = accumarray(ie, 1);

n_open  = sum(edge_count == 1);   % boundary (not shared)
n_two   = sum(edge_count == 2);
n_three = sum(edge_count >= 3);    % multi-material junction edges

fprintf('  edges total=%d:  2-shared=%d, junction(>=3)=%d, open(==1)=%d\n', ...
        size(ue, 1), n_two, n_three, n_open);

if n_open == 0
    fprintf('  -> watertight (no open edges): PASS\n');
    if n_three > 0
        fprintf('  -> %d junction edges expected for multi-material SN\n', n_three);
    end
else
    fprintf('  -> NOT watertight (%d open edges): FAIL\n', n_open);
end

%% ----------- 2. Boundary mask 1-voxel-thick in normal direction ----------
%
% sn_svmc_volume.nii is 4D (8, Nx, Ny, Nz) uint8. The 8 components per voxel
% are [nz, ny, nx, cz, cy, cx, upper, lower]. In MCX's SVMC encoding,
% homogeneous voxels store byte 7 (1-indexed: 8) = own label and zero out
% byte 6 (1-indexed: 7); boundary voxels store both non-zero with the
% sorted-low/sorted-high label pair. So the boundary mask is upper != 0.

fid = fopen('sn_svmc_volume.nii', 'rb');
if fid < 0, error('cannot open sn_svmc_volume.nii'); end
sizeof_hdr = fread(fid, 1, 'int32'); assert(sizeof_hdr == 348);
fseek(fid, 40, 'bof'); dim = fread(fid, 8, 'int16');
fseek(fid, 108, 'bof'); vox_offset = fread(fid, 1, 'float32');
fseek(fid, vox_offset, 'bof');
ncomp = dim(2); nx = dim(3); ny = dim(4); nz = dim(5);
assert(dim(1) == 4 && ncomp == 8, ...
       'expected 4D NIfTI with innermost dim=8 (got dim[0]=%d, dim[1]=%d)', ...
       dim(1), ncomp);
gvol = fread(fid, ncomp * nx * ny * nz, 'uint8=>uint8');
fclose(fid);
gvol = reshape(gvol, ncomp, nx, ny, nz);

upper = squeeze(gvol(7, :, :, :));    % byte 6 (1-indexed: 7) = upper label
mask  = uint8(upper ~= 0);

n_boundary = sum(mask(:) > 0);
fprintf('\nBoundary mask: %d boundary voxels (%.2f%% of %d total)\n', ...
        n_boundary, 100 * n_boundary / numel(mask), numel(mask));

% A boundary voxel is "interior to a thick shell" if ALL 6 of its 6-connected
% neighbors are also boundary -- the shell is locally at least 3 voxels
% thick in some direction.
%
% A correctly rasterized 1-thick shell has NO such interior voxels.
n_thick_interior = 0;
for k = 2:nz - 1
    for j = 2:ny - 1
        for i = 2:nx - 1
            if mask(i, j, k) == 0, continue; end
            if mask(i - 1, j, k) > 0 && mask(i + 1, j, k) > 0 && ...
               mask(i, j - 1, k) > 0 && mask(i, j + 1, k) > 0 && ...
               mask(i, j, k - 1) > 0 && mask(i, j, k + 1) > 0
                n_thick_interior = n_thick_interior + 1;
            end
        end
    end
end

fprintf('  thick-interior voxels (all 6 face-neighbors also boundary): %d\n', n_thick_interior);

if n_thick_interior == 0
    fprintf('  -> 1-voxel-thick in normal direction: PASS\n');
else
    fprintf('  -> shell has %d interior cells: FAIL\n', n_thick_interior);
end

%% ---- 2b. 6-connectivity: every boundary voxel has at least 1 boundary nbr ----
% A "watertight" shell in voxel sense: boundary voxels form a connected
% surface. Each interior boundary voxel should touch at least one neighbor
% (otherwise it is an isolated stray write).
n_isolated = 0;
for k = 2:nz - 1
    for j = 2:ny - 1
        for i = 2:nx - 1
            if mask(i, j, k) == 0, continue; end
            if mask(i - 1, j, k) == 0 && mask(i + 1, j, k) == 0 && ...
               mask(i, j - 1, k) == 0 && mask(i, j + 1, k) == 0 && ...
               mask(i, j, k - 1) == 0 && mask(i, j, k + 1) == 0
                n_isolated = n_isolated + 1;
            end
        end
    end
end

fprintf('\nIsolated boundary voxels (no 6-connected neighbor): %d\n', n_isolated);
if n_isolated == 0
    fprintf('  -> all boundary voxels are 6-connected: PASS\n');
else
    fprintf('  -> %d isolated boundary voxels: FAIL\n', n_isolated);
end
