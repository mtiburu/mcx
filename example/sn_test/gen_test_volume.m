%% Generate the two-sphere intersection test volume for SN-SVMC debugging.
%
%   60 x 60 x 60 cube, MCX column-major uint8 storage.
%     label 1   : background
%     label 2   : sphere centered at (20,20,20), radius 15, exclusive
%     label 3   : sphere centered at (30,30,30), radius 15, exclusive
%     label 4   : intersection of the two spheres
%
%   Voxel (i,j,k) has its lower-bottom corner at MCX (i,j,k) and its center
%   at (i+0.5, j+0.5, k+0.5), matching MCX's convention.
%
%   Output: test_sn_vol.bin (216000 bytes, column-major)

dim = 60;
[X, Y, Z] = ndgrid(0.5:dim - 0.5, 0.5:dim - 0.5, 0.5:dim - 0.5);

sphere1 = (X - 20).^2 + (Y - 20).^2 + (Z - 20).^2 <= 15 ^ 2;
sphere2 = (X - 30).^2 + (Y - 30).^2 + (Z - 30).^2 <= 15 ^ 2;

vol = ones(dim, dim, dim, 'uint8');
vol(sphere1) = 2;
vol(sphere2) = 3;
vol(sphere1 & sphere2) = 4;

fid = fopen('test_sn_vol.bin', 'wb');
fwrite(fid, vol, 'uint8');
fclose(fid);

fprintf('Wrote test_sn_vol.bin  (%d voxels: bg=%d, A=%d, B=%d, A&B=%d)\n', ...
        numel(vol), sum(vol(:) == 1), sum(vol(:) == 2), sum(vol(:) == 3), sum(vol(:) == 4));
