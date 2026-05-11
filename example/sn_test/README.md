SN-SVMC two-sphere intersection test
====================================

A small reproducible benchmark for the SurfaceNets -> SVMC rasterization
pipeline. The input is a 60^3 labeled volume with two intersecting spheres:

    label 1   background
    label 2   sphere at center (20,20,20), radius 15, EXCLUDING intersection
    label 3   sphere at center (30,30,30), radius 15, EXCLUDING intersection
    label 4   intersection of the two spheres

There are 4 distinct material-pair boundaries (1-2, 1-3, 2-4, 3-4), so the
SN mesh must contain at least 4 surface patches that meet at junction
edges/corners. This exercises both ordinary surface generation and
sharp-feature preservation.


How to run
----------

  1. Build the test volume (one-time):
       octave gen_test_volume.m
     -> writes test_sn_vol.bin  (216000 bytes, MCX column-major uint8)

  2. Run MCX with SN-based SVMC preprocessing:
       ../../bin/mcx -f test_sn.json --svmc 2 -n 0

     --svmc 2  selects the SurfaceNets preprocessor (mode 1 = MC, mode 2 = SN).
     -n 0      runs 0 photons; we only want the preprocess outputs.

     This writes two files in the current directory:
       sn_surface.off          relaxed SN mesh, vertices in MCX coords
       sn_svmc_volume.nii      uint8 4D NIfTI-1 (8, Nx, Ny, Nz) holding the
                               full 8-byte-per-voxel SVMC volume, with the
                               8 components as the innermost (fastest-varying)
                               axis. The 8 bytes per voxel are
                               [nz, ny, nx, cz, cy, cx, upper, lower]; a
                               voxel is a boundary voxel iff upper != lower.

  3. Verify the rasterization:
       octave verify_shell.m

     Reports whether:
       (a) the OFF mesh is watertight + manifold
           = every edge shared by exactly 2 triangles
           = no boundary edges, no >2 incident triangles
       (b) the boundary mask shell is 1-voxel thick
           = no two consecutive boundary voxels along any of x, y, z


What to expect
--------------

A correctly rasterized output should:
  - have a watertight + manifold mesh (closed surfaces around each material),
  - have a SVMC boundary shell that is exactly 1 voxel thick everywhere.

If verify_shell.m reports thicker-than-1-voxel runs along some axes, the
rasterizer is over-painting; if it reports open edges in the OFF, the SN
mesh itself is broken.


Visualization
-------------

OFF mesh:
  - MeshLab, Blender, or any DCC that imports OFF.
  - Octave/iso2mesh: plotmesh(verts, tris).
  - Each label-pair boundary appears as its own quad shell; the four pairs
    meet at curve junctions that should look sharp.

SVMC volume:
  - The 4D NIfTI (8, Nx, Ny, Nz) is best inspected in Octave:
      gvol  = niftiread('sn_svmc_volume.nii');    % 8 x Nx x Ny x Nz, uint8
      upper = squeeze(gvol(7, :, :, :));          % byte 6 = upper label
      lower = squeeze(gvol(8, :, :, :));          % byte 7 = lower label
      mask  = upper ~= lower;
      isosurface(mask, 0.5);
  - Generic NIfTI viewers (3D Slicer, fslview, ITK-Snap) will show 8 frames
    along the component axis; pick frames 7 (upper) or 8 (lower) to see the
    label volume.
