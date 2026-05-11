/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2024
**
**  \section sref Reference
**  \li \c (\b Yan2020) Shijie Yan and Qianqian Fang* (2020), "Hybrid mesh and
**          voxel based Monte Carlo algorithm for accurate and efficient photon
**          transport modeling in complex bio-tissues," Biomed. Opt. Express,
**          11(11) pp. 6262-6270. https://doi.org/10.1364/BOE.409468
**  \li \c (\b Frisken2022) S. Frisken, "SurfaceNets for Multi-Label
**          Segmentations with Preservation of Sharp Boundaries," J. Computer
**          Graphics Techniques, 11(1), pp. 34-54, 2022.
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    mcx_svmc_surfacenets.cu

@brief   Surface-Nets driven Split-Voxel MC (SVMC) preprocessor.

The MMSurfaceNet machinery builds a single watertight surface mesh that
separates every pair of materials in the input label volume. Vertex
relaxation produces smooth, multi-material surfaces but lets quad/triangle
patches drift across voxel boundaries. We rasterize that relaxed mesh
into the 8-byte-per-voxel SVMC volume that MCX's photon kernel consumes.

Coordinate alignment (read this before changing anything):

  * MCX stores a column-major Nx*Ny*Nz label volume. Voxel (i,j,k) occupies
    real-world coordinates [i, i+1) x [j, j+1) x [k, k+1) in voxel units
    (multiply by cfg->unitinmm for mm). Its center is at (i+0.5, j+0.5, k+0.5).
  * The Surface-Nets lattice lives BETWEEN the input voxel centers: an SN
    "cube" at padded cell index ci has its 8 corner labels sampled from the
    voxel centers around it. After conversion to MCX coordinates, an SN cube
    occupies [ci-0.5, ci+0.5] per axis. SN vertices live inside that range.
  * Therefore: mcx_coord = padded_world / voxelsize - 0.5 (per axis).
  * MMCellMap pads the input volume by one cell on each side, so padded
    cell index ci=1 holds the label of MCX voxel index 0; the SN trace
    iterates the padded array, but every output coordinate must be shifted
    by -0.5 *voxelsize* to land in MCX space.

Rasterization (this file's job):

  We rasterize the RELAXED SN surface using DDA-style per-triangle
  4-point sampling: for each surface triangle in the relaxed SN mesh (two
  triangles per quad, three quads per active cube), compute 4 sample
  points (the triangle's centroid plus the midpoints between centroid
  and each corner), and accumulate each sample's contribution into the
  MCX voxel that contains it. This traces the actual relaxed (smoothed)
  surface across the voxel grid: a voxel receives a record if and only if
  the relaxed surface passes through it (up to sample-point precision).

  This is intentionally NOT a strict-1-voxel-thick rasterization. Where
  the relaxed surface is locally tangent to a voxel face plane (e.g. the
  equator of a sphere whose center sits on a voxel corner), a single
  triangle physically straddles two voxels along the normal direction
  and both voxels get a record. That two-voxel band is geometrically
  correct: the smooth surface really does cross both voxels. The
  rasterized voxel set is DDA-contiguous in the 26-connectivity sense
  (single connected component, coherent normals across adjacent voxels).

  Each voxel accumulates a sum of:
    - the unit outward normal (oriented from lower-label to upper-label
      material so the SVMC photon code reads byte 8 as the -n^ tissue
      and byte 7 as the +n^ tissue), weighted by triangle_area / 4 per
      sample,
    - the triangle centroid (in MCX coords), same weighting,
    - the dominant {lower, upper} label pair (chosen by largest single
      triangle area to date).
  After accumulation, normal is re-normalized to unit length and the
  centroid is converted to a [0,1] local offset within the receiving
  voxel (clamped, since the triangle centroid can sit slightly outside
  the voxel a single sample point landed in).

Output layout (matches mcx_svmc.cu exactly):
  Per voxel, packed 8 bytes [nz, ny, nx, cz, cy, cx, upper, lower], then
  repacked into MCX's interleaved [cy,cx,upper,lower | nz,ny,nx,cz] format.

*******************************************************************************/

#include "MMCellMap.cuh"
#include "MMSurfaceNet.cuh"
#include "MMCellFlag.cuh"
#include "mcx_svmc.h"
#include "mcx_tictoc.h"
#include "mcx_const.h"
#include "mcx_vector_math.cu"
#include "nifti1.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

/**
 * Per-voxel accumulator. Filled on the CPU by rasterize_svmc(), then
 * encoded to the 8-byte SVMC layout by finalize_svmc_kernel().
 *
 * During accumulation:
 *   cx/cy/cz   - area-weighted MCX-coordinate centroid sums.
 *   nx/ny/nz   - area-weighted unit-normal sums (lower-to-upper oriented).
 *   w          - total area weight accumulated at this voxel.
 *   lower/upper- the label pair from the largest single contribution.
 *   best_area  - area of that largest single contribution.
 *
 * After normalization (end of rasterize_svmc):
 *   cx/cy/cz   - centroid local offset within this voxel, in [0, 1].
 *   nx/ny/nz   - unit normal pointing from lower-label to upper-label tissue.
 *   w          - retained as "non-empty" marker for the finalize kernel.
 */
struct SVMCRecord {
    float    cx, cy, cz;
    float    nx, ny, nz;
    float    w;
    uint32_t lower;
    uint32_t upper;
    float    best_area;
};

/* ---------------------------------------------------------------------------
 * Forward declarations
 * ------------------------------------------------------------------------ */

__global__ void finalize_svmc_kernel(unsigned char* gvol, const SVMCRecord* records,
                                     const unsigned short* vol_labels, long vol_length,
                                     int dimx, int dimy, int dimz, int nMedia);

__global__ void repack_kernel(unsigned int* newvol, const unsigned char* gvol,
                              long vol_length);

static void rasterize_svmc(MMCellMap* cellmap, int dims[3], float voxelsize[3],
                           SVMCRecord* records);

static void dump_sn_surface_off(MMSurfaceNet* sn, float voxelsize[3],
                                const char* filename);

static void dump_svmc_volume_nii(const unsigned char* gvol_bytes, int dims[3],
                                 float voxelsize[3], const char* filename);

/* ---------------------------------------------------------------------------
 * Main entry point: build SN -> relax -> rasterize -> encode
 * ------------------------------------------------------------------------ */

void mcx_svmc_preprocess_surfacenets(Config* cfg, GPUInfo* gpu) {
    if (cfg->mediabyte > 4 || !cfg->issvmc) {
        return;
    }

    MCX_FPRINTF(cfg->flog, "Surface Nets SVMC preprocessing...\n");
    unsigned int tic = StartTimer();

    int   dims[3]      = {(int)cfg->dim.x, (int)cfg->dim.y, (int)cfg->dim.z};
    float voxelsize[3] = {cfg->unitinmm,   cfg->unitinmm,   cfg->unitinmm};
    long  vol_length   = (long)dims[0] * dims[1] * dims[2];

    /* 1. Snapshot the label volume the SN expects (unsigned short, with
     *    0xFFFF reserved for padding). MCX may carry extra bits in cfg->vol;
     *    keep only the medium index. */
    unsigned short* h_vol_labels = new unsigned short[vol_length];

    for (long i = 0; i < vol_length; i++) {
        unsigned short lbl = (unsigned short)(cfg->vol[i] & MED_MASK);
        h_vol_labels[i]    = (lbl == 0xFFFF) ? 0 : lbl;
    }

    /* 2. Build the Surface-Nets mesh and relax it. */
    MMSurfaceNet sn(h_vol_labels, dims, voxelsize);
    MMSurfaceNet::RelaxAttrs relaxAttrs{10, 0.5f, 0.45f};
    sn.relax(relaxAttrs);

    /* 3. Dump the relaxed SN surface as an OFF mesh in MCX coordinates. */
    dump_sn_surface_off(&sn, voxelsize, "sn_surface.off");

    /* 4. Rasterize relaxed mesh into per-voxel SVMC records. */
    SVMCRecord* h_records = new SVMCRecord[vol_length]();
    rasterize_svmc(sn.cellMap(), dims, voxelsize, h_records);

    long active = 0;

    for (long i = 0; i < vol_length; i++) {
        if (h_records[i].w > 0.0f) {
            active++;
        }
    }

    MCX_FPRINTF(cfg->flog, "[SN] active surface voxels: %ld / %ld\n", active, vol_length);

    /* 5. Encode records to MCX's interleaved 8-byte-per-voxel SVMC format on GPU. */
    SVMCRecord*     d_records;
    unsigned short* d_vol_labels;
    unsigned char*  d_gvol;
    unsigned int*   d_newvol;

    cudaMalloc(&d_records,    vol_length * sizeof(SVMCRecord));
    cudaMalloc(&d_vol_labels, vol_length * sizeof(unsigned short));
    cudaMalloc(&d_gvol,       vol_length * 8);
    cudaMalloc(&d_newvol,     vol_length * 2 * sizeof(unsigned int));

    cudaMemcpy(d_records,    h_records,    vol_length * sizeof(SVMCRecord),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_vol_labels, h_vol_labels, vol_length * sizeof(unsigned short), cudaMemcpyHostToDevice);

    int threads      = 256;
    int voxel_blocks = (int)((vol_length + threads - 1) / threads);

    finalize_svmc_kernel <<< voxel_blocks, threads>>>(d_gvol, d_records, d_vol_labels,
            vol_length, dims[0], dims[1], dims[2],
            cfg->medianum);
    cudaDeviceSynchronize();

    /* 6. Dump the encoded 8-byte-per-voxel volume as a 4D NIfTI for inspection. */
    {
        unsigned char* h_gvol = (unsigned char*)malloc(vol_length * 8);
        cudaMemcpy(h_gvol, d_gvol, vol_length * 8, cudaMemcpyDeviceToHost);
        dump_svmc_volume_nii(h_gvol, dims, voxelsize, "sn_svmc_volume.nii");
        free(h_gvol);
    }

    repack_kernel <<< voxel_blocks, threads>>>(d_newvol, d_gvol, vol_length);
    cudaDeviceSynchronize();

    /* 7. Copy back to host and hand off to MCX. */
    unsigned int* h_newvol = (unsigned int*)malloc(vol_length * 2 * sizeof(unsigned int));
    cudaMemcpy(h_newvol, d_newvol, vol_length * 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    MCX_FPRINTF(cfg->flog, "Surface Nets complete: %d ms\n", GetTimeMillis() - tic);

    cudaFree(d_records);
    cudaFree(d_vol_labels);
    cudaFree(d_gvol);
    cudaFree(d_newvol);
    delete[] h_records;
    delete[] h_vol_labels;
    free(cfg->vol);

    cfg->vol       = h_newvol;
    cfg->mediabyte = MEDIA_2LABEL_SPLIT;

    /* Post-processing: source position fixup + detector mask (identical to MC). */
    if (cfg->srctype <= MCX_SRC_CONE || cfg->srctype == MCX_SRC_ARCSINE ||
            cfg->srctype == MCX_SRC_ZGAUSSIAN) {
        if (cfg->srcpos.x < 0.f || cfg->srcpos.y < 0.f || cfg->srcpos.z < 0.f ||
                cfg->srcpos.x >= cfg->dim.x || cfg->srcpos.y >= cfg->dim.y ||
                cfg->srcpos.z >= cfg->dim.z) {
            *((uint*)&cfg->srcparam2.z) = 0;
            *((uint*)&cfg->srcparam2.w) = 0;
        } else {
            uint idx1dorig =
                ((int)floorf(cfg->srcpos.z)) * (cfg->dim.y * cfg->dim.x) +
                ((int)floorf(cfg->srcpos.y)) * cfg->dim.x +
                ((int)floorf(cfg->srcpos.x));
            *((uint*)&cfg->srcparam2.z) = idx1dorig;
            *((uint*)&cfg->srcparam2.w) = (cfg->vol[idx1dorig] & MED_MASK);
        }

        if (cfg->extrasrclen) {
            for (unsigned int i = 0; i < cfg->extrasrclen; i++) {
                float sx = cfg->srcdata[i].srcpos.x;
                float sy = cfg->srcdata[i].srcpos.y;
                float sz = cfg->srcdata[i].srcpos.z;

                if (sx < 0.f || sy < 0.f || sz < 0.f ||
                        sx >= cfg->dim.x || sy >= cfg->dim.y || sz >= cfg->dim.z) {
                    *((uint*)&cfg->srcdata[i].srcparam2.z) = 0;
                    *((uint*)&cfg->srcdata[i].srcparam2.w) = 0;
                } else {
                    uint idx1dorig =
                        ((int)floorf(sz)) * (cfg->dim.y * cfg->dim.x) +
                        ((int)floorf(sy)) * cfg->dim.x +
                        ((int)floorf(sx));
                    *((uint*)&cfg->srcdata[i].srcparam2.z) = idx1dorig;
                    *((uint*)&cfg->srcdata[i].srcparam2.w) =
                        (cfg->vol[idx1dorig] & MED_MASK);
                }
            }
        }
    }

    mcx_maskdet(cfg);
}

/* ---------------------------------------------------------------------------
 * CPU: rasterize_svmc
 *
 * One pass over every surface triangle in the relaxed SN mesh. Quads are
 * split into two triangles (ABC + ACD); each triangle is sampled at
 * (centroid, midpoint(centroid,A), midpoint(centroid,B), midpoint(centroid,C)).
 * Each sample's enclosing MCX voxel accumulates triangle_area/4 worth of
 * area-weighted contribution to normal, centroid, and dominant material pair.
 * Voxels that the relaxed surface does not pass through get no record.
 *
 * Ownership: each active cube emits its 3 "back" edges (left-bottom,
 * back-bottom, left-back); the 9 forward edges are owned by neighbor cubes
 * - this matches the CPU SurfaceNets reference and ensures every quad is
 * rasterized exactly once.
 *
 * Padded->MCX conversion:
 *   mcx_coord = padded_world / voxelsize - 0.5
 *   (Padded cell 1 holds MCX voxel 0; MCX voxel 0's center is at MCX 0.5;
 *    padded cell index 1 maps to MCX 0.5 via the -0.5 shift.)
 *
 * Outward normal:
 *   Each per-triangle cross product (B-A)x(C-A) is oriented "from labels[0]
 *   side to labels[1] side" in cell-ordering (a property of MMCellMap's
 *   getEdgeQuadVtxIndices winding). We flip the sign when labels[0] >
 *   labels[1] so the final normal always points from the lower-numerical
 *   material to the higher one. SVMC photon code reads byte 8 as the
 *   -n^ tissue (= lower) and byte 7 as the +n^ tissue (= upper).
 *
 * Ownership: each active cube emits its 3 "back" edges (left-bottom,
 * back-bottom, left-back); the 9 forward edges are owned by neighbor cubes
 * - this matches the CPU SurfaceNets reference and ensures every quad is
 * rasterized exactly once.
 *
 * Padded->MCX conversion:
 *   mcx_coord = padded_world / voxelsize - 0.5
 *   (Padded cell 1 holds MCX voxel 0; MCX voxel 0's center is at MCX 0.5;
 *    padded cell index 1 maps to MCX 0.5 via the -0.5 shift.)
 *
 * Outward normal:
 *   The per-triangle cross product (B-A)x(C-A) is oriented "from labels[0]
 *   side to labels[1] side" in cell-ordering (a property of MMCellMap's
 *   getEdgeQuadVtxIndices winding). We flip the sign when labels[0] >
 *   labels[1] so the final normal always points from the lower-numerical
 *   material to the higher one. SVMC photon code reads byte 8 as the
 *   -n^ tissue (= lower) and byte 7 as the +n^ tissue (= upper).
 * ------------------------------------------------------------------------ */

static void rasterize_svmc(MMCellMap* cellmap, int dims[3], float voxelsize[3],
                           SVMCRecord* records) {
    if (!cellmap) {
        return;
    }

    /* Each cube owns its three back edges; the other nine are processed when
     * the neighbor cube reaches its back-edge. This avoids duplicate
     * rasterization of the same quad from four different cubes. */
    const MMCellFlag::Edge owned_edges[3] = {
        MMCellFlag::LeftBottomEdge,
        MMCellFlag::BackBottomEdge,
        MMCellFlag::LeftBackEdge
    };

    /* Two triangles per quad: ABC and ACD. */
    const int tri_corner_idx[2][3] = {{0, 1, 2}, {0, 2, 3}};

    const float inv_vs[3] = {1.0f / voxelsize[0], 1.0f / voxelsize[1], 1.0f / voxelsize[2]};
    const int   nv        = cellmap->numVertices();

    for (int v = 0; v < nv; v++) {
        for (int e = 0; e < 3; e++) {
            float          corners_padded[12];
            unsigned short labels[2];

            if (!cellmap->getEdgeQuad(v, owned_edges[e], corners_padded, labels)) {
                continue;
            }

            /* Skip same-material edges and any edge that touches the
             * 0xFFFF padding label. */
            if (labels[0] == labels[1] ||
                    labels[0] == 0xFFFF || labels[1] == 0xFFFF) {
                continue;
            }

            /* Sort labels into (lo < hi) and remember whether the cell-order
             * cross product needs flipping to point lo -> hi. */
            unsigned short lo, hi;
            float          sgn;

            if (labels[0] < labels[1]) {
                lo  = labels[0];
                hi  = labels[1];
                sgn = +1.0f;
            } else {
                lo  = labels[1];
                hi  = labels[0];
                sgn = -1.0f;
            }

            /* Convert the 4 padded-world quad corners into MCX coordinates. */
            float corners[12];

            for (int i = 0; i < 4; i++) {
                corners[i * 3 + 0] = corners_padded[i * 3 + 0] * inv_vs[0] - 0.5f;
                corners[i * 3 + 1] = corners_padded[i * 3 + 1] * inv_vs[1] - 0.5f;
                corners[i * 3 + 2] = corners_padded[i * 3 + 2] * inv_vs[2] - 0.5f;
            }

            /* Two triangles per quad. */
            for (int t = 0; t < 2; t++) {
                float A[3], B[3], C[3];

                for (int d = 0; d < 3; d++) {
                    A[d] = corners[tri_corner_idx[t][0] * 3 + d];
                    B[d] = corners[tri_corner_idx[t][1] * 3 + d];
                    C[d] = corners[tri_corner_idx[t][2] * 3 + d];
                }

                /* Triangle centroid, in MCX coordinates. */
                float Tc[3] = {
                    (A[0] + B[0] + C[0]) / 3.0f,
                    (A[1] + B[1] + C[1]) / 3.0f,
                    (A[2] + B[2] + C[2]) / 3.0f
                };

                /* Outward unit normal: (B-A) x (C-A), sign-adjusted to point lo -> hi. */
                float u[3] = {B[0] - A[0], B[1] - A[1], B[2] - A[2]};
                float w[3] = {C[0] - A[0], C[1] - A[1], C[2] - A[2]};
                float n[3] = {
                    u[1]* w[2] - u[2]* w[1],
                    u[2]* w[0] - u[0]* w[2],
                    u[0]* w[1] - u[1]* w[0]
                };
                float n_mag = sqrtf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);

                if (n_mag < 1e-12f) {
                    continue;
                }

                float area  = 0.5f * n_mag;
                float nu[3] = {sgn* n[0] / n_mag, sgn* n[1] / n_mag, sgn* n[2] / n_mag};

                /* 4 sample points: centroid + midpoints (centroid, corner). */
                float samples[4][3] = {
                    {Tc[0], Tc[1], Tc[2]},
                    {0.5f * (Tc[0] + A[0]), 0.5f * (Tc[1] + A[1]), 0.5f * (Tc[2] + A[2])},
                    {0.5f * (Tc[0] + B[0]), 0.5f * (Tc[1] + B[1]), 0.5f * (Tc[2] + B[2])},
                    {0.5f * (Tc[0] + C[0]), 0.5f * (Tc[1] + C[1]), 0.5f * (Tc[2] + C[2])}
                };

                float sample_w = area * 0.25f;

                for (int s = 0; s < 4; s++) {
                    /* Voxel selection matches mcx_svmc.cu's convention: a
                     * cube spanning MCX (V-0.5, V+0.5) writes its record at
                     * voxel V (the voxel whose center is the cube's upper
                     * corner). That is, samples are rounded to the nearest
                     * MCX integer rather than floored. Without this shift
                     * SN records sit half a voxel "earlier" than MC's. */
                    int vx = (int)floorf(samples[s][0] + 0.5f);
                    int vy = (int)floorf(samples[s][1] + 0.5f);
                    int vz = (int)floorf(samples[s][2] + 0.5f);

                    if (vx < 0 || vx >= dims[0] ||
                            vy < 0 || vy >= dims[1] ||
                            vz < 0 || vz >= dims[2]) {
                        continue;
                    }

                    long        idx = (long)vx + (long)vy * dims[0] + (long)vz * dims[0] * dims[1];
                    SVMCRecord& r   = records[idx];

                    r.cx += Tc[0] * sample_w;
                    r.cy += Tc[1] * sample_w;
                    r.cz += Tc[2] * sample_w;
                    r.nx += nu[0] * sample_w;
                    r.ny += nu[1] * sample_w;
                    r.nz += nu[2] * sample_w;
                    r.w  += sample_w;

                    /* Take the dominant material pair (largest single-triangle area). */
                    if (area > r.best_area) {
                        r.best_area = area;
                        r.lower     = lo;
                        r.upper     = hi;
                    }
                }
            }
        }
    }

    /* Normalize: convert accumulated centroid to a local [0,1] offset within
     * each voxel and renormalize the accumulated normal to unit length. */
    long total_voxels = (long)dims[0] * dims[1] * dims[2];

    for (long idx = 0; idx < total_voxels; idx++) {
        SVMCRecord& r = records[idx];

        if (r.w < 1e-12f) {
            continue;
        }

        int vx = (int)(idx % dims[0]);
        int vy = (int)((idx / dims[0]) % dims[1]);
        int vz = (int)(idx / ((long)dims[0] * dims[1]));

        /* Encode the centroid in cube-local [0,1] coords where 0 = cube
         * lower corner = MCX (V-0.5), 1 = cube upper corner = MCX (V+0.5).
         * This matches mcx_svmc.cu's split_voxel encoding so the photon
         * kernel reads SN and MC SVMC records on the same coordinate
         * frame (no half-voxel shift between the two preprocessors). */
        float cx = r.cx / r.w - (float)vx + 0.5f;
        float cy = r.cy / r.w - (float)vy + 0.5f;
        float cz = r.cz / r.w - (float)vz + 0.5f;

        r.cx = fminf(fmaxf(cx, 0.0f), 1.0f);
        r.cy = fminf(fmaxf(cy, 0.0f), 1.0f);
        r.cz = fminf(fmaxf(cz, 0.0f), 1.0f);

        float n_mag = sqrtf(r.nx * r.nx + r.ny * r.ny + r.nz * r.nz);

        if (n_mag > 1e-12f) {
            r.nx /= n_mag;
            r.ny /= n_mag;
            r.nz /= n_mag;
        }
    }
}


/* ---------------------------------------------------------------------------
 * GPU: finalize_svmc_kernel
 *
 * One thread per MCX voxel. If the SN rasterizer did not write here
 * (r.w == 0), emit a homogeneous-voxel record with byte 8 = own label.
 * Otherwise pack normalized centroid, normal, and label pair into the
 * 8-byte layout matching mcx_svmc.cu:
 *   [nz, ny, nx, cz, cy, cx, upper, lower]
 *
 * Encoding (matches MC's split_voxel):
 *   centroid byte = floor(c * 255)             c in [0,1]
 *   normal byte   = min(floor((n+1)*127.5), 254)  n in [-1,1]
 * ------------------------------------------------------------------------ */

__global__ void finalize_svmc_kernel(unsigned char* gvol, const SVMCRecord* records,
                                     const unsigned short* vol_labels, long vol_length,
                                     int dimx, int dimy, int dimz, int nMedia) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= vol_length) {
        return;
    }

    unsigned char*    voxel = &gvol[idx * 8];
    const SVMCRecord* r     = &records[idx];

    /* Homogeneous voxel: byte 8 = own label, all other bytes = 0. */
    if (r->w < 1e-12f) {
        unsigned short lab = vol_labels[idx];

        if (lab >= (unsigned short)nMedia) {
            lab = 0;
        }

        voxel[0] = 0;
        voxel[1] = 0;
        voxel[2] = 0;
        voxel[3] = 0;
        voxel[4] = 0;
        voxel[5] = 0;
        voxel[6] = 0;
        voxel[7] = (unsigned char)lab;
        return;
    }

    /* Boundary voxel: normalize, clamp, encode. */
    float nx = r->nx;
    float ny = r->ny;
    float nz = r->nz;

    /* Renormalize defensively (the CPU pass already unit-normalized; this is
     * cheap insurance against drift through the encode path). */
    float n_mag = sqrtf(nx * nx + ny * ny + nz * nz);

    if (n_mag > 1e-12f) {
        nx /= n_mag;
        ny /= n_mag;
        nz /= n_mag;
    } else {
        nx = 0.0f;
        ny = 0.0f;
        nz = 1.0f;
    }

    unsigned short lower = (unsigned short)(r->lower & 0xFFFF);
    unsigned short upper = (unsigned short)(r->upper & 0xFFFF);

    if (lower >= (unsigned short)nMedia) {
        lower = 0;
    }

    if (upper >= (unsigned short)nMedia) {
        upper = lower;
    }

    if (lower > upper) {
        unsigned short t = lower;
        lower = upper;
        upper = t;
    }

    float cx = fminf(fmaxf(r->cx, 0.0f), 1.0f);
    float cy = fminf(fmaxf(r->cy, 0.0f), 1.0f);
    float cz = fminf(fmaxf(r->cz, 0.0f), 1.0f);

    unsigned char ecx = (unsigned char)(cx * 255.0f);
    unsigned char ecy = (unsigned char)(cy * 255.0f);
    unsigned char ecz = (unsigned char)(cz * 255.0f);

    unsigned char enx = (unsigned char)fminf((nx + 1.0f) * 127.5f, 254.0f);
    unsigned char eny = (unsigned char)fminf((ny + 1.0f) * 127.5f, 254.0f);
    unsigned char enz = (unsigned char)fminf((nz + 1.0f) * 127.5f, 254.0f);

    voxel[0] = enz;
    voxel[1] = eny;
    voxel[2] = enx;
    voxel[3] = ecz;
    voxel[4] = ecy;
    voxel[5] = ecx;
    voxel[6] = (unsigned char)upper;
    voxel[7] = (unsigned char)lower;
}

/* ---------------------------------------------------------------------------
 * GPU: repack_kernel
 *
 * Contiguous 8-byte layout -> MCX interleaved format (identical to MC).
 *   Input  (per voxel): [nz, ny, nx, cz, cy, cx, upper, lower]
 *   Output first  half: [cy, cx, upper, lower]
 *   Output second half: [nz, ny, nx, cz]
 * ------------------------------------------------------------------------ */

__global__ void repack_kernel(unsigned int* newvol, const unsigned char* gvol,
                              long vol_length) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= vol_length) {
        return;
    }

    const unsigned char* src = &gvol[idx * 8];
    unsigned char*       dst = (unsigned char*)newvol;

    dst[idx * 4 + 0] = src[4];
    dst[idx * 4 + 1] = src[5];
    dst[idx * 4 + 2] = src[6];
    dst[idx * 4 + 3] = src[7];

    dst[(idx + vol_length) * 4 + 0] = src[0];
    dst[(idx + vol_length) * 4 + 1] = src[1];
    dst[(idx + vol_length) * 4 + 2] = src[2];
    dst[(idx + vol_length) * 4 + 3] = src[3];
}

/* ---------------------------------------------------------------------------
 * Debug dump: relaxed SN surface as an OFF mesh in MCX coordinates
 *
 * Each active SN cube emits its three "owned" back-edges. Every owned quad
 * with a non-trivial label transition is split into two triangles (ABC, ACD)
 * and written referencing the SN vertex indices.
 *
 * Vertex positions are stored after the padded->MCX conversion
 *   mcx = padded/voxelsize - 0.5
 * so the resulting OFF aligns 1:1 with the input voxel grid (lower-bottom
 * corner of input voxel 0 is at (0,0,0)).
 * ------------------------------------------------------------------------ */

static void dump_sn_surface_off(MMSurfaceNet* sn, float voxelsize[3],
                                const char* filename) {
    if (!sn) {
        return;
    }

    MMCellMap* cellmap = sn->cellMap();

    if (!cellmap) {
        return;
    }

    const MMCellFlag::Edge owned_edges[3] = {
        MMCellFlag::LeftBottomEdge,
        MMCellFlag::BackBottomEdge,
        MMCellFlag::LeftBackEdge
    };

    const int nv = cellmap->numVertices();

    /* First pass: collect triangles (3-tuples of SN vertex ids). */
    std::vector<int> tris;
    tris.reserve(nv * 18);   /* up to 3 quads * 2 tris * 3 vids per cube */

    for (int v = 0; v < nv; v++) {
        for (int e = 0; e < 3; e++) {
            int            quad_vids[4];
            unsigned short labels[2];

            if (!cellmap->getEdgeQuad(v, owned_edges[e], quad_vids, labels)) {
                continue;
            }

            if (labels[0] == labels[1] ||
                    labels[0] == 0xFFFF || labels[1] == 0xFFFF) {
                continue;
            }

            if (quad_vids[0] < 0 || quad_vids[1] < 0 ||
                    quad_vids[2] < 0 || quad_vids[3] < 0) {
                continue;
            }

            tris.push_back(quad_vids[0]);
            tris.push_back(quad_vids[1]);
            tris.push_back(quad_vids[2]);

            tris.push_back(quad_vids[0]);
            tris.push_back(quad_vids[2]);
            tris.push_back(quad_vids[3]);
        }
    }

    int n_tris = (int)(tris.size() / 3);

    FILE* fp = fopen(filename, "w");

    if (!fp) {
        return;
    }

    fprintf(fp, "OFF\n");
    fprintf(fp, "%d %d 0\n", nv, n_tris);

    /* Vertex coordinates: padded world -> MCX (subtract 0.5 per axis after
     * dividing by voxelsize). */
    const float inv_vs[3] = {1.0f / voxelsize[0], 1.0f / voxelsize[1], 1.0f / voxelsize[2]};

    for (int v = 0; v < nv; v++) {
        float pos_padded[3];
        sn->getVertexPosition(v, pos_padded);
        float mcx_x = pos_padded[0] * inv_vs[0] - 0.5f;
        float mcx_y = pos_padded[1] * inv_vs[1] - 0.5f;
        float mcx_z = pos_padded[2] * inv_vs[2] - 0.5f;
        fprintf(fp, "%.6f %.6f %.6f\n", mcx_x, mcx_y, mcx_z);
    }

    for (int t = 0; t < n_tris; t++) {
        fprintf(fp, "3 %d %d %d\n", tris[t * 3 + 0], tris[t * 3 + 1], tris[t * 3 + 2]);
    }

    fclose(fp);
}

/* ---------------------------------------------------------------------------
 * Debug dump: full SVMC volume as a 4D NIfTI-1 (.nii) volume
 *
 * Writes a uint8 (8, Nx, Ny, Nz) volume with the 8 SVMC bytes as the
 * fastest-varying (innermost) dimension so it lines up with mcx_svmc.cu's
 * pre-repack layout per voxel:
 *   bytes 0..7 = [nz, ny, nx, cz, cy, cx, upper, lower]
 *
 * A boundary voxel is one whose lower label (byte 7) differs from its upper
 * label (byte 6). Use this to confirm the rasterized shell is 1-voxel thick
 * and watertight in 3D Slicer / fslview / Octave.
 *
 * Output is a standard single-file NIfTI-1 (.nii) with sform = identity
 * scaled by voxelsize. No extensions, no header swapping.
 * ------------------------------------------------------------------------ */

static void dump_svmc_volume_nii(const unsigned char* gvol_bytes, int dims[3],
                                 float voxelsize[3], const char* filename) {
    nifti_1_header hdr;
    memset(&hdr, 0, sizeof(hdr));

    hdr.sizeof_hdr = 348;
    hdr.regular    = 'r';
    hdr.dim_info   = 0;

    /* 4D layout: innermost dim = 8 SVMC bytes per voxel. */
    hdr.dim[0] = 4;
    hdr.dim[1] = 8;
    hdr.dim[2] = (short)dims[0];
    hdr.dim[3] = (short)dims[1];
    hdr.dim[4] = (short)dims[2];
    hdr.dim[5] = 1;
    hdr.dim[6] = 1;
    hdr.dim[7] = 1;

    hdr.datatype = 2;     /* DT_UINT8 */
    hdr.bitpix   = 8;

    hdr.pixdim[0] = 1.0f;
    hdr.pixdim[1] = 1.0f;            /* component axis: dimensionless */
    hdr.pixdim[2] = voxelsize[0];
    hdr.pixdim[3] = voxelsize[1];
    hdr.pixdim[4] = voxelsize[2];
    hdr.pixdim[5] = 1.0f;
    hdr.pixdim[6] = 1.0f;
    hdr.pixdim[7] = 1.0f;

    hdr.vox_offset = 352.0f;
    hdr.xyzt_units = 2;   /* NIFTI_UNITS_MM */

    hdr.qform_code = 0;
    hdr.sform_code = 1;

    hdr.srow_x[0] = voxelsize[0];
    hdr.srow_y[1] = voxelsize[1];
    hdr.srow_z[2] = voxelsize[2];

    strncpy(hdr.descrip, "SN-SVMC volume (8 bytes/voxel: nz ny nx cz cy cx up lo)", 79);
    strncpy(hdr.magic, "n+1", 4);

    FILE* fp = fopen(filename, "wb");

    if (!fp) {
        return;
    }

    fwrite(&hdr, 1, 348, fp);

    /* 4-byte extension marker: no extensions. */
    unsigned char ext_zero[4] = {0, 0, 0, 0};
    fwrite(ext_zero, 1, 4, fp);

    /* gvol_bytes already has 8 bytes per voxel laid out contiguously, which
     * matches NIfTI's column-major (innermost-first) ordering with dim[1]=8. */
    long vol_length = (long)dims[0] * dims[1] * dims[2];
    fwrite(gvol_bytes, 1, vol_length * 8, fp);
    fclose(fp);
}
