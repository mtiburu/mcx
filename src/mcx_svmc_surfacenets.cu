// =============================================================================
// mcx_svmc_sn.cu  –  Surface Nets SVMC preprocessing
//
// Mirrors the Marching Cubes pipeline in mcx_svmc.cu exactly, replacing
// MC's per-label Gaussian-blur + triangle extraction with SN's vertex-centric
// approach:
//
//   MC pipeline (per label):
//     gaussian_blur → split_voxel kernel (MC lookup table, one thread/voxel)
//
//   SN pipeline (all labels at once):
//     build SN → relax → CPU vertex loop → finalize_svmc_kernel (one thread/voxel)
//
// Key insight: SN already gives us one surface vertex per active cell.
// That vertex's position IS the centroid (= vertexOffset), recoverable from
// the first quad corner since getEdgeQuadVtxIndices always places the current
// cell's vertex at quadVtxIndices[0] / corners[0..2].
//
// Storage convention (must match MC exactly):
//   MC stores the geometry for the cube at blockIdx=(bx,by,bz) at storage
//   index  idx1d = (bx+1) + (by+1)*dimx + (bz+1)*dimx*dimy,
//   i.e., at the UPPER corner of the cube.
//   The SN vertex at padded cell ci IS that upper corner (ci[i] = bx+i+1).
//   We therefore store each vertex's record at idx = ci[0] + ci[1]*dimx + ci[2]*dimx*dimy,
//   NOT at (ci[0]-1) + ... (lower corner), which would shift all geometry by
//   one voxel in every dimension and cause the simulation to read the geometry
//   for cube N when the photon is in cube N+1.
//
// Normal orientation convention (matches MC):
//   MC computes  isosurface_normal = -normal  (points toward lower-label material).
//   The analytical proof for all 12 edge types shows that the natural SN
//   cross product (B-A)x(C-A) always points from labels[0] toward labels[1].
//   Since labels are sorted so that labels[0] < labels[1] (lower < upper),
//   the natural normal points lower->upper.  Negating gives upper->lower =
//   toward-lower-label = MC convention.  No probe needed or used.
//
// Data flow:
//   1. Build MMSurfaceNet + relax
//   2. CPU vertex loop -> fills SVMCRecord[vol_length] (upper-corner indexed)
//   3. Upload SVMCRecord array to GPU
//   4. finalize_svmc_kernel: encode each record to 8-byte SVMC voxel
//   5. repack_kernel: 8-byte contiguous -> MCX interleaved format
// =============================================================================

#include "MMCellMap.cuh"
#include "MMSurfaceNet.cuh"
#include "MMCellFlag.cuh"
#include "mcx_svmc.h"
#include "mcx_tictoc.h"
#include "mcx_const.h"
#include "mcx_vector_math.cu"

#include <cstdio>
#include <cmath>
#include <algorithm>

// =============================================================================
// Data structures
// =============================================================================

/**
 * One record per voxel, populated on the CPU from SN vertex data.
 * Stored at the UPPER-CORNER index (ci[0], ci[1], ci[2]) to match MC.
 */
struct SVMCRecord {
    float    cx, cy, cz;   /**< centroid_local in [0,1] (= SN vertexOffset) */
    float    nx, ny, nz;   /**< area-weighted normal sum, MC convention (toward lower label) */
    float    w;            /**< total quad area; 0 means homogeneous voxel */
    uint32_t lower;        /**< lower material label (smaller index) */
    uint32_t upper;        /**< upper material label (larger index) */
};

// =============================================================================
// Forward declarations
// =============================================================================

__global__ void finalize_svmc_kernel(
    unsigned char*        gvol,
    const SVMCRecord*     records,
    const unsigned short* vol_labels,
    long   vol_length,
    int    dimx, int dimy, int dimz,
    float  vsx,  float vsy,  float vsz,
    int    nMedia);

__global__ void repack_kernel(
    unsigned int*        newvol,
    const unsigned char* gvol,
    long                 vol_length);

static void build_svmc_records(
    MMCellMap*            cellmap,
    const unsigned short* vol_labels,
    int                   dims[3],
    float                 voxelsize[3],
    SVMCRecord*           records);

// =============================================================================
// Main entry point
// =============================================================================

void mcx_svmc_preprocess_surfacenets(Config* cfg, GPUInfo* gpu) {
    if (cfg->mediabyte > 4 || !cfg->issvmc) return;

    MCX_FPRINTF(cfg->flog, "Surface Nets SVMC preprocessing...\n");
    unsigned int tic = StartTimer();

    int   dims[3]      = {(int)cfg->dim.x, (int)cfg->dim.y, (int)cfg->dim.z};
    float voxelsize[3] = {cfg->unitinmm,   cfg->unitinmm,   cfg->unitinmm};
    long  vol_length   = (long)dims[0] * dims[1] * dims[2];

    // ----- 1. Prepare host label volume -----
    unsigned short* h_vol_labels = new unsigned short[vol_length];
    for (long i = 0; i < vol_length; i++) {
        unsigned short lbl = (unsigned short)(cfg->vol[i] & MED_MASK);
        h_vol_labels[i]    = (lbl == 0xFFFF) ? 0 : lbl;
    }

    // ----- 2. Build and relax Surface Net -----
    MCX_FPRINTF(cfg->flog, "[SN] Building mesh...\n");
    MMSurfaceNet sn(h_vol_labels, dims, voxelsize);

    MCX_FPRINTF(cfg->flog, "[SN] Relaxing mesh...\n");
    // maxDistFromCenter MUST stay < 0.5 so vertices remain inside their own cell.
    // Values >= 0.5 allow drift into a neighbouring cell, placing the centroid
    // outside [0,1] in local coordinates (clamped to the cell face = wrong geometry).
    // 0.45 gives a small safety margin.
    MMSurfaceNet::RelaxAttrs relaxAttrs{10, 0.5f, 0.45f};
    sn.relax(relaxAttrs);

    // ----- 3. CPU vertex loop: build SVMCRecord array -----
    MCX_FPRINTF(cfg->flog, "[SN] Building SVMC records from vertices...\n");
    SVMCRecord* h_records = new SVMCRecord[vol_length]();   // zero-initialised

    build_svmc_records(sn.cellMap(), h_vol_labels, dims, voxelsize, h_records);

    // Count active voxels for logging
    long active = 0;
    for (long i = 0; i < vol_length; i++)
        if (h_records[i].w > 0.0f) active++;
    MCX_FPRINTF(cfg->flog, "[SN] Active surface voxels: %ld / %ld\n", active, vol_length);

    // ----- Debug: comparison at known MC boundary voxels -----
    {
        const int check_voxels[][3] = {
            {28,26,15}, {29,26,15}, {30,26,15}, {31,26,15}, {32,26,15}
        };
        MCX_FPRINTF(cfg->flog, "[SN] Record check at MC-known boundary voxels:\n");
        for (int k = 0; k < 5; k++) {
            int x = check_voxels[k][0];
            int y = check_voxels[k][1];
            int z = check_voxels[k][2];
            if (x>=dims[0]||y>=dims[1]||z>=dims[2]) continue;
            long i = (long)x + (long)y*dims[0] + (long)z*dims[0]*dims[1];
            const SVMCRecord& r = h_records[i];
            unsigned short own = h_vol_labels[i];
            if (r.w > 0.f) {
                float len = sqrtf(r.nx*r.nx + r.ny*r.ny + r.nz*r.nz);
                float nnx = len>0 ? r.nx/len : 0.f;
                float nny = len>0 ? r.ny/len : 0.f;
                float nnz = len>0 ? r.nz/len : 0.f;
                MCX_FPRINTF(cfg->flog,
                    "  SN vox(%d,%d,%d): own=%u lo=%u hi=%u "
                    "c=(%.3f,%.3f,%.3f) n=(%.3f,%.3f,%.3f)\n",
                    x, y, z, own, r.lower, r.upper,
                    r.cx, r.cy, r.cz, nnx, nny, nnz);
            } else {
                MCX_FPRINTF(cfg->flog,
                    "  SN vox(%d,%d,%d): own=%u  NO RECORD\n", x, y, z, own);
            }
        }
    }

    // ----- Dump centroids + normals for MATLAB visualization -----
    // Format: vol_length x 6 float32 values per voxel: [wx, wy, wz, nx, ny, nz]
    // wx/wy/wz are world-space centroid positions in voxel units.
    {
        FILE* fv = fopen("dump_vec_sn.bin", "wb");
        if (fv) {
            for (long i = 0; i < vol_length; i++) {
                const SVMCRecord& r = h_records[i];
                float row[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                if (r.w > 1e-12f) {
                    // Upper-corner idx: vx=ci[0]. Cube lower corner is (vx-1).
                    int vx = (int)(i % dims[0]);
                    int vy = (int)((i / dims[0]) % dims[1]);
                    int vz = (int)(i / ((long)dims[0] * dims[1]));
                    row[0] = (float)(vx - 1) + r.cx;
                    row[1] = (float)(vy - 1) + r.cy;
                    row[2] = (float)(vz - 1) + r.cz;
                    float len = sqrtf(r.nx*r.nx + r.ny*r.ny + r.nz*r.nz);
                    if (len > 1e-12f) {
                        row[3] = r.nx / len;
                        row[4] = r.ny / len;
                        row[5] = r.nz / len;
                    }
                }
                fwrite(row, sizeof(float), 6, fv);
            }
            fclose(fv);
            MCX_FPRINTF(cfg->flog, "[SN] Wrote centroids+normals -> dump_vec_sn.bin\n");
        }
    }

    // ----- 4. GPU upload and kernel launch -----
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

    MCX_FPRINTF(cfg->flog, "[SN] Finalizing SVMC format...\n");
    finalize_svmc_kernel<<<voxel_blocks, threads>>>(
        d_gvol, d_records, d_vol_labels, vol_length,
        dims[0], dims[1], dims[2],
        voxelsize[0], voxelsize[1], voxelsize[2],
        cfg->medianum);
    cudaDeviceSynchronize();

    MCX_FPRINTF(cfg->flog, "[SN] Repacking to MCX interleaved format...\n");
    repack_kernel<<<voxel_blocks, threads>>>(d_newvol, d_gvol, vol_length);
    cudaDeviceSynchronize();

    // ----- 5. Copy result back to host -----
    unsigned int* h_newvol = (unsigned int*)malloc(vol_length * 2 * sizeof(unsigned int));
    cudaMemcpy(h_newvol, d_newvol, vol_length * 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    MCX_FPRINTF(cfg->flog, "Surface Nets complete: %d ms\n", GetTimeMillis() - tic);

    FILE* fp = fopen("dump_svmc_sn.bin", "wb");
    if (fp) {
        fwrite(h_newvol, sizeof(unsigned int), vol_length * 2, fp);
        fclose(fp);
        MCX_FPRINTF(cfg->flog, "[SN] Wrote packed volume -> dump_svmc_sn.bin\n");
    }

    // ----- 6. Cleanup -----
    cudaFree(d_records);
    cudaFree(d_vol_labels);
    cudaFree(d_gvol);
    cudaFree(d_newvol);
    delete[] h_records;
    delete[] h_vol_labels;
    free(cfg->vol);

    cfg->vol       = h_newvol;
    cfg->mediabyte = MEDIA_2LABEL_SPLIT;

    // Post-processing: source position fixup + detector mask (identical to MC)
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
                if (sx<0.f||sy<0.f||sz<0.f||
                        sx>=cfg->dim.x||sy>=cfg->dim.y||sz>=cfg->dim.z) {
                    *((uint*)&cfg->srcdata[i].srcparam2.z) = 0;
                    *((uint*)&cfg->srcdata[i].srcparam2.w) = 0;
                } else {
                    uint idx1dorig =
                        ((int)floorf(sz))*(cfg->dim.y*cfg->dim.x) +
                        ((int)floorf(sy))*cfg->dim.x +
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


// =============================================================================
// CPU: build_svmc_records
//
// One pass over all SN vertices. Each vertex lives in padded cell ci and
// carries the surface geometry for that cube. We store its record at the
// UPPER-CORNER index  idx = ci[0] + ci[1]*dims[0] + ci[2]*dims[0]*dims[1]
// to match MC's storage convention.
//
// Centroid:
//   corners[0..2] from getEdgeQuadPositions is always the current vertex's
//   world position in padded space (getEdgeQuadVtxIndices always puts the
//   current cell's vertex at index 0).
//   world_pos[d] = voxelsize[d] * (ci[d] + vertexOffset[d])
//   centroid_local[d] = world_pos[d]/voxelsize[d] - ci[d]
//                     = vertexOffset[d]   in [0, 1]
//   Equivalently: corners[d]/voxelsize[d] - ci[d]
//   (The -1 padding offset implicit in corners[] cancels with ci[d]-1+1.)
//
// Normal:
//   Analytical proof for all 12 edge types shows that the natural cross
//   product (B-A)x(C-A) always points from the lower-indexed label toward
//   the higher-indexed label. After sorting qlo < qhi the natural normal
//   points lower->upper. We negate to match MC's convention (toward lower).
//   No probe is used: probes are structurally unreliable when the vertex sits
//   near the center of its cell (offset ~0.5) because no single probe offset
//   can both cross the near boundary and stay within the correct far cell.
//
// Storage filter: NONE.
//   Every active vertex writes regardless of what label occupies the
//   upper-corner cell. MC likewise writes to every cube that has a surface
//   crossing, regardless of the cube's own label.
// =============================================================================

static void build_svmc_records(
    MMCellMap*            cellmap,
    const unsigned short* vol_labels,
    int                   dims[3],
    float                 voxelsize[3],
    SVMCRecord*           records)   // pre-zeroed, dims[0]*dims[1]*dims[2] entries
{
    if (!cellmap) return;

    const MMCellFlag::Edge all_edges[12] = {
        MMCellFlag::LeftBottomEdge,  MMCellFlag::RightBottomEdge,
        MMCellFlag::BackBottomEdge,  MMCellFlag::FrontBottomEdge,
        MMCellFlag::LeftTopEdge,     MMCellFlag::RightTopEdge,
        MMCellFlag::BackTopEdge,     MMCellFlag::FrontTopEdge,
        MMCellFlag::LeftBackEdge,    MMCellFlag::RightBackEdge,
        MMCellFlag::LeftFrontEdge,   MMCellFlag::RightFrontEdge
    };

    const int nv = cellmap->numVertices();

    for (int v = 0; v < nv; v++) {
        int ci[3];
        cellmap->getVertexCellIndex(v, ci);

        // Interior range: ci[i] in [1, dims[i]-1] gives storage indices in
        // [1, dims[i]-1] matching MC's range of (bx+1) in [1, dimx-1].
        // ci[i]==0 is a padding cell. ci[i]==dims[i] would be out-of-bounds.
        if (ci[0] < 1 || ci[0] >= dims[0] ||
            ci[1] < 1 || ci[1] >= dims[1] ||
            ci[2] < 1 || ci[2] >= dims[2]) continue;

        // Upper-corner storage index -- matches MC exactly.
        long idx = (long)ci[0] +
                   (long)ci[1] * dims[0] +
                   (long)ci[2] * dims[0] * dims[1];

        float total_nx = 0.f, total_ny = 0.f, total_nz = 0.f;
        float total_w  = 0.f;
        float vcx = 0.f, vcy = 0.f, vcz = 0.f;
        bool  got_position = false;
        uint32_t lo_best   = 0xFFFFFFFFu;
        uint32_t hi_best   = 0;
        float    best_area = -1.f;

        for (int e = 0; e < 12; e++) {
            float          corners[12];
            unsigned short labels[2];
            if (!cellmap->getEdgeQuad(v, all_edges[e], corners, labels))
                continue;

            // Skip padding-label or same-material quads
            if (labels[0] == labels[1] ||
                labels[0] == 0xFFFF   ||
                labels[1] == 0xFFFF) continue;

            // Centroid from first valid quad.
            // corners[0..2] = world_pos = voxelsize*(ci + vertexOffset)
            // centroid_local = corners/voxelsize - ci = vertexOffset in [0,1]
            if (!got_position) {
                vcx = corners[0] / voxelsize[0] - (float)ci[0];
                vcy = corners[1] / voxelsize[1] - (float)ci[1];
                vcz = corners[2] / voxelsize[2] - (float)ci[2];
                got_position = true;
            }

            // Split quad ABCD -> triangles ABC + ACD
            float ax=corners[0], ay=corners[1], az=corners[2];
            float bx=corners[3], by=corners[4], bz=corners[5];
            float ccx=corners[6],ccy=corners[7],ccz=corners[8];
            float dx=corners[9], dy=corners[10],dz=corners[11];

            float n1x=(by-ay)*(ccz-az)-(bz-az)*(ccy-ay);
            float n1y=(bz-az)*(ccx-ax)-(bx-ax)*(ccz-az);
            float n1z=(bx-ax)*(ccy-ay)-(by-ay)*(ccx-ax);

            float n2x=(ccy-ay)*(dz-az)-(ccz-az)*(dy-ay);
            float n2y=(ccz-az)*(dx-ax)-(ccx-ax)*(dz-az);
            float n2z=(ccx-ax)*(dy-ay)-(ccy-ay)*(dx-ax);

            float a1   = 0.5f * sqrtf(n1x*n1x + n1y*n1y + n1z*n1z);
            float a2   = 0.5f * sqrtf(n2x*n2x + n2y*n2y + n2z*n2z);
            float area = a1 + a2;
            if (area < 1e-12f) continue;

            // Unit normal from combined cross products (area-weighted by magnitude)
            float snx=n1x+n2x, sny=n1y+n2y, snz=n1z+n2z;
            float slen = sqrtf(snx*snx + sny*sny + snz*snz);
            if (slen < 1e-12f) continue;
            snx /= slen; sny /= slen; snz /= slen;

            total_nx += snx * area;
            total_ny += sny * area;
            total_nz += snz * area;
            total_w  += area;

            // Track dominant label pair by largest quad area
            uint32_t qlo = (labels[0] < labels[1]) ? labels[0] : labels[1];
            uint32_t qhi = (labels[0] < labels[1]) ? labels[1] : labels[0];
            if (area > best_area) {
                best_area = area;
                lo_best   = qlo;
                hi_best   = qhi;
            }
        }

        if (total_w < 1e-12f || !got_position) continue;

        // Negate to match MC convention: normal points toward lower-label material.
        // The natural cross product points lower->upper (proven analytically for
        // all 12 edge types). Negating gives upper->lower = toward-lower = MC.
        total_nx = -total_nx;
        total_ny = -total_ny;
        total_nz = -total_nz;

        SVMCRecord& r = records[idx];
        r.cx    = vcx;
        r.cy    = vcy;
        r.cz    = vcz;
        r.nx    = total_nx;
        r.ny    = total_ny;
        r.nz    = total_nz;
        r.w     = total_w;
        r.lower = lo_best;
        r.upper = hi_best;
    }
}


// =============================================================================
// GPU Kernel 1: finalize_svmc_kernel
//
// One thread per voxel. Reads SVMCRecord, normalises the normal, encodes
// centroid and normal to bytes, packs into 8-byte contiguous SVMC layout:
//   byte[0]=nz  [1]=ny  [2]=nx  [3]=cz  [4]=cy  [5]=cx  [6]=upper  [7]=lower
//
// Storage convention reminder:
//   idx decomposes as vx=ci[0], vy=ci[1], vz=ci[2] (upper corner).
//   The cube spans (vx-1, vy-1, vz-1) to (vx, vy, vz) in original 0-indexed.
//   r->cx/cy/cz are vertexOffset in [0,1], already the correct local coords.
//
// No probe for normal orientation -- build_svmc_records negated analytically.
// =============================================================================

__global__ void finalize_svmc_kernel(
    unsigned char*        gvol,
    const SVMCRecord*     records,
    const unsigned short* vol_labels,
    long   vol_length,
    int    dimx, int dimy, int dimz,
    float  vsx,  float vsy,  float vsz,
    int    nMedia)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vol_length) return;

    unsigned char*    voxel = &gvol[idx * 8];
    const SVMCRecord* r     = &records[idx];

    // ---- Homogeneous voxel ----
    if (r->w < 1e-12f) {
        unsigned short lab = vol_labels[idx];
        if (lab >= (unsigned short)nMedia) lab = 0;
        voxel[0]=0; voxel[1]=0; voxel[2]=0; voxel[3]=0;
        voxel[4]=0; voxel[5]=0; voxel[6]=0;
        voxel[7] = (unsigned char)lab;
        return;
    }

    // ---- Normalise surface normal ----
    float len = sqrtf(r->nx*r->nx + r->ny*r->ny + r->nz*r->nz);
    float nx, ny, nz;
    if (len > 1e-12f) { nx=r->nx/len; ny=r->ny/len; nz=r->nz/len; }
    else              { nx=0.f;       ny=0.f;       nz=1.f;       }

    // ---- Labels (sorted lower < upper from build_svmc_records) ----
    unsigned short lower = (unsigned short)(r->lower & 0xFFFF);
    unsigned short upper = (unsigned short)(r->upper & 0xFFFF);
    if (lower >= (unsigned short)nMedia) lower = 0;
    if (upper >= (unsigned short)nMedia) upper = lower;
    if (lower > upper) { unsigned short t=lower; lower=upper; upper=t; }

    // ---- Centroid: vertexOffset already in [0,1], just clamp ----
    float cx = fminf(fmaxf(r->cx, 0.f), 1.f);
    float cy = fminf(fmaxf(r->cy, 0.f), 1.f);
    float cz = fminf(fmaxf(r->cz, 0.f), 1.f);

    // ---- Encode to bytes (same encoding as MC split_voxel) ----
    unsigned char ecx = (unsigned char)(cx * 255.f);
    unsigned char ecy = (unsigned char)(cy * 255.f);
    unsigned char ecz = (unsigned char)(cz * 255.f);
    // Normal: [-1,1] -> [0,254]  (max 254 matches MC's min(...,254) clamp)
    unsigned char enx = (unsigned char)fminf((nx + 1.f) * 127.5f, 254.f);
    unsigned char eny = (unsigned char)fminf((ny + 1.f) * 127.5f, 254.f);
    unsigned char enz = (unsigned char)fminf((nz + 1.f) * 127.5f, 254.f);

    // SVMC 8-byte layout: [nz, ny, nx, cz, cy, cx, upper, lower]
    voxel[0] = enz;
    voxel[1] = eny;
    voxel[2] = enx;
    voxel[3] = ecz;
    voxel[4] = ecy;
    voxel[5] = ecx;
    voxel[6] = (unsigned char)upper;
    voxel[7] = (unsigned char)lower;
}


// =============================================================================
// GPU Kernel 2: repack_kernel
//
// Contiguous 8-byte layout -> MCX interleaved format (identical to MC output).
//
// Input  (per voxel): [nz, ny, nx, cz, cy, cx, upper, lower]
// Output first  half: [cy, cx, upper, lower]
// Output second half: [nz, ny, nx, cz]
// =============================================================================

__global__ void repack_kernel(
    unsigned int*        newvol,
    const unsigned char* gvol,
    long                 vol_length)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vol_length) return;

    const unsigned char* src = &gvol[idx * 8];
    unsigned char*       dst = (unsigned char*)newvol;

    dst[idx * 4 + 0] = src[4];  // cy
    dst[idx * 4 + 1] = src[5];  // cx
    dst[idx * 4 + 2] = src[6];  // upper
    dst[idx * 4 + 3] = src[7];  // lower

    dst[(idx + vol_length) * 4 + 0] = src[0];  // nz
    dst[(idx + vol_length) * 4 + 1] = src[1];  // ny
    dst[(idx + vol_length) * 4 + 2] = src[2];  // nx
    dst[(idx + vol_length) * 4 + 3] = src[3];  // cz
}
