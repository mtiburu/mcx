#include "MMCellMap.cuh"
#include "MMSurfaceNet.cuh"
#include "MMCellFlag.cuh"
#include "mcx_svmc.h"
#include "mcx_tictoc.h"
#include "mcx_const.h"

#include <vector>
#include <cstdio>
#include <set>
#include <map>
#include <cmath>

// ==========================================================
// Data Structures 
// ==========================================================

// Quad with ownership tracking - knows which cube edge it came from
struct SNQuad {
    float corners[12];      // 4 vertices × 3 coords
    unsigned short labels[2]; // [lower, upper] material labels
    int voxel1[3];     // Voxel with lower label
    int voxel2[3];   // Voxel with upper label
    int edge_axis;          // 0=X, 1=Y, 2=Z - axis of the edge

    int owner[3];   // in ORIGINAL (unpadded) voxel coordinates
};

// Per-voxel accumulator 
struct VoxelAccum {
    float cx, cy, cz;       // Weighted centroid sum
    float nx, ny, nz;       // Weighted normal sum  
    float total_weight;     // Total area weight
    unsigned int lower;   // Min label seen
    unsigned int upper;   // Max label seen
    int quad_count;         // Number of contributing quads
};

enum SNDStage {
    SNDBG_BEFORE_RELAX,
    SNDBG_AFTER_RELAX,
    SNDBG_AFTER_ASSIGN,
    SNDBG_AFTER_FINALIZE
};

// Gate heavy debug with a macro so normal runs stay clean.
#ifndef MCX_SN_DEBUG
#define MCX_SN_DEBUG 0
#endif

static void debug_function(
    Config* cfg,
    SNDStage stage,
    const std::vector<SNQuad>* quads,
    const unsigned short* vol_labels,
    const int dims[3],
    const float voxelsize[3],
    const VoxelAccum* h_accums,   // optional (only after assign)
    long vol_length,
    const unsigned int* packed2u32 // optional (only after finalize)
);

__device__ __forceinline__ float3 compute_quad_centroid_proper(
    float3 A, float3 B, float3 C, float3 D,
    float3 n1, float3 n2  // Pre-computed triangle normals
) {
    // Triangle ABC centroid
    float3 c1 = make_float3(
        (A.x + B.x + C.x) / 3.0f,
        (A.y + B.y + C.y) / 3.0f,
        (A.z + B.z + C.z) / 3.0f
    );
    
    // Triangle ACD centroid
    float3 c2 = make_float3(
        (A.x + C.x + D.x) / 3.0f,
        (A.y + C.y + D.y) / 3.0f,
        (A.z + C.z + D.z) / 3.0f
    );
    
    // Triangle areas
    float a1 = sqrtf(n1.x*n1.x + n1.y*n1.y + n1.z*n1.z);
    float a2 = sqrtf(n2.x*n2.x + n2.y*n2.y + n2.z*n2.z);
    
    float total = a1 + a2;
    if (total < 1e-12f) {
        return make_float3(
            (A.x + B.x + C.x + D.x) * 0.25f,
            (A.y + B.y + C.y + D.y) * 0.25f,
            (A.z + B.z + C.z + D.z) * 0.25f
        );
    }
    
    return make_float3(
        (c1.x * a1 + c2.x * a2) / total,
        (c1.y * a1 + c2.y * a2) / total,
        (c1.z * a1 + c2.z * a2) / total
    );
}
__device__ __forceinline__ int voxel_idx_3d_to_1d(int x, int y, int z, int dimx, int dimy);
__device__ __forceinline__ float3 compute_triangle_normal(float3 A, float3 B, float3 C);
__device__ __forceinline__ float triangle_area(float3 n);
__device__ __forceinline__ float3 normalize_vec(float3 v);
__device__ bool triangle_aabb_intersect( float3 v0, float3 v1, float3 v2, float3 box_min, float3 box_max);
__global__ void assign_quads_to_voxels_kernel( const SNQuad* quads, int num_quads, VoxelAccum* accums,
    const unsigned short* vol_labels,int dimx, int dimy, int dimz, float vsx, float vsy, float vsz);
__global__ void finalize_svmc_kernel( unsigned char* gvol, const VoxelAccum* accums, const unsigned short* vol_labels,
    long vol_length, int dimx, int dimy, int dimz, float vsx, float vsy, float vsz, int nMedia);
std::vector<SNQuad> extract_quads_with_tracking( MMCellMap* cellmap, const unsigned short* vol_labels, int dims[3], float voxelsize[3]);
void analyze_quads(const std::vector<SNQuad>& quads);
void analyze_voxel_assignment(const std::vector<SNQuad>& quads,  const unsigned short* vol_labels, int dims[3]);


void dump_quads_obj(const std::vector<SNQuad>& quads, const char* filename);
// ==========================================================
// Main Processing Function
// ==========================================================

void mcx_svmc_preprocess_surfacenets(Config* cfg, GPUInfo* gpu) {
    if (cfg->mediabyte > 4 || !cfg->issvmc) return;
    
    MCX_FPRINTF(cfg->flog, "Surface Nets SVMC preprocessing...\n");
    unsigned int tic = StartTimer();
    
    int dims[3] = {(int)cfg->dim.x, (int)cfg->dim.y, (int)cfg->dim.z};
    float voxelsize[3] = {cfg->unitinmm, cfg->unitinmm, cfg->unitinmm};
    long vol_length = (long)dims[0] * dims[1] * dims[2];
    
    // Prepare label volume
    unsigned short* h_vol_labels = new unsigned short[vol_length];
    for (long i = 0; i < vol_length; i++) {
        unsigned short lbl = (unsigned short)(cfg->vol[i] & MED_MASK);
        h_vol_labels[i] = (lbl == 0xFFFF) ? 0 : lbl;
    }
    
    // Build and relax Surface Net
    MCX_FPRINTF(cfg->flog, "[SN] Building mesh...\n");
    MMSurfaceNet sn(h_vol_labels, dims, voxelsize);
    // Extract and dump BEFORE relax
    std::vector<SNQuad> quads_before = extract_quads_with_tracking(sn.cellMap(), h_vol_labels, dims, voxelsize);
    dump_quads_obj(quads_before, "dump_BEFORE_relax.obj");

    debug_function(cfg, SNDBG_BEFORE_RELAX, &quads_before, h_vol_labels, dims, voxelsize, nullptr, vol_length, nullptr);
        
    MCX_FPRINTF(cfg->flog, "[SN] Relaxing mesh...\n");
/*
 Surface relaxation parameters for Surface Nets smoothing.

 numRelaxIterations
   Number of relaxation passes applied to surface vertices.
   Higher values increase smoothness but also increase runtime
   and amplify drift toward the center of the mesh.
   Typical range: 10–40 for tuning, 60–120 for final output.

 relaxFactor
   Per-iteration movement scale applied to each vertex.
   Range is strictly between 0.0 and 1.0.
   Larger values move vertices faster but increase the risk
   of instability, folding, and corner collapse.
   This controls how aggressive each relaxation step is.

 maxDistFromCellCenter
   Hard clamp on how far a vertex is allowed to move from the
   center of its original surface cube, measured in voxel units.
   This constraint prevents surface shrinkage, self-intersection,
   and loss of label fidelity near boundaries.
   Values around 1.0 keep vertices tightly bound,
   values above 1.5 allow stronger smoothing.
 */

    MMSurfaceNet::RelaxAttrs relaxAttrs{10, 0.5f, 1.0f}; 
    sn.relax(relaxAttrs);

    // Extract and dump AFTER relax
    std::vector<SNQuad> quads = extract_quads_with_tracking(sn.cellMap(), h_vol_labels, dims, voxelsize);
    dump_quads_obj(quads, "dump_AFTER_relax.obj");

    debug_function(cfg, SNDBG_AFTER_RELAX, &quads, h_vol_labels, dims, voxelsize, nullptr, vol_length, nullptr);

    MCX_FPRINTF(cfg->flog, "[SN Debug] voxelsize=[%f,%f,%f]\n", voxelsize[0], voxelsize[1], voxelsize[2]);

    // ADD THESE TWO LINES:
    analyze_quads(quads);
    analyze_voxel_assignment(quads, h_vol_labels, dims);
    
    // // Extract quads with tracking
    // MCX_FPRINTF(cfg->flog, "[SN] Extracting quads...\n");
    // std::vector<SNQuad> quads = extract_quads_with_tracking(sn.cellMap(), h_vol_labels, dims);
    // MCX_FPRINTF(cfg->flog, "[SN] Extracted %zu quads\n", quads.size());
    // Count label pairs in quads
int pair_01 = 0, pair_02 = 0, pair_12 = 0, other = 0;
for (const auto& q : quads) {
    if (q.labels[0] == 0 && q.labels[1] == 1) pair_01++;
    else if (q.labels[0] == 0 && q.labels[1] == 2) pair_02++;
    else if (q.labels[0] == 1 && q.labels[1] == 2) pair_12++;
    else other++;
}
MCX_FPRINTF(cfg->flog, "[SN Debug] Quad label pairs: 0-1=%d, 0-2=%d, 1-2=%d, other=%d\n",
    pair_01, pair_02, pair_12, other);

// Count labels in volume
int label_counts[10] = {0};
for (long i = 0; i < vol_length; i++) {
    if (h_vol_labels[i] < 10) label_counts[h_vol_labels[i]]++;
}
MCX_FPRINTF(cfg->flog, "[SN Debug] Volume labels: ");
for (int l = 0; l < 10; l++) {
    if (label_counts[l] > 0) {
        MCX_FPRINTF(cfg->flog, "L%d=%d ", l, label_counts[l]);
    }
}
MCX_FPRINTF(cfg->flog, "\n");

    MCX_FPRINTF(cfg->flog, "[SN Debug] First 5 quads:\n");
    for (int i = 0; i < min(5, (int)quads.size()); i++) {
        MCX_FPRINTF(cfg->flog, "  Quad %d: labels=[%d,%d] voxel1=[%d,%d,%d] voxel2=[%d,%d,%d]\n",
            i, quads[i].labels[0], quads[i].labels[1],
            quads[i].voxel1[0], quads[i].voxel1[1], quads[i].voxel1[2],
            quads[i].voxel2[0], quads[i].voxel2[1], quads[i].voxel2[2]);
    }
    
    if (quads.empty()) {
        MCX_FPRINTF(stderr, "[SN Error] No quads extracted!\n");
        delete[] h_vol_labels;
        return;
    }
    
    // Allocate device memory
    SNQuad* d_quads;
    unsigned short* d_vol_labels;
    VoxelAccum* d_accums;
    unsigned char* d_gvol;
    
    cudaMalloc(&d_quads, quads.size() * sizeof(SNQuad));
    cudaMalloc(&d_vol_labels, vol_length * sizeof(unsigned short));
    cudaMalloc(&d_accums, vol_length * sizeof(VoxelAccum));
    cudaMalloc(&d_gvol, vol_length * 2 * sizeof(unsigned int)); 
    
    // Initialize
    cudaMemcpy(d_quads, quads.data(), quads.size() * sizeof(SNQuad), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vol_labels, h_vol_labels, vol_length * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaMemset(d_accums, 0, vol_length * sizeof(VoxelAccum));
    
    // Initialize accums with sentinel values for min/max
    std::vector<VoxelAccum> init_accums(vol_length);
    for (long i = 0; i < vol_length; i++) {
        init_accums[i].cx = 0.0f;
        init_accums[i].cy = 0.0f;
        init_accums[i].cz = 0.0f;
        init_accums[i].nx = 0.0f;
        init_accums[i].ny = 0.0f;
        init_accums[i].nz = 0.0f;
        init_accums[i].total_weight = 0.0f;
        init_accums[i].lower = 0xFFFF;
        init_accums[i].upper = 0;
        init_accums[i].quad_count = 0;
    }
    cudaMemcpy(d_accums, init_accums.data(), vol_length * sizeof(VoxelAccum), cudaMemcpyHostToDevice);
    
    // Run kernels
    int threads = 256;
    int quad_blocks = ((int)quads.size() + threads - 1) / threads;
    int voxel_blocks = (vol_length + threads - 1) / threads;
    
    MCX_FPRINTF(cfg->flog, "[SN] Assigning quads to voxels...\n");
    assign_quads_to_voxels_kernel<<<quad_blocks, threads>>>( d_quads, (int)quads.size(), d_accums, d_vol_labels,dims[0], dims[1], dims[2], voxelsize[0], voxelsize[1], voxelsize[2]);
    cudaDeviceSynchronize();

    

    // Debug: count voxel types
    VoxelAccum* h_accums = new VoxelAccum[vol_length];
    cudaMemcpy(h_accums, d_accums, vol_length * sizeof(VoxelAccum), cudaMemcpyDeviceToHost);

    debug_function(cfg, SNDBG_AFTER_ASSIGN, nullptr, h_vol_labels, dims, voxelsize, h_accums, vol_length, nullptr);


    int surface_count = 0, homogeneous_count = 0;
    for (long i = 0; i < vol_length; i++) {
        if (h_accums[i].total_weight < 1e-12f || h_accums[i].quad_count == 0) {
            homogeneous_count++;
        } else {
            surface_count++;
        }
    }
    MCX_FPRINTF(cfg->flog, "[SN Debug] Surface voxels: %d, Homogeneous: %d (total: %ld)\n", surface_count, homogeneous_count, vol_length);

// Check for different label pairs in assigned voxels
int pair_counts[5] = {0, 0, 0, 0, 0}; // 0-1, 0-2, 0-3, 1-2, other
for (long i = 0; i < vol_length; i++) {
    if (h_accums[i].quad_count > 0) {
        int lo = h_accums[i].lower;
        int hi = h_accums[i].upper;
        if (lo == 0 && hi == 1) pair_counts[0]++;
        else if (lo == 0 && hi == 2) pair_counts[1]++;
        else if (lo == 0 && hi == 3) pair_counts[2]++;
        else if (lo == 1 && hi == 2) pair_counts[3]++;
        else pair_counts[4]++;
    }
}
MCX_FPRINTF(cfg->flog, "[SN Debug] Assigned voxel pairs: 0-1=%d, 0-2=%d, 0-3=%d, 1-2=%d, other=%d\n",
    pair_counts[0], pair_counts[1], pair_counts[2], pair_counts[3], pair_counts[4]);

    // Also check first few surface voxels' upper values
    int shows = 0;
    for (long i = 0; i < vol_length && shows < 5; i++) {
        if (h_accums[i].quad_count > 0) {
            MCX_FPRINTF(cfg->flog, "[SN Debug] Surface voxel %ld: lower=%d upper=%d weight=%.3f\n",
                        i, h_accums[i].lower, h_accums[i].upper, h_accums[i].total_weight);
            shows++;
        }
    }

    // ========== ADD NEW DIAGNOSTICS HERE ==========
    
    // How many quads assigned per voxel?
    int voxels_with_1_quad = 0;
    int voxels_with_2_quads = 0;
    int voxels_with_3plus_quads = 0;

    for (long i = 0; i < vol_length; i++) {
        if (h_accums[i].quad_count == 1) voxels_with_1_quad++;
        else if (h_accums[i].quad_count == 2) voxels_with_2_quads++;
        else if (h_accums[i].quad_count >= 3) voxels_with_3plus_quads++;
    }

    MCX_FPRINTF(cfg->flog, "[SN Debug] Voxels by quad count: 1=%d, 2=%d, 3+=%d\n", voxels_with_1_quad, voxels_with_2_quads, voxels_with_3plus_quads);

    // Check if centroids are within expected voxel bounds
    int out_of_bounds = 0;
    for (long i = 0; i < vol_length; i++) {
        if (h_accums[i].quad_count > 0) {
            float inv_w = 1.0f / h_accums[i].total_weight;
            float cx = h_accums[i].cx * inv_w;
            float cy = h_accums[i].cy * inv_w;
            float cz = h_accums[i].cz * inv_w;
            
            int vx = i % dims[0];
            int vy = (i / dims[0]) % dims[1];
            int vz = i / (dims[0] * dims[1]);
            
            // Check if centroid is near this voxel
            if (cx < vx - 1 || cx > vx + 2 || 
                cy < vy - 1 || cy > vy + 2 || 
                cz < vz - 1 || cz > vz + 2) {
                out_of_bounds++;
            }
        }
    }
    MCX_FPRINTF(cfg->flog, "[SN Debug] Centroids far from assigned voxel: %d\n", out_of_bounds);
    
    // ========== END NEW DIAGNOSTICS ==========

    delete[] h_accums;
    
    MCX_FPRINTF(cfg->flog, "[SN] Finalizing SVMC format...\n");
    finalize_svmc_kernel<<<voxel_blocks, threads>>>( d_gvol, d_accums, d_vol_labels, vol_length, dims[0], dims[1], dims[2], voxelsize[0], voxelsize[1], voxelsize[2], cfg->medianum);
    cudaDeviceSynchronize();

    
    // Copy back and repack to MCX format
    unsigned int* h_newvol = (unsigned int*)malloc(vol_length * 2 * sizeof(unsigned int));
    unsigned char* h_gvol = (unsigned char*)malloc(vol_length * 8);
    if (!h_gvol) {
        MCX_FPRINTF(stderr, "[SN Error] Failed to allocate h_gvol\n");
        // cleanup and return
    }
    cudaMemcpy(h_gvol, d_gvol, vol_length * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_newvol, d_gvol, vol_length * 2 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    debug_function(cfg, SNDBG_AFTER_FINALIZE, nullptr, h_vol_labels, dims, voxelsize,
               nullptr, vol_length, (const unsigned int*)h_newvol);


    // After finalize kernel, check for anomalies
int anomalies = 0;
int label_mismatch = 0;
for (long i = 0; i < vol_length; i++) {
    unsigned char* first = &h_gvol[i * 4];
    unsigned char lower = first[3];
    unsigned char upper = first[2];
    unsigned short vol_label = h_vol_labels[i];
    
    // Check: lower should match volume label for homogeneous, 
    // or be one of the boundary labels for surface voxels
    if (upper > 0 && upper != lower) {
        // Surface voxel - lower should equal vol_label
        if (lower != vol_label) {
            if (label_mismatch < 10) {
                int x = i % dims[0];
                int y = (i / dims[0]) % dims[1];
                int z = i / (dims[0] * dims[1]);
                MCX_FPRINTF(cfg->flog, "[SN Anomaly] Voxel %ld [%d,%d,%d]: lower=%d but vol_label=%d (upper=%d)\n",
                    i, x, y, z, lower, vol_label, upper);
            }
            label_mismatch++;
        }
    }
    
    // Check for unexpected high values
    if (upper > 3 || lower > 3) {
        anomalies++;
    }
}
MCX_FPRINTF(cfg->flog, "[SN Debug] Label mismatches: %d, Anomalies (label>3): %d\n", 
    label_mismatch, anomalies);


    int sn_surface[4] = {0}, sn_interior[4] = {0};
for (long i = 0; i < vol_length; i++) {
    unsigned char* first = &h_gvol[i * 4];
    unsigned char lower = first[3];
    unsigned char upper = first[2];
    
    if (lower < 4) {
        if (upper > 0 && upper != lower) {
            sn_surface[lower]++;
        } else {
            sn_interior[lower]++;
        }
    }
}
MCX_FPRINTF(cfg->flog, "[SN Stats] Surface voxels by label: L0=%d L1=%d L2=%d L3=%d\n",
    sn_surface[0], sn_surface[1], sn_surface[2], sn_surface[3]);
MCX_FPRINTF(cfg->flog, "[SN Stats] Interior voxels by label: L0=%d L1=%d L2=%d L3=%d\n",
    sn_interior[0], sn_interior[1], sn_interior[2], sn_interior[3]);
    
    // Debug: check actual bytes in surface voxels
    int shown = 0;
    for (long i = 0; i < vol_length && shown < 5; i++) {
        unsigned char* v = &h_gvol[i * 8];
        // Check if it's a surface voxel (upper != 0 and upper != lower)
        if (v[6] > 0 && v[6] != v[7]) {
            MCX_FPRINTF(cfg->flog, "[SN Debug] Voxel %ld raw bytes: [%d,%d,%d,%d,%d,%d,%d,%d]\n",
                        i, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
            MCX_FPRINTF(cfg->flog, "           nz=%d ny=%d nx=%d cz=%d cy=%d cx=%d upper=%d lower=%d\n",
                        v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
            shown++;
        }
    }

    // Also check a known homogeneous interior voxel (center of volume)
    long center_idx = 32 + 32*64 + 32*64*64;  // Assuming 64^3 volume
    unsigned char* vc = &h_gvol[center_idx * 8];
    MCX_FPRINTF(cfg->flog, "[SN Debug] Center voxel %ld raw: [%d,%d,%d,%d,%d,%d,%d,%d]\n",
                center_idx, vc[0], vc[1], vc[2], vc[3], vc[4], vc[5], vc[6], vc[7]);

    cudaError_t err = cudaMemcpy(h_gvol, d_gvol, vol_length * 8, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        MCX_FPRINTF(stderr, "[SN Error] cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    }

    MCX_FPRINTF(cfg->flog, "Surface Nets complete: %d ms\n", GetTimeMillis() - tic);
    
    // After repack loop, dump the PACKED format (same as MC)
    FILE* fp_packed = fopen("dump_svmc_packed.bin", "wb");
    if (fp_packed) {
        fwrite(h_newvol, sizeof(unsigned int), vol_length * 2, fp_packed);
        fclose(fp_packed);
        MCX_FPRINTF(cfg->flog, "[SN] Wrote packed volume to dump_svmc_packed.bin\n");
    }

    // Read back and verify
    FILE* fp_verify = fopen("dump_svmc_packed.bin", "rb");
    if (fp_verify) {
        unsigned int verify_val;
        fseek(fp_verify, 59036 * 4, SEEK_SET);
        fread(&verify_val, sizeof(unsigned int), 1, fp_verify);
        fclose(fp_verify);
        MCX_FPRINTF(cfg->flog, "[SN Debug] File readback voxel 59036 = 0x%08X\n", verify_val);
    }

    // DEBUG: Verify repack worked correctly
    long test_idx = 59036;  // First surface voxel from earlier debug
    unsigned char* check = (unsigned char*)h_newvol;
    MCX_FPRINTF(cfg->flog, "[SN Debug] After repack - surface voxel %ld:\n", test_idx);
    MCX_FPRINTF(cfg->flog, "  First uint bytes [0,1,2,3]: [%d,%d,%d,%d] (cy,cx,upper,lower)\n",
        check[test_idx * 4 + 0], check[test_idx * 4 + 1], 
        check[test_idx * 4 + 2], check[test_idx * 4 + 3]);
    MCX_FPRINTF(cfg->flog, "  Second uint bytes [0,1,2,3]: [%d,%d,%d,%d] (nz,ny,nx,cz)\n",
        check[(test_idx + vol_length) * 4 + 0], check[(test_idx + vol_length) * 4 + 1],
        check[(test_idx + vol_length) * 4 + 2], check[(test_idx + vol_length) * 4 + 3]);

    // Check center (homogeneous) voxel
    test_idx = 133152;
    MCX_FPRINTF(cfg->flog, "[SN Debug] After repack - center voxel %ld:\n", test_idx);
    MCX_FPRINTF(cfg->flog, "  First uint bytes [0,1,2,3]: [%d,%d,%d,%d] (cy,cx,upper,lower)\n",
        check[test_idx * 4 + 0], check[test_idx * 4 + 1],
        check[test_idx * 4 + 2], check[test_idx * 4 + 3]);
    // Write raw 8-byte SVMC volume for debugging
    FILE* fp_svmc = fopen("dump_svmc_surfacenets.bin", "wb");
    if (fp_svmc) {
        fwrite(h_gvol, 1, vol_length * 8, fp_svmc);
        fclose(fp_svmc);
        MCX_FPRINTF(cfg->flog, "[SN] Wrote SVMC 8-byte volume to dump_svmc_surfacenets.bin\n");
    }
    // Cleanup
    free(h_gvol);
    cudaFree(d_quads);
    cudaFree(d_vol_labels);
    cudaFree(d_accums);
    cudaFree(d_gvol);
    delete[] h_vol_labels;
    free(cfg->vol);
    
    cfg->vol = h_newvol;
    cfg->mediabyte = MEDIA_2LABEL_SPLIT;
}

// ==========================================================
// Device Helper Functions
// ==========================================================

__device__ __forceinline__ int voxel_idx_3d_to_1d(int x, int y, int z, int dimx, int dimy) {
    return x + y * dimx + z * dimx * dimy;
}

__device__ __forceinline__ float3 compute_triangle_normal(float3 A, float3 B, float3 C) {
    float3 AB = make_float3(B.x - A.x, B.y - A.y, B.z - A.z);
    float3 AC = make_float3(C.x - A.x, C.y - A.y, C.z - A.z);
    return make_float3( AB.y * AC.z - AB.z * AC.y, AB.z * AC.x - AB.x * AC.z, AB.x * AC.y - AB.y * AC.x);
}

__device__ __forceinline__ float triangle_area(float3 n) {
    return 0.5f * sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
}

__device__ __forceinline__ float3 normalize_vec(float3 v) {
    float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if (len > 1e-12f) {
        return make_float3(v.x/len, v.y/len, v.z/len);
    }
    return make_float3(0.0f, 0.0f, 0.0f);
}

// Separating Axis Theorem for triangle-AABB intersection
__device__ bool triangle_aabb_intersect( float3 v0, float3 v1, float3 v2, float3 box_min, float3 box_max
) {
    // Translate so box is centered at origin
    float3 c = make_float3( (box_min.x + box_max.x) * 0.5f, (box_min.y + box_max.y) * 0.5f, (box_min.z + box_max.z) * 0.5f);
    float3 e = make_float3(
        (box_max.x - box_min.x) * 0.5f,
        (box_max.y - box_min.y) * 0.5f,
        (box_max.z - box_min.z) * 0.5f
    );
    
    v0 = make_float3(v0.x - c.x, v0.y - c.y, v0.z - c.z);
    v1 = make_float3(v1.x - c.x, v1.y - c.y, v1.z - c.z);
    v2 = make_float3(v2.x - c.x, v2.y - c.y, v2.z - c.z);
    
    // Test AABB axes
    float minX = fminf(fminf(v0.x, v1.x), v2.x);
    float maxX = fmaxf(fmaxf(v0.x, v1.x), v2.x);
    if (minX > e.x || maxX < -e.x) return false;
    
    float minY = fminf(fminf(v0.y, v1.y), v2.y);
    float maxY = fmaxf(fmaxf(v0.y, v1.y), v2.y);
    if (minY > e.y || maxY < -e.y) return false;
    
    float minZ = fminf(fminf(v0.z, v1.z), v2.z);
    float maxZ = fmaxf(fmaxf(v0.z, v1.z), v2.z);
    if (minZ > e.z || maxZ < -e.z) return false;
    
    // Test triangle normal
    float3 n = compute_triangle_normal(v0, v1, v2);
    float d = v0.x*n.x + v0.y*n.y + v0.z*n.z;
    float r = e.x*fabsf(n.x) + e.y*fabsf(n.y) + e.z*fabsf(n.z);
    if (fabsf(d) > r) return false;
    
    return true;  
}

// ==========================================================
// Kernel 1: Direct Quad-to-Voxel Assignment
// ==========================================================
__global__ void assign_quads_to_voxels_kernel(
    const SNQuad* quads, int num_quads, VoxelAccum* accums,
    const unsigned short* vol_labels,
    int dimx, int dimy, int dimz,
    float vsx, float vsy, float vsz
) {
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= num_quads) return;
    
    SNQuad q = quads[qid];
    if (q.labels[0] == q.labels[1] || q.labels[0] == 0xFFFF) return;
    
    float3 A = make_float3(q.corners[0], q.corners[1], q.corners[2]);
    float3 B = make_float3(q.corners[3], q.corners[4], q.corners[5]);
    float3 C = make_float3(q.corners[6], q.corners[7], q.corners[8]);
    float3 D = make_float3(q.corners[9], q.corners[10], q.corners[11]);
    
    float3 n1 = compute_triangle_normal(A, B, C);
    float3 n2 = compute_triangle_normal(A, C, D);
    float area = triangle_area(n1) + triangle_area(n2);
    if (area < 1e-12f) return;
    
    float3 normal = normalize_vec(make_float3(n1.x + n2.x, n1.y + n2.y, n1.z + n2.z));
    
    // OVERSAMPLE: Create sample points - 4 per triangle = 8 per quad
    float3 samples[8];
    int num_samples = 0;
    
    // Triangle 1: A-B-C
    float3 c1 = make_float3( (A.x + B.x + C.x) / 3.0f, (A.y + B.y + C.y) / 3.0f, (A.z + B.z + C.z) / 3.0f);
    samples[num_samples++] = c1;
    samples[num_samples++] = make_float3( (c1.x + A.x) * 0.5f, (c1.y + A.y) * 0.5f, (c1.z + A.z) * 0.5f);
    samples[num_samples++] = make_float3( (c1.x + B.x) * 0.5f, (c1.y + B.y) * 0.5f, (c1.z + B.z) * 0.5f);
    samples[num_samples++] = make_float3( (c1.x + C.x) * 0.5f, (c1.y + C.y) * 0.5f, (c1.z + C.z) * 0.5f);

    
    // Triangle 2: A-C-D
    float3 c2 = make_float3( (A.x + C.x + D.x) / 3.0f, (A.y + C.y + D.y) / 3.0f, (A.z + C.z + D.z) / 3.0f);
    samples[num_samples++] = c2;
    samples[num_samples++] = make_float3( (c2.x + A.x) * 0.5f, (c2.y + A.y) * 0.5f, (c2.z + A.z) * 0.5f);
    samples[num_samples++] = make_float3( (c2.x + C.x) * 0.5f, (c2.y + C.y) * 0.5f, (c2.z + C.z) * 0.5f);
    samples[num_samples++] = make_float3( (c2.x + D.x) * 0.5f, (c2.y + D.y) * 0.5f, (c2.z + D.z) * 0.5f);
    
    // Split area equally among samples
    float sample_area = area / num_samples;
    
    // For each sample point, find its voxel and assign
    for (int i = 0; i < num_samples; i++) {
        float3 sample_grid = make_float3(
            samples[i].x / vsx,
            samples[i].y / vsy,
            samples[i].z / vsz
        );
        
        // Determine which voxel this sample is in
        int sx = (int)floorf(sample_grid.x);
        int sy = (int)floorf(sample_grid.y);
        int sz = (int)floorf(sample_grid.z);
        
        // Clamp to bounds
        sx = max(0, min(sx, dimx - 1));
        sy = max(0, min(sy, dimy - 1));
        sz = max(0, min(sz, dimz - 1));
        
        int idx = sx + sy * dimx + sz * dimx * dimy;
        
        // Only assign if voxel has one of the interface labels
        unsigned short voxel_label = vol_labels[idx];
        if (voxel_label == q.labels[0] || voxel_label == q.labels[1]) {
            // Use SAMPLE position, not centroid
            atomicAdd(&accums[idx].cx, samples[i].x * sample_area);
            atomicAdd(&accums[idx].cy, samples[i].y * sample_area);
            atomicAdd(&accums[idx].cz, samples[i].z * sample_area);
            atomicAdd(&accums[idx].nx, normal.x * sample_area);
            atomicAdd(&accums[idx].ny, normal.y * sample_area);
            atomicAdd(&accums[idx].nz, normal.z * sample_area);
            atomicAdd(&accums[idx].total_weight, sample_area);
            atomicAdd(&accums[idx].quad_count, 1);
            atomicMin(&accums[idx].lower, (unsigned int)q.labels[0]);
            atomicMax(&accums[idx].upper, (unsigned int)q.labels[1]);
        }
    }
}
// ==========================================================
// Kernel 2: Finalize Voxels to SVMC Format
// ==========================================================

__global__ void finalize_svmc_kernel(
    unsigned char* gvol,  // Now interleaved: first vol_length*4 bytes, then second vol_length*4 bytes
    const VoxelAccum* accums,
    const unsigned short* vol_labels,
    long vol_length,
    int dimx, int dimy, int dimz,
    float vsx, float vsy, float vsz,
    int nMedia
) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vol_length) return;
    
    // Interleaved format pointers (like MC does)
    unsigned char* first_uint = &gvol[idx * 4];              // First half
    unsigned char* second_uint = &gvol[(idx + vol_length) * 4];  // Second half
    
    const VoxelAccum* A = &accums[idx];
    
    int vx = idx % dimx;
    int vy = (idx / dimx) % dimy;
    int vz = idx / (dimx * dimy);
    
    if (A->total_weight < 1e-12f || A->quad_count == 0) {
        // Homogeneous voxel
        unsigned short lab = vol_labels[idx];
        if (lab >= nMedia) lab = 0;
        
        first_uint[0] = 0;                    // cy
        first_uint[1] = 0;                    // cx
        first_uint[2] = 0;                    // upper
        first_uint[3] = (unsigned char)lab;   // lower
        
        second_uint[0] = 0;  // nz
        second_uint[1] = 0;  // ny
        second_uint[2] = 0;  // nx
        second_uint[3] = 0;  // cz
        return;
    }
    
    // Compute final centroid and normal
    float inv_w = 1.0f / A->total_weight;
    float3 centroid_world = make_float3(
        A->cx * inv_w,
        A->cy * inv_w,
        A->cz * inv_w
    );
    float3 normal = normalize_vec(make_float3(A->nx, A->ny, A->nz));
    
    // Get labels
    unsigned short lower = (A->lower == 0xFFFF) ? 0 : (unsigned short)A->lower;
    unsigned short upper = (A->upper == 0) ? 0 : (unsigned short)A->upper;
    
    if (lower >= nMedia) lower = 0;
    if (upper >= nMedia) upper = lower;
    if (lower > upper) {
        unsigned short tmp = lower;
        lower = upper;
        upper = tmp;
    }
    
    // Convert centroid to voxel-local coordinates [0, 1]
    float3 centroid_grid = make_float3(
        centroid_world.x / vsx,
        centroid_world.y / vsy,
        centroid_world.z / vsz
    );
    float3 centroid_local = make_float3(
        centroid_grid.x - (float)vx,
        centroid_grid.y - (float)vy,
        centroid_grid.z - (float)vz
    );
    
    centroid_local.x = fminf(fmaxf(centroid_local.x, 0.0f), 1.0f);
    centroid_local.y = fminf(fmaxf(centroid_local.y, 0.0f), 1.0f);
    centroid_local.z = fminf(fmaxf(centroid_local.z, 0.0f), 1.0f);
    
    // Orient normal
    float offset = 0.1f * fminf(fminf(vsx, vsy), vsz);
    int tx = (int)floorf((centroid_world.x + offset * normal.x) / vsx);
    int ty = (int)floorf((centroid_world.y + offset * normal.y) / vsy);
    int tz = (int)floorf((centroid_world.z + offset * normal.z) / vsz);
    int ax = (int)floorf((centroid_world.x - offset * normal.x) / vsx);
    int ay = (int)floorf((centroid_world.y - offset * normal.y) / vsy);
    int az = (int)floorf((centroid_world.z - offset * normal.z) / vsz);
    
    unsigned short toward_label = 0, away_label = 0;
    if (tx >= 0 && tx < dimx && ty >= 0 && ty < dimy && tz >= 0 && tz < dimz)
        toward_label = vol_labels[voxel_idx_3d_to_1d(tx, ty, tz, dimx, dimy)];
    if (ax >= 0 && ax < dimx && ay >= 0 && ay < dimy && az >= 0 && az < dimz)
        away_label = vol_labels[voxel_idx_3d_to_1d(ax, ay, az, dimx, dimy)];
    
    if (away_label == upper && toward_label == lower) {
        normal.x = -normal.x;
        normal.y = -normal.y;
        normal.z = -normal.z;
    }
    
    // Encode to bytes
    unsigned char cx = (unsigned char)(centroid_local.x * 255.0f);
    unsigned char cy = (unsigned char)(centroid_local.y * 255.0f);
    unsigned char cz = (unsigned char)(centroid_local.z * 255.0f);
    unsigned char nx = (unsigned char)((normal.x + 1.0f) * 127.5f);
    unsigned char ny = (unsigned char)((normal.y + 1.0f) * 127.5f);
    unsigned char nz = (unsigned char)((normal.z + 1.0f) * 127.5f);
    
    // Write interleaved format (same as MC)
    first_uint[0] = cy;
    first_uint[1] = cx;
    first_uint[2] = (unsigned char)(upper & 0xFF);
    first_uint[3] = (unsigned char)(lower & 0xFF);
    
    second_uint[0] = nz;
    second_uint[1] = ny;
    second_uint[2] = nx;
    second_uint[3] = cz;
}

// ==========================================================
// Fixed Quad Extraction with Per-Edge Boundary Conditions
// 
// Key fix: The original code applied uniform boundary checks to all edges,
// but each edge type only needs checks on its PERPENDICULAR dimensions.
// This was causing systematic gaps in the mesh.
// ==========================================================

std::vector<SNQuad> extract_quads_with_tracking(MMCellMap* cellmap, const unsigned short* vol_labels, int dims[3], float voxelsize[3]) {
    std::vector<SNQuad> quads;
    if (!cellmap) return quads;
    
    quads.reserve((long)dims[0] * dims[1] * dims[2]);
    
    // Canonical edges
    const MMCellFlag::Edge canon_edges[3] = {
        MMCellFlag::LeftBottomEdge,   // e=0: quad in XZ plane, cells differ in Y
        MMCellFlag::BackBottomEdge,   // e=1: quad in YZ plane, cells differ in X
        MMCellFlag::LeftBackEdge      // e=2: quad in XY plane, cells differ in Z
    };
    
    // For each edge type, which dimensions need ci >= 1 (to have -1 neighbor for quad)
    const int perp_dims[3][2] = {
        {0, 2},  // Edge 0
        {1, 2},  // Edge 1
        {0, 1}   // Edge 2
    };
    
    // Direction of label difference for each edge
    const int edge_offsets[3][3] = {
        {0, 1, 0},  // e=0: +Y
        {1, 0, 0},  // e=1: +X
        {0, 0, 1}   // e=2: +Z
    };
    
    auto get_label = [&](int x, int y, int z) -> unsigned short {
        if (x < 0 || x >= dims[0] || y < 0 || y >= dims[1] || z < 0 || z >= dims[2])
            return 0xFFFF;
        return vol_labels[z * dims[0] * dims[1] + y * dims[0] + x];
    };
    
    for (int v = 0; v < cellmap->numVertices(); v++) {
        int ci[3];
        cellmap->getVertexCellIndex(v, ci);
        
        for (int e = 0; e < 3; e++) {
            // Per-edge boundary check: only check perpendicular dimensions
            int d1 = perp_dims[e][0];
            int d2 = perp_dims[e][1];
            if (ci[d1] < 1 || ci[d2] < 1) continue;
            
            // Check upper bound on the edge-parallel axis
            int axis = (e == 0) ? 1 : (e == 1) ? 0 : 2;
            if (ci[axis] > dims[axis]) continue;
            
            SNQuad q;
            unsigned short labs[2];
            if (!cellmap->getEdgeQuad(v, canon_edges[e], q.corners, labs)) continue;
            
            // Convert from padded to original coordinates
            int vox1[3] = {ci[0] - 1, ci[1] - 1, ci[2] - 1};
            int vox2[3] = {
                vox1[0] + edge_offsets[e][0],
                vox1[1] + edge_offsets[e][1],
                vox1[2] + edge_offsets[e][2]
            };

            q.owner[0] = vox1[0];
            q.owner[1] = vox1[1];
            q.owner[2] = vox1[2];
            
            unsigned short label1 = get_label(vox1[0], vox1[1], vox1[2]);
            unsigned short label2 = get_label(vox2[0], vox2[1], vox2[2]);
            
            // Skip boundary quads (one or both voxels in padding)
            if (label1 == 0xFFFF || label2 == 0xFFFF) continue;
            
            // Skip if same label (no material boundary)
            if (label1 == label2) continue;
            
            // Valid material boundary quad
            q.labels[0] = std::min(label1, label2);
            q.labels[1] = std::max(label1, label2);
            
            int* lo_vox = (label1 < label2) ? vox1 : vox2;
            int* hi_vox = (label1 < label2) ? vox2 : vox1;
            for (int i = 0; i < 3; i++) {
                q.voxel1[i] = lo_vox[i];
                q.voxel2[i] = hi_vox[i];
            }
            
            // Convert corners from padded to original coordinates
            for (int c = 0; c < 4; c++) {
                q.corners[c*3 + 0] -= voxelsize[0];
                q.corners[c*3 + 1] -= voxelsize[1];
                q.corners[c*3 + 2] -= voxelsize[2];
            }

            // Verify voxel labels match
unsigned short check1 = get_label(q.voxel1[0], q.voxel1[1], q.voxel1[2]);
unsigned short check2 = get_label(q.voxel2[0], q.voxel2[1], q.voxel2[2]);
if (check1 != q.labels[0] || check2 != q.labels[1]) {
    static int mismatch_count = 0;
    if (mismatch_count++ < 10) {
        printf("EXTRACTION MISMATCH: labels=[%d,%d] but voxel1[%d,%d,%d] has %d, voxel2[%d,%d,%d] has %d\n",
            q.labels[0], q.labels[1],
            q.voxel1[0], q.voxel1[1], q.voxel1[2], check1,
            q.voxel2[0], q.voxel2[1], q.voxel2[2], check2);
    }
}
            
            q.edge_axis = e;
            quads.push_back(q);
        }
    }
    
    return quads;
}
// void dump_quads_obj(const std::vector<SNQuad>& quads, const char* filename) {
//     FILE* fobj = fopen(filename, "w");
//     if (!fobj) return;
    
//     for (size_t i = 0; i < quads.size(); ++i) {
//         const SNQuad& q = quads[i];
//         fprintf(fobj,
//             "v %.6f %.6f %.6f\n"
//             "v %.6f %.6f %.6f\n"
//             "v %.6f %.6f %.6f\n"
//             "v %.6f %.6f %.6f\n",
//             q.corners[0],  q.corners[1],  q.corners[2],
//             q.corners[3],  q.corners[4],  q.corners[5],
//             q.corners[6],  q.corners[7],  q.corners[8],
//             q.corners[9],  q.corners[10], q.corners[11]);
//         size_t base = i * 4;
//         fprintf(fobj, "f %zu %zu %zu\n", base + 1, base + 2, base + 3);
//         fprintf(fobj, "f %zu %zu %zu\n", base + 1, base + 3, base + 4);
//     }
//     fclose(fobj);
//     printf("[SN] Dumped %zu quads to %s\n", quads.size(), filename);
// }

// Diagnostic function to dump extracted quads as OBJ
// Add this to your mcx_svmc_sn.cu to visualize raw quad extraction

void dump_quads_obj(const std::vector<SNQuad>& quads, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }
    
    fprintf(fp, "# Surface Nets quads diagnostic\n");
    fprintf(fp, "# Total quads: %zu\n", quads.size());
    
    // Write all vertices first
    int vertex_count = 0;
    for (size_t i = 0; i < quads.size(); i++) {
        const SNQuad& q = quads[i];
        // 4 corners per quad
        for (int c = 0; c < 4; c++) {
            fprintf(fp, "v %f %f %f\n", 
                q.corners[c*3 + 0],
                q.corners[c*3 + 1], 
                q.corners[c*3 + 2]);
        }
        vertex_count += 4;
    }
    
    fprintf(fp, "# Vertices: %d\n", vertex_count);
    
    // Write faces (1-indexed in OBJ format)
    for (size_t i = 0; i < quads.size(); i++) {
        int base = i * 4 + 1;  // OBJ is 1-indexed
        // Quad as two triangles: ABC and ACD
        fprintf(fp, "f %d %d %d\n", base, base+1, base+2);
        fprintf(fp, "f %d %d %d\n", base, base+2, base+3);
    }
    
    fclose(fp);
    printf("Wrote %zu quads (%d vertices) to %s\n", quads.size(), vertex_count, filename);
}

// Also add per-label-pair statistics:
void analyze_quads(const std::vector<SNQuad>& quads) {
    std::map<std::pair<int,int>, int> label_pair_counts;
    std::map<int, int> edge_axis_counts;
    
    float min_area = 1e30f, max_area = 0.0f, total_area = 0.0f;
    int degenerate_count = 0;
    
    for (const auto& q : quads) {
        // Count by label pair
        label_pair_counts[{q.labels[0], q.labels[1]}]++;
        
        // Count by edge axis
        edge_axis_counts[q.edge_axis]++;
        
        // Calculate quad area
        float3 A = {q.corners[0], q.corners[1], q.corners[2]};
        float3 B = {q.corners[3], q.corners[4], q.corners[5]};
        float3 C = {q.corners[6], q.corners[7], q.corners[8]};
        float3 D = {q.corners[9], q.corners[10], q.corners[11]};
        
        // Cross products for triangle areas
        float3 AB = {B.x-A.x, B.y-A.y, B.z-A.z};
        float3 AC = {C.x-A.x, C.y-A.y, C.z-A.z};
        float3 AD = {D.x-A.x, D.y-A.y, D.z-A.z};
        
        float3 n1 = {AB.y*AC.z - AB.z*AC.y, AB.z*AC.x - AB.x*AC.z, AB.x*AC.y - AB.y*AC.x};
        float3 n2 = {AC.y*AD.z - AC.z*AD.y, AC.z*AD.x - AC.x*AD.z, AC.x*AD.y - AC.y*AD.x};
        
        float area = 0.5f * (sqrtf(n1.x*n1.x + n1.y*n1.y + n1.z*n1.z) + sqrtf(n2.x*n2.x + n2.y*n2.y + n2.z*n2.z));
        
        if (area < 1e-10f) {
            degenerate_count++;
        } else {
            min_area = fminf(min_area, area);
            max_area = fmaxf(max_area, area);
            total_area += area;
        }
    }
    
    printf("\n=== Quad Analysis ===\n");
    printf("Total quads: %zu\n", quads.size());
    printf("Degenerate (zero area): %d\n", degenerate_count);
    printf("Area stats: min=%.6f max=%.6f avg=%.6f total=%.2f\n",
           min_area, max_area, total_area/quads.size(), total_area);
    
    printf("\nBy label pair:\n");
    for (const auto& kv : label_pair_counts) {
        printf("  [%d,%d]: %d quads\n", kv.first.first, kv.first.second, kv.second);
    }
    
    printf("\nBy edge axis:\n");
    printf("  X-edge (e=0, Y-diff): %d\n", edge_axis_counts[0]);
    printf("  Y-edge (e=1, X-diff): %d\n", edge_axis_counts[1]);
    printf("  Z-edge (e=2, Z-diff): %d\n", edge_axis_counts[2]);
}

// Diagnostic for checking voxel coverage
void analyze_voxel_assignment(const std::vector<SNQuad>& quads,  const unsigned short* vol_labels, int dims[3]) {
    // Track which voxels receive quads
    std::set<int> assigned_voxels;
    std::set<int> boundary_voxels;
    
    auto idx3d = [&](int x, int y, int z) { return x + y*dims[0] + z*dims[0]*dims[1]; };
    
    // Find all boundary voxels (have neighbor with different label)
    for (int z = 0; z < dims[2]; z++) {
        for (int y = 0; y < dims[1]; y++) {
            for (int x = 0; x < dims[0]; x++) {
                unsigned short lab = vol_labels[idx3d(x,y,z)];
                bool is_boundary = false;
                
                // Check 6 neighbors
                if (x > 0 && vol_labels[idx3d(x-1,y,z)] != lab) is_boundary = true;
                if (x < dims[0]-1 && vol_labels[idx3d(x+1,y,z)] != lab) is_boundary = true;
                if (y > 0 && vol_labels[idx3d(x,y-1,z)] != lab) is_boundary = true;
                if (y < dims[1]-1 && vol_labels[idx3d(x,y+1,z)] != lab) is_boundary = true;
                if (z > 0 && vol_labels[idx3d(x,y,z-1)] != lab) is_boundary = true;
                if (z < dims[2]-1 && vol_labels[idx3d(x,y,z+1)] != lab) is_boundary = true;
                
                if (is_boundary) {
                    boundary_voxels.insert(idx3d(x,y,z));
                }
            }
        }
    }
    
    // Check which voxels would receive quads (using HIGHER label strategy)
    for (const auto& q : quads) {
        unsigned short l1 = vol_labels[idx3d(q.voxel1[0], q.voxel1[1], q.voxel1[2])];
        unsigned short l2 = vol_labels[idx3d(q.voxel2[0], q.voxel2[1], q.voxel2[2])];
        
        int vx, vy, vz;
        if (l1 >= l2) {
            vx = q.voxel1[0]; vy = q.voxel1[1]; vz = q.voxel1[2];
        } else {
            vx = q.voxel2[0]; vy = q.voxel2[1]; vz = q.voxel2[2];
        }
        
        if (vx >= 0 && vx < dims[0] && vy >= 0 && vy < dims[1] && vz >= 0 && vz < dims[2]) {
            assigned_voxels.insert(idx3d(vx, vy, vz));
        }
    }
    
    // Find boundary voxels without any quad assignment
    std::vector<int> missing;
    for (int idx : boundary_voxels) {
        if (assigned_voxels.find(idx) == assigned_voxels.end()) {
            missing.push_back(idx);
        }
    }
    
    printf("\n=== Voxel Assignment Analysis ===\n");
    printf("Total boundary voxels: %zu\n", boundary_voxels.size());
    printf("Voxels with quad assignments: %zu\n", assigned_voxels.size());
    printf("Boundary voxels missing coverage: %zu (%.1f%%)\n", 
           missing.size(), 100.0 * missing.size() / boundary_voxels.size());
    
    if (!missing.empty() && missing.size() <= 20) {
        printf("Missing voxel indices (first 20):\n");
        for (size_t i = 0; i < std::min(missing.size(), (size_t)20); i++) {
            int idx = missing[i];
            int x = idx % dims[0];
            int y = (idx / dims[0]) % dims[1];
            int z = idx / (dims[0] * dims[1]);
            printf("  [%d,%d,%d] label=%d\n", x, y, z, vol_labels[idx]);
        }
    }
}

static void debug_function(
    Config* cfg,
    SNDStage stage,
    const std::vector<SNQuad>* quads,
    const unsigned short* vol_labels,
    const int dims[3],
    const float voxelsize[3],
    const VoxelAccum* h_accums,   // optional (only after assign)
    long vol_length,
    const unsigned int* packed2u32 // optional (only after finalize)
) {
#if MCX_SN_DEBUG
    FILE* log = (cfg && cfg->flog) ? cfg->flog : stderr;

    auto idx3d = [&](int x,int y,int z){ return x + y*dims[0] + z*dims[0]*dims[1]; };

    if (stage == SNDBG_BEFORE_RELAX || stage == SNDBG_AFTER_RELAX) {
        if (!quads) return;

        const char* objname =
            (stage == SNDBG_BEFORE_RELAX) ? "dump_SN_BEFORE_surfacenets.obj" : "dump_SN_AFTER_surfacenets.obj";

        dump_quads_obj(*quads, objname);

        analyze_quads(*quads);
        analyze_voxel_assignment(*quads, vol_labels, (int*)dims);

        // small sample print
        fprintf(log, "[SNDBG] First 5 quads:\n");
        for (int i = 0; i < (int)std::min<size_t>(5, quads->size()); i++) {
            const auto& q = (*quads)[i];
            fprintf(log,
                "  q%d labels=[%u,%u] owner=[%d,%d,%d] v1=[%d,%d,%d] v2=[%d,%d,%d] axis=%d\n",
                i, q.labels[0], q.labels[1],
                q.owner[0], q.owner[1], q.owner[2],
                q.voxel1[0], q.voxel1[1], q.voxel1[2],
                q.voxel2[0], q.voxel2[1], q.voxel2[2],
                q.edge_axis);
        }
        return;
    }

    if (stage == SNDBG_AFTER_ASSIGN && h_accums && vol_length > 0) {
        long surface = 0, homo = 0;
        for (long i = 0; i < vol_length; i++) {
            if (h_accums[i].quad_count > 0 && h_accums[i].total_weight > 1e-12f) surface++;
            else homo++;
        }
        fprintf(log, "[SNDBG] After assign: surface=%ld homo=%ld total=%ld\n", surface, homo, vol_length);
        return;
    }

    if (stage == SNDBG_AFTER_FINALIZE && packed2u32 && vol_length > 0) {
        // Write contiguous 8-byte-per-voxel file: [nz,ny,nx,cz, cy,cx,upper,lower]
        std::vector<unsigned char> out8((size_t)vol_length * 8);

        const unsigned char* first  = (const unsigned char*)packed2u32; // [cy,cx,upper,lower]
        const unsigned char* second = first + (size_t)vol_length * 4;   // [nz,ny,nx,cz]

        for (long i = 0; i < vol_length; i++) {
            // second then first
            memcpy(&out8[i*8 + 0], &second[i*4], 4);
            memcpy(&out8[i*8 + 4], &first[i*4], 4);
        }

        FILE* fp = fopen("dump_svmc_surfacenets.bin", "wb");
        if (fp) {
            fwrite(out8.data(), 1, out8.size(), fp);
            fclose(fp);
            fprintf(log, "[SNDBG] Wrote dump_svmc_surfacenets.bin (%zu bytes)\n", out8.size());
        }
        return;
    }
#else
    (void)cfg; (void)stage; (void)quads; (void)vol_labels; (void)dims; (void)voxelsize;
    (void)h_accums; (void)vol_length; (void)packed2u32;
#endif
}
