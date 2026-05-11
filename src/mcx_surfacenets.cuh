// ==========================================================
// File: mcx_surfacenets.cuh
// GPU-friendly version of MMSurfaceNet (header only)
// ==========================================================

#ifndef MCX_SURFACENETS_CUH
#define MCX_SURFACENETS_CUH

#include <cuda_runtime.h>

// Forward declaration (will be defined in mcx_sn_cellmap.cuh)
struct MMCellMap;

// CUDA-friendly MMSurfaceNet
struct MMSurfaceNet {
    // Constructor & destructor equivalents (to be implemented in .cu)
    __host__ MMSurfaceNet(unsigned short* labels, int arraySize[3], float voxelSize[3]);
    __host__ ~MMSurfaceNet();

    __host__ void dump_obj(const char* filename) const;


    // Access cell map (device pointer to struct)
    __host__ MMCellMap* cellMap() const {
        return m_cellMap;
    }

    // Surface smoothing (relaxation)
    struct RelaxAttrs {
        int   numRelaxIterations;      // More iterations --> smoother but slower
        float relaxFactor;             // Range (0.0, 1.0); larger --> faster but less stable
        float maxDistFromCellCenter;   // Max displacement in voxel units
    };

    __host__ void relax(const RelaxAttrs relaxAttrs); // implemented in .cu
    __host__ void reset();                            // implemented in .cu

    // Get the unique material labels for this SurfaceNet

    __host__ void labels(int** outLabels, int* count);
    __host__ void getVertexPosition(int vid, float pos[3]) const;


    // Label used internally. Not available as material index.
    enum ReservedLabel { Padding = 65535 };

  private:
    MMCellMap* m_cellMap;  // device/host pointer depending on context
};

#endif // MCX_SURFACENETS_CUH
