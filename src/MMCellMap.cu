// ==========================================================
// File: MMCellMap.cu
// GPU implementation of MMCellMap methods
// ==========================================================
#include <cub/cub.cuh>
#include "MMCellMap.cuh"
#include "MMCellFlag.cuh"


// // Constructor
// __host__ MMCellMap::MMCellMap(unsigned short *labels, int arraySize[3], float voxelSize[3]) : m_cellArray(nullptr), m_numVertices(0), m_vertices(nullptr)
// {
//     for (int i = 0; i < 3; i++) {
//         m_arraySize[i] = arraySize[i] + 2; // Pad by 1 voxel on each side
//         m_voxelSize[i] = voxelSize[i];
//     }

//     int numCells = m_arraySize[0] * m_arraySize[1] * m_arraySize[2];

//     try {
//         m_cellArray = new Cell[numCells];
//     } catch (std::bad_alloc&) {
//         m_cellArray = nullptr;
//         return;
//     }

//     // Initialize interior with original labels
//     Cell* pCell = m_cellArray;
//     unsigned short* pLabel = labels;
//     for (int k = 0; k < m_arraySize[2]; k++) {
//         for (int j = 0; j < m_arraySize[1]; j++) {
//             for (int i = 0; i < m_arraySize[0]; i++) {
//                 if (i == 0 || i == m_arraySize[0] - 1 ||
//                     j == 0 || j == m_arraySize[1] - 1 ||
//                     k == 0 || k == m_arraySize[2] - 1) {
//                     // To be filled with replicate padding below
//                     initCell(pCell++, 0); // Temporary placeholder
//                 } else {
//                     initCell(pCell++, *pLabel++);
//                 }
//             }
//         }
//     }

//     // Replicate padding: Copy nearest interior voxel labels
//     // Note: m_arraySize = arraySize + 2 (padding of 1 voxel per side)
//     int dimx = arraySize[0], dimy = arraySize[1], dimz = arraySize[2];
//     int new_dimx = m_arraySize[0], new_dimy = m_arraySize[1], new_dimz = m_arraySize[2];

//     // Helper function to get label at (i,j,k) in m_cellArray
//     auto getCellLabel = [&](int i, int j, int k) -> unsigned short {
//         int idx = i + j * new_dimx + k * new_dimx * new_dimy;
//         return m_cellArray[idx].label; // Assuming Cell has a 'label' field
//     };

//     // Helper function to set label at (i,j,k) in m_cellArray
//     auto setCellLabel = [&](int i, int j, int k, unsigned short label) {
//         int idx = i + j * new_dimx + k * new_dimx * new_dimy;
//         initCell(&m_cellArray[idx], label);
//     };

//     // Pad along -x (i=0)
//     for (int j = 0; j < new_dimy; j++) {
//         for (int k = 0; k < new_dimz; k++) {
//             if (j == 0 || j == new_dimy - 1 || k == 0 || k == new_dimz - 1) continue; // Handle later
//             setCellLabel(0, j, k, getCellLabel(1, j, k));
//         }
//     }
//     // Pad along +x (i=new_dimx-1)
//     for (int j = 0; j < new_dimy; j++) {
//         for (int k = 0; k < new_dimz; k++) {
//             if (j == 0 || j == new_dimy - 1 || k == 0 || k == new_dimz - 1) continue;
//             setCellLabel(new_dimx - 1, j, k, getCellLabel(new_dimx - 2, j, k));
//         }
//     }

//     // Pad along -y (j=0)
//     for (int i = 0; i < new_dimx; i++) {
//         for (int k = 0; k < new_dimz; k++) {
//             if (i == 0 || i == new_dimx - 1 || k == 0 || k == new_dimz - 1) continue;
//             setCellLabel(i, 0, k, getCellLabel(i, 1, k));
//         }
//     }
//     // Pad along +y (j=new_dimy-1)
//     for (int i = 0; i < new_dimx; i++) {
//         for (int k = 0; k < new_dimz; k++) {
//             if (i == 0 || i == new_dimx - 1 || k == 0 || k == new_dimz - 1) continue;
//             setCellLabel(i, new_dimy - 1, k, getCellLabel(i, new_dimy - 2, k));
//         }
//     }

//     // Pad along -z (k=0)
//     for (int i = 0; i < new_dimx; i++) {
//         for (int j = 0; j < new_dimy; j++) {
//             if (i == 0 || i == new_dimx - 1 || j == 0 || j == new_dimy - 1) continue;
//             setCellLabel(i, j, 0, getCellLabel(i, j, 1));
//         }
//     }
//     // Pad along +z (k=new_dimz-1)
//     for (int i = 0; i < new_dimx; i++) {
//         for (int j = 0; j < new_dimy; j++) {
//             if (i == 0 || i == new_dimx - 1 || j == 0 || j == new_dimy - 1) continue;
//             setCellLabel(i, j, new_dimz - 1, getCellLabel(i, j, new_dimz - 2));
//         }
//     }

//     // Handle edges and corners (e.g., (0,0,k), (0,j,0), etc.)
//     for (int k = 0; k < new_dimz; k++) {
//         for (int j = 0; j < new_dimy; j++) {
//             for (int i = 0; i < new_dimx; i++) {
//                 if ((i == 0 || i == new_dimx - 1) && (j == 0 || j == new_dimy - 1)) {
//                     setCellLabel(i, j, k, getCellLabel(i == 0 ? 1 : new_dimx - 2, j == 0 ? 1 : new_dimy - 2, k));
//                 }
//                 if ((i == 0 || i == new_dimx - 1) && (k == 0 || k == new_dimz - 1)) {
//                     setCellLabel(i, j, k, getCellLabel(i == 0 ? 1 : new_dimx - 2, j, k == 0 ? 1 : new_dimz - 2));
//                 }
//                 if ((j == 0 || j == new_dimy - 1) && (k == 0 || k == new_dimz - 1)) {
//                     setCellLabel(i, j, k, getCellLabel(i, j == 0 ? 1 : new_dimy - 2, k == 0 ? 1 : new_dimz - 2));
//                 }
//             }
//         }
//     }

//     setCellVertices();
// }

// =============================================================
// relaxation kernel
// =============================================================
// __global__ void kernel_relax(
//     const float3* d_in,          // per-vertex offsets (input of this iter)
//     float3*       d_out,         // per-vertex offsets (output of this iter)
//     const int*    d_cellIdx,     // per-vertex PADDED linear cell index
//     const Cell*   d_cells,       // full padded cell map
//     int           numVertices,
//     int3          dims,          // dims = (m_arraySize[0], m_arraySize[1], m_arraySize[2])
//     MMSurfaceNet::RelaxAttrs attrs )
// {
//     int vid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (vid >= numVertices) return;

//     // Which padded cell does this vertex live in?
//     int cell1D = d_cellIdx[vid];
//     int cx = cell1D % dims.x;
//     int cy = (cell1D / dims.x) % dims.y;
//     int cz = cell1D / (dims.x * dims.y);

//     const Cell* pCell = &d_cells[cell1D];
//     MMCellFlag::VertexType vtype = pCell->flag.vertexType();

//     // Accumulate neighbor positions
//     float3 avg = make_float3(0.f, 0.f, 0.f);
//     int count  = 0;

//     // Padded neighbor offsets (matches getFaceNeighborCellAndIndex)
//     const int3 off[6] = {
//         { -1,  0,  0 }, // Left
//         {  1,  0,  0 }, // Right
//         {  0, -1,  0 }, // Back
//         {  0,  1,  0 }, // Front
//         {  0,  0, -1 }, // Bottom
//         {  0,  0,  1 }  // Top
//     };

//     const MMCellFlag::Face faces[6] = {
//         MMCellFlag::LeftFace,   MMCellFlag::RightFace,
//         MMCellFlag::BackFace,   MMCellFlag::FrontFace,
//         MMCellFlag::BottomFace, MMCellFlag::TopFace
//     };

//     for (int f = 0; f < 6; ++f) {
//         int nx = cx + off[f].x;
//         int ny = cy + off[f].y;
//         int nz = cz + off[f].z;

//         // stay inside padded grid
//         if (nx < 0 || nx >= dims.x || ny < 0 || ny >= dims.y || nz < 0 || nz >= dims.z)
//             continue;

//         int n1D = nx + dims.x * (ny + dims.y * nz);
//         const Cell* nbr = &d_cells[n1D];

//         // same logic as CPU: which neighbors are used?
//         MMCellFlag::FaceCrossingType fct = pCell->flag.faceCrossingType(faces[f]);

//         bool useNeighbor = (vtype == MMCellFlag::SurfaceVertex) ? (fct != MMCellFlag::NoFaceCrossing) : (fct == MMCellFlag::JunctionFaceCrossing);

//         if (!useNeighbor)
//             continue;

//         int nid = nbr->vertexIndex;
//         if (nid < 0) // neighbor cell has no vertex
//             continue;

//         // USE OFFSETS FROM PREVIOUS ITERATION (Jacobi)
//         float3 np = d_in[nid];

//         // same math as CPU: neighbor offset + (nbrIdx - cellIdx)
//         avg.x += np.x + float(nx - cx);
//         avg.y += np.y + float(ny - cy);
//         avg.z += np.z + float(nz - cz);
//         ++count;
//     }

//     float3 p = d_in[vid]; // current vertex position

//     if (count > 0) {
//         float inv = 1.0f / float(count);
//         avg.x *= inv;
//         avg.y *= inv;
//         avg.z *= inv;

//         float a = attrs.relaxFactor;
//         // exact CPU formula
//         p.x = (1.0f - a) * p.x + a * avg.x;
//         p.y = (1.0f - a) * p.y + a * avg.y;
//         p.z = (1.0f - a) * p.z + a * avg.z;

//         // constrain distance from center of cell
//         float minc = 0.5f - attrs.maxDistFromCellCenter; 
//         float maxc = 0.5f + attrs.maxDistFromCellCenter;

//         p.x = fminf(fmaxf(p.x, minc), maxc);
//         p.y = fminf(fmaxf(p.y, minc), maxc);
//         p.z = fminf(fmaxf(p.z, minc), maxc);
//     }

//     d_out[vid] = p;
// }

__global__ void kernel_assign_vertices(Cell* d_cells, int nx, int ny, int nz, int pitchX, int pitchXY, MMCellMap::Vertex* d_vertices, int* d_counter)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nz;
    if (idx >= total) return;

    int k = idx / (nx * ny);
    int j = (idx / nx) % ny;
    int i = idx % nx;

    int cell1D = (k * pitchXY) + (j * pitchX) + i;

    if (d_cells[cell1D].flag.vertexType() != MMCellFlag::NoVertex)
    {
        int vid = atomicAdd(d_counter, 1);
        d_vertices[vid].cellIndex[0] = i;
        d_vertices[vid].cellIndex[1] = j;
        d_vertices[vid].cellIndex[2] = k;

        d_cells[cell1D].vertexIndex = vid;
    }
}
__global__ void kernel_count_vertices(Cell* d_cells, int nx, int ny, int nz, int pitchX, int pitchXY, int* d_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nz;
    if (idx >= total) return;

    int k = idx / (nx * ny);
    int j = (idx / nx) % ny;
    int i = idx % nx;

    int cell1D = (k * pitchXY) + (j * pitchX) + i;

    if (d_cells[cell1D].flag.vertexType() != MMCellFlag::NoVertex)
        atomicAdd(d_count, 1);
}

__global__ void kernel_classify_cells( Cell* d_cells, int nx, int ny, int nz, int pitchX, int pitchXY)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nz;
    if (idx >= total) return;

    int k = idx / (nx * ny);
    int j = (idx / nx) % ny;
    int i = idx % nx;

    // i,j,k are interior cells: 0..nx-1, etc.
    int cell1D = k * pitchXY + j * pitchX + i;

    Cell* pCell = &d_cells[cell1D];

    unsigned short lab[8];
    lab[0] = pCell->label;
    lab[1] = (pCell + 1)->label;
    lab[2] = (pCell + 1 + pitchX)->label;
    lab[3] = (pCell + pitchX)->label;
    lab[4] = (pCell + pitchXY)->label;
    lab[5] = (pCell + 1 + pitchXY)->label;
    lab[6] = (pCell + 1 + pitchX + pitchXY)->label;
    lab[7] = (pCell + pitchX + pitchXY)->label;

    // This must be __host__ __device__ and defined in the header
    pCell->flag.set(lab);
}
// Constructor
__host__ MMCellMap::MMCellMap(unsigned short* labels, int arraySize[3], float voxelSize[3]): m_cellArray(nullptr), m_numVertices(0), m_vertices(nullptr),
      d_cells(nullptr), d_vertexOffsets(nullptr), d_vertexCellIdx(nullptr)
{
    // 1) store padded array size and voxel size
    for (int i = 0; i < 3; i++) {
        m_arraySize[i] = arraySize[i] + 2;   // Pad by 1 voxel on each side
        m_voxelSize[i] = voxelSize[i];
    }

    int numCells = m_arraySize[0] * m_arraySize[1] * m_arraySize[2];

    // 2) allocate host cell array
    try {
        m_cellArray = new Cell[numCells];
    } catch (std::bad_alloc&) {
        m_cellArray = nullptr;
        return;
    }

    // 3) fill host cells with labels + padding
    unsigned short padLabel = (unsigned short)MMSurfaceNet::ReservedLabel::Padding;
    Cell*          pCell    = m_cellArray;
    unsigned short* pLabel  = labels;

    for (int k = 0; k < m_arraySize[2]; k++) {
        for (int j = 0; j < m_arraySize[1]; j++) {
            for (int i = 0; i < m_arraySize[0]; i++) {
                if (i == 0 || i == m_arraySize[0] - 1 ||
                    j == 0 || j == m_arraySize[1] - 1 ||
                    k == 0 || k == m_arraySize[2] - 1)
                {
                    // padding voxel
                    initCell(pCell++, padLabel);
                } else {
                    // interior voxel from input labels
                    initCell(pCell++, *pLabel++);
                }
            }
        }
    }

    // 4) create device mirror BEFORE calling setCellVertices()
    size_t totalCells = (size_t)m_arraySize[0] * m_arraySize[1] * m_arraySize[2];

    cudaMalloc(&d_cells, totalCells * sizeof(Cell));
    cudaMemcpy(d_cells, m_cellArray, totalCells * sizeof(Cell), cudaMemcpyHostToDevice);

    // 5) build flags + vertices on GPU (uses d_cells internally)
    setCellVertices();   // fills m_vertices + m_numVertices and updates m_cellArray from d_cells

    // 6) allocate relax buffers only if we actually have vertices
    if (m_numVertices > 0) {
        // d_vertexOffsets
        cudaMalloc(&d_vertexOffsets, m_numVertices * sizeof(float3));

        // d_vertexCellIdx
        cudaMalloc(&d_vertexCellIdx, m_numVertices * sizeof(int));

        // 6a) upload initial offsets from host cells
        float3* h_offsets = (float3*)malloc(m_numVertices * sizeof(float3));
        for (int i = 0; i < m_numVertices; i++) {
            int ci[3];
            getVertexCellIndex(i, ci);             // uses m_vertices (host)
            Cell* c = getCell(ci);                 // host path → m_cellArray
            h_offsets[i] = make_float3(c->vertexOffset[0], c->vertexOffset[1], c->vertexOffset[2]);
        }
        cudaMemcpy(d_vertexOffsets, h_offsets, m_numVertices * sizeof(float3), cudaMemcpyHostToDevice);
        free(h_offsets);

        // 6b) upload vertex→cell index map
        int* hidx = (int*)malloc(m_numVertices * sizeof(int));
        for (int i = 0; i < m_numVertices; i++) {
            int ci[3];
            getVertexCellIndex(i, ci);
            hidx[i] = cellArrayIndex(ci);          // host index in padded grid
        }
        cudaMemcpy(d_vertexCellIdx, hidx, m_numVertices * sizeof(int), cudaMemcpyHostToDevice);
        free(hidx);
    } else {
        // no vertices → keep device relax buffers null
        d_vertexOffsets = nullptr;
        d_vertexCellIdx = nullptr;
    }
}


// Destructor
__host__ MMCellMap::~MMCellMap() {
    if (m_cellArray) {
        delete[] m_cellArray;
    }

    if (m_vertices) {
        delete[] m_vertices;
    }
    if (d_vertexOffsets) cudaFree(d_vertexOffsets);
    if (d_vertexCellIdx) cudaFree(d_vertexCellIdx);
    if (d_cells) cudaFree(d_cells);

}
__device__ inline float3 face_dir(MMCellFlag::Face f) {
    switch (f) {
        case MMCellFlag::LeftFace:   return make_float3(-1,  0,  0);
        case MMCellFlag::RightFace:  return make_float3( 1,  0,  0);
        case MMCellFlag::BackFace:   return make_float3( 0, -1,  0);
        case MMCellFlag::FrontFace:  return make_float3( 0,  1,  0);
        case MMCellFlag::BottomFace: return make_float3( 0,  0, -1);
        case MMCellFlag::TopFace:    return make_float3( 0,  0,  1);
        default:                     return make_float3( 0,  0,  0);
    }
}

// =============================================================
// Red-Black Gauss-Seidel relaxation kernel
// =============================================================
__global__ void kernel_relax_redblack(
    float3*       d_offsets,     // single buffer - in-place updates
    const int*    d_cellIdx,     // per-vertex PADDED linear cell index
    const Cell*   d_cells,       // full padded cell map
    int           numVertices, int3 dims, MMSurfaceNet::RelaxAttrs attrs,
    int           parity)        // 0 = red, 1 = black
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVertices) return;

    // Which padded cell does this vertex live in?
    int cell1D = d_cellIdx[vid];
    int cx = cell1D % dims.x;
    int cy = (cell1D / dims.x) % dims.y;
    int cz = cell1D / (dims.x * dims.y);

    // Only process vertices matching current parity
    if (((cx + cy + cz) & 1) != parity) return;

    const Cell* pCell = &d_cells[cell1D];
    MMCellFlag::VertexType vtype = pCell->flag.vertexType();

    // Accumulate neighbor positions
    float3 avg = make_float3(0.f, 0.f, 0.f);
    int count  = 0;

    const int3 off[6] = {
        {-1,  0,  0},  // Left
        { 1,  0,  0},  // Right
        { 0, -1,  0},  // Back
        { 0,  1,  0},  // Front
        { 0,  0, -1},  // Bottom
        { 0,  0,  1}   // Top
    };

    const MMCellFlag::Face faces[6] = {
        MMCellFlag::LeftFace,   MMCellFlag::RightFace,
        MMCellFlag::BackFace,   MMCellFlag::FrontFace,
        MMCellFlag::BottomFace, MMCellFlag::TopFace
    };

    for (int f = 0; f < 6; ++f) {
        int nx = cx + off[f].x;
        int ny = cy + off[f].y;
        int nz = cz + off[f].z;

        // Stay inside padded grid
        if (nx < 0 || nx >= dims.x || 
            ny < 0 || ny >= dims.y || 
            nz < 0 || nz >= dims.z)
            continue;

        int n1D = nx + dims.x * (ny + dims.y * nz);
        const Cell* nbr = &d_cells[n1D];

        MMCellFlag::FaceCrossingType fct = pCell->flag.faceCrossingType(faces[f]);

        bool useNeighbor = (vtype == MMCellFlag::SurfaceVertex) ? (fct != MMCellFlag::NoFaceCrossing) : (fct == MMCellFlag::JunctionFaceCrossing);

        if (!useNeighbor)
            continue;

        int nid = nbr->vertexIndex;
        if (nid < 0)
            continue;

        // Read from SAME buffer (Gauss-Seidel: may have current-iteration values)
        float3 np = d_offsets[nid];

        avg.x += np.x + float(nx - cx);
        avg.y += np.y + float(ny - cy);
        avg.z += np.z + float(nz - cz);
        ++count;
    }

    float3 p = d_offsets[vid];

    if (count > 0) {
        float inv = 1.0f / float(count);
        avg.x *= inv;
        avg.y *= inv;
        avg.z *= inv;

        float a = attrs.relaxFactor;
        p.x = (1.0f - a) * p.x + a * avg.x;
        p.y = (1.0f - a) * p.y + a * avg.y;
        p.z = (1.0f - a) * p.z + a * avg.z;

        float minc = 0.5f - attrs.maxDistFromCellCenter;
        float maxc = 0.5f + attrs.maxDistFromCellCenter;

        p.x = fminf(fmaxf(p.x, minc), maxc);
        p.y = fminf(fmaxf(p.y, minc), maxc);
        p.z = fminf(fmaxf(p.z, minc), maxc);
    }

    // In-place update
    d_offsets[vid] = p;
}

__global__ void kernel_write_back_offsets( Cell* d_cells, const float3* d_offsets, const int* d_cellIdx, int numVertices)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= numVertices) return;

    int cell1D = d_cellIdx[vid];
    float3 p   = d_offsets[vid];

    d_cells[cell1D].vertexOffset[0] = p.x;
    d_cells[cell1D].vertexOffset[1] = p.y;
    d_cells[cell1D].vertexOffset[2] = p.z;
}

__host__ void MMCellMap::relax(MMSurfaceNet::RelaxAttrs attrs)
{
    if (m_numVertices == 0) return;

    int threads = 256;
    int blocks  = (m_numVertices + threads - 1) / threads;
    int3 dims   = make_int3(m_arraySize[0], m_arraySize[1], m_arraySize[2]);

    // No temporary buffer needed - Red-Black updates in place
    for (int iter = 0; iter < attrs.numRelaxIterations; ++iter) {
        // Red pass (parity 0)
        kernel_relax_redblack<<<blocks, threads>>>( d_vertexOffsets, d_vertexCellIdx, d_cells, m_numVertices, dims, attrs, 0);
        cudaDeviceSynchronize();

        // Black pass (parity 1)
        kernel_relax_redblack<<<blocks, threads>>>( d_vertexOffsets, d_vertexCellIdx, d_cells, m_numVertices, dims, attrs, 1);
        cudaDeviceSynchronize();
    }

    // Write final offsets back into per-cell vertexOffset[]
    kernel_write_back_offsets<<<blocks, threads>>>( d_cells, d_vertexOffsets, d_vertexCellIdx, m_numVertices);
    cudaDeviceSynchronize();

    // Sync device cells → host cells
    size_t totalCells = (size_t)m_arraySize[0] * m_arraySize[1] * m_arraySize[2];
    cudaMemcpy(m_cellArray, d_cells, totalCells * sizeof(Cell), cudaMemcpyDeviceToHost);
}


// __global__ void kernel_write_back_offsets( Cell* d_cells, const float3* d_offsets, const int* d_cellIdx, int numVertices)
// {
//     int vid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (vid >= numVertices) return;

//     int cell1D = d_cellIdx[vid];  // PADDED linear index
//     float3 p   = d_offsets[vid];

//     d_cells[cell1D].vertexOffset[0] = p.x;
//     d_cells[cell1D].vertexOffset[1] = p.y;
//     d_cells[cell1D].vertexOffset[2] = p.z;
// }

// __host__ void MMCellMap::relax(MMSurfaceNet::RelaxAttrs attrs)
// {
//     if (m_numVertices == 0) return;

//     int threads = 256;
//     int blocks  = (m_numVertices + threads - 1) / threads;
//     int3 dims   = make_int3(m_arraySize[0], m_arraySize[1], m_arraySize[2]);

//     // Temporary buffer for Jacobi iterations
//     float3* d_tmp = nullptr;
//     cudaMalloc(&d_tmp, m_numVertices * sizeof(float3));

//     float3* d_in  = d_vertexOffsets; // initial offsets from constructor
//     float3* d_out = d_tmp;

//     for (int iter = 0; iter < attrs.numRelaxIterations; ++iter) {
//         kernel_relax<<<blocks, threads>>>( d_in, d_out, d_vertexCellIdx,  d_cells, m_numVertices, dims, /*0,*/ attrs );
//         cudaDeviceSynchronize();

//         // ping-pong swap
//         float3* tmp = d_in;
//         d_in  = d_out;
//         d_out = tmp;

//         // kernel_relax<<<blocks, threads>>>( d_in, d_out, d_vertexCellIdx,  d_cells, m_numVertices, dims, 1, attrs );
//         // cudaDeviceSynchronize();

//         // // ping-pong swap
//         // tmp = d_in;
//         // d_in  = d_out;
//         // d_out = tmp;
//     }

//     // Make sure d_vertexOffsets holds the final result
//     if (d_in != d_vertexOffsets) {
//         cudaMemcpy(d_vertexOffsets, d_in, m_numVertices * sizeof(float3), cudaMemcpyDeviceToDevice);
//     }

//     cudaFree(d_tmp);

//     // 2) Write final offsets back into per-cell vertexOffset[]
//     kernel_write_back_offsets<<<blocks, threads>>>( d_cells, d_vertexOffsets, d_vertexCellIdx, m_numVertices );
//     cudaDeviceSynchronize();

//     // 3) Sync device cells → host cells so CPU getEdgeQuad() sees smoothed vertices
//     size_t totalCells = (size_t)m_arraySize[0] * m_arraySize[1] * m_arraySize[2];
//     cudaMemcpy(m_cellArray, d_cells, totalCells * sizeof(Cell), cudaMemcpyDeviceToHost);
// }

// // Relax vertex positions
// __host__ void MMCellMap::relax(MMSurfaceNet::RelaxAttrs relaxAttrs) {
//     for (int iter = 0; iter < relaxAttrs.numRelaxIterations; iter++) {
//         for (int idxVtx = 0; idxVtx < m_numVertices; idxVtx++) {
//             int cellIdx[3];
//             getVertexCellIndex(idxVtx, cellIdx);
//             Cell* pCell = getCell(cellIdx);

//             int numNeighbors = 0;
//             float avgP[3] = {0.0f, 0.0f, 0.0f};

//             if (pCell->flag.vertexType() == MMCellFlag::SurfaceVertex) {
//                 for (MMCellFlag::Face face = MMCellFlag::LeftFace; face <= MMCellFlag::TopFace; ++face) {
//                     if (pCell->flag.faceCrossingType(face) != MMCellFlag::NoFaceCrossing) {
//                         int nbrIdx[3];
//                         Cell* nbrCell = getFaceNeighborCellAndIndex(cellIdx, face, nbrIdx);
//                         avgP[0] += nbrCell->vertexOffset[0] + nbrIdx[0] - cellIdx[0];
//                         avgP[1] += nbrCell->vertexOffset[1] + nbrIdx[1] - cellIdx[1];
//                         avgP[2] += nbrCell->vertexOffset[2] + nbrIdx[2] - cellIdx[2];
//                         numNeighbors++;
//                     }
//                 }
//             } else {
//                 for (MMCellFlag::Face face = MMCellFlag::LeftFace; face <= MMCellFlag::TopFace; ++face) {
//                     if (pCell->flag.faceCrossingType(face) == MMCellFlag::JunctionFaceCrossing) {
//                         int nbrIdx[3];
//                         Cell* nbrCell = getFaceNeighborCellAndIndex(cellIdx, face, nbrIdx);
//                         avgP[0] += nbrCell->vertexOffset[0] + nbrIdx[0] - cellIdx[0];
//                         avgP[1] += nbrCell->vertexOffset[1] + nbrIdx[1] - cellIdx[1];
//                         avgP[2] += nbrCell->vertexOffset[2] + nbrIdx[2] - cellIdx[2];
//                         numNeighbors++;
//                     }
//                 }
//             }
//             float* p = pCell->vertexOffset;
//             if (numNeighbors > 0) {
//                 avgP[0] /= (float)numNeighbors;
//                 avgP[1] /= (float)numNeighbors;
//                 avgP[2] /= (float)numNeighbors;
//                 float alpha = relaxAttrs.relaxFactor;
//                 p[0] = (1.0f - alpha) * p[0] + alpha * avgP[0];
//                 p[1] = (1.0f - alpha) * p[1] + alpha * avgP[1];
//                 p[2] = (1.0f - alpha) * p[2] + alpha * avgP[2];

//                 // Constrain vertex location to a max distance from the original voxel
//                 float min = 0.5f - relaxAttrs.maxDistFromCellCenter;
//                 float max = 0.5f + relaxAttrs.maxDistFromCellCenter;
//                 if (p[0] < min) p[0] = min;
//                 if (p[0] > max) p[0] = max;
//                 if (p[1] < min) p[1] = min;
//                 if (p[1] > max) p[1] = max;
//                 if (p[2] < min) p[2] = min;
//                 if (p[2] > max) p[2] = max;
//             }
//         }
//     }
// }

// Reset all vertices to cell centers
__host__ void MMCellMap::reset() {
    for (int idxVtx = 0; idxVtx < m_numVertices; idxVtx++) {
        int cellIdx[3];
        getVertexCellIndex(idxVtx, cellIdx);
        Cell* pCell = getCell(cellIdx);
        pCell->vertexOffset[0] = 0.5f;
        pCell->vertexOffset[1] = 0.5f;
        pCell->vertexOffset[2] = 0.5f;
    }
}

// Initialize a cell
__host__ __device__ void MMCellMap::initCell(Cell* cell, unsigned short label) {
    cell->label = label;
    cell->flag.clear();
    cell->vertexIndex = -1;
    cell->vertexOffset[0] = 0.5f;
    cell->vertexOffset[1] = 0.5f;
    cell->vertexOffset[2] = 0.5f;
}
__host__ void MMCellMap::setCellVertices() {
    int dimx = m_arraySize[0];
    int dimy = m_arraySize[1];
    int dimz = m_arraySize[2];

    int nx = dimx - 1;
    int ny = dimy - 1;
    int nz = dimz - 1;

    int totalInterior = nx * ny * nz;
    if (totalInterior <= 0) {
        m_numVertices = 0;
        if (m_vertices) {
            delete[] m_vertices;
            m_vertices = nullptr;
        }
        return;
    }

    int pitchX  = dimx;
    int pitchXY = dimx * dimy;

    int threads = 256;
    int blocks  = (totalInterior + threads - 1) / threads;

    // 1) Classify cells on device (compute flags)
    kernel_classify_cells<<<blocks, threads>>>(d_cells, nx, ny, nz, pitchX, pitchXY);
    cudaDeviceSynchronize();

    // 2) Count vertices
    int* d_count = nullptr;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));

    kernel_count_vertices<<<blocks, threads>>>(d_cells, nx, ny, nz, pitchX, pitchXY, d_count);
    cudaDeviceSynchronize();

    cudaMemcpy(&m_numVertices, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_count);

    // If no vertices, still need flags on host
    if (m_numVertices == 0) {
        size_t totalCells = (size_t)dimx * dimy * dimz;
        cudaMemcpy(m_cellArray, d_cells, totalCells * sizeof(Cell), cudaMemcpyDeviceToHost);
        if (m_vertices) {
            delete[] m_vertices;
            m_vertices = nullptr;
        }
        return;
    }

    // 3) Allocate host + device vertex arrays
    if (m_vertices) {
        delete[] m_vertices;
    }
    m_vertices = new Vertex[m_numVertices];

    MMCellMap::Vertex* d_vertices = nullptr;
    cudaMalloc(&d_vertices, m_numVertices * sizeof(MMCellMap::Vertex));

    int* d_counter = nullptr;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));

    // 4) Assign vertices & vertexIndex into d_cells
    kernel_assign_vertices<<<blocks, threads>>>(d_cells, nx, ny, nz, pitchX, pitchXY, d_vertices, d_counter);
    cudaDeviceSynchronize();

    cudaFree(d_counter);

    // 5) Copy vertex list back to host
    cudaMemcpy(m_vertices, d_vertices, m_numVertices * sizeof(Vertex), cudaMemcpyDeviceToHost);
    cudaFree(d_vertices);

    // 6) Sync cell flags + vertexIndex back to host for CPU use
    size_t totalCells = (size_t)dimx * dimy * dimz;
    cudaMemcpy(m_cellArray, d_cells, totalCells * sizeof(Cell), cudaMemcpyDeviceToHost);
}




// __host__  void MMCellMap::setCellVertices()
// {
//     m_numVertices = 0;

//     for (int k = 0; k < m_arraySize[2] - 1; ++k) {
//         for (int j = 0; j < m_arraySize[1] - 1; ++j) {
//             for (int i = 0; i < m_arraySize[0] - 1; ++i) {
//                 Cell* pCell = getCell(i, j, k);
//                 unsigned short cellLabels[8];
//                 getCellLabels(pCell, cellLabels);

//                 int idx1d = i + j * (m_arraySize[0] - 1) +
//                             k * (m_arraySize[0] - 1) * (m_arraySize[1] - 1);
//                 pCell->flag.set(cellLabels, idx1d);

//                 if (pCell->flag.vertexType() != MMCellFlag::NoVertex)
//                     m_numVertices++;
//             }
//         }
//     }

//     // Free any existing vertex buffer
//     if (m_vertices) {
//         free(m_vertices);
//         m_vertices = nullptr;
//     }

//     if (m_numVertices == 0) return;

//     m_vertices = (Vertex*)calloc(m_numVertices, sizeof(Vertex));
//     if (!m_vertices) {
//         free(m_cellArray);
//         m_cellArray = nullptr;
//         m_numVertices = 0;
//         return;
//     }

//     int vtxIdx = 0;
//     for (int k = 0; k < m_arraySize[2] - 1; ++k) {
//         for (int j = 0; j < m_arraySize[1] - 1; ++j) {
//             for (int i = 0; i < m_arraySize[0] - 1; ++i) {
//                 Cell* pCell = getCell(i, j, k);
//                 if (pCell->flag.vertexType() != MMCellFlag::NoVertex) {
//                     pCell->vertexIndex = vtxIdx;
//                     Vertex* v = &m_vertices[vtxIdx++];
//                     v->cellIndex[0] = i;
//                     v->cellIndex[1] = j;
//                     v->cellIndex[2] = k;
//                 } else {
//                     pCell->vertexIndex = -1;
//                 }
//             }
//         }
//     }
// }

// __host__ __device__ Cell* MMCellMap::getCell(int cellArrayIndex) const {
//     //return &(m_cellArray[cellArrayIndex]);
//     #ifdef __CUDA_ARCH__
//     return &d_cells[cellArrayIndex];     // GPU path
//     #else
//         return &m_cellArray[cellArrayIndex]; // CPU path
//     #endif
// }
// __host__ __device__ Cell* MMCellMap::getCell(int i, int j, int k) const {
//     int idx = cellArrayIndex(i,j,k);
//     #ifdef __CUDA_ARCH__
//         return &d_cells[idx];
//     #else
//         return &m_cellArray[idx];
//     #endif
// }

// __host__ __device__ Cell* MMCellMap::getCell(int cellIndex[3]) const {
    
//     return getCell(cellArrayIndex(cellIndex));
// }

// __host__ __device__ int MMCellMap::cellArrayIndex(int cellIndex[3]) const {
//     return cellArrayIndex(cellIndex[0], cellIndex[1], cellIndex[2]);
// }
// __host__ __device__ int MMCellMap::cellArrayIndex(int i, int j, int k) const {
//     return i + m_arraySize[0] * j + m_arraySize[0] * m_arraySize[1] * k;
// }

__host__ __device__ void MMCellMap::getCellLabels(Cell* pCell, unsigned short labels[8]) {
    labels[0] = pCell->label;
    labels[1] = (pCell + 1)->label;
    labels[2] = (pCell + 1 + m_arraySize[0])->label;
    labels[3] = (pCell + m_arraySize[0])->label;
    labels[4] = (pCell + m_arraySize[0] * m_arraySize[1])->label;
    labels[5] = (pCell + 1 + m_arraySize[0] * m_arraySize[1])->label;
    labels[6] = (pCell + 1 + m_arraySize[0] + m_arraySize[0] * m_arraySize[1])->label;
    labels[7] = (pCell + m_arraySize[0] + m_arraySize[0] * m_arraySize[1])->label;
}

// __host__ __device__ bool MMCellMap::isEdgeCrossing(int cellArrayIndex, MMCellFlag::Edge edge) const {
//     Cell* pCell = getCell(cellArrayIndex);
//     return pCell->flag.isEdgeCrossing(edge);
// }

// __host__ __device__ MMCellFlag::VertexType MMCellMap::vertexType(int vertexIndex) const {
//     int cellIndex[3];
//     getVertexCellIndex(vertexIndex, cellIndex);
//     return (cellVertexType(cellArrayIndex(cellIndex)));
// }
// __host__ __device__ void MMCellMap::getVertexPosition(int vertexIndex, float position[3]) const {
// 	getVertexPosition(m_vertices[vertexIndex].cellIndex, position);
// }
// __host__ __device__ MMCellFlag::VertexType MMCellMap::cellVertexType(int cellMapIndex) const {
//     Cell*  pCell = getCell(cellMapIndex);
//     return (pCell->flag.vertexType());
// }

// __host__ __device__ void MMCellMap::getVertexCellIndex(int vertexIndex, int cellIndex[3]) const {
//     Vertex* pVertex = &(m_vertices[vertexIndex]);
//     cellIndex[0] = pVertex->cellIndex[0];
//     cellIndex[1] = pVertex->cellIndex[1];
//     cellIndex[2] = pVertex->cellIndex[2];
// }

// __host__ __device__ void MMCellMap::getVertexPosition(int cellIndex[3], float position[3]) const {
//     Cell* pCell = getCell(cellArrayIndex(cellIndex));
//     position[0] = m_voxelSize[0] * (cellIndex[0] + pCell->vertexOffset[0]);
//     position[1] = m_voxelSize[1] * (cellIndex[1] + pCell->vertexOffset[1]);
//     position[2] = m_voxelSize[2] * (cellIndex[2] + pCell->vertexOffset[2]);
// }

// __host__ __device__ void MMCellMap::getVertexPosition(int i, int j, int k, float position[3]) const {
//     Cell* pCell = getCell(i, j, k);
//     position[0] = m_voxelSize[0] * (i + pCell->vertexOffset[0]);
//     position[1] = m_voxelSize[1] * (j + pCell->vertexOffset[1]);
//     position[2] = m_voxelSize[2] * (k + pCell->vertexOffset[2]);
// }

__host__ __device__ int MMCellMap::vertexFaceNeighborVertexIndex(int vertexIndex, MMCellFlag::Face face) const {
    int cellMapIndex = cellArrayIndex(m_vertices[vertexIndex].cellIndex);

    switch (face) {
        case MMCellFlag::LeftFace:
            return cellMapIndex - 1;

        case MMCellFlag::RightFace:
            return cellMapIndex + 1;

        case MMCellFlag::BackFace:
            return cellMapIndex - m_arraySize[0];

        case MMCellFlag::FrontFace:
            return cellMapIndex + m_arraySize[0];

        case MMCellFlag::BottomFace:
            return cellMapIndex - m_arraySize[0] * m_arraySize[1];

        case MMCellFlag::TopFace:
            return cellMapIndex + m_arraySize[0] * m_arraySize[1];

        default:
            return cellMapIndex;
    }
}

__host__ __device__ Cell* MMCellMap::getFaceNeighborCellAndIndex(int cellIndex[3], MMCellFlag::Face face, int nbrCellIndex[3]) {
    nbrCellIndex[0] = cellIndex[0];
    nbrCellIndex[1] = cellIndex[1];
    nbrCellIndex[2] = cellIndex[2];

    switch (face) {
        case MMCellFlag::LeftFace:
            nbrCellIndex[0] -= 1;
            break;

        case MMCellFlag::RightFace:
            nbrCellIndex[0] += 1;
            break;

        case MMCellFlag::BackFace:
            nbrCellIndex[1] -= 1;
            break;

        case MMCellFlag::FrontFace:
            nbrCellIndex[1] += 1;
            break;

        case MMCellFlag::BottomFace:
            nbrCellIndex[2] -= 1;
            break;

        case MMCellFlag::TopFace:
            nbrCellIndex[2] += 1;
            break;

        default:
            break;
    }

    return &(m_cellArray[cellArrayIndex(nbrCellIndex)]);
}

// ==========================================================
// Edge quad methods
// ==========================================================

// __host__ __device__ void MMCellMap::getEdgeLabels(int cellIndex[3], MMCellFlag::Edge edge, unsigned short quadLabels[2]) {
//     Cell* pCell = getCell(cellIndex);
//     Cell* pCellFirstLabel, *pCellSecondLabel;

//     switch (edge) {
//         case MMCellFlag::LeftBottomEdge:
//             pCellFirstLabel = pCell;
//             pCellSecondLabel = pCell + m_arraySize[0];
//             break;

//         case MMCellFlag::RightBottomEdge:
//             pCellFirstLabel = pCell + 1;
//             pCellSecondLabel = pCell + 1 + m_arraySize[0];
//             break;

//         case MMCellFlag::BackBottomEdge:
//             pCellFirstLabel = pCell;
//             pCellSecondLabel = pCell + 1;
//             break;

//         case MMCellFlag::FrontBottomEdge:
//             pCellFirstLabel = pCell + m_arraySize[0];
//             pCellSecondLabel = pCell + 1 + m_arraySize[0];
//             break;

//         case MMCellFlag::LeftTopEdge:
//             pCellFirstLabel = pCell + m_arraySize[0] * m_arraySize[1];
//             pCellSecondLabel = pCell + m_arraySize[0] + m_arraySize[0] * m_arraySize[1];
//             break;

//         case MMCellFlag::RightTopEdge:
//             pCellFirstLabel = pCell + 1 + m_arraySize[0] * m_arraySize[1];
//             pCellSecondLabel = pCell + 1 + m_arraySize[0] + m_arraySize[0] * m_arraySize[1];
//             break;

//         case MMCellFlag::BackTopEdge:
//             pCellFirstLabel = pCell + m_arraySize[0] * m_arraySize[1];
//             pCellSecondLabel = pCell + 1 + m_arraySize[0] * m_arraySize[1];
//             break;

//         case MMCellFlag::FrontTopEdge:
//             pCellFirstLabel = pCell + m_arraySize[0] + m_arraySize[0] * m_arraySize[1];
//             pCellSecondLabel = pCell + 1 + m_arraySize[0] + m_arraySize[0] * m_arraySize[1];
//             break;

//         case MMCellFlag::LeftBackEdge:
//             pCellFirstLabel = pCell;
//             pCellSecondLabel = pCell + m_arraySize[0] * m_arraySize[1];
//             break;

//         case MMCellFlag::RightBackEdge:
//             pCellFirstLabel = pCell + 1;
//             pCellSecondLabel = pCell + 1 + m_arraySize[0] * m_arraySize[1];
//             break;

//         case MMCellFlag::LeftFrontEdge:
//             pCellFirstLabel = pCell + m_arraySize[0];
//             pCellSecondLabel = pCell + m_arraySize[0] + m_arraySize[0] * m_arraySize[1];
//             break;

//         case MMCellFlag::RightFrontEdge:
//             pCellFirstLabel = pCell + 1 + m_arraySize[0];
//             pCellSecondLabel = pCell + 1 + m_arraySize[0] + m_arraySize[0] * m_arraySize[1];
//             break;

//         default:
//             pCellFirstLabel = pCell;
//             pCellSecondLabel = pCell;
//             break;
//     }

//     quadLabels[0] = pCellFirstLabel->label;
//     quadLabels[1] = pCellSecondLabel->label;
// }

// __host__ __device__ void MMCellMap::getEdgeQuadVtxIndices(int cellIndex[3], MMCellFlag::Edge edge, int quadVtxIndices[4]) {
//     Cell* pCell = getCell(cellIndex);
//     int length = m_arraySize[0];
//     int area   = m_arraySize[0] * m_arraySize[1];
//     quadVtxIndices[0] = pCell->vertexIndex;

//     switch (edge) {
//         case MMCellFlag::LeftBottomEdge:
//             quadVtxIndices[1] = (pCell - area)->vertexIndex;
//             quadVtxIndices[2] = (pCell - 1 - area)->vertexIndex;
//             quadVtxIndices[3] = (pCell - 1)->vertexIndex;
//             break;

//         case MMCellFlag::RightBottomEdge:
//             quadVtxIndices[1] = (pCell + 1)->vertexIndex;
//             quadVtxIndices[2] = (pCell + 1 - area)->vertexIndex;
//             quadVtxIndices[3] = (pCell - area)->vertexIndex;
//             break;

//         case MMCellFlag::BackBottomEdge:
//             quadVtxIndices[1] = (pCell - length)->vertexIndex;
//             quadVtxIndices[2] = (pCell - length - area)->vertexIndex;
//             quadVtxIndices[3] = (pCell - area)->vertexIndex;
//             break;

//         case MMCellFlag::FrontBottomEdge:
//             quadVtxIndices[1] = (pCell - area)->vertexIndex;
//             quadVtxIndices[2] = (pCell + length - area)->vertexIndex;
//             quadVtxIndices[3] = (pCell + length)->vertexIndex;
//             break;

//         case MMCellFlag::LeftTopEdge:
//             quadVtxIndices[1] = (pCell - 1)->vertexIndex;
//             quadVtxIndices[2] = (pCell - 1 + area)->vertexIndex;
//             quadVtxIndices[3] = (pCell + area)->vertexIndex;
//             break;

//         case MMCellFlag::RightTopEdge:
//             quadVtxIndices[1] = (pCell + area)->vertexIndex;
//             quadVtxIndices[2] = (pCell + 1 + area)->vertexIndex;
//             quadVtxIndices[3] = (pCell + 1)->vertexIndex;
//             break;

//         case MMCellFlag::BackTopEdge:
//             quadVtxIndices[1] = (pCell + area)->vertexIndex;
//             quadVtxIndices[2] = (pCell - length + area)->vertexIndex;
//             quadVtxIndices[3] = (pCell - length)->vertexIndex;
//             break;

//         case MMCellFlag::FrontTopEdge:
//             quadVtxIndices[1] = (pCell + length)->vertexIndex;
//             quadVtxIndices[2] = (pCell + length + area)->vertexIndex;
//             quadVtxIndices[3] = (pCell + area)->vertexIndex;
//             break;

//         case MMCellFlag::LeftBackEdge:
//             quadVtxIndices[1] = (pCell - 1)->vertexIndex;
//             quadVtxIndices[2] = (pCell - 1 - length)->vertexIndex;
//             quadVtxIndices[3] = (pCell - length)->vertexIndex;
//             break;

//         case MMCellFlag::RightBackEdge:
//             quadVtxIndices[1] = (pCell - length)->vertexIndex;
//             quadVtxIndices[2] = (pCell + 1 - length)->vertexIndex;
//             quadVtxIndices[3] = (pCell + 1)->vertexIndex;
//             break;

//         case MMCellFlag::LeftFrontEdge:
//             quadVtxIndices[1] = (pCell + length)->vertexIndex;
//             quadVtxIndices[2] = (pCell - 1 + length)->vertexIndex;
//             quadVtxIndices[3] = (pCell - 1)->vertexIndex;
//             break;

//         case MMCellFlag::RightFrontEdge:
//             quadVtxIndices[1] = (pCell + 1)->vertexIndex;
//             quadVtxIndices[2] = (pCell + 1 + length)->vertexIndex;
//             quadVtxIndices[3] = (pCell + length)->vertexIndex;
//             break;

//         default:
//             quadVtxIndices[1] = pCell->vertexIndex;
//             quadVtxIndices[2] = pCell->vertexIndex;
//             quadVtxIndices[3] = pCell->vertexIndex;
//             break;
//     }
// }

// __host__ __device__ void MMCellMap::getEdgeQuadPositions(int cellIndex[3], MMCellFlag::Edge edge, float quadCorners[12]) {
//     int vtxIndices[4];
//     getEdgeQuadVtxIndices(cellIndex, edge, vtxIndices);

//     for (int i = 0; i < 4; i++) {
//         int ci[3];
//         getVertexCellIndex(vtxIndices[i], ci);
//         getVertexPosition(ci, &(quadCorners[i * 3]));
//     }
// }

// __host__ __device__ inline bool MMCellMap::getEdgeQuad(int vertexIndex, MMCellFlag::Edge edge, float quadCorners[12], unsigned short quadLabels[2]) {
//     int cellIndex[3];
//     getVertexCellIndex(vertexIndex, cellIndex);

//     if (!isEdgeCrossing(cellArrayIndex(cellIndex), edge)) {
//         return false;
//     }

//     getEdgeLabels(cellIndex, edge, quadLabels);
//     getEdgeQuadPositions(cellIndex, edge, quadCorners);
//     return true;
// }


// __host__ __device__ inline bool MMCellMap::getEdgeQuad(int vertexIndex, MMCellFlag::Edge edge, int quadVtxIndices[4], unsigned short quadLabels[2]) {
//     int cellIndex[3];
//     getVertexCellIndex(vertexIndex, cellIndex);

//     if (!isEdgeCrossing(cellArrayIndex(cellIndex), edge)) {
//         return false;
//     }

//     getEdgeLabels(cellIndex, edge, quadLabels);
//     getEdgeQuadVtxIndices(cellIndex, edge, quadVtxIndices);
//     return true;
// }

__host__ __device__ int MMCellMap::numEdgeCrossings() const {
    int numCrossings = 0;
    int cellMapIdx = 0;

    for (int k = 0; k < m_arraySize[2]; k++) {
        for (int j = 0; j < m_arraySize[1]; j++) {
            for (int i = 0; i < m_arraySize[0]; i++) {
                if (isEdgeCrossing(cellMapIdx, MMCellFlag::LeftBackEdge)) numCrossings++;
                if (isEdgeCrossing(cellMapIdx, MMCellFlag::LeftBottomEdge)) numCrossings++;
                if (isEdgeCrossing(cellMapIdx, MMCellFlag::BackBottomEdge)) numCrossings++;
                cellMapIdx++;
            }
        }
    }
    return numCrossings;
}

