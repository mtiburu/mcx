// ==========================================================
// File: mcx_sn_cellmap.cu
// GPU implementation of MMCellMap methods
// ==========================================================
#include <cub/cub.cuh>
#include "mcx_sn_cellmap.cuh"
#include "mcx_sn_cellflag.cuh"

// =============================================================
// Vertex enumeration via prefix scan (Bug 4 fix)
//
// Replaces the non-deterministic atomicAdd-based scheme with a
// flag -> exclusive-scan -> scatter pipeline so vertex IDs are
// assigned in scanline (k, j, i) order matching the CPU reference.
// =============================================================

__global__ void kernel_mark_vertex_flags(const Cell* d_cells, int nx, int ny, int nz,
        int pitchX, int pitchXY, int* d_flags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nz;

    if (idx >= total) {
        return;
    }

    int k = idx / (nx * ny);
    int j = (idx / nx) % ny;
    int i = idx % nx;

    int cell1D = (k * pitchXY) + (j * pitchX) + i;
    d_flags[idx] = (d_cells[cell1D].flag.vertexType() != MMCellFlag::NoVertex) ? 1 : 0;
}

__global__ void kernel_scatter_vertices(Cell* d_cells, int nx, int ny, int nz,
                                        int pitchX, int pitchXY,
                                        const int* d_flags, const int* d_scan,
                                        MMCellMap::Vertex* d_vertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nz;

    if (idx >= total) {
        return;
    }

    if (d_flags[idx] == 0) {
        return;
    }

    int k = idx / (nx * ny);
    int j = (idx / nx) % ny;
    int i = idx % nx;

    int cell1D = (k * pitchXY) + (j * pitchX) + i;
    int vid    = d_scan[idx];                 // deterministic, scanline-ordered

    d_vertices[vid].cellIndex[0] = i;
    d_vertices[vid].cellIndex[1] = j;
    d_vertices[vid].cellIndex[2] = k;

    d_cells[cell1D].vertexIndex = vid;
}

__global__ void kernel_classify_cells( Cell* d_cells, int nx, int ny, int nz, int pitchX, int pitchXY) {
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nz;

    if (idx >= total) {
        return;
    }

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
    d_cells(nullptr), d_vertices(nullptr), d_vertexOffsets(nullptr), d_vertexCellIdx(nullptr) {
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
                        k == 0 || k == m_arraySize[2] - 1) {
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

    if (d_vertexOffsets) {
        cudaFree(d_vertexOffsets);
    }

    if (d_vertexCellIdx) {
        cudaFree(d_vertexCellIdx);
    }

    if (d_cells) {
        cudaFree(d_cells);
    }

    if (d_vertices) {
        cudaFree(d_vertices);
    }
}
__device__ inline float3 face_dir(MMCellFlag::Face f) {
    switch (f) {
        case MMCellFlag::LeftFace:
            return make_float3(-1,  0,  0);

        case MMCellFlag::RightFace:
            return make_float3( 1,  0,  0);

        case MMCellFlag::BackFace:
            return make_float3( 0, -1,  0);

        case MMCellFlag::FrontFace:
            return make_float3( 0,  1,  0);

        case MMCellFlag::BottomFace:
            return make_float3( 0,  0, -1);

        case MMCellFlag::TopFace:
            return make_float3( 0,  0,  1);

        default:
            return make_float3( 0,  0,  0);
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
    int           parity) {      // 0 = red, 1 = black
    int vid = blockIdx.x * blockDim.x + threadIdx.x;

    if (vid >= numVertices) {
        return;
    }

    // Which padded cell does this vertex live in?
    int cell1D = d_cellIdx[vid];
    int cx = cell1D % dims.x;
    int cy = (cell1D / dims.x) % dims.y;
    int cz = cell1D / (dims.x * dims.y);

    // Only process vertices matching current parity
    if (((cx + cy + cz) & 1) != parity) {
        return;
    }

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
                nz < 0 || nz >= dims.z) {
            continue;
        }

        int n1D = nx + dims.x * (ny + dims.y * nz);
        const Cell* nbr = &d_cells[n1D];

        MMCellFlag::FaceCrossingType fct = pCell->flag.faceCrossingType(faces[f]);

        bool useNeighbor = (vtype == MMCellFlag::SurfaceVertex) ? (fct != MMCellFlag::NoFaceCrossing) : (fct == MMCellFlag::JunctionFaceCrossing);

        if (!useNeighbor) {
            continue;
        }

        int nid = nbr->vertexIndex;

        if (nid < 0) {
            continue;
        }

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

__global__ void kernel_write_back_offsets( Cell* d_cells, const float3* d_offsets, const int* d_cellIdx, int numVertices) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;

    if (vid >= numVertices) {
        return;
    }

    int cell1D = d_cellIdx[vid];
    float3 p   = d_offsets[vid];

    d_cells[cell1D].vertexOffset[0] = p.x;
    d_cells[cell1D].vertexOffset[1] = p.y;
    d_cells[cell1D].vertexOffset[2] = p.z;
}

__host__ void MMCellMap::relax(MMSurfaceNet::RelaxAttrs attrs) {
    if (m_numVertices == 0) {
        return;
    }

    int threads = 256;
    int blocks  = (m_numVertices + threads - 1) / threads;
    int3 dims   = make_int3(m_arraySize[0], m_arraySize[1], m_arraySize[2]);

    // No temporary buffer needed - Red-Black updates in place
    for (int iter = 0; iter < attrs.numRelaxIterations; ++iter) {
        // Red pass (parity 0)
        kernel_relax_redblack <<< blocks, threads>>>( d_vertexOffsets, d_vertexCellIdx, d_cells, m_numVertices, dims, attrs, 0);
        cudaDeviceSynchronize();

        // Black pass (parity 1)
        kernel_relax_redblack <<< blocks, threads>>>( d_vertexOffsets, d_vertexCellIdx, d_cells, m_numVertices, dims, attrs, 1);
        cudaDeviceSynchronize();
    }

    // Write final offsets back into per-cell vertexOffset[]
    kernel_write_back_offsets <<< blocks, threads>>>( d_cells, d_vertexOffsets, d_vertexCellIdx, m_numVertices);
    cudaDeviceSynchronize();

    // Sync device cells → host cells
    size_t totalCells = (size_t)m_arraySize[0] * m_arraySize[1] * m_arraySize[2];
    cudaMemcpy(m_cellArray, d_cells, totalCells * sizeof(Cell), cudaMemcpyDeviceToHost);
}

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

        if (d_vertices) {
            cudaFree(d_vertices);
            d_vertices = nullptr;
        }

        return;
    }

    int pitchX  = dimx;
    int pitchXY = dimx * dimy;

    int threads = 256;
    int blocks  = (totalInterior + threads - 1) / threads;

    // 1) Classify cells on device (compute flags)
    kernel_classify_cells <<< blocks, threads>>>(d_cells, nx, ny, nz, pitchX, pitchXY);
    cudaDeviceSynchronize();

    // 2) Mark per-cell vertex flags (1 if cell has a vertex, 0 otherwise)
    int* d_flags = nullptr;
    int* d_scan  = nullptr;
    cudaMalloc(&d_flags, totalInterior * sizeof(int));
    cudaMalloc(&d_scan,  totalInterior * sizeof(int));

    kernel_mark_vertex_flags <<< blocks, threads>>>(d_cells, nx, ny, nz, pitchX, pitchXY, d_flags);
    cudaDeviceSynchronize();

    // 3) Exclusive scan -> vertex IDs in scanline order
    void*  d_temp     = nullptr;
    size_t d_temp_sz  = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, d_temp_sz, d_flags, d_scan, totalInterior);
    cudaMalloc(&d_temp, d_temp_sz);
    cub::DeviceScan::ExclusiveSum(d_temp, d_temp_sz, d_flags, d_scan, totalInterior);
    cudaFree(d_temp);

    // 4) Total count = (last flag) + (last exclusive-scan result)
    int last_flag = 0, last_scan = 0;
    cudaMemcpy(&last_flag, d_flags + (totalInterior - 1), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_scan, d_scan  + (totalInterior - 1), sizeof(int), cudaMemcpyDeviceToHost);
    m_numVertices = last_flag + last_scan;

    if (m_numVertices == 0) {
        cudaFree(d_flags);
        cudaFree(d_scan);
        size_t totalCells = (size_t)dimx * dimy * dimz;
        cudaMemcpy(m_cellArray, d_cells, totalCells * sizeof(Cell), cudaMemcpyDeviceToHost);

        if (m_vertices) {
            delete[] m_vertices;
            m_vertices = nullptr;
        }

        if (d_vertices) {
            cudaFree(d_vertices);
            d_vertices = nullptr;
        }

        return;
    }

    // 5) Allocate persistent host + device vertex arrays
    if (m_vertices) {
        delete[] m_vertices;
    }

    m_vertices = new Vertex[m_numVertices];

    if (d_vertices) {
        cudaFree(d_vertices);
    }

    cudaMalloc(&d_vertices, m_numVertices * sizeof(Vertex));

    // 6) Scatter: each cell with flag=1 writes its (i,j,k) to d_vertices[d_scan[idx]]
    //    and writes vertexIndex back into d_cells[cell1D]
    kernel_scatter_vertices <<< blocks, threads>>>(d_cells, nx, ny, nz, pitchX, pitchXY,
            d_flags, d_scan, d_vertices);
    cudaDeviceSynchronize();

    cudaFree(d_flags);
    cudaFree(d_scan);

    // 7) Copy vertex list back to host (used by host-side getVertexCellIndex etc.)
    cudaMemcpy(m_vertices, d_vertices, m_numVertices * sizeof(Vertex), cudaMemcpyDeviceToHost);

    // 8) Sync cell flags + vertexIndex back to host for CPU use
    size_t totalCells = (size_t)dimx * dimy * dimz;
    cudaMemcpy(m_cellArray, d_cells, totalCells * sizeof(Cell), cudaMemcpyDeviceToHost);
}

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

__host__ __device__ int MMCellMap::vertexFaceNeighborVertexIndex(int vertexIndex, MMCellFlag::Face face) const {
    int ci[3];
    getVertexCellIndex(vertexIndex, ci);
    int cellMapIndex = cellArrayIndex(ci);

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

    return getCell(nbrCellIndex);
}

// ==========================================================
// Edge quad methods
// ==========================================================
__host__ __device__ int MMCellMap::numEdgeCrossings() const {
    int numCrossings = 0;
    int cellMapIdx = 0;

    for (int k = 0; k < m_arraySize[2]; k++) {
        for (int j = 0; j < m_arraySize[1]; j++) {
            for (int i = 0; i < m_arraySize[0]; i++) {
                if (isEdgeCrossing(cellMapIdx, MMCellFlag::LeftBackEdge)) {
                    numCrossings++;
                }

                if (isEdgeCrossing(cellMapIdx, MMCellFlag::LeftBottomEdge)) {
                    numCrossings++;
                }

                if (isEdgeCrossing(cellMapIdx, MMCellFlag::BackBottomEdge)) {
                    numCrossings++;
                }

                cellMapIdx++;
            }
        }
    }

    return numCrossings;
}

