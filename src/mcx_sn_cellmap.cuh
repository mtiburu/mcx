// ==========================================================
// File: mcx_sn_cellmap.cuh
// GPU-friendly version of MMCellMap (header only)
// ==========================================================

#ifndef MCX_SN_CELLMAP_CUH
#define MCX_SN_CELLMAP_CUH

#include <new>
#include <cuda_runtime.h>
#include "mcx_surfacenets.cuh"
#include "mcx_sn_cellflag.cuh"

struct Cell {
    unsigned short label;
    MMCellFlag flag;
    int vertexIndex;
    float vertexOffset[3];
};

// CUDA-friendly MMCellMap
struct MMCellMap {
    // Constructor & destructor (to be implemented in .cu)
    __host__ MMCellMap(unsigned short* labels, int arraySize[3], float voxelSize[3]);
    __host__ ~MMCellMap();

    // Relaxation and reset
    __host__ void relax(MMSurfaceNet::RelaxAttrs relaxAttrs); // .cu
    __host__ void reset();                                   // .cu

    // Export array size / voxel size
    __host__ __device__ inline void getArraySize(int arraySize[3]) const {
        arraySize[0] = m_arraySize[0];
        arraySize[1] = m_arraySize[1];
        arraySize[2] = m_arraySize[2];
    }
    __host__ __device__ inline void getVoxelSize(float voxelSize[3]) const {
        voxelSize[0] = m_voxelSize[0];
        voxelSize[1] = m_voxelSize[1];
        voxelSize[2] = m_voxelSize[2];
    }

    // Vertex/edge/quad queries (to be implemented in .cu)
    __host__ __device__ int numVertices() const {
        return m_numVertices;
    }
    __host__ __device__ int numEdgeCrossings() const;
    __host__ __device__ __forceinline__ MMCellFlag::VertexType vertexType(int vertexIndex) const {
        int cellIndex[3];
        getVertexCellIndex(vertexIndex, cellIndex);
        return cellVertexType(cellArrayIndex(cellIndex));
    }

    __host__ __device__ inline bool getEdgeQuad(int vertexIndex, MMCellFlag::Edge edge, float quadCorners[12], unsigned short quadLabels[2]) {
        int cellIndex[3];
        getVertexCellIndex(vertexIndex, cellIndex);

        if (!isEdgeCrossing(cellArrayIndex(cellIndex), edge)) {
            return false;
        }

        getEdgeLabels(cellIndex, edge, quadLabels);
        getEdgeQuadPositions(cellIndex, edge, quadCorners);
        return true;
    }


    __host__ __device__ inline bool getEdgeQuad(int vertexIndex, MMCellFlag::Edge edge, int quadVtxIndices[4], unsigned short quadLabels[2]) {
        int cellIndex[3];
        getVertexCellIndex(vertexIndex, cellIndex);

        if (!isEdgeCrossing(cellArrayIndex(cellIndex), edge)) {
            return false;
        }

        getEdgeLabels(cellIndex, edge, quadLabels);
        getEdgeQuadVtxIndices(cellIndex, edge, quadVtxIndices);
        return true;
    }

    struct Vertex {
        int cellIndex[3];
    };

    __host__ __device__ __forceinline__ void getVertexCellIndex(int vertexIndex, int cellIndex[3]) const {
#ifdef __CUDA_ARCH__
        const Vertex* pVertex = &(d_vertices[vertexIndex]);
#else
        const Vertex* pVertex = &(m_vertices[vertexIndex]);
#endif
        cellIndex[0] = pVertex->cellIndex[0];
        cellIndex[1] = pVertex->cellIndex[1];
        cellIndex[2] = pVertex->cellIndex[2];
    }

    __host__ Cell* get_d_cells() const {
        return d_cells;
    }
    __host__ Vertex* get_vertices() const {
        return m_vertices;
    }
    __host__ Vertex* get_d_vertices() const {
        return d_vertices;
    }
    friend struct MMSurfaceNet;

  private:

    // Device data
    float3* d_vertexOffsets = nullptr;
    int* d_vertexCellIdx = nullptr;
    // Dimensions and voxel size
    int   m_arraySize[3];
    float m_voxelSize[3];

    Cell*   d_cells         = nullptr;
    Vertex* d_vertices      = nullptr;   // device mirror of m_vertices (Bug 3 fix)

    // Cell definition

    Cell* m_cellArray;  // device/host pointer

    // Vertex definition

    int     m_numVertices;
    Vertex* m_vertices;

    // Internal helpers (to be implemented in .cu)
    __host__ __device__ void initCell(Cell* cell, unsigned short label);
    __host__ void setCellVertices();
    __host__ __device__ inline Cell* getCell(int cellArrayIndex) const {
#ifdef __CUDA_ARCH__
        return &d_cells[cellArrayIndex];
#else
        return &m_cellArray[cellArrayIndex];
#endif
    }
    __host__ __device__ inline Cell* getCell(int i, int j, int k) const {
        int idx = cellArrayIndex(i, j, k);
#ifdef __CUDA_ARCH__
        return &d_cells[idx];
#else
        return &m_cellArray[idx];
#endif
    }
    __host__ __device__ inline Cell* getCell(int cellIndex[3]) const {
        return getCell(cellArrayIndex(cellIndex));
    }
    __host__ __device__ inline int cellArrayIndex(int cellIndex[3]) const {
        return cellArrayIndex(cellIndex[0], cellIndex[1], cellIndex[2]);
    }
    __host__ __device__ inline int cellArrayIndex(int i, int j, int k) const {
        return i + m_arraySize[0] * (j + m_arraySize[1] * k);
    }

    __host__ __device__ void getCellLabels(Cell* cell, unsigned short labels[8]);
    __host__ __device__ inline bool isEdgeCrossing(int cellArrayIndex, MMCellFlag::Edge edge) const {
        Cell* pCell = getCell(cellArrayIndex);
        return pCell->flag.isEdgeCrossing(edge);
    }
    __host__ __device__ inline MMCellFlag::VertexType cellVertexType(int cellArrayIndex) const {
        Cell*  pCell = getCell(cellArrayIndex);
        return pCell->flag.vertexType();
    }
    __host__ __device__ void getEdgeLabels(int cellIndex[3], MMCellFlag::Edge edge, unsigned short quadLabels[2]) {
        Cell* pCell = getCell(cellIndex);
        Cell* pCellFirstLabel, *pCellSecondLabel;

        switch (edge) {
            case MMCellFlag::LeftBottomEdge:
                pCellFirstLabel = pCell;
                pCellSecondLabel = pCell + m_arraySize[0];
                break;

            case MMCellFlag::RightBottomEdge:
                pCellFirstLabel = pCell + 1;
                pCellSecondLabel = pCell + 1 + m_arraySize[0];
                break;

            case MMCellFlag::BackBottomEdge:
                pCellFirstLabel = pCell;
                pCellSecondLabel = pCell + 1;
                break;

            case MMCellFlag::FrontBottomEdge:
                pCellFirstLabel = pCell + m_arraySize[0];
                pCellSecondLabel = pCell + 1 + m_arraySize[0];
                break;

            case MMCellFlag::LeftTopEdge:
                pCellFirstLabel = pCell + m_arraySize[0] * m_arraySize[1];
                pCellSecondLabel = pCell + m_arraySize[0] + m_arraySize[0] * m_arraySize[1];
                break;

            case MMCellFlag::RightTopEdge:
                pCellFirstLabel = pCell + 1 + m_arraySize[0] * m_arraySize[1];
                pCellSecondLabel = pCell + 1 + m_arraySize[0] + m_arraySize[0] * m_arraySize[1];
                break;

            case MMCellFlag::BackTopEdge:
                pCellFirstLabel = pCell + m_arraySize[0] * m_arraySize[1];
                pCellSecondLabel = pCell + 1 + m_arraySize[0] * m_arraySize[1];
                break;

            case MMCellFlag::FrontTopEdge:
                pCellFirstLabel = pCell + m_arraySize[0] + m_arraySize[0] * m_arraySize[1];
                pCellSecondLabel = pCell + 1 + m_arraySize[0] + m_arraySize[0] * m_arraySize[1];
                break;

            case MMCellFlag::LeftBackEdge:
                pCellFirstLabel = pCell;
                pCellSecondLabel = pCell + m_arraySize[0] * m_arraySize[1];
                break;

            case MMCellFlag::RightBackEdge:
                pCellFirstLabel = pCell + 1;
                pCellSecondLabel = pCell + 1 + m_arraySize[0] * m_arraySize[1];
                break;

            case MMCellFlag::LeftFrontEdge:
                pCellFirstLabel = pCell + m_arraySize[0];
                pCellSecondLabel = pCell + m_arraySize[0] + m_arraySize[0] * m_arraySize[1];
                break;

            case MMCellFlag::RightFrontEdge:
                pCellFirstLabel = pCell + 1 + m_arraySize[0];
                pCellSecondLabel = pCell + 1 + m_arraySize[0] + m_arraySize[0] * m_arraySize[1];
                break;

            default:
                pCellFirstLabel = pCell;
                pCellSecondLabel = pCell;
                break;
        }

        quadLabels[0] = pCellFirstLabel->label;
        quadLabels[1] = pCellSecondLabel->label;
    }
    __host__ __device__ inline void getEdgeQuadPositions(int cellIndex[3], MMCellFlag::Edge edge, float quadCorners[12]) {
        int vtxIndices[4];
        getEdgeQuadVtxIndices(cellIndex, edge, vtxIndices);

        for (int i = 0; i < 4; i++) {
            int ci[3];
            getVertexCellIndex(vtxIndices[i], ci);
            getVertexPosition(ci, &(quadCorners[i * 3]));
        }
    }
    __host__ __device__ void getEdgeQuadVtxIndices(int cellIndex[3], MMCellFlag::Edge edge, int quadVtxIndices[4]) {
        Cell* pCell = getCell(cellIndex);
        int length = m_arraySize[0];
        int area   = m_arraySize[0] * m_arraySize[1];
        quadVtxIndices[0] = pCell->vertexIndex;

        switch (edge) {
            case MMCellFlag::LeftBottomEdge:
                quadVtxIndices[1] = (pCell - area)->vertexIndex;
                quadVtxIndices[2] = (pCell - 1 - area)->vertexIndex;
                quadVtxIndices[3] = (pCell - 1)->vertexIndex;
                break;

            case MMCellFlag::RightBottomEdge:
                quadVtxIndices[1] = (pCell + 1)->vertexIndex;
                quadVtxIndices[2] = (pCell + 1 - area)->vertexIndex;
                quadVtxIndices[3] = (pCell - area)->vertexIndex;
                break;

            case MMCellFlag::BackBottomEdge:
                quadVtxIndices[1] = (pCell - length)->vertexIndex;
                quadVtxIndices[2] = (pCell - length - area)->vertexIndex;
                quadVtxIndices[3] = (pCell - area)->vertexIndex;
                break;

            case MMCellFlag::FrontBottomEdge:
                quadVtxIndices[1] = (pCell - area)->vertexIndex;
                quadVtxIndices[2] = (pCell + length - area)->vertexIndex;
                quadVtxIndices[3] = (pCell + length)->vertexIndex;
                break;

            case MMCellFlag::LeftTopEdge:
                quadVtxIndices[1] = (pCell - 1)->vertexIndex;
                quadVtxIndices[2] = (pCell - 1 + area)->vertexIndex;
                quadVtxIndices[3] = (pCell + area)->vertexIndex;
                break;

            case MMCellFlag::RightTopEdge:
                quadVtxIndices[1] = (pCell + area)->vertexIndex;
                quadVtxIndices[2] = (pCell + 1 + area)->vertexIndex;
                quadVtxIndices[3] = (pCell + 1)->vertexIndex;
                break;

            case MMCellFlag::BackTopEdge:
                quadVtxIndices[1] = (pCell + area)->vertexIndex;
                quadVtxIndices[2] = (pCell - length + area)->vertexIndex;
                quadVtxIndices[3] = (pCell - length)->vertexIndex;
                break;

            case MMCellFlag::FrontTopEdge:
                quadVtxIndices[1] = (pCell + length)->vertexIndex;
                quadVtxIndices[2] = (pCell + length + area)->vertexIndex;
                quadVtxIndices[3] = (pCell + area)->vertexIndex;
                break;

            case MMCellFlag::LeftBackEdge:
                quadVtxIndices[1] = (pCell - 1)->vertexIndex;
                quadVtxIndices[2] = (pCell - 1 - length)->vertexIndex;
                quadVtxIndices[3] = (pCell - length)->vertexIndex;
                break;

            case MMCellFlag::RightBackEdge:
                quadVtxIndices[1] = (pCell - length)->vertexIndex;
                quadVtxIndices[2] = (pCell + 1 - length)->vertexIndex;
                quadVtxIndices[3] = (pCell + 1)->vertexIndex;
                break;

            case MMCellFlag::LeftFrontEdge:
                quadVtxIndices[1] = (pCell + length)->vertexIndex;
                quadVtxIndices[2] = (pCell - 1 + length)->vertexIndex;
                quadVtxIndices[3] = (pCell - 1)->vertexIndex;
                break;

            case MMCellFlag::RightFrontEdge:
                quadVtxIndices[1] = (pCell + 1)->vertexIndex;
                quadVtxIndices[2] = (pCell + 1 + length)->vertexIndex;
                quadVtxIndices[3] = (pCell + length)->vertexIndex;
                break;

            default:
                quadVtxIndices[1] = pCell->vertexIndex;
                quadVtxIndices[2] = pCell->vertexIndex;
                quadVtxIndices[3] = pCell->vertexIndex;
                break;
        }
    }

    /* Vertex position queries in padded-world coordinates (= cellIndex + vertexOffset, scaled by voxelSize). */
    __host__ __device__ __forceinline__ void getVertexPosition(int vertexIndex, float position[3]) const {
        int ci[3];
        getVertexCellIndex(vertexIndex, ci);
        getVertexPosition(ci, position);
    }
    __host__ __device__ __forceinline__ void getVertexPosition(int i, int j, int k, float position[3]) const {
        Cell* pCell = getCell(i, j, k);
        position[0] = m_voxelSize[0] * (i + pCell->vertexOffset[0]);
        position[1] = m_voxelSize[1] * (j + pCell->vertexOffset[1]);
        position[2] = m_voxelSize[2] * (k + pCell->vertexOffset[2]);
    }

    __host__ __device__ __forceinline__ void getVertexPosition(int cellIndex[3], float position[3]) const {
        Cell* pCell = getCell(cellArrayIndex(cellIndex));
        position[0] = m_voxelSize[0] * (cellIndex[0] + pCell->vertexOffset[0]);
        position[1] = m_voxelSize[1] * (cellIndex[1] + pCell->vertexOffset[1]);
        position[2] = m_voxelSize[2] * (cellIndex[2] + pCell->vertexOffset[2]);
    }
    __host__ __device__ int  vertexFaceNeighborVertexIndex(int vertexIndex, MMCellFlag::Face face) const;

    // Neighbor access
    __host__ __device__ Cell* getFaceNeighborCellAndIndex(int cellIndex[3], MMCellFlag::Face face, int nbrCellIndex[3]);
};

#endif // MCX_SN_CELLMAP_CUH
