// ==========================================================
// File: mcx_surfacenets.cu
// GPU implementation of MMSurfaceNet methods
// ==========================================================

#include "mcx_surfacenets.cuh"
#include "mcx_sn_cellmap.cuh"

#include <set>
#include <cstdio>


// __host__ __device__ void MMSurfaceNet::getVertexPosition(int vid, float pos[3]) const {
//     if (m_cellMap) m_cellMap->getVertexPosition(vid, pos);
// }

// Constructor
__host__ MMSurfaceNet::MMSurfaceNet(unsigned short* labels, int arraySize[3], float voxelSize[3]) : m_cellMap(nullptr) {
    if (m_cellMap != nullptr) {
        delete m_cellMap;
    }

    m_cellMap = new MMCellMap(labels, arraySize, voxelSize);
}

// Destructor
__host__ MMSurfaceNet::~MMSurfaceNet() {
    if (m_cellMap) {
        delete m_cellMap;
    }
}

// Relax surface
__host__ void MMSurfaceNet::relax(const RelaxAttrs relaxAttrs) {
    if (!m_cellMap) {
        return;
    }

    m_cellMap->relax(relaxAttrs);
}

__host__ void MMSurfaceNet::getVertexPosition(int vertexIndex, float position[3]) const {
    if (m_cellMap) {
        m_cellMap->getVertexPosition(vertexIndex, position);
    }
}

// Reset surface
__host__ void MMSurfaceNet::reset() {
    if (!m_cellMap) {
        return;
    }

    m_cellMap->reset();
}

// Get unique labels (host-only)
__host__ void MMSurfaceNet::labels(int** outLabels, int* count) {
    if (!m_cellMap) {
        *outLabels = nullptr;
        *count = 0;
        return;
    }

    std::set<int> labelSet;

    for (int idxVtx = 0; idxVtx < m_cellMap->numVertices(); idxVtx++) {
        int vertexIndices[4];
        unsigned short quadLabels[2];

        // Back-bottom edge
        if (m_cellMap->getEdgeQuad(idxVtx, MMCellFlag::BackBottomEdge, vertexIndices, quadLabels)) {
            labelSet.insert((int)quadLabels[0]);
            labelSet.insert((int)quadLabels[1]);
        }

        // Left-bottom edge
        if (m_cellMap->getEdgeQuad(idxVtx, MMCellFlag::LeftBottomEdge, vertexIndices, quadLabels)) {
            labelSet.insert((int)quadLabels[0]);
            labelSet.insert((int)quadLabels[1]);
        }

        // Left-back edge
        if (m_cellMap->getEdgeQuad(idxVtx, MMCellFlag::LeftBackEdge, vertexIndices, quadLabels)) {
            labelSet.insert((int)quadLabels[0]);
            labelSet.insert((int)quadLabels[1]);
        }
    }

    // Remove reserved padding
    labelSet.erase((int)ReservedLabel::Padding);

    // Copy to output array
    *count = (int)labelSet.size();
    *outLabels = new int[*count];
    int idx = 0;

    for (int val : labelSet) {
        (*outLabels)[idx++] = val;
    }
}
