// ==========================================================
// File: MMCellFlag.cu
// GPU implementation of MMCellFlag methods
// ==========================================================

#include "MMCellFlag.cuh"

// Set components of the cell flag from 8 corner labels
__host__ __device__ void MMCellFlag::set(const unsigned short cellLabels[8], int debugIdx) {
    /* By default the cell has no vertex and no face or edge crossings. */
    m_bitFlag = 0;

    // Find edge crossings
    int numEdgeCrossings = 0;

    if (cellLabels[0] != cellLabels[3]) {
        m_bitFlag |= m_leftBottomEdgeCrossingBit;
        numEdgeCrossings++;
    }

    if (cellLabels[1] != cellLabels[2]) {
        m_bitFlag |= m_rightBottomEdgeCrossingBit;
        numEdgeCrossings++;
    }

    if (cellLabels[0] != cellLabels[1]) {
        m_bitFlag |= m_backBottomEdgeCrossingBit;
        numEdgeCrossings++;
    }

    if (cellLabels[2] != cellLabels[3]) {
        m_bitFlag |= m_frontBottomEdgeCrossingBit;
        numEdgeCrossings++;
    }

    if (cellLabels[4] != cellLabels[7]) {
        m_bitFlag |= m_leftTopEdgeCrossingBit;
        numEdgeCrossings++;
    }

    if (cellLabels[5] != cellLabels[6]) {
        m_bitFlag |= m_rightTopEdgeCrossingBit;
        numEdgeCrossings++;
    }

    if (cellLabels[4] != cellLabels[5]) {
        m_bitFlag |= m_backTopEdgeCrossingBit;
        numEdgeCrossings++;
    }

    if (cellLabels[6] != cellLabels[7]) {
        m_bitFlag |= m_frontTopEdgeCrossingBit;
        numEdgeCrossings++;
    }

    if (cellLabels[0] != cellLabels[4]) {
        m_bitFlag |= m_leftBackEdgeCrossingBit;
        numEdgeCrossings++;
    }

    if (cellLabels[1] != cellLabels[5]) {
        m_bitFlag |= m_rightBackEdgeCrossingBit;
        numEdgeCrossings++;
    }

    if (cellLabels[3] != cellLabels[7]) {
        m_bitFlag |= m_leftFrontEdgeCrossingBit;
        numEdgeCrossings++;
    }

    if (cellLabels[2] != cellLabels[6]) {
        m_bitFlag |= m_rightFrontEdgeCrossingBit;
        numEdgeCrossings++;
    }

    if (numEdgeCrossings == 0) {
        return;
    }

    // Find face crossings
    unsigned int faceTypeBits;
    faceTypeBits = faceCrossingTypeAsBits(cellLabels[0], cellLabels[3], cellLabels[7], cellLabels[4]);
    m_bitFlag |= (faceTypeBits << LeftFaceShift);
    faceTypeBits = faceCrossingTypeAsBits(cellLabels[1], cellLabels[2], cellLabels[6], cellLabels[5]);
    m_bitFlag |= (faceTypeBits << RightFaceShift);

    faceTypeBits = faceCrossingTypeAsBits(cellLabels[0], cellLabels[1], cellLabels[5], cellLabels[4]);
    m_bitFlag |= (faceTypeBits << BackFaceShift);
    faceTypeBits = faceCrossingTypeAsBits(cellLabels[3], cellLabels[2], cellLabels[6], cellLabels[7]);
    m_bitFlag |= (faceTypeBits << FrontFaceShift);

    faceTypeBits = faceCrossingTypeAsBits(cellLabels[0], cellLabels[1], cellLabels[2], cellLabels[3]);
    m_bitFlag |= (faceTypeBits << BottomFaceShift);
    faceTypeBits = faceCrossingTypeAsBits(cellLabels[4], cellLabels[5], cellLabels[6], cellLabels[7]);
    m_bitFlag |= (faceTypeBits << TopFaceShift);


    // Determine vertex type
    int numFaceCrossings = 0;
    int numJunctionCrossings = 0;

    for (Face face = Face::LeftFace; face <= Face::TopFace; ++face) {
        if (faceCrossingType(face) != FaceCrossingType::NoFaceCrossing) {
            numFaceCrossings++;

            if (faceCrossingType(face) == FaceCrossingType::JunctionFaceCrossing) {
                numJunctionCrossings++;
            }
        }
    }

    if (numFaceCrossings != 0) {
        unsigned int vertexTypeBits = 0;

        if (numJunctionCrossings < 1) {
            vertexTypeBits = (unsigned int)VertexType::SurfaceVertex;
        } else if (numJunctionCrossings <= 2) {
            vertexTypeBits = (unsigned int)VertexType::EdgeVertex;
        } else {
            vertexTypeBits = (unsigned int)VertexType::CornerVertex;
        }

        m_bitFlag |= (vertexTypeBits << VertexTypeShift);
    }

}
