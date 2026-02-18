// ==========================================================
// File: MMCellFlag.cu
// GPU implementation of MMCellFlag methods
// ==========================================================

#include "MMCellFlag.cuh"




// Set components of the cell flag from 8 labels
__host__ __device__ void MMCellFlag::set(const unsigned short cellLabels[8], int debugIdx) {
    // #ifndef __CUDA_ARCH__
    //     fprintf(stderr, "[DEBUG][MMCellFlag::set] Function entered!\n");
    // #endif

    // By default the cell has no vertex and no face or edge crossings

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
    m_bitFlag |= (faceTypeBits << TopFaceShift);

        // --- Bottom face: z- (or however you defined it)
    // faceTypeBits = faceCrossingTypeAsBits(cellLabels[0], cellLabels[1], cellLabels[2], cellLabels[3]);
    // m_bitFlag |= (faceTypeBits << BottomFaceShift);

    // // --- Top face: z+ (must use the top four corners)
    // faceTypeBits = faceCrossingTypeAsBits(cellLabels[4], cellLabels[5], cellLabels[6], cellLabels[7]);
    // m_bitFlag |= (faceTypeBits << TopFaceShift);


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

    // #ifndef __CUDA_ARCH__
    //     // CPU side: direct logging
    //     fprintf(stderr,
    //         "[DEBUG][MMCellFlag::set] labels=[%hu %hu %hu %hu %hu %hu %hu %hu], m_bitFlag=0x%08X\n",
    //         cellLabels[0], cellLabels[1], cellLabels[2], cellLabels[3],
    //         cellLabels[4], cellLabels[5], cellLabels[6], cellLabels[7],
    //         m_bitFlag);
    // #else
    //     // GPU side: write into managed debug buffer
    //     if (debugIdx >= 0 && debugIdx < 1024) {
    //         g_debugFlags[debugIdx] = m_bitFlag;
    //     }
    // #endif
}

// __host__ __device__ FaceCrossingType faceCrossingType(Face face) const
// {
//         switch (face) {
//         case MMCellFlag::Face::LeftFace:
//             return (MMCellFlag::FaceCrossingType)((MMCellFlag::m_bitFlag >> MMCellFlag::LeftFaceShift) & 0x3u);

//         case MMCellFlag::Face::RightFace:
//             return (MMCellFlag::FaceCrossingType)((MMCellFlag::m_bitFlag >> MMCellFlag::RightFaceShift) & 0x3u);

//         case MMCellFlag::Face::BackFace:
//             return (MMCellFlag::FaceCrossingType)((MMCellFlag::m_bitFlag >> MMCellFlag::BackFaceShift) & 0x3u);

//         case MMCellFlag::Face::FrontFace:
//             return (MMCellFlag::FaceCrossingType)((MMCellFlag::m_bitFlag >> MMCellFlag::FrontFaceShift) & 0x3u);

//         case MMCellFlag::Face::BottomFace:
//             return (MMCellFlag::FaceCrossingType)((MMCellFlag::m_bitFlag >> MMCellFlag::BottomFaceShift) & 0x3u);

//         case MMCellFlag::Face::TopFace:
//             return (MMCellFlag::FaceCrossingType)((MMCellFlag::m_bitFlag >> MMCellFlag::TopFaceShift) & 0x3u);

//         default:
//             return MMCellFlag::FaceCrossingType::NoFaceCrossing;
//     }
// }
// Face crossing type bits helper
// __host__ __device__ unsigned int MMCellFlag::faceCrossingTypeAsBits(
//     unsigned short c0, unsigned short c1, unsigned short c2, unsigned short c3) {
//     int numUniqueTypes = 0;
//     unsigned short uniqueTypes[4];
//     uniqueTypes[numUniqueTypes++] = c0;

//     if (c1 != uniqueTypes[0])  uniqueTypes[numUniqueTypes++] = c1;

//     int idx = 0;

//     while (idx < numUniqueTypes && c2 != uniqueTypes[idx]) {
//         idx++;
//     }

//     if (idx == numUniqueTypes) uniqueTypes[numUniqueTypes++] = c2;

//     idx = 0;

//     while (idx < numUniqueTypes && c3 != uniqueTypes[idx]) {
//         idx++;
//     }

//     if (idx == numUniqueTypes) uniqueTypes[numUniqueTypes++] = c3;

//     FaceCrossingType crossingType = NoFaceCrossing;

//     switch (numUniqueTypes) {
//         case 0:
//         case 1:
//             crossingType = NoFaceCrossing;
//             break;

//         case 2:
//             if (c0 == c2 && c1 == c3) crossingType = JunctionFaceCrossing;
//             else crossingType = SurfaceFaceCrossing;
//             break;

//         case 3:
//         case 4:
//             crossingType = JunctionFaceCrossing;
//             break;

//         default:
//             crossingType = NoFaceCrossing;
//             break;
//     }

//     return (unsigned int)crossingType;
// }

// __host__ __device__ MMCellFlag::FaceCrossingType MMCellFlag::faceCrossingType(Face face) const {
//     unsigned int faceTypeBits = 0;
//     switch (face) {
//         case Face::LeftFace:
//             faceTypeBits = (m_bitFlag & m_leftFaceCrossingBits) >> LeftFaceShift;
//             break;

//         case Face::RightFace:
//             faceTypeBits = (m_bitFlag & m_rightFaceCrossingBits) >> RightFaceShift;
//             break;

//         case Face::BackFace:
//             faceTypeBits = (m_bitFlag & m_backFaceCrossingBits) >> BackFaceShift;
//             break;

//         case Face::FrontFace:
//             faceTypeBits = (m_bitFlag & m_frontFaceCrossingBits) >> FrontFaceShift;
//             break;

//         case Face::BottomFace:
//             faceTypeBits = (m_bitFlag & m_bottomFaceCrossingBits) >> BottomFaceShift;
//             break;

//         case Face::TopFace:
//             faceTypeBits = (m_bitFlag & m_topFaceCrossingBits) >> TopFaceShift;
//             break;

//         default:            
//             faceTypeBits = 0;

//     }

//     switch (faceTypeBits) {
//         case 0: return(FaceCrossingType::NoFaceCrossing);
//         case 1: return(FaceCrossingType::SurfaceFaceCrossing);
//         case 2: return(FaceCrossingType::JunctionFaceCrossing);
//         default: return(FaceCrossingType::NoFaceCrossing);
//     }
    
// }



