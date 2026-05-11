// ==========================================================
// File: MMCellFlag.cuh
// GPU-friendly version of MMCellFlag (header only)
// ==========================================================

#ifndef MM_CELL_FLAG_CUH
#define MM_CELL_FLAG_CUH


#include <cuda_runtime.h>



// CUDA-friendly MMCellFlag
struct MMCellFlag {
    // The bitflag
    unsigned int m_bitFlag;

    // Enums (same as CPU version)
    enum VertexType {
        NoVertex, SurfaceVertex, EdgeVertex, CornerVertex
    };
    enum FaceCrossingType {
        NoFaceCrossing, SurfaceFaceCrossing, JunctionFaceCrossing
    };
    enum Face {
        LeftFace, RightFace, BackFace, FrontFace, BottomFace, TopFace
    };
    enum Edge {
        LeftBottomEdge, RightBottomEdge, BackBottomEdge, FrontBottomEdge,
        LeftTopEdge, RightTopEdge, BackTopEdge, FrontTopEdge,
        LeftBackEdge, RightBackEdge, LeftFrontEdge, RightFrontEdge
    };

    // Constructor-like init
    __device__ __host__ inline void clear() {
        m_bitFlag = 0u;
    }

    __device__ __host__ inline void operator=(const MMCellFlag& t) {
        m_bitFlag = t.m_bitFlag;
    }

    // Set components of the cell flag from 8 labels (implemented in .cu)
    __host__ __device__ void set(const unsigned short cellLabels[8], int debugIdx);

    __host__ __device__ inline void set(unsigned short cellLabels[8]) {
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
        faceTypeBits = faceCrossingTypeAsBits(cellLabels[4], cellLabels[5], cellLabels[6], cellLabels[7]);
        m_bitFlag |= (faceTypeBits << TopFaceShift);

        // Determine vertex type
        int numFaceCrossings = 0;
        int numJunctionCrossings = 0;

        for (int fi = (int)LeftFace; fi <= (int)TopFace; ++fi) {
            Face face = (Face)fi;
            FaceCrossingType fct = faceCrossingType(face);

            if (fct != FaceCrossingType::NoFaceCrossing) {
                numFaceCrossings++;

                if (fct == FaceCrossingType::JunctionFaceCrossing) {
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

    // Accessors
    __device__ __host__ inline VertexType vertexType() const {
        return static_cast<VertexType>((m_bitFlag >> VertexTypeShift) & 0x3u);
    }

    __host__ __device__ inline FaceCrossingType faceCrossingType(Face face) const {
        unsigned int faceTypeBits = 0;

        switch (face) {
            case LeftFace:
                faceTypeBits = (m_bitFlag & m_leftFaceCrossingBits) >> LeftFaceShift;
                break;

            case RightFace:
                faceTypeBits = (m_bitFlag & m_rightFaceCrossingBits) >> RightFaceShift;
                break;

            case BackFace:
                faceTypeBits = (m_bitFlag & m_backFaceCrossingBits) >> BackFaceShift;
                break;

            case FrontFace:
                faceTypeBits = (m_bitFlag & m_frontFaceCrossingBits) >> FrontFaceShift;
                break;

            case BottomFace:
                faceTypeBits = (m_bitFlag & m_bottomFaceCrossingBits) >> BottomFaceShift;
                break;

            case TopFace:
                faceTypeBits = (m_bitFlag & m_topFaceCrossingBits) >> TopFaceShift;
                break;

            default:
                faceTypeBits = 0;

        }

        switch (faceTypeBits) {
            case 0:
                return (FaceCrossingType::NoFaceCrossing);

            case 1:
                return (FaceCrossingType::SurfaceFaceCrossing);

            case 2:
                return (FaceCrossingType::JunctionFaceCrossing);

            default:
                return (FaceCrossingType::NoFaceCrossing);
        }
    }
    __host__ __device__ inline bool isEdgeCrossing(Edge edge) const {
        bool result = false;

        switch (edge) {
            case LeftBottomEdge:
                result = (m_bitFlag & m_leftBottomEdgeCrossingBit);
                break;

            case RightBottomEdge:
                result = (m_bitFlag & m_rightBottomEdgeCrossingBit);
                break;

            case BackBottomEdge:
                result = (m_bitFlag & m_backBottomEdgeCrossingBit);
                break;

            case FrontBottomEdge:
                result = (m_bitFlag & m_frontBottomEdgeCrossingBit);
                break;

            case LeftTopEdge:
                result = (m_bitFlag & m_leftTopEdgeCrossingBit);
                break;

            case RightTopEdge:
                result = (m_bitFlag & m_rightTopEdgeCrossingBit);
                break;

            case BackTopEdge:
                result = (m_bitFlag & m_backTopEdgeCrossingBit);
                break;

            case FrontTopEdge:
                result = (m_bitFlag & m_frontTopEdgeCrossingBit);
                break;

            case LeftBackEdge:
                result = (m_bitFlag & m_leftBackEdgeCrossingBit);
                break;

            case RightBackEdge:
                result = (m_bitFlag & m_rightBackEdgeCrossingBit);
                break;

            case LeftFrontEdge:
                result = (m_bitFlag & m_leftFrontEdgeCrossingBit);
                break;

            case RightFrontEdge:
                result = (m_bitFlag & m_rightFrontEdgeCrossingBit);
                break;

            default:
                return false;
        }

        return result;
    }

  private:
    // Bit shifts
    enum BitShift {
        VertexTypeShift   = 0,
        LeftFaceShift     = 2,
        RightFaceShift    = 4,
        BackFaceShift     = 6,
        FrontFaceShift    = 8,
        BottomFaceShift   = 10,
        TopFaceShift      = 12,
    };

    // Flag bits associated with each component
    static constexpr unsigned int m_vertexTypeBits            = (1 << VertexTypeShift) | (1 << (VertexTypeShift + 1));
    static constexpr unsigned int m_leftFaceCrossingBits      = (1 << LeftFaceShift) | (1 << (LeftFaceShift + 1));
    static constexpr unsigned int m_rightFaceCrossingBits     = (1 << RightFaceShift) | (1 << (RightFaceShift + 1));
    static constexpr unsigned int m_backFaceCrossingBits      = (1 << BackFaceShift) | (1 << (BackFaceShift + 1));
    static constexpr unsigned int m_frontFaceCrossingBits     = (1 << FrontFaceShift) | (1 << (FrontFaceShift + 1));
    static constexpr unsigned int m_bottomFaceCrossingBits    = (1 << BottomFaceShift) | (1 << (BottomFaceShift + 1));
    static constexpr unsigned int m_topFaceCrossingBits       = (1 << TopFaceShift) | (1 << (TopFaceShift + 1));
    static constexpr unsigned int m_leftBottomEdgeCrossingBit = 1 << 14;
    static constexpr unsigned int m_rightBottomEdgeCrossingBit = 1 << 15;
    static constexpr unsigned int m_backBottomEdgeCrossingBit = 1 << 16;
    static constexpr unsigned int m_frontBottomEdgeCrossingBit = 1 << 17;
    static constexpr unsigned int m_leftTopEdgeCrossingBit    = 1 << 18;
    static constexpr unsigned int m_rightTopEdgeCrossingBit   = 1 << 19;
    static constexpr unsigned int m_backTopEdgeCrossingBit    = 1 << 20;
    static constexpr unsigned int m_frontTopEdgeCrossingBit   = 1 << 21;
    static constexpr unsigned int m_leftBackEdgeCrossingBit   = 1 << 22;
    static constexpr unsigned int m_rightBackEdgeCrossingBit  = 1 << 23;
    static constexpr unsigned int m_leftFrontEdgeCrossingBit  = 1 << 24;
    static constexpr unsigned int m_rightFrontEdgeCrossingBit = 1 << 25;

    // Helper for face crossing
    __host__ __device__ inline unsigned int faceCrossingTypeAsBits( unsigned short c0, unsigned short c1, unsigned short c2, unsigned short c3) {
        int numUniqueTypes = 0;
        unsigned short uniqueTypes[4];
        uniqueTypes[numUniqueTypes++] = c0;

        if (c1 != uniqueTypes[0]) {
            uniqueTypes[numUniqueTypes++] = c1;
        }

        int idx = 0;

        while (idx < numUniqueTypes && c2 != uniqueTypes[idx]) {
            idx++;
        }

        if (idx == numUniqueTypes) {
            uniqueTypes[numUniqueTypes++] = c2;
        }

        idx = 0;

        while (idx < numUniqueTypes && c3 != uniqueTypes[idx]) {
            idx++;
        }

        if (idx == numUniqueTypes) {
            uniqueTypes[numUniqueTypes++] = c3;
        }

        FaceCrossingType crossingType = NoFaceCrossing;

        switch (numUniqueTypes) {
            case 0:
            case 1:
                crossingType = NoFaceCrossing;
                break;

            case 2:
                if (c0 == c2 && c1 == c3) {
                    crossingType = JunctionFaceCrossing;
                } else {
                    crossingType = SurfaceFaceCrossing;
                }

                break;

            case 3:
            case 4:
                crossingType = JunctionFaceCrossing;
                break;

            default:
                crossingType = NoFaceCrossing;
                break;
        }

        return (unsigned int)crossingType;
    }
};

// Inline operator++ for Face (host+device)
__device__ __host__ inline MMCellFlag::Face& operator++ (MMCellFlag::Face& f) {
    f = MMCellFlag::Face((unsigned int)(f) + 1);
    return f;
}
__device__ __host__ inline MMCellFlag::Face operator++ (MMCellFlag::Face& f, int) {
    MMCellFlag::Face old = f;
    f = MMCellFlag::Face((unsigned int)(f) + 1);
    return old;
}

#endif // MM_CELL_FLAG_CUH
