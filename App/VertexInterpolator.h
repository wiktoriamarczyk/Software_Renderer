#pragma once
#include "Common.h"
#include "TransformedVertex.h"

class VertexInterpolator
{
public:
    VertexInterpolator(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C);
    void Interpolate(const Vector3f& baricentric, TransformedVertex& out);

private:
    struct InterpolatedSource
    {
        Vector3f    worldPositionOverW;
        Vector3f    normalOverW;
        Vector2f    uvOverW;
        Vector4f    colorOverW;
        float       oneOverW;
        float       screenPositionZ;
    };

    InterpolatedSource m_A;
    InterpolatedSource m_B;
    InterpolatedSource m_C;
};
