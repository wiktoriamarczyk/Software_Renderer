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
        Vector3f    worldPositionOverZ;
        Vector3f    normalOverZ;
        Vector2f    uvOverZ;
        Vector4f    colorOverZ;
        float       oneOverZ;
    };

    InterpolatedSource m_A;
    InterpolatedSource m_B;
    InterpolatedSource m_C;
};
