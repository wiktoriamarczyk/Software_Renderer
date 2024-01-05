/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Common.h"
#include "TransformedVertex.h"

class VertexInterpolator
{
public:
    VertexInterpolator(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C);
    void InterpolateZ(const Vector3f& baricentric, TransformedVertex& out);
    void InterpolateAllButZ(const Vector3f& baricentric, TransformedVertex& out);

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

inline void VertexInterpolator::InterpolateZ(const Vector3f& baricentric, TransformedVertex& out)
{
    out.screenPosition.z = baricentric.x * m_A.screenPositionZ + baricentric.y * m_B.screenPositionZ + baricentric.z * m_C.screenPositionZ;
}

inline void VertexInterpolator::InterpolateAllButZ(const Vector3f& baricentric, TransformedVertex& out)
{
    float oneOverW = baricentric.x * m_A.oneOverW + baricentric.y * m_B.oneOverW + baricentric.z * m_C.oneOverW;

    float w = 1.0f / oneOverW;

    out.color           = (baricentric.x * m_A.colorOverW + baricentric.y * m_B.colorOverW + baricentric.z * m_C.colorOverW) * w;
    out.normal          = (baricentric.x * m_A.normalOverW + baricentric.y * m_B.normalOverW + baricentric.z * m_C.normalOverW) * w;
    out.uv              = (baricentric.x * m_A.uvOverW + baricentric.y * m_B.uvOverW + baricentric.z * m_C.uvOverW) * w;
    out.worldPosition   = (baricentric.x * m_A.worldPositionOverW + baricentric.y * m_B.worldPositionOverW + baricentric.z * m_C.worldPositionOverW) * w;
    out.uv              = (baricentric.x * m_A.uvOverW + baricentric.y * m_B.uvOverW + baricentric.z * m_C.uvOverW) * w;
}