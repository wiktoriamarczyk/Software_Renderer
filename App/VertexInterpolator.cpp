#include "VertexInterpolator.h"

VertexInterpolator::VertexInterpolator(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C)
{
    m_A.oneOverZ = 1.0f / A.zValue;
    m_A.colorOverZ = A.color * m_A.oneOverZ;
    m_A.normalOverZ = A.normal * m_A.oneOverZ;
    m_A.uvOverZ = A.uv * m_A.oneOverZ;
    m_A.worldPositionOverZ = A.worldPosition * m_A.oneOverZ;

    m_B.oneOverZ = 1.0f / B.zValue;
    m_B.colorOverZ = B.color * m_B.oneOverZ;
    m_B.normalOverZ = B.normal * m_B.oneOverZ;
    m_B.uvOverZ = B.uv * m_B.oneOverZ;
    m_B.worldPositionOverZ = B.worldPosition * m_B.oneOverZ;

    m_C.oneOverZ = 1.0f / C.zValue;
    m_C.colorOverZ = C.color * m_C.oneOverZ;
    m_C.normalOverZ = C.normal * m_C.oneOverZ;
    m_C.uvOverZ = C.uv * m_C.oneOverZ;
    m_C.worldPositionOverZ = C.worldPosition * m_C.oneOverZ;
}

void VertexInterpolator::Interpolate(const Vector3f& baricentric, TransformedVertex& out)
{
    float oneOverZ = baricentric.x * m_A.oneOverZ + baricentric.y * m_B.oneOverZ + baricentric.z * m_C.oneOverZ;
    out.zValue = 1.0f / oneOverZ;

    out.color = (baricentric.x * m_A.colorOverZ + baricentric.y * m_B.colorOverZ + baricentric.z * m_C.colorOverZ) * out.zValue;
    out.normal = (baricentric.x * m_A.normalOverZ + baricentric.y * m_B.normalOverZ + baricentric.z * m_C.normalOverZ) * out.zValue;
    out.uv = (baricentric.x * m_A.uvOverZ + baricentric.y * m_B.uvOverZ + baricentric.z * m_C.uvOverZ) * out.zValue;
    out.worldPosition = (baricentric.x * m_A.worldPositionOverZ + baricentric.y * m_B.worldPositionOverZ + baricentric.z * m_C.worldPositionOverZ) * out.zValue;
}