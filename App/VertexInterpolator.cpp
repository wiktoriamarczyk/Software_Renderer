#include "VertexInterpolator.h"

VertexInterpolator::VertexInterpolator(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C)
{
    m_A.oneOverW = 1.0f / A.screenPosition.w;
    m_A.colorOverW = A.color * m_A.oneOverW;
    m_A.normalOverW = A.normal * m_A.oneOverW;
    m_A.uvOverW = A.uv * m_A.oneOverW;
    m_A.worldPositionOverW = A.worldPosition * m_A.oneOverW;
    m_A.screenPositionZ = A.screenPosition.z;

    m_B.oneOverW = 1.0f / B.screenPosition.w;
    m_B.colorOverW = B.color * m_B.oneOverW;
    m_B.normalOverW = B.normal * m_B.oneOverW;
    m_B.uvOverW = B.uv * m_B.oneOverW;
    m_B.worldPositionOverW = B.worldPosition * m_B.oneOverW;
    m_B.screenPositionZ = B.screenPosition.z;

    m_C.oneOverW = 1.0f / C.screenPosition.w;
    m_C.colorOverW = C.color * m_C.oneOverW;
    m_C.normalOverW = C.normal * m_C.oneOverW;
    m_C.uvOverW = C.uv * m_C.oneOverW;
    m_C.worldPositionOverW = C.worldPosition * m_C.oneOverW;
    m_C.screenPositionZ = C.screenPosition.z;
}

void VertexInterpolator::Interpolate(const Vector3f& baricentric, TransformedVertex& out)
{
    float oneOverW = baricentric.x * m_A.oneOverW + baricentric.y * m_B.oneOverW + baricentric.z * m_C.oneOverW;
    out.screenPosition.z = baricentric.x * m_A.screenPositionZ + baricentric.y * m_B.screenPositionZ + baricentric.z * m_C.screenPositionZ;

    float w = 1.0f / oneOverW;

    out.color = (baricentric.x * m_A.colorOverW + baricentric.y * m_B.colorOverW + baricentric.z * m_C.colorOverW) * w;
    out.normal = (baricentric.x * m_A.normalOverW + baricentric.y * m_B.normalOverW + baricentric.z * m_C.normalOverW) * w;
    out.uv = (baricentric.x * m_A.uvOverW + baricentric.y * m_B.uvOverW + baricentric.z * m_C.uvOverW) * w;
    out.worldPosition = (baricentric.x * m_A.worldPositionOverW + baricentric.y * m_B.worldPositionOverW + baricentric.z * m_C.worldPositionOverW) * w;
    out.uv = (baricentric.x * m_A.uvOverW + baricentric.y * m_B.uvOverW + baricentric.z * m_C.uvOverW) * w;
}