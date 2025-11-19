/*
* Master’s thesis - Analysis of selected optimization techniques for a 3D software renderer
* Author: Wiktoria Marczyk
* Year: 2025
*/

#include "VertexInterpolator.h"

VertexInterpolator::VertexInterpolator(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C)
{
    m_A.m_OneOverW = 1.0f / A.m_ScreenPosition.w;
    m_A.m_NormalOverW = A.m_Normal * m_A.m_OneOverW;
    m_A.m_ColorOverW = A.m_Color * m_A.m_OneOverW;
    m_A.m_UVOverW = A.m_UV * m_A.m_OneOverW;
    m_A.m_WorldPositionOverW = A.m_WorldPosition * m_A.m_OneOverW;
    m_A.m_ScreenPositionZ = A.m_ScreenPosition.z;

    m_B.m_OneOverW = 1.0f / B.m_ScreenPosition.w;
    m_B.m_NormalOverW = B.m_Normal * m_B.m_OneOverW;
    m_B.m_ColorOverW = B.m_Color * m_B.m_OneOverW;
    m_B.m_UVOverW = B.m_UV * m_B.m_OneOverW;
    m_B.m_WorldPositionOverW = B.m_WorldPosition * m_B.m_OneOverW;
    m_B.m_ScreenPositionZ = B.m_ScreenPosition.z;

    m_C.m_OneOverW = 1.0f / C.m_ScreenPosition.w;
    m_C.m_NormalOverW = C.m_Normal * m_C.m_OneOverW;
    m_C.m_ColorOverW = C.m_Color * m_C.m_OneOverW;
    m_C.m_UVOverW = C.m_UV * m_C.m_OneOverW;
    m_C.m_WorldPositionOverW = C.m_WorldPosition * m_C.m_OneOverW;
    m_C.m_ScreenPositionZ = C.m_ScreenPosition.z;
}

VertexInterpolator::VertexInterpolator(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& Color)
{
    m_A.m_OneOverW = 1.0f / A.m_ScreenPosition.w;
    m_A.m_NormalOverW = A.m_Normal * m_A.m_OneOverW;
    m_A.m_ColorOverW = A.m_Color * Color * m_A.m_OneOverW;
    m_A.m_UVOverW = A.m_UV * m_A.m_OneOverW;
    m_A.m_WorldPositionOverW = A.m_WorldPosition * m_A.m_OneOverW;
    m_A.m_ScreenPositionZ = A.m_ScreenPosition.z;

    m_B.m_OneOverW = 1.0f / B.m_ScreenPosition.w;
    m_B.m_NormalOverW = B.m_Normal * m_B.m_OneOverW;
    m_B.m_ColorOverW = B.m_Color * Color * m_B.m_OneOverW;
    m_B.m_UVOverW = B.m_UV * m_B.m_OneOverW;
    m_B.m_WorldPositionOverW = B.m_WorldPosition * m_B.m_OneOverW;
    m_B.m_ScreenPositionZ = B.m_ScreenPosition.z;

    m_C.m_OneOverW = 1.0f / C.m_ScreenPosition.w;
    m_C.m_NormalOverW = C.m_Normal * m_C.m_OneOverW;
    m_C.m_ColorOverW = C.m_Color * Color * m_C.m_OneOverW;
    m_C.m_UVOverW = C.m_UV * m_C.m_OneOverW;
    m_C.m_WorldPositionOverW = C.m_WorldPosition * m_C.m_OneOverW;
    m_C.m_ScreenPositionZ = C.m_ScreenPosition.z;
}