/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Common.h"
#include "Math.h"

struct ALIGN_FOR_AVX TransformedVertex
{
    Vector4f m_Color;           // 0-3
    Vector3f m_Normal;          // 4-6
    Vector2f m_UV;              // 7-8
    Vector3f m_WorldPosition;   // 9-11
    Vector4f m_ScreenPosition;

    float*       Data()      { return m_Color.Data(); }
    const float* Data()const { return m_Color.Data(); }

    TransformedVertex operator*(float value)const
    {
        TransformedVertex result;
        result.m_ScreenPosition = m_ScreenPosition * value;
       // result.zValue = zValue * value;
        result.m_Normal = m_Normal * value;
        result.m_WorldPosition = m_WorldPosition * value;
        result.m_Color = m_Color * value;
        result.m_UV = m_UV * value;
        return result;
    }

    TransformedVertex operator+(const TransformedVertex& vertex)const
    {
        TransformedVertex result;
        result.m_ScreenPosition = m_ScreenPosition + vertex.m_ScreenPosition;
        //result.zValue = zValue + vertex.zValue;
        result.m_Normal = m_Normal + vertex.m_Normal;
        result.m_WorldPosition = m_WorldPosition + vertex.m_WorldPosition;
        result.m_Color = m_Color + vertex.m_Color;
        result.m_UV = m_UV + vertex.m_UV;
        return result;
    }

    void ProjToScreen(const Vertex& v, const Matrix4f& worldMatrix, const Matrix4f& mvpMatrix);
};

inline void TransformedVertex::ProjToScreen(const Vertex& v, const Matrix4f& worldMatrix, const Matrix4f& mvpMatrix)
{
    m_WorldPosition = v.position.Multiplied(worldMatrix);
    m_Normal        = v.normal.TransformedVec(worldMatrix).Normalized();
    m_Color         = v.color;
    m_UV            = v.uv;

    m_ScreenPosition = Vector4f(v.position, 1.0f).Transformed(mvpMatrix);

    float oneOverW = 1.0f / m_ScreenPosition.w;

    m_ScreenPosition.x *= oneOverW;
    m_ScreenPosition.y *= oneOverW;
    m_ScreenPosition.z *= oneOverW;

    m_ScreenPosition.x = (m_ScreenPosition.x + 1) * SCREEN_WIDTH / 2;
    m_ScreenPosition.y = (m_ScreenPosition.y + 1) * SCREEN_HEIGHT / 2;
}