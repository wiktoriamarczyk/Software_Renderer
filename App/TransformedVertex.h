#pragma once
#include "Common.h"
#include "Math.h"

struct TransformedVertex
{
    Vector4f screenPosition;
    Vector3f normal;
    Vector3f worldPosition;
    Vector4f color;
    Vector2f uv;

    TransformedVertex operator*(float value)const
    {
        TransformedVertex result;
        result.screenPosition = screenPosition * value;
       // result.zValue = zValue * value;
        result.normal = normal * value;
        result.worldPosition = worldPosition * value;
        result.color = color * value;
        result.uv = uv * value;
        return result;
    }

    TransformedVertex operator+(const TransformedVertex& vertex)const
    {
        TransformedVertex result;
        result.screenPosition = screenPosition + vertex.screenPosition;
        //result.zValue = zValue + vertex.zValue;
        result.normal = normal + vertex.normal;
        result.worldPosition = worldPosition + vertex.worldPosition;
        result.color = color + vertex.color;
        result.uv = uv + vertex.uv;
        return result;
    }

    void ProjToScreen(Vertex v, Matrix4f worldMatrix, Matrix4f mvpMatrix);
};
