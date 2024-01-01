/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2023
*/

#include "TransformedVertex.h"
#include "Matrix4.h"

void TransformedVertex::ProjToScreen(const Vertex& v, const Matrix4f& worldMatrix, const Matrix4f& mvpMatrix)
{
    worldPosition = v.position.Multiplied(worldMatrix);
    normal = (v.normal.Multiplied(worldMatrix) - Vector3f{ 0,0,0 }.Multiplied(worldMatrix)).Normalized();
    color = v.color;
    uv = v.uv;

    screenPosition = Vector4f(v.position, 1.0f).Transformed(mvpMatrix);

    float oneOverW = 1.0f / screenPosition.w;

    screenPosition.x *= oneOverW;
    screenPosition.y *= oneOverW;
    screenPosition.z *= oneOverW;

    screenPosition.x = (screenPosition.x + 1) * SCREEN_WIDTH / 2;
    screenPosition.y = (screenPosition.y + 1) * SCREEN_HEIGHT / 2;
}