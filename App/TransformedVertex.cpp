#include "TransformedVertex.h"
#include "Matrix4.h"

void TransformedVertex::ProjToScreen(Vertex v, Matrix4f worldMatrix, Matrix4f mvpMatrix)
{
    worldPosition = v.position.Transformed(worldMatrix);
    normal = (v.normal.Transformed(worldMatrix) - Vector3f{ 0,0,0 }.Transformed(worldMatrix)).Normalized();
    color = v.color;
    uv = v.uv;

    Vector4f clip = Vector4f(v.position, 1.0f).Transformed(mvpMatrix);

    zValue = clip.z;

    screenPosition.x = (clip.x / clip.w + 1) * SCREEN_WIDTH / 2;
    screenPosition.y = (clip.y / clip.w + 1) * SCREEN_HEIGHT / 2;
}