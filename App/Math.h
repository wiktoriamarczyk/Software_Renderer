#pragma once
#include "Vector2f.h"
#include "Vector3f.h"
#include "Vector4f.h"

struct Vertex
{
    Vector3f position;
    Vector3f normal;
    Vector4f color;
    Vector2f uv;
};