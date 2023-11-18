#pragma once
#include "Common.h"

class Matrix4f;

class Vector4f
{
public:
    Vector4f() = default;
    Vector4f(float x, float y, float z, float w);

    Vector4f operator+(Vector4f other)const;
    Vector4f operator-(Vector4f other)const;
    Vector4f operator*(float value)const;
    Vector4f operator/(float value)const;

    float GetLength()const;
    float Dot(const Vector4f& other)const;
    Vector4f& Normalize();
    Vector4f Normalized()const;
    Vector4f CWiseMin(const Vector4f& other)const;
    Vector4f CWiseMax(const Vector4f& other)const;

    static uint32_t ToARGB(const Vector4f& color);
    static Vector4f FromARGB(uint32_t color);

    float x = 0;
    float y = 0;
    float z = 0;
    float w = 0;
};

