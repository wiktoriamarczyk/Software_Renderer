#pragma once
#include "Common.h"

class Matrix4f;
class Vector3f;

class Vector2f {
public:
    Vector2f() = default;
    Vector2f(float x, float y);
    Vector2f(Vector3f vector);

    Vector2f operator+(Vector2f other)const;
    Vector2f operator-(Vector2f other)const;
    Vector2f operator*(float value)const;
    Vector2f operator/(float value)const;

    float GetLength()const;
    float Dot(const Vector2f& other)const;
    Vector2f& Normalize();
    Vector2f Normalized()const;
    Vector2f CWiseMin(const Vector2f& other)const;
    Vector2f CWiseMax(const Vector2f& other)const;

    float x = 0;
    float y = 0;
};