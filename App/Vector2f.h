/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

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

    friend Vector2f operator*(float value, const Vector2f& v);

    float GetLength()const;
    float Dot(const Vector2f& other)const;
    Vector2f& Normalize();
    Vector2f Normalized()const;
    Vector2f CWiseMin(const Vector2f& other)const;
    Vector2f CWiseMax(const Vector2f& other)const;

    float* Data() { return &x; }
    const float* Data() const { return &x; }

    float x = 0;
    float y = 0;
};

inline Vector2f::Vector2f(float x, float y)
{
    this->x = x;
    this->y = y;
}

inline Vector2f Vector2f::operator+(Vector2f other)const
{
    return Vector2f(x + other.x, y + other.y);
}

inline Vector2f Vector2f::operator-(Vector2f other)const
{
    return Vector2f(x - other.x, y - other.y);
}

inline Vector2f Vector2f::operator*(float value)const
{
    return Vector2f(x * value, y * value);
}

inline Vector2f Vector2f::operator/(float value)const
{
    return Vector2f(x / value, y / value);
}

inline Vector2f operator*(float value, const Vector2f& v)
{
    return Vector2f(v.x * value, v.y * value);
}