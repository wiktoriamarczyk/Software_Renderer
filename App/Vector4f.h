/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2023
*/

#pragma once
#include "Common.h"
#include "Vector2f.h"
#include "Vector3f.h"

class Matrix4f;
class Vector2f;
class Vector3f;

class Vector4f
{
public:
    Vector4f() = default;
    Vector4f(const Vector3f& v,float w);
    Vector4f(float x, float y, float z, float w);

    Vector4f operator+(Vector4f other)const;
    Vector4f operator-(Vector4f other)const;
    Vector4f operator*(Vector4f other)const;
    Vector4f operator*(float value)const;
    Vector4f operator/(float value)const;

    friend Vector4f operator*(float value, const Vector4f& v);

    float GetLength()const;
    float Dot(const Vector4f& other)const;
    Vector4f& Normalize();
    Vector4f Normalized()const;
    Vector4f Transformed(const Matrix4f& m) const;
    Vector4f CWiseMin(const Vector4f& other)const;
    Vector4f CWiseMax(const Vector4f& other)const;

    Vector2f xy()const;
    Vector3f xyz()const;

    static uint32_t ToARGB(const Vector4f& color);
    static Vector4f FromARGB(uint32_t color);

    float x = 0;
    float y = 0;
    float z = 0;
    float w = 0;
};

inline Vector4f::Vector4f(float x, float y, float z, float w)
{
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
}

inline Vector4f Vector4f::operator+(Vector4f other)const
{
    return Vector4f(x + other.x, y + other.y, z + other.z, w + other.w);
}

inline Vector4f Vector4f::operator-(Vector4f other)const
{
    return Vector4f(x - other.x, y - other.y, z - other.z, w - other.w);
}

inline Vector4f Vector4f::operator*(Vector4f other) const
{
    return Vector4f(x * other.x, y * other.y, z * other.z, w * other.w);
}

inline Vector4f Vector4f::operator*(float value)const
{
    return Vector4f(x * value, y * value, z * value, w * value);
}

inline Vector4f Vector4f::operator/(float value)const
{
    return Vector4f(x / value, y / value, z / value, w / value);
}

inline Vector4f operator*(float value, const Vector4f& v)
{
    return Vector4f(v.x * value, v.y * value, v.z * value, v.w * value);
}

inline Vector3f Vector4f::xyz() const
{
    return Vector3f(x, y, z);
}

inline Vector2f Vector4f::xy() const
{
    return Vector2f(x, y);
}

inline uint32_t Vector4f::ToARGB(const Vector4f& color)
{
    return (uint32_t)(color.w * 255) << 24 | (uint32_t)(color.z * 255) << 16 | (uint32_t)(color.y * 255) << 8 | (uint32_t)(color.x * 255);
}