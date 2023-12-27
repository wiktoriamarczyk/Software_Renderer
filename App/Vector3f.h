#pragma once
#include "Common.h"

class Matrix4f;

class Vector3f
{
public:
    Vector3f()=default;
    Vector3f(float x, float y, float z);

    Vector3f operator+(Vector3f other)const;
    Vector3f operator-(Vector3f other)const;
    Vector3f operator*(float value)const;
    Vector3f operator/(float value)const;

    friend Vector3f operator*(float value, const Vector3f& v);

    float GetLength()const;
    float Dot(const Vector3f& other)const;
    float MaxComponent()const;
    Vector3f& Normalize();
    Vector3f Normalized()const;
    Vector3f Transformed(const Matrix4f& m) const;
    Vector3f Cross(const Vector3f& other)const;
    Vector3f CWiseMin(const Vector3f& other)const;
    Vector3f CWiseMax(const Vector3f& other)const;
    Vector3f CWiseAbs(const Vector3f& other)const;
    // TO CHECK!
    Vector3f Reflect(const Vector3f& normal)const;

    float x = 0;
    float y = 0;
    float z = 0;
};


inline Vector3f::Vector3f(float x, float y, float z)
{
    this->x = x;
    this->y = y;
    this->z = z;
}

inline Vector3f Vector3f::operator+(Vector3f other)const
{
    return Vector3f(x + other.x, y + other.y, z + other.z);
}

inline Vector3f Vector3f::operator-(Vector3f other)const
{
    return Vector3f(x - other.x, y - other.y, z - other.z);
}

inline Vector3f Vector3f::operator*(float value)const
{
    return Vector3f(x * value, y * value, z * value);
}

inline Vector3f Vector3f::operator/(float value)const
{
    return Vector3f(x / value, y / value, z / value);
}

inline Vector3f operator*(float value, const Vector3f& v)
{
    return Vector3f(v.x * value, v.y * value, v.z * value);
}

