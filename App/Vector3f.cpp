#include "Vector3f.h"
#include "Matrix4.h"

Vector3f::Vector3f(float x, float y, float z)
{
    this->x = x;
    this->y = y;
    this->z = z;
}

Vector3f Vector3f::operator+(Vector3f other)const
{
    return Vector3f(x + other.x, y + other.y, z + other.z);
}

Vector3f Vector3f::operator-(Vector3f other)const
{
    return Vector3f(x - other.x, y - other.y, z - other.z);
}

Vector3f Vector3f::operator*(float value)const
{
    return Vector3f(x * value, y * value, z * value);
}

Vector3f Vector3f::operator/(float value)const
{
    return Vector3f(x / value, y / value, z / value);
}

float Vector3f::GetLength()const
{
    return sqrt(x * x + y * y + z * z);
}

Vector3f& Vector3f::Normalize()
{
    float length = GetLength();
    x = x / length;
    y = y / length;
    z = z / length;

    return *this;
}

Vector3f Vector3f::Normalized()const
{
    return Vector3f(x / GetLength(), y / GetLength(), z / GetLength());
}

Vector3f Vector3f::Transformed(const Matrix4f& m) const
{
    float w = 1.f / (m[3] * x + m[7] * y + m[11] * z + m[15]);
    return Vector3f(
        (m[0] * x + m[4] * y + m[8] * z + m[12]) * w,
        (m[1] * x + m[5] * y + m[9] * z + m[13]) * w,
        (m[2] * x + m[6] * y + m[10] * z + m[14]) * w);
}

float Vector3f::Dot(const Vector3f& other)const
{
    return x * other.x + y * other.y + z * other.z;
}

float Vector3f::MaxComponent() const
{
    return std::max(std::max(x, y), z);
}

Vector3f Vector3f::Cross(const Vector3f& other)const
{
    return Vector3f(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
}

Vector3f Vector3f::CWiseMin(const Vector3f& other) const
{
    return Vector3f(std::min(x, other.x), std::min(y, other.y), std::min(z, other.z));
}

Vector3f Vector3f::CWiseMax(const Vector3f& other) const
{
    return Vector3f(std::max(x, other.x), std::max(y, other.y), std::max(z, other.z));
}

Vector3f Vector3f::CWiseAbs(const Vector3f& other) const
{
    return Vector3f(std::abs(x), std::abs(y), std::abs(z));
}

Vector3f Vector3f::Reflect(const Vector3f& normal) const
{
    return *this - normal * 2.f * Dot(normal);
}
