#include "Vector4f.h"
#include "Matrix4.h"

Vector4f::Vector4f(float x, float y, float z, float w)
{
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
}

Vector4f Vector4f::operator+(Vector4f other)const
{
    return Vector4f(x + other.x, y + other.y, z + other.z, w + other.w);
}

Vector4f Vector4f::operator-(Vector4f other)const
{
    return Vector4f(x - other.x, y - other.y, z - other.z, w - other.w);
}

Vector4f Vector4f::operator*(float value)const
{
    return Vector4f(x * value, y * value, z * value, w * value);
}

Vector4f Vector4f::operator/(float value)const
{
    return Vector4f(x / value, y / value, z / value, w / value);
}

float Vector4f::GetLength()const
{
    return sqrt(x * x + y * y + z * z + w * w);
}

Vector4f& Vector4f::Normalize()
{
    float length = GetLength();
    x = x / length;
    y = y / length;
    z = z / length;
    w = w / length;

    return *this;
}

Vector4f Vector4f::Normalized()const
{
    return Vector4f(x / GetLength(), y / GetLength(), z / GetLength(), w / GetLength());
}


float Vector4f::Dot(const Vector4f& other)const
{
    return x * other.x + y * other.y + z * other.z + w * other.w;
}


Vector4f Vector4f::CWiseMin(const Vector4f& other) const
{
    return Vector4f(std::min(x, other.x), std::min(y, other.y), std::min(z, other.z), std::min(w, other.w));
}

Vector4f Vector4f::CWiseMax(const Vector4f& other) const
{
    return Vector4f(std::max(x, other.x), std::max(y, other.y), std::max(z, other.z), std::max(w, other.w));
}

uint32_t Vector4f::ToARGB(const Vector4f& color)
{
    return (uint32_t)(color.w * 255) << 24 | (uint32_t)(color.z * 255) << 16 | (uint32_t)(color.y * 255) << 8 | (uint32_t)(color.x * 255);
}

Vector4f Vector4f::FromARGB(uint32_t color)
{
    return Vector4f((float)(color & 0xFF), (float)((color >> 8) & 0xFF), (float)((color >> 16) & 0xFF), (float)((color >> 24))) * 1.0f / 255.0f;
}
