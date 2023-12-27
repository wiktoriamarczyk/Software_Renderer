#include "Vector4f.h"
#include "Matrix4.h"
#include "Vector2f.h"
#include "Vector3f.h"

Vector4f::Vector4f(const Vector3f& v, float w)
{
    this->x = v.x;
    this->y = v.y;
    this->z = v.z;
    this->w = w;
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


Vector4f Vector4f::Transformed(const Matrix4f& m) const
{
    return Vector4f(m[0] * x + m[4] * y + m[8] * z + m[12] * w,
                    m[1] * x + m[5] * y + m[9] * z + m[13] * w,
                    m[2] * x + m[6] * y + m[10] * z + m[14] * w,
                    m[3] * x + m[7] * y + m[11] * z + m[15] * w);
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

Vector4f Vector4f::FromARGB(uint32_t color)
{
    return Vector4f((float)(color & 0xFF), (float)((color >> 8) & 0xFF), (float)((color >> 16) & 0xFF), (float)((color >> 24))) * 1.0f / 255.0f;
}


