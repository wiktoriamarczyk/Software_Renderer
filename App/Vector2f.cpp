#include "Vector2f.h"
#include "Matrix4.h"
#include "Vector3f.h"

Vector2f::Vector2f(float x, float y)
{
    this->x = x;
    this->y = y;
}

Vector2f::Vector2f(Vector3f vector)
{
    x = vector.x;
    y = vector.y;
}

Vector2f Vector2f::operator+(Vector2f other)const
{
    return Vector2f(x + other.x, y + other.y);
}

Vector2f Vector2f::operator-(Vector2f other)const
{
    return Vector2f(x - other.x, y - other.y);
}

Vector2f Vector2f::operator*(float value)const
{
    return Vector2f(x * value, y * value);
}

Vector2f Vector2f::operator/(float value)const
{
    return Vector2f(x / value, y / value);
}

float Vector2f::GetLength()const
{
    return sqrt(x * x + y * y);
}

Vector2f& Vector2f::Normalize()
{
    float length = GetLength();
    x = x / length;
    y = y / length;

    return *this;
}

Vector2f Vector2f::Normalized()const
{
    return Vector2f(x / GetLength(), y / GetLength());
}


float Vector2f::Dot(const Vector2f& other)const
{
    return x * other.x + y * other.y;
}

Vector2f Vector2f::CWiseMin(const Vector2f& other) const
{
    return Vector2f(std::min(x, other.x), std::min(y, other.y));
}

Vector2f Vector2f::CWiseMax(const Vector2f& other) const
{
    return Vector2f(std::max(x, other.x), std::max(y, other.y));
}