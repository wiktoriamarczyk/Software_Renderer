
#include "Math.h"


Plane::Plane(const Vector3f& a, const Vector3f& b, const Vector3f& c)
{
    m_Normal = (b - a).Cross(c - a).Normalized();
    m_D = -m_Normal.Dot(a);
}

Plane::Plane(const Vector3f& normal, float N)
    : m_Normal(normal)
    , m_D(N)
{
}

float Plane::Distance(const Vector3f& point) const
{
    return m_Normal.Dot(point) + m_D;
}

Plane::eSide Plane::GetSide(const Vector3f& point, float epsilon) const
{
    float distance = Distance(point);
    if (distance < -epsilon)
        return eSide::Back;
    else if (distance > epsilon)
        return eSide::Front;

    return eSide::On;
}

bool Plane::LineIntersection(const Vector3f& start, const Vector3f& end, float& scale) const
{
    Vector3f dir;
    dir = (end - start);
    float d1 = m_Normal.Dot(start) + m_D;
    float d2 = m_Normal.Dot(dir);

    if (d2 == 0.0f)
        return false;

    scale = -(d1 / d2);
    return true;
}
