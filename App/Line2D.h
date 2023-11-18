#pragma once
#include "Vector2f.h"

class Line2D
{
public:
    Line2D() = default;
    Line2D(const Vector2f& start, const Vector2f& end);

    float DistanceToPoint(const Vector2f& point) const;
    bool IsRightFromLine(const Vector2f& point, float epsilon = 0.01f) const { return DistanceToPoint(point) >= 0; }

private:
    Vector2f    m_Normal = {};
    float       m_Distance = {};
};

inline Line2D::Line2D(const Vector2f& start, const Vector2f& end)
{
    auto dir = end - start;

    m_Normal.x = -dir.y;
    m_Normal.y = dir.x;
    m_Normal.Normalize();
    m_Distance = -(m_Normal.Dot(start));
}

inline float Line2D::DistanceToPoint(const Vector2f& point) const
{
    return m_Normal.Dot(point) + m_Distance;
}