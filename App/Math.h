#pragma once
#include "Vector2f.h"
#include "Vector3f.h"
#include "Vector4f.h"

struct Vertex
{
    Vector3f position;
    Vector3f normal;
    Vector4f color;
    Vector2f uv;

    Vertex operator*(float value)const
    {
        Vertex result;
        result.position = position * value;
        result.normal = normal * value;
        result.color = color * value;
        result.uv = uv * value;
        return result;
    }

    Vertex operator+(const Vertex& vertex)const
    {
        Vertex result;
        result.position = position + vertex.position;
        result.normal = normal + vertex.normal;
        result.color = color + vertex.color;
        result.uv = uv + vertex.uv;
        return result;
    }
};

class Plane
{
public:
    enum class eSide : uint8_t
    {
        Back = 0,
        Front = 1,
        On = 2
    };

    Plane() = default;
    Plane(const Vector3f& normal, float N);
    Plane(const Vector3f& a, const Vector3f& b, const Vector3f& c);

    float Distance(const Vector3f& point)const;
    bool  LineIntersection(const Vector3f& start, const Vector3f& end, float& scale) const;
    const Vector3f& GetNormal()const { return m_Normal; }
    float GetD()const { return m_D; }
    eSide GetSide(const Vector3f& point, float Epsilon = 0.0f)const;
private:
    Vector3f m_Normal;
    float    m_D = 0;
};

const vector<Vertex>& ClipTraingles(const Plane& ClipPlane, const float Epsilon, const vector<Vertex>& Verts);