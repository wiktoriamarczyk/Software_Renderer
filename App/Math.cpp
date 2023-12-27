#include "Math.h"


inline constexpr int IsIntSignBitNotSet(int i) { return (~static_cast<const unsigned long>(i)) >> 31; }

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



template< typename T >
static T LerpT(const T& a, const T& b, float alpha)
{
    float a1 = 1.0f - alpha;
    return a * a1 + b * alpha;
}

const vector<Vertex>& ClipTraingles(const Plane& ClipPlane, const float Epsilon, const vector<Vertex>& Verts)
{
    static vector<Vertex> EMPTY;
    vector<uint8_t> VertsRelation;
    vector<Vertex> SplitedVertices;
    vector<int> EdgeSplitVertex;
    static thread_local vector<Vertex> ClippedVerts;

    using eSide = Plane::eSide;

    float            fDist = 0;
    const int        OldVerticesCount = Verts.size();
    const int        OldEdgesCount = OldVerticesCount;
    const int        OldTrianglesCount = OldVerticesCount / 3;
    int              FrontBack[3] = {};
    VertsRelation.resize(OldVerticesCount);

    for (int i = 0; i < OldVerticesCount; ++i)
    {
        VertsRelation[i] = (uint8_t)ClipPlane.GetSide(Verts[i].position, Epsilon);
        FrontBack[VertsRelation[i]]++;
    }

    // all vertices behind clipping plane - clip all
    if (!FrontBack[(int)eSide::Back])
    {
        return EMPTY;
    }

    // all vertices in front of clipping plane - no clipping
    if (!FrontBack[(int)eSide::Front])
    {
        return Verts;
    }

    struct edge
    {
        uint8_t VertexIndex[2];
    };

    constexpr edge TriangleEdges[3] = {
        { 0,1 },
        { 1,2 },
        { 2,0 }
    };

    SplitedVertices.clear();
    EdgeSplitVertex.resize(OldEdgesCount);

    for (int t = 0, e = 0; t < OldTrianglesCount; ++t)
    {
        int BaseVertex = t * 3;
        for (auto& Edge : TriangleEdges)
        {
            int vi0 = BaseVertex + Edge.VertexIndex[0];
            int vi1 = BaseVertex + Edge.VertexIndex[1];

            if ((VertsRelation[vi0] ^ VertsRelation[vi1]) && !((VertsRelation[vi0] | VertsRelation[vi1]) & (uint8_t)eSide::On))
            {
                ClipPlane.LineIntersection(Verts[vi0].position, Verts[vi1].position, fDist);
                Vertex NewVertex = LerpT(Verts[vi0], Verts[vi1], fDist);

                EdgeSplitVertex[e] = (int)SplitedVertices.size();

                SplitedVertices.push_back(NewVertex);
            }
            else
            {
                // no split
                EdgeSplitVertex[e] = -1;
            }
            ++e;
        }
    }

    ClippedVerts.reserve(OldVerticesCount + SplitedVertices.size() / 2);
    ClippedVerts.clear();

    for (int E = 0; E < OldEdgesCount; E += 3)
    {
        const int EdgeSplit0 = EdgeSplitVertex[E + 0];
        const int EdgeSplit1 = EdgeSplitVertex[E + 1];
        const int EdgeSplit2 = EdgeSplitVertex[E + 2];

        const int vi0 = E + 0;
        const int vi1 = E + 1;
        const int vi2 = E + 2;

        const uint8_t val = IsIntSignBitNotSet(EdgeSplit0) | (IsIntSignBitNotSet(EdgeSplit1) << 1) | (IsIntSignBitNotSet(EdgeSplit2) << 2);
        switch (val)
        {
        case 0:
            // no split

            // all vertices behind clipping plane - skip
            if ((VertsRelation[vi0] | VertsRelation[vi1] | VertsRelation[vi2]) & uint8_t(eSide::Front))
                break; // skip this triangle

            // copy all
            ClippedVerts.push_back(Verts[vi0]);
            ClippedVerts.push_back(Verts[vi1]);
            ClippedVerts.push_back(Verts[vi2]);
            break;
        case 1:
            // edge 0 slitted
            if (!(VertsRelation[vi0] & uint8_t(eSide::Front)))
            {
                ClippedVerts.push_back(Verts[vi0]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit0]);
                ClippedVerts.push_back(Verts[vi2]);
            }
            else {
                ClippedVerts.push_back(SplitedVertices[EdgeSplit0]);
                ClippedVerts.push_back(Verts[vi1]);
                ClippedVerts.push_back(Verts[vi2]);
            }
            break;
        case 2:
            if (!(VertsRelation[vi1] & uint8_t(eSide::Front)))
            {
                ClippedVerts.push_back(Verts[vi1]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit1]);
                ClippedVerts.push_back(Verts[vi0]);
            }
            else
            {
                ClippedVerts.push_back(SplitedVertices[EdgeSplit1]);
                ClippedVerts.push_back(Verts[vi2]);
                ClippedVerts.push_back(Verts[vi0]);
            }
            break;
        case 4:
            if (!(VertsRelation[vi2] & uint8_t(eSide::Front)))
            {
                ClippedVerts.push_back(Verts[vi2]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit2]);
                ClippedVerts.push_back(Verts[vi1]);
            }
            else {

                ClippedVerts.push_back(SplitedVertices[EdgeSplit2]);
                ClippedVerts.push_back(Verts[vi0]);
                ClippedVerts.push_back(Verts[vi1]);
            }
            break;
        case 3:
            // edge 0 and 1 slitted
            if (!(VertsRelation[vi1] & uint8_t(eSide::Front)))
            {
                ClippedVerts.push_back(Verts[vi1]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit1]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit0]);
            }
            else {
                ClippedVerts.push_back(Verts[vi0]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit0]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit1]);

                ClippedVerts.push_back(SplitedVertices[EdgeSplit1]);
                ClippedVerts.push_back(Verts[vi2]);
                ClippedVerts.push_back(Verts[vi0]);
            }
            break;

        case 5:
            // edge 0 and 2 slitted
            if (!(VertsRelation[vi0] & uint8_t(eSide::Front)))
            {
                ClippedVerts.push_back(Verts[vi0]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit0]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit2]);
            }
            else {

                ClippedVerts.push_back(SplitedVertices[EdgeSplit0]);
                ClippedVerts.push_back(Verts[vi1]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit2]);

                ClippedVerts.push_back(Verts[vi1]);
                ClippedVerts.push_back(Verts[vi2]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit2]);
            }
            break;
        case 6:
            // edge 1 and 2 slitted
            if (!(VertsRelation[vi2] & uint8_t(eSide::Front)))
            {
                ClippedVerts.push_back(Verts[vi2]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit2]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit1]);
            }
            else {
                ClippedVerts.push_back(SplitedVertices[EdgeSplit2]);
                ClippedVerts.push_back(Verts[vi1]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit1]);

                ClippedVerts.push_back(Verts[vi0]);
                ClippedVerts.push_back(Verts[vi1]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit2]);
            }
            break;
        }
    }
    return ClippedVerts;
}