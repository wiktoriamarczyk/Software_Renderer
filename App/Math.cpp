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

const vector<Vertex>& ClipTraingles(const Plane& clipPlane, const float epsilon, const vector<Vertex>& verts)
{
    static vector<Vertex> EMPTY;
    vector<uint8_t> vertsRelation;
    vector<Vertex> splitedVertices;
    vector<int> edgeSplitVertex;
    static thread_local vector<Vertex> clippedVerts;

    using eSide = Plane::eSide;

    float       distance = 0;
    const int   oldVerticesCount = verts.size();
    const int   oldEdgesCount = oldVerticesCount;
    const int   oldTrianglesCount = oldVerticesCount / 3;
    int         frontOnBackCount[3] = {};
    vertsRelation.resize(oldVerticesCount);

    for (int i = 0; i < oldVerticesCount; ++i)
    {
        vertsRelation[i] = (uint8_t)clipPlane.GetSide(verts[i].position, epsilon);
        frontOnBackCount[vertsRelation[i]]++;
    }

    // all vertices behind clipping plane - clip all
    if (!frontOnBackCount[(int)eSide::Back])
    {
        return EMPTY;
    }

    // all vertices in front of clipping plane - no clipping
    if (!frontOnBackCount[(int)eSide::Front])
    {
        return verts;
    }

    struct edge
    {
        uint8_t VertexIndex[2];
    };

    constexpr edge triangleEdges[3] = {
        { 0,1 },
        { 1,2 },
        { 2,0 }
    };

    splitedVertices.clear();
    edgeSplitVertex.resize(oldEdgesCount);

    for (int t = 0, e = 0; t < oldTrianglesCount; ++t)
    {
        int baseVertex = t * 3;
        for (auto& edge : triangleEdges)
        {
            int vi0 = baseVertex + edge.VertexIndex[0];
            int vi1 = baseVertex + edge.VertexIndex[1];

            if ((vertsRelation[vi0] ^ vertsRelation[vi1]) && !((vertsRelation[vi0] | vertsRelation[vi1]) & (uint8_t)eSide::On))
            {
                clipPlane.LineIntersection(verts[vi0].position, verts[vi1].position, distance);
                Vertex newVertex = LerpT(verts[vi0], verts[vi1], distance);

                edgeSplitVertex[e] = (int)splitedVertices.size();

                splitedVertices.push_back(newVertex);
            }
            else
            {
                // no split
                edgeSplitVertex[e] = -1;
            }
            ++e;
        }
    }

    clippedVerts.reserve(oldVerticesCount + splitedVertices.size() / 2);
    clippedVerts.clear();

    for (int e = 0; e < oldEdgesCount; e += 3)
    {
        const int edgeSplit0 = edgeSplitVertex[e + 0];
        const int edgeSplit1 = edgeSplitVertex[e + 1];
        const int edgeSplit2 = edgeSplitVertex[e + 2];

        const int vi0 = e + 0;
        const int vi1 = e + 1;
        const int vi2 = e + 2;

        const uint8_t val = IsIntSignBitNotSet(edgeSplit0) | (IsIntSignBitNotSet(edgeSplit1) << 1) | (IsIntSignBitNotSet(edgeSplit2) << 2);
        switch (val)
        {
        case 0:
            // no split

            // all vertices behind clipping plane - skip
            if ((vertsRelation[vi0] | vertsRelation[vi1] | vertsRelation[vi2]) & uint8_t(eSide::Front))
                break; // skip this triangle

            // copy all
            clippedVerts.push_back(verts[vi0]);
            clippedVerts.push_back(verts[vi1]);
            clippedVerts.push_back(verts[vi2]);
            break;
        case 1:
            // edge 0 slitted
            if (!(vertsRelation[vi0] & uint8_t(eSide::Front)))
            {
                clippedVerts.push_back(verts[vi0]);
                clippedVerts.push_back(splitedVertices[edgeSplit0]);
                clippedVerts.push_back(verts[vi2]);
            }
            else {
                clippedVerts.push_back(splitedVertices[edgeSplit0]);
                clippedVerts.push_back(verts[vi1]);
                clippedVerts.push_back(verts[vi2]);
            }
            break;
        case 2:
            if (!(vertsRelation[vi1] & uint8_t(eSide::Front)))
            {
                clippedVerts.push_back(verts[vi1]);
                clippedVerts.push_back(splitedVertices[edgeSplit1]);
                clippedVerts.push_back(verts[vi0]);
            }
            else
            {
                clippedVerts.push_back(splitedVertices[edgeSplit1]);
                clippedVerts.push_back(verts[vi2]);
                clippedVerts.push_back(verts[vi0]);
            }
            break;
        case 4:
            if (!(vertsRelation[vi2] & uint8_t(eSide::Front)))
            {
                clippedVerts.push_back(verts[vi2]);
                clippedVerts.push_back(splitedVertices[edgeSplit2]);
                clippedVerts.push_back(verts[vi1]);
            }
            else {

                clippedVerts.push_back(splitedVertices[edgeSplit2]);
                clippedVerts.push_back(verts[vi0]);
                clippedVerts.push_back(verts[vi1]);
            }
            break;
        case 3:
            // edge 0 and 1 slitted
            if (!(vertsRelation[vi1] & uint8_t(eSide::Front)))
            {
                clippedVerts.push_back(verts[vi1]);
                clippedVerts.push_back(splitedVertices[edgeSplit1]);
                clippedVerts.push_back(splitedVertices[edgeSplit0]);
            }
            else {
                clippedVerts.push_back(verts[vi0]);
                clippedVerts.push_back(splitedVertices[edgeSplit0]);
                clippedVerts.push_back(splitedVertices[edgeSplit1]);

                clippedVerts.push_back(splitedVertices[edgeSplit1]);
                clippedVerts.push_back(verts[vi2]);
                clippedVerts.push_back(verts[vi0]);
            }
            break;

        case 5:
            // edge 0 and 2 slitted
            if (!(vertsRelation[vi0] & uint8_t(eSide::Front)))
            {
                clippedVerts.push_back(verts[vi0]);
                clippedVerts.push_back(splitedVertices[edgeSplit0]);
                clippedVerts.push_back(splitedVertices[edgeSplit2]);
            }
            else {

                clippedVerts.push_back(splitedVertices[edgeSplit0]);
                clippedVerts.push_back(verts[vi1]);
                clippedVerts.push_back(splitedVertices[edgeSplit2]);

                clippedVerts.push_back(verts[vi1]);
                clippedVerts.push_back(verts[vi2]);
                clippedVerts.push_back(splitedVertices[edgeSplit2]);
            }
            break;
        case 6:
            // edge 1 and 2 slitted
            if (!(vertsRelation[vi2] & uint8_t(eSide::Front)))
            {
                clippedVerts.push_back(verts[vi2]);
                clippedVerts.push_back(splitedVertices[edgeSplit2]);
                clippedVerts.push_back(splitedVertices[edgeSplit1]);
            }
            else {
                clippedVerts.push_back(splitedVertices[edgeSplit2]);
                clippedVerts.push_back(verts[vi1]);
                clippedVerts.push_back(splitedVertices[edgeSplit1]);

                clippedVerts.push_back(verts[vi0]);
                clippedVerts.push_back(verts[vi1]);
                clippedVerts.push_back(splitedVertices[edgeSplit2]);
            }
            break;
        }
    }
    return clippedVerts;
}