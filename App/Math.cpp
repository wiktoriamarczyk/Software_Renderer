/*
* Master’s thesis - Analysis of selected optimization techniques for a 3D software renderer
* Author: Wiktoria Marczyk
* Year: 2025
*/

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

span<const Vertex> ClipTriangles(const Plane& clipPlane, const float epsilon, span<const Vertex> verts)
{
    vector<uint8_t> vertsRelation;

    using eSide = Plane::eSide;

    const int   oldVerticesCount = verts.size();
    const int   oldEdgesCount = oldVerticesCount;
    const int   oldTrianglesCount = oldVerticesCount / 3;
    int         frontOnBackCount[3] = {};
    vertsRelation.resize(oldVerticesCount);

    // get all vertices relation to clipping plane
    for (int i = 0; i < oldVerticesCount; ++i)
    {
        vertsRelation[i] = (uint8_t)clipPlane.GetSide(verts[i].m_Position, epsilon);
        // count vertices in front/on/back of clipping plane
        frontOnBackCount[vertsRelation[i]]++;
    }

    // all vertices are behind clipping plane - clip all
    if (!frontOnBackCount[(int)eSide::Back])
        return {};

    // all vertices are in front of clipping plane - no clipping
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

    ZoneScopedN("Execute Clip");

    vector<Vertex> splitedVertices;
    vector<int> edgeSplitVertex(oldEdgesCount);

    // go through all triangles ...
    for (int triangle = 0, edgeIndex = 0; triangle < oldTrianglesCount; ++triangle)
    {
        int baseVertex = triangle * 3;
        // ... and all edges of each triangle
        for (auto& edge : triangleEdges)
        {
            int vi0 = baseVertex + edge.VertexIndex[0];
            int vi1 = baseVertex + edge.VertexIndex[1];

            // if edge is split by clip plane, add new vertex
            if ((vertsRelation[vi0] ^ vertsRelation[vi1]) && !((vertsRelation[vi0] | vertsRelation[vi1]) & (uint8_t)eSide::On))
            {
                float distance = 0;
                // calculate intersection point
                clipPlane.LineIntersection(verts[vi0].m_Position, verts[vi1].m_Position, distance);
                // create new vertex by interpolation of two vertices
                Vertex newVertex = LerpT(verts[vi0], verts[vi1], distance);

                // this edge is spitted - store index of new vertex
                edgeSplitVertex[edgeIndex] = (int)splitedVertices.size();

                splitedVertices.push_back(newVertex);
            }
            else
            {
                // no split - store -1 as index
                edgeSplitVertex[edgeIndex] = -1;
            }
            ++edgeIndex;
        }
    }

    static thread_local vector<Vertex> clippedVerts;
    clippedVerts.reserve(oldVerticesCount + splitedVertices.size() / 2);
    clippedVerts.clear();

    // go through all edges ...
    for (int e = 0; e < oldEdgesCount; e += 3)
    {
        const int edgeSplit0 = edgeSplitVertex[e + 0];
        const int edgeSplit1 = edgeSplitVertex[e + 1];
        const int edgeSplit2 = edgeSplitVertex[e + 2];

        const int vi0 = e + 0;
        const int vi1 = e + 1;
        const int vi2 = e + 2;

        // create mask of split edges by using IsIntSignBitNotSet
        // if edge is split, then IsIntSignBitNotSet returns 1, otherwise 0
        // mask is 3 bits, each bit represents one edge
        const uint8_t edgesSplitMask = IsIntSignBitNotSet(edgeSplit0) | (IsIntSignBitNotSet(edgeSplit1) << 1) | (IsIntSignBitNotSet(edgeSplit2) << 2);

        // handle all cases
        switch (edgesSplitMask)
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
            // edge 1 slitted
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
            // edge 2 slitted
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

void Frustum::Update( const Matrix4f& mvpMatrix )
{
    mvpMatrix.GetFrustumNearPlane  ( m_Planes[0] );
    mvpMatrix.GetFrustumFarPlane   ( m_Planes[1] );
    mvpMatrix.GetFrustumLeftPlane  ( m_Planes[2] );
    mvpMatrix.GetFrustumRightPlane ( m_Planes[3] );
    mvpMatrix.GetFrustumTopPlane   ( m_Planes[4] );
    mvpMatrix.GetFrustumBottomPlane( m_Planes[5] );
}

bool Frustum::IsInside( const Vector3f& point ) const
{
    for( const Plane& plane : m_Planes )
    {
        if( plane.GetSide( point ) == Plane::eSide::Back )
            return false;
    }
    return true;
}

bool Frustum::IsBoundingBoxInside(const Vector3f& Min, const Vector3f& Max) const
{
    Vector3f BBPoints[] =
    {
        Vector3f(Min.x, Min.y, Min.z),
        Vector3f(Max.x, Min.y, Min.z),
        Vector3f(Min.x, Max.y, Min.z),
        Vector3f(Max.x, Max.y, Min.z),
        Vector3f(Min.x, Min.y, Max.z),
        Vector3f(Max.x, Min.y, Max.z),
        Vector3f(Min.x, Max.y, Max.z),
        Vector3f(Max.x, Max.y, Max.z)
    };

    // check box outside/inside of frustum
    for( int i=0; i<6; i++ )
    {
        int out = 0;
        for( auto& p : BBPoints )
            out += m_Planes[i].GetSide(p) == Plane::eSide::Front ? 1: 0;

        if( out==8 )
            return false;
    }

    return true;
}
