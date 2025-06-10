#pragma once

#include "Vector2f.h"
#include "Vector3f.h"
#include "Vector4f.h"
#include "Matrix4.h"

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

    static const uint32_t Stride;
    static const uint32_t PositionOffset;
    static const uint32_t NormalOffset;
    static const uint32_t ColorOffset;
    static const uint32_t UVOffset;
};

constexpr inline const uint32_t Vertex::Stride          = sizeof(Vertex);
constexpr inline const uint32_t Vertex::PositionOffset  = offsetof(Vertex, position);
constexpr inline const uint32_t Vertex::NormalOffset    = offsetof(Vertex, normal);
constexpr inline const uint32_t Vertex::ColorOffset     = offsetof(Vertex, color);
constexpr inline const uint32_t Vertex::UVOffset        = offsetof(Vertex, uv);

class IMath
{
public:
    virtual void MultiplyVec4ByScalar( const float* v, float scalar , float* pOut )const=0;
    virtual void MultiplyVec8ByScalar( const float* v, float scalar , float* pOut )const=0;
    virtual void MulVec4ByScalarAdd( const float* v, float scalar , const float* c , float* pOut )const=0;
    virtual void MulVec8ByScalarAdd( const float* v, float scalar , const float* c , float* pOut )const=0;
    virtual void AddVec4( const float* a, const float* b , float* pOut )const=0;
    virtual void AddVec8( const float* a, const float* b , float* pOut )const=0;
    virtual void SubVec4( const float* a, const float* b , float* pOut )const=0;
    virtual void SubVec8( const float* a, const float* b , float* pOut )const=0;
    virtual void MulVec4( const float* a, const float* b , float* pOut )const=0;
    virtual void MulVec8( const float* a, const float* b , float* pOut )const=0;
    virtual void MulAddVec4( const float* a, const float* b , const float* c , float* pOut )const=0;
    virtual void MulAddVec8( const float* a, const float* b , const float* c , float* pOut )const=0;
    virtual void log()const = 0;
};

class MathCPU final : public IMath
{
public:
    virtual void MultiplyVec4ByScalar( const float* v, float scalar , float* pOut )const override;
    virtual void MultiplyVec8ByScalar( const float* v, float scalar , float* pOut )const override;
    virtual void MulVec4ByScalarAdd( const float* v, float scalar , const float* c , float* pOut )const override;
    virtual void MulVec8ByScalarAdd( const float* v, float scalar , const float* c , float* pOut )const override;
    virtual void AddVec4( const float* a, const float* b , float* pOut )const override;
    virtual void AddVec8( const float* a, const float* b , float* pOut )const override;
    virtual void SubVec4( const float* a, const float* b , float* pOut )const override;
    virtual void SubVec8( const float* a, const float* b , float* pOut )const override;
    virtual void MulVec4( const float* a, const float* b , float* pOut )const override;
    virtual void MulVec8( const float* a, const float* b , float* pOut )const override;
    virtual void MulAddVec4( const float* a, const float* b , const float* c , float* pOut )const override;
    virtual void MulAddVec8( const float* a, const float* b , const float* c , float* pOut )const override;
    virtual void log()const override { printf("MathCPU\n"); }
};

class MathSSE final : public IMath
{
public:
    virtual void MultiplyVec4ByScalar( const float* v, float scalar , float* pOut )const override;
    virtual void MultiplyVec8ByScalar( const float* v, float scalar , float* pOut )const override;
    virtual void MulVec4ByScalarAdd( const float* v, float scalar , const float* c , float* pOut )const override;
    virtual void MulVec8ByScalarAdd( const float* v, float scalar , const float* c , float* pOut )const override;
    virtual void AddVec4( const float* a, const float* b , float* pOut )const override;
    virtual void AddVec8( const float* a, const float* b , float* pOut )const override;
    virtual void SubVec4( const float* a, const float* b , float* pOut )const override;
    virtual void SubVec8( const float* a, const float* b , float* pOut )const override;
    virtual void MulVec4( const float* a, const float* b , float* pOut )const override;
    virtual void MulVec8( const float* a, const float* b , float* pOut )const override;
    virtual void MulAddVec4( const float* a, const float* b , const float* c , float* pOut )const override;
    virtual void MulAddVec8( const float* a, const float* b , const float* c , float* pOut )const override;
    virtual void log()const override { printf("MathSSE\n"); }
};

class MathAVX final : public IMath
{
public:
    virtual void MultiplyVec4ByScalar( const float* v, float scalar , float* pOut )const override;
    virtual void MultiplyVec8ByScalar( const float* v, float scalar , float* pOut )const override;
    virtual void MulVec4ByScalarAdd( const float* v, float scalar , const float* c , float* pOut )const override;
    virtual void MulVec8ByScalarAdd( const float* v, float scalar , const float* c , float* pOut )const override;
    virtual void AddVec4( const float* a, const float* b , float* pOut )const override;
    virtual void AddVec8( const float* a, const float* b , float* pOut )const override;
    virtual void SubVec4( const float* a, const float* b , float* pOut )const override;
    virtual void SubVec8( const float* a, const float* b , float* pOut )const override;
    virtual void MulVec4( const float* a, const float* b , float* pOut )const override;
    virtual void MulVec8( const float* a, const float* b , float* pOut )const override;
    virtual void MulAddVec4( const float* a, const float* b , const float* c , float* pOut )const override;
    virtual void MulAddVec8( const float* a, const float* b , const float* c , float* pOut )const override;
    virtual void log()const override { printf("MathAVX\n"); }
};




//****************************************************************
//                          Math CPU
//****************************************************************

inline void MathCPU::MultiplyVec4ByScalar( const float* v, float scalar , float* pOut )const
{
    pOut[0] = v[0] * scalar;
    pOut[1] = v[1] * scalar;
    pOut[2] = v[2] * scalar;
    pOut[3] = v[3] * scalar;
}

inline void MathCPU::MultiplyVec8ByScalar( const float* v, float scalar , float* pOut )const
{
    MultiplyVec4ByScalar( v + 0 , scalar , pOut + 0 );
    MultiplyVec4ByScalar( v + 4 , scalar , pOut + 4 );
}

inline void MathCPU::AddVec4( const float* a, const float* b , float* pOut )const
{
    pOut[0] = a[0] + b[0];
    pOut[1] = a[1] + b[1];
    pOut[2] = a[2] + b[2];
    pOut[3] = a[3] + b[3];
}

inline void MathCPU::AddVec8(const float* a, const float* b, float* pOut) const
{
    AddVec4(a + 0, b + 0, pOut + 0);
    AddVec4(a + 4, b + 4, pOut + 4);
}

inline void MathCPU::SubVec4( const float* a, const float* b , float* pOut )const
{
    pOut[0] = a[0] - b[0];
    pOut[1] = a[1] - b[1];
    pOut[2] = a[2] - b[2];
    pOut[3] = a[3] - b[3];
}

inline void MathCPU::SubVec8(const float* a, const float* b, float* pOut) const
{
    SubVec4(a + 0, b + 0, pOut + 0);
    SubVec4(a + 4, b + 4, pOut + 4);
}

inline void MathCPU::MulVec4( const float* a, const float* b , float* pOut )const
{
    pOut[0] = a[0] * b[0];
    pOut[1] = a[1] * b[1];
    pOut[2] = a[2] * b[2];
    pOut[3] = a[3] * b[3];
}

inline void MathCPU::MulVec8(const float* a, const float* b, float* pOut) const
{
    MulVec4(a + 0, b + 0, pOut + 0);
    MulVec4(a + 4, b + 4, pOut + 4);
}

inline void MathCPU::MulAddVec4( const float* a, const float* b , const float* c , float* pOut )const
{
    pOut[0] = ( a[0] * b[0] ) + c[0];
    pOut[1] = ( a[1] * b[1] ) + c[1];
    pOut[2] = ( a[2] * b[2] ) + c[2];
    pOut[3] = ( a[3] * b[3] ) + c[3];
}

inline void MathCPU::MulAddVec8( const float* a, const float* b , const float* c , float* pOut )const
{
    MulAddVec4( a + 0 , b + 0 , c + 0 , pOut + 0 );
    MulAddVec4( a + 4 , b + 4 , c + 4 , pOut + 4 );
}

inline void MathCPU::MulVec4ByScalarAdd( const float* a, float scalar , const float* c , float* pOut )const
{
    pOut[0] = ( a[0] * scalar ) + c[0];
    pOut[1] = ( a[1] * scalar ) + c[1];
    pOut[2] = ( a[2] * scalar ) + c[2];
    pOut[3] = ( a[3] * scalar ) + c[3];
}

inline void MathCPU::MulVec8ByScalarAdd( const float* a, float scalar , const float* c , float* pOut )const
{
    MulVec4ByScalarAdd( a + 0 , scalar , c + 0 , pOut + 0 );
    MulVec4ByScalarAdd( a + 4 , scalar , c + 4 , pOut + 4 );
}

//****************************************************************
//                          Math SSE
//****************************************************************

inline void MathSSE::MultiplyVec4ByScalar( const float* v, float scalar , float* pOut )const
{
    __m128 a = _mm_load_ps (v);         // _mm_loadu_ps (float const* mem_addr)
    __m128 b = _mm_set1_ps (scalar);    // _mm_set_ps1 (float a)
    __m128 c = _mm_mul_ps  (a, b);      // _mm_mul_ps (__m128 a, __m128 b)
                _mm_store_ps(pOut, c);   // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathSSE::MultiplyVec8ByScalar( const float* v, float scalar , float* pOut )const
{
    MultiplyVec4ByScalar( v + 0 , scalar , pOut + 0 );
    MultiplyVec4ByScalar( v + 4 , scalar , pOut + 4 );
}

inline void MathSSE::AddVec4( const float* a, const float* b , float* pOut )const
{
    __m128 a1 = _mm_load_ps (a);         // _mm_loadu_ps (float const* mem_addr)
    __m128 b1 = _mm_load_ps (b);         // _mm_loadu_ps (float const* mem_addr)
    __m128 c  = _mm_add_ps  (a1, b1);    // _mm_add_ps (__m128 a, __m128 b)
                _mm_store_ps(pOut, c);   // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathSSE::AddVec8(const float* a, const float* b, float* pOut) const
{
    AddVec4(a + 0, b + 0, pOut + 0);
    AddVec4(a + 4, b + 4, pOut + 4);
}

inline void MathSSE::SubVec4( const float* a, const float* b , float* pOut )const
{
    __m128 a1 = _mm_load_ps (a);         // _mm_loadu_ps (float const* mem_addr)
    __m128 b1 = _mm_load_ps (b);         // _mm_loadu_ps (float const* mem_addr)
    __m128 c  = _mm_sub_ps  (a1, b1);    // _mm_sub_ps (__m128 a, __m128 b)
                _mm_store_ps(pOut, c);   // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathSSE::SubVec8(const float* a, const float* b, float* pOut) const
{
    SubVec4(a + 0, b + 0, pOut + 0);
    SubVec4(a + 4, b + 4, pOut + 4);
}

inline void MathSSE::MulVec4( const float* a, const float* b , float* pOut )const
{
    __m128 a1 = _mm_load_ps (a);         // _mm_loadu_ps (float const* mem_addr)
    __m128 b1 = _mm_load_ps (b);         // _mm_loadu_ps (float const* mem_addr)
    __m128 c  = _mm_mul_ps  (a1, b1);    // _mm_mul_ps (__m128 a, __m128 b)
                _mm_store_ps(pOut, c);   // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathSSE::MulVec8( const float* a, const float* b, float* pOut ) const
{
    MulVec4(a + 0, b + 0, pOut + 0);
    MulVec4(a + 4, b + 4, pOut + 4);
}

inline void MathSSE::MulAddVec4( const float* a, const float* b , const float* c , float* pOut )const
{
    __m128 a1 = _mm_load_ps (a);         // _mm_loadu_ps (float const* mem_addr)
    __m128 b1 = _mm_load_ps (b);         // _mm_loadu_ps (float const* mem_addr)
    __m128 c1 = _mm_load_ps (c);         // _mm_loadu_ps (float const* mem_addr)
    __m128 r  = _mm_fmadd_ps(a1, b1, c1);// _mm_fmadd_ps (__m128 a, __m128 b)
                _mm_store_ps(pOut, r);   // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathSSE::MulAddVec8( const float* a, const float* b , const float* c , float* pOut )const
{
    MulAddVec4(a + 0, b + 0, c + 0, pOut + 0);
    MulAddVec4(a + 4, b + 4, c + 4, pOut + 4);
}

inline void MathSSE::MulVec4ByScalarAdd( const float* a, float b , const float* c , float* pOut )const
{
    //__m128 a1 = _mm_load_ps (a);         // _mm_loadu_ps (float const* mem_addr)
    //__m128 b1 = _mm_set1_ps (b);         // _mm_loadu_ps (float const* mem_addr)
    //__m128 c1 = _mm_load_ps (c);         // _mm_loadu_ps (float const* mem_addr)
    //__m128 r  = _mm_fmadd_ps(a1, b1, c1);// _mm_fmadd_ps (__m128 a, __m128 b)
    //            _mm_store_ps(pOut, r);   // _mm_storeu_ps (float* mem_addr, __m128 a)

    __m128 a1 = _mm_load_ps (a);         // _mm_loadu_ps (float const* mem_addr)
    __m128 b1 = _mm_set1_ps (b);         // _mm_loadu_ps (float const* mem_addr)
    __m128 c1 = _mm_load_ps (c);         // _mm_loadu_ps (float const* mem_addr)
    __m128 r  = _mm_mul_ps  (a1, b1);    // _mm_fmadd_ps (__m128 a, __m128 b)
           r  = _mm_add_ps  (r , c1);    // _mm_fmadd_ps (__m128 a, __m128 b)
                _mm_store_ps(pOut, r);   // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathSSE::MulVec8ByScalarAdd( const float* a, float b , const float* c , float* pOut )const
{
    //__m256 a1 = _mm256_load_ps (a);         // _mm_loadu_ps (float const* mem_addr)
    //__m256 b1 = _mm256_set1_ps (b);         // _mm_loadu_ps (float const* mem_addr)
    //__m256 c1 = _mm256_load_ps (c);         // _mm_loadu_ps (float const* mem_addr)
    //__m256 r  = _mm256_fmadd_ps(a1, b1, c1);// _mm_add_ps (__m128 a, __m128 b)
    //           _mm256_store_ps(pOut, r);    // _mm_storeu_ps (float* mem_addr, __m128 a)

    MulVec4ByScalarAdd( a + 0, b, c + 0, pOut + 0 );
    MulVec4ByScalarAdd( a + 4, b, c + 4, pOut + 4 );
}

//****************************************************************
//                          Math AVX
//****************************************************************

inline void MathAVX::MultiplyVec4ByScalar( const float* v, float scalar , float* pOut )const
{
    __m128 a = _mm_load_ps (v);             // _mm_loadu_ps (float const* mem_addr)
    __m128 b = _mm_set1_ps (scalar);        // _mm_set_ps1 (float a)
    __m128 c = _mm_mul_ps  (a, b);          // _mm_mul_ps (__m128 a, __m128 b)
                _mm_store_ps(pOut, c);      // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathAVX::MultiplyVec8ByScalar( const float* v, float scalar , float* pOut )const
{
    __m256 a = _mm256_load_ps (v);          // _mm_loadu_ps (float const* mem_addr)
    __m256 b = _mm256_set1_ps (scalar);     // _mm_set_ps1 (float a)
    __m256 c = _mm256_mul_ps  (a, b);       // _mm_mul_ps (__m128 a, __m128 b)
               _mm256_store_ps(pOut, c);    // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathAVX::AddVec4( const float* a, const float* b , float* pOut )const
{
    __m128 a1 = _mm_load_ps (a);            // _mm_loadu_ps (float const* mem_addr)
    __m128 b1 = _mm_load_ps (b);            // _mm_loadu_ps (float const* mem_addr)
    __m128 c  = _mm_add_ps  (a1, b1);       // _mm_add_ps (__m128 a, __m128 b)
                _mm_store_ps(pOut, c);      // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathAVX::AddVec8(const float* a, const float* b, float* pOut) const
{
    __m256 a1 = _mm256_load_ps (a);         // _mm_loadu_ps (float const* mem_addr)
    __m256 b1 = _mm256_load_ps (b);         // _mm_loadu_ps (float const* mem_addr)
    __m256 c  = _mm256_add_ps  (a1, b1);    // _mm_add_ps (__m128 a, __m128 b)
               _mm256_store_ps(pOut, c);    // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathAVX::SubVec4( const float* a, const float* b , float* pOut )const
{
    __m128 a1 = _mm_load_ps (a);            // _mm_loadu_ps (float const* mem_addr)
    __m128 b1 = _mm_load_ps (b);            // _mm_loadu_ps (float const* mem_addr)
    __m128 c  = _mm_sub_ps  (a1, b1);       // _mm_sub_ps (__m128 a, __m128 b)
                _mm_store_ps(pOut, c);      // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathAVX::SubVec8(const float* a, const float* b, float* pOut) const
{
    __m256 a1 = _mm256_load_ps (a);         // _mm256_load_ps (float const* mem_addr)
    __m256 b1 = _mm256_load_ps (b);         // _mm256_load_ps (float const* mem_addr)
    __m256 c  = _mm256_sub_ps  (a1, b1);    // _mm_add_ps (__m256 a, __m256 b)
               _mm256_store_ps(pOut, c);    // _mm_storeu_ps (float* mem_addr, __m256 a)
}

inline void MathAVX::MulVec4( const float* a, const float* b , float* pOut )const
{
    __m128 a1 = _mm_load_ps (a);            // _mm_loadu_ps (float const* mem_addr)
    __m128 b1 = _mm_load_ps (b);            // _mm_loadu_ps (float const* mem_addr)
    __m128 c  = _mm_mul_ps  (a1, b1);       // _mm_add_ps (__m128 a, __m128 b)
                _mm_store_ps(pOut, c);      // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathAVX::MulVec8(const float* a, const float* b, float* pOut) const
{
    __m256 a1 = _mm256_load_ps (a);         // _mm_loadu_ps (float const* mem_addr)
    __m256 b1 = _mm256_load_ps (b);         // _mm_loadu_ps (float const* mem_addr)
    __m256 c  = _mm256_mul_ps  (a1, b1);    // _mm_add_ps (__m128 a, __m128 b)
               _mm256_store_ps(pOut, c);    // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathAVX::MulAddVec4( const float* a, const float* b , const float* c , float* pOut )const
{
    __m128 a1 = _mm_load_ps (a);         // _mm_loadu_ps (float const* mem_addr)
    __m128 b1 = _mm_load_ps (b);         // _mm_loadu_ps (float const* mem_addr)
    __m128 c1 = _mm_load_ps (c);         // _mm_loadu_ps (float const* mem_addr)
    __m128 r  = _mm_fmadd_ps(a1, b1, c1);// _mm_fmadd_ps (__m128 a, __m128 b)
                _mm_store_ps(pOut, r);   // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathAVX::MulAddVec8( const float* a, const float* b , const float* c , float* pOut )const
{
    __m256 a1 = _mm256_load_ps (a);         // _mm_loadu_ps (float const* mem_addr)
    __m256 b1 = _mm256_load_ps (b);         // _mm_loadu_ps (float const* mem_addr)
    __m256 c1 = _mm256_load_ps (c);         // _mm_loadu_ps (float const* mem_addr)
    __m256 r  = _mm256_fmadd_ps(a1, b1, c1);// _mm_add_ps (__m128 a, __m128 b)
               _mm256_store_ps(pOut, r);    // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathAVX::MulVec4ByScalarAdd( const float* a, float b , const float* c , float* pOut )const
{
    __m128 a1 = _mm_load_ps (a);         // _mm_loadu_ps (float const* mem_addr)
    __m128 b1 = _mm_set1_ps (b);         // _mm_loadu_ps (float const* mem_addr)
    __m128 c1 = _mm_load_ps (c);         // _mm_loadu_ps (float const* mem_addr)
    __m128 r  = _mm_fmadd_ps(a1, b1, c1);// _mm_fmadd_ps (__m128 a, __m128 b)
                _mm_store_ps(pOut, r);   // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathAVX::MulVec8ByScalarAdd( const float* a, float b , const float* c , float* pOut )const
{
    __m256 a1 = _mm256_load_ps (a);         // _mm_loadu_ps (float const* mem_addr)
    __m256 b1 = _mm256_set1_ps (b);         // _mm_loadu_ps (float const* mem_addr)
    __m256 c1 = _mm256_load_ps (c);         // _mm_loadu_ps (float const* mem_addr)
    __m256 r  = _mm256_fmadd_ps(a1, b1, c1);// _mm_add_ps (__m128 a, __m128 b)
               _mm256_store_ps(pOut, r);    // _mm_storeu_ps (float* mem_addr, __m128 a)
}

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
    eSide GetSide(const Vector3f& point, float epsilon = 0.0f)const;
private:
    Vector3f m_Normal;
    float    m_D = 0;
};

class Frustum
{
public:
    Frustum() = default;
    void Update(const Matrix4f& mvpMatrix);
    bool IsInside(const Vector3f& point)const;
    bool IsBoundingBoxInside(const Vector3f& Min, const Vector3f& Max)const;
private:
    Plane m_Planes[6];
};

const vector<Vertex>& ClipTriangles(const Plane& clipPlane, const float epsilon, const vector<Vertex>& verts);