#pragma once

#include "Vector2f.h"
#include "Vector3f.h"
#include "Vector4f.h"
#include "Matrix4.h"

#if defined(__clang__)

#include <ammintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <avxintrin.h>
#include <fmaintrin.h>
#include <fma4intrin.h>
#include <softintrin.h>
#include <intrin.h>

#endif

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

template< std::integral T >
inline bool IsPowerOfTwo(T value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

constexpr inline const uint32_t Vertex::Stride          = sizeof(Vertex);
constexpr inline const uint32_t Vertex::PositionOffset  = offsetof(Vertex, position);
constexpr inline const uint32_t Vertex::NormalOffset    = offsetof(Vertex, normal);
constexpr inline const uint32_t Vertex::ColorOffset     = offsetof(Vertex, color);
constexpr inline const uint32_t Vertex::UVOffset        = offsetof(Vertex, uv);

class IMath
{
public:
    virtual void MultiplyVec4ByScalar( const float* v, float scalar , float* pOut )const=0;
    virtual void MultiplyVec4ByScalar884( const float* v, float scalar , float* pOut )const=0;
    virtual void MultiplyVec8ByScalar( const float* v, float scalar , float* pOut )const=0;
    virtual void MulVec4ByScalarAdd( const float* v, float scalar , const float* c , float* pOut )const=0;
    virtual void MulVec8ByScalarAdd( const float* v, float scalar , const float* c , float* pOut )const=0;
    virtual void AddVec4( const float* a, const float* b , float* pOut )const=0;
    virtual void AddVec8( const float* a, const float* b , float* pOut )const=0;
    virtual void AddVec4ToScalar( const float* v, float scalar , float* pOut )const=0;
    virtual void AddVec8ToScalar( const float* v, float scalar , float* pOut )const=0;
    virtual void SubVec4( const float* a, const float* b , float* pOut )const=0;
    virtual void SubVec8( const float* a, const float* b , float* pOut )const=0;
    virtual void MulVec4( const float* a, const float* b , float* pOut )const=0;
    virtual void MulVec8( const float* a, const float* b , float* pOut )const=0;
    virtual void MulAddVec4( const float* a, const float* b , const float* c , float* pOut )const=0;
    virtual void MulAddVec8( const float* a, const float* b , const float* c , float* pOut )const=0;
    virtual void PairAddVec4( const float* a, const float* b , float* pOut )const = 0;
    virtual void PairAddVec8( const float* a, const float* b , float* pOut )const = 0;
    virtual void Interleave4( float a, float b , float* pOut )const = 0;
    virtual void Interleave8( float a, float b , float* pOut )const = 0;
    virtual void LessThan4( const float* a, const float* b , float* pOut )const=0;
    virtual void LessThan8( const float* a, const float* b , float* pOut )const=0;
    virtual void BitOr4( const float* a, const float* b , float* pOut )const=0;
    virtual void BitOr8( const float* a, const float* b , float* pOut )const=0;
    virtual void log()const = 0;

    virtual void EdgeFunction3x( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const = 0;
    virtual bool EdgeFunction3xToBool( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const = 0;
    virtual void EdgeFunction3xToBoolx8( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const = 0;
};

class MathCPU final : public IMath
{
public:
    virtual void MultiplyVec4ByScalar( const float* v, float scalar , float* pOut )const override;
    virtual void MultiplyVec4ByScalar884( const float* v, float scalar , float* pOut )const override;
    virtual void MultiplyVec8ByScalar( const float* v, float scalar , float* pOut )const override;
    virtual void MulVec4ByScalarAdd( const float* v, float scalar , const float* c , float* pOut )const override;
    virtual void MulVec8ByScalarAdd( const float* v, float scalar , const float* c , float* pOut )const override;
    virtual void AddVec4( const float* a, const float* b , float* pOut )const override;
    virtual void AddVec8( const float* a, const float* b , float* pOut )const override;
    virtual void AddVec4ToScalar( const float* v, float scalar , float* pOut )const override;
    virtual void AddVec8ToScalar( const float* v, float scalar , float* pOut )const override;
    virtual void SubVec4( const float* a, const float* b , float* pOut )const override;
    virtual void SubVec8( const float* a, const float* b , float* pOut )const override;
    virtual void MulVec4( const float* a, const float* b , float* pOut )const override;
    virtual void MulVec8( const float* a, const float* b , float* pOut )const override;
    virtual void MulAddVec4( const float* a, const float* b , const float* c , float* pOut )const override;
    virtual void MulAddVec8( const float* a, const float* b , const float* c , float* pOut )const override;
    virtual void PairAddVec4( const float* a, const float* b , float* pOut )const override;
    virtual void PairAddVec8( const float* a, const float* b , float* pOut )const override;
    virtual void Interleave4( float a, float b , float* pOut )const override;
    virtual void Interleave8( float a, float b , float* pOut )const override;
    virtual void LessThan4( const float* a, const float* b , float* pOut )const override;
    virtual void LessThan8( const float* a, const float* b , float* pOut )const override;
    virtual void BitOr4( const float* a, const float* b , float* pOut )const override;
    virtual void BitOr8( const float* a, const float* b , float* pOut )const override;
    virtual void log()const override { printf("MathCPU\n"); }

    virtual void EdgeFunction3x( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const override;
    virtual bool EdgeFunction3xToBool( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const override;
    virtual void EdgeFunction3xToBoolx8( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const override;
};

class MathSSE final : public IMath
{
public:
    virtual void MultiplyVec4ByScalar( const float* v, float scalar , float* pOut )const override;
    virtual void MultiplyVec4ByScalar884( const float* v, float scalar , float* pOut )const override;
    virtual void MultiplyVec8ByScalar( const float* v, float scalar , float* pOut )const override;
    virtual void MulVec4ByScalarAdd( const float* v, float scalar , const float* c , float* pOut )const override;
    virtual void MulVec8ByScalarAdd( const float* v, float scalar , const float* c , float* pOut )const override;
    virtual void AddVec4( const float* a, const float* b , float* pOut )const override;
    virtual void AddVec8( const float* a, const float* b , float* pOut )const override;
    virtual void AddVec4ToScalar( const float* v, float scalar , float* pOut )const override;
    virtual void AddVec8ToScalar( const float* v, float scalar , float* pOut )const override;
    virtual void SubVec4( const float* a, const float* b , float* pOut )const override;
    virtual void SubVec8( const float* a, const float* b , float* pOut )const override;
    virtual void MulVec4( const float* a, const float* b , float* pOut )const override;
    virtual void MulVec8( const float* a, const float* b , float* pOut )const override;
    virtual void MulAddVec4( const float* a, const float* b , const float* c , float* pOut )const override;
    virtual void MulAddVec8( const float* a, const float* b , const float* c , float* pOut )const override;
    virtual void PairAddVec4( const float* a, const float* b , float* pOut )const override;
    virtual void PairAddVec8( const float* a, const float* b , float* pOut )const override;
    virtual void Interleave4( float a, float b , float* pOut )const override;
    virtual void Interleave8( float a, float b , float* pOut )const override;
    virtual void LessThan4( const float* a, const float* b , float* pOut )const override;
    virtual void LessThan8( const float* a, const float* b , float* pOut )const override;
    virtual void BitOr4( const float* a, const float* b , float* pOut )const override;
    virtual void BitOr8( const float* a, const float* b , float* pOut )const override;
    virtual void log()const override { printf("MathSSE\n"); }

    virtual void EdgeFunction3x( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const override;
    virtual bool EdgeFunction3xToBool( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const override;
    virtual void EdgeFunction3xToBoolx8( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const override;
};

class MathAVX final : public IMath
{
public:
    virtual void MultiplyVec4ByScalar( const float* v, float scalar , float* pOut )const override;
    virtual void MultiplyVec4ByScalar884( const float* v, float scalar , float* pOut )const override;
    virtual void MultiplyVec8ByScalar( const float* v, float scalar , float* pOut )const override;
    virtual void MulVec4ByScalarAdd( const float* v, float scalar , const float* c , float* pOut )const override;
    virtual void MulVec8ByScalarAdd( const float* v, float scalar , const float* c , float* pOut )const override;
    virtual void AddVec4( const float* a, const float* b , float* pOut )const override;
    virtual void AddVec8( const float* a, const float* b , float* pOut )const override;
    virtual void AddVec4ToScalar( const float* v, float scalar , float* pOut )const override;
    virtual void AddVec8ToScalar( const float* v, float scalar , float* pOut )const override;
    virtual void SubVec4( const float* a, const float* b , float* pOut )const override;
    virtual void SubVec8( const float* a, const float* b , float* pOut )const override;
    virtual void MulVec4( const float* a, const float* b , float* pOut )const override;
    virtual void MulVec8( const float* a, const float* b , float* pOut )const override;
    virtual void MulAddVec4( const float* a, const float* b , const float* c , float* pOut )const override;
    virtual void MulAddVec8( const float* a, const float* b , const float* c , float* pOut )const override;
    virtual void PairAddVec4( const float* a, const float* b , float* pOut )const override;
    virtual void PairAddVec8( const float* a, const float* b , float* pOut )const override;
    virtual void Interleave4( float a, float b , float* pOut )const override;
    virtual void Interleave8( float a, float b , float* pOut )const override;
    virtual void LessThan4( const float* a, const float* b , float* pOut )const override;
    virtual void LessThan8( const float* a, const float* b , float* pOut )const override;
    virtual void BitOr4( const float* a, const float* b , float* pOut )const override;
    virtual void BitOr8( const float* a, const float* b , float* pOut )const override;
    virtual void log()const override { printf("MathAVX\n"); }

    virtual void EdgeFunction3x( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const override;
    virtual bool EdgeFunction3xToBool( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const override;
    virtual void EdgeFunction3xToBoolx8( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const override;
};



union simd_data
{
    float     f;
    uint32_t  u;
};

template< bool AVX = true >
struct simd_float
{
    simd_float() = default;
    simd_float( float vin )
    {
        if constexpr( AVX )
        {
            m_data[7].f = vin;
            m_data[6].f = vin;
            m_data[5].f = vin;
            m_data[4].f = vin;
        }
        m_data[3].f = vin;
        m_data[2].f = vin;
        m_data[1].f = vin;
        m_data[0].f = vin;
    }
    simd_float( const float* vin )
    {
        if constexpr( AVX )
        {
            m_data[7].f = vin[7];
            m_data[6].f = vin[6];
            m_data[5].f = vin[5];
            m_data[4].f = vin[4];
        }
        m_data[3].f = vin[3];
        m_data[2].f = vin[2];
        m_data[1].f = vin[1];
        m_data[0].f = vin[0];
    }
    simd_float( float v0 , float v1, float v2, float v3 ) requires( !AVX )
    {
        m_data[0].f = v0;
        m_data[1].f = v1;
        m_data[2].f = v2;
        m_data[3].f = v3;
    }
    simd_float( float v0 , float v1, float v2, float v3, float v4, float v5, float v6, float v7 ) requires( AVX )
    {
        m_data[0].f = v0;
        m_data[1].f = v1;
        m_data[2].f = v2;
        m_data[3].f = v3;
        m_data[4].f = v4;
        m_data[5].f = v5;
        m_data[6].f = v6;
        m_data[7].f = v7;
    }

    float*          data        ()      { return &this->m_data->f; }
    const float*    data        ()const { return &this->m_data->f; }
    operator        float*      ()      { return data(); }
    operator        const float*()const { return data(); }

    simd_float operator*( const simd_float& other ) const
    {
        simd_float result;

        if constexpr( AVX )
            MathCPU{}.MulVec8( *this , other , result );
        else
            MathCPU{}.MulVec4( *this , other , result );

        return result;
    }

    simd_float operator*( float other ) const
    {
        simd_float result;

        if constexpr( AVX )
            MathCPU{}.MultiplyVec8ByScalar( *this , other , result );
        else
            MathCPU{}.MultiplyVec4ByScalar( *this , other , result );

        return result;
    }

    auto lower()const requires( AVX ){ return simd_float<false>{ data() + 0 }; }
    auto upper()const requires( AVX ){ return simd_float<false>{ data() + 4 }; }

    simd_float operator+( const simd_float& other ) const
    {
        simd_float result;

        if constexpr( AVX )
            MathCPU{}.AddVec8( *this , other , result );
        else
            MathCPU{}.AddVec4( *this , other , result );

        return result;
    }

    simd_float operator+( float other ) const
    {
        simd_float result;

        if constexpr( AVX )
            MathCPU{}.AddVec8ToScalar( *this , other , result );
        else
            MathCPU{}.AddVec8ToScalar( *this , other , result );

        return result;
    }

    simd_float operator<( const simd_float& other ) const
    {
        simd_float result;

        if constexpr( AVX )
            MathCPU{}.LessThan8( *this , other , result );
        else
            MathCPU{}.LessThan4( *this , other , result );

        return result;
    }
    simd_float operator<( float v ) const
    {
        return *this < simd_float{ v };
    }

    simd_float operator|( const simd_float& other ) const
    {
        simd_float result;

        if constexpr( AVX )
            MathCPU{}.BitOr8( *this , other , result );
        else
            MathCPU{}.BitOr4( *this , other , result );

        return result;
    }

    void store( float* pOut ) const
    {
        memcpy( pOut , data() , sizeof(*this) );
    }

    static constexpr uint32_t Count = AVX ? 8 : 4;

    simd_data m_data[ Count ];
};

using sse_float = simd_float<false>;
using avx_float = simd_float<true>;


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

inline void MathCPU::MultiplyVec4ByScalar884( const float* v, float scalar , float* pOut )const
{
    return MultiplyVec4ByScalar(v, scalar, pOut);
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

inline void MathCPU::AddVec4ToScalar( const float* v, float scalar , float* pOut )const
{
    pOut[0] = v[0] + scalar;
    pOut[1] = v[1] + scalar;
    pOut[2] = v[2] + scalar;
    pOut[3] = v[3] + scalar;
}


inline void MathCPU::AddVec8ToScalar( const float* v, float scalar , float* pOut )const
{
    AddVec4ToScalar( v + 0 , scalar , pOut + 0 );
    AddVec4ToScalar( v + 4 , scalar , pOut + 4 );
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

inline void MathCPU::PairAddVec4( const float* a, const float* b , float* pOut )const
{
    pOut[0] = a[0] + a[1];
    pOut[1] = a[2] + a[3];
    pOut[2] = b[0] + b[1];
    pOut[3] = b[2] + b[3];
}

inline void MathCPU::PairAddVec8( const float* a, const float* b , float* pOut )const
{
    pOut[0] = a[0] + a[1];
    pOut[1] = a[2] + a[3];
    pOut[2] = a[4] + a[5];
    pOut[3] = a[6] + a[7];
    pOut[4] = b[0] + b[1];
    pOut[5] = b[2] + b[3];
    pOut[6] = b[4] + b[5];
    pOut[7] = b[6] + b[7];
}

inline void MathCPU::Interleave4( float a, float b , float* pOut )const
{
    pOut[0] = a;
    pOut[1] = b;
    pOut[2] = a;
    pOut[3] = b;
}

inline void MathCPU::Interleave8( float a, float b , float* pOut )const
{
    pOut[0] = a;
    pOut[1] = b;
    pOut[2] = a;
    pOut[3] = b;
    pOut[4] = a;
    pOut[5] = b;
    pOut[6] = a;
    pOut[7] = b;
}

inline void MathCPU::LessThan4( const float* a, const float* b , float* pOut )const
{
    constexpr uint32_t ALL = 0xFFFFFFFF;

    pOut[0] = a[0] < b[0] ? *reinterpret_cast<const float*>(&ALL) : 0;
    pOut[1] = a[1] < b[1] ? *reinterpret_cast<const float*>(&ALL) : 0;
    pOut[2] = a[2] < b[2] ? *reinterpret_cast<const float*>(&ALL) : 0;
    pOut[3] = a[3] < b[3] ? *reinterpret_cast<const float*>(&ALL) : 0;
}

inline void MathCPU::LessThan8( const float* a, const float* b , float* pOut )const
{
    LessThan4( a + 0 , b + 0 , pOut + 0 );
    LessThan4( a + 4 , b + 4 , pOut + 4 );
}

inline void MathCPU::BitOr4( const float* a, const float* b , float* pOut )const
{
    reinterpret_cast<uint32_t*>(pOut)[0] = reinterpret_cast<const uint32_t*>(a)[0] | reinterpret_cast<const uint32_t*>(b)[0];
    reinterpret_cast<uint32_t*>(pOut)[1] = reinterpret_cast<const uint32_t*>(a)[1] | reinterpret_cast<const uint32_t*>(b)[1];
    reinterpret_cast<uint32_t*>(pOut)[2] = reinterpret_cast<const uint32_t*>(a)[2] | reinterpret_cast<const uint32_t*>(b)[2];
    reinterpret_cast<uint32_t*>(pOut)[3] = reinterpret_cast<const uint32_t*>(a)[3] | reinterpret_cast<const uint32_t*>(b)[3];
}

inline void MathCPU::BitOr8( const float* a, const float* b , float* pOut )const
{
    BitOr4( a + 0 , b + 0 , pOut + 0 );
    BitOr4( a + 4 , b + 4 , pOut + 4 );
}

inline void MathCPU::EdgeFunction3x( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const
{
    float PS[8];

    Interleave8( P.x , P.y , PS );
    MulAddVec8( TBL0 , PS , TBL1 , PS );
    PairAddVec4( PS , PS + 4 , pResult );

    //PS[0] = ( TBL0[0] * P.x ) + TBL1[0];
    //PS[1] = ( TBL0[1] * P.y ) + TBL1[1];
    //PS[2] = ( TBL0[2] * P.x ) + TBL1[2];
    //PS[3] = ( TBL0[3] * P.y ) + TBL1[3];
    //PS[4] = ( TBL0[4] * P.x ) + TBL1[4];
    //PS[5] = ( TBL0[5] * P.y ) + TBL1[5];
    //PS[6] = ( TBL0[6] * P.x ) + TBL1[6];
    //PS[7] = ( TBL0[7] * P.y ) + TBL1[7];

    //pResult[0] = PS[0+0] + PS[0+1];
    //pResult[1] = PS[0+2] + PS[0+3];
    //pResult[2] = PS[4+0] + PS[4+1];
    //pResult[3] = PS[4+2] + PS[4+3];
}

inline bool MathCPU::EdgeFunction3xToBool( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const
{
    float    PS[8];

    Interleave8( P.x , P.y , PS );
    MulAddVec8( TBL0 , PS , TBL1 , PS );
    PairAddVec4( PS , PS + 4 , pResult );

    return (pResult[0] < 0) || (pResult[1] < 0) || (pResult[2] < 0);
}

struct ALIGN_FOR_AVX EdgeResult8_t
{
    float* data(){ return reinterpret_cast<float*>(this); }

    uint32_t SKIP[8] = {0};
    float    ABP [8] = {0};
    float    BCP [8] = {0};
    float    CAP [8] = {0};
};

inline void MathCPU::EdgeFunction3xToBoolx8( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const
{
    avx_float PX      { P.x+0
                      , P.x+1
                      , P.x+2
                      , P.x+3
                      , P.x+4
                      , P.x+5
                      , P.x+6
                      , P.x+7 };

    avx_float PY = P.y;

    avx_float TMP[6];

    EdgeResult8_t& EdgeResult = *reinterpret_cast<EdgeResult8_t*>(pResult);


    TMP[0] = PX * TBL0[0] + TBL1[0];
    TMP[1] = PY * TBL0[1] + TBL1[1];
    TMP[2] = PX * TBL0[2] + TBL1[2];
    TMP[3] = PY * TBL0[3] + TBL1[3];
    TMP[4] = PX * TBL0[4] + TBL1[4];
    TMP[5] = PY * TBL0[5] + TBL1[5];

    TMP[0] = TMP[0] + TMP[1];
    TMP[2] = TMP[2] + TMP[3];
    TMP[4] = TMP[4] + TMP[5];

    // skip ??
    PY = (TMP[0] < 0) | (TMP[2] < 0) | (TMP[4] < 0);

    PY    .store( pResult        );
    TMP[0].store( EdgeResult.ABP );
    TMP[2].store( EdgeResult.BCP );
    TMP[4].store( EdgeResult.CAP );
}

//inline void MathCPU::EdgeFunction3xToBoolx8( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const
//{
//    float    PS[8];
//    float    Final[4];
//
//    EdgeResult8_t& EdgeResult = *reinterpret_cast<EdgeResult8_t*>(pResult);
//
//    for( uint32_t i = 0; i < 8; ++i )
//    {
//        Interleave8( P.x + i , P.y , PS );
//        MulAddVec8( TBL0 , PS , TBL1 , PS );
//        PairAddVec4( PS , PS + 4 , Final );
//
//        EdgeResult.SKIP[i] = (Final[0] < 0) | (Final[1] < 0) | (Final[2] < 0);
//        EdgeResult.ABP [i] = Final[0];
//        EdgeResult.BCP [i] = Final[1];
//        EdgeResult.CAP [i] = Final[2];
//    }
//}
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

inline void MathSSE::MultiplyVec4ByScalar884( const float* v, float scalar , float* pOut )const
{
    return MultiplyVec4ByScalar(v, scalar, pOut);
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

inline void MathSSE::AddVec4ToScalar( const float* v, float scalar , float* pOut )const
{
    __m128 a = _mm_load_ps (v);         // _mm_loadu_ps (float const* mem_addr)
    __m128 b = _mm_set1_ps (scalar);    // _mm_set_ps1 (float a)
    __m128 c = _mm_add_ps  (a, b);      // _mm_mul_ps (__m128 a, __m128 b)
                _mm_store_ps(pOut, c);  // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathSSE::AddVec8(const float* a, const float* b, float* pOut) const
{
    AddVec4(a + 0, b + 0, pOut + 0);
    AddVec4(a + 4, b + 4, pOut + 4);
}

inline void MathSSE::AddVec8ToScalar( const float* v, float scalar , float* pOut )const
{
    AddVec4ToScalar(v + 0, scalar, pOut + 0);
    AddVec4ToScalar(v + 4, scalar, pOut + 4);
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
    __m128 r  = _mm_mul_ps(a1, b1);
           r  = _mm_add_ps(c1, r );
                _mm_store_ps(pOut, r);   // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathSSE::MulAddVec8( const float* a, const float* b , const float* c , float* pOut )const
{
    MulAddVec4(a + 0, b + 0, c + 0, pOut + 0);
    MulAddVec4(a + 4, b + 4, c + 4, pOut + 4);
}

inline void MathSSE::MulVec4ByScalarAdd( const float* a, float b , const float* c , float* pOut )const
{
    __m128 a1 = _mm_load_ps (a);         // _mm_loadu_ps (float const* mem_addr)
    __m128 b1 = _mm_set1_ps (b);         // _mm_loadu_ps (float const* mem_addr)
    __m128 c1 = _mm_load_ps (c);         // _mm_loadu_ps (float const* mem_addr)
    __m128 r  = _mm_mul_ps  (a1, b1);    // _mm_fmadd_ps (__m128 a, __m128 b)
           r  = _mm_add_ps  (r , c1);    // _mm_fmadd_ps (__m128 a, __m128 b)
                _mm_store_ps(pOut, r);   // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathSSE::MulVec8ByScalarAdd( const float* a, float b , const float* c , float* pOut )const
{
    MulVec4ByScalarAdd( a + 0, b, c + 0, pOut + 0 );
    MulVec4ByScalarAdd( a + 4, b, c + 4, pOut + 4 );
}

inline void MathSSE::PairAddVec4( const float* a, const float* b , float* pOut )const
{
    __m128 a1 = _mm_load_ps (a);         // _mm_load_ps (float const* mem_addr)
    __m128 b1 = _mm_load_ps (b);         // _mm_load_ps (float const* mem_addr)
    __m128 c  = _mm_hadd_ps (a1, b1);    // _mm_hadd_ps (__m128 a, __m128 b)
                _mm_store_ps(pOut, c);   // _mm_store_ps (float* mem_addr, __m128 a)
}

inline void MathSSE::PairAddVec8( const float* a, const float* b , float* pOut )const
{
    PairAddVec4(a + 0, b + 0, pOut + 0);
    PairAddVec4(a + 4, b + 4, pOut + 4);
}

inline void MathSSE::Interleave4( float a, float b , float* pOut )const
{
    __m128 a1 = _mm_set1_ps (a);
    __m128 b1 = _mm_set1_ps (b);
    __m128 r1 = _mm_blend_ps( a1 , b1 , 0xA );
                _mm_store_ps(pOut, r1);
}

inline void MathSSE::Interleave8( float a, float b , float* pOut )const
{
    __m128 a1 = _mm_set1_ps (a);
    __m128 b1 = _mm_set1_ps (b);
    _mm_store_ps( pOut+0 , _mm_blend_ps( a1 , b1 , 0xA ) );
    _mm_store_ps( pOut+4 , _mm_blend_ps( a1 , b1 , 0xA ) );
}

inline void MathSSE::LessThan4( const float* a, const float* b , float* pOut )const
{
    __m128 ma = _mm_load_ps (a);
    __m128 mb = _mm_load_ps (b);
    __m128 mc = _mm_cmplt_ps(ma, mb);
                _mm_store_ps(pOut, mc);
}

inline void MathSSE::LessThan8( const float* a, const float* b , float* pOut )const
{
    LessThan4( a + 0 , b + 0 , pOut + 0 );
    LessThan4( a + 4 , b + 4 , pOut + 4 );
}

inline void MathSSE::BitOr4( const float* a, const float* b , float* pOut )const
{
    __m128 ma = _mm_load_ps (a);
    __m128 mb = _mm_load_ps (b);
    __m128 mc = _mm_or_ps(ma, mb);
                _mm_store_ps(pOut, mc);
}

inline void MathSSE::BitOr8( const float* a, const float* b , float* pOut )const
{
    BitOr4( a + 0 , b + 0 , pOut + 0 );
    BitOr4( a + 4 , b + 4 , pOut + 4 );
}

//inline void MathSSE::EdgeFunction3x( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const
//{
//    // Interleave8( P.x , P.y , Arg1 );
//    // MulAddVec8( Arg0 , Arg1 , Arg2 , Arg1 );
//    // PairAddVec4( Arg1 , Arg1 + 4 , &EdgeResult.ABP );
//
//    __m128 PS = _mm_set_ps ( P.y , P.x , P.y , P.x );                       // PS ->  x , y , x , y
//    __m128 a1 = _mm_mul_ps( _mm_load_ps(TBL0  ) , PS );
//    __m128 b1 = _mm_mul_ps( _mm_load_ps(TBL0+4) , PS );
//
//           a1 = _mm_add_ps( _mm_load_ps(TBL1  ) , a1 );
//           b1 = _mm_add_ps( _mm_load_ps(TBL1+4) , b1 );
//
//
//
//           // TBL0[0] * x + TBL1[0]
//           // TBL0[1] * y + TBL1[1]
//           // TBL0[2] * x + TBL1[2]
//           // TBL0[3] * y + TBL1[3]
//
//           // TBL0[4] * x + TBL1[4]
//           // TBL0[5] * y + TBL1[5]
//           // TBL0[6] * x + TBL1[6]
//           // TBL0[7] * y + TBL1[7]
//
//
//    a1 = _mm_hadd_ps( a1 , b1 );
//    _mm_store_ps( pResult , a1 ); // Store the result in pResult
//}


inline void MathSSE::EdgeFunction3x( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const
{
    __m128 py = _mm_set_ss( P.x );
    __m128 px = _mm_set_ss( P.y );

    __m128 a1= _mm_mul_ss( _mm_load_ss( TBL0+0 ) , py );
           a1= _mm_add_ss( _mm_load_ss( TBL1+0 ) , a1 );
    __m128 a2= _mm_mul_ss( _mm_load_ss( TBL0+1 ) , px );
           a2= _mm_add_ss( _mm_load_ss( TBL1+1 ) , a2 );

           a1 = _mm_add_ss( a1 , a2 ); // a1 = TBL0[0] * y + TBL1[0] + TBL0[1] * x + TBL1[1]

    __m128 a3= _mm_mul_ss( _mm_load_ss( TBL0+2 ) , py );
           a3= _mm_add_ss( _mm_load_ss( TBL1+2 ) , a3 );
    __m128 a4= _mm_mul_ss( _mm_load_ss( TBL0+3 ) , px );
           a4= _mm_add_ss( _mm_load_ss( TBL1+3 ) , a4 );


           a3 = _mm_add_ss( a3 , a4 ); // a3 = TBL0[2] * y + TBL1[2] + TBL0[3] * y + TBL1[3]

    __m128 a5= _mm_mul_ss( _mm_load_ss( TBL0+4 ) , py );
           a5= _mm_add_ss( _mm_load_ss( TBL1+4 ) , a5 );
    __m128 a6= _mm_mul_ss( _mm_load_ss( TBL0+5 ) , px );
           a6= _mm_add_ss( _mm_load_ss( TBL1+5 ) , a6 );


           a5 = _mm_add_ss( a5 , a6 ); // a5 = TBL0[4] * y + TBL1[4] + TBL0[5] * y + TBL1[5]
                _mm_store_ss( pResult , a1 );
                _mm_store_ss( pResult+1 , a3 );
                _mm_store_ss( pResult+2 , a5 );
}

inline bool MathSSE::EdgeFunction3xToBool( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const
{
    __m128 py = _mm_set_ss( P.x );
    __m128 px = _mm_set_ss( P.y );

    __m128 a1= _mm_mul_ss( _mm_load_ss( TBL0+0 ) , py );
           a1= _mm_add_ss( _mm_load_ss( TBL1+0 ) , a1 );
    __m128 a2= _mm_mul_ss( _mm_load_ss( TBL0+1 ) , px );
           a2= _mm_add_ss( _mm_load_ss( TBL1+1 ) , a2 );

           a1 = _mm_add_ss( a1 , a2 ); // a1 = TBL0[0] * y + TBL1[0] + TBL0[1] * x + TBL1[1]

    __m128 a3= _mm_mul_ss( _mm_load_ss( TBL0+2 ) , py );
           a3= _mm_add_ss( _mm_load_ss( TBL1+2 ) , a3 );
    __m128 a4= _mm_mul_ss( _mm_load_ss( TBL0+3 ) , px );
           a4= _mm_add_ss( _mm_load_ss( TBL1+3 ) , a4 );


           a3 = _mm_add_ss( a3 , a4 ); // a3 = TBL0[2] * y + TBL1[2] + TBL0[3] * y + TBL1[3]

    __m128 a5= _mm_mul_ss( _mm_load_ss( TBL0+4 ) , py );
           a5= _mm_add_ss( _mm_load_ss( TBL1+4 ) , a5 );
    __m128 a6= _mm_mul_ss( _mm_load_ss( TBL0+5 ) , px );
           a6= _mm_add_ss( _mm_load_ss( TBL1+5 ) , a6 );


           a5 = _mm_add_ss( a5 , a6 ); // a5 = TBL0[4] * y + TBL1[4] + TBL0[5] * y + TBL1[5]
           px = _mm_setzero_ps();
                _mm_store_ss( pResult , a1 );
                _mm_store_ss( pResult+1 , a3 );
                _mm_store_ss( pResult+2 , a5 );


    int r1 = _mm_ucomilt_ss(a1, px); // compare each element of a1 with 0.0f
    int r2 = _mm_ucomilt_ss(a3, px);
    int r3 = _mm_ucomilt_ss(a5, px); // compare each element of a5 with 0.0f

    return r1 || r2 || r3; // is any of the first 3 elements < 0.0f?
}

//inline void MathSSE::EdgeFunction3xToBoolx8( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const
//{
//    __m128 PX = _mm_set_ps( P.x+3 , P.x+2 , P.x+1 , P.x );
//    __m128 PY = _mm_set_ps( P.y , P.y   , P.y   , P.y   );
//
//    __m128 _TBL0_0 = _mm_load1_ps(TBL0+0);
//    __m128 _TBL0_1 = _mm_load1_ps(TBL0+1);
//    __m128 _TBL0_2 = _mm_load1_ps(TBL0+2);
//    __m128 _TBL0_3 = _mm_load1_ps(TBL0+3);
//    __m128 _TBL0_4 = _mm_load1_ps(TBL0+4);
//    __m128 _TBL0_5 = _mm_load1_ps(TBL0+5);
//
//    __m128 _TBL1_0 = _mm_load1_ps(TBL1+0);
//    __m128 _TBL1_1 = _mm_load1_ps(TBL1+1);
//    __m128 _TBL1_2 = _mm_load1_ps(TBL1+2);
//    __m128 _TBL1_3 = _mm_load1_ps(TBL1+3);
//    __m128 _TBL1_4 = _mm_load1_ps(TBL1+4);
//    __m128 _TBL1_5 = _mm_load1_ps(TBL1+5);
//
//
//    __m128 TMP0 = _mm_mul_ps( _TBL0_0 , PX ); // TBL0[0] * x
//    __m128 TMP1 = _mm_mul_ps( _TBL0_1 , PY ); // TBL0[1] * y
//    __m128 TMP2 = _mm_mul_ps( _TBL0_2 , PX ); // TBL0[2] * x
//    __m128 TMP3 = _mm_mul_ps( _TBL0_3 , PY ); // TBL0[3] * y
//    __m128 TMP4 = _mm_mul_ps( _TBL0_4 , PX ); // TBL0[4] * x
//    __m128 TMP5 = _mm_mul_ps( _TBL0_5 , PY ); // TBL0[5] * y
//
//           TMP0 = _mm_add_ps( _TBL1_0 , TMP0 ); // TBL0[0] * x + TBL1[0]
//           TMP1 = _mm_add_ps( _TBL1_1 , TMP1 ); // TBL0[1] * y + TBL1[1]
//           TMP2 = _mm_add_ps( _TBL1_2 , TMP2 ); // TBL0[2] * x + TBL1[2]
//           TMP3 = _mm_add_ps( _TBL1_3 , TMP3 ); // TBL0[3] * y + TBL1[3]
//           TMP4 = _mm_add_ps( _TBL1_4 , TMP4 ); // TBL0[4] * x + TBL1[4]
//           TMP5 = _mm_add_ps( _TBL1_5 , TMP5 ); // TBL0[5] * y + TBL1[5]
//
//    __m128 ABP  = _mm_add_ps( TMP0 , TMP1 ); // ABP = TBL0[0] * x + TBL1[0] + TBL0[1] * y + TBL1[1]
//    __m128 BCP  = _mm_add_ps( TMP2 , TMP3 ); // BCP = TBL0[2] * x + TBL1[2] + TBL0[3] * y + TBL1[3]
//    __m128 CAP  = _mm_add_ps( TMP4 , TMP5 ); // CAP = TBL0[4] * x + TBL1[4] + TBL0[5] * y + TBL1[5]
//
//
//    __m128 ZERO = _mm_setzero_ps(); // Create a zero vector for comparison
//    __m128 ABP_Z= _mm_cmplt_ps( ABP , ZERO ); // Compare ABP with zero
//    __m128 BCP_Z= _mm_cmplt_ps( BCP , ZERO ); // Compare BCP with zero
//    __m128 CAP_Z= _mm_cmplt_ps( CAP , ZERO ); // Compare CAP with zero
//
//
//    __m128 SKIP = _mm_or_ps( ABP_Z , BCP_Z ); // Combine the results of the comparisons
//           SKIP = _mm_or_ps( SKIP  , CAP_Z ); // Combine the results of the comparisons
//
//    _mm_store_ps( pResult     , SKIP ); // Store the skip results in pResult
//    _mm_store_ps( pResult + 8 , ABP  ); // Store ABP results in pResult + 8
//    _mm_store_ps( pResult + 16, BCP  ); // Store BCP results in pResult + 16
//    _mm_store_ps( pResult + 24, CAP  ); // Store CAP results in pResult + 24
//
//           PX = _mm_set_ps( P.x+7 , P.x+6 , P.x+5 , P.x+4 );
//
//           TMP0 = _mm_mul_ps( _TBL0_0 , PX ); // TBL0[0] * x
//           TMP1 = _mm_mul_ps( _TBL0_1 , PY ); // TBL0[1] * y
//           TMP2 = _mm_mul_ps( _TBL0_2 , PX ); // TBL0[2] * x
//           TMP3 = _mm_mul_ps( _TBL0_3 , PY ); // TBL0[3] * y
//           TMP4 = _mm_mul_ps( _TBL0_4 , PX ); // TBL0[4] * x
//           TMP5 = _mm_mul_ps( _TBL0_5 , PY ); // TBL0[5] * y
//
//           TMP0 = _mm_add_ps( _TBL1_0 , TMP0 ); // TBL0[0] * x + TBL1[0]
//           TMP1 = _mm_add_ps( _TBL1_1 , TMP1 ); // TBL0[1] * y + TBL1[1]
//           TMP2 = _mm_add_ps( _TBL1_2 , TMP2 ); // TBL0[2] * x + TBL1[2]
//           TMP3 = _mm_add_ps( _TBL1_3 , TMP3 ); // TBL0[3] * y + TBL1[3]
//           TMP4 = _mm_add_ps( _TBL1_4 , TMP4 ); // TBL0[4] * x + TBL1[4]
//           TMP5 = _mm_add_ps( _TBL1_5 , TMP5 ); // TBL0[5] * y + TBL1[5]
//
//           ABP  = _mm_add_ps( TMP0 , TMP1 ); // ABP = TBL0[0] * x + TBL1[0] + TBL0[1] * y + TBL1[1]
//           BCP  = _mm_add_ps( TMP2 , TMP3 ); // BCP = TBL0[2] * x + TBL1[2] + TBL0[3] * y + TBL1[3]
//           CAP  = _mm_add_ps( TMP4 , TMP5 ); // CAP = TBL0[4] * x + TBL1[4] + TBL0[5] * y + TBL1[5]
//
//           ABP_Z= _mm_cmplt_ps( ABP , ZERO ); // Compare ABP with zero
//           BCP_Z= _mm_cmplt_ps( BCP , ZERO ); // Compare BCP with zero
//           CAP_Z= _mm_cmplt_ps( CAP , ZERO ); // Compare CAP with zero
//
//
//           SKIP = _mm_or_ps( ABP_Z , BCP_Z ); // Combine the results of the comparisons
//           SKIP = _mm_or_ps( SKIP  , CAP_Z ); // Combine the results of the comparisons
//
//    _mm_store_ps( pResult + 4 , SKIP ); // Store the skip results in pResult
//    _mm_store_ps( pResult + 12, ABP  ); // Store ABP results in pResult + 12
//    _mm_store_ps( pResult + 20, BCP  ); // Store BCP results in pResult + 20
//    _mm_store_ps( pResult + 28, CAP  ); // Store CAP results in pResult + 28
//}




inline void MathSSE::EdgeFunction3xToBoolx8( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const
{
//   A = TBL0[0] * x + TBL1[0]
//     + TBL0[1] * y + TBL1[1]
//
//   B = TBL0[2] * x + TBL1[2]
//     + TBL0[3] * y + TBL1[3]
//
//   C = TBL0[4] * x + TBL1[4]
//     + TBL0[5] * y + TBL1[5]



    __m128 TBL__0;
    __m128 TBL__1;
    __m128 TBL__2;
    __m128 TBL__3;
    __m128 YY;

    const __m128 ZERO  = _mm_setzero_ps();
    const __m128 PY    = _mm_set_ps( P.y   , P.y   , P.y   , P.y   );
    const __m128 PX1   = _mm_set_ps( P.x+3 , P.x+2 , P.x+1 , P.x   );
    const __m128 PX2   = _mm_set_ps( P.x+7 , P.x+6 , P.x+5 , P.x+4 );

    __m128 A1;
    __m128 A2;
    {

                TBL__0 = _mm_load1_ps(TBL0+0);
                TBL__1 = _mm_load1_ps(TBL0+1);
                TBL__2 = _mm_load1_ps(TBL1+0);
                TBL__3 = _mm_load1_ps(TBL1+1);

                    A1 = _mm_mul_ps( TBL__0 , PX1 ); // TBL0[0] * x1
                    A2 = _mm_mul_ps( TBL__0 , PX2 ); // TBL0[0] * x2
                    YY = _mm_mul_ps( TBL__1 ,  PY ); // TBL0[1] * y

                    A1 = _mm_add_ps( TBL__2 , A1  ); // TBL0[0] * x1 + TBL1[0]
                    A2 = _mm_add_ps( TBL__2 , A2  ); // TBL0[0] * x2 + TBL1[0]
                    YY = _mm_add_ps( TBL__3 , YY  ); // TBL0[1] * y  + TBL1[1]
    }

    __m128 B1;
    __m128 B2;
    {

                TBL__0 = _mm_load1_ps(TBL0+2);
                TBL__1 = _mm_load1_ps(TBL0+3);
                TBL__2 = _mm_load1_ps(TBL1+2);
                TBL__3 = _mm_load1_ps(TBL1+3);

                    A1 = _mm_add_ps( A1 , YY ); // A1 = TBL0[0] * x1 + TBL1[0] + TBL0[1] * y + TBL1[1]
                    A2 = _mm_add_ps( A2 , YY ); // A2 = TBL0[0] * x2 + TBL1[0] + TBL0[1] * y + TBL1[1]

                _mm_store_ps( pResult + 8  , A1  );
                _mm_store_ps( pResult + 12 , A2  );


                    B1 = _mm_mul_ps( TBL__0 , PX1 ); // TBL0[2] * x1
                    B2 = _mm_mul_ps( TBL__0 , PX2 ); // TBL0[2] * x2
                    YY = _mm_mul_ps( TBL__1 , PY  ); // TBL0[3] * y

                    B1 = _mm_add_ps( TBL__2 , B1  ); // TBL0[2] * x1 + TBL1[2]
                    B2 = _mm_add_ps( TBL__2 , B2  ); // TBL0[2] * x2 + TBL1[2]
                    YY = _mm_add_ps( TBL__3 , YY  ); // TBL0[3] * y  + TBL1[3]

                    _mm_store_ps( pResult + 16 , B1  );
                    _mm_store_ps( pResult + 20 , B2  );
    }
    __m128 C1;
    __m128 C2;
    {

                TBL__0 = _mm_load1_ps(TBL0+4);
                TBL__1 = _mm_load1_ps(TBL0+5);
                TBL__2 = _mm_load1_ps(TBL1+4);
                TBL__3 = _mm_load1_ps(TBL1+5);

                    A1 = _mm_cmplt_ps( A1 , ZERO );
                    A2 = _mm_cmplt_ps( A2 , ZERO );

                    B1 = _mm_add_ps( B1 , YY ); // B1 = TBL0[2] * x1 + TBL1[2] + TBL0[3] * y + TBL1[3]
                    B2 = _mm_add_ps( B2 , YY ); // B2 = TBL0[2] * x2 + TBL1[2] + TBL0[3] * y + TBL1[3]


                    C1 = _mm_mul_ps( TBL__0 , PX1 ); // TBL0[4] * x1
                    C2 = _mm_mul_ps( TBL__0 , PX2 ); // TBL0[4] * x2
                    YY = _mm_mul_ps( TBL__1 ,  PY ); // TBL0[5] * y

                    C1 = _mm_add_ps( TBL__2 , C1  ); // TBL0[4] * x1 + TBL1[4]
                    C2 = _mm_add_ps( TBL__2 , C2  ); // TBL0[4] * x2 + TBL1[4]
                    YY = _mm_add_ps( TBL__3 , YY  ); // TBL0[5] * y  + TBL1[5]

                    B1 = _mm_cmplt_ps( B1 , ZERO );
                    B2 = _mm_cmplt_ps( B2 , ZERO );
    }


    C1 = _mm_add_ps( C1 , YY ); // B1 = TBL0[4] * x1 + TBL1[4] + TBL0[5] * y + TBL1[5]
    C2 = _mm_add_ps( C2 , YY ); // B2 = TBL0[4] * x2 + TBL1[4] + TBL0[5] * y + TBL1[5]

    _mm_store_ps( pResult + 24 , C1  );
    _mm_store_ps( pResult + 28 , C2  );

    C1 = _mm_cmplt_ps( C1 , ZERO );
    C2 = _mm_cmplt_ps( C2 , ZERO );

    A1 = _mm_or_ps( A1 , B1 );
    A2 = _mm_or_ps( A2 , B2 );
    A1 = _mm_or_ps( A1 , C1 );
    A2 = _mm_or_ps( A2 , C2 );

    _mm_store_ps( pResult + 0  , A1 );
    _mm_store_ps( pResult + 4  , A2 );
}


//
//inline void MathSSE::EdgeFunction3xToBoolx8( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const
//{
////   A = TBL0[0] * x + TBL1[0]
////   A = TBL0[1] * y + TBL1[1]
////
////   A = TBL0[2] * x + TBL1[2]
////   A = TBL0[3] * y + TBL1[3]
////
////   A = TBL0[4] * x + TBL1[4]
////   A = TBL0[5] * y + TBL1[5]
//
//
//
//const __m128 PY  = _mm_set_ps( P.y   , P.y   , P.y   , P.y   );
//
//
//    __m128 TBL1_Y;
//    __m128 TBL3_Y;
//    __m128 TBL5_Y;
//
//    {
//        __m128 PX      = _mm_set_ps( P.x+3 , P.x+2 , P.x+1 , P.x   );
//        __m128 TMP_0;
//        __m128 TMP_1;
//        __m128 TMP_2;
//
//        {
//            __m128 TBL_0_0 = _mm_load1_ps(TBL0+0);
//            __m128 TBL_0_1 = _mm_load1_ps(TBL0+1);
//            __m128 TBL_0_2 = _mm_load1_ps(TBL0+2);
//
//            __m128 TBL_1_0 = _mm_load1_ps(TBL1+0);
//            __m128 TBL_1_1 = _mm_load1_ps(TBL1+1);
//            __m128 TBL_1_2 = _mm_load1_ps(TBL1+2);
//
//                   TMP_0  = _mm_mul_ps( TBL_0_0 , PX ); // TBL0[0] * x
//                   TBL1_Y = _mm_mul_ps( TBL_0_1 , PY ); // TBL0[1] * y
//                   TMP_1  = _mm_mul_ps( TBL_0_2 , PX ); // TBL0[2] * x
//
//                   TMP_0 = _mm_add_ps( TBL_1_0 , TMP_0  ); // TBL0[0] * x
//                  TBL1_Y = _mm_add_ps( TBL_1_1 , TBL1_Y ); // TBL0[1] * y
//                   TMP_1 = _mm_add_ps( TBL_1_2 , TMP_1  ); // TBL0[2] * x
//        }
//
//        {
//            __m128 TBL_0_0 = _mm_load1_ps(TBL0+3);
//            __m128 TBL_0_1 = _mm_load1_ps(TBL0+4);
//            __m128 TBL_0_2 = _mm_load1_ps(TBL0+5);
//
//            __m128 TBL_1_0 = _mm_load1_ps(TBL1+3);
//            __m128 TBL_1_1 = _mm_load1_ps(TBL1+4);
//            __m128 TBL_1_2 = _mm_load1_ps(TBL1+5);
//
//                  TBL3_Y = _mm_mul_ps( TBL_0_0 , PY ); // TBL0[3] * y
//                  TMP_2  = _mm_mul_ps( TBL_0_1 , PX ); // TBL0[0] * x
//                  TBL5_Y = _mm_mul_ps( TBL_0_2 , PY ); // TBL0[2] * x
//
//                  TBL3_Y = _mm_add_ps( TBL_1_0 , TBL3_Y ); // TBL0[3] * y
//                   TMP_2 = _mm_add_ps( TBL_1_1 , TMP_2  ); // TBL0[0] * x
//                  TBL5_Y = _mm_add_ps( TBL_1_2 , TBL5_Y ); // TBL0[2] * x
//        }
//
//        __m128 ABP   = _mm_add_ps( TMP_0 , TBL1_Y); // ABP = TBL0[0] * x + TBL1[0] + TBL0[1] * y + TBL1[1]
//        __m128 BCP   = _mm_add_ps( TMP_1 , TBL3_Y); // BCP = TBL0[2] * x + TBL1[2] + TBL0[3] * y + TBL1[3]
//        __m128 CAP   = _mm_add_ps( TMP_2 , TBL5_Y); // ABP = TBL0[0] * x + TBL1[0] + TBL0[1] * y + TBL1[1]
//
//        __m128 ZERO = _mm_setzero_ps(); // Create a zero vector for comparison
//
//              TMP_0 = _mm_cmplt_ps( ABP , ZERO ); // Compare ABP with zero
//              TMP_1 = _mm_cmplt_ps( BCP , ZERO ); // Compare BCP with zero
//              TMP_2 = _mm_cmplt_ps( CAP , ZERO ); // Compare CAP with zero
//
//        __m128 SKIP = _mm_or_ps( TMP_0 , TMP_1 );
//               SKIP = _mm_or_ps( SKIP  , TMP_2 );
//
//        _mm_store_ps( pResult + 0  , SKIP );
//        _mm_store_ps( pResult + 8  , ABP  );
//        _mm_store_ps( pResult + 16 , BCP  );
//        _mm_store_ps( pResult + 24 , CAP  );
//    }
//
//
//
//    {
//         __m128 PX      = _mm_set_ps( P.x+7 , P.x+6 , P.x+5 , P.x+4 );
//        __m128 TMP_0;
//        __m128 TMP_1;
//        __m128 TMP_2;
//        {
//
//         __m128 TBL_0_0 = _mm_load1_ps(TBL0+0);
//         __m128 TBL_0_2 = _mm_load1_ps(TBL0+2);
//         __m128 TBL_0_4 = _mm_load1_ps(TBL0+4);
//         __m128 TBL_1_0 = _mm_load1_ps(TBL1+0);
//         __m128 TBL_1_2 = _mm_load1_ps(TBL1+2);
//         __m128 TBL_1_4 = _mm_load1_ps(TBL1+4);
//
//               TMP_0  = _mm_mul_ps( TBL_0_0 , PX ); // TBL0[0] * x
//               TMP_1  = _mm_mul_ps( TBL_0_2 , PX ); // TBL0[2] * x
//               TMP_2  = _mm_mul_ps( TBL_0_4 , PX ); // TBL0[0] * x
//
//               TMP_0  = _mm_add_ps( TBL_1_0 , TMP_0  ); // TBL0[0] * x
//               TMP_1  = _mm_add_ps( TBL_1_2 , TMP_1  ); // TBL0[2] * x
//               TMP_2  = _mm_add_ps( TBL_1_4 , TMP_2  ); // TBL0[0] * x
//        }
//        {
//         __m128 ABP = _mm_add_ps( TMP_0 , TBL1_Y); // ABP = TBL0[0] * x + TBL1[0] + TBL0[1] * y + TBL1[1]
//         __m128 BCP = _mm_add_ps( TMP_1 , TBL3_Y); // BCP = TBL0[2] * x + TBL1[2] + TBL0[3] * y + TBL1[3]
//         __m128 CAP = _mm_add_ps( TMP_2 , TBL5_Y); // ABP = TBL0[0] * x + TBL1[0] + TBL0[1] * y + TBL1[1]
//
//        __m128 ZERO = _mm_setzero_ps(); // Create a zero vector for comparison
//
//              TMP_0 = _mm_cmplt_ps( ABP , ZERO ); // Compare ABP with zero
//              TMP_1 = _mm_cmplt_ps( BCP , ZERO ); // Compare BCP with zero
//              TMP_2 = _mm_cmplt_ps( CAP , ZERO ); // Compare CAP with zero
//
//        __m128 SKIP = _mm_or_ps( TMP_0 , TMP_1 );
//               SKIP = _mm_or_ps( SKIP  , TMP_2 );
//
//            _mm_store_ps( pResult + 4  , SKIP );
//            _mm_store_ps( pResult + 12 , ABP  );
//            _mm_store_ps( pResult + 20 , BCP  );
//            _mm_store_ps( pResult + 28 , CAP  );
//        }
//    }
//}

//
//inline void MathSSE::EdgeFunction3xToBoolx8( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const
//{
////   A = TBL0[0] * x + TBL1[0]
////   A = TBL0[1] * y + TBL1[1]
////
////   A = TBL0[2] * x + TBL1[2]
////   A = TBL0[3] * y + TBL1[3]
////
////   A = TBL0[4] * x + TBL1[4]
////   A = TBL0[5] * y + TBL1[5]
//
//
//
//    __m128 PX  = _mm_set_ps( P.x+3 , P.x+2 , P.x+1 , P.x   );
//    __m128 PY  = _mm_set_ps( P.y   , P.y   , P.y   , P.y   );
//
//    __m128 TBL_0  = _mm_load1_ps(TBL0+0);
//    __m128 TBL_1  = _mm_load1_ps(TBL0+1);
//    __m128 TBL_2  = _mm_load1_ps(TBL0+2);
//    __m128 TBL_3  = _mm_load1_ps(TBL0+3);
//    __m128 TBL_4  = _mm_load1_ps(TBL0+4);
//    __m128 TBL_5  = _mm_load1_ps(TBL0+5);
//
//    __m128 TMP_0  = _mm_mul_ps( TBL_0 , PX ); // TBL0[0] * x
//    __m128 TBL1_Y = _mm_mul_ps( TBL_1 , PY ); // TBL0[1] * y
//    __m128 TMP_2  = _mm_mul_ps( TBL_2 , PX ); // TBL0[2] * x
//    __m128 TBL3_Y = _mm_mul_ps( TBL_3 , PY ); // TBL0[3] * y
//    __m128 TMP_4  = _mm_mul_ps( TBL_4 , PX ); // TBL0[0] * x
//    __m128 TBL5_Y = _mm_mul_ps( TBL_5 , PY ); // TBL0[2] * x
//
//           TBL_0 = _mm_load1_ps(TBL1+0);
//           TBL_1 = _mm_load1_ps(TBL1+1);
//           TBL_2 = _mm_load1_ps(TBL1+2);
//           TBL_3 = _mm_load1_ps(TBL1+3);
//           TBL_4 = _mm_load1_ps(TBL1+4);
//           TBL_5 = _mm_load1_ps(TBL1+5);
//
//           TMP_0 = _mm_mul_ps( TBL_0 , TMP_0  ); // TBL0[0] * x
//    __m128 TMP_1 = _mm_mul_ps( TBL_1 , TBL1_Y ); // TBL0[1] * y
//           TMP_2 = _mm_mul_ps( TBL_2 , TMP_2  ); // TBL0[2] * x
//    __m128 TMP_3 = _mm_mul_ps( TBL_3 , TBL3_Y ); // TBL0[3] * y
//           TMP_4 = _mm_mul_ps( TBL_4 , TMP_4  ); // TBL0[0] * x
//    __m128 TMP_5 = _mm_mul_ps( TBL_5 , TBL5_Y ); // TBL0[2] * x
//
//    __m128 ABP_03 = _mm_add_ps( TMP_0 , TMP_1 ); // ABP = TBL0[0] * x + TBL1[0] + TBL0[1] * y + TBL1[1]
//    __m128 BCP_03 = _mm_add_ps( TMP_2 , TMP_3 ); // BCP = TBL0[2] * x + TBL1[2] + TBL0[3] * y + TBL1[3]
//    __m128 CAP_03 = _mm_add_ps( TMP_4 , TMP_5 ); // ABP = TBL0[0] * x + TBL1[0] + TBL0[1] * y + TBL1[1]
//
//
//           PX     = _mm_set_ps( P.x+7 , P.x+6 , P.x+5 , P.x+4 );
//
//           TBL_0  = _mm_load1_ps(TBL0+0);
//           TBL_2  = _mm_load1_ps(TBL0+2);
//           TBL_4  = _mm_load1_ps(TBL0+4);
//
//           TMP_0  = _mm_mul_ps( TBL_0 , PX ); // TBL0[0] * x
//           TMP_2  = _mm_mul_ps( TBL_2 , PX ); // TBL0[2] * x
//           TMP_4  = _mm_mul_ps( TBL_4 , PX ); // TBL0[0] * x
//
//           TBL_0 = _mm_load1_ps(TBL1+0);
//           TBL_1 = _mm_load1_ps(TBL1+1);
//           TBL_2 = _mm_load1_ps(TBL1+2);
//           TBL_3 = _mm_load1_ps(TBL1+3);
//           TBL_4 = _mm_load1_ps(TBL1+4);
//           TBL_5 = _mm_load1_ps(TBL1+5);
//
//           TMP_0 = _mm_mul_ps( TBL_0 , TMP_0  ); // TBL0[0] * x
//           TMP_1 = _mm_mul_ps( TBL_1 , TBL1_Y ); // TBL0[1] * y
//           TMP_2 = _mm_mul_ps( TBL_2 , TMP_2  ); // TBL0[2] * x
//           TMP_3 = _mm_mul_ps( TBL_3 , TBL3_Y ); // TBL0[3] * y
//           TMP_4 = _mm_mul_ps( TBL_4 , TMP_4  ); // TBL0[0] * x
//           TMP_5 = _mm_mul_ps( TBL_5 , TBL5_Y ); // TBL0[2] * x
//
//    __m128 ABP_47 = _mm_add_ps( TMP_0 , TMP_1 ); // ABP = TBL0[0] * x + TBL1[0] + TBL0[1] * y + TBL1[1]
//    __m128 BCP_47 = _mm_add_ps( TMP_2 , TMP_3 ); // BCP = TBL0[2] * x + TBL1[2] + TBL0[3] * y + TBL1[3]
//    __m128 CAP_47 = _mm_add_ps( TMP_4 , TMP_5 ); // ABP = TBL0[0] * x + TBL1[0] + TBL0[1] * y + TBL1[1]
//
//
//    __m128 ZERO = _mm_setzero_ps(); // Create a zero vector for comparison
//
//          TMP_0 = _mm_cmplt_ps( ABP_03 , ZERO ); // Compare ABP with zero
//          TMP_1 = _mm_cmplt_ps( BCP_03 , ZERO ); // Compare BCP with zero
//          TMP_2 = _mm_cmplt_ps( CAP_03 , ZERO ); // Compare CAP with zero
//
//          TMP_3 = _mm_cmplt_ps( ABP_47 , ZERO ); // Compare ABP with zero
//          TMP_4 = _mm_cmplt_ps( BCP_47 , ZERO ); // Compare BCP with zero
//          TMP_5 = _mm_cmplt_ps( CAP_47 , ZERO ); // Compare CAP with zero
//
//
//          TMP_0 = _mm_or_ps( TMP_0 , TMP_1 );
//          TMP_3 = _mm_or_ps( TMP_3 , TMP_4 );
//
//          TMP_0 = _mm_or_ps( TMP_0 , TMP_2 );
//          TMP_3 = _mm_or_ps( TMP_3 , TMP_5 );
//
//    _mm_store_ps( pResult + 0  , TMP_0  );
//    _mm_store_ps( pResult + 4  , TMP_3  );
//    _mm_store_ps( pResult + 8  , ABP_03 );
//    _mm_store_ps( pResult + 12 , ABP_47 );
//    _mm_store_ps( pResult + 16 , BCP_03 );
//    _mm_store_ps( pResult + 20 , BCP_47 );
//    _mm_store_ps( pResult + 24 , CAP_03 );
//    _mm_store_ps( pResult + 28 , CAP_47 );
//}

//inline bool MathSSE::EdgeFunction3xToBool( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const
//{
//    // Interleave8( P.x , P.y , Arg1 );
//    // MulAddVec8( Arg0 , Arg1 , Arg2 , Arg1 );
//    // PairAddVec4( Arg1 , Arg1 + 4 , &EdgeResult.ABP );
//
//    __m128 PS  = _mm_set_ps ( P.y , P.x , P.y , P.x );                       // PS ->  x , y , x , y
//    __m128 T00 = _mm_load_ps(TBL0);
//    __m128 T01 = _mm_load_ps(TBL0+4);
//    __m128 T10 = _mm_load_ps(TBL1);
//    __m128 T11 = _mm_load_ps(TBL1+4);
//
//    __m128 a1 = _mm_mul_ps( T00 , PS );
//    __m128 b1 = _mm_mul_ps( T01 , PS );
//           a1 = _mm_add_ps( T10 , a1 );
//           b1 = _mm_add_ps( T11 , b1 );
//
//
//
//            // TBL0[0] * x + TBL1[0]
//            // TBL0[1] * y + TBL1[1]
//            // TBL0[2] * x + TBL1[2]
//            // TBL0[3] * y + TBL1[3]
//
//            // TBL0[4] * x + TBL1[4]
//            // TBL0[5] * y + TBL1[5]
//            // TBL0[6] * x + TBL1[6]
//            // TBL0[7] * y + TBL1[7]
//
//
//    a1 = _mm_hadd_ps( a1 , b1 );
//    PS = _mm_setzero_ps();
//    PS = _mm_cmplt_ps(a1, PS); // compare each element of a1 with 0.0f
//    int mask = _mm_movemask_ps(PS); // get the sign bits of the first 4 elements of PS as a bitmask
//    _mm_store_ps(pResult, a1); // store the result in pResult
//    // is any of the first 3 elements < 0.0f?
//    return (mask & 0x7) != 0;
//}


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

inline void MathAVX::MultiplyVec4ByScalar884( const float* v, float scalar , float* pOut )const
{
    __m256 a = _mm256_load_ps (v);          // _mm_loadu_ps (float const* mem_addr)
    __m256 b = _mm256_set1_ps (scalar);     // _mm_set_ps1 (float a)
    __m256 c = _mm256_mul_ps  (a, b);       // _mm_mul_ps (__m128 a, __m128 b)
                _mm_store_ps(pOut, *(__m128*) &c);      // _mm_storeu_ps (float* mem_addr, __m128 a)
    //memccpy(pOut, &c, 0, 16); // Copy the first 4 elements to pOut
}


inline void MathAVX::AddVec4ToScalar( const float* v, float scalar , float* pOut )const
{
    __m128 a = _mm_load_ps (v);             // _mm_loadu_ps (float const* mem_addr)
    __m128 b = _mm_set1_ps (scalar);        // _mm_set_ps1 (float a)
    __m128 c = _mm_add_ps  (a, b);          // _mm_mul_ps (__m128 a, __m128 b)
                _mm_store_ps(pOut, c);      // _mm_storeu_ps (float* mem_addr, __m128 a)
}

inline void MathAVX::AddVec8ToScalar( const float* v, float scalar , float* pOut )const
{
    __m256 a = _mm256_load_ps (v);          // _mm_loadu_ps (float const* mem_addr)
    __m256 b = _mm256_set1_ps (scalar);     // _mm_set_ps1 (float a)
    __m256 c = _mm256_add_ps  (a, b);       // _mm_mul_ps (__m128 a, __m128 b)
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

inline void MathAVX::PairAddVec4( const float* a, const float* b , float* pOut )const
{
    __m128 a1 = _mm_load_ps (a);         // _mm_load_ps (float const* mem_addr)
    __m128 b1 = _mm_load_ps (b);         // _mm_load_ps (float const* mem_addr)
    __m128 c  = _mm_hadd_ps (a1, b1);    // _mm_hadd_ps (__m128 a, __m128 b)
                _mm_store_ps(pOut, c);   // _mm_store_ps (float* mem_addr, __m128 a)
}

inline void MathAVX::PairAddVec8( const float* a, const float* b , float* pOut )const
{
    __m256 a1 = _mm256_load_ps (a);         // _mm_load_ps (float const* mem_addr)
    __m256 b1 = _mm256_load_ps (b);         // _mm_load_ps (float const* mem_addr)
    __m256 c  = _mm256_hadd_ps (a1, b1);    // _mm_hadd_ps (__m128 a, __m128 b)
                _mm256_store_ps(pOut, c);   // _mm_store_ps (float* mem_addr, __m128 a)
}

inline void MathAVX::LessThan4( const float* a, const float* b , float* pOut )const
{
    __m128 ma = _mm_load_ps (a);
    __m128 mb = _mm_load_ps (b);
    __m128 mc = _mm_cmplt_ps(ma, mb);
                _mm_store_ps(pOut, mc);
}

inline void MathAVX::LessThan8( const float* a, const float* b , float* pOut )const
{
    __m256 ma = _mm256_load_ps (a);
    __m256 mb = _mm256_load_ps (b);
    __m256 mc = _mm256_cmp_ps  (ma, mb, _CMP_LT_OQ);
                _mm256_store_ps(pOut, mc);
}

inline void MathAVX::BitOr4( const float* a, const float* b , float* pOut )const
{
    __m128 ma = _mm_load_ps (a);
    __m128 mb = _mm_load_ps (b);
    __m128 mc = _mm_or_ps(ma, mb);
                _mm_store_ps(pOut, mc);
}

inline void MathAVX::BitOr8( const float* a, const float* b , float* pOut )const
{
    __m256 ma = _mm256_load_ps (a);
    __m256 mb = _mm256_load_ps (b);
    __m256 mc = _mm256_or_ps(ma, mb);
                _mm256_store_ps(pOut, mc);
}

inline void MathAVX::Interleave4( float a, float b , float* pOut )const
{
    __m128 a1 = _mm_set1_ps (a);
    __m128 b1 = _mm_set1_ps (b);
    __m128 r1 = _mm_blend_ps( a1 , b1 , 0xA );
                _mm_store_ps(pOut, r1);
}

inline void MathAVX::Interleave8( float a, float b , float* pOut )const
{
    __m256 a1 = _mm256_set1_ps (a);
    __m256 b1 = _mm256_set1_ps (b);
    __m256 r1 = _mm256_blend_ps( a1 , b1 , 0xAA );
                _mm256_store_ps(pOut, r1);
}

inline void MathAVX::EdgeFunction3x( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const
{
    __m256 a1 = _mm256_set1_ps (P.x);              //  x ->  x , x , x , x , x , x , x , x
    __m256 b1 = _mm256_set1_ps (P.y);              //  y ->  y , y , y , y , y , y , y , y
    __m256 PS = _mm256_blend_ps( a1 , b1 , 0xAA ); // PS ->  x , y , x , y , x , y , x , y

    PS = _mm256_fmadd_ps( reinterpret_cast<__m256*>(TBL0)[0] , PS , reinterpret_cast<__m256*>(TBL1)[0] ); // TBL0[0] * x + TBL1[0]
                                                                                                          // TBL0[1] * y + TBL1[1]
                                                                                                          // TBL0[2] * x + TBL1[2]
                                                                                                          // TBL0[3] * y + TBL1[3]
                                                                                                          // TBL0[4] * x + TBL1[4]
                                                                                                          // TBL0[5] * y + TBL1[5]
                                                                                                          // TBL0[6] * x + TBL1[6]
                                                                                                          // TBL0[7] * y + TBL1[7]

    reinterpret_cast<__m128*>(pResult)[0] = _mm_hadd_ps( reinterpret_cast<__m128*>(&PS)[0] , reinterpret_cast<__m128*>(&PS)[1] );
}

inline bool MathAVX::EdgeFunction3xToBool( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const
{
    __m256 a1 = _mm256_set1_ps (P.x);              //  x ->  x , x , x , x , x , x , x , x
    __m256 b1 = _mm256_set1_ps (P.y);              //  y ->  y , y , y , y , y , y , y , y
    __m256 PS = _mm256_blend_ps( a1 , b1 , 0xAA ); // PS ->  x , y , x , y , x , y , x , y

    PS = _mm256_fmadd_ps( reinterpret_cast<__m256*>(TBL0)[0] , PS , reinterpret_cast<__m256*>(TBL1)[0] ); // TBL0[0] * x + TBL1[0]
                                                                                                          // TBL0[1] * y + TBL1[1]
                                                                                                          // TBL0[2] * x + TBL1[2]
                                                                                                          // TBL0[3] * y + TBL1[3]
                                                                                                          // TBL0[4] * x + TBL1[4]
                                                                                                          // TBL0[5] * y + TBL1[5]
                                                                                                          // TBL0[6] * x + TBL1[6]
                                                                                                          // TBL0[7] * y + TBL1[7]


    // a1[0]  = a1[0] + a1[1]; // x + y
    // a1[2]  = a1[2] + a1[3]; // x + y
    // a1[4]  = a1[4] + a1[5]; // x + y
    // a1[6]  = a1[6] + a1[7]; // x + y

    {
        auto x = reinterpret_cast<__m128*>(&PS);
        auto a1 = _mm_hadd_ps( x[0] , x[1] );
        auto b1 = _mm_setzero_ps();
        auto PS = _mm_cmplt_ps(a1, b1); // compare each element of a1 with 0.0f
        int mask = _mm_movemask_ps(PS); // get the sign bits of the first 4 elements of PS as a bitmask
        _mm_store_ps(pResult, a1); // store the result in pResult
        return (mask & 0b0111) != 0; // is any of the first 3 elements < 0.0f?
    }
    //{
    //    a1 = _mm256_hadd_ps( PS , PS );
    //    b1 = _mm256_setzero_ps();
    //    PS = _mm256_cmp_ps(a1, b1, _CMP_LT_OQ); // compare each element of a1 with 0.0f
    //    int mask = _mm256_movemask_ps(PS); // get the sign bits of the first 4 elements of PS as a bitmask

    //    //_mm256_store_ps(pResult, a1); // store the result in pResult

    //    pResult[0] = a1.m256_f32[0];
    //    pResult[1] = a1.m256_f32[1];
    //    pResult[2] = a1.m256_f32[4];

    //    // is any of the first 3 elements < 0.0f?
    //    return (mask & 0b1011) != 0;
    //}
}



inline void MathAVX::EdgeFunction3xToBoolx8( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const
{
//   A = TBL0[0] * x + TBL1[0]
//     + TBL0[1] * y + TBL1[1]
//
//   B = TBL0[2] * x + TBL1[2]
//     + TBL0[3] * y + TBL1[3]
//
//   C = TBL0[4] * x + TBL1[4]
//     + TBL0[5] * y + TBL1[5]

    const __m256 PY    = _mm256_set1_ps(P.y);
    const __m256 PX    = _mm256_add_ps(
                            _mm256_set1_ps(P.x),
                            _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f)
                        );

        __m256 TBL_0_0 = _mm256_set1_ps(TBL0[0]);
        __m256 TBL_0_1 = _mm256_set1_ps(TBL0[1]);
        __m256 TBL_0_2 = _mm256_set1_ps(TBL0[2]);
        __m256 TBL_1_0 = _mm256_set1_ps(TBL1[0]);
        __m256 TBL_1_1 = _mm256_set1_ps(TBL1[1]);
        __m256 TBL_1_2 = _mm256_set1_ps(TBL1[2]);

             __m256 AX = _mm256_fmadd_ps( TBL_0_0 , PX , TBL_1_0 ); // TBL0[0] * x1
             __m256 AY = _mm256_fmadd_ps( TBL_0_1 , PY , TBL_1_1 ); // TBL0[1] * y
             __m256 BX = _mm256_fmadd_ps( TBL_0_2 , PX , TBL_1_2 ); // TBL0[2] * x1

        __m256 TBL_0_3 = _mm256_set1_ps(TBL0[3]);
        __m256 TBL_0_4 = _mm256_set1_ps(TBL0[4]);
        __m256 TBL_0_5 = _mm256_set1_ps(TBL0[5]);
        __m256 TBL_1_3 = _mm256_set1_ps(TBL1[3]);
        __m256 TBL_1_4 = _mm256_set1_ps(TBL1[4]);
        __m256 TBL_1_5 = _mm256_set1_ps(TBL1[5]);

             __m256 BY = _mm256_fmadd_ps( TBL_0_3 , PY , TBL_1_3 ); // TBL0[3] * y
             __m256 CX = _mm256_fmadd_ps( TBL_0_4 , PX , TBL_1_4 ); // TBL0[4] * x1
             __m256 CY = _mm256_fmadd_ps( TBL_0_5 , PY , TBL_1_5 ); // TBL0[5] * y

                    AX = _mm256_add_ps( AX , AY ); // A1 = TBL0[0] * x1 + TBL1[0] + TBL0[1] * y + TBL1[1]
                    _mm256_store_ps( pResult + 8  , AX  );
                    BX = _mm256_add_ps( BX , BY ); // A1 = TBL0[0] * x1 + TBL1[0] + TBL0[1] * y + TBL1[1]
                    _mm256_store_ps( pResult + 16 , BX  );
                    CX = _mm256_add_ps( CX , CY ); // A1 = TBL0[0] * x1 + TBL1[0] + TBL0[1] * y + TBL1[1]
                    _mm256_store_ps( pResult + 24 , CX  );


    // comparison with zero is redundant here, because we only need logical OR of the sign bits
    //
    //const __m256 ZERO  = _mm256_setzero_ps();
    //
    //
    //AX = _mm256_cmp_ps( AX , ZERO , _CMP_LT_OQ );
    //BX = _mm256_cmp_ps( BX , ZERO , _CMP_LT_OQ );
    //CX = _mm256_cmp_ps( CX , ZERO , _CMP_LT_OQ );

    AX = _mm256_or_ps( AX , BX );
    AX = _mm256_or_ps( AX , CX );

    _mm256_store_ps( pResult + 0  , AX  );
}


//inline void MathAVX::EdgeFunction3xToBoolx8( const Vector2f& P , float* TBL0 , float* TBL1 , float* pResult )const
//{
////   A = TBL0[0] * x + TBL1[0]
////     + TBL0[1] * y + TBL1[1]
////
////   B = TBL0[2] * x + TBL1[2]
////     + TBL0[3] * y + TBL1[3]
////
////   C = TBL0[4] * x + TBL1[4]
////     + TBL0[5] * y + TBL1[5]
//
//    // Przygotuj wektory X i Y
//    const __m256 PY = _mm256_set1_ps(P.y);
//    const __m256 PX = _mm256_add_ps(
//        _mm256_set1_ps(P.x),
//        _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f)
//    );
//
//    __m256 tb0_low = _mm256_load_ps(TBL0);
//    __m256 tb0_high= _mm256_castpd_ps( _mm256_permute4x64_pd( _mm256_castps_pd( tb0_low ) , 0xBB) );
//           tb0_low = _mm256_castpd_ps( _mm256_permute4x64_pd( _mm256_castps_pd( tb0_low ) , 0x11) );
//
//    __m256 tb1_low = _mm256_load_ps(TBL1);
//    __m256 tb1_high= _mm256_castpd_ps( _mm256_permute4x64_pd( _mm256_castps_pd( tb1_low ) , 0xBB) );
//           tb1_low = _mm256_castpd_ps( _mm256_permute4x64_pd( _mm256_castps_pd( tb1_low ) , 0x11) );
//
//    // Rozg³oœ (broadcast) poszczególne wartoœci do pe³nych rejestrów 256-bitowych
//    // U¿ywamy permutacji, aby efektywnie rozg³osiæ wartoœci bez dodatkowych odczytów z pamiêci
//    const __m256 TBL_0_0 = _mm256_permute_ps(tb0_low , 0xAA); // TBL0[0]
//    const __m256 TBL_0_1 = _mm256_permute_ps(tb0_low , 0xFF); // TBL0[1]
//    const __m256 TBL_0_2 = _mm256_permute_ps(tb0_low , 0x00); // TBL0[2]
//    const __m256 TBL_0_3 = _mm256_permute_ps(tb0_low , 0x55); // TBL0[3]
//    const __m256 TBL_0_4 = _mm256_permute_ps(tb0_high, 0xAA); // TBL0[4]
//    const __m256 TBL_0_5 = _mm256_permute_ps(tb0_high, 0xFF); // TBL0[5]
//
//    const __m256 TBL_1_0 =  _mm256_permute_ps(tb1_low , 0xAA); // TBL1[0]
//    const __m256 TBL_1_1 =  _mm256_permute_ps(tb1_low , 0xFF); // TBL1[1]
//    const __m256 TBL_1_2 =  _mm256_permute_ps(tb1_low , 0x00); // TBL1[2]
//    const __m256 TBL_1_3 =  _mm256_permute_ps(tb1_low , 0x55); // TBL1[3]
//    const __m256 TBL_1_4 =  _mm256_permute_ps(tb1_high, 0xAA); // TBL1[4]
//    const __m256 TBL_1_5 =  _mm256_permute_ps(tb1_high, 0xFF); // TBL1[5]
//
//    // Obliczenia s¹ takie same, ale dane wejœciowe s¹ przygotowane wydajniej
//    __m256 AX = _mm256_fmadd_ps( TBL_0_0 , PX , TBL_1_0 );
//    __m256 AY = _mm256_fmadd_ps( TBL_0_1 , PY , TBL_1_1 );
//    __m256 BX = _mm256_fmadd_ps( TBL_0_2 , PX , TBL_1_2 );
//    __m256 BY = _mm256_fmadd_ps( TBL_0_3 , PY , TBL_1_3 );
//    __m256 CX = _mm256_fmadd_ps( TBL_0_4 , PX , TBL_1_4 );
//    __m256 CY = _mm256_fmadd_ps( TBL_0_5 , PY , TBL_1_5 );
//
//    // Sumowanie i zapisywanie wyników
//    AX = _mm256_add_ps( AX , AY );
//    _mm256_store_ps( pResult + 8  , AX  );
//    BX = _mm256_add_ps( BX , BY );
//    _mm256_store_ps( pResult + 16 , BX  );
//    CX = _mm256_add_ps( CX , CY );
//    _mm256_store_ps( pResult + 24 , CX  );
//
//    // Porównanie i wygenerowanie ostatecznej maski
//    const __m256 ZERO  = _mm256_setzero_ps();
//    AX = _mm256_cmp_ps( AX , ZERO , _CMP_LT_OQ );
//    BX = _mm256_cmp_ps( BX , ZERO , _CMP_LT_OQ );
//    CX = _mm256_cmp_ps( CX , ZERO , _CMP_LT_OQ );
//
//    AX = _mm256_or_ps( AX , BX );
//    AX = _mm256_or_ps( AX , CX );
//
//    _mm256_store_ps( pResult + 0  , AX  );
//}

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