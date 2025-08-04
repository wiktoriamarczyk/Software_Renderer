/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Common.h"
#include "TransformedVertex.h"

class VertexInterpolator
{
public:
    VertexInterpolator( std::nullptr_t ){};
    VertexInterpolator(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C);
    VertexInterpolator(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& Color);
    void InterpolateZ(const Vector3f& baricentric, TransformedVertex& out);
    void InterpolateAllButZ(const Vector3f& baricentric, TransformedVertex& out);
    void Interpolate(const IMath& math, const Vector3f& baricentric, TransformedVertex& out);

    template< typename MathT >
    void InterpolateT(Vector3f baricentric, TransformedVertex& out)const;

    template< int Elements , eSimdType Type >
    void InterpolateZ(const Vector3<fsimd<Elements,Type>>& baricentric, SimdTransformedVertex<Elements,Type>& out)const;
    template< int Elements , eSimdType Type >
    void InterpolateAllButZ(const Vector3<fsimd<Elements,Type>>& baricentric, SimdTransformedVertex<Elements,Type>& out)const;
//private:
    struct ALIGN_FOR_AVX InterpolatedSource
    {
        Vector3f    m_NormalOverW;
        Vector4f    m_ColorOverW;
        Vector2f    m_UVOverW;
        Vector3f    m_WorldPositionOverW;
        float       m_OneOverW;
        float       _unused[2]; // Padding to ensure 16-byte alignment for SIMD operations
        float       m_ScreenPositionZ;

        float*       Data()      { return m_NormalOverW.data(); }
        const float* Data()const { return m_NormalOverW.data(); }
    };

    InterpolatedSource m_A;
    InterpolatedSource m_B;
    InterpolatedSource m_C;

};

inline void VertexInterpolator::InterpolateZ(const Vector3f& baricentric, TransformedVertex& out)
{
    out.m_ScreenPosition.z = baricentric.x * m_A.m_ScreenPositionZ + baricentric.y * m_B.m_ScreenPositionZ + baricentric.z * m_C.m_ScreenPositionZ;
}

inline void VertexInterpolator::InterpolateAllButZ(const Vector3f& baricentric, TransformedVertex& out)
{
    // 3 multiplications and 2 additions + ....
    float oneOverW = baricentric.x * m_A.m_OneOverW + baricentric.y * m_B.m_OneOverW + baricentric.z * m_C.m_OneOverW;

    // 1 division
    float w = 1.0f / oneOverW;

    //  4*( 3 + 2 ) = 20 multiplications + 6 additions
    //  3*( 3 + 2 ) = 15 multiplications + 6 additions
    //  2*( 3 + 2 ) = 10 multiplications + 6 additions
    //  3*( 3 + 2 ) = 15 multiplications + 6 additions
    out.m_Normal          = (baricentric.x * m_A.m_NormalOverW          + baricentric.y * m_B.m_NormalOverW         + baricentric.z * m_C.m_NormalOverW) * w;
    out.m_Color           = (baricentric.x * m_A.m_ColorOverW           + baricentric.y * m_B.m_ColorOverW          + baricentric.z * m_C.m_ColorOverW) * w;
    out.m_UV              = (baricentric.x * m_A.m_UVOverW              + baricentric.y * m_B.m_UVOverW             + baricentric.z * m_C.m_UVOverW) * w;
    out.m_WorldPosition   = (baricentric.x * m_A.m_WorldPositionOverW   + baricentric.y * m_B.m_WorldPositionOverW  + baricentric.z * m_C.m_WorldPositionOverW) * w;

    // all sums up to = 63 multiplications + 26 additions and one division
}

inline void VertexInterpolator::Interpolate(const IMath& math, const Vector3f& baricentric, TransformedVertex& out)
{
    float* AData = reinterpret_cast<float*>(&m_A);
    float* BData = reinterpret_cast<float*>(&m_B);
    float* CData = reinterpret_cast<float*>(&m_C);
    float* result = reinterpret_cast<float*>(&out);

    InterpolatedSource tmpA, tmpB, tmpC;

    //alignas(16) float TmpAResult[8];
    //alignas(16) float TmpBResult[8];
    //alignas(16) float TmpCResult[8];

    float* TmpAResult = reinterpret_cast<float*>(&tmpA);
    //float* TmpBResult = reinterpret_cast<float*>(&tmpB);
    //float* TmpCResult = reinterpret_cast<float*>(&tmpC);

    //math.MultiplyVec8ByScalar(AData, baricentric.x, TmpAResult);
    //math.MultiplyVec8ByScalar(BData, baricentric.y, TmpBResult);
    //math.MultiplyVec8ByScalar(CData, baricentric.z, TmpCResult);

    //math.AddVec8(TmpAResult, TmpBResult, TmpAResult);
    //math.AddVec8(TmpAResult, TmpCResult, TmpAResult);

    //math.MultiplyVec8ByScalar(AData + 8, baricentric.x, TmpAResult + 8);
    //math.MultiplyVec8ByScalar(BData + 8, baricentric.y, TmpBResult + 8);
    //math.MultiplyVec8ByScalar(CData + 8, baricentric.z, TmpCResult + 8);

    //math.AddVec8(TmpAResult + 8 , TmpBResult + 8 , TmpAResult + 8 );
    //math.AddVec8(TmpAResult + 8 , TmpCResult + 8 , TmpAResult + 8 );




    math.MultiplyVec8ByScalar(AData, baricentric.x, TmpAResult);
    math.MulVec8ByScalarAdd  (BData, baricentric.y, TmpAResult , TmpAResult );
    math.MulVec8ByScalarAdd  (CData, baricentric.z, TmpAResult , TmpAResult );


    math.MultiplyVec8ByScalar(AData + 8, baricentric.x, TmpAResult + 8 );
    math.MulVec8ByScalarAdd  (BData + 8, baricentric.y, TmpAResult + 8 , TmpAResult + 8 );
    math.MulVec8ByScalarAdd  (CData + 8, baricentric.z, TmpAResult + 8 , TmpAResult + 8 );


    ////// First part of data (8 floats)
    ////out.m_Color.x = (baricentric.x * m_A.m_ColorOverW.x + baricentric.y * m_B.m_ColorOverW.x + baricentric.z * m_C.m_ColorOverW.x);
    ////out.m_Color.y = (baricentric.x * m_A.m_ColorOverW.y + baricentric.y * m_B.m_ColorOverW.y + baricentric.z * m_C.m_ColorOverW.y);
    ////out.m_Color.z = (baricentric.x * m_A.m_ColorOverW.z + baricentric.y * m_B.m_ColorOverW.z + baricentric.z * m_C.m_ColorOverW.z);
    ////out.m_Color.w = (baricentric.x * m_A.m_ColorOverW.w + baricentric.y * m_B.m_ColorOverW.w + baricentric.z * m_C.m_ColorOverW.w);

    ////out.m_Normal.x = (baricentric.x * m_A.m_NormalOverW.x + baricentric.y * m_B.m_NormalOverW.x + baricentric.z * m_C.m_NormalOverW.x);
    ////out.m_Normal.y = (baricentric.x * m_A.m_NormalOverW.y + baricentric.y * m_B.m_NormalOverW.y + baricentric.z * m_C.m_NormalOverW.y);
    ////out.m_Normal.z = (baricentric.x * m_A.m_NormalOverW.z + baricentric.y * m_B.m_NormalOverW.z + baricentric.z * m_C.m_NormalOverW.z);

    ////out.m_UV.x = (baricentric.x * m_A.m_UVOverW.x + baricentric.y * m_B.m_UVOverW.x + baricentric.z * m_C.m_UVOverW.x);
    ////// Second part of data (5 floats)
    ////out.m_UV.y = (baricentric.x * m_A.m_UVOverW.y + baricentric.y * m_B.m_UVOverW.y + baricentric.z * m_C.m_UVOverW.y);

    ////out.m_WorldPosition.x = (baricentric.x * m_A.m_WorldPositionOverW.x + baricentric.y * m_B.m_WorldPositionOverW.x + baricentric.z * m_C.m_WorldPositionOverW.x);
    ////out.m_WorldPosition.y = (baricentric.x * m_A.m_WorldPositionOverW.y + baricentric.y * m_B.m_WorldPositionOverW.y + baricentric.z * m_C.m_WorldPositionOverW.y);
    ////out.m_WorldPosition.z = (baricentric.x * m_A.m_WorldPositionOverW.z + baricentric.y * m_B.m_WorldPositionOverW.z + baricentric.z * m_C.m_WorldPositionOverW.z);

    ////out.m_ScreenPosition.z = baricentric.x * m_A.m_ScreenPositionZ + baricentric.y * m_B.m_ScreenPositionZ + baricentric.z * m_C.m_ScreenPositionZ;

    //float oneOverW = baricentric.x * m_A.m_OneOverW + baricentric.y * m_B.m_OneOverW + baricentric.z * m_C.m_OneOverW;
    float w = 1.0f / tmpA.m_OneOverW;

    math.MultiplyVec8ByScalar(TmpAResult    , w, result    );
    math.MultiplyVec4ByScalar(TmpAResult + 8, w, result + 8);

    out.m_ScreenPosition.z = tmpA.m_ScreenPositionZ;

    //// First part of data (8 floats)
    //out.m_Color.x = out.m_Color.x * w;
    //out.m_Color.y = out.m_Color.y * w;
    //out.m_Color.z = out.m_Color.z * w;
    //out.m_Color.w = out.m_Color.w * w;

    //out.m_Normal.x = out.m_Normal.x * w;
    //out.m_Normal.y = out.m_Normal.y * w;
    //out.m_Normal.z = out.m_Normal.z * w;

    //out.m_UV.x = out.m_UV.x * w;
    //// Second part of data (6 floats)
    //out.m_UV.y = out.m_UV.y * w;

    //out.m_WorldPosition.x = out.m_WorldPosition.x * w;
    //out.m_WorldPosition.y = out.m_WorldPosition.y * w;
    //out.m_WorldPosition.z = out.m_WorldPosition.z * w;
}


template< typename MathT >
__forceinline void VertexInterpolator::InterpolateT(Vector3f baricentricCoordinates, TransformedVertex& out)const
{
    MathT math;
    InterpolatedSource tmp;

    const float* AData  = m_A.Data();
    const float* BData  = m_B.Data();
    const float* CData  = m_C.Data();
          float* pTmp   = tmp.Data();

    math.MultiplyVec8ByScalar(AData    , baricentricCoordinates.x, pTmp     );
    math.MulVec8ByScalarAdd  (BData    , baricentricCoordinates.y, pTmp     , pTmp );
    math.MulVec8ByScalarAdd  (CData    , baricentricCoordinates.z, pTmp     , pTmp );

    math.MultiplyVec8ByScalar(AData + 8, baricentricCoordinates.x, pTmp + 8 );
    math.MulVec8ByScalarAdd  (BData + 8, baricentricCoordinates.y, pTmp + 8 , pTmp + 8 );
    math.MulVec8ByScalarAdd  (CData + 8, baricentricCoordinates.z, pTmp + 8 , pTmp + 8 );

    float w = 1.0f / tmp.m_OneOverW;

    math.MultiplyVec8ByScalar(pTmp    , w, out.Data()    );
    math.MultiplyVec4ByScalar(pTmp + 8, w, out.Data() + 8);

    out.m_ScreenPosition.z = tmp.m_ScreenPositionZ;
}

template< int Elements , eSimdType Type >
inline void VertexInterpolator::InterpolateZ(const Vector3<fsimd<Elements,Type>>& baricentricCoordinates, SimdTransformedVertex<Elements,Type>& out)const
{
    //__m128 bx = _mm_load_ps(baricentricCoordinates.x);
    //__m128 by = _mm_load_ps(baricentricCoordinates.y);
    //__m128 bz = _mm_load_ps(baricentricCoordinates.z);

    //__m128 tmpA = _mm_set1_ps(m_A.m_ScreenPositionZ);
    //__m128 tmpB = _mm_set1_ps(m_B.m_ScreenPositionZ);
    //__m128 tmpC = _mm_set1_ps(m_C.m_ScreenPositionZ);

    //__m128 tmpA = _mm_mul_ps(tmpA, bx);
    //__m128 tmpB = _mm_mul_ps(tmpB, by);
    //__m128 tmpC = _mm_mul_ps(tmpC, bz);

    //__m128 z = _mm_add_ps(tmpA,tmpB);
    //       z = _mm_add_ps(z, tmpC);

    //_mm_store_ps(out.m_ScreenPosition.z, z);

    Vector3<fsimd<Elements,Type>> tmp(m_A.m_ScreenPositionZ,m_B.m_ScreenPositionZ,m_C.m_ScreenPositionZ);

    tmp *= baricentricCoordinates;

    simd z  = tmp.x + tmp.y;
         z += tmp.z;

    out.m_ScreenPosition.z = z;
}


template< int Elements , eSimdType Type >
inline void VertexInterpolator::InterpolateAllButZ(const Vector3<fsimd<Elements,Type>>& baricentricCoordinates, SimdTransformedVertex<Elements,Type>& out)const
{
    //const __m128 bx = _mm_load_ps(baricentricCoordinates.x);
    //const __m128 by = _mm_load_ps(baricentricCoordinates.y);
    //const __m128 bz = _mm_load_ps(baricentricCoordinates.z);

    //__m128 tmpA = _mm_set1_ps(m_A.m_OneOverW);
    //__m128 tmpB = _mm_set1_ps(m_B.m_OneOverW);
    //__m128 tmpC = _mm_set1_ps(m_C.m_OneOverW);

    //tmpA = _mm_mul_ps(tmpA, bx);
    //tmpB = _mm_mul_ps(tmpB, by);
    //tmpC = _mm_mul_ps(tmpC, bz);

    //__m128 w = _mm_add_ps(tmpA,tmpB);
    //       w = _mm_add_ps(w, tmpC);

    //tmpA = _mm_set1_ps(1.0f);

    //w = _mm_div_ps(tmpA, w);

    //auto interpolate = [&]( float BA , float BB , float BC , float* out ) [[msvc::forceinline]]
    //{
    //    tmpA = _mm_set1_ps(BA);
    //    tmpA = _mm_set1_ps(BB);
    //    tmpA = _mm_set1_ps(BC);

    //    tmpA = _mm_mul_ps(tmpA, bx);
    //    tmpB = _mm_mul_ps(tmpB, by);
    //    tmpC = _mm_mul_ps(tmpC, bz);

    //    __m128 r = _mm_add_ps(tmpA, tmpB);
    //           r = _mm_add_ps(   r, tmpC);
    //           r = _mm_mul_ps(   r, w   );

    //           _mm_store_ps(out, r);
    //};
    //interpolate( m_A.m_NormalOverW.x , m_B.m_NormalOverW.x , m_C.m_NormalOverW.x , out.m_Normal.x );
    //interpolate( m_A.m_NormalOverW.y , m_B.m_NormalOverW.y , m_C.m_NormalOverW.y , out.m_Normal.y );
    //interpolate( m_A.m_NormalOverW.z , m_B.m_NormalOverW.z , m_C.m_NormalOverW.z , out.m_Normal.z );

    //interpolate( m_A.m_ColorOverW.x , m_B.m_ColorOverW.x , m_C.m_ColorOverW.x , out.m_Color.x );
    //interpolate( m_A.m_ColorOverW.y , m_B.m_ColorOverW.y , m_C.m_ColorOverW.y , out.m_Color.y );
    //interpolate( m_A.m_ColorOverW.z , m_B.m_ColorOverW.z , m_C.m_ColorOverW.z , out.m_Color.z );
    //interpolate( m_A.m_ColorOverW.w , m_B.m_ColorOverW.w , m_C.m_ColorOverW.w , out.m_Color.w );

    //interpolate( m_A.m_UVOverW.x    , m_B.m_UVOverW.x    , m_C.m_UVOverW.x    , out.m_UV.x    );
    //interpolate( m_A.m_UVOverW.y    , m_B.m_UVOverW.y    , m_C.m_UVOverW.y    , out.m_UV.y    );

    //interpolate( m_A.m_WorldPositionOverW.x , m_B.m_WorldPositionOverW.x , m_C.m_WorldPositionOverW.x , out.m_WorldPosition.x );
    //interpolate( m_A.m_WorldPositionOverW.y , m_B.m_WorldPositionOverW.y , m_C.m_WorldPositionOverW.y , out.m_WorldPosition.y );
    //interpolate( m_A.m_WorldPositionOverW.z , m_B.m_WorldPositionOverW.z , m_C.m_WorldPositionOverW.z , out.m_WorldPosition.z );


    Vector3<fsimd<Elements,Type>> tmp(m_A.m_OneOverW, m_B.m_OneOverW, m_C.m_OneOverW);

    tmp *= baricentricCoordinates;

    simd w = tmp.x + tmp.y;
         w+= tmp.z;

    w = fsimd<Elements,Type>(1) / w;

    auto interpolate = [&]( float BA , float BB , float BC , fsimd<Elements,Type>& out ) [[msvc::forceinline]]
    {
        Vector3<fsimd<Elements,Type>> tmp(BA,BB,BC);

        tmp *= baricentricCoordinates;

        simd r = tmp.x + tmp.y;
             r+= tmp.z;
             r*= w;

        out = r;
    };

    interpolate( m_A.m_NormalOverW.x , m_B.m_NormalOverW.x , m_C.m_NormalOverW.x , out.m_Normal.x );
    interpolate( m_A.m_NormalOverW.y , m_B.m_NormalOverW.y , m_C.m_NormalOverW.y , out.m_Normal.y );
    interpolate( m_A.m_NormalOverW.z , m_B.m_NormalOverW.z , m_C.m_NormalOverW.z , out.m_Normal.z );

    interpolate( m_A.m_ColorOverW.x , m_B.m_ColorOverW.x , m_C.m_ColorOverW.x , out.m_Color.x );
    interpolate( m_A.m_ColorOverW.y , m_B.m_ColorOverW.y , m_C.m_ColorOverW.y , out.m_Color.y );
    interpolate( m_A.m_ColorOverW.z , m_B.m_ColorOverW.z , m_C.m_ColorOverW.z , out.m_Color.z );
    interpolate( m_A.m_ColorOverW.w , m_B.m_ColorOverW.w , m_C.m_ColorOverW.w , out.m_Color.w );

    interpolate( m_A.m_UVOverW.x    , m_B.m_UVOverW.x    , m_C.m_UVOverW.x    , out.m_UV.x    );
    interpolate( m_A.m_UVOverW.y    , m_B.m_UVOverW.y    , m_C.m_UVOverW.y    , out.m_UV.y    );

    interpolate( m_A.m_WorldPositionOverW.x , m_B.m_WorldPositionOverW.x , m_C.m_WorldPositionOverW.x , out.m_WorldPosition.x );
    interpolate( m_A.m_WorldPositionOverW.y , m_B.m_WorldPositionOverW.y , m_C.m_WorldPositionOverW.y , out.m_WorldPosition.y );
    interpolate( m_A.m_WorldPositionOverW.z , m_B.m_WorldPositionOverW.z , m_C.m_WorldPositionOverW.z , out.m_WorldPosition.z );
}