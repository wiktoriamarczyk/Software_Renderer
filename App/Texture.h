/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "IRenderer.h"
#include "Vector3f.h"

class Texture : public ITexture
{
public:
    Texture() = default;
    bool CreateWhite4x4Tex();
    bool Load(const char* fileName);
    bool IsValid()const override;
    Vector4f Sample(Vector2f uv)const;

    template< eSimdType Type >
    Vector4f256<Type> Sample(const Vector2f256<Type>& uv)const;
private:
    vector<uint32_t> m_Data;
    vector<Vector4f> m_fData;
    int              m_Width = 0;
    int              m_Height = 0;
    i256A            m_WidthSSE = 0;
    i256A            m_HeightSSE = 0;

    float            m_fWidth = 0;
    float            m_fHeight = 0;
    f256A            m_fWidthSSE = 0;
    f256A            m_fHeightSSE = 0;

    int              m_MaxWidth = 0;
    int              m_MaxHeight = 0;
    f256A            m_fMaxWidthSSE = 0;
    f256A            m_fMaxHeightSSE = 0;

    int              m_2Width = 0;
    int              m_2Height = 0;
    int              m_ShiftedWidth = 0;
    int              m_ShiftedHeight = 0;

    float            m_WidthBias = 0;
    float            m_HeightBias = 0;

    bool             m_Pow2   = false;
    bool             m_Clamp  = false;
    int              m_SizeMaskX = 0;
    int              m_SizeMaskY = 0;
    i256A            m_SizeMaskXSSE;
    i256A            m_SizeMaskYSSE;;
    f256A            m_10_SSE{ 10 };
    int              m_SizeShiftX = 0;
    int              m_SizeShiftY = 0;
};

inline Vector4f Texture::Sample(Vector2f uv) const
{
    int x;
    int y;
    int pixelIndex;
    if( !m_Pow2 )
    {
        if( m_Clamp )
        {
            x = std::clamp<int>( int32_t( uv.x * m_fWidth ) , 0 , m_MaxWidth  );
            y = std::clamp<int>( int32_t( uv.y * m_fHeight) , 0 , m_MaxHeight );
        }
        else
        {
            x = ( int32_t( ( 10 + uv.x ) * m_fWidth  ) ) % m_Width ;
            y = ( int32_t( ( 10 + uv.y ) * m_fHeight ) ) % m_Height;
        }
        pixelIndex = y * m_Width + x;
    }
    else
    {
        if( m_Clamp )
        {
            x = std::clamp<int>( int32_t( uv.x * m_fWidth ) , 0 , m_MaxWidth  );
            y = std::clamp<int>( int32_t( uv.y * m_fHeight) , 0 , m_MaxHeight );
        }
        else
        {
            x = ( int32_t( ( 10 + uv.x ) * m_fWidth  ) ) & m_SizeMaskX;
            y = ( int32_t( ( 10 + uv.y ) * m_fHeight ) ) & m_SizeMaskY;
        }
        pixelIndex = (y << m_SizeShiftX) + x;
    }

    return m_fData[pixelIndex];
}

FORCE_INLINE void Fill_AVX_Color( float* pOut , const Vector4f* pIn , const int* Indices , int Index )
{
    auto& Pixel = pIn[ Indices[Index] ];
    pOut[ 32*(Index/8)+(Index%8)+0  ] = Pixel.x;
    pOut[ 32*(Index/8)+(Index%8)+8  ] = Pixel.y;
    pOut[ 32*(Index/8)+(Index%8)+16 ] = Pixel.z;
    pOut[ 32*(Index/8)+(Index%8)+24 ] = Pixel.w;
}

FORCE_INLINE void Transpose8Vec4f_to_Vec4f256(const Vector4f* in, Vector4f256A& out)
{
    __m256 tmp;

    __m256 row0 = _mm256_loadu_ps( in[0].Data() ); // A B C D  E F G H
    __m256 row1 = _mm256_loadu_ps( in[2].Data() ); // I J K L  M N O P
    __m256 row2 = _mm256_loadu_ps( in[4].Data() ); // Q R S T  U V W X
    __m256 row3 = _mm256_loadu_ps( in[6].Data() ); // Y Z 1 2  3 4 5 6

           tmp  = _mm256_permute2f128_ps( row0 ,row2 , 0x31 );
           row0 = _mm256_permute2f128_ps( row0 ,row2 , 0x20 ); // A B C D  Q R S T
           row2 = tmp;                                         // E F G H  U V W X
           tmp  = _mm256_permute2f128_ps( row1 ,row3 , 0x31 );
           row1 = _mm256_permute2f128_ps( row1 ,row3 , 0x20 ); // I J K L  Y Z 1 2
           row3 = tmp;                                         // M N O P  3 4 5 6

    // Unpack low/high
    __m256 t0 = _mm256_unpacklo_ps(row0, row2); // A E B F  Q U R V
    __m256 t1 = _mm256_unpackhi_ps(row0, row2); // C G D H  S W T X
    __m256 t2 = _mm256_unpacklo_ps(row1, row3); // I M J N  Y 3 Z 4
    __m256 t3 = _mm256_unpackhi_ps(row1, row3); // K O L P  1 5 2 6

    // Final permutes
    out.x.v =_mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0)); // A E I M Q U Y 3
    out.y.v =_mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2)); // B F J N R V Z 4
    out.z.v =_mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0)); // C G K O S W 1 5
    out.w.v =_mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2)); // D H L P T X 2 6
}

template< eSimdType Type >
inline Vector4f256<Type> Texture::Sample(const Vector2f256<Type>& uv) const
{
    i256<Type> x;
    i256<Type> y;

    ALIGN_FOR_AVX int pixelIndex[8];

    if( !m_Pow2 )
    {
        if( m_Clamp )
        {
            x = i256A( (uv.x * m_fWidthSSE ).clamp( f256<Type>::One , m_fMaxWidthSSE  ) );
            y = i256A( (uv.y * m_fHeightSSE).clamp( f256<Type>::One , m_fMaxHeightSSE ) );
        }
        else
        {
            x = ( i256A( ( m_10_SSE + uv.x ) * m_fWidthSSE ) ) % m_WidthSSE ;
            y = ( i256A( ( m_10_SSE + uv.y ) * m_fHeightSSE) ) % m_HeightSSE;
        }
        (y * m_Width + x).store( pixelIndex , simd_alignment::AVX );
    }
    else
    {
        if( m_Clamp )
        {
            x = i256A( (uv.x * m_fWidthSSE ).clamp( f256<Type>::One , m_fMaxWidthSSE  ) );
            y = i256A( (uv.y * m_fHeightSSE).clamp( f256<Type>::One , m_fMaxHeightSSE ) );
        }
        else
        {
            x = i256A( ( m_10_SSE + uv.x ) * m_fWidthSSE ) & m_SizeMaskXSSE;
            y = i256A( ( m_10_SSE + uv.y ) * m_fHeightSSE) & m_SizeMaskYSSE;
        }
        ((y << m_SizeShiftX) + x).store( pixelIndex , simd_alignment::AVX );
    }

    ALIGN_FOR_AVX Vector4f samples[8];
    samples[0] = m_fData[pixelIndex[0]];
    samples[1] = m_fData[pixelIndex[1]];
    samples[2] = m_fData[pixelIndex[2]];
    samples[3] = m_fData[pixelIndex[3]];
    samples[4] = m_fData[pixelIndex[4]];
    samples[5] = m_fData[pixelIndex[5]];
    samples[6] = m_fData[pixelIndex[6]];
    samples[7] = m_fData[pixelIndex[7]];

    Vector4f256A Result;
    Transpose8Vec4f_to_Vec4f256( samples, Result );
    return Result;
}