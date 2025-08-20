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

    template< eSimdType Type , int Elements >
    Vector4<fsimd<Elements,Type>> Sample(const Vector2<fsimd<Elements,Type>>& uv)const;
private:
    template< int Elements , eSimdType Type = eSimdType::None >
    struct SampleHelper
    {
        wide_arithmetic<int,Elements,Type>      m_Width        = {};
        wide_arithmetic<int,Elements,Type>      m_Height       = {};

        wide_arithmetic<int,Elements,Type>      m_MaxWidth     = {};
        wide_arithmetic<int,Elements,Type>      m_MaxHeight    = {};

        wide_arithmetic<float,Elements,Type>    m_fWidth       = {};
        wide_arithmetic<float,Elements,Type>    m_fHeight      = {};

        wide_arithmetic<float,Elements,Type>    m_fMaxWidth    = {};
        wide_arithmetic<float,Elements,Type>    m_fMaxHeight   = {};

        wide_arithmetic<int,Elements,Type>      m_SizeShiftX   = {};
        wide_arithmetic<int,Elements,Type>      m_SizeShiftY   = {};
        wide_arithmetic<int,Elements,Type>      m_SizeMaskX    = {};
        wide_arithmetic<int,Elements,Type>      m_SizeMaskY    = {};
    };

    void InitSimdHelpers();

    template< int Elements , eSimdType Type = eSimdType::None >
    auto&            GetHelper()const
    {
        if constexpr( Elements == 1 )
            return m_HelperStandard;
        else if constexpr( Elements == 8 && Type == eSimdType::None )
            return m_HelperCPU;
        else if constexpr( Elements == 4 && Type == eSimdType::SSE )
            return m_HelperSSE;
        else if constexpr( Elements == 8 && Type == eSimdType::SSE )
            return m_HelperSSE8;
        else if constexpr( Elements == 8 && Type == eSimdType::AVX )
            return m_HelperAVX;
    }
    vector<uint32_t> m_Data;
    vector<Vector4f> m_fData;

    SampleHelper<1>                  m_HelperStandard;
    SampleHelper<8,eSimdType::None>  m_HelperCPU;
    SampleHelper<8,eSimdType::AVX>   m_HelperAVX;
    SampleHelper<8,eSimdType::SSE>   m_HelperSSE8;
    SampleHelper<4,eSimdType::SSE>   m_HelperSSE;

    float            m_WidthBias = 0;
    float            m_HeightBias = 0;

    bool             m_Pow2   = false;
    bool             m_Clamp  = false;
};

inline Vector4f Texture::Sample(Vector2f uv) const
{
    auto& helper = GetHelper<1>();
    int x;
    int y;
    int pixelIndex;
    if( !m_Pow2 )
    {
        if( m_Clamp )
        {
            x = std::clamp<int>( int32_t( uv.x * helper.m_fWidth ) , 0 , helper.m_MaxWidth  );
            y = std::clamp<int>( int32_t( uv.y * helper.m_fHeight) , 0 , helper.m_MaxHeight );
        }
        else
        {
            x = ( int32_t( ( 10 + uv.x ) * helper.m_fWidth  ) ) % helper.m_Width ;
            y = ( int32_t( ( 10 + uv.y ) * helper.m_fHeight ) ) % helper.m_Height;
        }
        pixelIndex = y * helper.m_Width + x;
    }
    else
    {
        if( m_Clamp )
        {
            x = std::clamp<int>( int32_t( uv.x * helper.m_fWidth ) , 0 , helper.m_MaxWidth  );
            y = std::clamp<int>( int32_t( uv.y * helper.m_fHeight) , 0 , helper.m_MaxHeight );
        }
        else
        {
            x = ( int32_t( ( 10 + uv.x ) * helper.m_fWidth  ) ) & helper.m_SizeMaskX;
            y = ( int32_t( ( 10 + uv.y ) * helper.m_fHeight ) ) & helper.m_SizeMaskY;
        }
        pixelIndex = (y << helper.m_SizeShiftX) + x;
    }

    return m_fData[pixelIndex];
}

template< eSimdType Type , int Elements >
Vector4<fsimd<Elements,Type>> Texture::Sample(const Vector2<fsimd<Elements,Type>>& uv)const
{
    using simd_int  = isimd<Elements,Type>;
    using simd_float= fsimd<Elements, Type>;
    using simd_vec4 = Vector4<simd_float>;
    simd_int x;
    simd_int y;

    auto& helper = GetHelper<Elements,Type>();

    ALIGN_FOR_AVX int pixelIndex[Elements];

    if( !m_Pow2 )
    {
        if( m_Clamp )
        {
            x = simd_int( (uv.x * helper.m_fWidth ).clamp( simd_float::One , helper.m_fMaxWidth  ) );
            y = simd_int( (uv.y * helper.m_fHeight).clamp( simd_float::One , helper.m_fMaxHeight ) );
        }
        else
        {
            x = ( simd_int( ( simd_float::Ten + uv.x ) * helper.m_fWidth ) ) % helper.m_Width ;
            y = ( simd_int( ( simd_float::Ten + uv.y ) * helper.m_fHeight) ) % helper.m_Height;
        }
        (y * helper.m_Width + x).store( pixelIndex , simd_alignment::AVX );
    }
    else
    {
        if( m_Clamp )
        {
            x = simd_int( (uv.x * helper.m_fWidth ).clamp( simd_float::One , helper.m_fMaxWidth  ) );
            y = simd_int( (uv.y * helper.m_fHeight).clamp( simd_float::One , helper.m_fMaxHeight ) );
        }
        else
        {
            x = simd_int( ( simd_float::Ten + uv.x ) * helper.m_fWidth ) & helper.m_SizeMaskX;
            y = simd_int( ( simd_float::Ten + uv.y ) * helper.m_fHeight) & helper.m_SizeMaskY;
        }
        ((y << helper.m_SizeShiftX) + x).store( pixelIndex , simd_alignment::AVX );
    }

    auto fData = m_fData.data();

    ALIGN_FOR_AVX Vector4f samples[Elements];
    samples[0] = fData[pixelIndex[0]];
    samples[1] = fData[pixelIndex[1]];
    samples[2] = fData[pixelIndex[2]];
    samples[3] = fData[pixelIndex[3]];
    if constexpr( Elements >= 8 )
    {
    samples[4] = fData[pixelIndex[4]];
    samples[5] = fData[pixelIndex[5]];
    samples[6] = fData[pixelIndex[6]];
    samples[7] = fData[pixelIndex[7]];
    }

    simd_vec4 simd_samples;
    data_array<float,eDataAlignment::AVX,Elements,data_array<float,eDataAlignment::AVX>> Data{ samples->data() };
    simd_samples.load( Data );
    simd_float::transpose_ARGBx_to_AxRxGxBx( simd_samples.x , simd_samples.y , simd_samples.z , simd_samples.w );
    return simd_samples;
}