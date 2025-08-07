/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#define STB_IMAGE_IMPLEMENTATION
#include "Texture.h"
#include "../stb/stb_image.h"

#include <bit>
#include <cmath>

bool Texture::CreateWhite4x4Tex()
{
    m_Data.resize(4 * 4);
    for (int i = 0; i < 4 * 4; ++i)
        m_Data[i] = 0xFFFFFFFF;
    m_Width = 4;
    m_Height = 4;

    m_Pow2 = true;

    m_fData.resize(m_Data.size());
    for(size_t i = 0; i < m_Data.size(); ++i)
    {
        m_fData[i] = Vector4f::FromARGB( m_Data[i] );
    }

    m_SizeMaskX = m_Width  - 1;
    m_SizeMaskY = m_Height - 1;
    m_SizeShiftX = std::countr_zero( uint32_t(m_Width) );
    m_SizeShiftY = std::countr_zero( uint32_t(m_Height) );

    m_ShiftedWidth   = m_Width  << 16;
    m_ShiftedHeight  = m_Height << 16;

    m_fWidth  = float(m_Width);
    m_fHeight = float(m_Height);

    m_fWidthSSE = f256A{ m_fWidth };
    m_fHeightSSE= f256A{ m_fHeight };

    m_SizeMaskXSSE = i256A{ m_SizeMaskX };
    m_SizeMaskYSSE = i256A{ m_SizeMaskY };

    m_MaxWidth  = m_Width  - 1;
    m_MaxHeight = m_Height - 1;

    m_fMaxWidthSSE = f256A{ m_fWidth-1 };
    m_fMaxHeightSSE= f256A{ m_fHeight-1 };

    m_WidthBias  = 0.001f + m_Width *100;
    m_HeightBias = 0.001f + m_Height*100;

    return true;
}

bool Texture::Load(const char* fileName)
{
    int width, height, channels;
    //STB::stbi_set_flip_vertically_on_load(true);

    auto result = STB::stbi_info(fileName, &width, &height, &channels);
    if (!result)
        return false;

    if (width > MAX_TEXTURE_SIZE || height > MAX_TEXTURE_SIZE)
    {
        printf("Texture size is too big: %d x %d, max is %d x %d\n", width, height, MAX_TEXTURE_SIZE , MAX_TEXTURE_SIZE);
        return false;
    }

    STB::stbi_uc* data = STB::stbi_load(fileName, &width, &height, &channels, 4);
    if (!data)
        return false;

    m_Data.resize(width * height);
    memcpy(m_Data.data(), data, width * height * 4);
    m_Width = width;
    m_Height = height;
    STB::stbi_image_free(data);

    m_fData.resize(m_Data.size());
    for(size_t i = 0; i < m_Data.size(); ++i)
    {
        m_fData[i] = Vector4f::FromARGB( m_Data[i] );
    }

    if( ( m_Pow2 = (IsPowerOfTwo( m_Width ) && IsPowerOfTwo( m_Height )) ) )
    {
        m_SizeMaskX = m_Width  - 1;
        m_SizeMaskY = m_Height - 1;
        m_SizeShiftX = std::countr_zero( uint32_t(m_Width) );
        m_SizeShiftY = std::countr_zero( uint32_t(m_Height) );
    }
    else
    {
        m_SizeMaskX = 0;
        m_SizeMaskY = 0;
    }

    m_ShiftedWidth   = m_Width  << 16;
    m_ShiftedHeight  = m_Height << 16;

    m_fWidth  = float(m_Width);
    m_fHeight = float(m_Height);

    m_fWidthSSE = f256A{ m_fWidth };
    m_fHeightSSE= f256A{ m_fHeight };

    m_SizeMaskXSSE = i256A{ m_SizeMaskX };
    m_SizeMaskYSSE = i256A{ m_SizeMaskY };

    m_MaxWidth  = m_Width  - 1;
    m_MaxHeight = m_Height - 1;

    m_fMaxWidthSSE = f256A{ m_fWidth-1 };
    m_fMaxHeightSSE= f256A{ m_fHeight-1 };

    m_WidthBias  = 0.001f + m_Width *100;
    m_HeightBias = 0.001f + m_Height*100;

    return true;
}

bool Texture::IsValid()const
{
    return m_Data.size() > 0;
}