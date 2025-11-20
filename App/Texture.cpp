/*
* Master’s thesis - Analysis of selected optimization techniques for a 3D software renderer
* Author: Wiktoria Marczyk
* Year: 2025
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

    m_HelperStandard.m_Width = 4;
    m_HelperStandard.m_Height = 4;

    InitSimdHelpers();

    return true;
}

void Texture::InitSimdHelpers()
{
    m_HelperStandard.m_fWidth = float(m_HelperStandard.m_Width);
    m_HelperStandard.m_fHeight = float(m_HelperStandard.m_Height);
    m_HelperStandard.m_MaxWidth = m_HelperStandard.m_Width - 1;
    m_HelperStandard.m_MaxHeight = m_HelperStandard.m_Height - 1;
    if ((m_Pow2 = (IsPowerOfTwo(m_HelperStandard.m_Width) && IsPowerOfTwo(m_HelperStandard.m_Height))))
    {
        m_HelperStandard.m_SizeMaskX = m_HelperStandard.m_Width - 1;
        m_HelperStandard.m_SizeMaskY = m_HelperStandard.m_Height - 1;
        m_HelperStandard.m_SizeShiftX = std::countr_zero(uint32_t(m_HelperStandard.m_Width));
    }
    else
    {
        m_HelperStandard.m_SizeMaskX = 0;
        m_HelperStandard.m_SizeMaskY = 0;
    }

    m_WidthBias = 0.001f + m_HelperStandard.m_Width * 100;
    m_HeightBias = 0.001f + m_HelperStandard.m_Height * 100;

    m_fData.resize(m_Data.size());
    for (size_t i = 0; i < m_Data.size(); ++i)
    {
        m_fData[i] = Vector4f::FromARGB(m_Data[i]);
    }

    auto Init = [this](auto& Helper)
        {
            Helper.m_Width = m_HelperStandard.m_Width;
            Helper.m_Height = m_HelperStandard.m_Height;

            Helper.m_MaxWidth = m_HelperStandard.m_MaxWidth;
            Helper.m_MaxHeight = m_HelperStandard.m_MaxHeight;

            Helper.m_fWidth = m_HelperStandard.m_fWidth;
            Helper.m_fHeight = m_HelperStandard.m_fHeight;

            Helper.m_SizeShiftX = m_HelperStandard.m_SizeShiftX;
            Helper.m_SizeMaskX = m_HelperStandard.m_SizeMaskX;
            Helper.m_SizeMaskY = m_HelperStandard.m_SizeMaskY;

            //m_HelperSSE. m_fMaxWidthSSE = f256A{ m_fWidth-1 };
            //m_HelperSSE. m_fMaxHeightSSE= f256A{ m_fHeight-1 };
        };

    Init(m_HelperCPU);
    Init(m_HelperSSE);
    Init(m_HelperSSE8);
    Init(m_HelperAVX);
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
        printf("Texture size is too big: %d x %d, max is %d x %d\n", width, height, MAX_TEXTURE_SIZE, MAX_TEXTURE_SIZE);
        return false;
    }

    STB::stbi_uc* data = STB::stbi_load(fileName, &width, &height, &channels, 4);
    if (!data)
        return false;

    m_Data.resize(width * height);
    memcpy(m_Data.data(), data, width * height * 4);
    m_HelperStandard.m_Width = width;
    m_HelperStandard.m_Height = height;
    STB::stbi_image_free(data);

    InitSimdHelpers();

    return true;
}

bool Texture::IsValid()const
{
    return m_Data.size() > 0;
}