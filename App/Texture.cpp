#include "Texture.h"
#include "../stb/stb_image.h"

bool Texture::Load(const char* fileName)
{
    int width, height, channels;
    //STB::stbi_set_flip_vertically_on_load(true);

    auto result = STB::stbi_info(fileName, &width, &height, &channels);
    if (!result)
        return false;

    if (width>MAX_TEXTURE_SIZE || height>MAX_TEXTURE_SIZE)
    {
        printf("Texture size is too big: %d x %d, max is (%d x %d)\n", width, height, MAX_TEXTURE_SIZE , MAX_TEXTURE_SIZE);
        return false;
    }

    STB::stbi_uc* data = STB::stbi_load(fileName, &width, &height, &channels, 4);
    if (!data) {
        return false;
    }

    m_Data.resize(width * height);
    memcpy(m_Data.data(), data, width * height * 4);
    m_Width = width;
    m_Height = height;
    STB::stbi_image_free(data);

    return true;
}

bool Texture::IsValid()const
{
    return m_Data.size() > 0;
}

Vector4f Texture::Sample(Vector2f uv) const
{
    int x = std::clamp<int>( uv.x * (m_Width  - 1) , 0 , m_Width - 1);
    int y = std::clamp<int>( uv.y * (m_Height - 1) , 0 , m_Height - 1);

    int pixelIndex = y * m_Width + x;

    return Vector4f::FromARGB(m_Data[pixelIndex]);
}