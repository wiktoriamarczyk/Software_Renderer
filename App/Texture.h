/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2023
*/

#pragma once
#include "IRenderer.h"

class Texture : public ITexture
{
public:
    Texture() = default;
    bool CreateWhite4x4Tex();
    bool Load(const char* fileName);
    bool IsValid()const override;
    Vector4f Sample(Vector2f uv)const;
private:
    vector<uint32_t> m_Data;
    int              m_Width = 0;
    int              m_Height = 0;

};