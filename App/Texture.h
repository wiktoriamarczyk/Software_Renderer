#pragma once

#include "Common.h"
#include "Vector2f.h"
#include "Vector4f.h"

class Texture
{
public:
    Texture() = default;
    bool Load(const char* fileName);
    bool IsValid()const;
    Vector4f Sample(Vector2f uv)const;

private:
    vector<uint32_t> m_Data;
    int              m_Width = 0;
    int              m_Height = 0;

};