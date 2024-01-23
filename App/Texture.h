/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "IRenderer.h"

/**
* Klasa tekstury renderera programowego. Implementuje interfejs ITexture.
*/
class Texture : public ITexture
{
public:
    Texture() = default;
    /**
    * Tworzy teksturê 4x4 wype³nion¹ bia³ym kolorem.
    */
    bool CreateWhite4x4Tex();
    /**
    * £aduje teksturê z pliku.
    * @param fileName œcie¿ka do pliku
    * @return true jeœli tekstura zosta³a za³adowana, false w przeciwnym wypadku
    */
    bool Load(const char* fileName);
    /**
    * Sprawdza czy tekstura jest poprawna.
    * @return true jeœli tekstura jest poprawna, false w przeciwnym wypadku
    */
    bool IsValid()const override;
    /**
    * Mapuje wspó³rzêdne UV na kolor tekstury.
    * @param uv wspó³rzêdne tekstury
    * @return kolor tekstury w danym punkcie
    */
    Vector4f Sample(Vector2f uv)const;
private:
    vector<uint32_t> m_Data; ///< dane tekstury
    int              m_Width = 0; ///< szerokoœæ tekstury
    int              m_Height = 0; ///< wysokoœæ tekstury

};