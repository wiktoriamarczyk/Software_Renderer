/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Common.h"
#include "Math.h"

/**
* Struktura reprezentuj¹ca wierzcho³ek po transformacji.
*/
struct TransformedVertex
{
    Vector4f screenPosition; ///< pozycja na ekranie
    Vector3f normal; ///< normalna
    Vector3f worldPosition; ///< pozycja w œwiecie
    Vector4f color; ///< kolor
    Vector2f uv; ///< wspó³rzêdne tekstury

    /**
    * Operator mno¿enia wierzcho³ka przez liczbê.
    * @param value liczba
    * @return wierzcho³ek wynikowy
    */
    TransformedVertex operator*(float value)const
    {
        TransformedVertex result;
        result.screenPosition = screenPosition * value;
       // result.zValue = zValue * value;
        result.normal = normal * value;
        result.worldPosition = worldPosition * value;
        result.color = color * value;
        result.uv = uv * value;
        return result;
    }

    /**
    * Operator dodawania wierzcho³ków.
    * @param vertex drugi parametr dodawania
    * @return wierzcho³ek wynikowy
    */
    TransformedVertex operator+(const TransformedVertex& vertex)const
    {
        TransformedVertex result;
        result.screenPosition = screenPosition + vertex.screenPosition;
        //result.zValue = zValue + vertex.zValue;
        result.normal = normal + vertex.normal;
        result.worldPosition = worldPosition + vertex.worldPosition;
        result.color = color + vertex.color;
        result.uv = uv + vertex.uv;
        return result;
    }

    /**
    * Transformuje wierzcho³ek z przestrzeni œwiata do przestrzeni ekranu.
    * @param v wierzcho³ek
    * @param worldMatrix macierz œwiata
    * @param mvpMatrix macierz MVP
    */
    void ProjToScreen(const Vertex& v, const Matrix4f& worldMatrix, const Matrix4f& mvpMatrix);
};

inline void TransformedVertex::ProjToScreen(const Vertex& v, const Matrix4f& worldMatrix, const Matrix4f& mvpMatrix)
{
    worldPosition = v.position.Multiplied(worldMatrix);
    normal        = v.normal.TransformedVec(worldMatrix).Normalized();
    color         = v.color;
    uv            = v.uv;

    screenPosition = Vector4f(v.position, 1.0f).Transformed(mvpMatrix);

    float oneOverW = 1.0f / screenPosition.w;

    screenPosition.x *= oneOverW;
    screenPosition.y *= oneOverW;
    screenPosition.z *= oneOverW;

    screenPosition.x = (screenPosition.x + 1) * SCREEN_WIDTH / 2;
    screenPosition.y = (screenPosition.y + 1) * SCREEN_HEIGHT / 2;
}