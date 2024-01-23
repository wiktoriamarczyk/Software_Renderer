/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Common.h"
#include "TransformedVertex.h"

/**
* Klasa pomocnicza s³u¿¹ca do interpolacji parametrów punktu w trójk¹cie.
*/
class VertexInterpolator
{
public:
    /**
    * Konstruktor klasy.
    * @param A pierwszy wierzcho³ek
    * @param B drugi wierzcho³ek
    * @param C trzeci wierzcho³ek
    */
    VertexInterpolator(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C);
    /**
    * Interpoluje wartoœæ Z dla punktu w trójk¹cie.
    * @param baricentric wspó³rzêdne barycentryczne punktu
    * @param out punkt wynikowy
    */
    void InterpolateZ(const Vector3f& baricentric, TransformedVertex& out);
    /**
    * Interpoluje pozosta³e atrybuty punktu w trójk¹cie.
    * @param baricentric wspó³rzêdne barycentryczne punktu
    * @param out punkt wynikowy
    */
    void InterpolateAllButZ(const Vector3f& baricentric, TransformedVertex& out);

private:
    /**
    * Struktura reprezentuj¹ca Ÿród³o interpolacji.
    */
    struct InterpolatedSource
    {
        Vector3f    worldPositionOverW; ///< pozycja w œwiecie
        Vector3f    normalOverW; ///< normalna
        Vector2f    uvOverW; ///< wspó³rzêdne tekstury
        Vector4f    colorOverW; ///< kolor
        float       oneOverW; ///< odwrotnoœæ wspó³rzêdnej w
        float       screenPositionZ; ///< pozycja na ekranie wspó³rzêdnej z
    };

    InterpolatedSource m_A; ///< pierwszy wierzcho³ek
    InterpolatedSource m_B; ///< drugi wierzcho³ek
    InterpolatedSource m_C; ///< trzeci wierzcho³ek
};

inline void VertexInterpolator::InterpolateZ(const Vector3f& baricentric, TransformedVertex& out)
{
    out.screenPosition.z = baricentric.x * m_A.screenPositionZ + baricentric.y * m_B.screenPositionZ + baricentric.z * m_C.screenPositionZ;
}

inline void VertexInterpolator::InterpolateAllButZ(const Vector3f& baricentric, TransformedVertex& out)
{
    float oneOverW = baricentric.x * m_A.oneOverW + baricentric.y * m_B.oneOverW + baricentric.z * m_C.oneOverW;

    float w = 1.0f / oneOverW;

    out.color           = (baricentric.x * m_A.colorOverW + baricentric.y * m_B.colorOverW + baricentric.z * m_C.colorOverW) * w;
    out.normal          = (baricentric.x * m_A.normalOverW + baricentric.y * m_B.normalOverW + baricentric.z * m_C.normalOverW) * w;
    out.uv              = (baricentric.x * m_A.uvOverW + baricentric.y * m_B.uvOverW + baricentric.z * m_C.uvOverW) * w;
    out.worldPosition   = (baricentric.x * m_A.worldPositionOverW + baricentric.y * m_B.worldPositionOverW + baricentric.z * m_C.worldPositionOverW) * w;
    out.uv              = (baricentric.x * m_A.uvOverW + baricentric.y * m_B.uvOverW + baricentric.z * m_C.uvOverW) * w;
}