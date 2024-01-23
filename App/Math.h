/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Vector2f.h"
#include "Vector3f.h"
#include "Vector4f.h"

/**
* Struktura reprezentuj¹ca wierzcho³ek.
*/
struct Vertex
{
    Vector3f position; ///< pozycja
    Vector3f normal; ///< normalna
    Vector4f color; ///< kolor
    Vector2f uv; ///< wspó³rzêdne tekstury

    /**
    * Operator mno¿enia wierzcho³ka przez liczbê.
    * @param value liczba
    * @return wierzcho³ek wynikowy
    */
    Vertex operator*(float value)const
    {
        Vertex result;
        result.position = position * value;
        result.normal = normal * value;
        result.color = color * value;
        result.uv = uv * value;
        return result;
    }
    /**
    * Operator dodawania wierzcho³ków.
    * @param vertex drugi parametr dodawania
    * @return wierzcho³ek wynikowy
    */
    Vertex operator+(const Vertex& vertex)const
    {
        Vertex result;
        result.position = position + vertex.position;
        result.normal = normal + vertex.normal;
        result.color = color + vertex.color;
        result.uv = uv + vertex.uv;
        return result;
    }

    static const uint32_t Stride; ///< rozmiar wierzcho³ka
    static const uint32_t PositionOffset; ///< offset pozycji
    static const uint32_t NormalOffset; ///< offset normalnej
    static const uint32_t ColorOffset; ///< offset koloru
    static const uint32_t UVOffset; ///< offset wspó³rzêdnych tekstury
};

constexpr inline const uint32_t Vertex::Stride          = sizeof(Vertex);
constexpr inline const uint32_t Vertex::PositionOffset  = offsetof(Vertex, position);
constexpr inline const uint32_t Vertex::NormalOffset    = offsetof(Vertex, normal);
constexpr inline const uint32_t Vertex::ColorOffset     = offsetof(Vertex, color);
constexpr inline const uint32_t Vertex::UVOffset        = offsetof(Vertex, uv);

/**
* Klasa p³aszczyzny.
*/
class Plane
{
public:
    /**
    * Enum okreœlaj¹cy po której stronie p³aszczyzny znajduje siê punkt.
    */
    enum class eSide : uint8_t
    {
        Back = 0,
        Front = 1,
        On = 2
    };

    Plane() = default;
    /**
    * Konstruktor p³aszczyzny.
    * @param normal normalna p³aszczyzny
    * @param N odleg³oœæ od pocz¹tku uk³adu wspó³rzêdnych
    */
    Plane(const Vector3f& normal, float N);
    /**
    * Konstruktor p³aszczyzny.
    * @param a pierwszy punkt
    * @param b drugi punkt
    * @param c trzeci punkt
    */
    Plane(const Vector3f& a, const Vector3f& b, const Vector3f& c);
    /**
    * Wylicza odleg³oœæ punktu od p³aszczyzny.
    * @param point punkt
    * @return odleg³oœæ punktu od p³aszczyzny
    */
    float Distance(const Vector3f& point)const;
    /**
    * Sprawdza czy odcinek przecina p³aszczyznê.
    * @param start pocz¹tek odcinka
    * @param end koniec odcinka
    * @param scale wspó³czynnik przeciêcia
    */
    bool  LineIntersection(const Vector3f& start, const Vector3f& end, float& scale) const;
    /**
    * Zwraca normaln¹ p³aszczyzny.
    * @return normalna p³aszczyzny
    */
    const Vector3f& GetNormal()const { return m_Normal; }
    /**
    * Zwraca odleg³oœæ od pocz¹tku uk³adu wspó³rzêdnych.
    * @return odleg³oœæ od pocz¹tku uk³adu wspó³rzêdnych
    */
    float GetD()const { return m_D; }
    /**
    * Zwraca po której stronie p³aszczyzny znajduje siê punkt.
    * @param point punkt
    * @param epsilon dok³adnoœæ
    * @return po której stronie p³aszczyzny znajduje siê punkt
    */
    eSide GetSide(const Vector3f& point, float epsilon = 0.0f)const;
private:
    Vector3f m_Normal; ///< normalna p³aszczyzny
    float    m_D = 0; ///< odleg³oœæ od pocz¹tku uk³adu wspó³rzêdnych
};

/**
* Przycina trójk¹ty do p³aszczyzny przycinania.
* @param clipPlane p³aszczyzna przycinania
* @param epsilon dok³adnoœæ
* @param verts wektor wierzcho³ków
* @return wektor przyciêtych wierzcho³ków
*/
const vector<Vertex>& ClipTriangles(const Plane& clipPlane, const float epsilon, const vector<Vertex>& verts);