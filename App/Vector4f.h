/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Common.h"
#include "Vector2f.h"
#include "Vector3f.h"

class Matrix4f;
class Vector2f;
class Vector3f;

/**
* Klasa reprezentuj¹ca wektor 4D.
*/
class Vector4f
{
public:
    Vector4f() = default;
    /**
    * Konstruktor klasy.
    * @param v wierzcho³ek 3D
    * @param w wspó³rzêdna w
    */
    Vector4f(const Vector3f& v,float w);
    /**
    * Konstruktor klasy.
    * @param x wspó³rzêdna x
    * @param y wspó³rzêdna y
    * @param z wspó³rzêdna z
    * @param w wspó³rzêdna w
    */
    Vector4f(float x, float y, float z, float w);

    /**
    * Operator dodawania wektorów.
    * @param other drugi parametr dodawania
    * @return wektor wynikowy
    */
    Vector4f operator+(Vector4f other)const;
    /**
    * Operator odejmowania wektorów.
    * @param other drugi parametr odejmowania
    * @return wektor wynikowy
    */
    Vector4f operator-(Vector4f other)const;
    /**
    * Operator mno¿enia wektorów.
    * @param other drugi parametr mno¿enia
    * @return wektor wynikowy
    */
    Vector4f operator*(Vector4f other)const;
    /**
    * Operator mno¿enia wektora przez liczbê.
    * @param value liczba
    * @return wektor wynikowy
    */
    Vector4f operator*(float value)const;
    /**
    * Operator dzielenia wektora przez liczbê.
    * @param value liczba
    * @return wektor wynikowy
    */
    Vector4f operator/(float value)const;

    /**
    * Operator mno¿enia liczby przez wektor.
    * @param value liczba
    * @param v wektor
    * @return wektor wynikowy
    */
    friend Vector4f operator*(float value, const Vector4f& v);

    /**
    * Zwraca d³ugoœæ wektora.
    * @return d³ugoœæ wektora
    */
    float GetLength()const;
    /**
    * Zwraca iloczyn skalarny wektorów.
    * @param other drugi wektor
    * @return iloczyn skalarny
    */
    float Dot(const Vector4f& other)const;
    /**
    * Normalizuje wektor.
    * @return wektor jednostkowy
    */
    Vector4f& Normalize();
    /**
    * Zwraca wektor jednostkowy.
    * @return wektor jednostkowy
    */
    Vector4f Normalized()const;
    /**
    * Zwraca wektor przetransformowany przez macierz.
    * @param m macierz
    * @return wektor wynikowy
    */
    Vector4f Transformed(const Matrix4f& m) const;
    /**
    * Zwraca wektor o najmniejszych wspó³rzêdnych.
    * @param other drugi wektor
    * @return wektor o najmniejszych wspó³rzêdnych
    */
    Vector4f CWiseMin(const Vector4f& other)const;
    /**
    * Zwraca wektor o najwiêkszych wspó³rzêdnych.
    * @param other drugi wektor
    * @return wektor o najwiêkszych wspó³rzêdnych
    */
    Vector4f CWiseMax(const Vector4f& other)const;
    /**
    * Zwraca wektor 2D.
    * @return wektor 2D
    */
    Vector2f xy()const;
    /**
    * Zwraca wektor 3D.
    * @return wektor 3D
    */
    Vector3f xyz()const;

    /**
    * Zamienia kolor z reprezentacji bêd¹cej wektorem na postaæ uint32_t.
    * @param color kolor jako wektor 4D
    * @return kolor jako uint32_t
    */
    static uint32_t ToARGB(const Vector4f& color);
    /**
    * Zamienia kolor z reprezentacji uint32_t na postaæ wektora.
    * @param color jako uint32_t
    * @return kolor jako wektor 4D
    */
    static Vector4f FromARGB(uint32_t color);

    float x = 0; ///< wspó³rzêdna x
    float y = 0; ///< wspó³rzêdna y
    float z = 0; ///< wspó³rzêdna z
    float w = 0; ///< wspó³rzêdna w
};

inline Vector4f::Vector4f(float x, float y, float z, float w)
{
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
}

inline Vector4f Vector4f::operator+(Vector4f other)const
{
    return Vector4f(x + other.x, y + other.y, z + other.z, w + other.w);
}

inline Vector4f Vector4f::operator-(Vector4f other)const
{
    return Vector4f(x - other.x, y - other.y, z - other.z, w - other.w);
}

inline Vector4f Vector4f::operator*(Vector4f other) const
{
    return Vector4f(x * other.x, y * other.y, z * other.z, w * other.w);
}

inline Vector4f Vector4f::operator*(float value)const
{
    return Vector4f(x * value, y * value, z * value, w * value);
}

inline Vector4f Vector4f::operator/(float value)const
{
    return Vector4f(x / value, y / value, z / value, w / value);
}

inline Vector4f operator*(float value, const Vector4f& v)
{
    return Vector4f(v.x * value, v.y * value, v.z * value, v.w * value);
}

inline Vector3f Vector4f::xyz() const
{
    return Vector3f(x, y, z);
}

inline Vector2f Vector4f::xy() const
{
    return Vector2f(x, y);
}

inline uint32_t Vector4f::ToARGB(const Vector4f& color)
{
    return (uint32_t)(color.w * 255) << 24 | (uint32_t)(color.z * 255) << 16 | (uint32_t)(color.y * 255) << 8 | (uint32_t)(color.x * 255);
}