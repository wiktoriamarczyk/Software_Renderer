/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Common.h"

class Matrix4f;
class Vector3f;

/**
* Klasa reprezentuj¹ca wektor 2D.
*/
class Vector2f {
public:
    Vector2f() = default;
    /**
    * Konstruktor klasy.
    * @param x wspó³rzêdna x
    * @param y wspó³rzêdna y
    */
    Vector2f(float x, float y);
    /**
    * Konstruktor klasy.
    * @param vector wektor 3D
    */
    Vector2f(Vector3f vector);

    /**
    * Operator dodawania wektorów.
    * @param other drugi parametr dodawania
    * @return wektor wynikowy
    */
    Vector2f operator+(Vector2f other)const;
    /**
    * Operator odejmowania wektorów.
    * @param other drugi parametr odejmowania
    * @return wektor wynikowy
    */
    Vector2f operator-(Vector2f other)const;
    /**
    * Operator mno¿enia wektora przez liczbê.
    * @param value liczba
    * @return wektor wynikowy
    */
    Vector2f operator*(float value)const;
    /**
    * Operator dzielenia wektora przez liczbê.
    * @param value liczba
    * @return wektor wynikowy
    */
    Vector2f operator/(float value)const;

    /**
    * Operator mno¿enia wektora przez liczbê.
    * @param value liczba
    * @param v wektor
    * @return wektor wynikowy
    */
    friend Vector2f operator*(float value, const Vector2f& v);

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
    float Dot(const Vector2f& other)const;
    /**
    * Normalizuje wektor.
    * @return wektor jednostkowy
    */
    Vector2f& Normalize();
    /**
    * Zwraca wektor jednostkowy.
    * @return wektor jednostkowy
    */
    Vector2f Normalized()const;
    /**
    * Zwraca wektor o najmniejszych wspó³rzêdnych.
    * @param other drugi wektor
    * @return wektor o najmniejszych wspó³rzêdnych
    */
    Vector2f CWiseMin(const Vector2f& other)const;
    /**
    * Zwraca wektor o najwiêkszych wspó³rzêdnych.
    * @param other drugi wektor
    * @return wektor o najwiêkszych wspó³rzêdnych
    */
    Vector2f CWiseMax(const Vector2f& other)const;

    float x = 0; ///< wspó³rzêdna x
    float y = 0; ///< wspó³rzêdna y
};

inline Vector2f::Vector2f(float x, float y)
{
    this->x = x;
    this->y = y;
}

inline Vector2f Vector2f::operator+(Vector2f other)const
{
    return Vector2f(x + other.x, y + other.y);
}

inline Vector2f Vector2f::operator-(Vector2f other)const
{
    return Vector2f(x - other.x, y - other.y);
}

inline Vector2f Vector2f::operator*(float value)const
{
    return Vector2f(x * value, y * value);
}

inline Vector2f Vector2f::operator/(float value)const
{
    return Vector2f(x / value, y / value);
}

inline Vector2f operator*(float value, const Vector2f& v)
{
    return Vector2f(v.x * value, v.y * value);
}