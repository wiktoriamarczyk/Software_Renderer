/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Common.h"

class Matrix4f;

/**
* Klasa reprezentuj¹ca wektor 3D.
*/
class Vector3f
{
public:
    Vector3f()=default;
    /**
    * Konstruktor klasy.
    * @param x wspó³rzêdna x
    * @param y wspó³rzêdna y
    */
    Vector3f(float x, float y, float z);

    /**
    * Operator dodawania wektorów.
    * @param other drugi parametr dodawania
    * @return wektor wynikowy
    */
    Vector3f operator+(Vector3f other)const;
    /**
    * Operator odejmowania wektorów.
    * @param other drugi parametr odejmowania
    * @return wektor wynikowy
    */
    Vector3f operator-(Vector3f other)const;
    /**
    * Operator mno¿enia wektora przez liczbê.
    * @param value liczba
    * @return wektor wynikowy
    */
    Vector3f operator*(float value)const;
    /**
    * Operator dzielenia wektora przez liczbê.
    * @param value liczba
    * @return wektor wynikowy
    */
    Vector3f operator/(float value)const;

    /**
    * Operator mno¿enia wektora przez liczbê.
    * @param value liczba
    * @param v wektor
    * @return wektor wynikowy
    */
    friend Vector3f operator*(float value, const Vector3f& v);

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
    float Dot(const Vector3f& other)const;
    /**
    * Zwraca najwiêksz¹ wspó³rzêdn¹ wektora.
    * @return najwiêksza wspó³rzêdna
    */
    float MaxComponent()const;
    /**
    * Normalizuje wektor.
    * @return wektor jednostkowy
    */
    Vector3f& Normalize();
    /**
    * Zwraca wektor jednostkowy.
    * @return wektor jednostkowy
    */
    Vector3f Normalized()const;
    /**
    * Zwraca wektor przemno¿ony przez macierz.
    * @param m macierz
    * @return wektor wynikowy
    */
    Vector3f Multiplied(const Matrix4f& m) const;
    /**
    * Zwraca wektor przetransformowany przez macierz.
    * @param m macierz
    * @return wektor wynikowy
    */
    Vector3f Transformed(const Matrix4f& m) const;
    /**
    * Zwraca wektor przetransformowany przez macierz.
    * @param m macierz
    * @return wektor wynikowy
    */
    Vector3f TransformedVec(const Matrix4f& m) const;
    /**
    * Zwraca iloczyn wektorowy wektorów.
    * @param other drugi wektor
    * @return iloczyn wektorowy
    */
    Vector3f Cross(const Vector3f& other)const;
    /**
    * Zwraca wektor o najmniejszych wspó³rzêdnych.
    * @param other drugi wektor
    * @return wektor o najmniejszych wspó³rzêdnych
    */
    Vector3f CWiseMin(const Vector3f& other)const;
    /**
    * Zwraca wektor o najwiêkszych wspó³rzêdnych.
    * @param other drugi wektor
    * @return wektor o najwiêkszych wspó³rzêdnych
    */
    Vector3f CWiseMax(const Vector3f& other)const;
    /**
    * Zwraca wektor z³o¿ony z wartoœci bezwzglêdnych wspó³rzêdnych.
    * @param other drugi wektor
    * @return wektor z³o¿ony z wartoœci bezwzglêdnych wspó³rzêdnych
    */
    Vector3f CWiseAbs(const Vector3f& other)const;
    /**
    * Zwraca wektor odbity wzglêdem normalnej.
    * @param normal normalna
    * @return wektor odbity
    */
    Vector3f Reflect(const Vector3f& normal)const;

    float x = 0; ///< wspó³rzêdna x
    float y = 0; ///< wspó³rzêdna y
    float z = 0; ///< wspó³rzêdna z
};


inline Vector3f::Vector3f(float x, float y, float z)
{
    this->x = x;
    this->y = y;
    this->z = z;
}

inline Vector3f Vector3f::operator+(Vector3f other)const
{
    return Vector3f(x + other.x, y + other.y, z + other.z);
}

inline Vector3f Vector3f::operator-(Vector3f other)const
{
    return Vector3f(x - other.x, y - other.y, z - other.z);
}

inline Vector3f Vector3f::operator*(float value)const
{
    return Vector3f(x * value, y * value, z * value);
}

inline Vector3f Vector3f::operator/(float value)const
{
    return Vector3f(x / value, y / value, z / value);
}

inline Vector3f operator*(float value, const Vector3f& v)
{
    return Vector3f(v.x * value, v.y * value, v.z * value);
}