/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Common.h"
#include "Vector3f.h"

class Plane;

/**
* Klasa reprezentuj¹ca macierz 4x4.
*/
class Matrix4f
{
public:
    constexpr Matrix4f();
    constexpr Matrix4f(const Matrix4f&)=default;
    constexpr Matrix4f(float m00, float m01, float m02, float m03,
                      float m10, float m11, float m12, float m13,
                      float m20, float m21, float m22, float m23,
                      float m30, float m31, float m32, float m33)
        : m_Matrix{{m00, m01, m02, m03},
            {m10, m11, m12, m13},
            {m20, m21, m22, m23},
            {m30, m31, m32, m33}}
    {
    }

    /**
    * Zwraca macierz jednostkow¹.
    * @return macierz jednostkowa
    */
    static constexpr Matrix4f Identity();
    /**
    * Operator mno¿enia macierzy przez inn¹ macierz.
    * @param other macierz
    * @return macierz wynikowa
    */
    Matrix4f operator*(const Matrix4f& other) const;
    /**
    * Operator mno¿enia macierzy przez inn¹ macierz z przypisaniem.
    * @param other macierz
    * @return macierz wynikowa
    */
    Matrix4f& operator*=(const Matrix4f& other);
    /**
    * Operator tablicowy.
    * @param index liniowy indeks
    * @return referencja do elementu macierzy
    */
    float& operator[](int index);
    /**
    * Operator tablicowy nie modyfikuj¹cy.
    * @param index liniowy indeks
    * @return referencja do elementu macierzy
    */
    const float& operator[](int index)const;
    /**
    * Tworzy macierz projekcji.
    * @param fieldOfView pole widzenia
    * @param aspectRatio wspó³czynnik proporcji ekranu
    * @param near odleg³oœæ od kamery do najbli¿szego punktu renderowania
    * @param far odleg³oœæ od kamery do najdalszego punktu renderowania
    * @return macierz projekcji
    */
    static Matrix4f CreateProjectionMatrix(float fieldOfView, float aspectRatio, float near, float far);
    /**
    * Tworzy macierz widoku.
    * @param eye pozycja kamery
    * @param target punkt, w którym kamera jest skierowana
    * @param up wektor okreœlaj¹cy kierunek góry
    * @return macierz widoku
    */
    static Matrix4f CreateLookAtMatrix(const Vector3f& eye, const Vector3f& target, const Vector3f& up);
    /**
    * Tworzy macierz translacji.
    * @param other wektor translacji
    * @return macierz translacji
    */
    static Matrix4f Translation(Vector3f other);
    /**
    * Tworzy macierz rotacji.
    * @param other wektor rotacji
    * @return macierz rotacji
    */
    static Matrix4f Rotation(Vector3f other);
    /**
    * Tworzy macierz skali.
    * @param other wektor skali
    * @return macierz skali
    */
    static Matrix4f Scale(Vector3f other);
    /**
    * Zwraca obiekt bliskiej p³aszczyzny odcinania.
    * @param outPlane p³aszczyzna odcinania
    * @return true, jeœli operacja siê powiod³a, false w przeciwnym wypadku
    */
    bool GetFrustumNearPlane(Plane& outPlane) const;
    bool GetFrustumFarPlane(Plane& OutPlane) const;
    bool GetFrustumLeftPlane(Plane& OutPlane) const;
    bool GetFrustumRightPlane(Plane& OutPlane) const;
    bool GetFrustumTopPlane(Plane& OutPlane) const;
    bool GetFrustumBottomPlane(Plane& OutPlane) const;

    /**
    * Transponuje macierz.
    * @return macierz transponowana
    */
    Matrix4f Transposed()const;
    /**
    * Odwraca macierz.
    * @return macierz odwrotna
    */
    Matrix4f Inversed() const;
    // matrix elements; first index is for rows, second for columns (row-major)
    float m_Matrix[4][4] = {}; ///< macierz
private:
    /**
    * Tworzy macierz przycinania.
    * @param a wspó³czynnik p³aszczyzny
    * @param b wspó³czynnik p³aszczyzny
    * @param c wspó³czynnik p³aszczyzny
    * @param d wspó³czynnik p³aszczyzny
    * @param outPlane p³aszczyzna przycinania
    * @return true, jeœli operacja siê powiod³a, false w przeciwnym wypadku
    */
    static bool MakeFrustumPlane(float a, float b, float c, float d, Plane& outPlane);
};

constexpr Matrix4f::Matrix4f()
    : Matrix4f(Identity())
{
}

constexpr Matrix4f Matrix4f::Identity()
{
    constexpr Matrix4f result(1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1);
    return result;
}

inline Vector3f Vector3f::Multiplied(const Matrix4f& m) const
{
    return Vector3f(
        (m[0] * x + m[4] * y + m[8] * z + m[12])  ,
        (m[1] * x + m[5] * y + m[9] * z + m[13])  ,
        (m[2] * x + m[6] * y + m[10] * z + m[14]) );
}

inline Vector3f Vector3f::Transformed(const Matrix4f& m) const
{
    float w = 1.f / (m[3] * x + m[7] * y + m[11] * z + m[15]);
    return Vector3f(
        (m[0] * x + m[4] * y + m[8] * z + m[12]) * w,
        (m[1] * x + m[5] * y + m[9] * z + m[13]) * w,
        (m[2] * x + m[6] * y + m[10] * z + m[14]) * w);
}

inline Vector3f Vector3f::TransformedVec(const Matrix4f& m) const
{
    return Vector3f(
        m[0] * x + m[4] * y + m[ 8] * z,
        m[1] * x + m[5] * y + m[ 9] * z,
        m[2] * x + m[6] * y + m[10] * z);
}

