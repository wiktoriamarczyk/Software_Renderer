/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Common.h"
#include "Vector4f.h"

class Plane;

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

    static constexpr Matrix4f Identity();

    Matrix4f operator*(const Matrix4f& other) const;
    Matrix4f& operator*=(const Matrix4f& other);
    float& operator[](int index);
    const float& operator[](int index)const;

    static Matrix4f CreateProjectionMatrix(float fieldOfView, float aspectRatio, float near, float far);
    static Matrix4f CreateLookAtMatrix(const Vector3f& eye, const Vector3f& target, const Vector3f& up);
    static Matrix4f Translation(Vector3f other);
    static Matrix4f Rotation(Vector3f other);
    static Matrix4f Scale(Vector3f other);

    bool GetFrustumNearPlane(Plane& outPlane) const;
    bool GetFrustumFarPlane(Plane& OutPlane) const;
    bool GetFrustumLeftPlane(Plane& OutPlane) const;
    bool GetFrustumRightPlane(Plane& OutPlane) const;
    bool GetFrustumTopPlane(Plane& OutPlane) const;
    bool GetFrustumBottomPlane(Plane& OutPlane) const;

    Matrix4f Transposed()const;
    Matrix4f Inversed() const;

    float m_Matrix[4][4] = {};   // matrix elements; first index is for rows, second for columns (row-major)
private:
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

template< typename T >
inline Vector3<T> Vector3<T>::Multiplied(const Matrix4f& m) const
{
    return Vector3f(
        (m[0] * x + m[4] * y + m[8] * z + m[12])  ,
        (m[1] * x + m[5] * y + m[9] * z + m[13])  ,
        (m[2] * x + m[6] * y + m[10] * z + m[14]) );
}

template< typename T >
inline Vector3<T> Vector3<T>::Transformed(const Matrix4f& m) const
{
    float w = 1.f / (m[3] * x + m[7] * y + m[11] * z + m[15]);
    return Vector3<T>(
        (m[0] * x + m[4] * y + m[8] * z + m[12]) * w,
        (m[1] * x + m[5] * y + m[9] * z + m[13]) * w,
        (m[2] * x + m[6] * y + m[10] * z + m[14]) * w);
}

template< typename T >
inline Vector3<T> Vector3<T>::TransformedVec(const Matrix4f& m) const
{
    return Vector3<T>(
        m[0] * x + m[4] * y + m[ 8] * z,
        m[1] * x + m[5] * y + m[ 9] * z,
        m[2] * x + m[6] * y + m[10] * z);
}

template< typename T >
Vector4<T> Vector4<T>::Transformed(const Matrix4f& m) const
{
    return Vector4<T>(m[0] * x + m[4] * y + m[8]  * z + m[12] * w,
                      m[1] * x + m[5] * y + m[9]  * z + m[13] * w,
                      m[2] * x + m[6] * y + m[10] * z + m[14] * w,
                      m[3] * x + m[7] * y + m[11] * z + m[15] * w);
}
