#pragma once
#include "Common.h"
#include "Vector3f.h"

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

    bool GetFrustumNearPlane(Plane& OuTPln) const;

    Matrix4f Transposed()const;
    Matrix4f Inversed() const;

    float m_Matrix[4][4] = {};   // matrix elements; first index is for rows, second for columns (row-major)
private:
    static bool MakeFrustumPlane(float A, float B, float C, float D, Plane& OutPlane);
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