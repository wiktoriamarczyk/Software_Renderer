#pragma once
#include "Common.h"
#include "Vector3f.h"

class Matrix4
{
public:
    constexpr Matrix4();
    constexpr Matrix4(const Matrix4&)=default;
    constexpr Matrix4(float m00, float m01, float m02, float m03,
                      float m10, float m11, float m12, float m13,
                      float m20, float m21, float m22, float m23,
                      float m30, float m31, float m32, float m33)
        : m_Matrix{{m00, m01, m02, m03},
            {m10, m11, m12, m13},
            {m20, m21, m22, m23},
            {m30, m31, m32, m33}}
    {
    }

    static constexpr Matrix4 Identity();

    Matrix4 operator*(const Matrix4& other) const;
    Matrix4& operator*=(const Matrix4& other);
    float& operator[](int index);
    const float& operator[](int index)const;


    static Matrix4 Translation(Vector3f other);
    static Matrix4 Rotation(Vector3f other);
    static Matrix4 Scale(Vector3f other);

    float m_Matrix[4][4] = {};   // matrix elements; first index is for rows, second for columns (row-major)

};

constexpr Matrix4::Matrix4()
    : Matrix4(Identity())
{
}

constexpr Matrix4 Matrix4::Identity()
{
    constexpr Matrix4 result(1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1);
    return result;
}
