#pragma once
#include "Common.h"
#include "Matrix4.h"


Matrix4 Matrix4::Translation(Vector3f other)
{
       Matrix4 result = Matrix4::Identity();
    result.m_Matrix[3][0] = other.x;
    result.m_Matrix[3][1] = other.y;
    result.m_Matrix[3][2] = other.z;
    return result;
}

Matrix4 Matrix4::Rotation(Vector3f other) {
    const float xs = sinf(other.x);
    const float xc = cosf(other.x);
    const float ys = sinf(other.y);
    const float yc = cosf(other.y);
    const float zs = sinf(other.z);
    const float zc = cosf(other.z);

    return Matrix4(
        yc * zc + ys * xs * zs, xc * zs, yc * xs * zs - ys * zc, 0.0f,
        ys * xs * zc - yc * zs, xc * zc, yc * xs * zc + ys * zs, 0.0f,
        ys * xc, -xs, yc * xc, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);
}

Matrix4 Matrix4::Scale(Vector3f other) {
    return Matrix4 {
        other.x, 0.0f, 0.0f, 0.0f,
        0.0f, other.y, 0.0f, 0.0f,
        0.0f, 0.0f, other.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f };
}

Matrix4 Matrix4::operator*(const Matrix4& m) const
{
    return Matrix4(m.m_Matrix[0][0] * m_Matrix[0][0] + m.m_Matrix[1][0] * m_Matrix[0][1] + m.m_Matrix[2][0] * m_Matrix[0][2] + m.m_Matrix[3][0] * m_Matrix[0][3],
        m.m_Matrix[0][1] * m_Matrix[0][0] + m.m_Matrix[1][1] * m_Matrix[0][1] + m.m_Matrix[2][1] * m_Matrix[0][2] + m.m_Matrix[3][1] * m_Matrix[0][3],
        m.m_Matrix[0][2] * m_Matrix[0][0] + m.m_Matrix[1][2] * m_Matrix[0][1] + m.m_Matrix[2][2] * m_Matrix[0][2] + m.m_Matrix[3][2] * m_Matrix[0][3],
        m.m_Matrix[0][3] * m_Matrix[0][0] + m.m_Matrix[1][3] * m_Matrix[0][1] + m.m_Matrix[2][3] * m_Matrix[0][2] + m.m_Matrix[3][3] * m_Matrix[0][3],

        m.m_Matrix[0][0] * m_Matrix[1][0] + m.m_Matrix[1][0] * m_Matrix[1][1] + m.m_Matrix[2][0] * m_Matrix[1][2] + m.m_Matrix[3][0] * m_Matrix[1][3],
        m.m_Matrix[0][1] * m_Matrix[1][0] + m.m_Matrix[1][1] * m_Matrix[1][1] + m.m_Matrix[2][1] * m_Matrix[1][2] + m.m_Matrix[3][1] * m_Matrix[1][3],
        m.m_Matrix[0][2] * m_Matrix[1][0] + m.m_Matrix[1][2] * m_Matrix[1][1] + m.m_Matrix[2][2] * m_Matrix[1][2] + m.m_Matrix[3][2] * m_Matrix[1][3],
        m.m_Matrix[0][3] * m_Matrix[1][0] + m.m_Matrix[1][3] * m_Matrix[1][1] + m.m_Matrix[2][3] * m_Matrix[1][2] + m.m_Matrix[3][3] * m_Matrix[1][3],
        m.m_Matrix[0][0] * m_Matrix[2][0] + m.m_Matrix[1][0] * m_Matrix[2][1] + m.m_Matrix[2][0] * m_Matrix[2][2] + m.m_Matrix[3][0] * m_Matrix[2][3],
        m.m_Matrix[0][1] * m_Matrix[2][0] + m.m_Matrix[1][1] * m_Matrix[2][1] + m.m_Matrix[2][1] * m_Matrix[2][2] + m.m_Matrix[3][1] * m_Matrix[2][3],
        m.m_Matrix[0][2] * m_Matrix[2][0] + m.m_Matrix[1][2] * m_Matrix[2][1] + m.m_Matrix[2][2] * m_Matrix[2][2] + m.m_Matrix[3][2] * m_Matrix[2][3],
        m.m_Matrix[0][3] * m_Matrix[2][0] + m.m_Matrix[1][3] * m_Matrix[2][1] + m.m_Matrix[2][3] * m_Matrix[2][2] + m.m_Matrix[3][3] * m_Matrix[2][3],

        m.m_Matrix[0][0] * m_Matrix[3][0] + m.m_Matrix[1][0] * m_Matrix[3][1] + m.m_Matrix[2][0] * m_Matrix[3][2] + m.m_Matrix[3][0] * m_Matrix[3][3],
        m.m_Matrix[0][1] * m_Matrix[3][0] + m.m_Matrix[1][1] * m_Matrix[3][1] + m.m_Matrix[2][1] * m_Matrix[3][2] + m.m_Matrix[3][1] * m_Matrix[3][3],
        m.m_Matrix[0][2] * m_Matrix[3][0] + m.m_Matrix[1][2] * m_Matrix[3][1] + m.m_Matrix[2][2] * m_Matrix[3][2] + m.m_Matrix[3][2] * m_Matrix[3][3],
        m.m_Matrix[0][3] * m_Matrix[3][0] + m.m_Matrix[1][3] * m_Matrix[3][1] + m.m_Matrix[2][3] * m_Matrix[3][2] + m.m_Matrix[3][3] * m_Matrix[3][3]);
}

float& Matrix4::operator[](int index) {
    return m_Matrix[0][index];
}

const float& Matrix4::operator[](int index)const {
   return m_Matrix[0][index];
}