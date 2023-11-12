#pragma once
#include "Common.h"
#include "Matrix4.h"


Matrix4f Matrix4f::Translation(Vector3f other)
{
    Matrix4f result = Matrix4f::Identity();
    result.m_Matrix[3][0] = other.x;
    result.m_Matrix[3][1] = other.y;
    result.m_Matrix[3][2] = other.z;
    return result;
}

Matrix4f Matrix4f::Rotation(Vector3f other) {
    const float xs = sinf(other.x);
    const float xc = cosf(other.x);
    const float ys = sinf(other.y);
    const float yc = cosf(other.y);
    const float zs = sinf(other.z);
    const float zc = cosf(other.z);

    return Matrix4f(
        yc * zc + ys * xs * zs, xc * zs, yc * xs * zs - ys * zc, 0.0f,
        ys * xs * zc - yc * zs, xc * zc, yc * xs * zc + ys * zs, 0.0f,
        ys * xc, -xs, yc * xc, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);
}

Matrix4f Matrix4f::Scale(Vector3f other) {
    return Matrix4f {
        other.x, 0.0f, 0.0f, 0.0f,
        0.0f, other.y, 0.0f, 0.0f,
        0.0f, 0.0f, other.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f };
}

Matrix4f Matrix4f::Transposed() const
{
    return Matrix4f(m_Matrix[0][0], m_Matrix[1][0], m_Matrix[2][0], m_Matrix[3][0],
                    m_Matrix[0][1], m_Matrix[1][1], m_Matrix[2][1], m_Matrix[3][1],
                    m_Matrix[0][2], m_Matrix[1][2], m_Matrix[2][2], m_Matrix[3][2],
                    m_Matrix[0][3], m_Matrix[1][3], m_Matrix[2][3], m_Matrix[3][3]);
}

Matrix4f Matrix4f::operator*(const Matrix4f& m) const
{
    return Matrix4f(m.m_Matrix[0][0] * m_Matrix[0][0] + m.m_Matrix[1][0] * m_Matrix[0][1] + m.m_Matrix[2][0] * m_Matrix[0][2] + m.m_Matrix[3][0] * m_Matrix[0][3],
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

float& Matrix4f::operator[](int index) {
    return m_Matrix[0][index];
}

const float& Matrix4f::operator[](int index)const {
   return m_Matrix[0][index];
}

/// <summary>
/// Projects 3D coordinates into 2D space
/// </summary>
/// <param name="fieldOfView"> the field of view in degrees</param>
/// <param name="near"> the near plane</param>
/// <param name="far"> the far plane</param>
Matrix4f Matrix4f::CreateProjectionMatrix(float fieldOfView, float aspectRatio, float near, float far)
{
    Matrix4f result;

    float const tanHalfFovy = tan((fieldOfView / 180 * std::numbers::pi) / 2.f);

    result.m_Matrix[0][0] = 1.f / (aspectRatio * tanHalfFovy);
    result.m_Matrix[1][1] = 1.f / (tanHalfFovy);
    result.m_Matrix[3][2] = 1.f;

    result.m_Matrix[2][2] = (far + near) / (far - near);
    result.m_Matrix[2][3] = -(2.f * far * near) / (far - near);
    result.m_Matrix[3][3] = 0.f;

    return result;
}

/// <summary>
///
/// </summary>
/// <param name="eye"></param>
/// <param name="target"></param>
/// <param name="up"></param>
/// <returns></returns>
Matrix4f Matrix4f::CreateLookAtMatrix(const Vector3f& eye, const Vector3f& target, const Vector3f& up)
{
    // Compute direction of gaze. (-Z)
    Vector3f zAxis = (eye - target).Normalized();

    // Compute orthogonal axes from cross product of gaze and up vector.
    Vector3f xAxis = up.Cross(zAxis).Normalized();
    Vector3f yAxis = zAxis.Cross(xAxis);

    Matrix4f result;
    // Set rotation and translate by eye
    result.m_Matrix[0][0] = xAxis.x;
    result.m_Matrix[1][0] = xAxis.y;
    result.m_Matrix[2][0] = xAxis.z;
    result.m_Matrix[3][0] = -xAxis.Dot(eye);
    result.m_Matrix[0][1] = yAxis.x;
    result.m_Matrix[1][1] = yAxis.y;
    result.m_Matrix[2][1] = yAxis.z;
    result.m_Matrix[3][1] = -yAxis.Dot(eye);
    result.m_Matrix[0][2] = zAxis.x;
    result.m_Matrix[1][2] = zAxis.y;
    result.m_Matrix[2][2] = zAxis.z;
    result.m_Matrix[3][2] = -zAxis.Dot(eye);
    result.m_Matrix[0][3] = 0.0f;
    result.m_Matrix[1][3] = 0.0f;
    result.m_Matrix[2][3] = 0.0f;
    result.m_Matrix[3][3] = 1.0f;

    return result;
}