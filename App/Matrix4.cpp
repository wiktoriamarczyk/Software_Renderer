/*
* Master’s thesis - Analysis of selected optimization techniques for a 3D software renderer
* Author: Wiktoria Marczyk
* Year: 2025
*/

#pragma once
#include "Common.h"
#include "Matrix4.h"
#include "Math.h"

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

Matrix4f Matrix4f::CreateProjectionMatrix(float fieldOfView, float aspectRatio, float near, float far)
{
    Matrix4f result;

    float const tanHalfFovy = tan((fieldOfView / 180 * std::numbers::pi) / 2.f);

    result.m_Matrix[0][0] = 1.f / (aspectRatio * tanHalfFovy);
    result.m_Matrix[1][1] = 1.f / (tanHalfFovy);
    result.m_Matrix[2][3] = -1.f;

    result.m_Matrix[2][2] = -(far + near) / (far - near);
    result.m_Matrix[3][2] = -(2.f * far * near) / (far - near);
    result.m_Matrix[3][3] = 0.f;

    return result;
}

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

Matrix4f Matrix4f::Inversed() const
{
    // wild magic 4 inverse - http://www.geometrictools.com/Documentation/LaplaceExpansionTheorem.pdf
    // 84 multiplications
    // 66 adds/subs
    //  1 division

    const float* data = (const float*)&m_Matrix[0][0];

    // 24
    const float fA0 = data[0] * data[5] - data[1] * data[4];
    const float fA1 = data[0] * data[6] - data[2] * data[4];
    const float fA2 = data[0] * data[7] - data[3] * data[4];
    const float fA3 = data[1] * data[6] - data[2] * data[5];
    const float fA4 = data[1] * data[7] - data[3] * data[5];
    const float fA5 = data[2] * data[7] - data[3] * data[6];
    const float fB0 = data[8] * data[13] - data[9] * data[12];
    const float fB1 = data[8] * data[14] - data[10] * data[12];
    const float fB2 = data[8] * data[15] - data[11] * data[12];
    const float fB3 = data[9] * data[14] - data[10] * data[13];
    const float fB4 = data[9] * data[15] - data[11] * data[13];
    const float fB5 = data[10] * data[15] - data[11] * data[14];

    // 6
    const float det = fA0 * fB5 - fA1 * fB4 + fA2 * fB3 + fA3 * fB2 - fA4 * fB1 + fA5 * fB0;
    if (det == 0.0f)
        return {};

    const float invDet = 1.0f / det;

    // float* MyMatrixData = (float*)&m_Matrix[0][0];

    // 36 + 16
    return Matrix4f(
        (data[5] * fB5 - data[6] * fB4 + data[7] * fB3) * invDet,
        (-data[1] * fB5 + data[2] * fB4 - data[3] * fB3) * invDet,
        (data[13] * fA5 - data[14] * fA4 + data[15] * fA3) * invDet,
        (-data[9] * fA5 + data[10] * fA4 - data[11] * fA3) * invDet,
        (-data[4] * fB5 + data[6] * fB2 - data[7] * fB1) * invDet,
        (data[0] * fB5 - data[2] * fB2 + data[3] * fB1) * invDet,
        (-data[12] * fA5 + data[14] * fA2 - data[15] * fA1) * invDet,
        (data[8] * fA5 - data[10] * fA2 + data[11] * fA1) * invDet,
        (data[4] * fB4 - data[5] * fB2 + data[7] * fB0) * invDet,
        (-data[0] * fB4 + data[1] * fB2 - data[3] * fB0) * invDet,
        (data[12] * fA4 - data[13] * fA2 + data[15] * fA0) * invDet,
        (-data[8] * fA4 + data[9] * fA2 - data[11] * fA0) * invDet,
        (-data[4] * fB3 + data[5] * fB1 - data[6] * fB0) * invDet,
        (data[0] * fB3 - data[1] * fB1 + data[2] * fB0) * invDet,
        (-data[12] * fA3 + data[13] * fA1 - data[14] * fA0) * invDet,
        (data[8] * fA3 - data[9] * fA1 + data[10] * fA0) * invDet);
}


bool Matrix4f::MakeFrustumPlane(float a, float b, float c, float d, Plane& outPlane)
{
    const float	lengthSquared = a * a + b * b + c * c;
    if (lengthSquared == 0)
        return false;

    const float	invLength = 1 / sqrt(lengthSquared);
    outPlane = Plane(Vector3f{ -a * invLength, -b * invLength, -c * invLength }, -d * invLength);
    return true;
}

bool Matrix4f::GetFrustumNearPlane(Plane& outPlane) const
{
    return MakeFrustumPlane(
        m_Matrix[0][3] + m_Matrix[0][2],
        m_Matrix[1][3] + m_Matrix[1][2],
        m_Matrix[2][3] + m_Matrix[2][2],
        m_Matrix[3][3] + m_Matrix[3][2],
        outPlane
    );
}

bool Matrix4f::GetFrustumFarPlane(Plane& OutPlane) const
{
    return MakeFrustumPlane(
        m_Matrix[0][3] - m_Matrix[0][2],
        m_Matrix[1][3] - m_Matrix[1][2],
        m_Matrix[2][3] - m_Matrix[2][2],
        m_Matrix[3][3] - m_Matrix[3][2],
        OutPlane
    );
}


bool Matrix4f::GetFrustumLeftPlane(Plane& OutPlane) const
{
    return MakeFrustumPlane(
        m_Matrix[0][3] + m_Matrix[0][0],
        m_Matrix[1][3] + m_Matrix[1][0],
        m_Matrix[2][3] + m_Matrix[2][0],
        m_Matrix[3][3] + m_Matrix[3][0],
        OutPlane
    );
}


bool Matrix4f::GetFrustumRightPlane(Plane& OutPlane) const
{
    return MakeFrustumPlane(
        m_Matrix[0][3] - m_Matrix[0][0],
        m_Matrix[1][3] - m_Matrix[1][0],
        m_Matrix[2][3] - m_Matrix[2][0],
        m_Matrix[3][3] - m_Matrix[3][0],
        OutPlane
    );
}


bool Matrix4f::GetFrustumTopPlane(Plane& OutPlane) const
{
    return MakeFrustumPlane(
        m_Matrix[0][3] - m_Matrix[0][1],
        m_Matrix[1][3] - m_Matrix[1][1],
        m_Matrix[2][3] - m_Matrix[2][1],
        m_Matrix[3][3] - m_Matrix[3][1],
        OutPlane
    );
}


bool Matrix4f::GetFrustumBottomPlane(Plane& OutPlane) const
{
    return MakeFrustumPlane(
        m_Matrix[0][3] + m_Matrix[0][1],
        m_Matrix[1][3] + m_Matrix[1][1],
        m_Matrix[2][3] + m_Matrix[2][1],
        m_Matrix[3][3] + m_Matrix[3][1],
        OutPlane
    );
}