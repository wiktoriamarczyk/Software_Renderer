#pragma once

#include "Common.h"
#include "Vector3f.h"
#include "Matrix4.h"
#include "Vector4f.h"
#include "Vector2f.h"

class SoftwareRenderer
{
public:
    SoftwareRenderer(int ScreenWidth, int ScreenHeight);

    void UpdateUI();
    void Render(const vector<Vector3f>& Vetices);
    const vector<uint32_t>& GetScreenBuffer() const;

    void SetModelMatrixx(const Matrix4f& other);
    void SetViewMatrix(const Matrix4f& other);
    void SetProjectionMatrix(const Matrix4f& other);

    Vector3f GetRotation() const;
    float GetScale() const;

private:
    void DrawFilledTriangle(const Vector3f& A, const Vector3f& B, const Vector3f& C, const Vector4f& color);
    void DrawTriangle(const Vector3f& A, const Vector3f& B, const Vector3f& C, const Vector4f& color);
    void DrawLine(const Vector3f& A, const Vector3f& B, const Vector4f& color);
    void PutPixel(int x, int y, uint32_t color);

    // 8 bit - one channel (8*4=32 - rgba)
    vector<uint32_t>    m_ScreenBuffer;
    Vector4f            m_Color = Vector4f(1, 1, 1, 1);
    Matrix4f            m_ModelMatrix;
    Matrix4f            m_ViewMatrix;
    Matrix4f            m_ProjectionMatrix;

    bool                m_SettingsOpen = false;
    Vector3f            m_Rotation;
    float               m_Scale = 1;

    const int           maxScale = 5;
};