#pragma once

#include "Common.h"
#include "Vector3f.h"
#include "Matrix4.h"

class SoftwareRenderer
{
public:
    SoftwareRenderer(int ScreenWidth, int ScreenHeight);

    void UpdateUI();
    void Render(const vector<Vector3f>& Vetices);

    // setters
    void SetModelMatrixx(const Matrix4f& other);
    void SetViewMatrix(const Matrix4f& other);
    void SetProjectionMatrix(const Matrix4f& other);

    // getters
    Vector3f GetRotation() const;
    float GetScale() const;

    const vector<uint32_t>& GetScreenBuffer() const;
private:
    void DrawTriangle(const Vector3f& A, const Vector3f& B, const Vector3f& C);
    void DrawLine(const Vector3f& A, const Vector3f& B );
    void PutPixel(int x, int y);

    // 8 bit - one channel (8*4=32 - rgba)
    vector<uint32_t>    m_ScreenBuffer;
    ImVec4              m_ImColor        = ImVec4(1,0,0,1);
    Matrix4f            m_ModelMatrix;
    Matrix4f            m_ViewMatrix;
    Matrix4f            m_ProjectionMatrix;

    int                 m_Color          = 0;
    bool                m_SettingsOpen   = false;
    Vector3f            m_Rotation;
    float               m_Scale = 1;

    // constants
    const int           maxScale = 5;
};