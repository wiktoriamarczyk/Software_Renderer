#pragma once

#include "Common.h"
#include "Vector3f.h"
#include "Matrix4.h"
#include "Vector4f.h"
#include "Vector2f.h"
#include "Math.h"
#include "TransformedVertex.h"


class Texture
{
public:
    Texture() = default;
    bool Load(const char* fileName);
    bool IsValid()const;
    Vector4f Sample(Vector2f uv)const;

private:
    vector<uint32_t> m_Data;
    int              m_Width = 0;
    int              m_Height = 0;

};

class SoftwareRenderer
{
public:
    SoftwareRenderer(int screenWidth, int screenHeight);

    void UpdateUI();
    void ClearScreen();
    void ClearZBuffer();
    void Render(const vector<Vertex>& vertices);
    void RenderWireframe(const vector<Vertex>& vertices);
    const vector<uint32_t>& GetScreenBuffer() const;

    void SetModelMatrixx(const Matrix4f& other);
    void SetViewMatrix(const Matrix4f& other);
    void SetProjectionMatrix(const Matrix4f& other);
    void SetTexture(shared_ptr<Texture> texture);

    Vector3f GetRotation() const;
    Vector3f GetTranslation() const;
    float GetScale() const;
    bool IsWireframe() const;

private:
    void DrawFilledTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C);
    void DrawTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color);
    void DrawLine(const TransformedVertex& A, const TransformedVertex& B, const Vector4f& color);
    void PutPixel(int x, int y, uint32_t color);
    void RenderLightSource();
    void UpdateMVPMatrix();

    static float EdgeFunction(const Vector2f& A, const Vector2f& B, const Vector2f& C);
    Vector4f FragmentShader(const TransformedVertex& vertex);

    // 8 bit - one channel (8*4=32 - rgba)
    vector<uint32_t>    m_ScreenBuffer;
    vector<float>       m_ZBuffer;

    Vector4f            m_WireFrameColor = Vector4f(1, 1, 1, 1);
    Vector4f            m_DiffuseColor = Vector4f(1, 1, 1, 1);
    Vector4f            m_AmbientColor = Vector4f(1, 1, 1, 1);
    Vector3f            m_LightPosition = Vector3f(0, 0, 15);
    Vector3f            m_Rotation;
    Vector3f            m_Translation;
    Vector3f            m_CameraPosition = Vector3f(0, 0, 0);

    Matrix4f            m_ModelMatrix;
    Matrix4f            m_ViewMatrix;
    Matrix4f            m_ProjectionMatrix;
    Matrix4f            m_MVPMatrix;

    bool                m_SettingsOpen = false;
    bool                m_Wireframe = false;
    float               m_Scale = 0.1;
    float               m_DiffuseStrength = 0.3f;
    float               m_AmbientStrength = 0.5f;
    float               m_SpecularStrength = 0.9f;
    float               m_Shininess = 32.0f;

    shared_ptr<Texture> m_Texture;

    const int           m_MaxScale = 5;
};