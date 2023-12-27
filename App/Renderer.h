#pragma once

#include "Common.h"
#include "Vector3f.h"
#include "Matrix4.h"
#include "Vector4f.h"
#include "Vector2f.h"
#include "Math.h"
#include "TransformedVertex.h"
#include "Texture.h"
#include "SimpleThreadPool.h"

class SoftwareRenderer
{
public:
    SoftwareRenderer(int screenWidth, int screenHeight);

    void ClearScreen();
    void ClearZBuffer();
    void Render(const vector<Vertex>& vertices);
    void RenderWireframe(const vector<Vertex>& vertices);
    const vector<uint32_t>& GetScreenBuffer() const;

    void SetModelMatrixx(const Matrix4f& other);
    void SetViewMatrix(const Matrix4f& other);
    void SetProjectionMatrix(const Matrix4f& other);
    void SetTexture(shared_ptr<Texture> texture);

    bool IsWireframe() const;


    void SetWireFrameColor(const Vector4f& wireFrameColor);
    void SetDiffuseColor(const Vector4f& diffuseColor);
    void SetAmbientColor(const Vector4f& ambientColor);
    void SetLightPosition(const Vector3f& lightPosition);
    void SetDiffuseStrength(float diffuseStrength);
    void SetAmbientStrength(float ambientStrength);
    void SetSpecularStrength(float specularStrength);
    void SetShininess(float shininess);
    void SetDrawWireframe(bool drawWireframe);
    void SetThreadsCount(uint8_t threadsCount);
private:
    void DrawFilledTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, int MinY, int MaxY);
    void DrawTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color);
    void DrawLine(const TransformedVertex& A, const TransformedVertex& B, const Vector4f& color);
    void PutPixel(int x, int y, uint32_t color);
    void PutPixelUnsafe(int x, int y, uint32_t color);
    void RenderLightSource();
    void UpdateMVPMatrix();
    void DoRender(const vector<Vertex>& vertices, int MinY, int MaxY);

    static float EdgeFunction(const Vector2f& A, const Vector2f& B, const Vector2f& C);
    Vector4f FragmentShader(const TransformedVertex& vertex);

    // 8 bit - one channel (8*4=32 - rgba)
    vector<uint32_t>    m_ScreenBuffer;
    vector<float>       m_ZBuffer;

    Vector4f            m_WireFrameColor = Vector4f(1, 1, 1, 1);
    Vector4f            m_DiffuseColor = Vector4f(1, 1, 1, 1);
    Vector4f            m_AmbientColor = Vector4f(1, 1, 1, 1);
    Vector3f            m_LightPosition = Vector3f(0, 0, -20);
    Vector3f            m_CameraPosition = Vector3f(0, 0, 0);

    Matrix4f            m_ModelMatrix;
    Matrix4f            m_ViewMatrix;
    Matrix4f            m_ProjectionMatrix;
    Matrix4f            m_MVPMatrix;

    bool                m_SettingsOpen = false;
    bool                m_DrawWireframe = false;
    float               m_DiffuseStrength = 0.3f;
    float               m_AmbientStrength = 0.5f;
    float               m_SpecularStrength = 0.9f;
    float               m_Shininess = 32.0f;
    uint8_t             m_ThreadsCount = 0;

    shared_ptr<Texture> m_Texture;
    SimpleThreadPool    m_ThreadPool;
};