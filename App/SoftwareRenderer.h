#pragma once

#include "IRenderer.h"
#include "TransformedVertex.h"
#include "Texture.h"
#include "SimpleThreadPool.h"

class SoftwareRenderer : public IRenderer
{
public:
    SoftwareRenderer(int screenWidth, int screenHeight);

    shared_ptr<ITexture> LoadTexture(const char* fileName)const override;

    void ClearScreen()override;
    void ClearZBuffer()override;
    void Render(const vector<Vertex>& vertices)override;
    const vector<uint32_t>& GetScreenBuffer() const override;

    void SetModelMatrixx(const Matrix4f& other)override;
    void SetViewMatrix(const Matrix4f& other)override;
    void SetProjectionMatrix(const Matrix4f& other)override;
    void SetTexture(shared_ptr<ITexture> texture)override;

    void SetWireFrameColor(const Vector4f& wireFrameColor)override;
    void SetDiffuseColor(const Vector4f& diffuseColor)override;
    void SetAmbientColor(const Vector4f& ambientColor)override;
    void SetLightPosition(const Vector3f& lightPosition)override;
    void SetDiffuseStrength(float diffuseStrength)override;
    void SetAmbientStrength(float ambientStrength)override;
    void SetSpecularStrength(float specularStrength)override;
    void SetShininess(float shininess)override;
    void SetThreadsCount(uint8_t threadsCount)override;
    void SetColorizeThreads(bool colorizeThreads)override;
    void SetDrawWireframe(bool Wireframe)override;
    void SetDrawBBoxes(bool drawBBoxes)override;
    void SetZWrite(bool zwrite)override;
    void SetZTest(bool ztest)override;
private:
    void DrawFilledTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int MinY, int MaxY);
    void DrawTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int MinY, int MaxY);
    void DrawTriangleBoundingBox(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int MinY, int MaxY);
    void DrawLine(const TransformedVertex& A, const TransformedVertex& B, const Vector4f& color, int MinY, int MaxY);
    void DrawLine(Vector2f A, Vector2f B, const Vector4f& color, int MinY, int MaxY);
    void PutPixel(int x, int y, uint32_t color);
    void PutPixelUnsafe(int x, int y, uint32_t color);
    void RenderLightSource();
    void UpdateMVPMatrix();
    void DoRender(const vector<Vertex>& vertices, int MinY, int MaxY, int threadID);

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
    Vector4f            m_ThreadColors[12];

    Matrix4f            m_ModelMatrix;
    Matrix4f            m_ViewMatrix;
    Matrix4f            m_ProjectionMatrix;
    Matrix4f            m_MVPMatrix;

    bool                m_DrawWireframe = false;
    bool                m_ColorizeThreads = false;
    bool                m_DrawBBoxes = false;
    bool                m_ZWrite = true;
    bool                m_ZTest = true;
    float               m_DiffuseStrength = 0.3f;
    float               m_AmbientStrength = 0.5f;
    float               m_SpecularStrength = 0.9f;
    float               m_Shininess = 32.0f;
    uint8_t             m_ThreadsCount = 0;

    shared_ptr<Texture> m_Texture;
    SimpleThreadPool    m_ThreadPool;
};