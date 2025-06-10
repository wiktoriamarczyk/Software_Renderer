/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

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
    void BeginFrame()override;
    void Render(const vector<Vertex>& vertices)override;
    void EndFrame()override;
    void RenderDepthBuffer()override;
    const vector<uint32_t>& GetScreenBuffer() const override;
    const DrawStats& GetDrawStats() const override;
    shared_ptr<ITexture> GetDefaultTexture() const override;

    void SetModelMatrix(const Matrix4f& other)override;
    void SetViewMatrix(const Matrix4f& other)override;
    void SetProjectionMatrix(const Matrix4f& other)override;
    void SetTexture(shared_ptr<ITexture> texture)override;

    void SetWireFrameColor(const Vector4f& wireFrameColor)override;
    void SetClearColor(const Vector4f& clearColor)override;
    void SetDiffuseColor(const Vector3f& diffuseColor)override;
    void SetAmbientColor(const Vector3f& ambientColor)override;
    void SetLightPosition(const Vector3f& lightPosition)override;
    void SetDiffuseStrength(float diffuseStrength)override;
    void SetAmbientStrength(float ambientStrength)override;
    void SetSpecularStrength(float specularStrength)override;
    void SetShininess(float shininess)override;
    void SetThreadsCount(uint8_t threadsCount)override;
    void SetColorizeThreads(bool colorizeThreads)override;
    void SetDrawWireframe(bool wireframe)override;
    void SetDrawBBoxes(bool drawBBoxes)override;
    void SetZWrite(bool zWrite)override;
    void SetZTest(bool zTest)override;
    void SetMathType(eMathType mathType)override;
private:
    template< typename MathT >
    void DrawFilledTriangles(const TransformedVertex* pVerts, size_t Count, const Vector4f& color, int minY, int maxY, DrawStats& stats);
    template< typename MathT >
    void DrawFilledTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int minY, int maxY, DrawStats& stats);
    void DrawFilledTriangleWTF(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int minY, int maxY, DrawStats& stats);
    void DrawTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int MinY, int maxY);
    void DrawTriangleBoundingBox(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int MinY, int maxY);
    void DrawLine(const TransformedVertex& A, const TransformedVertex& B, const Vector4f& color, int MinY, int maxY);
    void DrawLine(Vector2f A, Vector2f B, const Vector4f& color, int MinY, int maxY);
    void PutPixel(int x, int y, uint32_t color);
    void PutPixelUnsafe(int x, int y, uint32_t color);
    void UpdateMVPMatrix();
    void DoRender(const vector<Vertex>& vertices, int MinY, int maxY, int threadID);

    static float EdgeFunction(const Vector2f& A, const Vector2f& B, const Vector2f& C);
    Vector4f FragmentShader(const TransformedVertex& vertex);

    // 8 bit - one channel (8*4=32 - rgba)
    vector<uint32_t>    m_ScreenBuffer;
    vector<float>       m_ZBuffer;

    Vector4f            m_WireFrameColor = Vector4f(1, 1, 1, 1);
    Vector3f            m_DiffuseColor = Vector3f(1, 1, 1);
    Vector3f            m_AmbientColor = Vector3f(1, 1, 1);
    Vector3f            m_LightPosition = Vector3f(0, 0, -20);
    Vector3f            m_CameraPosition = Vector3f(0, 0, 0);
    Vector4f            m_ThreadColors[12];
    uint32_t            m_ClearColor = 0xFF000000;

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

    atomic_int          m_FrameTriangles = 0;
    atomic_int          m_FrameTrianglesDrawn = 0;
    atomic_int          m_FramePixels = 0;
    atomic_int          m_FramePixelsDrawn = 0;
    atomic_int          m_FrameRasterTimeUS = 0;
    atomic_int          m_FrameTransformTimeUS = 0;
    atomic_int          m_FrameDrawTimeThreadUS = 0;
    atomic_int          m_FrameDrawTimeMainUS = 0;
    atomic_int          m_FillrateKP = 0;

    DrawStats           m_DrawStats;


    shared_ptr<Texture> m_Texture;
    shared_ptr<Texture> m_DefaultTexture;
    SimpleThreadPool    m_ThreadPool;

    int                 m_MathIndex = 0;
    MathCPU             m_MathCPU;
    MathSSE             m_MathSSE;
    MathAVX             m_MathAVX;
    const IMath*        m_MathArray[3] = { &m_MathCPU , &m_MathSSE , &m_MathAVX };
    const IMath*        m_pSelectedMath = &m_MathCPU;
};
