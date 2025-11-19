/*
* Master’s thesis - Analysis of selected optimization techniques for a 3D software renderer
* Author: Wiktoria Marczyk
* Year: 2025
*/


#pragma once
#include "IRenderer.h"
#include "Math.h"


enum class ShaderType { Vertex, Fragment, Count };

enum class UniformType { TransformPVM, World, LightPos, LightAmbientColor, LightDiffuseColor, AmbientStrength, DiffuseStrength, SpecularStrength, Shininess, CameraPos, WireframeColor, Count };

enum class VertexAttribute { Position, Normal, Color, TexCoord, Count };

class GlTexture : public ITexture
{
public:
    GlTexture()=default;

    bool CreateWhite4x4Tex();
    bool Load(const char* fileName);
    bool Bind()const;
    virtual bool IsValid()const;
    static void Unbind();

private:
    sf::Texture m_Texture;
    bool m_Loaded = false;
};

class GlProgram
{
public:
    GlProgram()=default;
    ~GlProgram();

    bool LoadShaderFromMemory(const std::string& shaderData, ShaderType type);
    bool LoadMatrix(const Matrix4f& mat , UniformType uniform);
    bool LoadVector(const Vector3f& vec , UniformType uniform);
    bool LoadVector(const Vector4f& vec, UniformType uniform);
    bool LoadFloat(float val , UniformType uniform);
    bool Bind()const;

    static void Unbind();

private:
    uint32_t m_Program = 0;
    uint32_t m_Shader[static_cast<uint32_t>(ShaderType::Count)] = { 0 };
    int      m_Uniform[ static_cast<uint32_t>(UniformType::Count) ] = { 0 };
};

class GlVertexBuffer
{
public:
    GlVertexBuffer();
    ~GlVertexBuffer();

    bool Load(const vector<Vertex>& vertices);
    void Bind()const;

    static void Unbind();

private:
    uint32_t m_VertexBufferObject = 0;
    uint32_t m_VertexArrayObject = 0;

    uint32_t m_BufferCapacity = 0;
    uint32_t m_BufferSize = 0;
};

class GlRenderer : public IRenderer
{
public:
    GlRenderer(int screenWidth, int screenHeight);
    ~GlRenderer();

    virtual shared_ptr<ITexture> LoadTexture(const char* fileName)const override;

    virtual void ClearScreen() override;
    virtual void ClearZBuffer() override;
    virtual void BeginFrame() override;
    virtual void Render(const vector<Vertex>& vertices) override;
    virtual void EndFrame() override;
    virtual void RenderDepthBuffer()override;
    virtual const vector<uint32_t>& GetScreenBuffer() const override;
    virtual const DrawStats& GetDrawStats() const override;
    virtual shared_ptr<ITexture> GetDefaultTexture() const override;

    virtual void SetModelMatrix(const Matrix4f& other)override;
    virtual void SetViewMatrix(const Matrix4f& other)override;
    virtual void SetProjectionMatrix(const Matrix4f& other)override;
    virtual void SetTexture(shared_ptr<ITexture> texture)override;

    virtual void SetWireFrameColor(const Vector4f& wireFrameColor)override;
    virtual void SetClearColor(const Vector4f& clearColor)override;
    virtual void SetDiffuseColor(const Vector3f& diffuseColor)override;
    virtual void SetAmbientColor(const Vector3f& ambientColor)override;
    virtual void SetLightPosition(const Vector3f& lightPosition)override;
    virtual void SetDiffuseStrength(float diffuseStrength)override;
    virtual void SetAmbientStrength(float ambientStrength)override;
    virtual void SetSpecularStrength(float specularStrength)override;
    virtual void SetShininess(float shininess)override;
    virtual void SetThreadsCount(uint8_t threadsCount)override;
    virtual void SetColorizeThreads(bool colorizeThreads)override;
    virtual void SetDrawWireframe(bool wireframe)override;
    virtual void SetDrawBBoxes(bool drawBBoxes)override;
    virtual void SetZWrite(bool zwrite)override;
    virtual void SetZTest(bool ztest)override;

private:
    int                         m_ScreenWidth = 0;
    int                         m_ScreenHeight = 0;

    unique_ptr<GlProgram>       m_DefaultProgram;
    unique_ptr<GlProgram>       m_LineProgram;
    unique_ptr<GlProgram>       m_WireframeProgram;
    unique_ptr<GlVertexBuffer>  m_DefaultVertexBuffer;

    Vector4f                    m_WireFrameColor = Vector4f(1, 1, 1, 1);
    Vector3f                    m_DiffuseColor = Vector3f(1, 1, 1);
    Vector3f                    m_AmbientColor = Vector3f(1, 1, 1);
    Vector3f                    m_LightPosition = Vector3f(0, 0, -20);
    Vector3f                    m_CameraPosition = Vector3f(0, 0, 0);
    Vector4f                    m_ThreadColors[12];
    Vector4f                    m_ClearColor = Vector4f(0, 0, 0, 1);

    Matrix4f                    m_ModelMatrix;
    Matrix4f                    m_ViewMatrix;
    Matrix4f                    m_ProjectionMatrix;
    Matrix4f                    m_MVPMatrix;

    bool                        m_DrawWireframe = false;
    bool                        m_ZWrite = false;
    bool                        m_ZTest = false;
    float                       m_DiffuseStrength = 0.3f;
    float                       m_AmbientStrength = 0.5f;
    float                       m_SpecularStrength = 0.9f;
    float                       m_Shininess = 32.0f;
    int                         m_DefaultSFMLProgram = 0;

    shared_ptr<GlTexture>       m_Texture;
    shared_ptr<GlTexture>       m_DefaultTexture;
};