#pragma once

#include "Common.h"
#include "Vector3f.h"
#include "Matrix4.h"
#include "Vector4f.h"
#include "Vector2f.h"


struct Vertex
{
    Vector3f position;
    Vector3f normal;
    Vector4f color;
    Vector2f uv;
};

struct TransformedVertex
{
    Vector2f screenPosition;
    float    zValue = 0;
    Vector3f normal;
    Vector3f worldPosition;
    Vector4f color;
    Vector2f uv;

    TransformedVertex operator*(float value)const
    {
        TransformedVertex result;
        result.screenPosition = screenPosition * value;
        result.zValue = zValue * value;
        result.normal = normal * value;
        result.worldPosition = worldPosition * value;
        result.color = color * value;
        result.uv = uv * value;
        return result;
    }

    TransformedVertex operator+(const TransformedVertex& vertex)const
    {
        TransformedVertex result;
        result.screenPosition = screenPosition + vertex.screenPosition;
        result.zValue = zValue + vertex.zValue;
        result.normal = normal + vertex.normal;
        result.worldPosition = worldPosition + vertex.worldPosition;
        result.color = color + vertex.color;
        result.uv = uv + vertex.uv;
        return result;
    }

    void ProjToScreen(Vertex v, Matrix4f worldMatrix, Matrix4f mvpMatrix);
};

class Texture
{
public:
    Texture() = default;
    bool Load(const char* fileName);
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

private:
    void DrawFilledTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color);
    void DrawTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color);
    void DrawLine(const TransformedVertex& A, const TransformedVertex& B, const Vector4f& color);
    void PutPixel(int x, int y, uint32_t color);

    static float EdgeFunction(const Vector2f& A, const Vector2f& B, const Vector2f& C);
    Vector4f FragmentShader(const TransformedVertex& vertex);

    // 8 bit - one channel (8*4=32 - rgba)
    vector<uint32_t>    m_ScreenBuffer;
    vector<float>       m_ZBuffer;
    Vector4f            m_Color = Vector4f(1, 1, 1, 1);
    Matrix4f            m_ModelMatrix;
    Matrix4f            m_ViewMatrix;
    Matrix4f            m_ProjectionMatrix;
    Vector3f            m_LightPosition = Vector3f(0, 0, 15);

    bool                m_SettingsOpen = false;
    Vector3f            m_Rotation;
    Vector3f            m_Translation;
    float               m_Scale = 1;

    shared_ptr<Texture> m_Texture;

    const int           m_MaxScale = 5;
};