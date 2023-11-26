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
};

struct TransformedVertex
{
    Vector2f screenPosition;
    float    zValue=0;
    Vector3f normal;
    Vector3f worldPosition;
    Vector4f color;

    TransformedVertex operator*(float value)const
    {
        TransformedVertex result;
        result.screenPosition = screenPosition * value;
        result.zValue = zValue * value;
        result.normal = normal * value;
        result.worldPosition = worldPosition * value;
        result.color = color * value;
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
        return result;
    }
};

class SoftwareRenderer
{
public:
    SoftwareRenderer(int screenWidth, int screenHeight);

    void UpdateUI();
    void ClearZBuffer();
    void Render(const vector<Vertex>& vertices);
    const vector<uint32_t>& GetScreenBuffer() const;

    void SetModelMatrixx(const Matrix4f& other);
    void SetViewMatrix(const Matrix4f& other);
    void SetProjectionMatrix(const Matrix4f& other);

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

    const int           maxScale = 5;
};