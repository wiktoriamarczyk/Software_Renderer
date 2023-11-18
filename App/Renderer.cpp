#include "Renderer.h"
#include "Line2D.h"


static bool IsPixelInsideTriangle(const Line2D& AB, const Line2D& BC, const Line2D& CA, const Vector2f& pixel)
{
    if (AB.IsRightFromLine(pixel) && BC.IsRightFromLine(pixel) && CA.IsRightFromLine(pixel))
    {
        return true;
    }
    return false;
}

static bool IsPointInsideTheWindow(const Vector2f& point)
{
    if (point.x >= 0 && point.x <= SCREEN_WIDTH && point.y >= 0 && point.y <= SCREEN_HEIGHT)
    {
        return true;
    }
    return false;
}

// ---------------------------- SoftwareRenderer ----------------------------

SoftwareRenderer::SoftwareRenderer(int ScreenWidth, int ScreenHeight)
{
    m_ScreenBuffer.resize(ScreenWidth*ScreenHeight, 0);
}

void SoftwareRenderer::UpdateUI()
{
    ImGui::Begin("Settings", &m_SettingsOpen);
    ImGui::ColorEdit4("Color", &m_Color.x);

    ImGui::SliderFloat3("Rotation", &m_Rotation.x, 0, fullCircle);
    ImGui::SliderFloat("Scale", &m_Scale, 0, maxScale);

    ImGui::End();
}

Vector3f ProjToScreen(Vector3f v)
{
    Vector3f result = v;
    result.x = (v.x + 1) * SCREEN_WIDTH / 2;
    result.y = (v.y + 1) * SCREEN_HEIGHT / 2;
    return result;
}

void SoftwareRenderer::Render(const vector<Vector3f>& Vertices)
{
    auto mat = Matrix4f::Identity();

    // clear screen
    std::fill(m_ScreenBuffer.begin(), m_ScreenBuffer.end(), 0xFF000000);

    Matrix4f mvpMatrix = m_ModelMatrix * m_ViewMatrix * m_ProjectionMatrix;

    for(int i = 0; i < Vertices.size(); i += triangleVerticesCount)
    {
        DrawFilledTriangle(ProjToScreen(Vertices[i].Transformed(mvpMatrix)), ProjToScreen(Vertices[i + 1].Transformed(mvpMatrix)), ProjToScreen(Vertices[i + 2].Transformed(mvpMatrix)), m_Color);
    }
}

const vector<uint32_t>& SoftwareRenderer::GetScreenBuffer()const
{
    return m_ScreenBuffer;
}

void SoftwareRenderer::DrawFilledTriangle(const Vector3f& A, const Vector3f& B, const Vector3f& C, const Vector4f& color)
{
    // filling algorithm is working that way that we are going through all pixels in rectangle that is created by min and max points
    // and we are checking if pixel is inside triangle by using three lines and checking if pixel is on the same side of each line

    //            max
    //    -------B
    //   |      /|
    //   |     /*|
    //   |    /**|
    //   |   /***|
    //   |  /****|
    //   | /*****|
    //   A-------C
    // min

    // min and max points of rectangle
    Vector3f min = A.CWiseMin(B).CWiseMin(C);
    Vector3f max = A.CWiseMax(B).CWiseMax(C);

    // clamp min and max points to screen size so we don't calculate points that we don't see
    min = min.CWiseMin(Vector3f(SCREEN_WIDTH, SCREEN_HEIGHT, 0)).CWiseMax(Vector3f(0, 0, 0));
    max = max.CWiseMin(Vector3f(SCREEN_WIDTH, SCREEN_HEIGHT, 0)).CWiseMax(Vector3f(0, 0, 0));

    int maxX = max.x;
    int maxY = max.y;

    // clockwise order so we check if point is on the right side of line
    Line2D lineAB (A, B);
    Line2D lineBC (B, C);
    Line2D lineCA (C, A);

    for (int x = min.x; x <= maxX; ++x)
    {
        for (int y = min.y; y <= maxY; ++y)
        {
            if (IsPixelInsideTriangle(lineAB, lineBC, lineCA, Vector2f(x, y)))
            {
                PutPixel(x, y, Vector4f::ToARGB(color));
            }
        }
    }
}

void SoftwareRenderer::DrawTriangle(const Vector3f& A, const Vector3f& B, const Vector3f& C, const Vector4f& color)
{
    DrawLine(A, B, color);
    DrawLine(C, B, color);
    DrawLine(C, A, color);
}

void SoftwareRenderer::DrawLine(const Vector3f& A, const Vector3f& B, const Vector4f& color)
{
    Vector3f dir = B - A;

    // y = ax + b
    float a = (B.y - A.y) / (B.x - A.x);
    float b = B.y - a * B.x;
    uint32_t intColor = Vector4f::ToARGB(color);

    if (abs(dir.x) >= abs(dir.y)) {

        int startX = A.x;
        int endX = B.x;

        if (startX > endX) {
            startX = B.x;
            endX = A.x;
        }

        for (int x = startX; x < endX; ++x)
        {
            float y = a * x + b;
            PutPixel(x, y, intColor);
        }
    }
    else {

        float a = (B.x - A.x) / (B.y - A.y);
        float b = B.x - a * B.y;

        int startY = A.y;
        int endY = B.y;

        if (startY > endY) {
            startY = B.y;
            endY = A.y;
        }

        for (int y = startY; y < endY; ++y)
        {
            float x = a * y + b;
            PutPixel(x, y, intColor);
        }
    }
}

void SoftwareRenderer::PutPixel(int x, int y, uint32_t color)
{
    if (x >= SCREEN_WIDTH || x <= 0 || y >= SCREEN_HEIGHT || y <= 0) {
        return;
    }
   m_ScreenBuffer[y * SCREEN_WIDTH + x] = color;
}

void SoftwareRenderer::SetModelMatrixx(const Matrix4f& other)
{
    m_ModelMatrix = other;
}

void SoftwareRenderer::SetViewMatrix(const Matrix4f& other)
{
    m_ViewMatrix = other;
}

void SoftwareRenderer::SetProjectionMatrix(const Matrix4f& other)
{
    m_ProjectionMatrix = other;
}

Vector3f SoftwareRenderer::GetRotation() const
{
    return m_Rotation;
}

float SoftwareRenderer::GetScale() const
{
    return m_Scale;
}
