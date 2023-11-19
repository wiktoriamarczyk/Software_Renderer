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
    ImGui::SliderFloat3("Light Position", &m_LightPosition.x, -20, 20);

    ImGui::End();
}

TransformedVertex ProjToScreen(Vertex v, Matrix4f worldMatrix, Matrix4f mvpMatrix)
{
    TransformedVertex result;

    result.worldPosition  = v.position.Transformed(worldMatrix);
    result.normal         = v.normal.Transformed(worldMatrix).Normalized();


    result.screenPosition = v.position.Transformed(mvpMatrix);
    result.screenPosition.x = (result.screenPosition.x + 1) * SCREEN_WIDTH / 2;
    result.screenPosition.y = (result.screenPosition.y + 1) * SCREEN_HEIGHT / 2;

    return result;
}

void SoftwareRenderer::Render(const vector<Vertex>& vertices)
{
    auto mat = Matrix4f::Identity();

    // clear screen
    std::fill(m_ScreenBuffer.begin(), m_ScreenBuffer.end(), 0xFF000000);

    Matrix4f mvpMatrix = m_ModelMatrix * m_ViewMatrix * m_ProjectionMatrix;

    for(int i = 0; i < vertices.size(); i += triangleVerticesCount)
    {
        TransformedVertex transformedA = ProjToScreen(vertices[i+0], m_ModelMatrix, mvpMatrix);
        TransformedVertex transformedB = ProjToScreen(vertices[i+1], m_ModelMatrix, mvpMatrix);
        TransformedVertex transformedC = ProjToScreen(vertices[i+2], m_ModelMatrix, mvpMatrix);
        DrawFilledTriangle(transformedA, transformedB, transformedC, m_Color);
    }
}

const vector<uint32_t>& SoftwareRenderer::GetScreenBuffer()const
{
    return m_ScreenBuffer;
}

void SoftwareRenderer::DrawFilledTriangle(const TransformedVertex& VA, const TransformedVertex& VB, const TransformedVertex& VC, const Vector4f& color)
{
    // filling algorithm is working that way that we are going through all pixels in rectangle that is created by min and max points
    // and we are checking if pixel is inside triangle by using three lines and checking if pixel is on the same side of each line

    Vector2f A = VA.screenPosition;
    Vector2f B = VB.screenPosition;
    Vector2f C = VC.screenPosition;

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
    Vector2f min = A.CWiseMin(B).CWiseMin(C);
    Vector2f max = A.CWiseMax(B).CWiseMax(C);

    // clamp min and max points to screen size so we don't calculate points that we don't see
    min = min.CWiseMin(Vector2f(SCREEN_WIDTH, SCREEN_HEIGHT)).CWiseMax(Vector2f(0, 0));
    max = max.CWiseMin(Vector2f(SCREEN_WIDTH, SCREEN_HEIGHT)).CWiseMax(Vector2f(0, 0));

    int maxX = max.x;
    int maxY = max.y;

    // clockwise order so we check if point is on the right side of line
    Line2D lineAB (A, B);
    Line2D lineBC (B, C);
    Line2D lineCA (C, A);


    Vector3f pointToLightDir = (VA.worldPosition - m_LightPosition).Normalized();
    float diffuseFactor = std::max(pointToLightDir.Dot(VA.normal), 0.0f);
    Vector4f diffuseLight = color * diffuseFactor;
    diffuseLight.w = 1.0f;
    uint32_t finalColor = Vector4f::ToARGB(diffuseLight);

    for (int x = min.x; x <= maxX; ++x)
    {
        for (int y = min.y; y <= maxY; ++y)
        {
            if (IsPixelInsideTriangle(lineAB, lineBC, lineCA, Vector2f(x, y)))
            {
                PutPixel(x, y, finalColor);
            }
        }
    }
}

void SoftwareRenderer::DrawTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color)
{
    DrawLine(A, B, color);
    DrawLine(C, B, color);
    DrawLine(C, A, color);
}

void SoftwareRenderer::DrawLine(const TransformedVertex& VA, const TransformedVertex& VB, const Vector4f& color)
{
    Vector2f A = VA.screenPosition;
    Vector2f B = VB.screenPosition;

    Vector2f dir = B - A;

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
