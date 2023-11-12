#include "Renderer.h"


SoftwareRenderer::SoftwareRenderer(int ScreenWidth, int ScreenHeight)
{
    m_ScreenBuffer.resize(ScreenWidth*ScreenHeight, 0);
}


void SoftwareRenderer::UpdateUI()
{
    ImGui::Begin("Settings", &m_SettingsOpen);
    ImGui::ColorEdit4("Color", &m_ImColor.x);

    ImGui::SliderFloat("Rotation", &m_Rotation, 0, 360);
    ImGui::SliderFloat("Scale", &m_Scale, 0, 5);

    ImGui::End();

    sf::Color Col = m_ImColor;
    uint32_t COL = sf::Color{ Col.a,Col.b,Col.g,Col.r }.toInteger();
    m_Color = COL;
}

Vector3f ProjToScreen(Vector3f v)
{
    Vector3f result = v;
    result.x = (v.x + 1) * 400;
    result.y = (v.y + 1) * 300;
    return result;
}

void SoftwareRenderer::Render(const vector<Vector3f>& Vertices)
{
    auto mat = Matrix4f::Identity();

    // clear screen
    std::fill(m_ScreenBuffer.begin(), m_ScreenBuffer.end(), 0xFF000000);

    //for(int i=0; i<Vertices.size(); ++i)
    //{
    //    int x = Vertices[i].x;
    //    int y = Vertices[i].y;
    //    m_ScreenBuffer[y*800+x] = m_Color;
    //}

    //Matrix4f matTranslation = Matrix4f::Translation(Vector3f(-400, -300, 0));
    //Matrix4f matScale = Matrix4f::Scale(Vector3f(m_Scale, m_Scale, m_Scale));
    //Matrix4f matRotation = Matrix4f::Rotation(Vector3f(0, 0, (m_Rotation / 180.f) * std::numbers::pi));
    //Matrix4f matTranslationReturn = Matrix4f::Translation(Vector3f(400, 300, 0));
    //Matrix4f matWorld = matTranslation * matScale * matRotation * matTranslationReturn;

    Matrix4f mvpMatrix = m_ModelMatrix * m_ViewMatrix * m_ProjectionMatrix;

    //mvpMatrix = mvpMatrix.Transposed();

    for(int i = 0; i < Vertices.size(); i += 3)
    {
        DrawTriangle(ProjToScreen(Vertices[i].Transformed(mvpMatrix)), ProjToScreen(Vertices[i + 1].Transformed(mvpMatrix)), ProjToScreen(Vertices[i + 2].Transformed(mvpMatrix)));
    }
}


const vector<uint32_t>& SoftwareRenderer::GetScreenBuffer()const
{
    return m_ScreenBuffer;
}

void SoftwareRenderer::DrawTriangle(const Vector3f& A, const Vector3f& B, const Vector3f& C)
{
    DrawLine(A,B);
    DrawLine(C,B);
    DrawLine(C,A);

    // Vector3f P1 = { 400, 300 ,0};
    // Vector3f P2 = { mTestX ,mTestY ,0};
    // DrawLine(P1, P2);
}

void SoftwareRenderer::DrawLine(const Vector3f& A, const Vector3f& B)
{
    Vector3f dir = B - A;

    // y = ax + b
    float a = (B.y - A.y) / (B.x - A.x);
    float b = B.y - a * B.x;

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
            PutPixel(x, y);
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
            PutPixel(x, y);
        }
    }
}

void SoftwareRenderer::PutPixel(int x, int y)
{
    if (x >= 800 || x <= 0 || y >= 600 || y <= 0) {
        return;
    }
   m_ScreenBuffer[y*800+x] = m_Color;
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
