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

void SoftwareRenderer::Render(const vector<Vector3f>& Vertices)
{
    auto mat = Matrix4::Identity();

    // clear screen
    std::fill(m_ScreenBuffer.begin(), m_ScreenBuffer.end(), 0xFF000000);

    for(int i=0; i<Vertices.size(); ++i)
    {
        int x = Vertices[i].x;
        int y = Vertices[i].y;
        m_ScreenBuffer[y*800+x] = m_Color;
    }

    Matrix4 matTranslation = Matrix4::Translation(Vector3f(-400, -300, 0));
    Matrix4 matScale = Matrix4::Scale(Vector3f(m_Scale, m_Scale, m_Scale));
    Matrix4 matRotation = Matrix4::Rotation(Vector3f(0, 0, (m_Rotation / 180.f) * std::numbers::pi));
    Matrix4 matTranslationReturn = Matrix4::Translation(Vector3f(400, 300, 0));
    Matrix4 matWorld = matTranslation * matScale * matRotation * matTranslationReturn;

    for(int i=0; i<Vertices.size(); i += 3)
    {
        DrawTriangle(Vertices[i].Transformed(matWorld), Vertices[i + 1].Transformed(matWorld), Vertices[i + 2].Transformed(matWorld));
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
