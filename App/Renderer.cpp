#include "Renderer.h"
#include "Line2D.h"
#include "../stb/stb_image.h"


void TransformedVertex::ProjToScreen(Vertex v, Matrix4f worldMatrix, Matrix4f mvpMatrix)
{
    worldPosition = v.position.Transformed(worldMatrix);
    normal = v.normal.Transformed(worldMatrix).Normalized();
    color = v.color;

    auto screenXYZ = Vector4f(v.position, 1.0f).Transformed(mvpMatrix);
    zValue = screenXYZ.z;
    screenPosition = screenXYZ.xy();
    screenPosition.x = (screenPosition.x + 1) * SCREEN_WIDTH / 2;
    screenPosition.y = (screenPosition.y + 1) * SCREEN_HEIGHT / 2;
    uv = v.uv;
}


bool Texture::Load(const char* fileName)
{
    int width, height, channels;

    //STB::stbi_set_flip_vertically_on_load(true);

    STB::stbi_uc* data = STB::stbi_load(fileName, &width, &height, &channels, 4);
    if (!data) {
        return false;
    }
    m_Data.resize(width * height);
    memcpy(m_Data.data(), data, width * height * 4);
    m_Width = width;
    m_Height = height;
    STB::stbi_image_free(data);
    return true;

}

Vector4f Texture::Sample(Vector2f uv) const
{
    int x = uv.x * m_Width;
    int y = uv.y * m_Height;

    int pixelIndex = y * m_Width + x;

    return Vector4f::FromARGB(m_Data[pixelIndex]);
}

// ---------------------------- SoftwareRenderer ----------------------------

SoftwareRenderer::SoftwareRenderer(int ScreenWidth, int ScreenHeight)
{
    m_ScreenBuffer.resize(ScreenWidth * ScreenHeight, 0);
    m_ZBuffer.resize(ScreenWidth * ScreenHeight, 0);
}

void SoftwareRenderer::UpdateUI()
{
    ImGui::Begin("Settings", &m_SettingsOpen);
    ImGui::ColorEdit4("Color", &m_Color.x);

    ImGui::SliderFloat3("Rotation", &m_Rotation.x, 0, FULL_ANGLE);
    //ImGui::SliderFloat3("Translation", &m_Translation.x, -5, 5);
    ImGui::SliderFloat("Scale", &m_Scale, 0, m_MaxScale);
    ImGui::SliderFloat3("Light Position", &m_LightPosition.x, -20, 20);

    ImGui::End();
}

void SoftwareRenderer::ClearScreen()
{
    // clear screen
    std::fill(m_ScreenBuffer.begin(), m_ScreenBuffer.end(), 0xFF000000);
}

void SoftwareRenderer::ClearZBuffer()
{
    std::fill(m_ZBuffer.begin(), m_ZBuffer.end(), std::numeric_limits<float>::lowest());
}

void SoftwareRenderer::Render(const vector<Vertex>& vertices)
{
    auto mat = Matrix4f::Identity();

    Matrix4f mvpMatrix = m_ModelMatrix * m_ViewMatrix * m_ProjectionMatrix;

    for(int i = 0; i < vertices.size(); i += TRIANGLE_VERT_COUNT)
    {
        TransformedVertex transformedA;
        transformedA.ProjToScreen(vertices[i + 0], m_ModelMatrix, mvpMatrix);
        TransformedVertex transformedB;
        transformedB.ProjToScreen(vertices[i + 1], m_ModelMatrix, mvpMatrix);
        TransformedVertex transformedC;
        transformedC.ProjToScreen(vertices[i + 2], m_ModelMatrix, mvpMatrix);

        DrawFilledTriangle(transformedA, transformedB, transformedC, m_Color);
    }
}

void SoftwareRenderer::RenderWireframe(const vector<Vertex>& vertices)
{
    auto mat = Matrix4f::Identity();

    Matrix4f mvpMatrix = m_ModelMatrix * m_ViewMatrix * m_ProjectionMatrix;

    for (int i = 0; i < vertices.size(); i += TRIANGLE_VERT_COUNT)
    {
        TransformedVertex transformedA;
        transformedA.ProjToScreen(vertices[i + 0], m_ModelMatrix, mvpMatrix);
        TransformedVertex transformedB;
        transformedB.ProjToScreen(vertices[i + 1], m_ModelMatrix, mvpMatrix);
        TransformedVertex transformedC;
        transformedC.ProjToScreen(vertices[i + 2], m_ModelMatrix, mvpMatrix);

        DrawTriangle(transformedA, transformedB, transformedC, m_Color);
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
    min = min.CWiseMin(Vector2f(SCREEN_WIDTH-1, SCREEN_HEIGHT-1)).CWiseMax(Vector2f(0, 0));
    max = max.CWiseMin(Vector2f(SCREEN_WIDTH-1, SCREEN_HEIGHT-1)).CWiseMax(Vector2f(0, 0));

    int maxX = max.x;
    int maxY = max.y;

    // clockwise order so we check if point is on the right side of line
    const float ABC = EdgeFunction(A, B, C);
    const float inverseABC = 1.0f / ABC;

    // if our edge function (signed area x2) is negative, it's a back facing triangle and we can cull it
    if (ABC <= 0) {
        return;
    }

    float invABC = 1.0f / ABC;


    // loop through all pixels in rectangle
    for (int x = min.x; x <= maxX; ++x)
    {
        for (int y = min.y; y <= maxY; ++y)
        {
            const Vector2f P(x, y);
            // calculate value of edge function for each line
            const float ABP = EdgeFunction(A, B, P);
            const float BCP = EdgeFunction(B, C, P);
            const float CAP = EdgeFunction(C, A, P);
            // if pixel is inside triangle, draw it
            if (ABP >= 0 && BCP >= 0 && CAP >= 0)
            {
                // dividing edge function values by ABC will give us baricentric coordinates - how much each vertex contributes to final color in point P
                Vector3f baricentricCoordinates = Vector3f(ABP, BCP, CAP) * invABC;
                //TransformedVertex interpolatedVertex = VA * baricentricCoordinates.x + VB * baricentricCoordinates.y + VC * baricentricCoordinates.z;
                TransformedVertex interpolatedVertex = VA * baricentricCoordinates.y + VB * baricentricCoordinates.z + VC * baricentricCoordinates.x;
                interpolatedVertex.normal.Normalize();

                float& z = m_ZBuffer[y * SCREEN_WIDTH + x];
                if (interpolatedVertex.zValue > z) {
                    z = interpolatedVertex.zValue;
                }
                else {
                    continue;
                }

                Vector4f finalColor = FragmentShader(interpolatedVertex);
                PutPixel(x, y, Vector4f::ToARGB(finalColor));
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

float SoftwareRenderer::EdgeFunction(const Vector2f& A, const Vector2f& B, const Vector2f& C)
{
    return (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
}

Vector4f SoftwareRenderer::FragmentShader(const TransformedVertex& vertex)
{
    Vector4f sampledPixel;
    if (m_Texture) {
        sampledPixel = m_Texture->Sample(vertex.uv);
    }
    else {
        sampledPixel = Vector4f(1, 1, 1, 1);
    }
    Vector3f pointToLightDir = (vertex.worldPosition - m_LightPosition).Normalized();
    float diffuseFactor = std::max(pointToLightDir.Dot(vertex.normal), 0.0f);
    Vector4f diffuseLight = vertex.color * diffuseFactor * sampledPixel;
    diffuseLight.w = 1.0f;

    return diffuseLight;
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

void SoftwareRenderer::SetTexture(shared_ptr<Texture> texture)
{
    m_Texture = texture;
}

Vector3f SoftwareRenderer::GetRotation() const
{
    return m_Rotation;
}

Vector3f SoftwareRenderer::GetTranslation() const
{
    return m_Translation;
}

float SoftwareRenderer::GetScale() const
{
    return m_Scale;
}
