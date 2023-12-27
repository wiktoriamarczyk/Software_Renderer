#include "Renderer.h"
#include "Line2D.h"
#include "../stb/stb_image.h"
#include "TransformedVertex.h"
#include "VertexInterpolator.h"

// ---------------------------- SoftwareRenderer ----------------------------

SoftwareRenderer::SoftwareRenderer(int ScreenWidth, int ScreenHeight)
{
    m_ScreenBuffer.resize(ScreenWidth * ScreenHeight, 0);
    m_ZBuffer.resize(ScreenWidth * ScreenHeight, 0);
}

void SoftwareRenderer::UpdateUI()
{
    ImGui::Begin("Settings", &m_SettingsOpen);

    ImGui::ColorEdit4("WireFrame Color", &m_WireFrameColor.x);
    ImGui::ColorEdit4("Ambient Color", &m_AmbientColor.x);
    ImGui::ColorEdit4("Diffuse Color", &m_DiffuseColor.x);
    ImGui::SliderFloat("Ambient Strength", &m_AmbientStrength, 0, 1);
    ImGui::SliderFloat("Diffuse Strength", &m_DiffuseStrength, 0, 1);
    ImGui::SliderFloat("Specular Strength", &m_SpecularStrength, 0, 1);
    ImGui::InputFloat("Shininess", &m_Shininess, 2.f);
    ImGui::SliderFloat3("Rotation", &m_Rotation.x, 0, FULL_ANGLE);
    ImGui::SliderFloat3("Translation", &m_Translation.x, -15, 15);
    ImGui::SliderFloat("Scale", &m_Scale, 0, m_MaxScale);
    ImGui::SliderFloat3("Light Position", &m_LightPosition.x, -20, 20);
    ImGui::SliderInt("Thread Count", &m_ThreadsCount, 0, 12);
    ImGui::Checkbox("Wireframe", &m_Wireframe);

    m_ThreadPool.SetThreadCount(m_ThreadsCount);

    ImGui::End();
}

void SoftwareRenderer::ClearScreen()
{
    std::fill(m_ScreenBuffer.begin(), m_ScreenBuffer.end(), 0xFF000000);
}

void SoftwareRenderer::ClearZBuffer()
{
    std::fill(m_ZBuffer.begin(), m_ZBuffer.end(), std::numeric_limits<float>::max());
}

void SoftwareRenderer::Render(const vector<Vertex>& vertices)
{
    int threadsCount = m_ThreadPool.GetThreadCount();
    if (threadsCount>0)
    {
        int linesPerThread = SCREEN_HEIGHT / threadsCount;
        int lineStyartY = 0;
        int lineEndY = linesPerThread;

        vector<function<void()>> tasks(threadsCount);
        for (int i = 0; i < threadsCount; ++i)
        {
            if(i+1==threadsCount)
                lineEndY = SCREEN_HEIGHT - 1;
            tasks[i] = [this,&vertices, lineStyartY, lineEndY]
            {
                DoRender(vertices, lineStyartY, lineEndY);
            };
            lineStyartY += linesPerThread;
            lineEndY += linesPerThread;
        }

        m_ThreadPool.LaunchTasks(std::move(tasks));
    }
    else
    {
        DoRender(vertices, 0, SCREEN_HEIGHT - 1);
    }
}

void SoftwareRenderer::DoRender(const vector<Vertex>& inVertices, int MinY, int MaxY)
{
    Plane nearFrustumPlane;
    m_MVPMatrix.GetFrustumNearPlane(nearFrustumPlane);

    const vector<Vertex>& vertices = ClipTraingles(nearFrustumPlane, 0.001f, inVertices);

    for (int i = 0; i < vertices.size(); i += TRIANGLE_VERT_COUNT)
    {
        TransformedVertex transformedA;
        transformedA.ProjToScreen(vertices[i + 0], m_ModelMatrix, m_MVPMatrix);
        TransformedVertex transformedB;
        transformedB.ProjToScreen(vertices[i + 1], m_ModelMatrix, m_MVPMatrix);
        TransformedVertex transformedC;
        transformedC.ProjToScreen(vertices[i + 2], m_ModelMatrix, m_MVPMatrix);

        DrawFilledTriangle(transformedA, transformedB, transformedC,MinY,MaxY);
    }
}

void SoftwareRenderer::RenderLightSource()
{
    // render light source as cube

    TransformedVertex transformed;
    Vertex vertex(m_LightPosition);
    transformed.ProjToScreen(vertex, m_ModelMatrix, m_ViewMatrix*m_ProjectionMatrix);

    Vector2f lightPos = transformed.screenPosition.xy();

    float lightSize = 10;
    for (int x = lightPos.x - lightSize; x < lightPos.x + lightSize; ++x)
    {
        for (int y = lightPos.y - lightSize; y < lightPos.y + lightSize; ++y)
        {
            PutPixel(x, y, Vector4f::ToARGB(m_AmbientColor));
        }
    }
}

void SoftwareRenderer::UpdateMVPMatrix()
{
    m_MVPMatrix = m_ModelMatrix * m_ViewMatrix * m_ProjectionMatrix;
}

void SoftwareRenderer::RenderWireframe(const vector<Vertex>& vertices)
{
    auto mat = Matrix4f::Identity();

    for (int i = 0; i < vertices.size(); i += TRIANGLE_VERT_COUNT)
    {
        TransformedVertex transformedA;
        transformedA.ProjToScreen(vertices[i + 0], m_ModelMatrix, m_MVPMatrix);
        TransformedVertex transformedB;
        transformedB.ProjToScreen(vertices[i + 1], m_ModelMatrix, m_MVPMatrix);
        TransformedVertex transformedC;
        transformedC.ProjToScreen(vertices[i + 2], m_ModelMatrix, m_MVPMatrix);

        DrawTriangle(transformedA, transformedB, transformedC, m_WireFrameColor);
    }
}

const vector<uint32_t>& SoftwareRenderer::GetScreenBuffer()const
{
    return m_ScreenBuffer;
}

inline void SoftwareRenderer::PutPixelUnsafe(int x, int y, uint32_t color)
{
    m_ScreenBuffer[y * SCREEN_WIDTH + x] = color;
}

inline void SoftwareRenderer::PutPixel(int x, int y, uint32_t color)
{
    if (x >= SCREEN_WIDTH || x <= 0 || y >= SCREEN_HEIGHT || y <= 0) {
        return;
    }
    m_ScreenBuffer[y * SCREEN_WIDTH + x] = color;
}

void SoftwareRenderer::DrawFilledTriangle(const TransformedVertex& VA, const TransformedVertex& VB, const TransformedVertex& VC, int MinY, int MaxY)
{
    // filling algorithm is working that way that we are going through all pixels in rectangle that is created by min and max points
    // and we are checking if pixel is inside triangle by using three lines and checking if pixel is on the same side of each line

    Vector2f A = VA.screenPosition.xy();
    Vector2f B = VB.screenPosition.xy();
    Vector2f C = VC.screenPosition.xy();

    // clockwise order so we check if point is on the right side of line
    const float ABC = EdgeFunction(A, B, C);

    // if our edge function (signed area x2) is negative, it's a back facing triangle and we can cull it
    bool isBackFacing = ABC <= 0;
    if (isBackFacing)
    {
        return;
    }

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
    min = min.CWiseMin(Vector2f(SCREEN_WIDTH-1, MaxY)).CWiseMax(Vector2f(0, MinY));
    max = max.CWiseMin(Vector2f(SCREEN_WIDTH-1, MaxY)).CWiseMax(Vector2f(0, MinY));

    const float invABC = 1.0f / ABC;

    VertexInterpolator interpolator(VA, VB, VC);
    TransformedVertex interpolatedVertex;

    // loop through all pixels in rectangle
    for (int y = min.y; y <= max.y; ++y)
    {
        for (int x = min.x; x <= max.x; ++x)
        {
            const Vector2f P(x, y);
            // calculate value of edge function for each line
            const float ABP = EdgeFunction(A, B, P);
            const float BCP = EdgeFunction(B, C, P);
            const float CAP = EdgeFunction(C, A, P);
            // if pixel is inside triangle, draw it
            if (ABP >= 0 && BCP >= 0 && CAP >= 0)
            {
                // dividing edge function values by ABC will give us barycentric coordinates - how much each vertex contributes to final color in point P
                Vector3f baricentricCoordinates = Vector3f( BCP, CAP , ABP) * invABC;
                interpolator.InterpolateZ(baricentricCoordinates, interpolatedVertex);

                float& z = m_ZBuffer[y * SCREEN_WIDTH + x];
                if (interpolatedVertex.screenPosition.z < z) {
                    z = interpolatedVertex.screenPosition.z;
                }
                else {
                    continue;
                }

                interpolator.InterpolateAllButZ(baricentricCoordinates, interpolatedVertex);
                interpolatedVertex.normal.Normalize();
                Vector4f finalColor = FragmentShader(interpolatedVertex);
                PutPixelUnsafe(x, y, Vector4f::ToARGB(finalColor));
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
    Vector2f A = VA.screenPosition.xy();
    Vector2f B = VB.screenPosition.xy();

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

    Vector3f pointToLightDir = (m_LightPosition- vertex.worldPosition).Normalized();

    // ambient - light that is reflected from other objects

    Vector4f ambient = m_AmbientColor * m_AmbientStrength;
    // ----------------------------------------------


    // diffuse - light that is reflected from light source

    float diffuseFactor = std::max(pointToLightDir.Dot(vertex.normal), 0.0f);
    Vector4f diffuse = m_DiffuseColor * diffuseFactor * m_DiffuseStrength;
    diffuse.w = 1.0f;
    // ----------------------------------------------


    // specular - light that is reflected from light source and is reflected in one direction
    // specular = specularStrength * specularColor * pow(max(dot(viewDir, reflectDir), 0.0), shininess)

    Vector3f viewDir = (m_CameraPosition - vertex.worldPosition).Normalized();
    Vector3f reflectDir = (pointToLightDir * -1).Reflect(vertex.normal);
    float specularFactor = pow(max(viewDir.Dot(reflectDir), 0.0f), m_Shininess);
    Vector4f specular = m_DiffuseColor * m_SpecularStrength * specularFactor;

    // final light color = (ambient + diffuse + specular) * modelColor
    Vector4f sumOfLight = ambient + diffuse + specular;
    sumOfLight = sumOfLight.CWiseMin(Vector4f(1, 1, 1, 1));
    Vector4f finalColor = sumOfLight * sampledPixel * vertex.color;

    return finalColor;
}

void SoftwareRenderer::SetModelMatrixx(const Matrix4f& other)
{
    m_ModelMatrix = other;
    UpdateMVPMatrix();
}

void SoftwareRenderer::SetViewMatrix(const Matrix4f& other)
{
    m_ViewMatrix = other;
    Matrix4f inversedViewMatrix = m_ViewMatrix.Inversed();
    m_CameraPosition = Vector3f(inversedViewMatrix[12], inversedViewMatrix[13], inversedViewMatrix[14]);
    UpdateMVPMatrix();
}

void SoftwareRenderer::SetProjectionMatrix(const Matrix4f& other)
{
    m_ProjectionMatrix = other;
    UpdateMVPMatrix();
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

bool SoftwareRenderer::IsWireframe() const
{
    return m_Wireframe;
}