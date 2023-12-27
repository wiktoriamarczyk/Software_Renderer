#include "SoftwareRenderer.h"
#include "Line2D.h"
#include "TransformedVertex.h"
#include "VertexInterpolator.h"

// ---------------------------- SoftwareRenderer ----------------------------

SoftwareRenderer::SoftwareRenderer(int ScreenWidth, int ScreenHeight)
{
    m_ScreenBuffer.resize(ScreenWidth * ScreenHeight, 0);
    m_ZBuffer.resize(ScreenWidth * ScreenHeight, 0);

    m_ThreadColors[0]  = Vector4f(1.0f, 1.0f, 1.0f, 1.0f);
    m_ThreadColors[1]  = Vector4f(1.0f, 0.0f, 0.0f, 1.0f);
    m_ThreadColors[2]  = Vector4f(0.0f, 1.0f, 0.0f, 1.0f);
    m_ThreadColors[3]  = Vector4f(0.0f, 0.0f, 1.0f, 1.0f);
    m_ThreadColors[4]  = Vector4f(1.0f, 1.0f, 0.0f, 1.0f);
    m_ThreadColors[5]  = Vector4f(1.0f, 0.0f, 1.0f, 1.0f);
    m_ThreadColors[6]  = Vector4f(0.0f, 1.0f, 1.0f, 1.0f);
    m_ThreadColors[7]  = Vector4f(1.0f, 0.5f, 0.5f, 1.0f);
    m_ThreadColors[8]  = Vector4f(0.5f, 1.0f, 0.5f, 1.0f);
    m_ThreadColors[9]  = Vector4f(0.5f, 0.5f, 1.0f, 1.0f);
    m_ThreadColors[10] = Vector4f(1.0f, 1.0f, 0.5f, 1.0f);
    m_ThreadColors[11] = Vector4f(1.0f, 0.5f, 1.0f, 1.0f);
}

shared_ptr<ITexture> SoftwareRenderer::LoadTexture(const char* fileName) const
{
    auto texture = std::make_shared<Texture>();
    if (texture->Load(fileName))
        return texture;
    return nullptr;
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
            tasks[i] = [this,&vertices, lineStyartY, lineEndY,i]
            {
                DoRender(vertices, lineStyartY, lineEndY,i);
            };
            lineStyartY = lineEndY+1;
            lineEndY = lineStyartY+linesPerThread;
        }

        m_ThreadPool.LaunchTasks(std::move(tasks));
    }
    else
    {
        DoRender(vertices, 0, SCREEN_HEIGHT - 1,0);
    }
}

void SoftwareRenderer::DoRender(const vector<Vertex>& inVertices, int MinY, int MaxY, int threadID)
{
    Plane nearFrustumPlane;
    m_MVPMatrix.GetFrustumNearPlane(nearFrustumPlane);

    const vector<Vertex>& vertices = ClipTraingles(nearFrustumPlane, 0.001f, inVertices);

    TransformedVertex transformedA;
    TransformedVertex transformedB;
    TransformedVertex transformedC;

    if (m_DrawWireframe || m_DrawBBoxes)
    {
        const Vector4f Color = m_ColorizeThreads ? m_WireFrameColor*m_ThreadColors[threadID] : m_WireFrameColor;

        for (int i = 0; i < vertices.size(); i += TRIANGLE_VERT_COUNT)
        {
            transformedA.ProjToScreen(vertices[i + 0], m_ModelMatrix, m_MVPMatrix);
            transformedB.ProjToScreen(vertices[i + 1], m_ModelMatrix, m_MVPMatrix);
            transformedC.ProjToScreen(vertices[i + 2], m_ModelMatrix, m_MVPMatrix);

            if( m_DrawBBoxes )
            {
                DrawTriangleBoundingBox(transformedA, transformedB, transformedC,Color, MinY,MaxY);
            }
            else
            {
                DrawTriangle(transformedA, transformedB, transformedC,Color, MinY,MaxY);
            }
        }
    }
    else
    {
        const Vector4f Color = m_ColorizeThreads ? m_ThreadColors[threadID] : Vector4f(1.0f, 1.0f, 1.0f, 1.0f);

        for (int i = 0; i < vertices.size(); i += TRIANGLE_VERT_COUNT)
        {
            transformedA.ProjToScreen(vertices[i + 0], m_ModelMatrix, m_MVPMatrix);
            transformedB.ProjToScreen(vertices[i + 1], m_ModelMatrix, m_MVPMatrix);
            transformedC.ProjToScreen(vertices[i + 2], m_ModelMatrix, m_MVPMatrix);

            DrawFilledTriangle(transformedA, transformedB, transformedC,Color, MinY,MaxY);
        }
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

void SoftwareRenderer::DrawFilledTriangle(const TransformedVertex& VA, const TransformedVertex& VB, const TransformedVertex& VC, const Vector4f& color, int MinY, int MaxY)
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
                interpolatedVertex.color = interpolatedVertex.color * color;
                Vector4f finalColor = FragmentShader(interpolatedVertex);
                PutPixelUnsafe(x, y, Vector4f::ToARGB(finalColor));
            }
        }
    }
}

void SoftwareRenderer::DrawTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int MinY, int MaxY)
{
    DrawLine(A, B, color, MinY, MaxY);
    DrawLine(C, B, color, MinY, MaxY);
    DrawLine(C, A, color, MinY, MaxY);
}

void SoftwareRenderer::DrawTriangleBoundingBox(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int MinY, int MaxY)
{
    Vector2f min = A.screenPosition.CWiseMin( B.screenPosition ).CWiseMin( C.screenPosition ).xy();
    Vector2f max = A.screenPosition.CWiseMax( B.screenPosition ).CWiseMax( C.screenPosition ).xy();

    if( max.y < MinY ||
        min.y > MaxY )
        return;

    // clamp min and max points to screen size so we don't calculate points that we don't see
    min = min.CWiseMin(Vector2f(SCREEN_WIDTH-1, MaxY)).CWiseMax(Vector2f(0, MinY));
    max = max.CWiseMin(Vector2f(SCREEN_WIDTH-1, MaxY)).CWiseMax(Vector2f(0, MinY));

    DrawLine(min , Vector2f{max.x,min.y}, color, MinY, MaxY);
    DrawLine(Vector2f{max.x,min.y} , max, color, MinY, MaxY);
    DrawLine(max , Vector2f{min.x,max.y}, color, MinY, MaxY);
    DrawLine(Vector2f{min.x,max.y} , min, color, MinY, MaxY);
}

void SoftwareRenderer::DrawLine(const TransformedVertex& VA, const TransformedVertex& VB, const Vector4f& color, int MinY, int MaxY)
{
    Vector2f A = VA.screenPosition.xy();
    Vector2f B = VB.screenPosition.xy();

    return DrawLine(A, B, color, MinY,  MaxY);
}

void SoftwareRenderer::DrawLine(Vector2f A, Vector2f B, const Vector4f& color, int MinY, int MaxY)
{
    // Clip whole line against screen bounds
    if( (A.x < 0 && B.x < 0) ||
        (A.y < MinY && B.y < MinY) ||
        (A.x >= SCREEN_WIDTH  && B.x >= SCREEN_WIDTH) ||
        (A.y >= MaxY && B.y > MaxY) )
        return;

    // Handle case when start end end point are on the same pixel
    if ( int(A.x) == int(B.x) && int(A.y) == int(B.y) )
    {
        PutPixel( int(A.x) , int(A.y) , Vector4f::ToARGB(color) );
        return;
    }

    Vector2f dir = B - A;

    // Clip point A to minimum Y
    if( A.y < MinY )
    {
        float t = ( MinY - A.y ) / dir.y;
        A = A + dir * t;
    }
    // Clip point A to maximum Y
    else if( A.y > MaxY )
    {
        float t = ( MaxY - A.y ) / dir.y;
        A = A + dir * t;
    }

    // Clip point B to minimum Y
    if( B.y > MaxY )
    {
        float t = ( MaxY - A.y ) / dir.y;
        B = A + dir * t;
    }
    // Clip point B to maximum Y
    else if( B.y < MinY )
    {
        float t = ( MinY - A.y ) / dir.y;
        B = A + dir * t;
    }

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

void SoftwareRenderer::SetTexture(shared_ptr<ITexture> texture)
{
    m_Texture = dynamic_pointer_cast<Texture>(texture);
}

void SoftwareRenderer::SetWireFrameColor(const Vector4f& wireFrameColor)
{
    m_WireFrameColor = wireFrameColor;
}

void SoftwareRenderer::SetDiffuseColor(const Vector4f& diffuseColor)
{
    m_DiffuseColor = diffuseColor;
}

void SoftwareRenderer::SetAmbientColor(const Vector4f& ambientColor)
{
    m_AmbientColor = ambientColor;
}

void SoftwareRenderer::SetLightPosition(const Vector3f& lightPosition)
{
    m_LightPosition = lightPosition;
}

void SoftwareRenderer::SetDiffuseStrength(float diffuseStrength)
{
    m_DiffuseStrength = diffuseStrength;
}

void SoftwareRenderer::SetAmbientStrength(float ambientStrength)
{
    m_AmbientStrength = ambientStrength;
}

void SoftwareRenderer::SetSpecularStrength(float specularStrength)
{
    m_SpecularStrength = specularStrength;
}

void SoftwareRenderer::SetShininess(float shininess)
{
    m_Shininess = shininess;
}

void SoftwareRenderer::SetThreadsCount(uint8_t threadsCount)
{
    if (m_ThreadsCount==threadsCount)
        return;

    m_ThreadsCount = threadsCount;
    m_ThreadPool.SetThreadCount(m_ThreadsCount);
}

void SoftwareRenderer::SetColorizeThreads(bool colorizeThreads)
{
    m_ColorizeThreads = colorizeThreads;
}

void SoftwareRenderer::SetDrawWireframe(bool Wireframe)
{
    m_DrawWireframe = Wireframe;
}

void SoftwareRenderer::SetDrawBBoxes(bool drawBBoxes)
{
    m_DrawBBoxes = drawBBoxes;
}
