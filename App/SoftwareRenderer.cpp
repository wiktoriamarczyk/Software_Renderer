/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2023
*/

#include "SoftwareRenderer.h"
#include "TransformedVertex.h"
#include "VertexInterpolator.h"

SoftwareRenderer::SoftwareRenderer(int screenWidth, int screenHeight)
{
    m_ScreenBuffer.resize(screenWidth * screenHeight, 0);
    m_ZBuffer.resize(screenWidth * screenHeight, 0);

    m_ThreadColors[0]  = Vector4f(1.0f, 0.0f, 1.0f, 1.0f);
    m_ThreadColors[1]  = Vector4f(1.0f, 0.0f, 0.0f, 1.0f);
    m_ThreadColors[2]  = Vector4f(0.0f, 1.0f, 0.0f, 1.0f);
    m_ThreadColors[3]  = Vector4f(0.0f, 0.0f, 1.0f, 1.0f);
    m_ThreadColors[4]  = Vector4f(1.0f, 1.0f, 0.0f, 1.0f);
    m_ThreadColors[5]  = Vector4f(0.5f, 0.0f, 1.0f, 1.0f);
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
    std::fill(m_ScreenBuffer.begin(), m_ScreenBuffer.end(), m_ClearColor);
}

void SoftwareRenderer::ClearZBuffer()
{
    std::fill(m_ZBuffer.begin(), m_ZBuffer.end(), 1.f);
}

void SoftwareRenderer::BeginFrame()
{
    m_FramePixels = 0;
    m_FrameTriangles = 0;
    m_FramePixelsDrawn = 0;
    m_FrameTrianglesDrawn = 0;
    m_FrameDrawTimeUS = 0;
    m_FillrateKP = 0;
}

void SoftwareRenderer::EndFrame()
{
    m_DrawStats.m_FramePixels         = m_FramePixels;
    m_DrawStats.m_FramePixelsDrawn    = m_FramePixelsDrawn;
    m_DrawStats.m_FrameTriangles      = m_FrameTriangles;
    m_DrawStats.m_FrameTrianglesDrawn = m_FrameTrianglesDrawn;
    m_DrawStats.m_DrawTimeUS          = m_FrameDrawTimeUS;
    m_DrawStats.m_DrawTimePerThreadUS = m_FrameDrawTimeUS / ( m_ThreadPool.GetThreadCount() ? m_ThreadPool.GetThreadCount() : 1 );
    m_DrawStats.m_FillrateKP          = m_FillrateKP;
}

void SoftwareRenderer::Render(const vector<Vertex>& vertices)
{
    ZoneScoped;
    int threadsCount = m_ThreadPool.GetThreadCount();
    if (threadsCount > 0)
    {
        int linesPerThread = SCREEN_HEIGHT / threadsCount;
        int lineStyartY = 0;
        int lineEndY = linesPerThread;

        vector<function<void()>> tasks(threadsCount);
        for (int i = 0; i < threadsCount; ++i)
        {
            if (i + 1 == threadsCount)
                lineEndY = SCREEN_HEIGHT - 1;
            tasks[i] = [this,&vertices, lineStyartY, lineEndY,i]
            {
                ZoneScopedN("Render Task");
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

void SoftwareRenderer::RenderDepthBuffer()
{
    ZoneScoped;
    for( int i= 0 ; i < m_ScreenBuffer.size() ; ++i )
    {
        uint32_t Col = std::clamp( int(255 * m_ZBuffer[i] ) , 0 , 255 );
        m_ScreenBuffer[i] = 0xFF000000 | (Col<<16) | (Col<<8) | (Col);
    }
}

void SoftwareRenderer::DoRender(const vector<Vertex>& inVertices, int minY, int maxY, int threadID)
{
    ZoneScoped;
    Plane nearFrustumPlane;
    m_MVPMatrix.GetFrustumNearPlane(nearFrustumPlane);

    const auto startTime = std::chrono::high_resolution_clock::now();

    const vector<Vertex>& vertices = ClipTriangles(nearFrustumPlane, 0.001f, inVertices);

    TransformedVertex transformedA;
    TransformedVertex transformedB;
    TransformedVertex transformedC;

    DrawStats drawStats;

    if (m_DrawWireframe || m_DrawBBoxes)
    {
        const Vector4f color = m_ColorizeThreads ? m_ThreadColors[threadID] : m_WireFrameColor;

        for (int i = 0; i < vertices.size(); i += TRIANGLE_VERT_COUNT)
        {
            transformedA.ProjToScreen(vertices[i + 0], m_ModelMatrix, m_MVPMatrix);
            transformedB.ProjToScreen(vertices[i + 1], m_ModelMatrix, m_MVPMatrix);
            transformedC.ProjToScreen(vertices[i + 2], m_ModelMatrix, m_MVPMatrix);

            if (m_DrawWireframe)
                DrawTriangle(transformedA, transformedB, transformedC, color, minY, maxY);

            if (m_DrawBBoxes)
                DrawTriangleBoundingBox(transformedA, transformedB, transformedC, color, minY, maxY);
        }
    }
    else
    {
        const Vector4f color = m_ColorizeThreads ? m_ThreadColors[threadID] : Vector4f(1.0f, 1.0f, 1.0f, 1.0f);

        for (int i = 0; i < vertices.size(); i += TRIANGLE_VERT_COUNT)
        {
            transformedA.ProjToScreen(vertices[i + 0], m_ModelMatrix, m_MVPMatrix);
            transformedB.ProjToScreen(vertices[i + 1], m_ModelMatrix, m_MVPMatrix);
            transformedC.ProjToScreen(vertices[i + 2], m_ModelMatrix, m_MVPMatrix);

            DrawFilledTriangle(transformedA, transformedB, transformedC, color, minY, maxY, drawStats);
        }
    }

    auto timeUS = std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::high_resolution_clock::now() - startTime).count();

    m_FrameTriangles        += drawStats.m_FrameTriangles;
    m_FrameTrianglesDrawn   += drawStats.m_FrameTrianglesDrawn;
    m_FramePixels           += drawStats.m_FramePixels;
    m_FramePixelsDrawn      += drawStats.m_FramePixelsDrawn;
    m_FrameDrawTimeUS       += timeUS;
    m_FillrateKP            += drawStats.m_FramePixelsDrawn * ( 1000.0f / timeUS );
}

void SoftwareRenderer::UpdateMVPMatrix()
{
    m_MVPMatrix = m_ModelMatrix * m_ViewMatrix * m_ProjectionMatrix;
}

const vector<uint32_t>& SoftwareRenderer::GetScreenBuffer()const
{
    return m_ScreenBuffer;
}

const DrawStats& SoftwareRenderer::GetDrawStats() const
{
    return m_DrawStats;
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

inline float SoftwareRenderer::EdgeFunction(const Vector2f& A, const Vector2f& B, const Vector2f& C)
{
    return (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
}

void SoftwareRenderer::DrawFilledTriangle(const TransformedVertex& VA, const TransformedVertex& VB, const TransformedVertex& VC, const Vector4f& color, int minY, int maxY, DrawStats& stats)
{
    ZoneScoped;
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
        stats.m_FrameTriangles++;
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
    min = min.CWiseMin(Vector2f(SCREEN_WIDTH-1, maxY)).CWiseMax(Vector2f(0, minY));
    max = max.CWiseMin(Vector2f(SCREEN_WIDTH-1, maxY)).CWiseMax(Vector2f(0, minY));

    const float invABC = 1.0f / ABC;

    VertexInterpolator interpolator(VA, VB, VC);
    TransformedVertex interpolatedVertex;

    int pixelsDrawn = 0;

    // loop through all pixels in rectangle
    for (int y = min.y; y <= max.y; ++y)
    {
        for (int x = min.x; x <= max.x; ++x)
        {
            const Vector2f P(x+0.5f, y+0.5f);
            // calculate value of edge function for each line
            const float ABP = EdgeFunction(A, B, P);
            if (ABP < 0)
                continue;
            const float BCP = EdgeFunction(B, C, P);
            if (BCP < 0)
                continue;
            const float CAP = EdgeFunction(C, A, P);
            if (CAP < 0)
                continue;
            // if pixel is inside triangle, draw it
            //
            // dividing edge function values by ABC will give us barycentric coordinates - how much each vertex contributes to final color in point P
            Vector3f baricentricCoordinates = Vector3f( BCP, CAP , ABP) * invABC;
            interpolator.InterpolateZ(baricentricCoordinates, interpolatedVertex);

            float& z = m_ZBuffer[y * SCREEN_WIDTH + x];
            if (interpolatedVertex.screenPosition.z < z) {
                if (m_ZWrite)
                    z = interpolatedVertex.screenPosition.z;
            }
            else if (m_ZTest){
                continue;
            }

            interpolator.InterpolateAllButZ(baricentricCoordinates, interpolatedVertex);
            interpolatedVertex.normal.Normalize();
            interpolatedVertex.color = interpolatedVertex.color * color;
            Vector4f finalColor = FragmentShader(interpolatedVertex);
            PutPixelUnsafe(x, y, Vector4f::ToARGB(finalColor));
            pixelsDrawn++;
        }
    }

    stats.m_FramePixels += (1+max.y-min.y)*(1+max.x-min.x);
    stats.m_FramePixelsDrawn += pixelsDrawn;
    stats.m_FrameTriangles++;
    stats.m_FrameTrianglesDrawn++;
}

void SoftwareRenderer::DrawTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int minY, int maxY)
{
    DrawLine(A, B, color, minY, maxY);
    DrawLine(C, B, color, minY, maxY);
    DrawLine(C, A, color, minY, maxY);
}

void SoftwareRenderer::DrawTriangleBoundingBox(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int minY, int maxY)
{
    Vector2f min = A.screenPosition.CWiseMin( B.screenPosition ).CWiseMin( C.screenPosition ).xy();
    Vector2f max = A.screenPosition.CWiseMax( B.screenPosition ).CWiseMax( C.screenPosition ).xy();

    if( max.y < minY ||
        min.y > maxY )
        return;

    // clamp min and max points to screen size so we don't calculate points that we don't see
    min = min.CWiseMin(Vector2f(SCREEN_WIDTH-1, maxY)).CWiseMax(Vector2f(0, minY));
    max = max.CWiseMin(Vector2f(SCREEN_WIDTH-1, maxY)).CWiseMax(Vector2f(0, minY));

    DrawLine(min , Vector2f{max.x,min.y}, color, minY, maxY);
    DrawLine(Vector2f{max.x,min.y} , max, color, minY, maxY);
    DrawLine(max , Vector2f{min.x,max.y}, color, minY, maxY);
    DrawLine(Vector2f{min.x,max.y} , min, color, minY, maxY);
}

void SoftwareRenderer::DrawLine(const TransformedVertex& VA, const TransformedVertex& VB, const Vector4f& color, int minY, int maxY)
{
    Vector2f A = VA.screenPosition.xy();
    Vector2f B = VB.screenPosition.xy();

    return DrawLine(A, B, color, minY,  maxY);
}

void SoftwareRenderer::DrawLine(Vector2f A, Vector2f B, const Vector4f& color, int minY, int maxY)
{
    // Clip whole line against screen bounds
    if ((A.x < 0 && B.x < 0) ||
        (A.y < minY && B.y < minY) ||
        (A.x >= SCREEN_WIDTH  && B.x >= SCREEN_WIDTH) ||
        (A.y > maxY && B.y > maxY) )
        return;

    // Handle case when start end end point are on the same pixel
    if (int(A.x) == int(B.x) && int(A.y) == int(B.y))
    {
        PutPixel( int(A.x) , int(A.y) , Vector4f::ToARGB(color) );
        return;
    }

    Vector2f dir = B - A;

    // Clip point A to minimum Y
    if (A.y < minY)
    {
        float t = ( minY - A.y ) / dir.y;
        A = A + dir * t;
    }
    // Clip point A to maximum Y
    else if (A.y > maxY)
    {
        float t = ( maxY - A.y ) / dir.y;
        A = A + dir * t;
    }

    // Clip point B to minimum Y
    if (B.y > maxY)
    {
        float t = ( maxY - A.y ) / dir.y;
        B = A + dir * t;
    }
    // Clip point B to maximum Y
    else if (B.y < minY)
    {
        float t = ( minY - A.y ) / dir.y;
        B = A + dir * t;
    }

    // y = ax + b
    float a = (B.y - A.y) / (B.x - A.x);
    float b = B.y - a * B.x;
    uint32_t intColor = Vector4f::ToARGB(color);

    if (abs(dir.x) >= abs(dir.y))
    {
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
    else
    {
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
    Vector3f ambient = m_AmbientColor * m_AmbientStrength;

    // diffuse - light that is reflected from light source
    float diffuseFactor = std::max(pointToLightDir.Dot(vertex.normal), 0.0f);
    Vector3f diffuse = m_DiffuseColor * diffuseFactor * m_DiffuseStrength;

    // specular - light that is reflected from light source and is reflected in one direction
    // specular = specularStrength * specularColor * pow(max(dot(viewDir, reflectDir), 0.0), shininess)
    Vector3f viewDir = (m_CameraPosition - vertex.worldPosition).Normalized();
    Vector3f reflectDir = (pointToLightDir * -1).Reflect(vertex.normal);
    float specularFactor = pow(max(viewDir.Dot(reflectDir), 0.0f), m_Shininess);
    Vector3f specular = m_DiffuseColor * m_SpecularStrength * specularFactor;

    // final light color = (ambient + diffuse + specular) * modelColor
    Vector3f sumOfLight = ambient + diffuse + specular;
    sumOfLight = sumOfLight.CWiseMin(Vector3f(1, 1, 1));
    Vector4f finalColor = Vector4f(sumOfLight,1.0f) * sampledPixel * vertex.color;

    return finalColor;
}

void SoftwareRenderer::SetModelMatrix(const Matrix4f& other)
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

void SoftwareRenderer::SetClearColor(const Vector4f& clearColor)
{
    m_ClearColor = Vector4f::ToARGB(clearColor);
}

void SoftwareRenderer::SetDiffuseColor(const Vector3f& diffuseColor)
{
    m_DiffuseColor = diffuseColor;
}

void SoftwareRenderer::SetAmbientColor(const Vector3f& ambientColor)
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
    if (threadsCount == 1)
        // no need to use thread pool for just 1 thread - execute work on main thread
        threadsCount = 0;

    if (m_ThreadsCount == threadsCount)
        return;

    m_ThreadsCount = threadsCount;
    m_ThreadPool.SetThreadCount(m_ThreadsCount);
}

void SoftwareRenderer::SetColorizeThreads(bool colorizeThreads)
{
    m_ColorizeThreads = colorizeThreads;
}

void SoftwareRenderer::SetDrawWireframe(bool wireframe)
{
    m_DrawWireframe = wireframe;
}

void SoftwareRenderer::SetDrawBBoxes(bool drawBBoxes)
{
    m_DrawBBoxes = drawBBoxes;
}

void SoftwareRenderer::SetZWrite(bool zwrite)
{
    m_ZWrite = zwrite;
}

void SoftwareRenderer::SetZTest(bool ztest)
{
    m_ZTest = ztest;
}