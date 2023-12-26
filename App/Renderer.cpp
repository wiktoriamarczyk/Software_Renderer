#include "Renderer.h"
#include "Line2D.h"
#include "../stb/stb_image.h"
#include "TransformedVertex.h"
#include "VertexInterpolator.h"


constexpr int INTSIGNBITNOTSET(int i) { return ((~((const unsigned long)(i))) >> 31); }

template< typename T >
static T LerpT(const T& a, const T& b, float alpha)
{
    float a1 = 1.0f - alpha;
    return a * a1 + b * alpha;
}

const vector<Vertex>& ClipTraingles(const Plane& ClipPlane, const float Epsilon, const vector<Vertex>& Verts)
{
    static vector<Vertex> EMPTY;
    vector<uint8_t> VertsRelation;
    vector<Vertex> SplitedVertices;
    vector<int> EdgeSplitVertex;
    static thread_local vector<Vertex> ClippedVerts;

    using eSide = Plane::eSide;

    float            fDist = 0;
    const int        OldVerticesCount = Verts.size();
    const int        OldEdgesCount = OldVerticesCount;
    const int        OldTrianglesCount = OldVerticesCount / 3;
    int              FrontBack[3] = {};
    VertsRelation.resize(OldVerticesCount);

    for (int i = 0; i < OldVerticesCount; ++i)
    {
        VertsRelation[i] = (uint8_t)ClipPlane.GetSide(Verts[i].position, Epsilon);
        FrontBack[VertsRelation[i]]++;
    }

    // all vertices behind clipping plane - clip all
    if (!FrontBack[(int)eSide::Back])
    {
        return EMPTY;
    }

    // all vertices in front of clipping plane - no clipping
    if (!FrontBack[(int)eSide::Front])
    {
        return Verts;
    }

    struct edge
    {
        uint8_t VertexIndex[2];
    };

    constexpr edge TriangleEdges[3] = {
        { 0,1 },
        { 1,2 },
        { 2,0 }
    };

    SplitedVertices.clear();
    EdgeSplitVertex.resize(OldEdgesCount);

    for (int t = 0, e = 0; t < OldTrianglesCount; ++t)
    {
        int BaseVertex = t * 3;
        for (auto& Edge : TriangleEdges)
        {
            int vi0 = BaseVertex + Edge.VertexIndex[0];
            int vi1 = BaseVertex + Edge.VertexIndex[1];

            if ((VertsRelation[vi0] ^ VertsRelation[vi1]) && !((VertsRelation[vi0] | VertsRelation[vi1]) & (uint8_t)eSide::On))
            {
                ClipPlane.LineIntersection(Verts[vi0].position, Verts[vi1].position, fDist);
                Vertex NewVertex = LerpT(Verts[vi0], Verts[vi1], fDist);

                EdgeSplitVertex[e] = (int)SplitedVertices.size();

                SplitedVertices.push_back(NewVertex);
            }
            else
            {
                // no split
                EdgeSplitVertex[e] = -1;
            }
            ++e;
        }
    }

    ClippedVerts.reserve(OldVerticesCount + SplitedVertices.size() / 2);
    ClippedVerts.clear();

    for (int E = 0; E < OldEdgesCount; E += 3)
    {
        const int EdgeSplit0 = EdgeSplitVertex[E + 0];
        const int EdgeSplit1 = EdgeSplitVertex[E + 1];
        const int EdgeSplit2 = EdgeSplitVertex[E + 2];

        const int vi0 = E + 0;
        const int vi1 = E + 1;
        const int vi2 = E + 2;

        const uint8_t val = INTSIGNBITNOTSET(EdgeSplit0) | (INTSIGNBITNOTSET(EdgeSplit1) << 1) | (INTSIGNBITNOTSET(EdgeSplit2) << 2);
        switch (val)
        {
        case 0:
            // no split

            // all vertices behind clipping plane - skip
            if ((VertsRelation[vi0] | VertsRelation[vi1] | VertsRelation[vi2]) & uint8_t(eSide::Front))
                break; // skip this triangle

            // copy all
            ClippedVerts.push_back(Verts[vi0]);
            ClippedVerts.push_back(Verts[vi1]);
            ClippedVerts.push_back(Verts[vi2]);
            break;
        case 1:
            // edge 0 slitted
            if (!(VertsRelation[vi0] & uint8_t(eSide::Front)))
            {
                ClippedVerts.push_back(Verts[vi0]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit0]);
                ClippedVerts.push_back(Verts[vi2]);
            }
            else {
                ClippedVerts.push_back(SplitedVertices[EdgeSplit0]);
                ClippedVerts.push_back(Verts[vi1]);
                ClippedVerts.push_back(Verts[vi2]);
            }
            break;
        case 2:
            if (!(VertsRelation[vi1] & uint8_t(eSide::Front)))
            {
                ClippedVerts.push_back(Verts[vi1]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit1]);
                ClippedVerts.push_back(Verts[vi0]);
            }
            else
            {
                ClippedVerts.push_back(SplitedVertices[EdgeSplit1]);
                ClippedVerts.push_back(Verts[vi2]);
                ClippedVerts.push_back(Verts[vi0]);
            }
            break;
        case 4:
            if (!(VertsRelation[vi2] & uint8_t(eSide::Front)))
            {
                ClippedVerts.push_back(Verts[vi2]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit2]);
                ClippedVerts.push_back(Verts[vi1]);
            }
            else {

                ClippedVerts.push_back(SplitedVertices[EdgeSplit2]);
                ClippedVerts.push_back(Verts[vi0]);
                ClippedVerts.push_back(Verts[vi1]);
            }
            break;
        case 3:
            // edge 0 and 1 slitted
            if (!(VertsRelation[vi1] & uint8_t(eSide::Front)))
            {
                ClippedVerts.push_back(Verts[vi1]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit1]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit0]);
            }
            else {
                ClippedVerts.push_back(Verts[vi0]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit0]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit1]);

                ClippedVerts.push_back(SplitedVertices[EdgeSplit1]);
                ClippedVerts.push_back(Verts[vi2]);
                ClippedVerts.push_back(Verts[vi0]);
            }
            break;

        case 5:
            // edge 0 and 2 slitted
            if (!(VertsRelation[vi0] & uint8_t(eSide::Front)))
            {
                ClippedVerts.push_back(Verts[vi0]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit0]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit2]);
            }
            else {

                ClippedVerts.push_back(SplitedVertices[EdgeSplit0]);
                ClippedVerts.push_back(Verts[vi1]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit2]);

                ClippedVerts.push_back(Verts[vi1]);
                ClippedVerts.push_back(Verts[vi2]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit2]);
            }
            break;
        case 6:
            // edge 1 and 2 slitted
            if (!(VertsRelation[vi2] & uint8_t(eSide::Front)))
            {
                ClippedVerts.push_back(Verts[vi2]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit2]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit1]);
            }
            else {
                ClippedVerts.push_back(SplitedVertices[EdgeSplit2]);
                ClippedVerts.push_back(Verts[vi1]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit1]);

                ClippedVerts.push_back(Verts[vi0]);
                ClippedVerts.push_back(Verts[vi1]);
                ClippedVerts.push_back(SplitedVertices[EdgeSplit2]);
            }
            break;
        }
    }
    return ClippedVerts;
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

bool Texture::IsValid()const
{
    return m_Data.size() > 0;
}

Vector4f Texture::Sample(Vector2f uv) const
{
    int x = std::clamp<int>( uv.x * (m_Width  - 1) , 0 , m_Width - 1);
    int y = std::clamp<int>( uv.y * (m_Height - 1) , 0 , m_Height - 1);

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

    const vector<Vertex>& vertices = ClipTraingles( nearFrustumPlane, 0.001f, inVertices);

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

   // RenderLightSource();
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

    Matrix4f mvpMatrix = m_ModelMatrix * m_ViewMatrix * m_ProjectionMatrix;

    for (int i = 0; i < vertices.size(); i += TRIANGLE_VERT_COUNT)
    {
        TransformedVertex transformedA;
        transformedA.ProjToScreen(vertices[i + 0], m_ModelMatrix, mvpMatrix);
        TransformedVertex transformedB;
        transformedB.ProjToScreen(vertices[i + 1], m_ModelMatrix, mvpMatrix);
        TransformedVertex transformedC;
        transformedC.ProjToScreen(vertices[i + 2], m_ModelMatrix, mvpMatrix);

        DrawTriangle(transformedA, transformedB, transformedC, m_WireFrameColor);
    }
}

const vector<uint32_t>& SoftwareRenderer::GetScreenBuffer()const
{
    return m_ScreenBuffer;
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
    for (int x = min.x; x <= max.x; ++x)
    {
        for (int y = min.y; y <= max.y; ++y)
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
                Vector3f baricentricCoordinates = Vector3f( BCP, CAP , ABP) * invABC;
                //TransformedVertex interpolatedVertex = VA * baricentricCoordinates.y + VB * baricentricCoordinates.z + VC * baricentricCoordinates.x;
                interpolator.Interpolate(baricentricCoordinates, interpolatedVertex);
                interpolatedVertex.normal.Normalize();

                float& z = m_ZBuffer[y * SCREEN_WIDTH + x];
                if (interpolatedVertex.screenPosition.z < z) {
                    z = interpolatedVertex.screenPosition.z;
                }
                else {
                    continue;
                }

                interpolatedVertex.normal.Normalize();
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







SimlpeThreadPool::SimlpeThreadPool()
    : m_NewTaskSemaphore(0)
{
    m_Finlizing = false;
}

SimlpeThreadPool::~SimlpeThreadPool()
{
    {
        std::unique_lock lock(m_TasksCS);
        if (m_ThreadCount <= 0)
            return;

        m_Finlizing = true;
        m_NewTaskSemaphore.release(m_ThreadCount);
    }

    while (m_ThreadCount)
    {
    }
}

void SimlpeThreadPool::SetThreadCount(uint8_t Count)
{
    {
        std::unique_lock lock(m_TasksCS);
        int CurCount = m_ThreadCount;
        if (Count == CurCount)
            return;

        if (Count < CurCount)
        {
            int TasksToKill = CurCount - Count;
            for (int i = 0; i < TasksToKill; ++i)
                m_Tasks.push_back({});

            m_NewTaskSemaphore.release(TasksToKill);
        }
        else
        {
            int TasksToSpawn = Count - CurCount;
            for (int i = 0; i < TasksToSpawn; ++i)
                thread([this] { Worker(); }).detach();

        }
    }

    while (m_ThreadCount != Count)
    {
    }
}

void SimlpeThreadPool::Worker()
{
    m_ThreadCount++;
    while (true)
    {
        m_NewTaskSemaphore.acquire();
        if (m_Finlizing)
            break;

        optional<Task> Task = AcquireTask();
        if (!Task || !Task->m_Func)
            break;

        Task->m_Func();
        Task->m_FinishPromise.set_value();
    }
    m_ThreadCount--;
}

void SimlpeThreadPool::LaunchTasks(vector<TaskFunc> TaskFuncs)
{
    vector<future<void>> TasksAwaiters;
    {
        std::unique_lock lock(m_TasksCS);
        if (m_ThreadCount <= 0)
            return;

        for (auto& Func : TaskFuncs)
        {
            if (!Func)
                continue;

            m_Tasks.push_back(Task{ {}, std::move(Func) });
            TasksAwaiters.push_back(m_Tasks.back().m_FinishPromise.get_future());
        }

        m_NewTaskSemaphore.release(TasksAwaiters.size());
    }

    for (auto& Awaiter : TasksAwaiters)
    {
        Awaiter.get();
    }
}


optional<SimlpeThreadPool::Task> SimlpeThreadPool::AcquireTask()
{
    std::unique_lock lock(m_TasksCS);

    optional<SimlpeThreadPool::Task> Result;
    if (m_Tasks.empty())
        return Result;

    Result = std::move(m_Tasks.front());
    m_Tasks.erase(m_Tasks.begin());
    return Result;
}