#pragma once

#include "Common.h"
#include "Vector3f.h"
#include "Matrix4.h"
#include "Vector4f.h"
#include "Vector2f.h"
#include "Math.h"
#include "TransformedVertex.h"

#include <functional>
#include <future>
#include <semaphore>


class Texture
{
public:
    Texture() = default;
    bool Load(const char* fileName);
    bool IsValid()const;
    Vector4f Sample(Vector2f uv)const;

private:
    vector<uint32_t> m_Data;
    int              m_Width = 0;
    int              m_Height = 0;

};


class SimlpeThreadPool
{
public:
    using TaskFunc = function<void()>;

    SimlpeThreadPool();
    ~SimlpeThreadPool();
    void                SetThreadCount(uint8_t Count);
    uint8_t             GetThreadCount()const { return m_ThreadCount; }
    void                LaunchTasks(vector<TaskFunc> tasks);
public:
    struct Task
    {
        promise<void>   m_FinishPromise;
        TaskFunc        m_Func;
    };
    void                Worker();
    optional<Task>      AcquireTask();

    atomic_bool         m_Finlizing;
    counting_semaphore<>m_NewTaskSemaphore;
    vector<Task>        m_Tasks;
    std::mutex          m_TasksCS;
    atomic_int          m_ThreadCount = 0;
};

class SoftwareRenderer
{
public:
    SoftwareRenderer(int screenWidth, int screenHeight);

    void UpdateUI();
    void ClearScreen();
    void ClearZBuffer();
    void Render(const vector<Vertex>& vertices);
    void RenderWireframe(const vector<Vertex>& vertices);
    const vector<uint32_t>& GetScreenBuffer() const;

    void SetModelMatrixx(const Matrix4f& other);
    void SetViewMatrix(const Matrix4f& other);
    void SetProjectionMatrix(const Matrix4f& other);
    void SetTexture(shared_ptr<Texture> texture);

    Vector3f GetRotation() const;
    Vector3f GetTranslation() const;
    float GetScale() const;
    bool IsWireframe() const;

private:
    void DrawFilledTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, int MinY, int MaxY);
    void DrawTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color);
    void DrawLine(const TransformedVertex& A, const TransformedVertex& B, const Vector4f& color);
    void PutPixel(int x, int y, uint32_t color);
    void RenderLightSource();
    void UpdateMVPMatrix();
    void DoRender(const vector<Vertex>& vertices, int MinY, int MaxY);

    static float EdgeFunction(const Vector2f& A, const Vector2f& B, const Vector2f& C);
    Vector4f FragmentShader(const TransformedVertex& vertex);

    // 8 bit - one channel (8*4=32 - rgba)
    vector<uint32_t>    m_ScreenBuffer;
    vector<float>       m_ZBuffer;

    Vector4f            m_WireFrameColor = Vector4f(1, 1, 1, 1);
    Vector4f            m_DiffuseColor = Vector4f(1, 1, 1, 1);
    Vector4f            m_AmbientColor = Vector4f(1, 1, 1, 1);
    Vector3f            m_LightPosition = Vector3f(0, 0, -20);
    Vector3f            m_Rotation;
    Vector3f            m_Translation;
    Vector3f            m_CameraPosition = Vector3f(0, 0, 0);

    Matrix4f            m_ModelMatrix;
    Matrix4f            m_ViewMatrix;
    Matrix4f            m_ProjectionMatrix;
    Matrix4f            m_MVPMatrix;

    bool                m_SettingsOpen = false;
    bool                m_Wireframe = false;
    float               m_Scale = 0.1;
    float               m_DiffuseStrength = 0.3f;
    float               m_AmbientStrength = 0.5f;
    float               m_SpecularStrength = 0.9f;
    float               m_Shininess = 32.0f;
    int                 m_ThreadsCount = 0;

    shared_ptr<Texture> m_Texture;
    SimlpeThreadPool    m_ThreadPool;

    const int           m_MaxScale = 5;
};