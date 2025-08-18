/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Common.h"
#include "IRenderer.h"
#include "Math.h"

struct DrawSettings
{
    Vector3f    modelRotation = Vector3f(360 ,0,0);
    Vector3f    modelTranslation = Vector3f(0, 0, 0);//.75f);
    float       modelScale = 2.0f;//5.5;//0.3f;//;
    Vector4f    wireFrameColor = Vector4f(1, 0, 1, 1);
    Vector3f    diffuseColor = Vector3f(1, 1, 1);
    Vector3f    ambientColor = Vector3f(1, 1, 1);
    Vector3f    backgroundColor = Vector3f(0.18f, 0.24f, 0.44f);
    Vector3f    lightPosition = Vector3f(0, 0, -20);
    float       diffuseStrength = 0.7f;
    float       ambientStrength = 0.1f;
    float       specularStrength = 0.9f;
    float       shininessPower = 5; // 2 4 8 16 32
    int         threadsCount = 1;
    bool        drawWireframe = false;
    bool        drawBBoxes = false;
    bool        colorizeThreads = false;
    bool        useZBuffer = true;
    bool        renderDepthBuffer = false;
    bool        vSync = true;
    int         rendererType = 0;
    int         mathType = 4;
};

struct Model
{
    vector<Vertex> vertices;
    Vector3f Min;
    Vector3f Max;
};

struct MyModelPaths
{
    string modelPath;
    string texturePath;
};

struct RendererContext
{
    shared_ptr<IRenderer> pRenderer;
    shared_ptr<ITexture>  pModelTexture;
};

class aiScene;

class Application
{
public:
    Application()=default;

    bool Initialize();
    int Run();
private:
    static vector<Model> LoadModelVertices(const char* path);
    static vector<Model> LoadFromScene(const aiScene* pScene);
    static void NormalizeModelPosition(vector<Model>& models);
    static vector<Model> LoadFallbackModel();
    static void OpenDialog(const char* title, const char* filters, function<void()> callback);
    void OpenSceneDataDialog(MyModelPaths& selectedPaths);
    void DrawRenderingStats( int pixels );

    const uint8_t MAX_THREADS_COUNT = uint8_t(std::min<int>(16, std::thread::hardware_concurrency()));

    DrawSettings m_DrawSettings;
    RendererContext m_Contexts[2];
    MyModelPaths m_LastModelPaths;
    MyModelPaths m_ModelPaths;
    Matrix4f m_CameraMatrix;
    Matrix4f m_ProjectionMatrix;
    Matrix4f m_ModelMatrix;
    vector<Model> m_ModelsData;
    sf::RenderWindow m_MainWindow;
    sf::Texture m_ScreenTexture;
    sf::Sprite m_ScreenSprite;
    sf::Clock m_DeltaClock;
    std::chrono::steady_clock::time_point m_LastFrameTime;
    bool m_VSync = true;
};