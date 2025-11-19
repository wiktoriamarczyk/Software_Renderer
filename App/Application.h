/*
* Master’s thesis - Analysis of selected optimization techniques for a 3D software renderer
* Author: Wiktoria Marczyk
* Year: 2025
*/

#pragma once
#include "Common.h"
#include "IRenderer.h"
#include "Math.h"

struct DrawSettings
{
    Vector3f    m_ModelRotation = Vector3f(360, 0, 0);
    Vector3f    m_ModelTranslation = Vector3f(0, 0, 0);
    float       m_ModelScale = 2.0f;
    Vector4f    m_WireFrameColor = Vector4f(1, 0, 1, 1);
    Vector3f    m_DiffuseColor = Vector3f(1, 1, 1);
    Vector3f    m_AmbientColor = Vector3f(1, 1, 1);
    Vector3f    m_BackgroundColor = Vector3f(0.18f, 0.24f, 0.44f);
    Vector3f    m_LightPosition = Vector3f(0, 0, -20);
    float       m_DiffuseStrength = 0.7f;
    float       m_AmbientStrength = 0.1f;
    float       m_SpecularStrength = 0.9f;
    float       m_ShininessPower = 5;
    int         m_ThreadsCount = 1;
    bool        m_DrawWireframe = false;
    bool        m_DrawBBoxes = false;
    bool        m_ColorizeThreads = false;
    bool        m_UseZBuffer = true;
    bool        m_RenderDepthBuffer = false;
    bool        m_AlphaBlend = false;
    bool        m_BackfaceCulling = true;
    bool        m_VSync = false;
    int         m_RendererType = 0;   ///< (0 - software, 1 - hardware)
    int         m_MathType = 4;       ///< (0 - CPU, 1 - CPUx8, 2 - SSEx4, 3 - SSEx8, 4 - AVXx8)
    int         m_TileMode = 0;       ///< (0 - 32x32, 1 - 16x16, 2 - 8x8)
};

struct Model
{
    vector<Vertex> m_Vertices;
    Vector3f m_Min; ///< min model coordinates
    Vector3f m_Max; ///< max model coordinates
};

struct MyModelPaths
{
    string m_ModelPath;
    string m_TexturePath;
};

struct RendererContext
{
    shared_ptr<IRenderer> m_pRenderer;
    shared_ptr<ITexture>  m_pModelTexture;
};

class aiScene;

class Application
{
public:
    Application()=default;

    void SaveStatsToFile(int pixels, int frames);
    void RunPerformanceTests();
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
    void SetupPerformanceTests();

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

    // stats
    vector<pair<int ,bool>> m_SelectedModels;
    vector<pair<int ,bool>> m_SelectedRasterModes;
    vector<pair<int ,bool>> m_SelectedTileModes;
    vector<pair<bool,bool>> m_SelectedCompressed;
    vector<pair<bool,bool>> m_SelectedMultiTC;
    vector<pair<bool,bool>> m_SelectedFragmentShader;
};