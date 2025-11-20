/*
* Master’s thesis - Analysis of selected optimization techniques for a 3D software renderer
* Author: Wiktoria Marczyk
* Year: 2025
*/

#include "Application.h"

#include <iostream>

#include "Fallback.h"
#include "DrawStatsSystem.h"
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "../ImGuiFileDialog/ImGuiFileDialog.h"


struct PredefinedModel
{
    const char* modelPath = "";
    const char* texturePath = "";
    optional<float> RotationX;
    optional<float> RotationY;
    optional<float> RotationZ;
    optional<float> Scale;
    optional<float> PositionX;
    optional<float> PositionY;
    optional<float> PositionZ;
};

const PredefinedModel s_PredefinedModels[] =
{
    {.modelPath = "x"                              , .texturePath = "../Data/Checkerboard.png"     , .RotationX = 120.f    , .RotationY = 25.f  , .Scale = 2.38f },
    {.modelPath = "../Data/teapot/Teapot.gltf"     , .texturePath = "../Data/teapot/Teapot.png"    , .RotationX = 330.f    , .RotationY = 25.f  , .Scale = 5.5f },
    {.modelPath = "../Data/Shiba/shiba.fbx"        , .texturePath = "../Data/Shiba/shiba.png"      , .RotationX = 280.f    , .RotationY = 140.f , .Scale = 4.59f },
    {.modelPath = "../Data/dog/dog.glb"            , .texturePath = "../Data/dog/dog.png"          , .RotationY = 120.f    , .Scale = 6.3f },

    {.modelPath = "x"                              , .texturePath = "../Data/Checkerboard.png"                                , .RotationY = 25.f  , .Scale = 2.9f },
    {.modelPath = "../Data/teapot/Teapot.gltf"     , .texturePath = "../Data/teapot/Teapot.png"    , .RotationX = 330.f    , .RotationY = 25.f  , .Scale = 6.837f },
    {.modelPath = "../Data/Shiba/shiba.fbx"        , .texturePath = "../Data/Shiba/shiba.png"      , .RotationX = 280.f    , .RotationY = 140.f , .Scale = 5.64f },
    {.modelPath = "../Data/dog/dog.glb"            , .texturePath = "../Data/dog/dog.png"          , .RotationY = 120.f    , .Scale = 7.72f },

    {.modelPath = "x"                              , .texturePath = "../Data/Checkerboard.png"                                , .RotationY = 25.f  , .Scale = 5.05f },
    {.modelPath = "../Data/teapot/Teapot.gltf"     , .texturePath = "../Data/teapot/Teapot.png"    , .RotationX = 330.f    , .RotationY = 25.f  , .Scale = 12.937f },
    {.modelPath = "../Data/Shiba/shiba.fbx"        , .texturePath = "../Data/Shiba/shiba.png"      , .RotationX = 280.f    , .RotationY = 140.f , .Scale = 9.23f  , .PositionX = -2.7f    , .PositionY = -1.3f },
    {.modelPath = "../Data/dog/dog.glb"            , .texturePath = "../Data/dog/dog.png"          , .RotationY = 120.f    , .Scale = 15.37f , .PositionX = -10.7f   , .PositionY = -6.45f    , .PositionZ = -0.57f },

    {.modelPath = "x"                              , .texturePath = "../Data/Checkerboard.png"     , .RotationX = 0.f      , .RotationY = 25.f  , .Scale = 6.0f },
    {.modelPath = "../Data/teapot/Teapot.gltf"     , .texturePath = "../Data/teapot/Teapot.png"    , .RotationX = 330.f    , .RotationY = 25.f  , .Scale = 16.f    , .PositionY = 1.4f },
    {.modelPath = "../Data/Shiba/shiba.fbx"        , .texturePath = "../Data/Shiba/shiba.png"      , .RotationX = 280.f    , .RotationY = 140.f , .Scale = 10.8f   , .PositionX = -3.7f    , .PositionY = -1.75f },
    {.modelPath = "../Data/dog/dog.glb"            , .texturePath = "../Data/dog/dog.png"          , .RotationY = 90.f     , .Scale = 16.0f   , .PositionY = -2.1f    , .PositionZ = -4.2f },
};

extern bool g_showTilesBoundry;
extern bool g_showTilesGrid;
extern bool g_showTilestype;
extern bool g_showTriangleBoundry;
extern bool g_ThreadPerTile;
extern bool g_MultithreadedTransformAndClip;
extern bool g_TrivialFS;
extern bool g_CompressedPartialTile;
extern std::atomic<size_t> g_memory_resource_mem;
extern std::atomic<int> g_max_overdraw;

std::atomic<bool> STATS_DUMP{ true };
std::atomic<int> THREADS{ 1 };
std::atomic<int> FRAMES_RENDERED{ 0 };
std::atomic<int> LAST_FPS{ 0 };
std::atomic<int> LAST_FPS_STD_DEV{ 0 };

std::mutex dispath_mutex;
std::vector<std::function<void()>> dispath_queue;
std::barrier s_syncBarrier(2);

auto global_start = std::chrono::high_resolution_clock::now();

static bool show_panel = false;

uint32_t miliseconds_from_app_start()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - global_start).count();

}
void dispath(std::function<void()> func)
{
    std::lock_guard<std::mutex> lock(dispath_mutex);
    dispath_queue.push_back(func);
}

void process_dispatch_queue()
{
    std::lock_guard<std::mutex> lock(dispath_mutex);
    for (auto& func : dispath_queue)
    {
        func();
    }
    dispath_queue.clear();
}

int stats_ready(uint32_t wait_start, uint32_t mintime, uint32_t maxtime)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    if (!dispath_queue.empty())
        return 0;

    auto elapsed = miliseconds_from_app_start() - wait_start;
    if (elapsed > maxtime)
    {
        if (FRAMES_RENDERED >= 480)
            return 480;
        if (FRAMES_RENDERED >= 240)
            return 240;
        if (FRAMES_RENDERED >= 120)
            return 120;
        if (FRAMES_RENDERED >= 60)
            return 60;
        if (FRAMES_RENDERED >= 30)
            return 30;
    }

    if (FRAMES_RENDERED > 480 && (elapsed > mintime && LAST_FPS_STD_DEV < LAST_FPS / 20))
        return 480;

    return 0;
}

int wait_for_stats()
{
    FRAMES_RENDERED = 0;
    auto wait_start = miliseconds_from_app_start();
    int ready = 0;

    for (; !(ready = stats_ready(wait_start, 1000, 7000));) {}

    printf("%d frames done, waited %u ms (fps:%u, std:%u)\n", ready, miliseconds_from_app_start() - wait_start, LAST_FPS.load(), LAST_FPS_STD_DEV.load());
    return ready;
}


void LoadPredefined(MyModelPaths& paths, DrawSettings& Settings, int index)
{
    span<const PredefinedModel> PredefinedArray{ s_PredefinedModels };
    if (PredefinedArray.size() <= index)
    {
        paths = {};
        return;
    }

    const PredefinedModel& model = PredefinedArray[index];

    DrawSettings Def;

    paths.m_ModelPath = model.modelPath;
    paths.m_TexturePath = model.texturePath;
    Settings.m_ModelRotation.x = model.RotationX.value_or(Def.m_ModelRotation.x);
    Settings.m_ModelRotation.y = model.RotationY.value_or(Def.m_ModelRotation.y);
    Settings.m_ModelRotation.z = model.RotationZ.value_or(Def.m_ModelRotation.z);
    Settings.m_ModelTranslation.x = model.PositionX.value_or(Def.m_ModelTranslation.x);
    Settings.m_ModelTranslation.y = model.PositionY.value_or(Def.m_ModelTranslation.y);
    Settings.m_ModelTranslation.z = model.PositionZ.value_or(Def.m_ModelTranslation.z);
    Settings.m_ModelScale = model.Scale.value_or(Def.m_ModelScale);
}



void Application::RunPerformanceTests()
{
    if (!STATS_DUMP || THREADS > MAX_THREADS_COUNT)
        return;

    STATS_DUMP = false;

    std::thread([this]
        {
            auto max = 4 * 16 * 16;
            auto it = 0;

            for (auto& multi_tc : m_SelectedMultiTC)
            {
                if (!multi_tc.second)
                    continue;
                for (auto& compressed : m_SelectedCompressed)
                {
                    if (!compressed.second)
                        continue;
                    for (auto& tile_size : m_SelectedTileModes)
                    {
                        if (!tile_size.second)
                            continue;
                        for (auto& fragment_shader : m_SelectedFragmentShader)
                        {
                            if (!fragment_shader.second)
                                continue;
                            for (auto& raster : m_SelectedRasterModes)
                            {
                                if (!raster.second)
                                    continue;
                                m_DrawSettings.m_MathType = raster.first;
                                m_DrawSettings.m_TileMode = tile_size.first;
                                for (auto& model_type : m_SelectedModels)
                                {
                                    if (!model_type.second)
                                        continue;

                                    THREADS = 1;

                                    while (THREADS <= 16)
                                    {
                                        auto start = miliseconds_from_app_start();

                                        dispath([this, model_type, fragment_shader, compressed, multi_tc]
                                            {
                                                auto time = miliseconds_from_app_start();
                                                LoadPredefined(m_ModelPaths, m_DrawSettings, model_type.first);
                                                g_TrivialFS = !fragment_shader.first;
                                                g_CompressedPartialTile = compressed.first;
                                                g_MultithreadedTransformAndClip = multi_tc.first;
                                                auto load_time = miliseconds_from_app_start() - time;
                                                if (load_time > 5)
                                                    printf("model load done in %u ms\n", load_time);
                                                s_syncBarrier.arrive_and_wait();
                                            });

                                        s_syncBarrier.arrive_and_wait();

                                        int frames = wait_for_stats();

                                        dispath([this, frames]
                                            {
                                                auto renderer = m_Contexts[m_DrawSettings.m_RendererType].m_pRenderer;
                                                SaveStatsToFile(renderer->GetPixelsDrawn(), frames);
                                                s_syncBarrier.arrive_and_wait();
                                                THREADS++;
                                            });

                                        s_syncBarrier.arrive_and_wait();

                                        auto elapsed = miliseconds_from_app_start() - start;
                                        printf("time elapsed: %u ms, %%\n\n",
                                            elapsed, THREADS.load());

                                        it++;
                                    }
                                }

                            }
                        }
                    }
                }
            }
        }).detach();
}

vector<Model> Application::LoadFromScene(const aiScene* pScene)
{
    ZoneScoped;
    if (!pScene->HasMeshes())
    {
        printf("No meshes\n");
        return vector<Model>();
    }

    int totalTriangles = 0;

    for (int i = 0; i < pScene->mNumMeshes; ++i)
    {
        if ((pScene->mMeshes[i]->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) != aiPrimitiveType_TRIANGLE)
        {
            // skip non-triangle meshes
            continue;
        }
        totalTriangles += pScene->mMeshes[i]->mNumFaces;
    }

    if (totalTriangles > MAX_MODEL_TRIANGLES)
    {
        printf("Total triangles count %d exceeds max allowed number of triangles %d\n", totalTriangles, MAX_MODEL_TRIANGLES);
        return vector<Model>();
    }

    vector<Model> result;

    for (int i = 0; i < pScene->mNumMeshes; ++i)
    {
        result.push_back(Model());

        Model& model = result.back();

        auto mesh = pScene->mMeshes[i];
        if ((mesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) != aiPrimitiveType_TRIANGLE)
        {
            // skip non-triangle meshes
            continue;
        }

        model.m_Vertices.reserve(mesh->mNumFaces * 3);

        for (int t = 0; t < mesh->mNumFaces; ++t)
        {
            const aiFace* face = &mesh->mFaces[t];
            if (face->mNumIndices != 3)
            {
                continue;
            }

            for (int i = 0; i < face->mNumIndices; i++)
            {
                Vertex v;
                v.m_Color = Vector4f(1, 1, 1, 1);

                int index = face->mIndices[i];
                if (mesh->mColors[0] != NULL)
                    v.m_Color = Vector4f(mesh->mColors[0][index].r, mesh->mColors[0][index].g, mesh->mColors[0][index].b, mesh->mColors[0][index].a);

                if (mesh->mTextureCoords[0] != NULL)
                    v.m_UV = Vector2f(mesh->mTextureCoords[0][index].x, mesh->mTextureCoords[0][index].y);


                if (mesh->mNormals != NULL)
                    v.m_Normal = Vector3f(mesh->mNormals[index].x, mesh->mNormals[index].y, mesh->mNormals[index].z);

                v.m_Position = Vector3f(mesh->mVertices[index].x, mesh->mVertices[index].y, mesh->mVertices[index].z);
                model.m_Vertices.push_back(v);
            }
        }
    }

    return result;
}

void Application::NormalizeModelPosition(vector<Model>& models)
{
    ZoneScoped;
    float maxValue = std::numeric_limits<float>::max();
    float minValue = std::numeric_limits<float>::lowest();

    Vector3f min = Vector3f(maxValue, maxValue, maxValue);
    Vector3f max = Vector3f(minValue, minValue, minValue);

    for (auto& model : models)
    {
        model.m_Min = Vector3f(maxValue, maxValue, maxValue);
        model.m_Max = Vector3f(minValue, minValue, minValue);

        for (auto& v : model.m_Vertices)
        {
            min = min.CWiseMin(v.m_Position);
            max = max.CWiseMax(v.m_Position);

            model.m_Min = model.m_Min.CWiseMin(v.m_Position);
            model.m_Max = model.m_Max.CWiseMax(v.m_Position);
        }
    }

    // center and scale model
    Vector3f center = (min + max) * 0.5f;
    const float expectedFinalDistance = 2.0f;
    const float currentDistance = (max - min).MaxComponent();
    const float currentToExpectedMultiplier = expectedFinalDistance / currentDistance;

    for (auto& model : models)
    {
        for (auto& v : model.m_Vertices)
        {
            v.m_Position = (v.m_Position - center) * currentToExpectedMultiplier;
        }
    }
}

vector<Model> Application::LoadFallbackModel()
{
    ZoneScoped;
    vector<Model> result;
    result.push_back(Model());

    vector<Vertex>& vertices = result[0].m_Vertices;
    vertices.resize(FALLBACK_MODEL_VERT_COUNT);
    std::copy(fallbackVertices, fallbackVertices + FALLBACK_MODEL_VERT_COUNT, vertices.begin());

    NormalizeModelPosition(result);
    return result;
}



vector<Model> Application::LoadModelVertices(const char* path)
{
    ZoneScoped;
    vector<Model> result;
    auto scene = aiImportFile(path, aiProcessPreset_TargetRealtime_MaxQuality | aiProcess_Triangulate);
    if (scene)
    {
        result = LoadFromScene(scene);
        aiReleaseImport(scene);

        NormalizeModelPosition(result);
    }
    else
    {
        auto error = aiGetErrorString();
        printf("Error loading model: %s\n", error);
        printf("Loading fallback model...\n");

        result = LoadFallbackModel();
    }

    return result;
}

void Application::OpenDialog(const char* title, const char* filters, function<void()> callback)
{
    string filePathName;

    if (ImGui::Button(title))
        ImGuiFileDialog::Instance()->OpenDialog(title, title, filters, ".", 1, nullptr, ImGuiFileDialogFlags_Modal);

    // display
    if (ImGuiFileDialog::Instance()->Display(title))
    {
        // action if OK
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            if (callback)
                callback();
        }

        // close
        ImGuiFileDialog::Instance()->Close();
    }
}

void Application::OpenSceneDataDialog(MyModelPaths& selectedPaths)
{
    ImGui::Begin("Settings");

    OpenDialog("Choose Model", MODEL_FORMATS, [&selectedPaths]
        {
            selectedPaths.m_ModelPath = ImGuiFileDialog::Instance()->GetFilePathName();
        });

    ImGui::SameLine(); OpenDialog("Choose Model Texture", TEXTURE_FORMATS, [&selectedPaths]
        {
            selectedPaths.m_TexturePath = ImGuiFileDialog::Instance()->GetFilePathName();
        });


    ImGui::End();
}

sf::ContextSettings GetSFMLOpenGL4_0_WindowSettings()
{
    // specify the window context settings - require OpenGL 4.0
    sf::ContextSettings windowSettings;
    windowSettings.depthBits = 24;
    windowSettings.stencilBits = 8;
    windowSettings.antialiasingLevel = 0;
    windowSettings.majorVersion = 4;
    windowSettings.minorVersion = 0;
    return windowSettings;
}

bool Application::Initialize()
{
    ZoneScoped;
    // load default model
    m_ModelsData = LoadFallbackModel();

    // create the window
    m_MainWindow.create(sf::VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT), "Software renderer", sf::Style::Default, GetSFMLOpenGL4_0_WindowSettings());
    m_MainWindow.setActive(true);

    // create renderers
    m_Contexts[0].m_pRenderer = RendererFactory::CreateRenderer(eRendererType::Software, SCREEN_WIDTH, SCREEN_HEIGHT);
    m_Contexts[1].m_pRenderer = RendererFactory::CreateRenderer(eRendererType::Hardware, SCREEN_WIDTH, SCREEN_HEIGHT);

    // load default textures
    m_Contexts[0].m_pModelTexture = m_Contexts[0].m_pRenderer->LoadTexture(INIT_TEXTURE_PATH.c_str());
    m_Contexts[1].m_pModelTexture = m_Contexts[1].m_pRenderer->LoadTexture(INIT_TEXTURE_PATH.c_str());

    m_ModelPaths.m_TexturePath = INIT_TEXTURE_PATH;
    m_LastModelPaths = m_ModelPaths;

    // initialize ImGui
    ImGui::SFML::Init(m_MainWindow);

    // create screen texture and sprite
    m_ScreenTexture.create(SCREEN_WIDTH, SCREEN_HEIGHT);
    m_ScreenSprite.setTexture(m_ScreenTexture);
    // flip the sprite vertically
    m_ScreenSprite.setTextureRect(sf::IntRect(0, SCREEN_HEIGHT, SCREEN_WIDTH, -SCREEN_HEIGHT));

    // set initial matrices
    m_CameraMatrix = Matrix4f::CreateLookAtMatrix(Vector3f(0, 0, -10), Vector3f(0, 0, 0), Vector3f(0, 1, 0));
    m_ProjectionMatrix = Matrix4f::CreateProjectionMatrix(60, (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT, 0.8f, 40.0f);
    m_ModelMatrix = Matrix4f::Identity();

    m_LastFrameTime = std::chrono::high_resolution_clock::now();

    m_SelectedModels.resize(16);
    m_SelectedRasterModes.resize(5);
    m_SelectedTileModes.resize(3);

    m_SelectedCompressed.resize(2);
    m_SelectedMultiTC.resize(2);
    m_SelectedFragmentShader.resize(2);

    m_SelectedModels[0] = { 0 , true };
    for (int i = 1; i < m_SelectedModels.size(); ++i)
        m_SelectedModels[(i % 4) * 4 + i / 4] = { i , false };

    m_SelectedRasterModes[0] = { 0 , true };
    for (int i = 1; i < m_SelectedRasterModes.size(); ++i)
        m_SelectedRasterModes[i] = { i , false };

    m_SelectedTileModes[0] = { 0 , true };
    for (int i = 1; i < m_SelectedTileModes.size(); ++i)
        m_SelectedTileModes[i] = { i , false };

    for (int i = 0; i < m_SelectedCompressed.size(); ++i)
    {
        m_SelectedCompressed[i] = { !i , !i };
        m_SelectedMultiTC[i] = { !i , !i };
        m_SelectedFragmentShader[i] = { !i , !i };
    }

    sf::Image icon;
    icon.loadFromFile("../Data/icon.png");
    m_MainWindow.setIcon(icon.getSize().x, icon.getSize().y, icon.getPixelsPtr());

    return true;
}

int Application::Run()
{
    struct Frustum
    {
    public:
        static Frustum FromMatrix(const Matrix4f& m)
        {
            Frustum F;
            m.GetFrustumNearPlane(F.m_Planes[0]);
            m.GetFrustumFarPlane(F.m_Planes[1]);
            m.GetFrustumLeftPlane(F.m_Planes[2]);
            m.GetFrustumRightPlane(F.m_Planes[3]);
            m.GetFrustumTopPlane(F.m_Planes[4]);
            m.GetFrustumBottomPlane(F.m_Planes[5]);
            return F;
        }
        bool IsInside(const Model& model) const
        {
            Vector3f Points[8];
            Points[0] = Vector3f(model.m_Min.x, model.m_Min.y, model.m_Min.z);
            Points[1] = Vector3f(model.m_Max.x, model.m_Min.y, model.m_Min.z);
            Points[2] = Vector3f(model.m_Min.x, model.m_Min.y, model.m_Max.z);
            Points[3] = Vector3f(model.m_Max.x, model.m_Min.y, model.m_Max.z);
            Points[4] = Vector3f(model.m_Min.x, model.m_Max.y, model.m_Min.z);
            Points[5] = Vector3f(model.m_Max.x, model.m_Max.y, model.m_Min.z);
            Points[6] = Vector3f(model.m_Min.x, model.m_Max.y, model.m_Max.z);
            Points[7] = Vector3f(model.m_Max.x, model.m_Max.y, model.m_Max.z);

            auto AllPointsInFront = [&Points](const Plane& plane)
                {
                    for (auto& point : Points)
                    {
                        if (plane.GetSide(point) != Plane::eSide::Front)
                            return false;
                    }
                    return true;
                };

            for (auto& plane : m_Planes)
            {
                if (AllPointsInFront(plane))
                    return false;
            }
            return true;
        }
        Plane m_Planes[6];
    };


    ZoneScoped;
    // run the program as long as the window is open
    while (m_MainWindow.isOpen())
    {
        if (THREADS != m_DrawSettings.m_ThreadsCount && THREADS > 0 && THREADS < 17)
            m_DrawSettings.m_ThreadsCount = THREADS;

        FrameMark;
        ZoneScopedN("Main Loop");

        const auto now = std::chrono::high_resolution_clock::now();
        const auto durationUS = std::chrono::duration_cast<std::chrono::microseconds>(now - m_LastFrameTime).count();
        const int fps = int(durationUS ? 1000'000.0f / durationUS : 0.0f);
        FRAMES_RENDERED++;
        m_LastFrameTime = now;

        auto renderer = m_Contexts[m_DrawSettings.m_RendererType].m_pRenderer;
        auto modelTexture = m_Contexts[m_DrawSettings.m_RendererType].m_pModelTexture;

        // check all the window's events that were triggered since the last iteration of the loop
        {
            ZoneScopedN("pollEvents");
            sf::Event event;
            while (m_MainWindow.pollEvent(event))
            {
                ImGui::SFML::ProcessEvent(m_MainWindow, event);

                // "close requested" event: we close the window
                if (event.type == sf::Event::Closed)
                    m_MainWindow.close();

                if (event.type == sf::Event::KeyPressed)
                {
                    // space pressed
                    if (event.key.code == sf::Keyboard::Space)
                        m_DrawSettings.m_RendererType = (m_DrawSettings.m_RendererType + 1) % 2;
                }
            }
        }

        process_dispatch_queue();

        // load model and texture if user selected them
        if (m_LastModelPaths.m_ModelPath != m_ModelPaths.m_ModelPath)
        {
            m_ModelsData = LoadModelVertices(m_ModelPaths.m_ModelPath.c_str());
            m_LastModelPaths.m_ModelPath = m_ModelPaths.m_ModelPath;
            if (m_LastModelPaths.m_TexturePath == m_ModelPaths.m_TexturePath)
                m_ModelPaths.m_TexturePath = "";
        }

        if (m_LastModelPaths.m_TexturePath != m_ModelPaths.m_TexturePath)
        {
            m_Contexts[0].m_pModelTexture = m_Contexts[0].m_pRenderer->LoadTexture(m_ModelPaths.m_TexturePath.c_str());
            m_Contexts[1].m_pModelTexture = m_Contexts[1].m_pRenderer->LoadTexture(m_ModelPaths.m_TexturePath.c_str());
            m_LastModelPaths.m_TexturePath = m_ModelPaths.m_TexturePath;
        }

        m_ModelMatrix = Matrix4f::Rotation(m_DrawSettings.m_ModelRotation / 180.f * PI) * Matrix4f::Scale(Vector3f(m_DrawSettings.m_ModelScale, m_DrawSettings.m_ModelScale, m_DrawSettings.m_ModelScale)) * Matrix4f::Translation(m_DrawSettings.m_ModelTranslation);

        renderer->SetTexture(modelTexture);
        renderer->SetModelMatrix(m_ModelMatrix);
        renderer->SetViewMatrix(m_CameraMatrix);
        renderer->SetProjectionMatrix(m_ProjectionMatrix);

        // update UI
        ImGui::SFML::Update(m_MainWindow, m_DeltaClock.restart());

        // render settings window
        ImGui::Begin("Settings");

        //static float drag_speeed = 0.01f;
        //if( sf::Keyboard::isKeyPressed(sf::Keyboard::LShift) || sf::Keyboard::isKeyPressed(sf::Keyboard::RShift) )
        //    drag_speeed = 0.001f;

        float drag_speeed = 0.01f;

        ImGui::SliderFloat("Ambient Strength", &m_DrawSettings.m_AmbientStrength, 0, 1);
        ImGui::SliderFloat("Diffuse Strength", &m_DrawSettings.m_DiffuseStrength, 0, 1);
        ImGui::SliderFloat("Specular Strength", &m_DrawSettings.m_SpecularStrength, 0, 1);
        ImGui::SliderFloat("Shininess Power", &m_DrawSettings.m_ShininessPower, 1.f, 10.0f);
        ImGui::SliderFloat3("Rotation", &m_DrawSettings.m_ModelRotation.x, 0, FULL_ANGLE);
        ImGui::SliderFloat3("Translation", &m_DrawSettings.m_ModelTranslation.x, -12, 12);
        ImGui::SliderFloat("Scale", &m_DrawSettings.m_ModelScale, 0, 16);
        ImGui::SliderFloat3("Light Position", &m_DrawSettings.m_LightPosition.x, -20, 20);

        if (ImGui::SliderInt("Thread Count", &m_DrawSettings.m_ThreadsCount, 1, MAX_THREADS_COUNT))
        {
            THREADS = m_DrawSettings.m_ThreadsCount;
        }

        ImGui::Combo("Renderer Type", &m_DrawSettings.m_RendererType, "Software\0Hardware\0");

        ImGui::Checkbox("Wireframe", &m_DrawSettings.m_DrawWireframe);
        ImGui::SameLine(); ImGui::Checkbox("BBoxes", &m_DrawSettings.m_DrawBBoxes);
        ImGui::SameLine(); ImGui::Checkbox("Show Tiles Grid", &g_showTilesGrid);
        ImGui::SameLine(); ImGui::Checkbox("Visualize ZBuffer", &m_DrawSettings.m_RenderDepthBuffer);

        //ImGui::SameLine(); ImGui::Checkbox("Alpha Blend" , &m_DrawSettings.m_AlphaBlend);

        ImGui::Checkbox("Trivial FS", &g_TrivialFS);
        ImGui::SameLine(); ImGui::Checkbox("Colorize Threads", &m_DrawSettings.m_ColorizeThreads);
        ImGui::SameLine(); ImGui::Checkbox("Threaded T&C", &g_MultithreadedTransformAndClip);
        ImGui::SameLine(); ImGui::Checkbox("Vertical Sync", &m_DrawSettings.m_VSync);

        ImGui::Checkbox("Compressed Partial Tile", &g_CompressedPartialTile);
        ImGui::SameLine(); ImGui::Checkbox("Use ZBuffer", &m_DrawSettings.m_UseZBuffer);
        ImGui::SameLine(); ImGui::Checkbox("One Thread Per Tile", &g_ThreadPerTile);

        //ImGui::SameLine(); ImGui::Checkbox("Backface Culling", &m_DrawSettings.m_BackfaceCulling);

        if (ImGui::Button("0A")) LoadPredefined(m_ModelPaths, m_DrawSettings, 0); ImGui::SameLine();
        if (ImGui::Button("1A")) LoadPredefined(m_ModelPaths, m_DrawSettings, 1); ImGui::SameLine();
        if (ImGui::Button("2A")) LoadPredefined(m_ModelPaths, m_DrawSettings, 2); ImGui::SameLine();
        if (ImGui::Button("3A")) LoadPredefined(m_ModelPaths, m_DrawSettings, 3);

        if (ImGui::Button("0B")) LoadPredefined(m_ModelPaths, m_DrawSettings, 4); ImGui::SameLine();
        if (ImGui::Button("1B")) LoadPredefined(m_ModelPaths, m_DrawSettings, 5); ImGui::SameLine();
        if (ImGui::Button("2B")) LoadPredefined(m_ModelPaths, m_DrawSettings, 6); ImGui::SameLine();
        if (ImGui::Button("3B")) LoadPredefined(m_ModelPaths, m_DrawSettings, 7);

        if (ImGui::Button("0C")) LoadPredefined(m_ModelPaths, m_DrawSettings, 8); ImGui::SameLine();
        if (ImGui::Button("1C")) LoadPredefined(m_ModelPaths, m_DrawSettings, 9); ImGui::SameLine();
        if (ImGui::Button("2C")) LoadPredefined(m_ModelPaths, m_DrawSettings, 10); ImGui::SameLine();
        if (ImGui::Button("3C")) LoadPredefined(m_ModelPaths, m_DrawSettings, 11);

        if (ImGui::Button("0D")) LoadPredefined(m_ModelPaths, m_DrawSettings, 12); ImGui::SameLine();
        if (ImGui::Button("1D")) LoadPredefined(m_ModelPaths, m_DrawSettings, 13); ImGui::SameLine();
        if (ImGui::Button("2D")) LoadPredefined(m_ModelPaths, m_DrawSettings, 14); ImGui::SameLine();
        if (ImGui::Button("3D")) LoadPredefined(m_ModelPaths, m_DrawSettings, 15);

        if (ImGui::Button("Stats panel"))
        {
            show_panel = !show_panel;
        }

        if (show_panel)
        {
            ImGui::Begin("Stats panel", &show_panel);
            SetupPerformanceTests();
            ImGui::End();
        }

        ImGui::ColorEdit3("Ambient Color", &m_DrawSettings.m_AmbientColor.x);
        ImGui::ColorEdit3("Diffuse Color", &m_DrawSettings.m_DiffuseColor.x);
        ImGui::ColorEdit3("Background Color", &m_DrawSettings.m_BackgroundColor.x);
        ImGui::Combo("Math Type", &m_DrawSettings.m_MathType, "CPU\0CPUx8\0SSEx4\0SSEx8\0AVXx8\0");
        ImGui::Combo("Tile Size", &m_DrawSettings.m_TileMode, "32x32\0""16x16\08x8\0");

        ImGui::End();

        if (m_VSync != m_DrawSettings.m_VSync)
        {
            m_VSync = m_DrawSettings.m_VSync;
            m_MainWindow.setFramerateLimit(m_VSync ? 60 : 0);
        }

        // set render params
        renderer->SetWireFrameColor(m_DrawSettings.m_WireFrameColor);
        renderer->SetDiffuseColor(m_DrawSettings.m_DiffuseColor);
        renderer->SetAmbientColor(m_DrawSettings.m_AmbientColor);
        renderer->SetLightPosition(m_DrawSettings.m_LightPosition);
        renderer->SetDiffuseStrength(m_DrawSettings.m_DiffuseStrength);
        renderer->SetAmbientStrength(m_DrawSettings.m_AmbientStrength);
        renderer->SetSpecularStrength(m_DrawSettings.m_SpecularStrength);
        renderer->SetShininess(pow(2.0f, m_DrawSettings.m_ShininessPower));
        renderer->SetThreadsCount(m_DrawSettings.m_ThreadsCount);
        renderer->SetDrawWireframe(m_DrawSettings.m_DrawWireframe);
        renderer->SetColorizeThreads(m_DrawSettings.m_ColorizeThreads);
        renderer->SetDrawBBoxes(m_DrawSettings.m_DrawBBoxes);
        renderer->SetZTest(m_DrawSettings.m_UseZBuffer);
        renderer->SetZWrite(m_DrawSettings.m_UseZBuffer);
        renderer->SetClearColor(Vector4f{ m_DrawSettings.m_BackgroundColor,1 });
        renderer->SetAlphaBlending(m_DrawSettings.m_AlphaBlend);
        renderer->SetBackfaceCulling(m_DrawSettings.m_BackfaceCulling);
        renderer->ClearZBuffer();
        renderer->ClearScreen();
        renderer->SetBlockMathMode(static_cast<eBlockMathMode>(m_DrawSettings.m_MathType));
        renderer->SetTileMode(static_cast<eTileMode>(m_DrawSettings.m_TileMode));

        {
            ZoneScopedN("Frame");
            renderer->BeginFrame();

            Frustum frustum = Frustum::FromMatrix(m_ModelMatrix * m_CameraMatrix * m_ProjectionMatrix);

            // render stuff to screen buffer
            for (auto& model : m_ModelsData)
            {
                if (frustum.IsInside(model))
                    renderer->Render(model.m_Vertices);
            }

            renderer->EndFrame();
        }

        // for software renderer render screen buffer to window (hardware renderer renders directly to window and calling GetScreenBuffer on it returns empty buffer)
        if (auto& buf = renderer->GetScreenBuffer(); buf.size())
        {
            {
                ZoneScopedN("Update screen texture");
                // update texture

                if (m_DrawSettings.m_RenderDepthBuffer)
                    renderer->RenderDepthBuffer();

                m_ScreenTexture.update((uint8_t*)buf.data());
            }
            {
                ZoneScopedN("Draw screen texture");
                // render texture to screen
                m_MainWindow.draw(m_ScreenSprite);
            }
        }


        // display fps counter
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::Begin("FPS", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::SetWindowPos(ImVec2(0, 0));
        ImGui::Text("FPS: %d", fps);
        ImGui::Text("%s %s (%s)", CMAKE_BUILD_NAME, sizeof(void*) == 8 ? "x64" : "x86", COMPILER_NAME);
        ImGui::Text("Mem : %u KB", uint32_t(g_memory_resource_mem.load() / 1024));
        ImGui::Text("m_Max Overdraw: %d", g_max_overdraw.load());
        ImGui::End();

        DrawRenderingStats(renderer->GetPixelsDrawn());

        OpenSceneDataDialog(m_ModelPaths);

        // render UI on top
        ImGui::SFML::Render(m_MainWindow);

        // end the current frame
        m_MainWindow.display();

        // update stats
        auto drawStats = renderer->GetDrawStats();
        drawStats.m_DT = durationUS;
        DrawStatsSystem::AddSample(drawStats);
    }

    m_MainWindow.setActive(false);

    ImGui::SFML::Shutdown();

    exit(0); // TODO: Creating OpenGl renderer causes application crash at exit

    return 0;
}

template<typename T>
void SelectCheckboxOptions(span<pair<T, bool>> values, span<const string_view> labels)
{
    for (int i = 0; i < values.size(); ++i)
    {
        ImGui::Checkbox(labels[i].data(), &values[i].second);
    }
}

void Application::SetupPerformanceTests()
{
    constexpr string_view s_PredefinedModelNames[] = { "Cube 15%", "Cube 25%", "Cube 75%", "Cube 100%", "Teapot 15%", "Teapot 25%", "Teapot 75%", "Teapot 100%", "Shiba 15%", "Shiba 25%", "Shiba 75%", "Shiba 100%", "Dog 15%", "Dog 25%", "Dog 75%", "Dog 100%" };
    constexpr string_view s_TileModeNames[] = { "32x32", "16x16", "8x8" };
    constexpr string_view s_RasterModeNames[] = { "CPU", "CPUx8", "SSEx4", "SSEx8", "AVXx8" };
    constexpr string_view s_YesNo[] = { "With", "Without" };

    if (ImGui::CollapsingHeader("Models"))
        SelectCheckboxOptions<int>(m_SelectedModels, s_PredefinedModelNames);

    if (ImGui::CollapsingHeader("Tile mode"))
        SelectCheckboxOptions<int>(m_SelectedTileModes, s_TileModeNames);

    if (ImGui::CollapsingHeader("Raster mode"))
        SelectCheckboxOptions<int>(m_SelectedRasterModes, s_RasterModeNames);

    if (ImGui::CollapsingHeader("Compress Partial Tiles"))
        SelectCheckboxOptions<bool>(m_SelectedCompressed, s_YesNo);

    if (ImGui::CollapsingHeader("Multithreaded Transform and Clip"))
        SelectCheckboxOptions<bool>(m_SelectedMultiTC, s_YesNo);

    if (ImGui::CollapsingHeader("Fragment Shader On"))
        SelectCheckboxOptions<bool>(m_SelectedFragmentShader, s_YesNo);

    if (ImGui::Button("Run performance tests"))
    {
        STATS_DUMP = true;
        RunPerformanceTests();
    }
}

void Application::DrawRenderingStats(int pixels)
{
    if (!ImGui::Begin("Stats"))
    {
        ImGui::End();
        return;
    }

    auto& avg = DrawStatsSystem::GetAvg();
    auto& min = DrawStatsSystem::GetMin();
    auto& max = DrawStatsSystem::GetMax();
    auto& med = DrawStatsSystem::GetMed();
    auto& std = DrawStatsSystem::GetStd();

    auto ScreenPixels = SCREEN_WIDTH * SCREEN_HEIGHT;

    ImVec4 Col1{ 1.0f,1.0f,1.0f,1.0f };
    ImVec4 Col2{ 0.4f,0.8f,0.9f,1.0f };

    ImGui::TextColored(Col1, " __________________________________________________________________________________________________________");
    ImGui::TextColored(Col1, "|                               |      AVG     |      MIN     |      MAX     |    MEDIAN    |    STD DEV   |");
    ImGui::TextColored(Col1, "|-------------------------------|--------------|--------------|--------------|--------------|--------------|");
    ImGui::TextColored(Col1, "| FPS                           | %12d | %12d | %12d | %12d | %12d |", int(avg.m_FPS), int(min.m_FPS), int(max.m_FPS), int(med.m_FPS), int(std.m_FPS));
    ImGui::TextColored(Col2, "| Triangles analyzed per frame  | %12d | %12d | %12d | %12d | %12d |", int(avg.m_FrameTriangles), int(min.m_FrameTriangles), int(max.m_FrameTriangles), int(med.m_FrameTriangles), int(std.m_FrameTriangles));
    ImGui::TextColored(Col1, "| Triangles drawn per frame     | %12d | %12d | %12d | %12d | %12d |", int(avg.m_FrameTrianglesDrawn), int(min.m_FrameTrianglesDrawn), int(max.m_FrameTrianglesDrawn), int(med.m_FrameTrianglesDrawn), int(std.m_FrameTrianglesDrawn));
    ImGui::TextColored(Col2, "| Pixels analyzed  per frame    | %12d | %12d | %12d | %12d | %12d |", int(avg.m_FramePixels), int(min.m_FramePixels), int(max.m_FramePixels), int(med.m_FramePixels), int(std.m_FramePixels));
    ImGui::TextColored(Col1, "| Pixels drawn per frame        | %12d | %12d | %12d | %12d | %12d |", int(avg.m_FramePixelsDrawn), int(min.m_FramePixelsDrawn), int(max.m_FramePixelsDrawn), int(med.m_FramePixelsDrawn), int(std.m_FramePixelsDrawn));
    ImGui::TextColored(Col2, "| Pixels calculated per frame   | %12d | %12d | %12d | %12d | %12d |", int(avg.m_FramePixelsCalcualted), int(min.m_FramePixelsCalcualted), int(max.m_FramePixelsCalcualted), int(med.m_FramePixelsCalcualted), int(std.m_FramePixelsCalcualted));
    ImGui::TextColored(Col1, "| Avg Draws per tile            | %12f | %12f | %12f | %12f | %12f |", (avg.m_FrameDrawsPerTile) * 0.01f, (min.m_FrameDrawsPerTile) * 0.01f, (max.m_FrameDrawsPerTile) * 0.01f, (med.m_FrameDrawsPerTile) * 0.01f, (std.m_FrameDrawsPerTile) * 0.01f);
    ImGui::TextColored(Col2, "| Frame draw time (US)          | %12d | %12d | %12d | %12d | %12d |", int(avg.m_DrawTimeUS), int(min.m_DrawTimeUS), int(max.m_DrawTimeUS), int(med.m_DrawTimeUS), int(std.m_DrawTimeUS));
    ImGui::TextColored(Col1, "| Frame draw time per thread(US)| %12d | %12d | %12d | %12d | %12d |", int(avg.m_DrawTimePerThreadUS), int(min.m_DrawTimePerThreadUS), int(max.m_DrawTimePerThreadUS), int(med.m_DrawTimePerThreadUS), int(std.m_DrawTimePerThreadUS));
    ImGui::TextColored(Col2, "| Transform Time (US)           | %12d | %12d | %12d | %12d | %12d |", int(avg.m_TransformTimeUS), int(min.m_TransformTimeUS), int(max.m_TransformTimeUS), int(med.m_TransformTimeUS), int(std.m_TransformTimeUS));
    ImGui::TextColored(Col1, "| Transform Time per thread(US) | %12d | %12d | %12d | %12d | %12d |", int(avg.m_TransformTimePerThreadUS), int(min.m_TransformTimePerThreadUS), int(max.m_TransformTimePerThreadUS), int(med.m_TransformTimePerThreadUS), int(std.m_TransformTimePerThreadUS));
    ImGui::TextColored(Col2, "| Raster time (US)              | %12d | %12d | %12d | %12d | %12d |", int(avg.m_RasterTimeUS), int(min.m_RasterTimeUS), int(max.m_RasterTimeUS), int(med.m_RasterTimeUS), int(std.m_RasterTimeUS));
    ImGui::TextColored(Col1, "| Raster time per thread(US)    | %12d | %12d | %12d | %12d | %12d |", int(avg.m_RasterTimePerThreadUS), int(min.m_RasterTimePerThreadUS), int(max.m_RasterTimePerThreadUS), int(med.m_RasterTimePerThreadUS), int(std.m_RasterTimePerThreadUS));
    ImGui::TextColored(Col2, "| Fillrate (Kilo pixels/s)      | %12d | %12d | %12d | %12d | %12d |", int(avg.m_FillrateKP), int(min.m_FillrateKP), int(max.m_FillrateKP), int(med.m_FillrateKP), int(std.m_FillrateKP));
    ImGui::TextColored(Col1, "|_______________________________|______________|______________|______________|______________|______________|");
    ImGui::TextColored(Col1, " Screen coverage %u%% (%u pixels / %u pixels)", pixels * 100 / ScreenPixels, pixels, ScreenPixels);
    ImGui::End();
}

struct DrawStatsCollection
{
    const DrawStatsSystem::Stats& avg;
    const DrawStatsSystem::Stats& min;
    const DrawStatsSystem::Stats& max;
    const DrawStatsSystem::Stats& med;
    const DrawStatsSystem::Stats& std;
};

template< int FRAMES >
DrawStatsCollection GetDrawStatsCollectionImpl()
{
    return DrawStatsCollection
    {
        DrawStatsSystem::GetAvg<FRAMES>(),
        DrawStatsSystem::GetMin<FRAMES>(),
        DrawStatsSystem::GetMax<FRAMES>(),
        DrawStatsSystem::GetMed<FRAMES>(),
        DrawStatsSystem::GetStd<FRAMES>()
    };
}

DrawStatsCollection GetDrawStatsCollection(int frames)
{
    if (frames == 480)
        return GetDrawStatsCollectionImpl<480>();
    if (frames == 240)
        return GetDrawStatsCollectionImpl<240>();
    if (frames == 120)
        return GetDrawStatsCollectionImpl<120>();
    if (frames == 60)
        return GetDrawStatsCollectionImpl<60>();

    return GetDrawStatsCollectionImpl<30>();
}

void Application::SaveStatsToFile(int pixels, int frames)
{
    auto [avg2, min, max, med, std] = GetDrawStatsCollection(frames);

    auto ScreenPixels = SCREEN_WIDTH * SCREEN_HEIGHT;
    int screenCoverage = pixels * 100 / ScreenPixels;

    string modelPath = m_ModelPaths.m_TexturePath.empty() ? m_LastModelPaths.m_TexturePath : m_ModelPaths.m_TexturePath;

    string tileSize = "32x32";
    if (m_DrawSettings.m_TileMode == 1)
        tileSize = "16x16";
    else if (m_DrawSettings.m_TileMode == 2)
        tileSize = "8x8";

    std::filesystem::path p(modelPath);
    std::string baseName = p.stem().string();
    std::string filename = "MED_" + baseName +
        "_SC_" + std::to_string(screenCoverage) +
        "_T_" + std::to_string(m_DrawSettings.m_MathType) +
        "_FS_" + std::to_string(!g_TrivialFS) +
        "_MULTI_" + std::to_string(g_MultithreadedTransformAndClip) +
        "_COM_" + std::to_string(g_CompressedPartialTile) +
        "_" + tileSize + ".txt";

    static std::unordered_map<std::string, std::vector<int>> stats;
    static std::vector<std::string> metricOrder;

    int tIndex = m_DrawSettings.m_ThreadsCount - 1;

    auto setVal = [&](const std::string& key, int val)
        {
            if (!stats.contains(key))
            {
                stats[key] = std::vector<int>(MAX_THREADS_COUNT, 0);
                metricOrder.push_back(key);
            }
            stats[key][tIndex] = val;
        };

    LAST_FPS = med.m_FPS;
    LAST_FPS_STD_DEV = std.m_FPS;

    setVal("FPS----------------------------", int(med.m_FPS));
    setVal("Raster time (us)---------------", int(med.m_RasterTimeUS));
    setVal("Fillrate (Kilo pixels/s)-------", int(med.m_FillrateKP));
    setVal("Transform Time (us)------------", int(med.m_TransformTimeUS));
    setVal("Frame draw time (us)-----------", int(med.m_DrawTimeUS));
    setVal("Frame draw time per thread (us)", int(med.m_DrawTimePerThreadUS));
    setVal("Raster time per thread (us)----", int(med.m_RasterTimePerThreadUS));
    setVal("Triangles analyzed per frame---", int(med.m_FrameTriangles));
    setVal("Triangles drawn per frame------", int(med.m_FrameTrianglesDrawn));
    setVal("Pixels analyzed per frame------", int(med.m_FramePixels));
    setVal("Pixels drawn per frame---------", int(med.m_FramePixelsDrawn));

    std::ofstream fout(filename, std::ios::trunc);
    if (!fout.is_open())
    {
        std::cerr << "Couldn't open file " << filename << " to save.\n";
        return;
    }

    const size_t colWidth = 5;

    std::string header = "Metric-------------------------";
    fout << header;
    for (int t = 1; t <= MAX_THREADS_COUNT; t++)
    {
        std::string h = "T_" + std::to_string(t);
        if (h.size() < colWidth) h.resize(colWidth, ' ');
        fout << "\t" << h;
    }
    fout << "\n";

    for (const auto& metric : metricOrder)
    {
        fout << metric;
        for (int v : stats[metric])
        {
            std::string s = (v == 0 ? "-" : std::to_string(v));
            if (s.size() < colWidth) s.resize(colWidth, ' ');
            fout << "\t" << s;
        }
        fout << "\n";
    }

    fout.close();
    std::cout << "Stats saved/updated in: " << filename << " for " << std::to_string(m_DrawSettings.m_ThreadsCount) << " threads.""\n";
}