/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2023
*/

#include "Application.h"
#include "Fallback.h"
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "../ImGuiFileDialog/ImGuiFileDialog.h"


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
        if ((pScene->mMeshes[i]->mPrimitiveTypes & aiPrimitiveType_TRIANGLE)!=aiPrimitiveType_TRIANGLE)
        {
            // skip non-triangle meshes
            continue;
        }
        totalTriangles += pScene->mMeshes[i]->mNumFaces;
    }

    if (totalTriangles > MAX_MODEL_TRIANGLES)
    {
        printf("Total triangles count %d exceeds max allowed number of triangles %d\n" , totalTriangles , MAX_MODEL_TRIANGLES );
        return vector<Model>();
    }

    vector<Model> result;

    for (int i = 0; i < pScene->mNumMeshes; ++i)
    {
        result.push_back(Model());

        Model& model = result.back();

        auto mesh = pScene->mMeshes[i];
        if ((mesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE)!=aiPrimitiveType_TRIANGLE)
        {
            // skip non-triangle meshes
            continue;
        }

        model.vertices.reserve(mesh->mNumFaces * 3);

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
                v.color = Vector4f(1, 1, 1, 1);

                int index = face->mIndices[i];
                if (mesh->mColors[0] != NULL)
                    v.color = Vector4f(mesh->mColors[0][index].r, mesh->mColors[0][index].g, mesh->mColors[0][index].b, mesh->mColors[0][index].a);

                if (mesh->mTextureCoords[0] != NULL)
                    v.uv = Vector2f(mesh->mTextureCoords[0][index].x, mesh->mTextureCoords[0][index].y);


                if (mesh->mNormals != NULL)
                    v.normal = Vector3f(mesh->mNormals[index].x, mesh->mNormals[index].y, mesh->mNormals[index].z);

                v.position = Vector3f(mesh->mVertices[index].x, mesh->mVertices[index].y, mesh->mVertices[index].z);
                model.vertices.push_back(v);
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
        for (auto& v : model.vertices)
        {
            min = min.CWiseMin(v.position);
            max = max.CWiseMax(v.position);
        }
    }

    // center and scale model
    Vector3f center = (min + max) * 0.5f;
    const float expectedFinalDistance = 2.0f;
    const float currentDistance = (max - min).MaxComponent();
    const float currentToExpectedMultiplier = expectedFinalDistance / currentDistance;

    for (auto& model : models)
    {
        for (auto& v : model.vertices)
        {
            v.position = (v.position - center) * currentToExpectedMultiplier;
        }
    }
}

vector<Model> Application::LoadFallbackModel()
{
    ZoneScoped;
    vector<Model> result;
    result.push_back(Model());

    vector<Vertex>& vertices = result[0].vertices;
    vertices.resize(FALLBACK_MODEL_VERT_COUNT);
    std::copy(fallbackVertices, fallbackVertices + FALLBACK_MODEL_VERT_COUNT, vertices.begin());

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
    }
    else
    {
        auto error = aiGetErrorString();
        printf("Error loading model: %s\n", error);
        printf("Loading fallback model...\n");

        result = LoadFallbackModel();
    }

    NormalizeModelPosition(result);

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
        selectedPaths.modelPath = ImGuiFileDialog::Instance()->GetFilePathName();
    });

    ImGui::SameLine(); OpenDialog("Choose Model Texture", TEXTURE_FORMATS, [&selectedPaths]
    {
        selectedPaths.texturePath = ImGuiFileDialog::Instance()->GetFilePathName();
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
    m_MainWindow.setFramerateLimit(60);

    // create renderers
    m_Contexts[0].pRenderer = RendererFactory::CreateRenderer(eRendererType::Software, SCREEN_WIDTH, SCREEN_HEIGHT);
    m_Contexts[1].pRenderer = RendererFactory::CreateRenderer(eRendererType::Hardware, SCREEN_WIDTH, SCREEN_HEIGHT);

    // load default textures
    m_Contexts[0].pModelTexture = m_Contexts[0].pRenderer->LoadTexture(INIT_TEXTURE_PATH.c_str());
    m_Contexts[1].pModelTexture = m_Contexts[1].pRenderer->LoadTexture(INIT_TEXTURE_PATH.c_str());

    // initialize ImGui
    ImGui::SFML::Init(m_MainWindow);

    // create screen texture and sprite
    m_ScreenTexture.create(SCREEN_WIDTH, SCREEN_HEIGHT);
    m_ScreenSprite.setTexture(m_ScreenTexture);
    // flip the sprite vertically
    m_ScreenSprite.setTextureRect(sf::IntRect(0, SCREEN_HEIGHT, SCREEN_WIDTH, -SCREEN_HEIGHT));

    // set initial matrices
    m_CameraMatrix      = Matrix4f::CreateLookAtMatrix(Vector3f(0, 0, -10), Vector3f(0, 0, 0), Vector3f(0, 1, 0));
    m_ProjectionMatrix  = Matrix4f::CreateProjectionMatrix(60, (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT, 0.8f, 1000.0f);
    m_ModelMatrix       = Matrix4f::Identity();

    m_LastFrameTime = std::chrono::high_resolution_clock::now();

    return true;
}

int Application::Run()
{
    ZoneScoped;
    // run the program as long as the window is open
    while (m_MainWindow.isOpen())
    {
        FrameMark;
        ZoneScopedN("Main Loop");

        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_LastFrameTime).count();
        int fps = duration ? 1000 / duration : 0;
        m_LastFrameTime = now;

        auto renderer     = m_Contexts[m_DrawSettings.rendererType].pRenderer;
        auto modelTexture = m_Contexts[m_DrawSettings.rendererType].pModelTexture;

        // check all the window's events that were triggered since the last iteration of the loop
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
                    m_DrawSettings.rendererType = (m_DrawSettings.rendererType+1)%2;
            }
        }

        // load model and texture if user selected them
        if (m_LastModelPaths.modelPath != m_ModelPaths.modelPath)
        {
            m_ModelsData = LoadModelVertices(m_ModelPaths.modelPath.c_str());
            m_LastModelPaths = m_ModelPaths;
            m_ModelPaths.texturePath = DEFAULT_TEXTURE_PATH;
        }

        if (m_LastModelPaths.texturePath != m_ModelPaths.texturePath)
        {
            m_Contexts[0].pModelTexture = m_Contexts[0].pRenderer->LoadTexture(m_ModelPaths.texturePath.c_str());
            m_Contexts[1].pModelTexture = m_Contexts[1].pRenderer->LoadTexture(m_ModelPaths.texturePath.c_str());
            m_LastModelPaths = m_ModelPaths;
        }

        m_ModelMatrix = Matrix4f::Rotation(m_DrawSettings.modelRotation / 180.f * PI ) * Matrix4f::Scale(Vector3f(m_DrawSettings.modelScale, m_DrawSettings.modelScale, m_DrawSettings.modelScale)) * Matrix4f::Translation(m_DrawSettings.modelTranslation);

        renderer->SetTexture(modelTexture);
        renderer->SetModelMatrix(m_ModelMatrix);
        renderer->SetViewMatrix(m_CameraMatrix);
        renderer->SetProjectionMatrix(m_ProjectionMatrix);

        // update UI
        ImGui::SFML::Update(m_MainWindow, m_DeltaClock.restart());

        // render settings window
        ImGui::Begin("Settings");

        ImGui::SliderFloat("Ambient Strength", &m_DrawSettings.ambientStrength, 0, 1);
        ImGui::SliderFloat("Diffuse Strength", &m_DrawSettings.diffuseStrength, 0, 1);
        ImGui::SliderFloat("Specular Strength", &m_DrawSettings.specularStrength, 0, 1);
        ImGui::SliderFloat("Shininess Power", &m_DrawSettings.shininessPower, 1.f , 10.0f );
        ImGui::SliderFloat3("Rotation", &m_DrawSettings.modelRotation.x, 0, FULL_ANGLE);
        ImGui::SliderFloat3("Translation", &m_DrawSettings.modelTranslation.x, -15, 15);
        ImGui::SliderFloat("Scale", &m_DrawSettings.modelScale, 0, 5);
        ImGui::SliderFloat3("Light Position", &m_DrawSettings.lightPosition.x, -20, 20);
        ImGui::SliderInt("Thread Count", &m_DrawSettings.threadsCount, 1, MAX_THREADS_COUNT);
        ImGui::Combo("Renderer Type", &m_DrawSettings.rendererType, "Software\0Hardware\0");
        ImGui::Checkbox("Wireframe", &m_DrawSettings.drawWireframe);
        ImGui::SameLine(); ImGui::Checkbox("Colorize Threads", &m_DrawSettings.colorizeThreads);
        ImGui::SameLine(); ImGui::Checkbox("BBoxes", &m_DrawSettings.drawBBoxes);
        ImGui::SameLine(); ImGui::Checkbox("Use ZBuffer", &m_DrawSettings.useZBuffer);
        ImGui::ColorEdit3("Ambient Color", &m_DrawSettings.ambientColor.x);
        ImGui::ColorEdit3("Diffuse Color", &m_DrawSettings.diffuseColor.x);
        ImGui::ColorEdit3("Background Color", &m_DrawSettings.backgroundColor.x);

        ImGui::End();

        // set render params
        renderer->SetWireFrameColor(m_DrawSettings.wireFrameColor);
        renderer->SetDiffuseColor(m_DrawSettings.diffuseColor);
        renderer->SetAmbientColor(m_DrawSettings.ambientColor);
        renderer->SetLightPosition(m_DrawSettings.lightPosition);
        renderer->SetDiffuseStrength(m_DrawSettings.diffuseStrength);
        renderer->SetAmbientStrength(m_DrawSettings.ambientStrength);
        renderer->SetSpecularStrength(m_DrawSettings.specularStrength);
        renderer->SetShininess( pow(2.0f,m_DrawSettings.shininessPower) );
        renderer->SetThreadsCount(m_DrawSettings.threadsCount);
        renderer->SetDrawWireframe(m_DrawSettings.drawWireframe);
        renderer->SetColorizeThreads(m_DrawSettings.colorizeThreads);
        renderer->SetDrawBBoxes(m_DrawSettings.drawBBoxes);
        renderer->SetZTest(m_DrawSettings.useZBuffer);
        renderer->SetZWrite(m_DrawSettings.useZBuffer);
        renderer->SetClearColor(Vector4f{m_DrawSettings.backgroundColor,1});
        renderer->ClearZBuffer();
        renderer->ClearScreen();

        // render stuff to screen buffer
        for (auto& model : m_ModelsData)
        {
            renderer->Render(model.vertices);
        }

        // for software renderer render screen buffer to window (hardware renderer renders directly to window and calling GetScreenBuffer on it returns empty buffer)
        if (auto& buf = renderer->GetScreenBuffer() ; buf.size())
        {
            ZoneScopedN("Update screen texture");
            // update texture
            m_ScreenTexture.update((uint8_t*)buf.data());

            // render texture to screen
            m_MainWindow.draw(m_ScreenSprite);
        }

        // display fps counter
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::Begin("FPS", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::SetWindowPos(ImVec2(0, 0));
        ImGui::Text("FPS: %d", fps);
        ImGui::End();

        OpenSceneDataDialog(m_ModelPaths);

        // render UI on top
        ImGui::SFML::Render(m_MainWindow);

        // end the current frame
        m_MainWindow.display();
    }

    m_MainWindow.setActive(false);

    ImGui::SFML::Shutdown();

    exit(0); // TODO: Creating OpenGl renderer causes application crash at exit

    return 0;
}