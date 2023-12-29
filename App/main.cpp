#include "Common.h"
#include "IRenderer.h"
#include "Fallback.h"
#include "Math.h"
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "../ImGuiFileDialog/ImGuiFileDialog.h"

struct Model
{
    vector<Vertex> vertices;
};

vector<Model> LoadFromScene(const aiScene* pScene)
{
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

void NormalizeModelPosition(vector<Model>& models)
{
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

vector<Model> LoadFallbackModel()
{
    vector<Model> result;
    result.push_back(Model());

    vector<Vertex>& vertices = result[0].vertices;
    vertices.resize(FALLBACK_MODEL_VERT_COUNT);
    std::copy(fallbackVertices, fallbackVertices + FALLBACK_MODEL_VERT_COUNT, vertices.begin());

    return result;
}



vector<Model> LoadModelVertices(const char* path)
{
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

void OpenDialog(const char* title, const char* filters, function<void()> callback)
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

void OpenSceneDataDialog(MyModelPaths& selectedPaths)
{
    ImGui::Begin("Settings");

    OpenDialog("Choose Model", ".fbx,.glb,.gltf,.3ds,.blend,.obj", [&selectedPaths]
    {
        selectedPaths.modelPath = ImGuiFileDialog::Instance()->GetFilePathName();
    });

    ImGui::SameLine(); OpenDialog("Choose Model Texture", ".png,.jpg,.jpeg,.bmp", [&selectedPaths]
    {
        selectedPaths.texturePath = ImGuiFileDialog::Instance()->GetFilePathName();
    });


    ImGui::End();
}

struct RendererContext
{
    shared_ptr<IRenderer> pRenderer;
    shared_ptr<ITexture>  pModelTexture;
};

struct DrawSettings
{
    Vector3f    modelRotation;
    Vector3f    modelTranslation;
    float       modelScale = 1.0;
    Vector4f    wireFrameColor = Vector4f(1, 0, 1, 1);
    Vector3f    diffuseColor = Vector3f(1, 1, 1);
    Vector3f    ambientColor = Vector3f(1, 1, 1);
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
    int         rendererType = 0;
};

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

int main()
{
    MyModelPaths lastModelPaths;
    MyModelPaths modelPaths;

    // load default model
    vector<Model> modelsData = LoadFallbackModel();

    // create the window
    sf::RenderWindow window(sf::VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT), "Software renderer", sf::Style::Default, GetSFMLOpenGL4_0_WindowSettings());
    {
    window.setActive(true);
    window.setFramerateLimit(60);

    RendererContext contexts[2];

    contexts[0].pRenderer = IRenderer::CreateRenderer(eRendererType::SOFTWARE,SCREEN_WIDTH, SCREEN_HEIGHT);
    contexts[1].pRenderer = IRenderer::CreateRenderer(eRendererType::HARDWARE,SCREEN_WIDTH, SCREEN_HEIGHT);

    contexts[0].pModelTexture = contexts[0].pRenderer->LoadTexture(INIT_TEXTURE_PATH.c_str());
    contexts[1].pModelTexture = contexts[1].pRenderer->LoadTexture(INIT_TEXTURE_PATH.c_str());


    ImGui::SFML::Init(window);

    sf::Texture screenTexture;
    screenTexture.create(SCREEN_WIDTH, SCREEN_HEIGHT);

    sf::Sprite sprite;
    sprite.setTexture(screenTexture);
    // flip the sprite vertically
    sprite.setTextureRect(sf::IntRect(0, SCREEN_HEIGHT, SCREEN_WIDTH, -SCREEN_HEIGHT));

    sf::Clock deltaClock;

    const uint8_t MAX_THREADS_COUNT = uint8_t( std::min<int>( 12 , std::thread::hardware_concurrency() ) );

    Matrix4f    cameraMatrix = Matrix4f::CreateLookAtMatrix(Vector3f(0, 0, -10), Vector3f(0, 0, 0), Vector3f(0, 1, 0));
    Matrix4f    projectionMatrix = Matrix4f::CreateProjectionMatrix(60, (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT, 0.8f, 1000.0f);
    Matrix4f    modelMatrix = Matrix4f::Identity();

    DrawSettings drawSettings;

    auto lastFrameTime = std::chrono::high_resolution_clock::now();

    // run the program as long as the window is open
    while (window.isOpen())
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFrameTime).count();
        int fps = duration ? 1000 / duration : 0;
        lastFrameTime = now;

        auto renderer     = contexts[drawSettings.rendererType].pRenderer;
        auto modelTexture = contexts[drawSettings.rendererType].pModelTexture;

        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event))
        {
            ImGui::SFML::ProcessEvent(window, event);

            // "close requested" event: we close the window
            if (event.type == sf::Event::Closed)
                window.close();

            if (event.type == sf::Event::KeyPressed)
            {
                // space pressed
                if (event.key.code == sf::Keyboard::Space)
                    drawSettings.rendererType = (drawSettings.rendererType+1)%2;
            }
        }

        // load model and texture if user selected them
        if (lastModelPaths.modelPath != modelPaths.modelPath)
        {
            modelsData = LoadModelVertices(modelPaths.modelPath.c_str());
            modelPaths.texturePath = DEFAULT_TEXTURE_PATH;
            lastModelPaths = modelPaths;
        }

        if (lastModelPaths.texturePath != modelPaths.texturePath)
        {
            contexts[0].pModelTexture = contexts[0].pRenderer->LoadTexture(modelPaths.texturePath.c_str());
            contexts[1].pModelTexture = contexts[1].pRenderer->LoadTexture(modelPaths.texturePath.c_str());
            lastModelPaths = modelPaths;
        }

        modelMatrix = Matrix4f::Rotation(drawSettings.modelRotation / 180.f * PI ) * Matrix4f::Scale(Vector3f(drawSettings.modelScale, drawSettings.modelScale, drawSettings.modelScale)) * Matrix4f::Translation(drawSettings.modelTranslation);

        renderer->SetTexture(modelTexture);
        renderer->SetModelMatrixx(modelMatrix);
        renderer->SetViewMatrix(cameraMatrix);
        renderer->SetProjectionMatrix(projectionMatrix);

        // update UI
        ImGui::SFML::Update(window, deltaClock.restart());

        // render settings window
        ImGui::Begin("Settings");

        ImGui::SliderFloat("Ambient Strength", &drawSettings.ambientStrength, 0, 1);
        ImGui::SliderFloat("Diffuse Strength", &drawSettings.diffuseStrength, 0, 1);
        ImGui::SliderFloat("Specular Strength", &drawSettings.specularStrength, 0, 1);
        ImGui::SliderFloat("Shininess Power", &drawSettings.shininessPower, 1.f , 10.0f );
        ImGui::SliderFloat3("Rotation", &drawSettings.modelRotation.x, 0, FULL_ANGLE);
        ImGui::SliderFloat3("Translation", &drawSettings.modelTranslation.x, -15, 15);
        ImGui::SliderFloat("Scale", &drawSettings.modelScale, 0, 5);
        ImGui::SliderFloat3("Light Position", &drawSettings.lightPosition.x, -20, 20);
        ImGui::SliderInt("Thread Count", &drawSettings.threadsCount, 1, MAX_THREADS_COUNT);
        ImGui::Combo("Renderer Type", &drawSettings.rendererType, "Software\0Hardware\0");
        ImGui::Checkbox("Wireframe", &drawSettings.drawWireframe);
        ImGui::SameLine(); ImGui::Checkbox("Colorize Threads", &drawSettings.colorizeThreads);
        ImGui::SameLine(); ImGui::Checkbox("BBoxes", &drawSettings.drawBBoxes);
        ImGui::SameLine(); ImGui::Checkbox("Use ZBuffer", &drawSettings.useZBuffer);
      //ImGui::ColorEdit4("WireFrame Color", &drawSettings.wireFrameColor.x);
        ImGui::ColorEdit3("Ambient Color", &drawSettings.ambientColor.x);
        ImGui::ColorEdit3("Diffuse Color", &drawSettings.diffuseColor.x);

        ImGui::End();

        // set render params
        renderer->SetWireFrameColor(drawSettings.wireFrameColor);
        renderer->SetDiffuseColor(drawSettings.diffuseColor);
        renderer->SetAmbientColor(drawSettings.ambientColor);
        renderer->SetLightPosition(drawSettings.lightPosition);
        renderer->SetDiffuseStrength(drawSettings.diffuseStrength);
        renderer->SetAmbientStrength(drawSettings.ambientStrength);
        renderer->SetSpecularStrength(drawSettings.specularStrength);
        renderer->SetShininess( pow(2.0f,drawSettings.shininessPower) );
        renderer->SetThreadsCount(drawSettings.threadsCount);
        renderer->SetDrawWireframe(drawSettings.drawWireframe);
        renderer->SetColorizeThreads(drawSettings.colorizeThreads);
        renderer->SetDrawBBoxes(drawSettings.drawBBoxes);
        renderer->SetZTest(drawSettings.useZBuffer);
        renderer->SetZWrite(drawSettings.useZBuffer);
        renderer->ClearZBuffer();
        renderer->ClearScreen();

        // render stuff to screen buffer
        for (auto& model : modelsData)
        {
            renderer->Render(model.vertices);
        }

        // for software renderer render screen buffer to window (hardware renderer renders directly to window and calling GetScreenBuffer on it returns empty buffer)
        if (auto& buf = renderer->GetScreenBuffer() ; buf.size())
        {
            // update texture
            screenTexture.update((uint8_t*)buf.data());

            // render texture to screen
            window.draw(sprite);
        }

        // display fps counter
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::Begin("FPS", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::SetWindowPos(ImVec2(0, 0));
        ImGui::Text("FPS: %d", fps);
        ImGui::End();

        OpenSceneDataDialog(modelPaths);

        // render UI on top
        ImGui::SFML::Render(window);

        // end the current frame
        window.display();
    }

    }

    window.setActive(false);

    ImGui::SFML::Shutdown();

    exit(0); // TODO: Creating OpenGl renderer causes application crash at exit

    return 0;
}