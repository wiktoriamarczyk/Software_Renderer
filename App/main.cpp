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

    int totalVertices = 0;

    for (int i = 0; i < pScene->mNumMeshes; ++i)
    {
        if( (pScene->mMeshes[i]->mPrimitiveTypes & aiPrimitiveType_TRIANGLE)!=aiPrimitiveType_TRIANGLE )
        {
            // skip non-triangle meshes
            continue;
        }
        totalVertices += pScene->mMeshes[i]->mNumFaces * 3;
    }

    if (totalVertices > MAX_MODEL_VERTICES)
    {
        printf("Total vertices count %d exceeds max allowed number of vertices %d\n" , totalVertices , MAX_MODEL_VERTICES );
        return vector<Model>();
    }

    vector<Model> result;

    for (int i = 0; i < pScene->mNumMeshes; ++i)
    {
        result.push_back(Model());

        Model& model = result.back();

        auto mesh = pScene->mMeshes[i];
        if( (mesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE)!=aiPrimitiveType_TRIANGLE )
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
        auto Error = aiGetErrorString();
        printf("Error loading model: %s\n", Error);
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

    OpenDialog("Choose Model", ".fbx,.glb,.gltf", [&selectedPaths]
    {
        selectedPaths.modelPath = ImGuiFileDialog::Instance()->GetFilePathName();
    });

    OpenDialog("Choose Model Texture", ".png,.jpg,.jpeg,.bmp,.tga", [&selectedPaths]
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

int main()
{
    MyModelPaths lastModelPaths;
    MyModelPaths modelPaths;

    // load default model
    vector<Model> modelsData = LoadFallbackModel();


    // specify the window context settings - requie OpenGL 4.0
    sf::ContextSettings settings;
    settings.depthBits = 24;
    settings.stencilBits = 8;
    settings.antialiasingLevel = 0;
    settings.majorVersion = 4;
    settings.minorVersion = 0;

    // create the window
    sf::RenderWindow window(sf::VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT), "Software renderer", sf::Style::Default, settings);
    {

    settings = window.getSettings();

    printf( "depth bits: %u\n" , settings.depthBits );
    printf( "stencil bits: %u\n" , settings.stencilBits );
    printf( "antialiasing level: %u\n" , settings.antialiasingLevel );
    printf( "version: %u,%u\n" , settings.majorVersion , settings.minorVersion );

    window.setActive(true); // ??

    RendererContext Contexts[2];

    Contexts[0].pRenderer = IRenderer::CreateRenderer(eRendererType::SOFTWARE,SCREEN_WIDTH, SCREEN_HEIGHT);
    Contexts[1].pRenderer = IRenderer::CreateRenderer(eRendererType::HARDWARE,SCREEN_WIDTH, SCREEN_HEIGHT);

    Contexts[0].pModelTexture = Contexts[0].pRenderer->LoadTexture(INIT_TEXTURE_PATH.c_str());
    Contexts[1].pModelTexture = Contexts[1].pRenderer->LoadTexture(INIT_TEXTURE_PATH.c_str());

    RendererContext* pActiveContext = &Contexts[0];

    window.setFramerateLimit(60);

    ImGui::SFML::Init(window);

    sf::Texture screenTexture;
    screenTexture.create(SCREEN_WIDTH, SCREEN_HEIGHT);

    sf::Sprite sprite;
    sprite.setTexture(screenTexture);

    sf::Clock deltaClock;


    Matrix4f    cameraMatrix = Matrix4f::CreateLookAtMatrix(Vector3f(0, 0, -10), Vector3f(0, 0, 0), Vector3f(0, 1, 0));
    Matrix4f    projectionMatrix = Matrix4f::CreateProjectionMatrix(60, (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT, 0.8f, 1000.0f);
    Matrix4f    modelMatrix = Matrix4f::Identity();

    Vector3f    modelRotation;
    Vector3f    modelTranslation;
    float       modelScale = 1.0;
    Vector4f    wireFrameColor = Vector4f(1, 1, 1, 1);
    Vector4f    diffuseColor = Vector4f(1, 1, 1, 1);
    Vector4f    ambientColor = Vector4f(1, 1, 1, 1);
    Vector3f    lightPosition = Vector3f(0, 0, -20);
    float       diffuseStrength = 0.7f;
    float       ambientStrength = 0.3f;
    float       specularStrength = 0.9f;
    float       shininess = 32.0f;
    int         threadsCount = 0;
    bool        drawWireframe = false;
    bool        drawBBoxes = false;
    bool        colorizeThreads = false;

    auto lastFrameTime = std::chrono::high_resolution_clock::now();

    // run the program as long as the window is open
    while (window.isOpen())
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFrameTime).count();
        int fps = duration ? 1000 / duration : 0;
        lastFrameTime = now;

        auto renderer = pActiveContext->pRenderer;
        auto modelTexture = pActiveContext->pModelTexture;

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
                {
                    if (pActiveContext == &Contexts[0])
                        pActiveContext = &Contexts[1];
                    else
                        pActiveContext = &Contexts[0];
                }
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
            Contexts[0].pModelTexture = Contexts[0].pRenderer->LoadTexture(modelPaths.texturePath.c_str());
            Contexts[1].pModelTexture = Contexts[1].pRenderer->LoadTexture(modelPaths.texturePath.c_str());
            lastModelPaths = modelPaths;
        }

        float angleX = (modelRotation.x / 180.f * PI);
        float angleY = (modelRotation.y / 180.f * PI);
        float angleZ = (modelRotation.z / 180.f * PI);

        modelMatrix = Matrix4f::Rotation(Vector3f(angleX, angleY, angleZ)) * Matrix4f::Scale(Vector3f(modelScale, modelScale, modelScale)) * Matrix4f::Translation(modelTranslation);

        renderer->SetTexture(modelTexture);
        renderer->SetModelMatrixx(modelMatrix);
        renderer->SetViewMatrix(cameraMatrix);
        renderer->SetProjectionMatrix(projectionMatrix);

        // update UI
        ImGui::SFML::Update(window, deltaClock.restart());


        ImGui::Begin("Settings");

        ImGui::ColorEdit4("WireFrame Color", &wireFrameColor.x);
        ImGui::ColorEdit4("Ambient Color", &ambientColor.x);
        ImGui::ColorEdit4("Diffuse Color", &diffuseColor.x);
        ImGui::SliderFloat("Ambient Strength", &ambientStrength, 0, 1);
        ImGui::SliderFloat("Diffuse Strength", &diffuseStrength, 0, 1);
        ImGui::SliderFloat("Specular Strength", &specularStrength, 0, 1);
        ImGui::InputFloat("Shininess", &shininess, 2.f);
        ImGui::SliderFloat3("Rotation", &modelRotation.x, 0, FULL_ANGLE);
        ImGui::SliderFloat3("Translation", &modelTranslation.x, -15, 15);
        ImGui::SliderFloat("Scale", &modelScale, 0, 5);
        ImGui::SliderFloat3("Light Position", &lightPosition.x, -20, 20);
        ImGui::SliderInt("Thread Count", &threadsCount, 0, 12);
        ImGui::Checkbox("Wireframe", &drawWireframe); ImGui::SameLine() ; ImGui::Checkbox("Colorize Threads", &colorizeThreads); ImGui::SameLine(); ImGui::Checkbox("BBoxes", &drawBBoxes);

        ImGui::End();

        renderer->SetWireFrameColor(wireFrameColor);
        renderer->SetDiffuseColor(diffuseColor);
        renderer->SetAmbientColor(ambientColor);
        renderer->SetLightPosition(lightPosition);
        renderer->SetDiffuseStrength(diffuseStrength);
        renderer->SetAmbientStrength(ambientStrength);
        renderer->SetSpecularStrength(specularStrength);
        renderer->SetShininess(shininess);
        renderer->SetThreadsCount(threadsCount);
        renderer->SetDrawWireframe(drawWireframe);
        renderer->SetColorizeThreads(colorizeThreads);
        renderer->SetDrawBBoxes(drawBBoxes);

        // render stuff to screen buffer
        renderer->ClearZBuffer();
        renderer->ClearScreen();

        for (auto& model : modelsData) {
            renderer->Render(model.vertices);
        }

        if( auto buf = renderer->GetScreenBuffer() ; buf.size() )
        {
                  uint32_t* dst = (uint32_t*)buf.data();
            const uint32_t* src = (uint32_t*)renderer->GetScreenBuffer().data();

            for (int y = 0; y < SCREEN_HEIGHT; ++y)
            {
                memcpy(dst + y * SCREEN_WIDTH, src + (SCREEN_HEIGHT - 1 - y) * SCREEN_WIDTH, SCREEN_WIDTH * 4);
            }
            // update texture
            screenTexture.update((uint8_t*)dst);

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