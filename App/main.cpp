#include "Common.h"
#include "Renderer.h"
#include "teapot.h"
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

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

    vector<Model> result;

    for (int i = 0; i < pScene->mNumMeshes; ++i)
    {
        result.push_back(Model());

        Model& model = result.back();

        auto mesh = pScene->mMeshes[i];

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

    int verticesCount = teapot_count / TRIANGLE_VERT_COUNT;
    vertices.resize(verticesCount);

    Vector4f colors[3] = { Vector4f(1, 0, 0,1), Vector4f(0, 1, 0,1), Vector4f(0, 0, 1,1) };

    // we should have normals in the teapot data, but they are missing, so we calculate them here in simplified way assuming that each vertex has the same normal
    for (int i = 0; i < verticesCount; ++i)
    {
        vertices[i].position = Vector3f(teapot[i * TRIANGLE_VERT_COUNT + 0], teapot[i * TRIANGLE_VERT_COUNT + 1], teapot[i * TRIANGLE_VERT_COUNT + 2]);
        vertices[i].color = colors[i % 3];
    }

    for (int i = 0; i < verticesCount; i += TRIANGLE_VERT_COUNT)
    {
        Vector3f A = vertices[i + 0].position;
        Vector3f B = vertices[i + 1].position;
        Vector3f C = vertices[i + 2].position;
        Vector3f wholeTriangleNormal = (B - A).Cross(C - A).Normalized();
        vertices[i + 0].normal = wholeTriangleNormal;
        vertices[i + 1].normal = wholeTriangleNormal;
        vertices[i + 2].normal = wholeTriangleNormal;
    }

    return result;
}

vector<Model> LoadModelVertices()
{
    vector<Model> result;
    auto scene = aiImportFile("C:/Users/wikto/source/repos/Git-C++/SoftwareRenderer/Data/doghouse0908.fbx", aiProcessPreset_TargetRealtime_MaxQuality);
    if (scene)
    {
        result = LoadFromScene(scene);
        aiReleaseImport(scene);
    }
    else
    {
        result = LoadFallbackModel();
    }

    NormalizeModelPosition(result);

    return result;
}

int main()
{
    vector<Model> modelsData = LoadModelVertices();

    shared_ptr<Texture> modelTexture = make_shared<Texture>();
    modelTexture->Load("C:/Users/wikto/source/repos/Git-C++/SoftwareRenderer/Data/doghouse0908.png");

    // create the window
    sf::RenderWindow window(sf::VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT), "Software renderer");

    window.setFramerateLimit(60);

    ImGui::SFML::Init(window);

    sf::Texture screenTexture;
    screenTexture.create(SCREEN_WIDTH, SCREEN_HEIGHT);

    sf::Sprite sprite;
    sprite.setTexture(screenTexture);

    sf::Clock deltaClock;

    SoftwareRenderer renderer(SCREEN_WIDTH, SCREEN_HEIGHT);

    renderer.SetTexture(modelTexture);

    Matrix4f cameraMatrix = Matrix4f::CreateLookAtMatrix(Vector3f(0, 0, -10), Vector3f(0, 0, 1), Vector3f(0, 1, 0));
    Matrix4f projectionMatrix = Matrix4f::CreateProjectionMatrix(90, (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT, 0.1f, 1000.0f);
    Matrix4f modelMatrix = Matrix4f::Identity();

    auto lastFrameTime = std::chrono::high_resolution_clock::now();

    // run the program as long as the window is open
    while (window.isOpen())
    {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFrameTime).count();
        int fps = duration ? 1000 / duration : 0;
        lastFrameTime = now;

        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event))
        {
            ImGui::SFML::ProcessEvent(window, event);

            // "close requested" event: we close the window
            if (event.type == sf::Event::Closed)
                window.close();
        }

        float angleX = (renderer.GetRotation().x / 180.f * PI);
        float angleY = (renderer.GetRotation().y / 180.f * PI);
        float angleZ = (renderer.GetRotation().z / 180.f * PI);
        float scale = renderer.GetScale();
        Vector3f translation = renderer.GetTranslation();

        modelMatrix = Matrix4f::Rotation(Vector3f(angleX, angleY, angleZ)) * Matrix4f::Scale(Vector3f(scale, scale, scale)) * Matrix4f::Translation(translation);

        renderer.SetModelMatrixx(modelMatrix);
        renderer.SetViewMatrix(cameraMatrix);
        renderer.SetProjectionMatrix(projectionMatrix);

        // update UI
        ImGui::SFML::Update(window, deltaClock.restart());
        renderer.UpdateUI();

        // render stuff to screen buffer
        renderer.ClearZBuffer();
        renderer.ClearScreen();

        for (auto& model : modelsData) {
            renderer.Render(model.vertices);
           // renderer.RenderWireframe(model.vertices);
        }

        auto buf = renderer.GetScreenBuffer();

              uint32_t* dst = (uint32_t*)buf.data();
        const uint32_t* src = (uint32_t*)renderer.GetScreenBuffer().data();

        for (int y = 0; y < SCREEN_HEIGHT; ++y)
        {
            memcpy(dst + y * SCREEN_WIDTH, src + (SCREEN_HEIGHT - 1 - y) * SCREEN_WIDTH, SCREEN_WIDTH * 4);
        }
        // update texture
        screenTexture.update((uint8_t*)dst);

        // render texture to screen
        window.draw(sprite);

        // display fps counter
        ImGui::Begin("FPS", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs);
        ImGui::SetWindowPos(ImVec2(0, 0));
        ImGui::Text("FPS: %d", fps);
        ImGui::End();

        // render UI on top
        ImGui::SFML::Render(window);

        // end the current frame
        window.display();
    }
    ImGui::SFML::Shutdown();

    return 0;
}