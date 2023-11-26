#include "Common.h"
#include "Renderer.h"

#include "teapot.h"

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

vector<Vertex> LoadFromScene(const aiScene* scene)
{
    if (!scene->HasMeshes())
    {
        printf("No meshes\n");
        return vector<Vertex>();
    }
    auto mesh = scene->mMeshes[0];

    vector<Vertex> vertices;
    vertices.reserve(mesh->mNumFaces * 3);

    for (int t = 0; t < mesh->mNumFaces; ++t)
    {
        const aiFace* face = &mesh->mFaces[t];
        if(face->mNumIndices != 3)
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

            //if (mesh->mTextureCoords[0] != NULL)
            //    v.color = Vector4f(mesh->mTextureCoords[0][index].x, mesh->mTextureCoords[0][index].y,1,1);


            if (mesh->mNormals != NULL)
                v.normal = Vector3f(mesh->mNormals[index].x, mesh->mNormals[index].y, mesh->mNormals[index].z);

            v.position = Vector3f(mesh->mVertices[index].x, mesh->mVertices[index].y, mesh->mVertices[index].z);
            vertices.push_back(v);
        }
    }

    return vertices;
}


vector<Vertex> GenerateTeapotVertices()
{
    auto scene = aiImportFile("C:/Users/wikto/source/repos/Git-C++/SoftwareRenderer/Data/scene.gltf", aiProcessPreset_TargetRealtime_MaxQuality);
    if (scene)
    {
        auto vertices = LoadFromScene(scene);
        aiReleaseImport(scene);
        return vertices;
    };

    int verticesCount = teapot_count / TRIANGLE_VERT_COUNT;
    vector<Vertex> teapotData(verticesCount);

    Vector4f colors[3] = { Vector4f(1, 0, 0,1), Vector4f(0, 1, 0,1), Vector4f(0, 0, 1,1) };

    // we should have normals in the teapot data, but they are missing, so we calculate them here in simplified way assuming that each vertex has the same normal
    for (int i = 0; i < verticesCount; ++i)
    {
        teapotData[i].position = Vector3f(teapot[i * TRIANGLE_VERT_COUNT + 0], teapot[i * TRIANGLE_VERT_COUNT + 1], teapot[i * TRIANGLE_VERT_COUNT + 2]);

        teapotData[i].color = colors[i % 3];
    }
    for (int i = 0; i < verticesCount; i += TRIANGLE_VERT_COUNT)
    {
        Vector3f A = teapotData[i + 0].position;
        Vector3f B = teapotData[i + 1].position;
        Vector3f C = teapotData[i + 2].position;
        Vector3f wholeTriangleNormal = (B - A).Cross(C - A).Normalized();
        teapotData[i + 0].normal = wholeTriangleNormal;
        teapotData[i + 1].normal = wholeTriangleNormal;
        teapotData[i + 2].normal = wholeTriangleNormal;
    }

    return teapotData;
}

int main()
{
    vector<Vertex> teapotData = GenerateTeapotVertices();

    // create the window
    sf::RenderWindow window(sf::VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT), "Software renderer");

    window.setFramerateLimit(60);

    ImGui::SFML::Init(window);

    sf::Texture texture;
    texture.create(SCREEN_WIDTH, SCREEN_HEIGHT);

    sf::Sprite sprite;
    sprite.setTexture(texture);

    sf::Clock deltaClock;

    SoftwareRenderer renderer(SCREEN_WIDTH, SCREEN_HEIGHT);

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
        renderer.Render(teapotData);


        auto buf = renderer.GetScreenBuffer();

              uint32_t* dst = (uint32_t*)buf.data();
        const uint32_t* src = (uint32_t*)renderer.GetScreenBuffer().data();

        for (int y = 0; y < SCREEN_HEIGHT; ++y)
        {
            memcpy(dst + y * SCREEN_WIDTH, src + (SCREEN_HEIGHT - 1 - y) * SCREEN_WIDTH, SCREEN_WIDTH * 4);
        }
        // update texture
        texture.update((uint8_t*)dst);

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