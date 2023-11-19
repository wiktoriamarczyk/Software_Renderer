#include "Common.h"
#include "Renderer.h"

#include "teapot.h"

vector<Vertex> GenerateTeapotVertices()
{
    int verticesCount = teapot_count / triangleVerticesCount;
    vector<Vertex> teapotData(verticesCount);

    // we should have normals in the teapot data, but they are missing, so we calculate them here in simplified way assuming that each vertex has the same normal
    for (int i = 0; i < verticesCount; ++i)
    {
        teapotData[i].position = Vector3f(teapot[i * triangleVerticesCount + 0], teapot[i * triangleVerticesCount + 1], teapot[i * triangleVerticesCount + 2]);
    }
    for (int i = 0; i < verticesCount; i += 3)
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

        float angleX = (renderer.GetRotation().x / 180.f * pi);
        float angleY = (renderer.GetRotation().y / 180.f * pi);
        float angleZ = (renderer.GetRotation().z / 180.f * pi);
        float scale = renderer.GetScale();

        modelMatrix = Matrix4f::Rotation(Vector3f(angleX, angleY, angleZ)) * Matrix4f::Scale(Vector3f(scale, scale, scale));

        renderer.SetModelMatrixx(modelMatrix);
        renderer.SetViewMatrix(cameraMatrix);
        renderer.SetProjectionMatrix(projectionMatrix);

        // update UI
        ImGui::SFML::Update(window, deltaClock.restart());
        renderer.UpdateUI();

        // render stuff to screen buffer
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