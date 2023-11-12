#include "Common.h"
#include "Renderer.h"

#include "teapot.h"


int main()
{
    vector<Vector3f> TeapotData(teapot_count/3);

    memcpy( TeapotData.data() , teapot , sizeof(teapot) );

    // create the window
    sf::RenderWindow window(sf::VideoMode(800, 600), "My window");

    window.setFramerateLimit(60);

    ImGui::SFML::Init(window);

    sf::Texture texture;
    texture.create( 800, 600 );

    sf::Sprite sprite;
    sprite.setTexture(texture);

    sf::Clock deltaClock;

    SoftwareRenderer renderer(800, 600);

    vector<Vector3f> data = {{-1,  1, -1},
                             { 1,  1, -1},
                             { 1, -1, -1},

                             { 1, -1, -1},
                             {-1,  1, -1},
                             {-1, -1, -1}};

    Matrix4f cameraMatrix = Matrix4f::CreateLookAtMatrix(Vector3f(0, 0, -10), Vector3f(0, 0, 1), Vector3f(0, 1, 0));
    Matrix4f projectionMatrix = Matrix4f::CreateProjectionMatrix(90, 800.0f / 600.0f, 0.1f, 1000.0f);
    Matrix4f modelMatrix = Matrix4f::Identity();
    float rotation = 0;

    //for (auto& p : data)
    //    p.z += 5;



    // run the program as long as the window is open
    while (window.isOpen())
    {
        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event))
        {
            ImGui::SFML::ProcessEvent(window, event);

            // "close requested" event: we close the window
            if (event.type == sf::Event::Closed)
                window.close();
        }

        ++rotation;
        //modelMatrix = Matrix4f::Rotation(Vector3f(0, 0, (rotation / 180.f) * std::numbers::pi));

        float angle = (rotation / 180.f) * std::numbers::pi;

        modelMatrix = Matrix4f::Rotation(Vector3f(0, angle, angle));
        renderer.SetModelMatrixx(modelMatrix);
        renderer.SetViewMatrix(cameraMatrix);
        renderer.SetProjectionMatrix(projectionMatrix);

        // update UI
        ImGui::SFML::Update(window, deltaClock.restart());
        renderer.UpdateUI();

        // render stuff to screen buffer
        renderer.Render(TeapotData);


        auto buf = renderer.GetScreenBuffer();

              uint32_t* dst = (uint32_t*)buf.data();
        const uint32_t* src = (uint32_t*)renderer.GetScreenBuffer().data();

        for (int y = 0; y < 600; ++y)
        {
            memcpy(dst + y * 800, src + (599 - y) * 800, 800 * 4);
        }
        // update texture
        texture.update((uint8_t*)dst);

        // render texture to screen
        window.draw(sprite);

        // render UI on top
        ImGui::SFML::Render(window);

        // end the current frame
        window.display();
    }
    ImGui::SFML::Shutdown();

    return 0;
}