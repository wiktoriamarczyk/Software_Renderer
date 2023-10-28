#include "Common.h"
#include "Renderer.h"


int main()
{
    // create the window
    sf::RenderWindow window(sf::VideoMode(800, 600), "My window");

    window.setFramerateLimit(60);

    ImGui::SFML::Init(window);

    sf::Texture texture;
    texture.create( 800, 600 );

    sf::Sprite sprite;
    sprite.setTexture(texture);

    sf::Clock deltaClock;

    SoftwareRenderer Renderer(800, 600);

    vector<Vector3f> Data = { { 10  , 10  , 0  }
                                , { 700 , 10  , 0 }
                                , { 400 , 500 , 0 }};


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

        // update UI
        ImGui::SFML::Update(window, deltaClock.restart());
        Renderer.UpdateUI();

        // render stuff to screen buffer
        Renderer.Render( Data );


        auto buf = Renderer.GetScreenBuffer();

              uint32_t* dst = (uint32_t*)buf.data();
        const uint32_t* src = (uint32_t*)Renderer.GetScreenBuffer().data();

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