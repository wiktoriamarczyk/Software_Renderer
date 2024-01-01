/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2023
*/

#include "Application.h"

int main()
{
    ZoneScoped;
    Application app;
    if (!app.Initialize())
        return -1;
    return app.Run();
}