/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2023
*/

#include "Application.h"

int main()
{
    Application app;
    if (!app.Initialize())
        return -1;
    return app.Run();
}