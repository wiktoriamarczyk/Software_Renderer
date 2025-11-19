/*
* Master’s thesis - Analysis of selected optimization techniques for a 3D software renderer
* Author: Wiktoria Marczyk
* Year: 2025
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