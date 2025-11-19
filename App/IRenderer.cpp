/*
* Master’s thesis - Analysis of selected optimization techniques for a 3D software renderer
* Author: Wiktoria Marczyk
* Year: 2025
*/

#include "IRenderer.h"
#include "SoftwareRenderer.h"
#include "GlRenderer.h"

shared_ptr<IRenderer> RendererFactory::CreateRenderer(eRendererType rendererType, int screenWidth, int screenHeight)
{
    switch (rendererType)
    {
    case eRendererType::Software:
        return std::make_shared<SoftwareRenderer>(screenWidth, screenHeight);
    case eRendererType::Hardware:
        return std::make_shared<GlRenderer>(screenWidth, screenHeight);
    default:
        return nullptr;
    }
}
