/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2023
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
