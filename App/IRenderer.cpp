#include "IRenderer.h"
#include "SoftwareRenderer.h"
#include "GlRenderer.h"

shared_ptr<IRenderer> IRenderer::CreateRenderer(eRendererType rendererType, int screenWidth, int screenHeight)
{
    switch (rendererType)
    {
    case eRendererType::SOFTWARE:
        return std::make_shared<SoftwareRenderer>(screenWidth, screenHeight);
    case eRendererType::HARDWARE:
        return std::make_shared<GlRenderer>(screenWidth, screenHeight);
    default:
        return nullptr;
    }
}
