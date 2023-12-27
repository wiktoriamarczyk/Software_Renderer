#include "IRenderer.h"
#include "SoftwareRenderer.h"

shared_ptr<IRenderer> IRenderer::CreateRenderer(eRendererType rendererType, int screenWidth, int screenHeight)
{
    switch (rendererType)
    {
    case eRendererType::SOFTWARE:
        return std::make_shared<SoftwareRenderer>(screenWidth, screenHeight);
    default:
        return nullptr;
    }
}
