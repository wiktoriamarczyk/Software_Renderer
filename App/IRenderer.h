/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Common.h"
#include "Vector3f.h"
#include "Matrix4.h"
#include "Vector4f.h"
#include "Vector2f.h"
#include "Math.h"

enum class eRendererType : uint8_t
{
    Software,
    Hardware
};

class ITexture
{
public:
    virtual ~ITexture() = default;

    virtual bool IsValid()const=0;
};

class IRenderer;

class RendererFactory
{
public:
    static shared_ptr<IRenderer> CreateRenderer(eRendererType rendererType, int screenWidth, int screenHeight);
};

class IRenderer
{
public:
    virtual shared_ptr<ITexture> LoadTexture(const char* fileName)const=0;

    virtual void ClearScreen()=0;
    virtual void ClearZBuffer()=0;
    virtual void BeginFrame()=0;
    virtual void Render(const vector<Vertex>& vertices)=0;
    virtual void EndFrame()=0;
    virtual void RenderDepthBuffer()=0;
    virtual const vector<uint32_t>& GetScreenBuffer() const=0;
    virtual const DrawStats& GetDrawStats() const=0;
    virtual shared_ptr<ITexture> GetDefaultTexture() const=0;

    virtual void SetModelMatrix(const Matrix4f& other)=0;
    virtual void SetViewMatrix(const Matrix4f& other)=0;
    virtual void SetProjectionMatrix(const Matrix4f& other)=0;
    virtual void SetTexture(shared_ptr<ITexture> texture)=0;

    virtual void SetWireFrameColor(const Vector4f& wireFrameColor)=0;
    virtual void SetClearColor(const Vector4f& clearColor)=0;
    virtual void SetDiffuseColor(const Vector3f& diffuseColor)=0;
    virtual void SetAmbientColor(const Vector3f& ambientColor)=0;
    virtual void SetLightPosition(const Vector3f& lightPosition)=0;
    virtual void SetDiffuseStrength(float diffuseStrength)=0;
    virtual void SetAmbientStrength(float ambientStrength)=0;
    virtual void SetSpecularStrength(float specularStrength)=0;
    virtual void SetShininess(float shininess)=0;
    virtual void SetThreadsCount(uint8_t threadsCount)=0;
    virtual void SetColorizeThreads(bool colorizeThreads)=0;
    virtual void SetDrawWireframe(bool wireframe)=0;
    virtual void SetDrawBBoxes(bool drawBBoxes)=0;
    virtual void SetZWrite(bool zwrite)=0;
    virtual void SetZTest(bool ztest)=0;
};