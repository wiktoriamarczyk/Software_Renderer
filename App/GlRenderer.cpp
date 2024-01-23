/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#include "GlRenderer.h"
#include <iostream>


#define WIN32_LEAN_AND_MEAN
#define SOGL_MAJOR_VERSION 4
#define SOGL_MINOR_VERSION 0
#define SOGL_IMPLEMENTATION_WIN32

#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>

namespace SOGL
{

#include "simple-gl-loader.h"

}

using namespace SOGL;


namespace
{

void checkError(uint32_t l_shader, uint32_t l_flag, bool l_program, const std::string& l_errorMsg)
{
    int success = 0;
    char error[1024] = { 0 };
    if (l_program)
        glGetProgramiv(l_shader, l_flag, &success);
    else
        glGetShaderiv(l_shader, l_flag, &success);

    if (success)
        return;

    if (l_program)
        glGetProgramInfoLog(l_shader, sizeof(error), nullptr, error);
    else
        glGetShaderInfoLog(l_shader, sizeof(error), nullptr, error);

    std::cout << "Error: '" << error << "'" << std::endl;

}

///Creates and compiles a shader.
GLuint buildShader(const std::string& l_src, unsigned int l_type)
{
    GLuint shaderID = glCreateShader(l_type);
    if (!shaderID) {
        std::cout << "Bad shader type!";
        return 0;
    }
    const GLchar* sources[1];
    GLint lengths[1];
    sources[0] = l_src.c_str();
    lengths[0] = l_src.length();
    glShaderSource(shaderID, 1, sources, lengths);
    glCompileShader(shaderID);
    checkError(shaderID, GL_COMPILE_STATUS, false, "Shader compile error: ");
    return shaderID;
}

}

GlProgram::~GlProgram()
{
    for (uint32_t i = 0; i < static_cast<uint32_t>(ShaderType::Count); ++i)
    {
        if (m_Shader[i])
        {
            glDetachShader(m_Program, m_Shader[i]);
            glDeleteShader(m_Shader[i]);
        }
    }
    if (m_Program)
        glDeleteProgram(m_Program);
}

bool GlProgram::LoadShaderFromMemory(const std::string& shaderData, ShaderType type)
{
    if (shaderData.empty())
        return false;

    uint32_t& shaderID = m_Shader[ static_cast<uint32_t>(type) ];

    if (m_Program && shaderID)
    {
        glDetachShader(m_Program, shaderID);
        glDeleteShader(shaderID);
        shaderID = 0;
    }


    switch (type)
    {
    case ShaderType::Vertex:
        shaderID = buildShader(shaderData, GL_VERTEX_SHADER);
        break;
    case ShaderType::Fragment:
        shaderID = buildShader(shaderData, GL_FRAGMENT_SHADER);
        break;
    default:
        break;
    }

    if (!shaderID)
        return false;

    if (m_Program == 0)
    {
        m_Program = glCreateProgram();
    }

    glAttachShader(m_Program, shaderID);
    glBindAttribLocation(m_Program, static_cast<GLuint>(VertexAttribute::Position), "vs_position");
    glBindAttribLocation(m_Program, static_cast<GLuint>(VertexAttribute::Normal  ), "vs_normal");
    glBindAttribLocation(m_Program, static_cast<GLuint>(VertexAttribute::Color   ), "vs_color");
    glBindAttribLocation(m_Program, static_cast<GLuint>(VertexAttribute::TexCoord), "vs_uv");

    glLinkProgram(m_Program);
    checkError(m_Program, GL_LINK_STATUS, true, "Shader link error:");
    glValidateProgram(m_Program);
    checkError(m_Program, GL_VALIDATE_STATUS, true, "Invalid shader:");

    m_Uniform[uint32_t(UniformType::TransformPVM        )] = glGetUniformLocation(m_Program, "g_MatPVM");
    m_Uniform[uint32_t(UniformType::World               )] = glGetUniformLocation(m_Program, "g_MatW");
    m_Uniform[uint32_t(UniformType::LightPos            )] = glGetUniformLocation(m_Program, "g_LightPos");
    m_Uniform[uint32_t(UniformType::LightAmbientColor   )] = glGetUniformLocation(m_Program, "g_LightAmbientColor");
    m_Uniform[uint32_t(UniformType::LightDiffuseColor   )] = glGetUniformLocation(m_Program, "g_LightDiffuseColor");
    m_Uniform[uint32_t(UniformType::AmbientStrength     )] = glGetUniformLocation(m_Program, "g_LightAmbientStrength");
    m_Uniform[uint32_t(UniformType::DiffuseStrength     )] = glGetUniformLocation(m_Program, "g_LightDiffuseStrength");
    m_Uniform[uint32_t(UniformType::SpecularStrength    )] = glGetUniformLocation(m_Program, "g_SpecularStrength");
    m_Uniform[uint32_t(UniformType::CameraPos           )] = glGetUniformLocation(m_Program, "g_CameraPos");
    m_Uniform[uint32_t(UniformType::Shininess           )] = glGetUniformLocation(m_Program, "g_Shininess");
    m_Uniform[uint32_t(UniformType::WireframeColor      )] = glGetUniformLocation(m_Program, "g_WireframeColor");

    return true;
}

bool GlProgram::LoadMatrix(const Matrix4f& mat, UniformType uniformType)
{
    if (!m_Program)
        return false;

    auto uniform = (int)(m_Uniform[uint32_t(uniformType)]);
    if (uniform == -1)
        return false;

    glUniformMatrix4fv( uniform , 1, GL_FALSE, mat.m_Matrix[0] );
    return true;
}

bool GlProgram::LoadVector(const Vector4f& vec , UniformType uniform)
{
    if (!m_Program)
        return false;

    auto uniformID = (int)(m_Uniform[uint32_t(uniform)]);
    if (uniformID == -1)
        return false;

    glUniform4fv( uniformID , 1, &vec.x );
    return true;
}

bool GlProgram::LoadVector(const Vector3f& vec , UniformType uniform)
{
    if (!m_Program)
        return false;

    auto uniformID = (int)(m_Uniform[uint32_t(uniform)]);
    if (uniformID == -1)
        return false;

    glUniform3fv( uniformID , 1, &vec.x );
    return true;
}

bool GlProgram::LoadFloat(float val, UniformType uniform)
{
    if (!m_Program)
        return false;

    auto uniformID = (int)(m_Uniform[uint32_t(uniform)]);
    if (uniformID == -1)
        return false;

    glUniform1f( uniformID , val );
    return true;
}

bool GlProgram::Bind() const
{
    if (!m_Program)
        return false;
    glUseProgram(m_Program);
    return true;
}

void GlProgram::Unbind()
{
    glUseProgram(0);
}

void* ToGlOffset( uint32_t l_offset )
{
    return reinterpret_cast<char*>(0)+l_offset;
}

GlVertexBuffer::GlVertexBuffer()
{
    glGenVertexArrays(1, &m_VertexArrayObject);
    glBindVertexArray(m_VertexArrayObject);

    m_BufferCapacity = 1024*10;

    glGenBuffers(1, &m_VertexBufferObject);
    glBindBuffer(GL_ARRAY_BUFFER, m_VertexBufferObject);
    glBufferData(GL_ARRAY_BUFFER, Vertex::Stride*m_BufferCapacity, nullptr, GL_DYNAMIC_DRAW);


    glEnableVertexAttribArray(uint32_t(VertexAttribute::Position));
    glVertexAttribPointer(uint32_t(VertexAttribute::Position), 3, GL_FLOAT, GL_FALSE, Vertex::Stride, ToGlOffset(Vertex::PositionOffset));

    glEnableVertexAttribArray(int32_t(VertexAttribute::Normal));
    glVertexAttribPointer(uint32_t(VertexAttribute::Normal), 3, GL_FLOAT, GL_TRUE, Vertex::Stride, ToGlOffset(Vertex::NormalOffset));

    glEnableVertexAttribArray(uint32_t(VertexAttribute::Color));
    glVertexAttribPointer(uint32_t(VertexAttribute::Color), 4, GL_FLOAT, GL_FALSE, Vertex::Stride, ToGlOffset(Vertex::ColorOffset));

    glEnableVertexAttribArray(uint32_t(VertexAttribute::TexCoord));
    glVertexAttribPointer(uint32_t(VertexAttribute::TexCoord), 2, GL_FLOAT, GL_FALSE, Vertex::Stride, ToGlOffset(Vertex::UVOffset));

    //Make sure to bind the vertex array to null if you wish to define more objects.
    glBindVertexArray(0);
}

GlVertexBuffer::~GlVertexBuffer()
{
    glDeleteBuffers(1, &m_VertexBufferObject);
    glDeleteVertexArrays(1, &m_VertexArrayObject);
}

bool GlVertexBuffer::Load(const vector<Vertex>& vertices)
{
    if (vertices.empty())
        return false;

    glBindBuffer(GL_ARRAY_BUFFER, m_VertexBufferObject);

    if (vertices.size() > m_BufferCapacity)
    {
        m_BufferCapacity = vertices.size();
        glBufferData(GL_ARRAY_BUFFER, Vertex::Stride*m_BufferCapacity, nullptr, GL_DYNAMIC_DRAW);
    }

    m_BufferSize = vertices.size();
    glBufferSubData(GL_ARRAY_BUFFER, 0, Vertex::Stride*m_BufferSize, vertices.data());
    return true;
}

void GlVertexBuffer::Bind() const
{
    glBindVertexArray(m_VertexArrayObject);
}

void GlVertexBuffer::Unbind()
{
    glBindVertexArray(0);
}

bool GlTexture::CreateWhite4x4Tex()
{
    sf::Image image;
    image.create(4, 4, sf::Color::White);
    m_Texture.loadFromImage(image);
    m_Loaded = true;
    return true;
}

bool GlTexture::Load(const char* fileName)
{
    sf::Texture texture;
    if (!texture.loadFromFile(fileName))
        return false;

    texture.setSmooth(false);

    m_Texture.swap( texture );
    m_Loaded = true;
    return true;
}

bool GlTexture::Bind() const
{
    if (!m_Loaded)
        return false;

    sf::Texture::bind(&m_Texture);
    return true;
}

bool GlTexture::IsValid() const
{
    return m_Loaded;
}

void GlTexture::Unbind()
{
    glBindTexture( GL_TEXTURE_2D, 0 );
}

namespace ShadersCode
{
const std::string defaultVertexShader = R"(
#version 330
attribute vec3 vs_position;
attribute vec3 vs_normal;
attribute vec4 vs_color;
attribute vec2 vs_uv;

uniform mat4 g_MatW;
uniform mat4 g_MatPVM;

varying vec3 ps_position;
varying vec3 ps_normal;
varying vec4 ps_color;
varying vec2 ps_uv;

void main() {
    gl_Position = g_MatPVM * vec4(vs_position, 1.0);
    ps_position = (g_MatW * vec4(vs_position,1.0)).xyz;
    ps_normal = ((g_MatW * vec4(vs_normal,1.0f)) - (g_MatW * vec4(0,0,0,1))).xyz;
    ps_color = vs_color;
    ps_uv = vs_uv;
}
)";


const std::string defaultFragShader = R"(
#version 330

uniform sampler2D texture;
uniform vec3 g_LightPos;
uniform vec3 g_LightAmbientColor;
uniform vec3 g_LightDiffuseColor;
uniform vec3 g_CameraPos;
uniform float g_LightAmbientStrength;
uniform float g_LightDiffuseStrength;
uniform float g_SpecularStrength;
uniform float g_Shininess;

varying vec3 ps_position;
varying vec3 ps_normal;
varying vec4 ps_color;
varying vec2 ps_uv;

void main() {
    vec4 sampledPixel = texture2D(texture, ps_uv);
    vec3 normal = normalize(ps_normal);

    vec3 pointToLightDir = normalize(ps_position-g_LightPos);
    vec3 ambientLight = g_LightAmbientColor * g_LightAmbientStrength;

    float diffuseFactor = max(dot(pointToLightDir * -1.0,normal), 0.0);
    vec3 diffuseLight = g_LightDiffuseColor * diffuseFactor * g_LightDiffuseStrength;

    vec3 viewDir = normalize(g_CameraPos - ps_position);
    vec3 reflectDir = reflect(pointToLightDir,normal);

    float specularFactor = pow(max(dot(viewDir,reflectDir), 0.0), g_Shininess);
    vec3 specularLight = g_LightDiffuseColor * g_SpecularStrength * specularFactor;

    vec3 sumOfLight = ambientLight + diffuseLight + specularLight;
    sumOfLight = min(sumOfLight,vec3(1.0, 1.0, 1.0));

    gl_FragColor = vec4(sumOfLight,1) * sampledPixel * ps_color;
}
)";


const std::string lineVertexShader = R"(
#version 330
attribute vec3 vs_position;
attribute vec4 vs_color;

uniform mat4 g_MatPVM;

varying vec4 ps_color;

void main() {
    gl_Position = g_MatPVM * vec4(vs_position, 1.0);
    ps_color = vs_color;
}
)";

const std::string lineFragShader = R"(
#version 330

varying vec4 ps_color;

void main() {
    gl_FragColor = ps_color;
}
)";

const std::string wireframeVertexShader = R"(
#version 330
attribute vec3 vs_position;
uniform mat4 g_MatPVM;


void main() {
    gl_Position = g_MatPVM * vec4(vs_position, 1.0);
}
)";

const std::string wireframeFragShader = R"(
#version 330

uniform vec3 g_WireframeColor;

void main() {
    gl_FragColor = vec4(g_WireframeColor,1);
}
)";
}

GlRenderer::GlRenderer(int screenWidth, int screenHeight)
    : m_ScreenWidth(screenWidth)
    , m_ScreenHeight(screenHeight)
{

    if (!sogl_loadOpenGL())
    {
        const char **failures = sogl_getFailures();
        while (*failures)
        {
            char debugMessage[256];
            snprintf(debugMessage, 256, "SOGL WIN32 EXAMPLE: Failed to load function %s\n", *failures);
            OutputDebugStringA(debugMessage);
            failures++;
        }
        return;
    }

    glGetIntegerv(GL_CURRENT_PROGRAM,&m_DefaultSFMLProgram);

    m_DefaultVertexBuffer = make_unique<GlVertexBuffer>();

    m_DefaultProgram = std::make_unique<GlProgram>();
    m_DefaultProgram->LoadShaderFromMemory(ShadersCode::defaultVertexShader, ShaderType::Vertex);
    m_DefaultProgram->LoadShaderFromMemory(ShadersCode::defaultFragShader  , ShaderType::Fragment);

    m_LineProgram = std::make_unique<GlProgram>();
    m_LineProgram->LoadShaderFromMemory(ShadersCode::lineVertexShader, ShaderType::Vertex);
    m_LineProgram->LoadShaderFromMemory(ShadersCode::lineFragShader  , ShaderType::Fragment);

    m_WireframeProgram = std::make_unique<GlProgram>();
    m_WireframeProgram->LoadShaderFromMemory(ShadersCode::wireframeVertexShader, ShaderType::Vertex);
    m_WireframeProgram->LoadShaderFromMemory(ShadersCode::wireframeFragShader  , ShaderType::Fragment);

    m_DefaultProgram->Bind();

    m_DefaultTexture = std::make_shared<GlTexture>();
    m_DefaultTexture->CreateWhite4x4Tex();
}

GlRenderer::~GlRenderer()
{
    m_DefaultProgram.reset();
    m_LineProgram.reset();
    m_WireframeProgram.reset();
    m_DefaultVertexBuffer.reset();
    m_Texture.reset();

    GlTexture::Unbind();
    GlVertexBuffer::Unbind();
    GlProgram::Unbind();

    glUseProgram(m_DefaultSFMLProgram);
}

std::shared_ptr<ITexture> GlRenderer::LoadTexture(const char* fileName) const
{
    if (!fileName || fileName[0]==0)
        return m_DefaultTexture;
    auto texture = std::make_shared<GlTexture>();
    if (!texture->Load(fileName))
        return nullptr;
    return texture;
}

void GlRenderer::ClearScreen()
{
    // Configure the viewport (the same size as the window)
    glViewport(0, 0, m_ScreenWidth, m_ScreenHeight);

    glClearColor(m_ClearColor.x, m_ClearColor.y, m_ClearColor.z, m_ClearColor.w);
    // Clear the depth buffer
    glClear(GL_COLOR_BUFFER_BIT );
}

void GlRenderer::ClearZBuffer()
{
    glClear(GL_DEPTH_BUFFER_BIT);
}

void GlRenderer::BeginFrame()
{

}

void GlRenderer::Render(const vector<Vertex>& vertices)
{
    auto& pCurProgram = m_DrawWireframe ? m_WireframeProgram : m_DefaultProgram;
    pCurProgram->Bind();
    pCurProgram->LoadMatrix( m_MVPMatrix            , UniformType::TransformPVM );
    pCurProgram->LoadMatrix( m_ModelMatrix          , UniformType::World );
    pCurProgram->LoadVector( m_LightPosition        , UniformType::LightPos );
    pCurProgram->LoadVector( m_AmbientColor         , UniformType::LightAmbientColor );
    pCurProgram->LoadVector( m_DiffuseColor         , UniformType::LightDiffuseColor );
    pCurProgram->LoadVector( m_CameraPosition       , UniformType::CameraPos );
    pCurProgram->LoadFloat ( m_AmbientStrength      , UniformType::AmbientStrength );
    pCurProgram->LoadFloat ( m_DiffuseStrength      , UniformType::DiffuseStrength );
    pCurProgram->LoadFloat ( m_SpecularStrength     , UniformType::SpecularStrength );
    pCurProgram->LoadFloat ( m_Shininess            , UniformType::Shininess );
    pCurProgram->LoadVector( m_WireFrameColor.xyz() , UniformType::WireframeColor );


    m_DefaultVertexBuffer->Bind();
    m_DefaultVertexBuffer->Load(vertices);

    m_Texture->Bind();

    if (m_ZTest && !m_DrawWireframe)
        glEnable(GL_DEPTH_TEST);
    else
        glDisable(GL_DEPTH_TEST);

    if (m_ZWrite && !m_DrawWireframe)
        glDepthMask(GL_TRUE);
    else
        glDepthMask(GL_FALSE);

    glCullFace(GL_BACK);

    if (m_DrawWireframe)
    {
        glDisable(GL_CULL_FACE);
        glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
        glDrawArrays( GL_TRIANGLES , 0 , vertices.size() );
        glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
        glEnable(GL_CULL_FACE);
    }
    else
    {
        glEnable(GL_CULL_FACE);
        glDrawArrays( GL_TRIANGLES , 0 , vertices.size() );
    }

    GlVertexBuffer::Unbind();
    GlProgram::Unbind();
    GlTexture::Unbind();
}

void GlRenderer::EndFrame()
{

}

void GlRenderer::RenderDepthBuffer()
{
}

void GlRenderer::SetModelMatrix(const Matrix4f& other)
{
    m_ModelMatrix = other;
    m_MVPMatrix = m_ModelMatrix * m_ViewMatrix * m_ProjectionMatrix;
}

void GlRenderer::SetViewMatrix(const Matrix4f& other)
{
    m_ViewMatrix = other;
    m_MVPMatrix = m_ModelMatrix * m_ViewMatrix * m_ProjectionMatrix;

    Matrix4f inversedViewMatrix = m_ViewMatrix.Inversed();
    m_CameraPosition = Vector3f(inversedViewMatrix[12], inversedViewMatrix[13], inversedViewMatrix[14]);
}

void GlRenderer::SetProjectionMatrix(const Matrix4f& other)
{
    m_ProjectionMatrix = other;
    m_MVPMatrix = m_ModelMatrix * m_ViewMatrix * m_ProjectionMatrix;
}

void GlRenderer::SetTexture(shared_ptr<ITexture> texture)
{
    m_Texture = std::dynamic_pointer_cast<GlTexture>(texture);
    if (!m_Texture)
        m_Texture = m_DefaultTexture;
}

const std::vector<uint32_t>& GlRenderer::GetScreenBuffer() const
{
    static std::vector<uint32_t> None;
    return None;
}

const DrawStats& GlRenderer::GetDrawStats() const
{
    static const DrawStats DrawStats;
    return DrawStats;
}

shared_ptr<ITexture> GlRenderer::GetDefaultTexture() const
{
    return m_DefaultTexture;
}

void GlRenderer::SetWireFrameColor(const Vector4f& wireFrameColor)
{
    m_WireFrameColor = wireFrameColor;
}

void GlRenderer::SetClearColor(const Vector4f& clearColor)
{
    m_ClearColor = clearColor;
}

void GlRenderer::SetDiffuseColor(const Vector3f& diffuseColor)
{
    m_DiffuseColor = diffuseColor;
}

void GlRenderer::SetAmbientColor(const Vector3f& ambientColor)
{
    m_AmbientColor = ambientColor;
}

void GlRenderer::SetLightPosition(const Vector3f& lightPosition)
{
    m_LightPosition = lightPosition;
}

void GlRenderer::SetDiffuseStrength(float diffuseStrength)
{
    m_DiffuseStrength = diffuseStrength;
}

void GlRenderer::SetAmbientStrength(float ambientStrength)
{
    m_AmbientStrength = ambientStrength;
}

void GlRenderer::SetSpecularStrength(float specularStrength)
{
    m_SpecularStrength = specularStrength;
}

void GlRenderer::SetShininess(float shininess)
{
    m_Shininess = shininess;
}

void GlRenderer::SetThreadsCount(uint8_t threadsCount)
{
    // nothing
}

void GlRenderer::SetColorizeThreads(bool colorizeThreads)
{
    // nothing
}

void GlRenderer::SetDrawWireframe(bool Wireframe)
{
    m_DrawWireframe = Wireframe;
}

void GlRenderer::SetDrawBBoxes(bool drawBBoxes)
{
    // nothing
}

void GlRenderer::SetZWrite(bool zwrite)
{
    m_ZWrite = zwrite;
}

void GlRenderer::SetZTest(bool ztest)
{
    m_ZTest = ztest;
}
