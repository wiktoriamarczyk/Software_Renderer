/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "IRenderer.h"
#include "Math.h"

/**
* Enum okreœlaj¹cy typ jednostki cieniuj¹cej.
*/
enum class ShaderType { Vertex, Fragment, Count };
/**
* Enum okreœlaj¹cy sta³¹ jednostki cieniuj¹cej.
*/
enum class UniformType { TransformPVM, World, LightPos, LightAmbientColor, LightDiffuseColor, AmbientStrength, DiffuseStrength, SpecularStrength, Shininess, CameraPos, WireframeColor, Count };
/**
* Enum okreœlaj¹cy typ atrybutu wierzcho³ka.
*/
enum class VertexAttribute { Position, Normal, Color, TexCoord, Count };

/**
* Klasa tekstury wykorzystanej przez sprzêtowy renderer OpenGL. Implementuje interfejs ITexture.
*/
class GlTexture : public ITexture
{
public:
    GlTexture()=default;
    /**
    * Tworzy teksturê 4x4 wype³nion¹ bia³ym kolorem.
    * @return true jeœli tekstura zosta³a utworzona, false w przeciwnym wypadku.
    */
    bool CreateWhite4x4Tex();
    /**
    * £aduje teksturê z pliku.
    * @param fileName œcie¿ka do pliku.
    * @return true jeœli tekstura zosta³a za³adowana, false w przeciwnym wypadku.
    */
    bool Load(const char* fileName);
    /**
    * Ustawia teksturê jako aktualn¹.
    * @return true jeœli tekstura zosta³a ustawiona, false w przeciwnym wypadku.
    */
    bool Bind()const;
    /**
    * Sprawdza czy tekstura jest poprawna.
    * @return true jeœli tekstura jest poprawna, false w przeciwnym wypadku.
    */
    virtual bool IsValid()const;
    /**
    * Ustawia bie¿¹c¹ teksturê na null.
    */
    static void Unbind();
private:
    sf::Texture m_Texture; ///< tekstura
    bool m_Loaded = false; ///< czy tekstura zosta³a za³adowana
};

/**
* Klasa shadera wykorzystanego przez sprzêtowy renderer OpenGL.
*/
class GlProgram
{
public:
    GlProgram()=default;
    ~GlProgram();
    /**
    * £aduje shader z ³ancucha znaków.
    * @param shaderData ³ancuch znaków zawieraj¹cy kod shadera
    * @param type typ shadera
    */
    bool LoadShaderFromMemory(const std::string& shaderData, ShaderType type);
    /**
    * £aduje macierz.
    * @param mat macierz
    * @param uniform typ uniformu
    * @return true jeœli macierz zosta³a za³adowana, false w przeciwnym wypadku
    */
    bool LoadMatrix(const Matrix4f& mat, UniformType uniform);
    /**
    * £aduje wektor.
    * @param vec wektor.
    * @param uniform typ sta³ej shadera
    * @return true jeœli wektor zosta³ za³adowany, false w przeciwnym wypadku
    */
    bool LoadVector(const Vector3f& vec, UniformType uniform);
    /**
    * £aduje wektor.
    * @param vec wektor.
    * @param uniform typ sta³ej shadera
    * @return true jeœli wektor zosta³ za³adowany, false w przeciwnym wypadku
    */
    bool LoadVector(const Vector4f& vec, UniformType uniform);
    /**
    * £aduje liczbê zmiennoprzecinkow¹.
    * @param vec liczba zmiennoprzecinkowa
    * @param uniform typ sta³ej shadera.
    * @return true jeœli liczba zosta³a za³adowana, false w przeciwnym wypadku
    */
    bool LoadFloat(float val, UniformType uniform);
    /**
    * Ustawia program jako aktualny.
    * @return true jeœli program zosta³ ustawiony, false w przeciwnym wypadku
    */
    bool Bind()const;
    /**
    * Ustawia bie¿¹cy program jako null.
    */
    static void Unbind();
private:
    uint32_t m_Program = 0; ///< program
    uint32_t m_Shader [ static_cast<uint32_t>(ShaderType::Count)  ] = { 0 }; ///< shadery
    int      m_Uniform[ static_cast<uint32_t>(UniformType::Count) ] = { 0 }; ///< uniformy
};

/**
* Klasa bufora wierzcho³ków wykorzystanego przez sprzêtowy renderer OpenGL.
*/
class GlVertexBuffer
{
public:
    GlVertexBuffer();
    ~GlVertexBuffer();
    /**
    * £aduje wierzcho³ki do bufora.
    * @param vertices wektor wierzcho³ków
    * @return true, jeœli wierzcho³ki zosta³y za³adowane, false w przeciwnym wypadku
    */
    bool Load(const vector<Vertex>& vertices);
    /**
    * Ustawia bie¿¹cy bufor jako aktualny.
    */
    void Bind()const;
    /**
    * Ustawia bie¿¹cy bufor jako null.
    */
    static void Unbind();
private:
    uint32_t m_VertexBufferObject = 0; ///< bufor wierzcho³ków
    uint32_t m_VertexArrayObject = 0; ///< obiekt reprezentuj¹cy parametry wierzcho³ków

    uint32_t m_BufferCapacity = 0; ///< pojemnoœæ bufora
    uint32_t m_BufferSize = 0; ///< rozmiar bufora
};

/**
* Klasa renderera OpenGL. Implementuje interfejs IRenderer.
*/
class GlRenderer : public IRenderer
{
public:
    /**
    * Konstruktor klasy.
    * @param screenWidth szerokoœæ ekranu
    * @param screenHeight wysokoœæ ekranu
    */
    GlRenderer(int screenWidth, int screenHeight);
    ~GlRenderer();

    virtual shared_ptr<ITexture> LoadTexture(const char* fileName)const override;

    virtual void ClearScreen() override;
    virtual void ClearZBuffer() override;
    virtual void BeginFrame() override;
    virtual void Render(const vector<Vertex>& vertices) override;
    virtual void EndFrame() override;
    virtual void RenderDepthBuffer()override;
    virtual const vector<uint32_t>& GetScreenBuffer() const override;
    virtual const DrawStats& GetDrawStats() const override;
    virtual shared_ptr<ITexture> GetDefaultTexture() const override;

    virtual void SetModelMatrix(const Matrix4f& other)override;
    virtual void SetViewMatrix(const Matrix4f& other)override;
    virtual void SetProjectionMatrix(const Matrix4f& other)override;
    virtual void SetTexture(shared_ptr<ITexture> texture)override;

    virtual void SetWireFrameColor(const Vector4f& wireFrameColor)override;
    virtual void SetClearColor(const Vector4f& clearColor)override;
    virtual void SetDiffuseColor(const Vector3f& diffuseColor)override;
    virtual void SetAmbientColor(const Vector3f& ambientColor)override;
    virtual void SetLightPosition(const Vector3f& lightPosition)override;
    virtual void SetDiffuseStrength(float diffuseStrength)override;
    virtual void SetAmbientStrength(float ambientStrength)override;
    virtual void SetSpecularStrength(float specularStrength)override;
    virtual void SetShininess(float shininess)override;
    virtual void SetThreadsCount(uint8_t threadsCount)override;
    virtual void SetColorizeThreads(bool colorizeThreads)override;
    virtual void SetDrawWireframe(bool wireframe)override;
    virtual void SetDrawBBoxes(bool drawBBoxes)override;
    virtual void SetZWrite(bool zwrite)override;
    virtual void SetZTest(bool ztest)override;
private:
    int                         m_ScreenWidth = 0; ///< szerokoœæ ekranu
    int                         m_ScreenHeight = 0; ///< wysokoœæ ekranu

    unique_ptr<GlProgram>       m_DefaultProgram; ///< domyœlny program
    unique_ptr<GlProgram>       m_LineProgram; ///< program rysuj¹cy linie
    unique_ptr<GlProgram>       m_WireframeProgram; ///< program rysuj¹cy siatkê
    unique_ptr<GlVertexBuffer>  m_DefaultVertexBuffer; ///< domyœlny bufor wierzcho³ków

    Vector4f                    m_WireFrameColor = Vector4f(1, 1, 1, 1); ///< kolor siatki
    Vector3f                    m_DiffuseColor = Vector3f(1, 1, 1); ///< kolor œwiat³a rozproszenia
    Vector3f                    m_AmbientColor = Vector3f(1, 1, 1); ///< kolor œwiat³a otoczenia
    Vector3f                    m_LightPosition = Vector3f(0, 0, -20); ///< pozycja œwiat³a
    Vector3f                    m_CameraPosition = Vector3f(0, 0, 0); ///< pozycja kamery
    Vector4f                    m_ThreadColors[12]; ///< kolory w¹tków
    Vector4f                    m_ClearColor = Vector4f(0, 0, 0, 1); ///< kolor t³a

    Matrix4f                    m_ModelMatrix; ///< macierz modelu
    Matrix4f                    m_ViewMatrix; ///< macierz widoku
    Matrix4f                    m_ProjectionMatrix; ///< macierz projekcji
    Matrix4f                    m_MVPMatrix; ///< macierz MVP

    bool                        m_DrawWireframe = false; ///< czy rysowaæ siatkê
    bool                        m_ZWrite = false; ///< czy zapisywaæ do bufora g³êbokoœci
    bool                        m_ZTest = false; ///< czy testowaæ bufor g³êbokoœci
    float                       m_DiffuseStrength = 0.3f; ///< si³a œwiat³a rozproszenia
    float                       m_AmbientStrength = 0.5f; ///< si³a œwiat³a otoczenia
    float                       m_SpecularStrength = 0.9f; ///< si³a œwiat³a odbitego
    float                       m_Shininess = 32.0f; ///< moc œwiat³a odbitego
    int                         m_DefaultSFMLProgram = 0; ///< domyœlny program SFML

    shared_ptr<GlTexture>       m_Texture; ///< tekstura
    shared_ptr<GlTexture>       m_DefaultTexture; ///< domyœlna tekstura
};