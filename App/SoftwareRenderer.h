/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "IRenderer.h"
#include "TransformedVertex.h"
#include "Texture.h"
#include "SimpleThreadPool.h"

/**
* Klasa renderera programowego. Implementuje interfejs IRenderer.
*/
class SoftwareRenderer : public IRenderer
{
public:
    /**
    * Konstruktor klasy.
    * @param screenWidth szerokoœæ ekranu
    * @param screenHeight wysokoœæ ekranu
    */
    SoftwareRenderer(int screenWidth, int screenHeight);

    shared_ptr<ITexture> LoadTexture(const char* fileName)const override;

    void ClearScreen()override;
    void ClearZBuffer()override;
    void BeginFrame()override;
    void Render(const vector<Vertex>& vertices)override;
    void EndFrame()override;
    void RenderDepthBuffer()override;
    const vector<uint32_t>& GetScreenBuffer() const override;
    const DrawStats& GetDrawStats() const override;
    shared_ptr<ITexture> GetDefaultTexture() const override;

    void SetModelMatrix(const Matrix4f& other)override;
    void SetViewMatrix(const Matrix4f& other)override;
    void SetProjectionMatrix(const Matrix4f& other)override;
    void SetTexture(shared_ptr<ITexture> texture)override;

    void SetWireFrameColor(const Vector4f& wireFrameColor)override;
    void SetClearColor(const Vector4f& clearColor)override;
    void SetDiffuseColor(const Vector3f& diffuseColor)override;
    void SetAmbientColor(const Vector3f& ambientColor)override;
    void SetLightPosition(const Vector3f& lightPosition)override;
    void SetDiffuseStrength(float diffuseStrength)override;
    void SetAmbientStrength(float ambientStrength)override;
    void SetSpecularStrength(float specularStrength)override;
    void SetShininess(float shininess)override;
    void SetThreadsCount(uint8_t threadsCount)override;
    void SetColorizeThreads(bool colorizeThreads)override;
    void SetDrawWireframe(bool wireframe)override;
    void SetDrawBBoxes(bool drawBBoxes)override;
    void SetZWrite(bool zWrite)override;
    void SetZTest(bool zTest)override;
private:
    /**
    * Rysuje wype³niony trójk¹t.
    * @param A przetransformowany wierzcho³ek A
    * @param B przetransformowany wierzcho³ek B
    * @param C przetransformowany wierzcho³ek C
    * @param color kolor trójk¹ta
    * @param minY minimalna wartoœæ Y
    * @param maxY maksymalna wartoœæ Y
    * @param stats statystyki rysowania
    */
    void DrawFilledTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int minY, int maxY, DrawStats& stats);
    /**
    * Rysuje trójk¹t.
    * @param A przetransformowany wierzcho³ek A
    * @param B przetransformowany wierzcho³ek B
    * @param C przetransformowany wierzcho³ek C
    * @param color kolor trójk¹ta
    * @param minY minimalna wartoœæ Y
    * @param maxY maksymalna wartoœæ Y
    */
    void DrawTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int minY, int maxY);
    /**
    * Rysuje bry³ê brzegow¹ trójk¹ta.
    * @param A przetransformowany wierzcho³ek A
    * @param B przetransformowany wierzcho³ek B
    * @param C przetransformowany wierzcho³ek C
    * @param color kolor trójk¹ta
    * @param minY minimalna wartoœæ Y
    * @param maxY maksymalna wartoœæ Y
    */
    void DrawTriangleBoundingBox(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int minY, int maxY);
    /**
    * Rysuje liniê.
    * @param A przetransformowany wierzcho³ek A
    * @param B przetransformowany wierzcho³ek B
    * @param color kolor linii
    * @param minY minimalna wartoœæ Y
    * @param maxY maksymalna wartoœæ Y
    */
    void DrawLine(const TransformedVertex& A, const TransformedVertex& B, const Vector4f& color, int minY, int maxY);
    /**
    * Rysuje liniê.
    * @param A wierzcho³ek A
    * @param B wierzcho³ek B
    * @param color kolor linii
    * @param minY minimalna wartoœæ Y
    * @param maxY maksymalna wartoœæ Y
    */
    void DrawLine(Vector2f A, Vector2f B, const Vector4f& color, int minY, int maxY);
    /**
    * Ustawia kolor piksela
    * @param x wspó³rzêdna X
    * @param y wspó³rzêdna Y
    * @param color kolor piksela
    */
    void PutPixel(int x, int y, uint32_t color);
    /**
    * Ustawia kolor piksela bez sprawdzania granic ekranu.
    * @param x wspó³rzêdna X
    * @param y wspó³rzêdna Y
    * @param color kolor piksela
    */
    void PutPixelUnsafe(int x, int y, uint32_t color);
    /**
    * Aktualizuje macierz MVP.
    */
    void UpdateMVPMatrix();
    /**
    * Wykonuje renderowanie.
    * @param vertices wektor wierzcho³ków
    * @param minY minimalna wartoœæ Y
    * @param maxY maksymalna wartoœæ Y
    * @param threadID identyfikator w¹tku
    */
    void DoRender(const vector<Vertex>& vertices, int minY, int maxY, int threadID);
    /**
    * Funkcja wyliczaj¹ca kolor piksela.
    * @param vertex przetransformowany wierzcho³ek
    * @return kolor fragmentu
    */
    Vector4f FragmentShader(const TransformedVertex& vertex);
    /**
    * Funkcja krawêdzi.
    * @param A wierzcho³ek A
    * @param B wierzcho³ek B
    * @param C wierzcho³ek C
    * @return wartoœæ funkcji krawêdzi, dodatnia jeœli punkt C znajduje siê po lewej stronie odcinka AB, ujemna jeœli po prawej, 0 jeœli punkt C le¿y na odcinku AB
    */
    static float EdgeFunction(const Vector2f& A, const Vector2f& B, const Vector2f& C);

    // 8 bit - one channel (8*4=32 - rgba)
    vector<uint32_t>    m_ScreenBuffer; ///< bufor ekranu
    vector<float>       m_ZBuffer; ///< bufor g³êbokoœci

    Vector4f            m_WireFrameColor = Vector4f(1, 1, 1, 1); ///< kolor siatki
    Vector3f            m_DiffuseColor = Vector3f(1, 1, 1); ///< kolor œwiat³a rozproszenia
    Vector3f            m_AmbientColor = Vector3f(1, 1, 1); ///< kolor œwiat³a otoczenia
    Vector3f            m_LightPosition = Vector3f(0, 0, -20); ///< pozycja œwiat³a
    Vector3f            m_CameraPosition = Vector3f(0, 0, 0); ///< pozycja kamery
    Vector4f            m_ThreadColors[12]; ///< kolory w¹tków
    uint32_t            m_ClearColor = 0xFF000000; ///< kolor t³a

    Matrix4f            m_ModelMatrix; ///< macierz modelu
    Matrix4f            m_ViewMatrix; ///< macierz widoku
    Matrix4f            m_ProjectionMatrix; ///< macierz projekcji
    Matrix4f            m_MVPMatrix; ///< macierz MVP

    bool                m_DrawWireframe = false; ///< czy rysowaæ siatkê
    bool                m_ColorizeThreads = false; ///< czy kolorowaæ w¹tki
    bool                m_DrawBBoxes = false; ///< czy rysowaæ bry³y brzegowe
    bool                m_ZWrite = true; ///< czy zapisywaæ do bufora g³êbokoœci
    bool                m_ZTest = true; ///< czy testowaæ bufor g³êbokoœci
    float               m_DiffuseStrength = 0.3f; ///< si³a œwiat³a rozproszenia
    float               m_AmbientStrength = 0.5f; ///< si³a œwiat³a otoczenia
    float               m_SpecularStrength = 0.9f; ///< si³a œwiat³a odbitego
    float               m_Shininess = 32.0f; ///< moc œwiat³a odbitego
    uint8_t             m_ThreadsCount = 0; ///< liczba w¹tków

    atomic_int          m_FrameTriangles = 0; ///< liczba przetworzanych trójk¹tów
    atomic_int          m_FrameTrianglesDrawn = 0; ///< liczba narysowanych trójk¹tów
    atomic_int          m_FramePixels = 0; ///< liczba przetworzanych pikseli
    atomic_int          m_FramePixelsDrawn = 0; ///< liczba narysowanych pikseli
    atomic_int          m_FrameRasterTimeUS = 0; ///< czas rasteryzacji w mikrosekundach
    atomic_int          m_FrameTransformTimeUS = 0; ///< czas transformacji w mikrosekundach
    atomic_int          m_FrameDrawTimeThreadUS = 0; ///< czas rysowania w¹tków w mikrosekundach
    atomic_int          m_FrameDrawTimeMainUS = 0; ///< czas rysowania g³ównego w¹tku w mikrosekundach
    atomic_int          m_FillrateKP = 0; ///< liczba pikseli wype³nionych na sekundê

    DrawStats           m_DrawStats; ///< statystyki rysowania

    shared_ptr<Texture> m_Texture; ///< tekstura
    shared_ptr<Texture> m_DefaultTexture; ///< domyœlna tekstura
    SimpleThreadPool    m_ThreadPool; ///< pula w¹tków
};