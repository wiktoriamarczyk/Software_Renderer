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

/**
* Enum okreœlaj¹cy typ renderera.
*/
enum class eRendererType : uint8_t
{
    Software,
    Hardware
};

/**
* Interfejs tekstury.
*/
class ITexture
{
public:
    virtual ~ITexture() = default;
    /**
    * Sprawdza czy tekstura jest poprawna.
    * @return true, jeœli tekstura jest poprawna, false w przeciwnym wypadku.
    */
    virtual bool IsValid()const=0;
};

class IRenderer;

/**
* Klasa fabryki renderera. Pozwala na tworzenie obiektów renderera.
*/
class RendererFactory
{
public:
    /**
    * Tworzy obiekt renderera.
    * @param rendererType typ renderera
    * @param screenWidth szerokoœæ ekranu
    * @param screenHeight wysokoœæ ekranu
    * @return wskaŸnik na obiekt renderera
    */
    static shared_ptr<IRenderer> CreateRenderer(eRendererType rendererType, int screenWidth, int screenHeight);
};

/**
* Interfejs renderera.
*/
class IRenderer
{
public:
    /**
    * £aduje teksturê z pliku.
    * @param fileName œcie¿ka do pliku.
    * @return wskaŸnik na obiekt tekstury.
    */
    virtual shared_ptr<ITexture> LoadTexture(const char* fileName)const=0;
    /**
    * Czyœci bufor koloru.
    */
    virtual void ClearScreen()=0;
    /**
    * Czyœci bufor g³êbokoœci.
    */
    virtual void ClearZBuffer()=0;
    /**
    * Rozpoczyna rysowanie klatki.
    */
    virtual void BeginFrame()=0;
    /**
    * Rysuje obiekt.
    * @param vertices wektor wierzcho³ków sceny
    */
    virtual void Render(const vector<Vertex>& vertices)=0;
    /**
    * Koñczy rysowanie klatki.
    */
    virtual void EndFrame()=0;
    /**
    * Przerysowuje bufor g³êbokoœci do bufora koloru.
    */
    virtual void RenderDepthBuffer()=0;
    /**
    * Zwraca bufor ekranu (koloru).
    * @return bufor ekranu (koloru)
    */
    virtual const vector<uint32_t>& GetScreenBuffer() const=0;
    /**
    * Zwraca statystyki rysowania.
    * @return obiekt statystyk rysowania
    */
    virtual const DrawStats& GetDrawStats() const=0;
    /**
    * Zwraca domyœln¹ teksturê.
    * @return wskaŸnik na domyœln¹ teksturê
    */
    virtual shared_ptr<ITexture> GetDefaultTexture() const=0;
    /**
    * Ustawia macierz modelu.
    * @param other macierz modelu
    */
    virtual void SetModelMatrix(const Matrix4f& other)=0;
    /**
    * Ustawia macierz widoku.
    * @param other macierz widoku
    */
    virtual void SetViewMatrix(const Matrix4f& other)=0;
    /**
    * Ustawia macierz projekcji.
    * @param other macierz projekcji
    */
    virtual void SetProjectionMatrix(const Matrix4f& other)=0;
    /**
    * Ustawia teksturê.
    * @param texture wskaŸnik na teksturê
    */
    virtual void SetTexture(shared_ptr<ITexture> texture)=0;
    /**
    * Ustawia kolor siatki.
    * @param wireFrameColor kolor siatki
    */
    virtual void SetWireFrameColor(const Vector4f& wireFrameColor)=0;
    /**
    * Ustawia kolor t³a.
    * @param clearColor kolor t³a
    */
    virtual void SetClearColor(const Vector4f& clearColor)=0;
    /**
    * Ustawia kolor œwiat³a rozproszenia.
    * @param diffuseColor kolor œwiat³a rozproszenia
    */
    virtual void SetDiffuseColor(const Vector3f& diffuseColor)=0;
    /**
    * Ustawia kolor œwiat³a otoczenia.
    * @param ambientColor kolor œwiat³a otoczenia
    */
    virtual void SetAmbientColor(const Vector3f& ambientColor)=0;
    /**
    * Ustawia pozycjê œwiat³a.
    * @param lightPosition pozycja œwiat³a
    */
    virtual void SetLightPosition(const Vector3f& lightPosition)=0;
    /**
    * Ustawia si³ê œwiat³a rozproszenia.
    * @param diffuseStrength si³a œwiat³a rozproszenia
    */
    virtual void SetDiffuseStrength(float diffuseStrength)=0;
    /**
    * Ustawia si³ê œwiat³a otoczenia.
    * @param ambientStrength si³a œwiat³a otoczenia
    */
    virtual void SetAmbientStrength(float ambientStrength)=0;
    /**
    * Ustawia si³ê œwiat³a odbitego.
    * @param specularStrength si³a œwiat³a odbitego
    */
    virtual void SetSpecularStrength(float specularStrength)=0;
    /**
    * Ustawia moc œwiat³a odbitego.
    * @param shininess moc œwiat³a odbitego
    */
    virtual void SetShininess(float shininess)=0;
    /**
    * Ustawia liczbê w¹tków.
    * @param threadsCount liczba w¹tków
    */
    virtual void SetThreadsCount(uint8_t threadsCount)=0;
    /**
    * Ustawia czy kolorowaæ w¹tki.
    * @param colorizeThreads czy kolorowaæ w¹tki
    */
    virtual void SetColorizeThreads(bool colorizeThreads)=0;
    /**
    * Ustawia czy rysowaæ siatkê.
    * @param wireframe czy rysowaæ siatkê
    */
    virtual void SetDrawWireframe(bool wireframe)=0;
    /**
    * Ustawia czy rysowaæ bry³y brzegowe.
    * @param drawBBoxes czy rysowaæ bry³y brzegowe
    */
    virtual void SetDrawBBoxes(bool drawBBoxes)=0;
    /**
    * Ustawia czy wpisywaæ do bufora g³êbokoœci.
    * @param zwrite czy wpisywaæ bufora g³êbokoœci
    */
    virtual void SetZWrite(bool zwrite)=0;
    /**
    * Ustawia czy testowaæ bufor g³êbokoœci.
    * @param ztest czy testowaæ bufor g³êbokoœci
    */
    virtual void SetZTest(bool ztest)=0;
};