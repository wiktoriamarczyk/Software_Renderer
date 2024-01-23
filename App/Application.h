/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Common.h"
#include "IRenderer.h"
#include "Math.h"

/**
* Przechowuje ustawienia rysowania.
*/
struct DrawSettings
{
    Vector3f    modelRotation; ///< rotacja modelu
    Vector3f    modelTranslation; ///< translacja modelu
    float       modelScale = 1.0; ///< skalowanie modelu
    Vector4f    wireFrameColor = Vector4f(1, 1, 1, 1); ///< kolor siatki
    Vector3f    diffuseColor = Vector3f(1, 1, 1); ///< kolor œwiat³a rozproszenia
    Vector3f    ambientColor = Vector3f(1, 1, 1); ///< kolor œwiat³a otoczenia
    Vector3f    backgroundColor = Vector3f(0, 0, 0); ///< kolor t³a
    Vector3f    lightPosition = Vector3f(0, 0, -20); ///< pozycja œwiat³a
    float       diffuseStrength = 0.7f; ///< si³a œwiat³a rozproszenia
    float       ambientStrength = 0.1f; ///< si³a œwiat³a otoczenia
    float       specularStrength = 0.9f; ///< si³a œwiat³a odbitego
    float       shininessPower = 5; ///< moc œwiat³a odbitego
    int         threadsCount = 1; ///< liczba w¹tków
    bool        drawWireframe = false; ///< czy rysowaæ siatkê
    bool        drawBBoxes = false; ///< czy rysowaæ bry³y brzegowe
    bool        colorizeThreads = false; ///< czy kolorowaæ w¹tki
    bool        useZBuffer = true; ///< czy u¿ywaæ bufora g³êbokoœci
    bool        renderDepthBuffer = false; ///< czy rysowaæ bufor g³êbokoœci
    bool        vSync = true; ///< czy w³¹czyæ synchronizacjê pionow¹
    int         rendererType = 0; ///< typ renderera
};

/**
* Przechowuje dane modelu.
*/
struct Model
{
    vector<Vertex> vertices; ///< wierzcho³ki modelu
    Vector3f Min;
    Vector3f Max;
};

/**
* Przechowuje scie¿ki do modelu i tekstury.
*/
struct MyModelPaths
{
    string modelPath; ///< œcie¿ka do modelu
    string texturePath; ///< œcie¿ka do tekstury
};

/**
* Przechowuje kontekst renderera.
*/
struct RendererContext
{
    shared_ptr<IRenderer> pRenderer; ///< wskaŸnik na renderer
    shared_ptr<ITexture>  pModelTexture; ///< wskaŸnik na teksturê modelu
};

class aiScene;

/**
* G³ówna klasa aplikacji. Odpowiada za inicjalizacjê okna aplikacji i sceny pocz¹tkowej.
*/
class Application
{
public:
    Application()=default;
    /**
    * Odpowiada za inicjalizacjê aplikacji i sceny pocz¹tkowej.
    * @return true, jeœli inicjalizacja siê powiod³a, w przeciwnym wypadku false.
    */
    bool Initialize();
    /**
    * Odpowiada za g³ówn¹ pêtlê aplikacji.
    * @return 0, jeœli aplikacja zakoñczy³a siê poprawnie
    */
    int Run();
private:
    /**
    * £aduje wierzcho³ki modelu z pliku.
    * @param path œcie¿ka do modelu.
    * @return wektor zainicjalizowanych modeli.
    */
    static vector<Model> LoadModelVertices(const char* path);
    /**
    * Funkcja pomocnicza do ³adowania modelu z pliku.
    * @param pScene wskaŸnik na scenê.
    */
    static vector<Model> LoadFromScene(const aiScene* pScene);
    /**
    * Odpowiada za normalizacjê pozycji modelu.
    * @param models wektor modeli do normalizacji.
    */
    static void NormalizeModelPosition(vector<Model>& models);
    /**
    * Odpowiada za za³adowanie domyœlnego modelu.
    * @return wektor modeli.
    */
    static vector<Model> LoadFallbackModel();
    /**
    * Otwiera okno dialogowe.
    * @param title tytu³ okna dialogowego.
    * @param filters rodzaje plików do wyœwietlenia.
    * @param callback funkcja wywo³ywana po wybraniu pliku.
    */
    static void OpenDialog(const char* title, const char* filters, function<void()> callback);
    /**
    * Odpowiada za wyœwietlenie okna dialogowego do wgrania modelu.
    * @param selectedPaths œcie¿ki do modelu i tekstury.
    */
    void OpenSceneDataDialog(MyModelPaths& selectedPaths);
    /**
    * Odpowiada za wyœwietlenie okna statystyk.
    */
    void DrawRenderingStats();

    const uint8_t MAX_THREADS_COUNT = uint8_t(std::min<int>(12, std::thread::hardware_concurrency())); ///< Maksymalna liczba w¹tków.
    DrawSettings m_DrawSettings; ///< Ustawienia rysowania.
    RendererContext m_Contexts[2]; ///< Konteksty renderera.
    MyModelPaths m_LastModelPaths; ///< Ostatnie œcie¿ki do modelu i tekstury.
    MyModelPaths m_ModelPaths; ///< Œcie¿ki do modelu i tekstury.
    Matrix4f m_CameraMatrix; ///< Macierz kamery.
    Matrix4f m_ProjectionMatrix; ///< Macierz projekcji.
    Matrix4f m_ModelMatrix; ///< Macierz modelu.
    vector<Model> m_ModelsData; ///< Dane modelu.
    sf::RenderWindow m_MainWindow; ///< G³ówne okno aplikacji.
    sf::Texture m_ScreenTexture; ///< G³ówna tekstura aplikacji.
    sf::Sprite m_ScreenSprite; ///< G³ówny sprite aplikacji.
    sf::Clock m_DeltaClock; ///< Zegar aplikacji.
    std::chrono::steady_clock::time_point m_LastFrameTime; ///< Czas ostatniej klatki.
    bool m_VSync = true; ///< Czy w³¹czona synchronizacja pionowa.
};