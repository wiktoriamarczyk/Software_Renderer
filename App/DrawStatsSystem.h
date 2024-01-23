/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Common.h"

/**
* Klasa odpowiedzialna za organizacjê statystyk rysowania.
*/
class DrawStatsSystem
{
    DrawStatsSystem()=default;
public:
    static inline constexpr int FRAME_SAMPLES_COUNT = 60; ///< liczba próbek

    /**
    * Struktura reprezentuj¹ca statystykê rysowania.
    */
    struct Stats : DrawStats
    {
        float m_FPS = 0;
    };

    /**
    * Dodaje próbkê statystyk rysowania z danej klatki.
    * @param sample próbka statystyk rysowania
    */
    static void AddSample(DrawStats sample);
    /**
    * Zwraca œredni¹ statystyk rysowania.
    * @return obiekt zawieraj¹cy œredni¹ statystyk rysowania
    */
    static const Stats& GetAvg();
    /**
    * Zwraca minimum ze statystyk rysowania.
    * @return obiekt zawieraj¹cy minimum statystyk rysowania
    */
    static const Stats& GetMin();
    /**
    * Zwraca maksimum ze statystyk rysowania.
    * @return obiekt zawieraj¹cy maksimum statystyk rysowania
    */
    static const Stats& GetMax();
    /**
    * Zwraca medianê ze statystyk rysowania.
    * @return obiekt zawieraj¹cy medianê statystyk rysowania
    */
    static const Stats& GetMed();
    /**
    * Zwraca odchylenie standardowe ze statystyk rysowania.
    * @return obiekt zawieraj¹cy odchylenie standardowe statystyk rysowania
    */
    static const Stats& GetStd();
private:
    /**
    * Funkcja aktualizuj¹ca statystyki rysowania dla klatki.
    */
    void Update();

    static DrawStatsSystem s_Instance; ///< instancja klasy
    DrawStats m_FrameSamples[FRAME_SAMPLES_COUNT]; ///< próbki statystyk rysowania
    DrawStats m_MedianBuf[FRAME_SAMPLES_COUNT]; ///< bufor mediany
    Stats m_Avg; ///< œrednia statystyk rysowania
    Stats m_Min; ///< minimum statystyk rysowania
    Stats m_Max; ///< maksimum statystyk rysowania
    Stats m_Median; ///< mediana statystyk rysowania
    Stats m_StdDev; ///< odchylenie standardowe statystyk rysowania
    int m_FrameSampleIndex = 0; ///< indeks próbki statystyk rysowania
};

inline void DrawStatsSystem::AddSample(DrawStats sample)
{
    s_Instance.m_FrameSamples[s_Instance.m_FrameSampleIndex] = sample;
    s_Instance.m_FrameSampleIndex = (s_Instance.m_FrameSampleIndex + 1) % FRAME_SAMPLES_COUNT;
    s_Instance.Update();
}

inline auto DrawStatsSystem::GetAvg()->const Stats&
{
    return s_Instance.m_Avg;
}
inline auto DrawStatsSystem::GetMin()->const Stats&
{
    return s_Instance.m_Min;
}
inline auto DrawStatsSystem::GetMax()->const Stats&
{
    return s_Instance.m_Max;
}
inline auto DrawStatsSystem::GetMed()->const Stats&
{
    return s_Instance.m_Median;
}
inline auto DrawStatsSystem::GetStd()->const Stats&
{
    return s_Instance.m_StdDev;
}