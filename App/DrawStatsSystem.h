/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Common.h"

class DrawStatsSystem
{
    DrawStatsSystem()=default;
public:
    struct Stats : DrawStats
    {
        float m_FPS = 0;
    };

    static void AddSample(DrawStats sample);

    template< int SAMPLES = 120 >
    static const Stats& GetAvg();

    template< int SAMPLES = 120 >
    static const Stats& GetMin();

    template< int SAMPLES = 120 >
    static const Stats& GetMax();

    template< int SAMPLES = 120 >
    static const Stats& GetMed();

    template< int SAMPLES = 120 >
    static const Stats& GetStd();
private:

    template< int SAMPLES = 120 >
    struct Data
    {
        static inline constexpr int FRAME_SAMPLES_COUNT = SAMPLES; ///< liczba próbek

        void Update();

        static Data s_Instance; ///< instancja klasy
        DrawStats m_FrameSamples[FRAME_SAMPLES_COUNT]; ///< próbki statystyk rysowania
        DrawStats m_MedianBuf[FRAME_SAMPLES_COUNT]; ///< bufor mediany
        Stats m_Avg; ///< œrednia statystyk rysowania
        Stats m_Min; ///< minimum statystyk rysowania
        Stats m_Max; ///< maksimum statystyk rysowania
        Stats m_Median; ///< mediana statystyk rysowania
        Stats m_StdDev; ///< odchylenie standardowe statystyk rysowania
        int m_FrameSampleIndex = 0; ///< indeks próbki statystyk rysowania
        bool m_Dirty = false;

        inline void AddSample(DrawStats sample);
    };
};

template< int SAMPLES >
inline void DrawStatsSystem::Data<SAMPLES>::AddSample(DrawStats sample)
{
    s_Instance.m_FrameSamples[s_Instance.m_FrameSampleIndex] = sample;
    s_Instance.m_FrameSampleIndex = (s_Instance.m_FrameSampleIndex + 1) % FRAME_SAMPLES_COUNT;
    s_Instance.m_Dirty = true;
}

inline void DrawStatsSystem::AddSample(DrawStats sample)
{
    Data<480>::s_Instance.AddSample(sample);
    Data<240>::s_Instance.AddSample(sample);
    Data<120>::s_Instance.AddSample(sample);
    Data<60> ::s_Instance.AddSample(sample);
    Data<30> ::s_Instance.AddSample(sample);
}

template< int SAMPLES >
inline auto DrawStatsSystem::GetAvg()->const Stats&
{
    Data<SAMPLES>::s_Instance.Update();
    return Data<SAMPLES>::s_Instance.m_Avg;
}
template< int SAMPLES >
inline auto DrawStatsSystem::GetMin()->const Stats&
{
    Data<SAMPLES>::s_Instance.Update();
    return Data<SAMPLES>::s_Instance.m_Min;
}
template< int SAMPLES >
inline auto DrawStatsSystem::GetMax()->const Stats&
{
    Data<SAMPLES>::s_Instance.Update();
    return Data<SAMPLES>::s_Instance.m_Max;
}
template< int SAMPLES >
inline auto DrawStatsSystem::GetMed()->const Stats&
{
    Data<SAMPLES>::s_Instance.Update();
    return Data<SAMPLES>::s_Instance.m_Median;
}
template< int SAMPLES >
inline auto DrawStatsSystem::GetStd()->const Stats&
{
    Data<SAMPLES>::s_Instance.Update();
    return Data<SAMPLES>::s_Instance.m_StdDev;
}