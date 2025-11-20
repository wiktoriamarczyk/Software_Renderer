/*
* Master’s thesis - Analysis of selected optimization techniques for a 3D software renderer
* Author: Wiktoria Marczyk
* Year: 2025
*/

#pragma once
#include "Common.h"

class DrawStatsSystem
{
    DrawStatsSystem() = default;
public:
    struct Stats : DrawStats
    {
        float m_FPS = 0;
    };

    static void AddSample(DrawStats sample);

    template<int SAMPLES = 120>
    static const Stats& GetAvg();

    template<int SAMPLES = 120>
    static const Stats& GetMin();

    template< int SAMPLES = 120>
    static const Stats& GetMax();

    template<int SAMPLES = 120>
    static const Stats& GetMed();

    template<int SAMPLES = 120>
    static const Stats& GetStd();
private:

    template<int SAMPLES = 120 >
    struct Data
    {
        static inline constexpr int FRAME_SAMPLES_COUNT = SAMPLES;

        void Update();

        static Data s_Instance;
        DrawStats m_FrameSamples[FRAME_SAMPLES_COUNT];
        DrawStats m_MedianBuf[FRAME_SAMPLES_COUNT];
        Stats m_Avg;
        Stats m_Min;
        Stats m_Max;
        Stats m_Median;
        Stats m_StdDev;
        int m_FrameSampleIndex = 0;
        bool m_Dirty = false;

        inline void AddSample(DrawStats sample);
    };
};

template<int SAMPLES>
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

template<int SAMPLES>
inline auto DrawStatsSystem::GetAvg()->const Stats&
{
    Data<SAMPLES>::s_Instance.Update();
    return Data<SAMPLES>::s_Instance.m_Avg;
}
template<int SAMPLES>
inline auto DrawStatsSystem::GetMin()->const Stats&
{
    Data<SAMPLES>::s_Instance.Update();
    return Data<SAMPLES>::s_Instance.m_Min;
}
template<int SAMPLES>
inline auto DrawStatsSystem::GetMax()->const Stats&
{
    Data<SAMPLES>::s_Instance.Update();
    return Data<SAMPLES>::s_Instance.m_Max;
}
template<int SAMPLES>
inline auto DrawStatsSystem::GetMed()->const Stats&
{
    Data<SAMPLES>::s_Instance.Update();
    return Data<SAMPLES>::s_Instance.m_Median;
}
template<int SAMPLES>
inline auto DrawStatsSystem::GetStd()->const Stats&
{
    Data<SAMPLES>::s_Instance.Update();
    return Data<SAMPLES>::s_Instance.m_StdDev;
}