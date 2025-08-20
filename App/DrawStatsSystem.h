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
    static inline constexpr int FRAME_SAMPLES_COUNT = 120;

    struct Stats : DrawStats
    {
        float   m_FPS = 0;
    };

    static void AddSample(DrawStats sample);
    static const Stats& GetAvg();
    static const Stats& GetMin();
    static const Stats& GetMax();
    static const Stats& GetMed();
    static const Stats& GetStd();
private:
    void Update();

    static DrawStatsSystem s_Instance;
    DrawStats m_FrameSamples[FRAME_SAMPLES_COUNT];
    DrawStats m_MedianBuf[FRAME_SAMPLES_COUNT];
    Stats m_Avg;
    Stats m_Min;
    Stats m_Max;
    Stats m_Median;
    Stats m_StdDev;
    int m_FrameSampleIndex = 0;
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