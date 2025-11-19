/*
* Master’s thesis - Analysis of selected optimization techniques for a 3D software renderer
* Author: Wiktoria Marczyk
* Year: 2025
*/

#include "DrawStatsSystem.h"

template< int SAMPLES >
DrawStatsSystem::Data<SAMPLES> DrawStatsSystem::Data<SAMPLES>::s_Instance;


template< int SAMPLES >
void DrawStatsSystem::Data<SAMPLES>::Update()
{
    if (!m_Dirty)
        return;
    ZoneScoped;

    m_Avg = {};
    m_Median = {};
    m_StdDev = {};

    m_Min.m_FramePixels             = std::numeric_limits<int>::max();
    m_Min.m_FramePixelsDrawn        = std::numeric_limits<int>::max();
    m_Min.m_FramePixelsCalcualted   = std::numeric_limits<int>::max();
    m_Min.m_FrameTriangles          = std::numeric_limits<int>::max();
    m_Min.m_FrameTrianglesDrawn     = std::numeric_limits<int>::max();
    m_Min.m_FrameDrawsPerTile       = std::numeric_limits<int>::max();
    m_Min.m_DT                      = std::numeric_limits<int>::max();
    m_Min.m_DrawTimeUS              = std::numeric_limits<int>::max();
    m_Min.m_DrawTimePerThreadUS     = std::numeric_limits<int>::max();
    m_Min.m_FillrateKP              = std::numeric_limits<int>::max();

    m_Min.m_RasterTimeUS            = std::numeric_limits<int>::max();
    m_Min.m_RasterTimePerThreadUS   = std::numeric_limits<int>::max();
    m_Min.m_TransformTimeUS         = std::numeric_limits<int>::max();

    m_Max.m_FramePixels             = std::numeric_limits<int>::min();
    m_Max.m_FramePixelsDrawn        = std::numeric_limits<int>::min();
    m_Max.m_FramePixelsCalcualted   = std::numeric_limits<int>::min();
    m_Max.m_FrameTriangles          = std::numeric_limits<int>::min();
    m_Max.m_FrameTrianglesDrawn     = std::numeric_limits<int>::min();
    m_Max.m_FrameDrawsPerTile       = std::numeric_limits<int>::min();
    m_Max.m_DT                      = std::numeric_limits<int>::min();
    m_Max.m_DrawTimeUS              = std::numeric_limits<int>::min();
    m_Max.m_DrawTimePerThreadUS     = std::numeric_limits<int>::min();
    m_Max.m_FillrateKP              = std::numeric_limits<int>::min();

    m_Max.m_RasterTimeUS            = std::numeric_limits<int>::min();
    m_Max.m_RasterTimePerThreadUS   = std::numeric_limits<int>::min();
    m_Max.m_TransformTimeUS         = std::numeric_limits<int>::min();
    m_Max.m_TransformTimePerThreadUS= std::numeric_limits<int>::min();

    for (auto& sample : m_FrameSamples)
    {
        m_Avg.m_DT += sample.m_DT;
        m_Avg.m_FramePixels += sample.m_FramePixels;
        m_Avg.m_FramePixelsDrawn += sample.m_FramePixelsDrawn;
        m_Avg.m_FramePixelsCalcualted += sample.m_FramePixelsCalcualted;
        m_Avg.m_FrameTriangles += sample.m_FrameTriangles;
        m_Avg.m_FrameTrianglesDrawn += sample.m_FrameTrianglesDrawn;
        m_Avg.m_FrameDrawsPerTile += sample.m_FrameDrawsPerTile;
        m_Avg.m_DrawTimeUS += sample.m_DrawTimeUS;
        m_Avg.m_DrawTimePerThreadUS += sample.m_DrawTimePerThreadUS;
        m_Avg.m_FillrateKP += sample.m_FillrateKP;

        m_Avg.m_RasterTimeUS += sample.m_RasterTimeUS;
        m_Avg.m_RasterTimePerThreadUS += sample.m_RasterTimePerThreadUS;
        m_Avg.m_TransformTimeUS += sample.m_TransformTimeUS;
        m_Avg.m_TransformTimePerThreadUS += sample.m_TransformTimePerThreadUS;

        m_Min.m_FramePixels = std::min(m_Min.m_FramePixels, sample.m_FramePixels);
        m_Min.m_FramePixelsDrawn = std::min(m_Min.m_FramePixelsDrawn, sample.m_FramePixelsDrawn);
        m_Min.m_FramePixelsCalcualted = std::min(m_Min.m_FramePixelsCalcualted, sample.m_FramePixelsCalcualted);
        m_Min.m_FrameTriangles = std::min(m_Min.m_FrameTriangles, sample.m_FrameTriangles);
        m_Min.m_FrameTrianglesDrawn = std::min(m_Min.m_FrameTrianglesDrawn, sample.m_FrameTrianglesDrawn);
        m_Min.m_FrameDrawsPerTile = std::min(m_Min.m_FrameDrawsPerTile, sample.m_FrameDrawsPerTile);
        m_Min.m_DT = std::min(m_Min.m_DT, sample.m_DT);
        m_Min.m_DrawTimeUS = std::min(m_Min.m_DrawTimeUS, sample.m_DrawTimeUS);
        m_Min.m_DrawTimePerThreadUS = std::min(m_Min.m_DrawTimePerThreadUS, sample.m_DrawTimePerThreadUS);
        m_Min.m_FillrateKP = std::min(m_Min.m_FillrateKP, sample.m_FillrateKP);

        m_Min.m_RasterTimeUS = std::min(m_Min.m_RasterTimeUS, sample.m_RasterTimeUS);
        m_Min.m_RasterTimePerThreadUS = std::min(m_Min.m_RasterTimePerThreadUS, sample.m_RasterTimePerThreadUS);
        m_Min.m_TransformTimeUS = std::min(m_Min.m_TransformTimeUS, sample.m_TransformTimeUS);
        m_Min.m_TransformTimePerThreadUS = std::min(m_Min.m_TransformTimePerThreadUS, sample.m_TransformTimePerThreadUS);


        m_Max.m_FramePixels = std::max(m_Max.m_FramePixels, sample.m_FramePixels);
        m_Max.m_FramePixelsDrawn = std::max(m_Max.m_FramePixelsDrawn, sample.m_FramePixelsDrawn);
        m_Max.m_FramePixelsCalcualted = std::max(m_Max.m_FramePixelsCalcualted, sample.m_FramePixelsCalcualted);
        m_Max.m_FrameTriangles = std::max(m_Max.m_FrameTriangles, sample.m_FrameTriangles);
        m_Max.m_FrameTrianglesDrawn = std::max(m_Max.m_FrameTrianglesDrawn, sample.m_FrameTrianglesDrawn);
        m_Max.m_FrameDrawsPerTile = std::max(m_Max.m_FrameDrawsPerTile, sample.m_FrameDrawsPerTile);
        m_Max.m_DT = std::max(m_Max.m_DT, sample.m_DT);
        m_Max.m_DrawTimeUS = std::max(m_Max.m_DrawTimeUS, sample.m_DrawTimeUS);
        m_Max.m_DrawTimePerThreadUS = std::max(m_Max.m_DrawTimePerThreadUS, sample.m_DrawTimePerThreadUS);
        m_Max.m_FillrateKP = std::max(m_Max.m_FillrateKP, sample.m_FillrateKP);

        m_Max.m_RasterTimeUS = std::max(m_Max.m_RasterTimeUS, sample.m_RasterTimeUS);
        m_Max.m_RasterTimePerThreadUS = std::max(m_Max.m_RasterTimePerThreadUS, sample.m_RasterTimePerThreadUS);
        m_Max.m_TransformTimeUS = std::max(m_Max.m_TransformTimeUS, sample.m_TransformTimeUS);
        m_Max.m_TransformTimePerThreadUS = std::max(m_Max.m_TransformTimePerThreadUS, sample.m_TransformTimePerThreadUS);
    }

    m_Avg.m_DT /= FRAME_SAMPLES_COUNT;
    m_Avg.m_FramePixels /= FRAME_SAMPLES_COUNT;
    m_Avg.m_FramePixelsDrawn /= FRAME_SAMPLES_COUNT;
    m_Avg.m_FramePixelsCalcualted /= FRAME_SAMPLES_COUNT;
    m_Avg.m_FrameTriangles /= FRAME_SAMPLES_COUNT;
    m_Avg.m_FrameTrianglesDrawn /= FRAME_SAMPLES_COUNT;
    m_Avg.m_FrameDrawsPerTile /= FRAME_SAMPLES_COUNT;
    m_Avg.m_DrawTimeUS /= FRAME_SAMPLES_COUNT;
    m_Avg.m_DrawTimePerThreadUS /= FRAME_SAMPLES_COUNT;
    m_Avg.m_FillrateKP /= FRAME_SAMPLES_COUNT;

    m_Avg.m_RasterTimeUS  /= FRAME_SAMPLES_COUNT;
    m_Avg.m_RasterTimePerThreadUS  /= FRAME_SAMPLES_COUNT;
    m_Avg.m_TransformTimeUS  /= FRAME_SAMPLES_COUNT;
    m_Avg.m_TransformTimePerThreadUS  /= FRAME_SAMPLES_COUNT;

    memcpy(m_MedianBuf, m_FrameSamples, sizeof(m_FrameSamples));
    std::sort(std::begin(m_MedianBuf), std::end(m_MedianBuf), [](const DrawStats& a, const DrawStats& b) { return a.m_DT < b.m_DT; });
    m_Median.m_DT = m_MedianBuf[FRAME_SAMPLES_COUNT / 2].m_DT;

    memcpy(m_MedianBuf, m_FrameSamples, sizeof(m_FrameSamples));
    std::sort(std::begin(m_MedianBuf), std::end(m_MedianBuf), [](const DrawStats& a, const DrawStats& b) { return a.m_FramePixels < b.m_FramePixels; });
    m_Median.m_FramePixels = m_MedianBuf[FRAME_SAMPLES_COUNT / 2].m_FramePixels;

    memcpy(m_MedianBuf, m_FrameSamples, sizeof(m_FrameSamples));
    std::sort(std::begin(m_MedianBuf), std::end(m_MedianBuf), [](const DrawStats& a, const DrawStats& b) { return a.m_FramePixelsDrawn < b.m_FramePixelsDrawn; });
    m_Median.m_FramePixelsDrawn = m_MedianBuf[FRAME_SAMPLES_COUNT / 2].m_FramePixelsDrawn;

    memcpy(m_MedianBuf, m_FrameSamples, sizeof(m_FrameSamples));
    std::sort(std::begin(m_MedianBuf), std::end(m_MedianBuf), [](const DrawStats& a, const DrawStats& b) { return a.m_FramePixelsCalcualted < b.m_FramePixelsCalcualted; });
    m_Median.m_FramePixelsCalcualted = m_MedianBuf[FRAME_SAMPLES_COUNT / 2].m_FramePixelsCalcualted;

    memcpy(m_MedianBuf, m_FrameSamples, sizeof(m_FrameSamples));
    std::sort(std::begin(m_MedianBuf), std::end(m_MedianBuf), [](const DrawStats& a, const DrawStats& b) { return a.m_FrameTriangles < b.m_FrameTriangles; });
    m_Median.m_FrameTriangles = m_MedianBuf[FRAME_SAMPLES_COUNT / 2].m_FrameTriangles;

    memcpy(m_MedianBuf, m_FrameSamples, sizeof(m_FrameSamples));
    std::sort(std::begin(m_MedianBuf), std::end(m_MedianBuf), [](const DrawStats& a, const DrawStats& b) { return a.m_FrameTrianglesDrawn < b.m_FrameTrianglesDrawn; });
    m_Median.m_FrameTrianglesDrawn = m_MedianBuf[FRAME_SAMPLES_COUNT / 2].m_FrameTrianglesDrawn;

    memcpy(m_MedianBuf, m_FrameSamples, sizeof(m_FrameSamples));
    std::sort(std::begin(m_MedianBuf), std::end(m_MedianBuf), [](const DrawStats& a, const DrawStats& b) { return a.m_FrameDrawsPerTile < b.m_FrameDrawsPerTile; });
    m_Median.m_FrameDrawsPerTile = m_MedianBuf[FRAME_SAMPLES_COUNT / 2].m_FrameDrawsPerTile;

    memcpy(m_MedianBuf, m_FrameSamples, sizeof(m_FrameSamples));
    std::sort(std::begin(m_MedianBuf), std::end(m_MedianBuf), [](const DrawStats& a, const DrawStats& b) { return a.m_DrawTimeUS < b.m_DrawTimeUS; });
    m_Median.m_DrawTimeUS = m_MedianBuf[FRAME_SAMPLES_COUNT / 2].m_DrawTimeUS;

    memcpy(m_MedianBuf, m_FrameSamples, sizeof(m_FrameSamples));
    std::sort(std::begin(m_MedianBuf), std::end(m_MedianBuf), [](const DrawStats& a, const DrawStats& b) { return a.m_DrawTimePerThreadUS < b.m_DrawTimePerThreadUS; });
    m_Median.m_DrawTimePerThreadUS = m_MedianBuf[FRAME_SAMPLES_COUNT / 2].m_DrawTimePerThreadUS;

    memcpy(m_MedianBuf, m_FrameSamples, sizeof(m_FrameSamples));
    std::sort(std::begin(m_MedianBuf), std::end(m_MedianBuf), [](const DrawStats& a, const DrawStats& b) { return a.m_FillrateKP < b.m_FillrateKP; });
    m_Median.m_FillrateKP = m_MedianBuf[FRAME_SAMPLES_COUNT / 2].m_FillrateKP;

    memcpy(m_MedianBuf, m_FrameSamples, sizeof(m_FrameSamples));
    std::sort(std::begin(m_MedianBuf), std::end(m_MedianBuf), [](const DrawStats& a, const DrawStats& b) { return a.m_RasterTimeUS < b.m_RasterTimeUS; });
    m_Median.m_RasterTimeUS = m_MedianBuf[FRAME_SAMPLES_COUNT / 2].m_RasterTimeUS;

    memcpy(m_MedianBuf, m_FrameSamples, sizeof(m_FrameSamples));
    std::sort(std::begin(m_MedianBuf), std::end(m_MedianBuf), [](const DrawStats& a, const DrawStats& b) { return a.m_RasterTimePerThreadUS < b.m_RasterTimePerThreadUS; });
    m_Median.m_RasterTimePerThreadUS = m_MedianBuf[FRAME_SAMPLES_COUNT / 2].m_RasterTimePerThreadUS;

    memcpy(m_MedianBuf, m_FrameSamples, sizeof(m_FrameSamples));
    std::sort(std::begin(m_MedianBuf), std::end(m_MedianBuf), [](const DrawStats& a, const DrawStats& b) { return a.m_TransformTimeUS < b.m_TransformTimeUS; });
    m_Median.m_TransformTimeUS = m_MedianBuf[FRAME_SAMPLES_COUNT / 2].m_TransformTimeUS;

    memcpy(m_MedianBuf, m_FrameSamples, sizeof(m_FrameSamples));
    std::sort(std::begin(m_MedianBuf), std::end(m_MedianBuf), [](const DrawStats& a, const DrawStats& b) { return a.m_TransformTimePerThreadUS < b.m_TransformTimePerThreadUS; });
    m_Median.m_TransformTimePerThreadUS = m_MedianBuf[FRAME_SAMPLES_COUNT / 2].m_TransformTimePerThreadUS;

    auto intsqr = [](int64_t v)
    {
        int64_t r = v*v;
        return r;
    };

    m_Avg.m_FPS     = m_Avg.m_DT    ? 1000'000.0f / m_Avg.m_DT : 0;
    m_Min.m_FPS     = m_Max.m_DT    ? 1000'000.0f / m_Max.m_DT : 0;
    m_Max.m_FPS     = m_Min.m_DT    ? 1000'000.0f / m_Min.m_DT : 0;
    m_Median.m_FPS  = m_Median.m_DT ? 1000'000.0f / m_Median.m_DT : 0;

    int64_t StdDev_FPS                      = 0;
    int64_t StdDev_DT                       = 0;
    int64_t StdDev_DrawTimePerThreadUS      = 0;
    int64_t StdDev_DrawTimeUS               = 0;
    int64_t StdDev_FillrateKT               = 0;
    int64_t StdDev_FramePixels              = 0;
    int64_t StdDev_FramePixelsDrawn         = 0;
    int64_t StdDev_FramePixelsCalcualted    = 0;
    int64_t StdDev_FrameTriangles           = 0;
    int64_t StdDev_FrameTrianglesDrawn      = 0;
    int64_t StdDev_FrameDrawsPerTile        = 0;
    int64_t StdDev_RasterTimeUS             = 0;
    int64_t StdDev_RasterTimePerThreadUS    = 0;
    int64_t StdDev_TransformTimeUS          = 0;
    int64_t StdDev_TransformTimePerThreadUS = 0;

    for (auto& sample : m_FrameSamples)
    {
        StdDev_FPS                      += intsqr( 1000'000.0f / sample.m_DT        - m_Median.m_FPS                        );
        StdDev_DT                       += intsqr(sample.m_DT                       - m_Median.m_DT                         );
        StdDev_DrawTimePerThreadUS      += intsqr(sample.m_DrawTimePerThreadUS      - m_Median.m_DrawTimePerThreadUS        );
        StdDev_DrawTimeUS               += intsqr(sample.m_DrawTimeUS               - m_Median.m_DrawTimeUS                 );
        StdDev_FillrateKT               += intsqr(sample.m_FillrateKP               - m_Median.m_FillrateKP                 );
        StdDev_FramePixels              += intsqr(sample.m_FramePixels              - m_Median.m_FramePixels                );
        StdDev_FramePixelsDrawn         += intsqr(sample.m_FramePixelsDrawn         - m_Median.m_FramePixelsDrawn           );
        StdDev_FramePixelsCalcualted    += intsqr(sample.m_FramePixelsCalcualted    - m_Median.m_FramePixelsCalcualted      );
        StdDev_FrameTriangles           += intsqr(sample.m_FrameTriangles           - m_Median.m_FrameTriangles             );
        StdDev_FrameTrianglesDrawn      += intsqr(sample.m_FrameTrianglesDrawn      - m_Median.m_FrameTrianglesDrawn        );
        StdDev_FrameDrawsPerTile        += intsqr(sample.m_FrameDrawsPerTile        - m_Median.m_FrameDrawsPerTile          );
        StdDev_RasterTimeUS             += intsqr(sample.m_RasterTimeUS             - m_Median.m_RasterTimeUS               );
        StdDev_RasterTimePerThreadUS    += intsqr(sample.m_RasterTimePerThreadUS    - m_Median.m_RasterTimePerThreadUS      );
        StdDev_TransformTimeUS          += intsqr(sample.m_TransformTimeUS          - m_Median.m_TransformTimeUS            );
        StdDev_TransformTimePerThreadUS += intsqr(sample.m_TransformTimePerThreadUS - m_Median.m_TransformTimePerThreadUS   );
    }

    m_StdDev.m_FPS                      = int( sqrt( StdDev_FPS                     / double(FRAME_SAMPLES_COUNT) ) );
    m_StdDev.m_DT                       = int( sqrt( StdDev_DT                      / double(FRAME_SAMPLES_COUNT) ) );
    m_StdDev.m_DrawTimePerThreadUS      = int( sqrt( StdDev_DrawTimePerThreadUS     / double(FRAME_SAMPLES_COUNT) ) );
    m_StdDev.m_DrawTimeUS               = int( sqrt( StdDev_DrawTimeUS              / double(FRAME_SAMPLES_COUNT) ) );
    m_StdDev.m_FillrateKP               = int( sqrt( StdDev_FillrateKT              / double(FRAME_SAMPLES_COUNT) ) );
    m_StdDev.m_FramePixels              = int( sqrt( StdDev_FramePixels             / double(FRAME_SAMPLES_COUNT) ) );
    m_StdDev.m_FramePixelsDrawn         = int( sqrt( StdDev_FramePixelsDrawn        / double(FRAME_SAMPLES_COUNT) ) );
    m_StdDev.m_FramePixelsCalcualted    = int( sqrt( StdDev_FramePixelsCalcualted   / double(FRAME_SAMPLES_COUNT) ) );
    m_StdDev.m_FrameTriangles           = int( sqrt( StdDev_FrameTriangles          / double(FRAME_SAMPLES_COUNT) ) );
    m_StdDev.m_FrameTrianglesDrawn      = int( sqrt( StdDev_FrameTrianglesDrawn     / double(FRAME_SAMPLES_COUNT) ) );
    m_StdDev.m_FrameDrawsPerTile        = int( sqrt( StdDev_FrameDrawsPerTile       / double(FRAME_SAMPLES_COUNT) ) );
    m_StdDev.m_RasterTimeUS             = int( sqrt( StdDev_RasterTimeUS            / double(FRAME_SAMPLES_COUNT) ) );
    m_StdDev.m_RasterTimePerThreadUS    = int( sqrt( StdDev_RasterTimePerThreadUS   / double(FRAME_SAMPLES_COUNT) ) );
    m_StdDev.m_TransformTimeUS          = int( sqrt( StdDev_TransformTimeUS         / double(FRAME_SAMPLES_COUNT) ) );
    m_StdDev.m_TransformTimePerThreadUS = int( sqrt(StdDev_TransformTimePerThreadUS / double(FRAME_SAMPLES_COUNT) ) );

    m_Dirty = false;
}


template struct DrawStatsSystem::Data<480>;
template struct DrawStatsSystem::Data<240>;
template struct DrawStatsSystem::Data<120>;
template struct DrawStatsSystem::Data<60>;
template struct DrawStatsSystem::Data<30>;