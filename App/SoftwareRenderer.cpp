/*
* Master’s thesis - Analysis of selected optimization techniques for a 3D software renderer
* Author: Wiktoria Marczyk
* Year: 2025
*/

#include "immintrin.h"

#include "SoftwareRenderer.h"
#include "TransformedVertex.h"
#include "VertexInterpolator.h"
#include <array>
#include <chrono>


bool g_showTilesBoundry = false; ///< zmienna globalna do pokazywania granic kafelków
bool g_showTilestype = false; ///< zmienna globalna do pokazywania typu kafelków
bool g_showTriangleBoundry = false; ///< zmienna globalna do pokazywania granic trójk¹tów
bool g_showCornersClassify = false; ///< zmienna globalna do pokazywania klasyfikacji naro¿ników
bool g_showTilesGrid = false; ///< zmienna globalna do pokazywania siatki kafelków
bool g_ThreadPerTile = true; ///< zmienna globalna do ustawiania zadañ dla w¹tków per kafelek
bool g_MultithreadedTransformAndClip = true; ///< zmienna globalna do ustawiania wielow¹tkowego przekszta³cania i przycinania wierzcho³ków
bool g_TrivialFS = false; ///< zmienna globalna do wy³¹czania obliczania oœwietlenia
bool g_CompressedPartialTile = false; ///< zmienna globalna do ustawiania kompresji czêœciowych kafelków
auto g_max_overdraw = std::atomic<int>{ 0 }; ///< zmienna globalna do przechowywania maksymalnego przes³oniêcia


Vector4f ThreadColors[16] =
{
   { 1.0f, 0.0f, 1.0f, 1.0f },
   { 1.0f, 0.0f, 0.0f, 1.0f },
   { 0.0f, 1.0f, 0.0f, 1.0f },
   { 0.0f, 0.0f, 1.0f, 1.0f },
   { 1.0f, 1.0f, 0.0f, 1.0f },
   { 0.5f, 0.0f, 1.0f, 1.0f },
   { 0.0f, 1.0f, 1.0f, 1.0f },
   { 1.0f, 0.5f, 0.5f, 1.0f },
   { 0.5f, 1.0f, 0.5f, 1.0f },
   { 0.5f, 0.5f, 1.0f, 1.0f },
   { 1.0f, 1.0f, 0.5f, 1.0f },
   { 1.0f, 0.5f, 1.0f, 1.0f },

   { 0.2f, 0.5f, 1.0f, 1.0f },
   { 1.0f, 0.5f, 0.2f, 1.0f },
   { 0.5f, 0.2f, 1.0f, 1.0f },
   { 1.0f, 0.2f, 0.2f, 1.0f },
};

/**
 * Struktura pomocnicza do przechowywania pokrycia kafelka.
 * @tparam T typ danych (np. uint8_t, uint16_t, uint32_t)
 * @tparam size rozmiar tablicy
 */
template< typename T, int size >
struct tile_coverage_impl
{
    static inline constexpr int Bits = size;
    using type = T;

    T Buffer[size] = {}; ///< tablica przechowuj¹ca pokrycie kafelka

    FORCE_INLINE operator T* ()noexcept { return Buffer; }
    FORCE_INLINE T* data()noexcept { return Buffer; }
    FORCE_INLINE const T* data()const noexcept { return Buffer; }
};

template< uint8_t TILE_SIZE >
struct tile_coverage_t;

template<> struct tile_coverage_t<8> : tile_coverage_impl<uint8_t, 8> {};
template<> struct tile_coverage_t<16> : tile_coverage_impl<uint16_t, 16> {};
template<> struct tile_coverage_t<32> : tile_coverage_impl<uint32_t, 32> {};

optional<Vector4f> GetThreadColor(bool UseThreadColor)
{
    optional<Vector4f> Result;
    if (!UseThreadColor)
        return Result;

    if (auto ThreadId = SimpleThreadPool::GetThreadID(); ThreadId >= 0 && ThreadId < 16)
    {
        auto col = ThreadColors[ThreadId];
        col = (col * 0.4f) + 0.6;
        Result = col;
    }
    return Result;
}

/**
 * Struktura przechowuj¹ca dane wspó³dzielone w potoku renderera.
 */
struct PipelineSharedData
{
    Plane          m_NearFrustumPlane; ///< p³aszczyzna bliskiego przyciêcia
    CommandBuffer* m_pProcessTrianglesCmdBuffer = nullptr; ///< bufor komend do przetwarzania trójk¹tów
    SyncBarrier* m_pProcessTrianglesCmdBufferSync = nullptr; ///< bariera synchronizacji bufora komend do przetwarzania trójk¹tów
    CommandBuffer* m_pRenderTilesCmdBuffer = nullptr; ///< bufor komend do renderowania kafelków
    SyncBarrier* m_pRenderTilesCmdBufferSync = nullptr; ///< bariera synchronizacji bufora komend do renderowania kafelków
    DrawConfig* m_pDrawConfig = nullptr; ///< konfiguracja rysowania
};

/**
 * Struktura przechowuj¹ca dane w¹tku renderuj¹cego.
 */
struct SoftwareRenderer::RenderThreadData
{
    monotonic_stack_unsynchronized_memory_resource m_ThreadfastMemResource; ///< szybki zasób pamiêci tymczasowej
    DrawStats* m_pDrawStats = nullptr; ///< wskaŸnik na statystyki rysowania
};

/**
 * Struktura przechowuj¹ca funkcje krawêdzi trójk¹ta oraz ich wartoœci startowe i przyrosty.
 * @tparam T typ danych (np. float, int)
 */
template< typename T >
struct EdgeFunctionRails
{
    EdgeFunctionRails(std::nullptr_t) {}
    EdgeFunctionRails(const Vector2<T>& A, const Vector2<T>& B, const Vector2<T>& C, const Vector2<T>& SP)
        : m_ABP_Stride{ A.y - B.y , B.x - A.x }
        , m_BCP_Stride{ B.y - C.y , C.x - B.x }
        , m_CAP_Stride{ C.y - A.y , A.x - C.x }

        , m_ABP_Start{ m_ABP_Stride.x * SP.x , m_ABP_Stride.y * SP.y - m_ABP_Stride.Dot(A) }
        , m_BCP_Start{ m_BCP_Stride.x * SP.x , m_BCP_Stride.y * SP.y - m_BCP_Stride.Dot(B) }
        , m_CAP_Start{ m_CAP_Stride.x * SP.x , m_CAP_Stride.y * SP.y - m_CAP_Stride.Dot(C) }

        , m_SP{ SP.x , SP.y }
    {
    };

    Vector3<T> GetEdgeFunctionsXStride()const
    {
        return Vector3<T>(m_ABP_Stride.x, m_BCP_Stride.x, m_CAP_Stride.x);
    }
    Vector3<T> GetEdgeFunctionsYStride()const
    {
        return Vector3<T>(m_ABP_Stride.y, m_BCP_Stride.y, m_CAP_Stride.y);
    }

    Vector3<T> GetEdgeFunctionsXStart()const
    {
        return Vector3<T>(m_ABP_Start.x, m_BCP_Start.x, m_CAP_Start.x);
    }
    Vector3<T> GetEdgeFunctionsYStart()const
    {
        return Vector3<T>(m_ABP_Start.y, m_BCP_Start.y, m_CAP_Start.y);
    }

    struct Start
    {
        Vector3<T> x;
        Vector3<T> y;
    };

    Start GetStartFor(const Vector2<T>& P)const
    {
        auto PDiff = P - m_SP;
        Start Result;

        Result.x = GetEdgeFunctionsXStart() + GetEdgeFunctionsXStride() * PDiff.x;
        Result.y = GetEdgeFunctionsYStart() + GetEdgeFunctionsYStride() * PDiff.y;
        return Result;
    };

    Vector2<T> m_ABP_Stride; ///< przyrost funkcji krawêdzi AB
    Vector2<T> m_BCP_Stride; ///< przyrost funkcji krawêdzi BC
    Vector2<T> m_CAP_Stride; ///< przyrost funkcji krawêdzi CA

    Vector2<T> m_ABP_Start; ///< wartoœæ startowa funkcji krawêdzi AB
    Vector2<T> m_BCP_Start; ///< wartoœæ startowa funkcji krawêdzi BC
    Vector2<T> m_CAP_Start; ///< wartoœæ startowa funkcji krawêdzi CA

    Vector2<T> m_SP; ///< punkt startowy (lewy górny róg)
};

/**
 * Typ wyliczeniowy reprezentuj¹cy pokrycie kafelka.
 */
enum eTileCoverage : uint8_t
{
    Undefined = 0b000, // undefined state, should not be used
    Outside = 0b001,
    Partial = 0b011,
    Inside = 0b111,
};

eTileCoverage operator|(eTileCoverage a, eTileCoverage b)
{
    return static_cast<eTileCoverage>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}
eTileCoverage operator&(eTileCoverage a, eTileCoverage b)
{
    return static_cast<eTileCoverage>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
}

template< typename T >
inline T SoftwareRenderer::EdgeFunction(const Vector2<T>& A, const Vector2<T>& B, const Vector2<T>& C)
{
    return (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
}

template< typename T >
inline Vector2<T> SoftwareRenderer::EdgeFunctionSeparate(const Vector2<T>& A, const Vector2<T>& B, const Vector2<T>& C)
{
    return { (B.x - A.x) * (C.y - A.y) , -(B.y - A.y) * (C.x - A.x) };
}

inline void DrawStats::FinishDrawCallStats(Vector2<int> min, Vector2<int> max, int pixelsDrawn)
{
    m_FramePixels += (1 + max.y - min.y) * (1 + max.x - min.x);
    m_FramePixelsDrawn += pixelsDrawn;
    m_FrameTriangles++;
    m_FrameTrianglesDrawn++;
}

shared_ptr<ITexture> SoftwareRenderer::LoadTexture(const char* fileName) const
{
    if (!fileName || fileName[0] == 0)
        return m_DefaultTexture;
    auto texture = std::make_shared<Texture>();
    if (texture->Load(fileName))
        return texture;
    return nullptr;
}

void SoftwareRenderer::ClearScreen()
{
    ZoneScoped;
    //std::fill(m_ScreenBuffer.begin(), m_ScreenBuffer.end(), m_ClearColor);
    //return;
    m_pCommandBuffer->PushCommand<CommandClear>(m_ClearColor);

}

void SoftwareRenderer::ClearZBuffer()
{
    ZoneScoped;
    //std::fill(m_ZBuffer.begin(), m_ZBuffer.end(), 1.f);
    //return;
    m_pCommandBuffer->PushCommand<CommandClear>(std::nullopt, 1.0F);
}

inline Vector4f SoftwareRenderer::FragmentShader(const TransformedVertex& vertex)
{
    if (g_TrivialFS)
        return vertex.m_Color;
    // Normalize the interpolated normal vector
    auto vertexNormal = vertex.m_Normal.FastNormalized();

    Vector4f sampledPixel = m_Texture->Sample(vertex.m_UV);

    Vector3f pointToLightDir = (m_LightPosition - vertex.m_WorldPosition).FastNormalized();

    // ambient - light that is reflected from other objects
    Vector3f ambient = m_AmbientColor * m_AmbientStrength;

    // diffuse - light that is reflected from light source
    float diffuseFactor = std::max(pointToLightDir.Dot(vertexNormal), 0.0f);
    Vector3f diffuse = m_DiffuseColor * diffuseFactor * m_DiffuseStrength;

    // specular - light that is reflected from light source and is reflected in one direction
    // specular = specularStrength * specularColor * pow(max(dot(viewDir, reflectDir), 0.0), shininess)
    Vector3f viewDir = (m_CameraPosition - vertex.m_WorldPosition).FastNormalized();
    Vector3f reflectDir = (pointToLightDir * -1).Reflect(vertexNormal);
    float specularFactor = pow(max(viewDir.Dot(reflectDir), 0.0f), m_Shininess);
    Vector3f specular = m_DiffuseColor * m_SpecularStrength * specularFactor;

    // final light color = (ambient + diffuse + specular) * modelColor
    Vector3f sumOfLight = ambient + diffuse + specular;
    sumOfLight = sumOfLight.CWiseMin(Vector3f(1, 1, 1));
    Vector4f finalColor = Vector4f(sumOfLight, 1.0f) * sampledPixel * vertex.m_Color;

    return finalColor;
}

template< int Elements, eSimdType Type >
inline Vector4<fsimd<Elements, Type>> SoftwareRenderer::FragmentShader(const SimdTransformedVertex<Elements, Type>& vertex)
{
    using float_simd = fsimd<Elements, Type>;
    using Vector3Simd = Vector3<float_simd>;
    using Vector4Simd = Vector4<float_simd>;

    if (g_TrivialFS)
        return vertex.m_Color;

    auto& renderParams = GetRenderParams<Elements, Type>();

    // Normalize the interpolated normal vector
    auto vertexNormal = vertex.m_Normal.FastNormalized();

    Vector4Simd sampledPixel = m_Texture->Sample(vertex.m_UV);

    Vector3Simd pointToLightDir = (renderParams.m_LightPosition - vertex.m_WorldPosition).FastNormalized();

    // ambient - light that is reflected from other objects
    Vector3Simd ambient = renderParams.m_AmbientColor * m_AmbientStrength;

    // diffuse - light that is reflected from light source
    auto diffuseFactor = Math::Max(pointToLightDir.Dot(vertexNormal), float_simd::Zero);
    Vector3Simd diffuse = renderParams.m_DiffuseColor * diffuseFactor * m_DiffuseStrength;

    // specular - light that is reflected from light source and is reflected in one direction
    // specular = specularStrength * specularColor * pow(max(dot(viewDir, reflectDir), 0.0), shininess)
    Vector3Simd viewDir = (renderParams.m_CameraPosition - vertex.m_WorldPosition).FastNormalized();
    Vector3Simd reflectDir = (pointToLightDir * -1).Reflect(vertexNormal);
    auto specularFactor = Math::Max(viewDir.Dot(reflectDir), float_simd::Zero).pow(m_Shininess);
    Vector3Simd specular = renderParams.m_DiffuseColor * m_SpecularStrength * specularFactor;

    // final light color = (ambient + diffuse + specular) * modelColor
    Vector3Simd sumOfLight = ambient + diffuse + specular;
    sumOfLight = sumOfLight.CWiseMin(Vector3Simd(1, 1, 1));
    Vector4Simd finalColor = Vector4Simd(sumOfLight, 1.0f) * sampledPixel * vertex.m_Color;

    return finalColor;
}

/**
 * Struktura definiuj¹ca precyzjê liczb ca³kowitych u¿ywanych w rasteryzacji.
 */
struct IntegerPrecision
{
    int Bits = 4;
    int Multiplier = 1 << Bits;
    int Mask = Multiplier - 1;
};
constexpr IntegerPrecision Precision{ 0 };

/**
 * Struktura przechowuj¹ca dane trójk¹ta potrzebne do rasteryzacji.
 */
struct ALIGN_FOR_AVX TriangleData
{
    TriangleData(std::nullptr_t)
        : m_Interpolator(nullptr)
        , m_EdgeFunctionRails(nullptr)
    {
    }
    TriangleData(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& Color, Vector2<int64_t> SPA, Vector2<int64_t> SPB, Vector2<int64_t> SPC, Vector2<int64_t> SP)
        : m_Interpolator(A, B, C, Color)
        , m_EdgeFunctionRails{ SPA , SPB , SPC , SP }
    {
    }

    VertexInterpolator         m_Interpolator; ///< interpolator wierzcho³ków
    EdgeFunctionRails<int64_t> m_EdgeFunctionRails; ///< funkcje krawêdzi trójk¹ta
    float                      m_InvABC = 0.f; ///< odwrotnoœæ pola trójk¹ta
    float                      m_MinZ = 0.f; ///< minimalna wartoœæ Z w trójk¹cie
    float                      m_MaxZ = 0.f; ///< maksymalna wartoœæ Z w trójk¹cie
    uint32_t                   m_TriIndex = 0; ///< indeks trójk¹ta
};

void SoftwareRenderer::RecreateBuffers(uint8_t TileSize, int inScreenWidth, int inScreenHeight)
{
    if (TileSize <= 8)
    {
        m_TileSize = 8;
        m_TileMode = eTileMode::Tile_8x8;
    }
    else if (TileSize <= 16)
    {
        m_TileSize = 16;
        m_TileMode = eTileMode::Tile_16x16;
    }
    else
    {
        m_TileSize = 32;
        m_TileMode = eTileMode::Tile_32x32;
    }

    m_PublicScreenSize.x = std::clamp(inScreenWidth, 1, 8192);
    m_PublicScreenSize.y = std::clamp(inScreenHeight, 1, 8192);

    const auto TILE_PIXELS_COUNT = m_TileSize * m_TileSize;

    m_ScreenBuffer.resize(m_PublicScreenSize.x * m_PublicScreenSize.y, 0);

    m_ScreenSize.x = (static_cast<uint16_t>(m_PublicScreenSize.x + m_TileSize - 1) / m_TileSize) * m_TileSize;
    m_ScreenSize.y = (static_cast<uint16_t>(m_PublicScreenSize.y + m_TileSize - 1) / m_TileSize) * m_TileSize;

    m_TilesGridSize.x = (m_ScreenSize.x + m_TileSize - 1) / m_TileSize;
    m_TilesGridSize.y = (m_ScreenSize.y + m_TileSize - 1) / m_TileSize;

    m_ZBuffer.resize(m_TilesGridSize.x * m_TilesGridSize.y * TILE_PIXELS_COUNT, 0);
    m_TilesBuffer.resize(m_TilesGridSize.x * m_TilesGridSize.y * TILE_PIXELS_COUNT);

    m_LastTile = Vector2si(m_TilesGridSize.x - 1, m_TilesGridSize.y - 1);
    m_TilesGrid.reset(new TileInfo[m_TilesGridSize.x * m_TilesGridSize.y]);


    for (int y = 0; y < m_TilesGridSize.y; ++y)
    {
        for (int x = 0; x < m_TilesGridSize.x; ++x)
        {
            auto& Tile = m_TilesGrid[y * m_TilesGridSize.x + x];
            Tile.m_TileIndex = Vector2si(x, y);
            Tile.m_TileMemOffset = (y * m_TilesGridSize.x + x) * TILE_PIXELS_COUNT;
            Tile.m_TileZOffset = (y * m_TilesGridSize.x + x) * TILE_PIXELS_COUNT;
        }
    }
}

SoftwareRenderer::SoftwareRenderer(int inScreenWidth, int inScreenHeight)
{
    RecreateBuffers(m_TileSize, inScreenWidth, inScreenHeight);

    m_DefaultTexture = std::make_shared<Texture>();
    m_DefaultTexture->CreateWhite4x4Tex();

    m_Texture = m_DefaultTexture;
    m_TransientMemoryResource.reset();
    m_pCommandBuffer = AllocTransientCommandBuffer();
    m_TileThreadPool.SetThreadCount(16);
}

SoftwareRenderer::~SoftwareRenderer()
{
}

template< typename T >
FORCE_INLINE bool EdgeFunctionTest(const Vector3<T>& EdgeFunctionsValue)
{
    static_assert(std::is_signed_v<T>);

    if constexpr (std::is_integral_v<T>)
    {
        constexpr T NegativeTestBit = T(1) << (sizeof(T) * 8 - 1);

        return !!((EdgeFunctionsValue.x | EdgeFunctionsValue.y | EdgeFunctionsValue.z) & NegativeTestBit);
    }
    else
    {
        return (EdgeFunctionsValue.x | EdgeFunctionsValue.y | EdgeFunctionsValue.z) < 0;
    }
}

template< uint8_t TILE_SIZE >
inline void SoftwareRenderer::DrawTileImpl(const CommandRenderTile& _InitTD, RenderThreadData& data)
{
    ZoneScopedN(Partial ? "DrawTileImpl" : "DrawTileImplFull");
    ZoneColor(0xE0E000);
    const auto pTileInfo = _InitTD.TileInfo;

    using tile_coverage = tile_coverage_t<TILE_SIZE>;

    // local storage for pixels and zbuffer - it seems to be faster than using TileMem directly
    ALIGN_FOR_AVX Vector4f      Pixels[TILE_SIZE * TILE_SIZE];
    ALIGN_FOR_AVX float         ZBuffer[TILE_SIZE * TILE_SIZE];
    ALIGN_FOR_AVX tile_coverage TileCoverage;
    using coverage_line_t = typename tile_coverage::type;

    // lock tile info to prevent concurrent access
    std::scoped_lock lock(pTileInfo->m_Lock);

    const auto pTilePixels = m_TilesBuffer.data() + pTileInfo->m_TileMemOffset;
    const auto pTileZBuffer = m_ZBuffer.data() + pTileInfo->m_TileZOffset;
    auto       pDrawCommand = &_InitTD;
    auto       pZBuffer = pTileZBuffer;
    const bool AlphaBlend = pDrawCommand->DrawControl.m_AlphaBlend;

    if (g_ThreadPerTile)
    {
        // copy tile memory to local storage
        if (AlphaBlend)
            memcpy(Pixels, pTilePixels, sizeof(Pixels));
        memcpy(ZBuffer, pTileZBuffer, sizeof(ZBuffer));

        pDrawCommand = pTileInfo->m_pRenderTileCmd.load(std::memory_order_relaxed);
        pZBuffer = ZBuffer;
    }

    auto       pixelsDrawn = 0;
    int32_t    CommandIndex = 0;
    const auto TilePosition = pTileInfo->m_TileIndex * TILE_SIZE;
    const int  StartY = TilePosition.y;
    const int  EndY = StartY + TILE_SIZE;

    ALIGN_FOR_AVX TransformedVertex interpolatedVertex;

    auto DrawTileIteration = [&]<bool Partial>(const CommandRenderTile * pDrawCommand, Vector4f * pCurPixel, float* pZBuffer, coverage_line_t * pTileCoverage)
    {
        ZoneScopedN("Iteration");
        ZoneColor(CommandIndex > 0 ? 0x00E000 : 0x0000E0);

        const auto pTriangle = pDrawCommand->Triangle;
        const auto EdgeStrideX = pTriangle->m_EdgeFunctionRails.GetEdgeFunctionsXStride().ToVector3<int>();
        const auto EdgeStrideY = pTriangle->m_EdgeFunctionRails.GetEdgeFunctionsYStride().ToVector3<int>();
        const auto _EdgeStart = pTriangle->m_EdgeFunctionRails.GetStartFor(TilePosition.ToVector2<int64_t>());
        const auto EdgeStartX = _EdgeStart.x.ToVector3<int>();
        auto       EdgeStartY = _EdgeStart.y.ToVector3<int>();
        const auto invABC = pTriangle->m_InvABC;
        const bool ZTest = pDrawCommand->DrawControl.m_ZTest;
        const bool ZWrite = pDrawCommand->DrawControl.m_ZWrite;

        // loop through all pixels in tile square
        for (int y = StartY; y < EndY; y++, EdgeStartY += EdgeStrideY)
        {
            auto EdgeFunctions = EdgeStartX + EdgeStartY;

            for (int ix = 0; ix < TILE_SIZE; ix++, EdgeFunctions += EdgeStrideX)
            {
                if constexpr (Partial)
                {
                    if (EdgeFunctionTest(EdgeFunctions))
                        continue; // outside triangle
                }

                Vector3f baricentricCoordinates = EdgeFunctions.ToVector3<float>() * invABC;

                pTriangle->m_Interpolator.InterpolateT<MathCPU>(Vector3f(baricentricCoordinates.y, baricentricCoordinates.z, baricentricCoordinates.x), interpolatedVertex);

                Vector4f col = Vector4f{ 1,1,1,1 };

                float& z = pZBuffer[ix];
                if (interpolatedVertex.m_ScreenPosition.z < z)
                {
                    if (ZWrite)
                        z = interpolatedVertex.m_ScreenPosition.z;
                }
                else if (ZTest)
                {
                    continue;
                }

                Vector4f finalColor = FragmentShader(interpolatedVertex);

                pCurPixel[ix] = finalColor * col;
                pixelsDrawn++;

                pTileCoverage[0] |= 1 << ix;
            }

            pCurPixel += TILE_SIZE;
            pZBuffer += TILE_SIZE; // move to next row in Z-buffer
            pTileCoverage++;
        }
    };

    if (g_ThreadPerTile)
    {
        for (; pDrawCommand; ++CommandIndex, pDrawCommand = pDrawCommand->pNext.load(std::memory_order_relaxed))
        {
            if (pDrawCommand->DrawControl.m_IsFullTile)
                DrawTileIteration.template operator() < false > (pDrawCommand, Pixels, pZBuffer, TileCoverage);
            else
                DrawTileIteration.template operator() < true > (pDrawCommand, Pixels, pZBuffer, TileCoverage);
        }

        if (AlphaBlend)
            memcpy(pTilePixels, Pixels, sizeof(Pixels));
        memcpy(pTileZBuffer, ZBuffer, sizeof(ZBuffer));
    }
    else
    {
        if (pDrawCommand->DrawControl.m_IsFullTile)
            DrawTileIteration.template operator() < false > (pDrawCommand, Pixels, pZBuffer, TileCoverage);
        else
            DrawTileIteration.template operator() < true > (pDrawCommand, Pixels, pZBuffer, TileCoverage);
    }


    auto* pTileCoverage = TileCoverage.data();
    Vector4f* pPixels = Pixels;

    optional<Vector4f> White = GetThreadColor(m_ColorizeThreads);

    for (int y = StartY; y < EndY; y++)
    {
        auto pCurScreenPixel = m_ScreenBuffer.data() + (y * m_ScreenSize.x + TilePosition.x);

        if (White)
        {
            for (uint32_t x = 0, cov = 1; x < TILE_SIZE; ++x, cov <<= 1)
            {
                if (pTileCoverage[0] & cov)
                {
                    pPixels[x].xyz() = (pPixels[x].xyz() * 0.8f) + Vector3f(0.2f, 0.2f, 0.2f);
                    pCurScreenPixel[x] = Vector4f::ToARGB(pPixels[x] * *White);
                }
            }
        }
        else
        {
            for (uint32_t x = 0, cov = 1; x < TILE_SIZE; ++x, cov <<= 1)
            {
                if (pTileCoverage[0] & cov)
                    pCurScreenPixel[x] = Vector4f::ToARGB(pPixels[x]);
            }
        }
        pTileCoverage++;
        pPixels += TILE_SIZE; // move to next row in Pixels
    }

    if (data.m_pDrawStats)
    {
        data.m_pDrawStats->m_FramePixelsDrawn += pixelsDrawn;
        data.m_pDrawStats->m_FramePixelsCalcualted += pixelsDrawn;
        data.m_pDrawStats->m_FramePixels += TILE_SIZE * TILE_SIZE;
        data.m_pDrawStats->m_FrameDrawsPerTile += CommandIndex;
    }
}

template< eSimdType Type, int Elements >
void ARGB_ToLanesAVXTile(const Vector2si& TilePos, uint32_t TileSize, const uint32_t* pScreenBuf, Vector4f* pOut, Vector2si ScreenSize)
{
    const auto LANE_SIZE = TileSize * TileSize / 4;
    using simd_float = fsimd<Elements, Type>;
    using simd_int = isimd<Elements, Type>;

    float* Out[4] =
    {
        pOut->data() + LANE_SIZE * 0,
        pOut->data() + LANE_SIZE * 1,
        pOut->data() + LANE_SIZE * 2,
        pOut->data() + LANE_SIZE * 3
    };

    simd_int   ByteMask = int(0xFF);
    simd_float Mul = (1.0f / 255.0f);

    for (int y = TilePos.y, i = 0; y < TileSize + TilePos.y; y++)
    {
        const int32_t* In = reinterpret_cast<const int32_t*>(pScreenBuf) + y * ScreenSize.x + TilePos.x;

        for (int x = 0; x < TileSize; x += simd_float::elements_count, i += simd_float::elements_count)
        {
            simd_int P{ In + i , simd_alignment::AVX };

            // Unpack ARGB to 4 lanes
            const auto A = ((P >> 24) & ByteMask).template static_cast_to<float>() * Mul;
            const auto R = ((P >> 16) & ByteMask).template static_cast_to<float>() * Mul;
            const auto G = ((P >> 8) & ByteMask).template static_cast_to<float>() * Mul;
            const auto B = ((P)&ByteMask).template static_cast_to<float>() * Mul;

            // Store unpacked values to output lanes
            B.store(Out[0] + i, simd_alignment::AVX);
            G.store(Out[1] + i, simd_alignment::AVX);
            R.store(Out[2] + i, simd_alignment::AVX);
            A.store(Out[3] + i, simd_alignment::AVX);
        }
    }
}

FORCE_INLINE uint32_t SwapBytes(uint32_t data)
{
    data = (data >> 16) | (data << 16);
    data = ((data & 0x00FF00FF) << 8) | ((data & 0xFF00FF00) >> 8);
    return data;
}

FORCE_INLINE uint16_t SwapBytes(uint16_t data)
{
    data = (data >> 8) | (data << 8);
    return data;
}

FORCE_INLINE uint16_t SwapBytes(uint8_t data)
{
    return data;
}

template< eSimdType Type, int Elements, bool Colorize, int TILE_SIZE >
uint32_t Lanes_To_ARGBTile(const Vector2si& TilePos, const Vector4f* pPixels, const tile_coverage_t<TILE_SIZE>& CoverageData, uint32_t* pScreenBuf, optional<Vector4f> threadColor, Vector2si ScreenSize)
{
    constexpr auto LANE_SIZE = TILE_SIZE * TILE_SIZE;

    using simd_float = fsimd<Elements, Type>;
    using simd_int = isimd<Elements, Type>;

    const float* In[4] =
    {
        pPixels->data() + LANE_SIZE * 0,
        pPixels->data() + LANE_SIZE * 1,
        pPixels->data() + LANE_SIZE * 2,
        pPixels->data() + LANE_SIZE * 3
    };

    simd_int    ByteMask = int(0xFF);
    simd_float  Mul = 255.0f;
    uint32_t    PixelsDrawn = 0;
    auto        pCoverageData = CoverageData.data();

    for (int y = TilePos.y, d = 0; y < TILE_SIZE + TilePos.y; y++)
    {
        int32_t* Out = reinterpret_cast<int32_t*>(pScreenBuf) + y * ScreenSize.x + TilePos.x;

        auto CoverageLine = SwapBytes(pCoverageData[0]);
        if constexpr (simd_float::elements_count == 4)
            CoverageLine = ((CoverageLine & 0x0F0F0F0F) << 4) | ((CoverageLine & 0xF0F0F0F0) >> 4);

        simd_int Mask = simd_int{ int(CoverageLine) } << (simd_int::NToZero + (32 - CoverageData.Bits));
        PixelsDrawn += Math::CountBitsSetTo1(pCoverageData[0]);

        for (int x = 0; x < TILE_SIZE; x += simd_float::elements_count, d += simd_float::elements_count)
        {
            simd_float B{ In[0] + d , simd_alignment::AVX };
            simd_float G{ In[1] + d , simd_alignment::AVX };
            simd_float R{ In[2] + d , simd_alignment::AVX };
            simd_float A{ In[3] + d , simd_alignment::AVX };

            if constexpr (Colorize)
            {
                // Apply thread color if available
                if (threadColor)
                {
                    B = (B * 0.8f + 0.2f) * threadColor->x;
                    G = (G * 0.8f + 0.2f) * threadColor->y;
                    R = (R * 0.8f + 0.2f) * threadColor->z;
                    A *= threadColor->w;
                }
            }

            // Pack 4 lanes to ARGB
            simd_int P = (((A * Mul).template static_cast_to<int>() & ByteMask) << 24) |
                (((R * Mul).template static_cast_to<int>() & ByteMask) << 16) |
                (((G * Mul).template static_cast_to<int>() & ByteMask) << 8) |
                (((B * Mul).template static_cast_to<int>() & ByteMask));

            // Store packed ARGB to output
            P.store(Out + x, Mask);

            Mask <<= simd_float::elements_count;
        }
        pCoverageData++;
    }

    return PixelsDrawn;
}

template< uint8_t TILE_SIZE, eSimdType Type, int Elements  >
inline void SoftwareRenderer::DrawTileImplSimd(const CommandRenderTile& _InitTD, RenderThreadData& data)
{
    using f256t = fsimd<Elements, Type>;
    using i256t = isimd<Elements, Type>;
    using Vector3i256t = Vector3<i256t>;
    using Vector3f256t = Vector3<f256t>;
    using Vector4f256t = Vector4<f256t>;
    using TransformedVertexT = SimdTransformedVertex<Elements, Type>;
    using tile_coverage = tile_coverage_t<TILE_SIZE>;
    using coverage_line_t = typename tile_coverage::type;
    constexpr auto LANE_SIZE = TILE_SIZE * TILE_SIZE;
    constexpr int pack_size = f256t::elements_count;
    constexpr auto PackCoverageMask = (coverage_line_t(1) << pack_size) - 1;

    ZoneScopedN(Partial ? "DrawTileImplSimd" : "DrawTileImplSimdFull");
    ZoneColor(0xE0E000);
    const auto pTileInfo = _InitTD.TileInfo;

    // local storage for pixels and zbuffer - it seems to be faster than using TileMem directly
    ALIGN_FOR_AVX Vector4f      Pixels[TILE_SIZE * TILE_SIZE];
    ALIGN_FOR_AVX float         ZBuffer[TILE_SIZE * TILE_SIZE];
    ALIGN_FOR_AVX tile_coverage TileCoverage;
    const auto                  TilePosition = pTileInfo->m_TileIndex * TILE_SIZE;
    const auto                  pTilePixels = m_TilesBuffer.data() + pTileInfo->m_TileMemOffset;
    const auto                  pTileZBuffer = m_ZBuffer.data() + pTileInfo->m_TileZOffset;
    float* const                 ZBufferStart = g_ThreadPerTile ? ZBuffer : pTileZBuffer;
    auto                        pDrawCommand = g_ThreadPerTile ? pTileInfo->m_pRenderTileCmd.load(std::memory_order_relaxed) : &_InitTD;
    const bool                  AlphaBlend = pDrawCommand->DrawControl.m_AlphaBlend;

    // lock tile info to prevent concurrent access
    std::scoped_lock lock(pTileInfo->m_Lock);

    // copy tile memory to local storage
    if (g_ThreadPerTile)
    {
        if (AlphaBlend)
            ARGB_ToLanesAVXTile<Type, Elements>(TilePosition, TILE_SIZE, m_ScreenBuffer.data(), Pixels, m_ScreenSize);
        memcpy(ZBuffer, pTileZBuffer, sizeof(ZBuffer));
    }

    auto       pixelsDrawn = 0;
    auto       pixelsCalc = 0;
    int32_t    CommandIndex = 0;
    const int  StartY = TilePosition.y;
    const int  EndY = StartY + TILE_SIZE;


    auto DrawTileIteration = [&]<bool Partial>(const CommandRenderTile * pDrawCommand, float* pCurPixel, float* pZBuffer, coverage_line_t * CoverageMask)
    {

        ZoneScopedN("Iteration");
        ZoneColor(CommandIndex > 0 ? 0x00E000 : 0x0000E0);

        const auto pTriangle = pDrawCommand->Triangle;
        const auto _EdgeStrideX = pTriangle->m_EdgeFunctionRails.GetEdgeFunctionsXStride().ToVector3<int>().Swizzle<1, 2, 0>();
        const auto _EdgeStrideY = pTriangle->m_EdgeFunctionRails.GetEdgeFunctionsYStride().ToVector3<int>().Swizzle<1, 2, 0>();
        auto EdgeStrideX = Vector3i256t{ _EdgeStrideX.x , _EdgeStrideX.y , _EdgeStrideX.z };
        const auto EdgeStrideY = Vector3i256t{ _EdgeStrideY.x , _EdgeStrideY.y , _EdgeStrideY.z };
        const auto _EdgeStart = pTriangle->m_EdgeFunctionRails.GetStartFor(TilePosition.ToVector2<int64_t>());
        const auto _EdgeStartX = _EdgeStart.x.ToVector3<int>().Swizzle<1, 2, 0>();
        const auto _EdgeStartY = _EdgeStart.y.ToVector3<int>().Swizzle<1, 2, 0>();
        const auto EdgeStartX = Vector3i256t{ _EdgeStartX.x , _EdgeStartX.y , _EdgeStartX.z } + EdgeStrideX * i256t::ZeroToN;
        auto EdgeStartY = Vector3i256t{ _EdgeStartY.x , _EdgeStartY.y , _EdgeStartY.z };
        const auto invABC = f256t{ pTriangle->m_InvABC };
        const bool ZTest = pDrawCommand->DrawControl.m_ZTest;
        const bool ZWrite = pDrawCommand->DrawControl.m_ZWrite;
        EdgeStrideX *= pack_size;

        TransformedVertexT      interpolatedVertex;
        i256t                   write_mask;
        auto cmdPixelsDrawn = 0;
        auto cmdPixelsCalc = 0;

        // loop through all pixels in tile square
        for (int y = StartY; y < EndY; y++, EdgeStartY += EdgeStrideY)
        {
            // Calculate edge functions for the first pixel in the current line
            auto EdgeFunctions = EdgeStartX + EdgeStartY;

            coverage_line_t LineCoverageMask = 0;

            for (int ix = 0; ix < TILE_SIZE
                ; ix += pack_size
                , EdgeFunctions += EdgeStrideX
                )
            {
                // prepare coverage mask for bits from current pixel pack
                LineCoverageMask >>= pack_size;

                // initialize coverage with 1s - initially we assume that all pixels in current pack will be written
                coverage_line_t CurrentPackCoverageMask = std::numeric_limits<coverage_line_t>::max();

                if constexpr (Partial)
                {
                    write_mask = (EdgeFunctions.x | EdgeFunctions.y | EdgeFunctions.z) >= i256t::Zero;

                    // convert write_mask to 32-bit mask (8 least significant bits represent coverage for each pixel in the pack)
                    CurrentPackCoverageMask = write_mask.to_mask_32();

                    // perform edge function test for current pixel pack
                    if (!CurrentPackCoverageMask)
                        // all pixels failed edge function test - all of them are outside triangle - we can skip further processing
                        continue;
                }
                else
                {
                    // if we are drawing full tile, we assume that initially all pixels are inside triangle
                    write_mask = i256t::AllBitsSet;
                }

                // interpolate z value
                auto baricentricCoordinates = EdgeFunctions.ToVector3<f256t>() * invABC;
                pTriangle->m_Interpolator.InterpolateZ(baricentricCoordinates, interpolatedVertex);

                // load z-values for current pixel pack from Z-buffer
                f256t CurrentZValue(pZBuffer + ix, simd_alignment::AVX);

                if (ZTest)
                {
                    // Perform Z-test on whole pixel pack:
                    //      All pixels with z-value less than current z-value will generate 1 in write_mask, rest will generate 0
                    write_mask &= (interpolatedVertex.m_ScreenPosition.z < CurrentZValue).reinterpret_cast_to<int>();

                    // Select pixels that passed the Z-test - bit 1 will select second value, 0 will select first value
                    CurrentZValue = CurrentZValue.select(interpolatedVertex.m_ScreenPosition.z, write_mask);

                    if (ZWrite)
                        // store the current z-value back to the Z-buffer (we can write all pixels at once because we changed only pixels that passed the Z-test)
                        CurrentZValue.store(pZBuffer + ix, simd_alignment::AVX);

                    // convert write_mask to 8 least significant bits representing coverage for each pixel in the pack
                    CurrentPackCoverageMask &= write_mask.to_mask_32();
                    if (!CurrentPackCoverageMask)
                        continue; // zest failed for all pixels - no need to interpolate color
                }
                else if (ZWrite)
                {
                    // Only z-write is enabled, so we simply store the z-value for all pixels that passed edge function test
                    CurrentZValue = CurrentZValue.select(interpolatedVertex.m_ScreenPosition.z, write_mask);

                    CurrentZValue.store(pZBuffer + ix, simd_alignment::AVX);
                }

                // Interpolate all other vertex attributes (UV, normal, etc.) for all pixels in the current pack
                pTriangle->m_Interpolator.InterpolateAllButZ(baricentricCoordinates, interpolatedVertex);

                // Call fragment shader to compute final color for all pixels in the current pack
                Vector4f256t finalColor = FragmentShader(interpolatedVertex);

                simd_array_lvl2<float, 1, LANE_SIZE, eDataAlignment::AVX> data{ pCurPixel + ix };

                // If alpha blending is enabled, we need to blend the final color with the current pixel color
                if (AlphaBlend)
                {
                    Vector3f256t curValue{ data };

                    finalColor.xyz() = finalColor.xyz() * finalColor.w + curValue * (f256t::One - finalColor.w);
                }

                // store the final color in the local pixel buffer - we will write it to the screen buffer later
                finalColor.store(data, write_mask);

                cmdPixelsCalc += pack_size;
                if constexpr (Partial)
                    cmdPixelsDrawn += Math::CountBitsSetTo1(CurrentPackCoverageMask);

                // Add mask of current pixel pack to the coverage mask
                LineCoverageMask |= CurrentPackCoverageMask << (TileCoverage.Bits - pack_size); // accumulate coverage mask for the whole pixel pack
            }

            CoverageMask[0] |= LineCoverageMask;
            CoverageMask++;
            pCurPixel += TILE_SIZE;
            pZBuffer += TILE_SIZE;
        }
        if constexpr (!Partial)
            cmdPixelsDrawn = cmdPixelsCalc;

        pixelsCalc += cmdPixelsCalc;
        pixelsDrawn += cmdPixelsDrawn;

    };

    auto DrawTileIterationCompressed = [&]<bool Partial>(const CommandRenderTile * pDrawCommand, float* const pCurPixel, float* pZBuffer, coverage_line_t* const CoverageMask)
    {

        ZoneScopedN("Iteration compressed");
        ZoneColor(CommandIndex > 0 ? 0x00E000 : 0x0000E0);

        constexpr int pack_size = f256t::elements_count;
        const auto pTriangle = pDrawCommand->Triangle;
        const auto _EdgeStrideX = pTriangle->m_EdgeFunctionRails.GetEdgeFunctionsXStride().ToVector3<int>().Swizzle<1, 2, 0>();
        const auto _EdgeStrideY = pTriangle->m_EdgeFunctionRails.GetEdgeFunctionsYStride().ToVector3<int>().Swizzle<1, 2, 0>();
        auto EdgeStrideX = Vector3i256t{ _EdgeStrideX.x , _EdgeStrideX.y , _EdgeStrideX.z };
        const auto EdgeStrideY = Vector3i256t{ _EdgeStrideY.x , _EdgeStrideY.y , _EdgeStrideY.z };
        const auto _EdgeStart = pTriangle->m_EdgeFunctionRails.GetStartFor(TilePosition.ToVector2<int64_t>());
        const auto _EdgeStartX = _EdgeStart.x.ToVector3<int>().Swizzle<1, 2, 0>();
        const auto _EdgeStartY = _EdgeStart.y.ToVector3<int>().Swizzle<1, 2, 0>();
        const auto EdgeStartX = Vector3i256t{ _EdgeStartX.x , _EdgeStartX.y , _EdgeStartX.z } + EdgeStrideX * i256t::ZeroToN;
        auto EdgeStartY = Vector3i256t{ _EdgeStartY.x , _EdgeStartY.y , _EdgeStartY.z };
        const auto invABC = f256t{ pTriangle->m_InvABC };
        const bool ZTest = pDrawCommand->DrawControl.m_ZTest;
        const bool ZWrite = pDrawCommand->DrawControl.m_ZWrite;
        EdgeStrideX *= pack_size;

        TransformedVertexT      interpolatedVertex;
        i256t                   write_mask;

        ALIGN_FOR_AVX float         PixelsWorkBuffer[LANE_SIZE * 4];
        ALIGN_FOR_AVX tile_coverage LocalTileCoverage;

        auto pPixel = PixelsWorkBuffer;
        auto pCoverage = LocalTileCoverage.data();
        Vector3f256t baricentricCoordinates;

        // loop through all pixels in tile square
        for (int y = StartY; y < EndY; y++, EdgeStartY += EdgeStrideY)
        {
            // Calculate edge functions for the first pixel in the current line
            auto EdgeFunctions = EdgeStartX + EdgeStartY;

            coverage_line_t LineCoverageMask = 0;

            for (int ix = 0; ix < TILE_SIZE
                ; ix += pack_size
                , EdgeFunctions += EdgeStrideX
                )
            {
                // prepare coverage mask for bits from current pixel pack
                LineCoverageMask >>= pack_size;

                // initialize coverage with 1s - initially we assume that all pixels in current pack will be written
                coverage_line_t CurrentPackCoverageMask = std::numeric_limits<coverage_line_t>::max();

                // compute edge functions for current pixel pack and generate mask with 1s for pixels that are inside triangle
                write_mask = (EdgeFunctions.x | EdgeFunctions.y | EdgeFunctions.z) >= i256t::Zero;

                // convert write_mask to 32-bit mask (8 least significant bits represent coverage for each pixel in the pack)
                CurrentPackCoverageMask = write_mask.to_mask_32();

                // perform edge function test for current pixel pack
                if (!CurrentPackCoverageMask)
                    // all pixels failed edge function test - all of them are outside triangle - we can skip further processing
                    continue;

                // interpolate z value
                baricentricCoordinates = EdgeFunctions.ToVector3<f256t>() * invABC;
                pTriangle->m_Interpolator.InterpolateZ(baricentricCoordinates, interpolatedVertex);

                if (ZTest)
                {
                    // load z-values for current pixel pack from Z-buffer
                    f256t CurrentZValue(pZBuffer + ix, simd_alignment::AVX);

                    // Perform Z-test on whole pixel pack:
                    //      All pixels with z-value less than current z-value will generate 1 in write_mask, rest will generate 0
                    write_mask &= (interpolatedVertex.m_ScreenPosition.z < CurrentZValue).reinterpret_cast_to<int>();

                    // Select pixels that passed the Z-test - bit 1 will select second value, 0 will select first value
                    CurrentZValue = CurrentZValue.select(interpolatedVertex.m_ScreenPosition.z, write_mask);

                    if (ZWrite)
                        // store the current z-value back to the Z-buffer (we can write all pixels at once because we changed only pixels that passed the Z-test)
                        CurrentZValue.store(pZBuffer + ix, simd_alignment::AVX);

                    // convert write_mask to 8 least significant bits representing coverage for each pixel in the pack
                    CurrentPackCoverageMask &= write_mask.to_mask_32();
                    if (!CurrentPackCoverageMask)
                        continue; // zest failed for all pixels - no need to interpolate color
                }
                else if (ZWrite)
                {
                    // load z-values for current pixel pack from Z-buffer
                    f256t CurrentZValue(pZBuffer + ix, simd_alignment::AVX);

                    // Only z-write is enabled, so we simply store the z-value for all pixels that passed edge function test
                    CurrentZValue = CurrentZValue.select(interpolatedVertex.m_ScreenPosition.z, write_mask);

                    CurrentZValue.store(pZBuffer + ix, simd_alignment::AVX);
                }

                // Add mask for current pixel pack to the coverage mask
                LineCoverageMask |= CurrentPackCoverageMask << (TileCoverage.Bits - pack_size);

                // compress all barycentric coordinates and z-values so we don't have discard pixels in work buffer
                f256t::compressed_store_4(&baricentricCoordinates.x
                    , &baricentricCoordinates.y
                    , &baricentricCoordinates.z
                    , &interpolatedVertex.m_ScreenPosition.z
                    , pPixel + LANE_SIZE * 0
                    , pPixel + LANE_SIZE * 1
                    , pPixel + LANE_SIZE * 2
                    , pPixel + LANE_SIZE * 3, write_mask, std::true_type{});

                pPixel += Math::CountBitsSetTo1(CurrentPackCoverageMask);
            }

            pZBuffer += TILE_SIZE;

            pCoverage[0] |= LineCoverageMask; // accumulate coverage mask for the whole tile
            pCoverage++;
        }

        pixelsDrawn += pPixel - PixelsWorkBuffer;
        pixelsCalc += pPixel - PixelsWorkBuffer;

        for (auto pWork = PixelsWorkBuffer; pWork < pPixel; pWork += pack_size)
        {
            simd_array_lvl2<float, 1, LANE_SIZE, eDataAlignment::AVX> WorkBufferData{ pWork };

            baricentricCoordinates.load(WorkBufferData);
            interpolatedVertex.m_ScreenPosition.z.load(WorkBufferData[3]);

            // Interpolate all other vertex attributes (UV, normal, etc.) for all pixels in the current pack
            pTriangle->m_Interpolator.InterpolateAllButZ(baricentricCoordinates, interpolatedVertex);

            // Call fragment shader to compute final color for all pixels in the current pack
            Vector4f256t finalColor = FragmentShader(interpolatedVertex);

            finalColor.store(WorkBufferData);
        }


        auto pWritePixel = pCurPixel;
        auto pReadPixel = PixelsWorkBuffer;
        pCoverage = LocalTileCoverage.data();

        for (size_t y = 0; y < TILE_SIZE; ++y)
        {
            auto CoverageLine = *pCoverage++;
            if (!CoverageLine)
            {
                pWritePixel += TILE_SIZE; // skip empty line
                continue;
            }

            CoverageMask[y] |= CoverageLine;
            Vector4f256t finalColor;

            for (size_t x = 0; x < TILE_SIZE; x += pack_size, pWritePixel += pack_size)
            {
                // mask out bits that are not in the current pack
                coverage_line_t PackCoverage = CoverageLine & PackCoverageMask;
                // shift coverage line to the next pack
                CoverageLine >>= pack_size;
                if (!PackCoverage)
                    continue; // no pixels in this pack

                simd_array_lvl2<const float, 1, LANE_SIZE, eDataAlignment::AVX> WorkBufferData{ pReadPixel };

                auto mask = i256t::from_mask_32(PackCoverage);

                f256t::expand_load_4(&finalColor.x
                    , &finalColor.y
                    , &finalColor.z
                    , &finalColor.w
                    , WorkBufferData[0].data()
                    , WorkBufferData[1].data()
                    , WorkBufferData[2].data()
                    , WorkBufferData[3].data()
                    , mask);

                pReadPixel += Math::CountBitsSetTo1(PackCoverage);

                simd_array_lvl2<float, 1, LANE_SIZE, eDataAlignment::AVX> DstData{ pWritePixel };

                // If alpha blending is enabled, we need to blend the final color with the current pixel color
                if (AlphaBlend)
                {
                    Vector3f256t curValue{ DstData };

                    finalColor.xyz() = finalColor.xyz() * finalColor.w + curValue * (f256t::One - finalColor.w);
                }

                finalColor.store(DstData, mask);
            }
        }

    };

    for (; pDrawCommand; ++CommandIndex, pDrawCommand = pDrawCommand->pNext.load(std::memory_order_relaxed))
    {
        if (pDrawCommand->DrawControl.m_IsFullTile)
            DrawTileIteration.template operator() < false > (pDrawCommand, Pixels->data(), ZBufferStart, TileCoverage);
        else if (g_CompressedPartialTile)
            DrawTileIterationCompressed.template operator() < true > (pDrawCommand, Pixels->data(), ZBufferStart, TileCoverage);
        else
            DrawTileIteration.template operator() < true > (pDrawCommand, Pixels->data(), ZBufferStart, TileCoverage);

        if (!g_ThreadPerTile)
            break;
    }

    optional<Vector4f> White = GetThreadColor(m_ColorizeThreads);

    {

        //if( AlphaBlend )
        //    memcpy( pTilePixels  , Pixels  , sizeof(Pixels)  );

        if (g_ThreadPerTile)
            memcpy(pTileZBuffer, ZBuffer, sizeof(ZBuffer));

        Lanes_To_ARGBTile<Type, Elements, true, TILE_SIZE>(TilePosition, Pixels, TileCoverage, m_ScreenBuffer.data(), White, m_ScreenSize);
    }


    // update max overdraw count
    auto old_max_overdraw = g_max_overdraw.load(std::memory_order_relaxed);
    for (;;)
    {
        CommandIndex = std::max(old_max_overdraw, CommandIndex);
        if (g_max_overdraw.compare_exchange_strong(old_max_overdraw, CommandIndex, std::memory_order_relaxed))
            break;
    }

    // update pixels draw stats
    if (data.m_pDrawStats)
    {
        data.m_pDrawStats->m_FramePixelsDrawn += pixelsDrawn;
        data.m_pDrawStats->m_FramePixelsCalcualted += pixelsCalc;
        data.m_pDrawStats->m_FramePixels += TILE_SIZE * TILE_SIZE;
        data.m_pDrawStats->m_FrameDrawsPerTile += CommandIndex;
    }
}

constexpr auto TileMode(eTileMode TileMode, eBlockMathMode BlockMode)
{
    return (uint32_t(TileMode) << 3) | uint32_t(BlockMode);
}

void SoftwareRenderer::DrawTile(const CommandRenderTile& TD, RenderThreadData& data)
{
    auto Now = std::chrono::high_resolution_clock::now();

    switch (TileMode(m_TileMode, m_BlockMathMode))
    {
    case TileMode(eTileMode::Tile_32x32, eBlockMathMode::CPUx32):  DrawTileImpl    <32>(TD, data);                      break;
    case TileMode(eTileMode::Tile_32x32, eBlockMathMode::CPUx256):  DrawTileImplSimd<32, eSimdType::CPU, 8>(TD, data);    break;
    case TileMode(eTileMode::Tile_32x32, eBlockMathMode::SSEx128):  DrawTileImplSimd<32, eSimdType::SSE, 4>(TD, data);    break;
    case TileMode(eTileMode::Tile_32x32, eBlockMathMode::SSEx256):  DrawTileImplSimd<32, eSimdType::SSE, 8>(TD, data);    break;
    case TileMode(eTileMode::Tile_32x32, eBlockMathMode::AVXx256):  DrawTileImplSimd<32, eSimdType::AVX, 8>(TD, data);    break;

    case TileMode(eTileMode::Tile_16x16, eBlockMathMode::CPUx32):  DrawTileImpl    <16>(TD, data);                      break;
    case TileMode(eTileMode::Tile_16x16, eBlockMathMode::CPUx256):  DrawTileImplSimd<16, eSimdType::CPU, 8>(TD, data);    break;
    case TileMode(eTileMode::Tile_16x16, eBlockMathMode::SSEx128):  DrawTileImplSimd<16, eSimdType::SSE, 4>(TD, data);    break;
    case TileMode(eTileMode::Tile_16x16, eBlockMathMode::SSEx256):  DrawTileImplSimd<16, eSimdType::SSE, 8>(TD, data);    break;
    case TileMode(eTileMode::Tile_16x16, eBlockMathMode::AVXx256):  DrawTileImplSimd<16, eSimdType::AVX, 8>(TD, data);    break;

    case TileMode(eTileMode::Tile_8x8, eBlockMathMode::CPUx32):  DrawTileImpl    < 8>(TD, data);                      break;
    case TileMode(eTileMode::Tile_8x8, eBlockMathMode::CPUx256):  DrawTileImplSimd< 8, eSimdType::CPU, 8>(TD, data);    break;
    case TileMode(eTileMode::Tile_8x8, eBlockMathMode::SSEx128):  DrawTileImplSimd< 8, eSimdType::SSE, 4>(TD, data);    break;
    case TileMode(eTileMode::Tile_8x8, eBlockMathMode::SSEx256):  DrawTileImplSimd< 8, eSimdType::SSE, 8>(TD, data);    break;
    case TileMode(eTileMode::Tile_8x8, eBlockMathMode::AVXx256):  DrawTileImplSimd< 8, eSimdType::AVX, 8>(TD, data);    break;
    }

    data.m_pDrawStats->m_RasterTimeUS += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - Now).count();
}

eTileCoverage ClassifyX(Vector2<int64_t> A, Vector2<int64_t> B, Vector2<int64_t> C, Vector2<int64_t> TilePos, const Vector2<int64_t> Close[3], const Vector2<int64_t> Far[3])
{
    //Vector2i TilePos = TileCell * TILE_SIZE * 16;

    int64_t Dists[6] =
    {
        SoftwareRenderer::EdgeFunction(A , B , TilePos + Close[0]),
        SoftwareRenderer::EdgeFunction(A , B , TilePos + Far[0]),
        SoftwareRenderer::EdgeFunction(B , C , TilePos + Close[1]),
        SoftwareRenderer::EdgeFunction(B , C , TilePos + Far[1]),
        SoftwareRenderer::EdgeFunction(C , A , TilePos + Close[2]),
        SoftwareRenderer::EdgeFunction(C , A , TilePos + Far[2]),
    };


    eTileCoverage E0 = (Dists[0] < 0 ? eTileCoverage(0x1) : eTileCoverage(0x3))
        | (Dists[1] < 0 ? eTileCoverage(0x1) : eTileCoverage(0x5));
    eTileCoverage E1 = (Dists[2] < 0 ? eTileCoverage(0x1) : eTileCoverage(0x3))
        | (Dists[3] < 0 ? eTileCoverage(0x1) : eTileCoverage(0x5));
    eTileCoverage E2 = (Dists[4] < 0 ? eTileCoverage(0x1) : eTileCoverage(0x3))
        | (Dists[5] < 0 ? eTileCoverage(0x1) : eTileCoverage(0x5));


    return E0 & E1 & E2;
}

inline void ImGuiAddLine(ImVec2 p1, ImVec2 p2, ImU32 col, Vector2si ScreenSize, float thickness = 1)
{
    p1.y = ScreenSize.y - p1.y;
    p2.y = ScreenSize.y - p2.y;

    ImGui::GetBackgroundDrawList()->AddLine(p1, p2, col, thickness);
}

inline void ImGuiAddRectFilled(ImVec2 p_min, ImVec2 p_max, ImU32 col, Vector2si ScreenSize, float rounding = 0.0f, ImDrawFlags flags = 0)
{
    p_min.y = ScreenSize.y - p_min.y;
    p_max.y = ScreenSize.y - p_max.y;

    ImGui::GetBackgroundDrawList()->AddRectFilled(p_min, p_max, col, rounding, flags);
}


inline void ImGuiAddRect(ImVec2 p_min, ImVec2 p_max, ImU32 col, Vector2si ScreenSize, float rounding = 0.0f, ImDrawFlags flags = 0, float thickness = 1.0f)
{
    p_min.y = ScreenSize.y - p_min.y;
    p_max.y = ScreenSize.y - p_max.y;
    ImGui::GetBackgroundDrawList()->AddRect(p_min, p_max, col, rounding, flags, thickness);
}

inline void ImGuiAddX(ImVec2 p_min, ImVec2 p_max, ImU32 col, Vector2si ScreenSize, float thickness = 1.0f)
{
    ImVec2 P[4] = { ImVec2{ p_min.x , p_min.y }
                  , ImVec2{ p_max.x , p_min.y }
                  , ImVec2{ p_min.x , p_max.y }
                  , ImVec2{ p_max.x , p_max.y } };

    ImGuiAddLine(P[0], P[3], col, ScreenSize, thickness);
    ImGuiAddLine(P[1], P[2], col, ScreenSize, thickness);
}

void SoftwareRenderer::GenerateTileJobs(const TransformedVertex& VA, const TransformedVertex& VB, const TransformedVertex& VC, DrawStats& stats, const PipelineSharedData* pPipelineSharedData, uint32_t tri_index, pmr::vector<const Command*>& outCommmands)
{
    ZoneScoped;
    ZoneColor(0x00E0E0);
    // filling algorithm is working that way that we are going through all pixels in rectangle that is created by min and max points
    // and we are checking if pixel is inside triangle by using three lines and checking if pixel is on the same side of each line

    Vector2 A = (VA.m_ScreenPosition.xy() * Precision.Multiplier).ToVector2<int64_t, eRoundMode::Round>();
    Vector2 B = (VB.m_ScreenPosition.xy() * Precision.Multiplier).ToVector2<int64_t, eRoundMode::Round>();
    Vector2 C = (VC.m_ScreenPosition.xy() * Precision.Multiplier).ToVector2<int64_t, eRoundMode::Round>();

    // clockwise order so we check if point is on the right side of line
    const auto ABC = EdgeFunction(A, B, C);

    // if our edge function (signed area x2) is negative, it's a back facing triangle and we can cull it
    bool isBackFacing = ABC <= 0;
    if (isBackFacing && m_BackfaceCulling)
    {
        stats.m_FrameTriangles++;
        return;
    }

    if (m_DrawBBoxes | m_DrawWireframe)
    {
        if (m_DrawWireframe)
            return DrawTriangle(VA, VB, VC, m_WireFrameColor, 0, m_ScreenSize.y);
        else
            return DrawTriangleBoundingBox(VA, VB, VC, m_WireFrameColor, 0, m_ScreenSize.y);
    }

    //if( g_tri_index != size_t(g_selected_tri) )
    //    return;

    //            max
    //    -------B
    //   |      /|
    //   |     /*|
    //   |    /**|
    //   |   /***|
    //   |  /****|
    //   | /*****|
    //   A-------C
    // min

    // min and max points of rectangle

    Vector2 min = A.CWiseMin(B).CWiseMin(C).CWiseMax(Vector2<int64_t>(0, 0));
    Vector2 max = A.CWiseMax(B).CWiseMax(C).CWiseMax(Vector2<int64_t>(0, 0));

    if (max.x < 0 || min.x >= m_ScreenSize.x || max.y < 0 || min.y >= m_ScreenSize.y)
        // if rectangle is outside of screen, we can return
        return;

    min = min.CWiseMin(Vector2<int64_t>(m_ScreenSize.x * Precision.Multiplier, m_ScreenSize.y * Precision.Multiplier));
    max = max.CWiseMin(Vector2<int64_t>(m_ScreenSize.x * Precision.Multiplier, m_ScreenSize.y * Precision.Multiplier));

    stats.m_FrameTriangles++;

    VertexInterpolator interpolator(VA, VB, VC);

    const auto StartP = (min);

    constexpr auto PixelOffset = Vector2<int64_t>{ Precision.Multiplier , Precision.Multiplier } / 2;

    TriangleData& Data = *m_TransientAllocator.allocate<TriangleData>(VA, VB, VC, pPipelineSharedData->m_pDrawConfig->m_Color, A, B, C, StartP);
    Data.m_InvABC = (1.0f / ABC);
    Data.m_TriIndex = tri_index;
    Data.m_MaxZ = Data.m_MinZ = VA.m_ScreenPosition.z;
    Data.m_MinZ = std::min(Data.m_MinZ, VB.m_ScreenPosition.z);
    Data.m_MaxZ = std::max(Data.m_MaxZ, VB.m_ScreenPosition.z);

    Data.m_MinZ = std::min(Data.m_MinZ, VC.m_ScreenPosition.z);
    Data.m_MaxZ = std::max(Data.m_MaxZ, VC.m_ScreenPosition.z);


    const auto TILE_SIZE_M = m_TileSize * Precision.Multiplier;

    Vector2i MinTile;
    Vector2i MaxTile;

    {
        auto Min = min.ToVector2<int>() / Precision.Multiplier;
        auto Max = max.ToVector2<int>() / Precision.Multiplier;
        MinTile = Min / m_TileSize;
        MaxTile = ((Max + m_TileSize) / m_TileSize).CWiseMin(m_LastTile.ToVector2<int>());
    }

    struct Edge_t
    {
        Vector2<int64_t> A;
        Vector2<int64_t> B;
    };
    Vector2<int64_t> Closest[3];
    Vector2<int64_t> Farthest[3];

    Edge_t Edges[3];
    Edges[0] = { A , B };
    Edges[1] = { B , C };
    Edges[2] = { C , A };

    const Vector2<int64_t> Corners[] =
    {
        { 0 , 0 },
        { 0 , 1 },
        { 1 , 0 },
        { 1 , 1 },
    };

    {
        ZoneScopedN("Classify Corners");
        for (int i = 0; auto& Edge : Edges)
        {
            auto A = Edge.A;
            auto B = Edge.B;
            auto AB = B - A;
            auto Shift = (A + AB / 2) + (-AB.Rotated90()) * 10;

            uint8_t ClosestI = 0;
            uint8_t FarthestI = 0;
            int64_t ClosestD = std::numeric_limits<int>::max();
            int64_t FarthestD = std::numeric_limits<int>::min();

            for (uint8_t i = 0; i < 4; ++i)
            {
                auto P = Shift + Corners[i];

                int dist = SoftwareRenderer::EdgeFunction(B, A, P);
                if (dist < ClosestD)
                {
                    ClosestD = dist;
                    ClosestI = i;
                }
                if (dist > FarthestD)
                {
                    FarthestD = dist;
                    FarthestI = i;
                }
            }
            Closest[i] = Corners[ClosestI] * TILE_SIZE_M;
            Farthest[i++] = Corners[FarthestI] * TILE_SIZE_M;
        }
    }

    if (g_showTriangleBoundry)
    {
        ImGuiAddLine(A, B, ImColor(255, 0, 64), m_ScreenSize, 3);
        ImGuiAddLine(B, C, ImColor(0, 255, 64), m_ScreenSize, 3);
        ImGuiAddLine(C, A, ImColor(64, 0, 255), m_ScreenSize, 3);
    }

    if (g_showCornersClassify)
    {
        Vector2 TilePos = Vector2<int64_t>(4, m_TilesGridSize.y - 4) * TILE_SIZE_M;

        ImGuiAddRect(TilePos, TilePos + m_TileSize, ImColor(255, 0, 0, 128), m_ScreenSize);

        ImGuiAddRect(TilePos + Closest[0], TilePos + Closest[0] + 8, ImColor(255, 0, 0, 255), m_ScreenSize, 0, 0, 3);
        ImGuiAddRect(TilePos + Closest[1], TilePos + Closest[1] + 8, ImColor(0, 255, 0, 255), m_ScreenSize, 0, 0, 3);
        ImGuiAddRect(TilePos + Closest[2], TilePos + Closest[2] + 8, ImColor(0, 0, 255, 255), m_ScreenSize, 0, 0, 3);

        ImGuiAddX(TilePos + Farthest[0], TilePos + Farthest[0] + 8, ImColor(255, 0, 0, 255), m_ScreenSize, 2);
        ImGuiAddX(TilePos + Farthest[1], TilePos + Farthest[1] + 8, ImColor(0, 255, 0, 255), m_ScreenSize, 2);
        ImGuiAddX(TilePos + Farthest[2], TilePos + Farthest[2] + 8, ImColor(0, 0, 255, 255), m_ScreenSize, 2);
    }

    CommandBuffer* pCommandBuffer = pPipelineSharedData ? pPipelineSharedData->m_pRenderTilesCmdBuffer : m_pCommandBuffer;

    transient_allocator alloc{ m_TransientMemoryResource };

    DrawControl DC = pPipelineSharedData->m_pDrawConfig->m_DrawControl;
    CommandRenderTile* pCommand = nullptr;

    auto AddCommand = [&](const Vector2<int64_t>& TilePos, const TileInfo* pTileInfo, auto FullTile)
        {
            constexpr bool IsFullTile = FullTile();

            auto ID = pTileInfo->m_DrawCount++;

            pCommand = alloc.allocate<CommandRenderTile>();
            auto LogicPos = TilePos.ToVector2<int>();
            pCommand->DrawControl = DC;
            pCommand->DrawControl.m_IsFullTile = IsFullTile;
            pCommand->TileDrawID = ID;
            pCommand->Triangle = &Data;
            pCommand->TileInfo = pTileInfo;

            CommandRenderTile* pCurTop = nullptr;

            if (g_ThreadPerTile)
            {
                pCurTop = pTileInfo->m_pRenderTileCmd.load(std::memory_order_relaxed);

                for (;;)
                {
                    pCommand->pNext.store(pCurTop, std::memory_order_relaxed);
                    if (!pTileInfo->m_pRenderTileCmd.compare_exchange_weak(pCurTop, pCommand, std::memory_order_acq_rel))
                        continue;

                    break;
                }
            }

            if (!pCurTop)
            {
                ZoneColor(0x0000FF);
                outCommmands.push_back(pCommand);
            }
            else
            {
                ZoneColor(0x00FF00);
            }

            if (g_showTilestype)
            {
                if constexpr (IsFullTile)
                    ImGuiAddRectFilled(TilePos, TilePos + m_TileSize, ImColor(64, 255, 0, 64), m_ScreenSize);
                else
                    ImGuiAddRectFilled(TilePos, TilePos + m_TileSize, ImColor(0, 64, 255, 64), m_ScreenSize);
            }
        };

    {
        ZoneScopedN("Classify Tiles");
        for (int y = MinTile.y; y <= MaxTile.y; ++y)
        {
            for (int x = MinTile.x; x <= MaxTile.x; ++x)
            {
                const auto TileIndex = Vector2si(x, y);
                auto* pTileInfo = GetTileInfo(TileIndex);
                Vector2 TilePos = (TileIndex * TILE_SIZE_M).ToVector2<int64_t>();

                auto Classified = ClassifyX(A, B, C, TilePos, Closest, Farthest);

                if (g_showTilesBoundry)
                    ImGuiAddRect(TilePos, TilePos + m_TileSize, ImColor(32, 32, 32, 255), m_ScreenSize);

                if (Classified == eTileCoverage::Inside)
                {
                    AddCommand(TilePos, pTileInfo, std::true_type{});
                }
                else if (Classified == eTileCoverage::Partial)
                {
                    AddCommand(TilePos, pTileInfo, std::false_type{});
                }
            }
        }
    }

    if (pCommand)
        stats.m_FrameTrianglesDrawn++;
}

void SoftwareRenderer::BeginFrame()
{
    m_FramePixels = 0;
    m_FrameTriangles = 0;
    m_FramePixelsDrawn = 0;
    m_FrameDrawsPerTile = 0;
    m_FramePixelsCalcualted = 0;
    m_FrameTrianglesDrawn = 0;
    m_FrameDrawTimeMainUS = 0;
    m_FrameDrawTimeThreadUS = 0;
    m_FillrateKP = 0;

    m_FrameRasterTimeUS = 0;
    m_FrameTransformTimeUS = 0;
    g_max_overdraw = 0;

    for (int i = 0; m_TilesGridSize.x * m_TilesGridSize.y > i; ++i)
    {
        m_TilesGrid[i].m_DrawCount = 0;
        m_TilesGrid[i].m_pRenderTileCmd = nullptr;
    }
}

void SoftwareRenderer::EndFrame()
{
    m_TransientMemoryResource.reset();

    const int ThreadsDivide = (m_TileThreadPool.GetThreadCount() ? m_TileThreadPool.GetThreadCount() : 1);

    m_DrawStats.m_FramePixels = m_FramePixels;
    m_DrawStats.m_FramePixelsDrawn = m_FramePixelsDrawn;
    m_DrawStats.m_FramePixelsCalcualted = m_FramePixelsCalcualted;
    m_DrawStats.m_FrameDrawsPerTile = (m_FrameDrawsPerTile * 100) / (m_TilesGridSize.x * m_TilesGridSize.y);
    m_DrawStats.m_FrameTriangles = m_FrameTriangles;
    m_DrawStats.m_FrameTrianglesDrawn = m_FrameTrianglesDrawn;
    m_DrawStats.m_DrawTimeUS = m_FrameDrawTimeThreadUS;
    m_DrawStats.m_DrawTimePerThreadUS = m_FrameDrawTimeThreadUS / ThreadsDivide;
    m_DrawStats.m_RasterTimeUS = m_FrameRasterTimeUS;
    m_DrawStats.m_RasterTimePerThreadUS = m_FrameRasterTimeUS / ThreadsDivide;
    m_DrawStats.m_FillrateKP = m_DrawStats.m_FramePixelsDrawn * (1000.0f / (m_DrawStats.m_RasterTimeUS / ThreadsDivide));
    m_DrawStats.m_TransformTimeUS = m_FrameTransformTimeUS;
    m_DrawStats.m_TransformTimePerThreadUS = m_FrameTransformTimeUS / ThreadsDivide;

    if (g_showTilesGrid)
    {
        for (int i = 0; i < m_ScreenSize.x; i += m_TileSize)
            ImGui::GetBackgroundDrawList()->AddLine(ImVec2(i, 0), ImVec2(i, m_ScreenSize.y), ImColor(255, 128, 255, 128));

        for (int i = 0; i < m_ScreenSize.y; i += m_TileSize)
            ImGui::GetBackgroundDrawList()->AddLine(ImVec2(0, i), ImVec2(m_ScreenSize.x, i), ImColor(255, 128, 255, 128));
    }

    m_TransientMemoryResource.reset();
    m_pCommandBuffer = AllocTransientCommandBuffer();
}

void SoftwareRenderer::Render(const vector<Vertex>& vertices)
{
    ZoneScoped;
    const auto RenderStartTime = std::chrono::high_resolution_clock::now();

    {
        FrameMarkNamed("Pre-Render cleanup");
        for (int i = 0; m_TilesGridSize.x * m_TilesGridSize.y > i; ++i)
        {
            m_TilesGrid[i].m_DrawCount = 0;
            m_TilesGrid[i].m_pRenderTileCmd = nullptr;
        }
    }

    auto pConfig = m_TransientAllocator.allocate<DrawConfig>();
    pConfig->m_Color = m_VertexColor;
    pConfig->m_DrawControl.m_AlphaBlend = m_AlphaBlend;
    pConfig->m_DrawControl.m_ZTest = m_ZTest;
    pConfig->m_DrawControl.m_ZWrite = m_ZWrite;

    m_pCommandBuffer->AddSyncBarrier("Pre-VertexAssemply Sync", m_ThreadsCount);
    m_pCommandBuffer->PushCommand<CommandVertexAssemply>(vertices, *pConfig);
    m_pCommandBuffer->AddSyncBarrier("Post VertexAssemply Sync", m_ThreadsCount);

    FrameMarkNamed("Tasks launched");
    m_TileThreadPool.LaunchTasks({ m_ThreadsCount , [this]() { RendererTaskWorker(); } });

    m_FrameDrawTimeMainUS += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - RenderStartTime).count();

    m_pCommandBuffer = AllocTransientCommandBuffer();
}

void SoftwareRenderer::RendererTaskWorker()
{
    ZoneScoped;
    const auto pCommandBuffer = m_pCommandBuffer;

    DrawStats localDrawStats;
    RenderThreadData data
    {
        .m_ThreadfastMemResource{ 128 * 1024 , std::min<size_t>(AVX_ALIGN , 8) , m_TransientMemoryResource } ,
        .m_pDrawStats = &localDrawStats,
    };

    auto Now = std::chrono::high_resolution_clock::now();

    for (;;)
    {
        auto pCommand = pCommandBuffer->GetNextCommand();
        if (!pCommand)
        {
            ExecuteExitCommand();
            break;
        }

        switch (pCommand->GetCommandID())
        {
        case eCommandID::ClearBuffers:           ClearBuffers(*pCommand->static_cast_to<CommandClear>());             continue;
        case eCommandID::Fill32BitBuffer:        Fill32BitBuffer(*pCommand->static_cast_to<CommandFill32BitBuffer>());             continue;
        case eCommandID::SyncBarier:             WaitForSync(*pCommand->static_cast_to<CommandSyncBarier>());             continue;
        case eCommandID::VertexAssemply:         VertexAssemply(*pCommand->static_cast_to<CommandVertexAssemply>());             continue;
        case eCommandID::VertexTransformAndClip: VertexTransformAndClip(*pCommand->static_cast_to<CommandVertexTransformAndClip>(), data);        continue;
        case eCommandID::ProcessTriangles:       ProcessTriangles(*pCommand->static_cast_to<CommandProcessTriangles>(), data);        continue;
        case eCommandID::RenderTile:             DrawTile(*pCommand->static_cast_to<CommandRenderTile>(), data);        continue;
            ;
        default:
            assert(false && "Unknown command");
            return;
        }
    }

    m_FramePixels += localDrawStats.m_FramePixels;
    m_FramePixelsDrawn += localDrawStats.m_FramePixelsDrawn;
    m_FramePixelsCalcualted += localDrawStats.m_FramePixelsCalcualted;
    m_FrameDrawsPerTile += localDrawStats.m_FrameDrawsPerTile;
    m_FrameTriangles += localDrawStats.m_FrameTriangles;
    m_FrameTrianglesDrawn += localDrawStats.m_FrameTrianglesDrawn;
    m_FrameDrawTimeThreadUS += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - Now).count();
    m_FillrateKP += localDrawStats.m_FillrateKP;
    m_FrameRasterTimeUS += localDrawStats.m_RasterTimeUS;
    m_FrameTransformTimeUS += localDrawStats.m_TransformTimeUS;
}

void SoftwareRenderer::ClearBuffers(const CommandClear& cmd)
{
    ZoneScoped;
    if (!cmd.m_ClearColor && !cmd.m_ZValue)
        return;

    auto BUFFER_SUB_SIZE = m_ScreenBuffer.size() / 8;//TILE_SIZE*TILE_SIZE*32;

    auto pWorkCmdBuffer = AllocTransientCommandBuffer();

    auto pSync = m_TransientAllocator.allocate<SyncBarrier>();
    pSync->m_Barrier.emplace(m_ThreadsCount);

    if (cmd.m_ClearColor)
    {
        span<uint32_t> BufferSpan(m_ScreenBuffer);

        for (int i = BUFFER_SUB_SIZE; i < BufferSpan.size(); i += BUFFER_SUB_SIZE)
        {
            if (BufferSpan.size() < BUFFER_SUB_SIZE)
                break;

            auto SubSpan = BufferSpan.subspan(0, BUFFER_SUB_SIZE);
            BufferSpan = BufferSpan.subspan(SubSpan.size());

            pWorkCmdBuffer->PushCommand<CommandFill32BitBuffer>(SubSpan, *cmd.m_ClearColor);
        }

        if (!BufferSpan.empty())
            pWorkCmdBuffer->PushCommand<CommandFill32BitBuffer>(BufferSpan, *cmd.m_ClearColor);
    }

    if (cmd.m_ZValue)
    {
        span<float> BufferSpan(m_ZBuffer);

        for (int i = BUFFER_SUB_SIZE; i < BufferSpan.size(); i += BUFFER_SUB_SIZE)
        {
            if (BufferSpan.size() < BUFFER_SUB_SIZE)
                break;

            auto SubSpan = BufferSpan.subspan(0, BUFFER_SUB_SIZE);
            BufferSpan = BufferSpan.subspan(SubSpan.size());

            pWorkCmdBuffer->PushCommand<CommandFill32BitBuffer>(SubSpan, 1.0f);//*cmd.m_ZValue);
        }

        if (!BufferSpan.empty())
            pWorkCmdBuffer->PushCommand<CommandFill32BitBuffer>(BufferSpan, 1.0f);// *cmd.m_ZValue);
    }

    //pWorkCmdBuffer->AddSyncPoint( *pSync , m_ThreadsCount );

    m_pCommandBuffer->PushCommandBuffer(*pWorkCmdBuffer);
}

void SoftwareRenderer::Fill32BitBuffer(const CommandFill32BitBuffer& cmd)
{
    ZoneScoped;
    auto pBuffer = cmd.m_pBuffer;
    if (!pBuffer)
        return;

    if (cmd.m_Value.m_U8Array[0] == cmd.m_Value.m_U8Array[1] && cmd.m_Value.m_U8Array[1] == cmd.m_Value.m_U8Array[2] && cmd.m_Value.m_U8Array[2] && cmd.m_Value.m_U8Array[3])
        memset(cmd.m_pBuffer, cmd.m_Value.m_U8Array[0], cmd.m_ElementsCount * sizeof(uint32_t));
    else
        std::fill_n(reinterpret_cast<uint32_t*>(cmd.m_pBuffer), cmd.m_ElementsCount, cmd.m_Value.m_UValue);
}

void SoftwareRenderer::VertexAssemply(const CommandVertexAssemply& cmd)
{
    ZoneScoped;
    auto vertices = cmd.m_Vertices;
    if (vertices.empty())
        return;

    auto VerticesCount = vertices.size();

    PipelineSharedData* pData = nullptr;
    CommandBuffer* pWorkCmdBuffer = nullptr;

    {
        ZoneScopedN("Prepare PipelineSharedData");
        pWorkCmdBuffer = AllocTransientCommandBuffer();

        pData = m_TransientAllocator.allocate<PipelineSharedData>();
        pData->m_pProcessTrianglesCmdBuffer = AllocTransientCommandBuffer();
        pData->m_pProcessTrianglesCmdBufferSync = m_TransientAllocator.allocate<SyncBarrier>();
        pData->m_pRenderTilesCmdBuffer = AllocTransientCommandBuffer();
        pData->m_pRenderTilesCmdBufferSync = m_TransientAllocator.allocate<SyncBarrier>();
        pData->m_pDrawConfig = cmd.m_pConfig;
    }

    m_MVPMatrix.GetFrustumNearPlane(pData->m_NearFrustumPlane);

    auto pVertexTransformAndClipSync = m_TransientAllocator.allocate<SyncBarrier>();
    {
        ZoneScopedN("VertexAssemply dispatch");
        constexpr auto TRIANGLES_PER_COMMAND = 100;

        size_t i = 0;
        if (g_MultithreadedTransformAndClip)
        {
            for (; i < VerticesCount; i += TRIANGLES_PER_COMMAND * 3)
            {
                if (vertices.size() < +TRIANGLES_PER_COMMAND * 3)
                    break;

                auto SubSpan = vertices.subspan(0, TRIANGLES_PER_COMMAND * 3);
                vertices = vertices.subspan(SubSpan.size());

                pWorkCmdBuffer->PushCommand<CommandVertexTransformAndClip>(SubSpan, *pData, i / 3);
            }
        }

        if (!vertices.empty())
            pWorkCmdBuffer->PushCommand<CommandVertexTransformAndClip>(vertices, *pData, i / 3);
    }

    pVertexTransformAndClipSync->m_Name = "VertexAssemply end";
    pWorkCmdBuffer->AddSyncPoint(*pVertexTransformAndClipSync, m_ThreadsCount);

    pVertexTransformAndClipSync->m_Barrier.emplace(m_ThreadsCount, triviall_function_ref{}.Assign(m_TransientMemoryResource, [=, this]
        {
            m_pCommandBuffer->PushCommandBuffer(*pData->m_pProcessTrianglesCmdBuffer);
            pData->m_pProcessTrianglesCmdBufferSync->m_Name = "transform triangles end";
            m_pCommandBuffer->AddSyncPoint(*pData->m_pProcessTrianglesCmdBufferSync, m_ThreadsCount);
        }));

    pData->m_pProcessTrianglesCmdBufferSync->m_Barrier.emplace(m_ThreadsCount, triviall_function_ref{}.Assign(m_TransientMemoryResource, [=, this]()
        {
            m_pCommandBuffer->PushCommandBuffer(*pData->m_pRenderTilesCmdBuffer);

            pData->m_pRenderTilesCmdBufferSync->m_Name = "render triles end";
            m_pCommandBuffer->AddSyncPoint(*pData->m_pRenderTilesCmdBufferSync, m_ThreadsCount);
            m_pCommandBuffer->Finish();
        }));

    pData->m_pRenderTilesCmdBufferSync->m_Barrier.emplace(m_ThreadsCount);

    m_pCommandBuffer->PushCommandBuffer(*pWorkCmdBuffer);
}

void SoftwareRenderer::ExecuteExitCommand()
{
    ZoneScoped;
}

void SoftwareRenderer::VertexTransformAndClip(const CommandVertexTransformAndClip& cmd, RenderThreadData& data)
{
    ZoneScoped;
    ZoneColor(0xE000E0);
    auto vertices = cmd.m_Input;
    if (vertices.empty())
        return;

    span<TransformedVertex> transformedVertices;

    {
        ZoneScopedN("Clip");
        ZoneColor(0xE000E0);
        vertices = ClipTriangles(cmd.m_pPipelineSharedData->m_NearFrustumPlane, 0.001f, vertices);
    }

    {
        ZoneScopedN("Transform");
        ZoneColor(0xE000E0);
        transformedVertices = m_TransientAllocator.allocate_array<TransformedVertex>(vertices.size());

        const auto startTime = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < vertices.size(); ++i)
            transformedVertices[i].ProjToScreen(vertices[i], m_ModelMatrix, m_MVPMatrix, m_ScreenSize);

        data.m_pDrawStats->m_TransformTimeUS += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTime).count();
    }

    if (!g_MultithreadedTransformAndClip)
    {
        ZoneScopedN("Process Tri dispatch");
        constexpr auto TRIANGLES_PER_COMMAND = 100;

        const auto VerticesCount = transformedVertices.size();

        if (g_MultithreadedTransformAndClip)
        {
            for (size_t i = 0; i < VerticesCount; i += TRIANGLES_PER_COMMAND * 3)
            {
                if (transformedVertices.size() < +TRIANGLES_PER_COMMAND * 3)
                    break;

                auto SubSpan = transformedVertices.subspan(0, TRIANGLES_PER_COMMAND * 3);
                transformedVertices = transformedVertices.subspan(SubSpan.size());

                cmd.m_pPipelineSharedData->m_pProcessTrianglesCmdBuffer->PushCommand<CommandProcessTriangles>(SubSpan, *cmd.m_pPipelineSharedData, cmd.m_StartTriIndex + i / 3);
            }
        }
    }

    //cmd.m_pPipelineSharedData->m_pProcessTrianglesCmdBuffer->PushCommandSync<CommandProcessTriangles>( cmd.m_pPipelineSharedData->m_pProcessTrianglesCmdBufferSync , transformedVertices , *cmd.m_pPipelineSharedData );
    cmd.m_pPipelineSharedData->m_pProcessTrianglesCmdBuffer->PushCommand<CommandProcessTriangles>(transformedVertices, *cmd.m_pPipelineSharedData, cmd.m_StartTriIndex);
}

void SoftwareRenderer::ProcessTriangles(const CommandProcessTriangles& cmd, RenderThreadData& data)
{
    ZoneScoped;
    ZoneColor(0x00E0E0);
    auto vertices = cmd.m_Vertices;
    if (vertices.empty())
        return;

    std::array<uint8_t, (1024 + 16) * sizeof(void*)> Buffer;
    transient_memory_resource transient;
    pmr::monotonic_buffer_resource BufferedRes{ Buffer.data() , Buffer.size() , &transient };
    pmr::vector<const Command*> Commands{ &BufferedRes };
    Commands.reserve(1024);

    int tri = cmd.m_StartTriIndex;
    for (int i = 2; i < vertices.size(); i += 3, ++tri)
        GenerateTileJobs(vertices[i - 2], vertices[i - 1], vertices[i], *data.m_pDrawStats, cmd.m_pPipelineSharedData, tri, Commands);

    auto pCmdBuf = CommandBuffer::CreateCommandBuffer(m_TransientMemoryResource, Commands, true);

    cmd.m_pPipelineSharedData->m_pRenderTilesCmdBuffer->PushCommandBuffer(*pCmdBuf);
}

void SoftwareRenderer::RenderDepthBuffer()
{
    ZoneScoped;


    for (int tile_index = 0; tile_index < m_TilesGridSize.x * m_TilesGridSize.y; ++tile_index)
    {
        auto& Tile = m_TilesGrid[tile_index];
        auto pZBuffer = m_ZBuffer.data() + Tile.m_TileZOffset;
        auto TileScreenPos = Tile.m_TileIndex * m_TileSize;

        uint32_t* pScreenBuffer = m_ScreenBuffer.data() + TileScreenPos.y * m_ScreenSize.x + TileScreenPos.x;

        for (int y = 0, z = 0; y < m_TileSize; ++y)
        {
            uint32_t* pScreenLine = pScreenBuffer + y * m_ScreenSize.x;
            for (int x = 0; x < m_TileSize; ++x, ++z)
            {
                uint32_t Col = std::clamp(int(255 * pZBuffer[z]), 0, 255);
                pScreenLine[x] = 0xFF000000 | (Col << 16) | (Col << 8) | (Col);
            }
        }
    }
}

void SoftwareRenderer::WaitForSync(const CommandSyncBarier& cmd)
{
    ZoneScoped;
    ZoneColor(0xFF0000);
    if (cmd.pAwaitSync)
        cmd.pAwaitSync->Wait();
}

void SoftwareRenderer::DoRender(const vector<Vertex>& inVertices, int minY, int maxY, int threadID)
{
    ZoneScoped;
    Plane nearFrustumPlane;
    m_MVPMatrix.GetFrustumNearPlane(nearFrustumPlane);

    const auto startTime = std::chrono::high_resolution_clock::now();

    span<const Vertex> vertices = ClipTriangles(nearFrustumPlane, 0.001f, inVertices);
    DrawStats drawStats;

    thread_local static vector<TransformedVertex> transformedVertices;
    transformedVertices.resize(vertices.size());

    {
        ZoneScopedN("Transform");
        const auto startTime = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < vertices.size(); ++i)
            transformedVertices[i].ProjToScreen(vertices[i], m_ModelMatrix, m_MVPMatrix, m_ScreenSize);

        m_FrameTransformTimeUS += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTime).count();
    }

    if (m_DrawWireframe || m_DrawBBoxes)
    {
        ZoneScopedN("Debug");
        const Vector4f color = m_ColorizeThreads ? ThreadColors[threadID] : m_WireFrameColor;

        for (int i = 0; i < vertices.size(); i += TRIANGLE_VERT_COUNT)
        {
            if (m_DrawWireframe)
                DrawTriangle(transformedVertices[i + 0], transformedVertices[i + 1], transformedVertices[i + 2], color, minY, maxY);

            if (m_DrawBBoxes)
                DrawTriangleBoundingBox(transformedVertices[i + 0], transformedVertices[i + 1], transformedVertices[i + 2], color, minY, maxY);
        }
    }
    else
    {
        ZoneScopedN("Draw");
        const auto startTime = std::chrono::high_resolution_clock::now();

        const Vector4f color = m_ColorizeThreads ? ThreadColors[threadID] : Vector4f(1.0f, 1.0f, 1.0f, 1.0f);


        DrawFunctionConfig c;
        c.m_ZTest = m_ZTest;
        c.m_ZWrite = m_ZWrite;
        const auto Index = c.ToIndex();


        DrawFilledTriangles < MathSSE, DrawFunctionConfig{} > (transformedVertices.data(), transformedVertices.size(), color, minY, maxY, drawStats);

        m_FrameRasterTimeUS += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTime).count();
    }

    const auto timeUS = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - startTime).count();

    m_FrameTriangles += drawStats.m_FrameTriangles;
    m_FrameTrianglesDrawn += drawStats.m_FrameTrianglesDrawn;
    m_FramePixels += drawStats.m_FramePixels;
    m_FramePixelsDrawn += drawStats.m_FramePixelsDrawn;
    m_FillrateKP += drawStats.m_FramePixelsDrawn * (1000.0f / timeUS);
    m_FrameDrawTimeThreadUS += timeUS;
}

void SoftwareRenderer::UpdateMVPMatrix()
{
    m_MVPMatrix = m_ModelMatrix * m_ViewMatrix * m_ProjectionMatrix;
}

const vector<uint32_t>& SoftwareRenderer::GetScreenBuffer()const
{
    return m_ScreenBuffer;
}

const DrawStats& SoftwareRenderer::GetDrawStats() const
{
    return m_DrawStats;
}

shared_ptr<ITexture> SoftwareRenderer::GetDefaultTexture() const
{
    return m_DefaultTexture;
}

inline void SoftwareRenderer::PutPixelUnsafe(int x, int y, uint32_t color)
{
    m_ScreenBuffer[y * m_ScreenSize.x + x] = color;
}

inline void SoftwareRenderer::PutPixel(int x, int y, uint32_t color)
{
    if (x >= m_ScreenSize.x || x <= 0 || y >= m_ScreenSize.y || y <= 0)
    {
        return;
    }
    m_ScreenBuffer[y * m_ScreenSize.x + x] = color;
}

template< typename MathT, DrawFunctionConfig Config >
void SoftwareRenderer::DrawFilledTriangles(const TransformedVertex* pVerts, size_t Count, const Vector4f& color, int minY, int maxY, DrawStats& stats)
{
    switch (m_DrawTriVersion)
    {
    case eDrawTriVersion::DrawTriBaseline:
        for (size_t i = 0; i < Count; i += TRIANGLE_VERT_COUNT)
        {
            DrawFilledTriangleBaseline<MathT, Config>(pVerts[i + 0], pVerts[i + 1], pVerts[i + 2], color, minY, maxY, stats);
            //DrawFilledTriangle<MathT>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
        }
        return;

    case eDrawTriVersion::DrawTriv2:
        for (size_t i = 0; i < Count; i += TRIANGLE_VERT_COUNT)
        {
            DrawFilledTriangle_v2<MathT, Config>(pVerts[i + 0], pVerts[i + 1], pVerts[i + 2], color, minY, maxY, stats);
            //DrawFilledTriangle<MathT>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
        }
        return;

    case eDrawTriVersion::DrawTriv3:
        for (size_t i = 0; i < Count; i += TRIANGLE_VERT_COUNT)
        {
            DrawFilledTriangle_v3<MathT, Config>(pVerts[i + 0], pVerts[i + 1], pVerts[i + 2], color, minY, maxY, stats);
            //DrawFilledTriangle<MathT>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
        }
        return;


    default:
        for (size_t i = 0; i < Count; i += TRIANGLE_VERT_COUNT)
        {
            DrawFilledTriangle<MathT, Config>(pVerts[i + 0], pVerts[i + 1], pVerts[i + 2], color, minY, maxY, stats);
            //DrawFilledTriangle<MathT>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
        }
        return;
    }

}

template< typename, DrawFunctionConfig >
void SoftwareRenderer::DrawFilledTriangleBaseline(const TransformedVertex& VA, const TransformedVertex& VB, const TransformedVertex& VC, const Vector4f& color, int minY, int maxY, DrawStats& stats)
{
    // this is baseline implementation of triangle filling algorithm as it was written in my engineering thesis.
    // It uses 'Edge Function' algorithm to determine if pixel is inside triangle. and only major optimization is
    // that we are using fast method to interpolate vertex attributes in 'Vertex Interpolator'. It exploits fact
    // that we can use screen space interpolation of vertex attributes as long as we are interpolating A/w
    // where A is vertex attribute and w is homogeneous coordinate of vertex position in screen space.

    ZoneScoped;
    // filling algorithm is working that way that we are going through all pixels in rectangle that is created by min and max points
    // and we are checking if pixel is inside triangle by using three lines and checking if pixel is on the same side of each line

    Vector2f A = VA.m_ScreenPosition.xy();
    Vector2f B = VB.m_ScreenPosition.xy();
    Vector2f C = VC.m_ScreenPosition.xy();

    // clockwise order so we check if point is on the right side of line
    const float ABC = EdgeFunction(A, B, C);

    // if our edge function (signed area x2) is negative, it's a back facing triangle and we can cull it
    bool isBackFacing = ABC <= 0;
    if (isBackFacing)
    {
        stats.m_FrameTriangles++;
        return;
    }

    //            max
    //    -------B
    //   |      /|
    //   |     /*|
    //   |    /**|
    //   |   /***|
    //   |  /****|
    //   | /*****|
    //   A-------C
    // min

    // min and max points of rectangle clamped to screen size so we don't calculate points that we don't see
    Vector2i min = A.CWiseMin(B).CWiseMin(C).CWiseMin(Vector2f(m_ScreenSize.x - 1, maxY)).CWiseMax(Vector2f(0, minY)).ToVector2i();
    Vector2i max = A.CWiseMax(B).CWiseMax(C).CWiseMin(Vector2f(m_ScreenSize.x - 1, maxY)).CWiseMax(Vector2f(0, minY)).ToVector2i();

    const float invABC = 1.0f / ABC;

    VertexInterpolator interpolator(VA, VB, VC);
    TransformedVertex interpolatedVertex;

    int pixelsDrawn = 0;

    // loop through all pixels in rectangle
    for (int y = min.y; y <= max.y; ++y)
    {
        for (int x = min.x; x <= max.x; ++x)
        {
            const Vector2f P(x + 0.5f, y + 0.5f);
            // calculate value of edge function for each line
            const float ABP = EdgeFunction(A, B, P);
            if (ABP < 0)
                continue;
            const float BCP = EdgeFunction(B, C, P);
            if (BCP < 0)
                continue;
            const float CAP = EdgeFunction(C, A, P);
            if (CAP < 0)
                continue;
            // if pixel is inside triangle, draw it
            //
            // dividing edge function values by ABC will give us barycentric coordinates - how much each vertex contributes to final color in point P
            Vector3f baricentricCoordinates = Vector3f(BCP, CAP, ABP) * invABC;
            interpolator.InterpolateZ(baricentricCoordinates, interpolatedVertex);

            float& z = m_ZBuffer[y * m_ScreenSize.x + x];
            if (interpolatedVertex.m_ScreenPosition.z < z)
            {
                if (m_ZWrite)
                    z = interpolatedVertex.m_ScreenPosition.z;
            }
            else if (m_ZTest)
            {
                continue;
            }


            interpolator.InterpolateAllButZ(baricentricCoordinates, interpolatedVertex);
            interpolatedVertex.m_Color = interpolatedVertex.m_Color * color;
            PutPixelUnsafe(x, y, FragmentShader(interpolatedVertex).ToARGB());
            pixelsDrawn++;
        }
    }

    stats.FinishDrawCallStats(min, max, pixelsDrawn);
}

template< typename MathT, DrawFunctionConfig >
void SoftwareRenderer::DrawFilledTriangle_v2(const TransformedVertex& VA, const TransformedVertex& VB, const TransformedVertex& VC, const Vector4f& color, int minY, int maxY, DrawStats& stats)
{
    // this version uses SIMD version of 'VertexInterpolator::Interpolate' that can
    // interpolate all attributes in 4 SIMD additions and 4 multiplications (using AVX, SSE uses 2x more) and one division.
    // Non-SIMD version uses 63 multiplications + 26 additions and one division.

    ZoneScoped;
    // filling algorithm is working that way that we are going through all pixels in rectangle that is created by min and max points
    // and we are checking if pixel is inside triangle by using three lines and checking if pixel is on the same side of each line

    Vector2f A = VA.m_ScreenPosition.xy();
    Vector2f B = VB.m_ScreenPosition.xy();
    Vector2f C = VC.m_ScreenPosition.xy();

    // clockwise order so we check if point is on the right side of line
    const float ABC = EdgeFunction(A, B, C);

    // if our edge function (signed area x2) is negative, it's a back facing triangle and we can cull it
    bool isBackFacing = ABC <= 0;
    if (isBackFacing)
    {
        stats.m_FrameTriangles++;
        return;
    }

    //            max
    //    -------B
    //   |      /|
    //   |     /*|
    //   |    /**|
    //   |   /***|
    //   |  /****|
    //   | /*****|
    //   A-------C
    // min

    // min and max points of rectangle clamped to screen size so we don't calculate points that we don't see
    Vector2i min = A.CWiseMin(B).CWiseMin(C).CWiseMin(Vector2f(m_ScreenSize.x - 1, maxY)).CWiseMax(Vector2f(0, minY)).ToVector2i();
    Vector2i max = A.CWiseMax(B).CWiseMax(C).CWiseMin(Vector2f(m_ScreenSize.x - 1, maxY)).CWiseMax(Vector2f(0, minY)).ToVector2i();

    const float invABC = 1.0f / ABC;

    VertexInterpolator interpolator(VA, VB, VC);
    TransformedVertex interpolatedVertex;

    int pixelsDrawn = 0;
    MathT mathInstance;

    // loop through all pixels in rectangle
    for (int y = min.y; y <= max.y; ++y)
    {
        for (int x = min.x; x <= max.x; ++x)
        {
            const Vector2f P(x + 0.5f, y + 0.5f);
            // calculate value of edge function for each line
            const float ABP = EdgeFunction(A, B, P);
            if (ABP < 0)
                continue;
            const float BCP = EdgeFunction(B, C, P);
            if (BCP < 0)
                continue;
            const float CAP = EdgeFunction(C, A, P);
            if (CAP < 0)
                continue;
            // if pixel is inside triangle, draw it
            //
            // dividing edge function values by ABC will give us barycentric coordinates - how much each vertex contributes to final color in point P
            Vector3f baricentricCoordinates = Vector3f(BCP, CAP, ABP) * invABC;
            interpolator.Interpolate(mathInstance, baricentricCoordinates, interpolatedVertex);

            float& z = m_ZBuffer[y * m_ScreenSize.x + x];
            if (interpolatedVertex.m_ScreenPosition.z < z)
            {
                if (m_ZWrite)
                    z = interpolatedVertex.m_ScreenPosition.z;
            }
            else if (m_ZTest)
            {
                continue;
            }

            interpolatedVertex.m_Color = interpolatedVertex.m_Color * color;
            PutPixelUnsafe(x, y, FragmentShader(interpolatedVertex).ToARGB());
            pixelsDrawn++;
        }
    }

    stats.FinishDrawCallStats(min, max, pixelsDrawn);
}

/**
 * Struktura do przechowywania wstêpnie obliczonych wartoœci funkcji krawêdziowych
 */
struct ALIGN_FOR_AVX EdgeFunctionHelper1
{
    inline EdgeFunctionHelper1(Vector2f A, Vector2f B, Vector2f C)
        : PrecalculatedA{
          -(B.y - A.y) ,
            (B.x - A.x) ,
          -(C.y - B.y) ,
            (C.x - B.x) ,
          -(A.y - C.y) ,
            (A.x - C.x) ,
            0,
            0,
        }
        ,
        PrecalculatedB{
            -PrecalculatedA[0] * A.x ,
            -PrecalculatedA[1] * A.y ,
            -PrecalculatedA[2] * B.x ,
            -PrecalculatedA[3] * B.y ,
            -PrecalculatedA[4] * C.x ,
            -PrecalculatedA[5] * C.y ,
              0,
        }
    {
    }

    float PrecalculatedA[8];
    float PrecalculatedB[8];
};

/**
 * Struktura do przechowywania wyników funkcji krawêdziowych
 */
struct ALIGN_FOR_AVX EdgeFunctionResult
{
    float ABP = 0; ///< Wynik funkcji krawêdziowej dla krawêdzi AB i punktu P
    float BCP = 0; ///< Wynik funkcji krawêdziowej dla krawêdzi BC i punktu P
    float CAP = 0; ///< Wynik funkcji krawêdziowej dla krawêdzi CA i punktu P
    int   SKIP = 0; ///< Flaga pominiêcia (1 jeœli punkt jest poza trójk¹tem, 0 jeœli wewn¹trz)
};

template< typename MathT, DrawFunctionConfig >
void SoftwareRenderer::DrawFilledTriangle_v3(const TransformedVertex& VA, const TransformedVertex& VB, const TransformedVertex& VC, const Vector4f& color, int minY, int maxY, DrawStats& stats)
{
    // this version uses SIMD version of 'VertexInterpolator::Interpolate' that can
    // interpolate all attributes in 4 SIMD additions and 4 multiplications (using AVX, SSE uses 2x more) and one division.
    // Non-SIMD version uses 63 multiplications + 26 additions and one division.

    ZoneScoped;
    // filling algorithm is working that way that we are going through all pixels in rectangle that is created by min and max points
    // and we are checking if pixel is inside triangle by using three lines and checking if pixel is on the same side of each line

    Vector2f A = VA.m_ScreenPosition.xy();
    Vector2f B = VB.m_ScreenPosition.xy();
    Vector2f C = VC.m_ScreenPosition.xy();

    // clockwise order so we check if point is on the right side of line
    const float ABC = EdgeFunction(A, B, C);

    // if our edge function (signed area x2) is negative, it's a back facing triangle and we can cull it
    bool isBackFacing = ABC <= 0;
    if (isBackFacing)
    {
        stats.m_FrameTriangles++;
        return;
    }

    //            max
    //    -------B
    //   |      /|
    //   |     /*|
    //   |    /**|
    //   |   /***|
    //   |  /****|
    //   | /*****|
    //   A-------C
    // min

    // min and max points of rectangle clamped to screen size so we don't calculate points that we don't see
    Vector2i min = A.CWiseMin(B).CWiseMin(C).CWiseMin(Vector2f(m_ScreenSize.x - 1, maxY)).CWiseMax(Vector2f(0, minY)).ToVector2i();
    Vector2i max = A.CWiseMax(B).CWiseMax(C).CWiseMin(Vector2f(m_ScreenSize.x - 1, maxY)).CWiseMax(Vector2f(0, minY)).ToVector2i();

    const float invABC = 1.0f / ABC;

    VertexInterpolator  interpolator(VA, VB, VC);
    TransformedVertex   interpolatedVertex;
    EdgeFunctionHelper1 edgeHelper(A, B, C);
    EdgeFunctionResult  edgeResult;

    int pixelsDrawn = 0;
    MathT mathInstance;

    // loop through all pixels in rectangle
    for (int y = min.y; y <= max.y; ++y)
    {
        for (int x = min.x; x <= max.x; ++x)
        {
            const Vector2f P(x + 0.5f, y + 0.5f);

            mathInstance.EdgeFunction3x(P, edgeHelper.PrecalculatedA, edgeHelper.PrecalculatedB, &(edgeResult.ABP));

            if (edgeResult.ABP < 0 || edgeResult.BCP < 0 || edgeResult.CAP < 0)
                continue;

            Vector3f baricentricCoordinates = Vector3f(edgeResult.BCP, edgeResult.CAP, edgeResult.ABP) * invABC;
            interpolator.Interpolate(mathInstance, baricentricCoordinates, interpolatedVertex);

            float& z = m_ZBuffer[y * m_ScreenSize.x + x];
            if (interpolatedVertex.m_ScreenPosition.z < z)
            {
                if (m_ZWrite)
                    z = interpolatedVertex.m_ScreenPosition.z;
            }
            else if (m_ZTest)
            {
                continue;
            }

            interpolatedVertex.m_Color = interpolatedVertex.m_Color * color;
            PutPixelUnsafe(x, y, FragmentShader(interpolatedVertex).ToARGB());
            pixelsDrawn++;
        }
    }

    stats.FinishDrawCallStats(min, max, pixelsDrawn);
}

template< typename MathT, DrawFunctionConfig Config >
void SoftwareRenderer::DrawFilledTriangle(const TransformedVertex& VA, const TransformedVertex& VB, const TransformedVertex& VC, const Vector4f& color, int minY, int maxY, DrawStats& stats)
{
    ZoneScoped;

    // filling algorithm is working that way that we are going through all pixels in rectangle that is created by min and max points
    // and we are checking if pixel is inside triangle by using three lines and checking if pixel is on the same side of each line

    Vector2 A = VA.m_ScreenPosition.xy().ToVector2<int64_t>();
    Vector2 B = VB.m_ScreenPosition.xy().ToVector2<int64_t>();
    Vector2 C = VC.m_ScreenPosition.xy().ToVector2<int64_t>();

    // clockwise order so we check if point is on the right side of line
    const float ABC = EdgeFunction(A, B, C);

    // if our edge function (signed area x2) is negative, it's a back facing triangle and we can cull it
    bool isBackFacing = ABC <= 0;
    if (isBackFacing)
    {
        stats.m_FrameTriangles++;
        return;
    }

    //            max
    //    -------B
    //   |      /|
    //   |     /*|
    //   |    /**|
    //   |   /***|
    //   |  /****|
    //   | /*****|
    //   A-------C
    // min

    // min and max points of rectangle
    Vector2 min = A.CWiseMin(B).CWiseMin(C);
    Vector2 max = A.CWiseMax(B).CWiseMax(C);

    // clamp min and max points to screen size so we don't calculate points that we don't see
    min = min.CWiseMin(Vector2<int64_t>(m_ScreenSize.x - 1, maxY)).CWiseMax(Vector2<int64_t>(0, minY));
    max = max.CWiseMin(Vector2<int64_t>(m_ScreenSize.x - 1, maxY)).CWiseMax(Vector2<int64_t>(0, minY));

    const float invABC = 1.0f / ABC;

    VertexInterpolator interpolator(VA, VB, VC);
    TransformedVertex interpolatedVertex;

    int pixelsDrawn = 0;

    Vector4f baricentricCoordinates;

    const auto StartP = (min);


    auto CAP_Stride = Vector2(C.y - A.y, A.x - C.x);
    auto BCP_Stride = Vector2(B.y - C.y, C.x - B.x);
    auto ABP_Stride = Vector2(A.y - B.y, B.x - A.x);

    const auto CAP_Start = CAP_Stride * (StartP - Vector2(C.x, C.y));
    const auto BCP_Start = BCP_Stride * (StartP - Vector2(B.x, B.y));
    const auto ABP_Start = ABP_Stride * (StartP - Vector2(A.x, A.y));


    auto ABP_Y = ABP_Start.y;
    auto BCP_Y = BCP_Start.y;
    auto CAP_Y = CAP_Start.y;

    // loop through all pixels in rectangle
    for (auto ScrrenY = min.y; ScrrenY <= max.y; ScrrenY++, ABP_Y += ABP_Stride.y, BCP_Y += BCP_Stride.y, CAP_Y += CAP_Stride.y)
    {
        int linearpos = ScrrenY * m_ScreenSize.x + min.x;
        uint32_t LineIndex = 0;

        auto ABP_X = ABP_Start.x;
        auto BCP_X = BCP_Start.x;
        auto CAP_X = CAP_Start.x;

        for (auto ScrrenX = min.x; ScrrenX <= max.x; ScrrenX++, ++linearpos, ABP_X += ABP_Stride.x, BCP_X += BCP_Stride.x, CAP_X += CAP_Stride.x)
        {
            auto ABP = (ABP_X + ABP_Y);
            auto BCP = (BCP_X + BCP_Y);
            auto CAP = (CAP_X + CAP_Y);

            if ((ABP | BCP | CAP) < 0)
                continue;

            // if pixel is inside triangle, draw it

            // dividing edge function values by ABC will give us barycentric coordinates - how much each vertex contributes to final color in point P
            Vector3f baricentricCoordinates = Vector3f(ABP, BCP, CAP) * invABC;

            interpolator.InterpolateT<MathT>(Vector3f(baricentricCoordinates.y, baricentricCoordinates.z, baricentricCoordinates.x), interpolatedVertex);

            float& z = m_ZBuffer[linearpos];
            if (interpolatedVertex.m_ScreenPosition.z < z)
            {
                if (m_ZWrite)
                    z = interpolatedVertex.m_ScreenPosition.z;
            }
            else if (m_ZTest)
            {
                continue;
            }

            interpolatedVertex.m_Color = interpolatedVertex.m_Color * color;
            Vector4f finalColor = FragmentShader(interpolatedVertex);
            PutPixelUnsafe(ScrrenX, ScrrenY, Vector4f::ToARGB(finalColor));
            pixelsDrawn++;
        }
    }

    ///stats.FinishDrawCallStats(min,max,pixelsDrawn);
}

void SoftwareRenderer::DrawTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int minY, int maxY)
{
    DrawLine(A, B, color, minY, maxY);
    DrawLine(C, B, color, minY, maxY);
    DrawLine(C, A, color, minY, maxY);
}

void SoftwareRenderer::DrawTriangleBoundingBox(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int minY, int maxY)
{
    Vector2f min = A.m_ScreenPosition.CWiseMin(B.m_ScreenPosition).CWiseMin(C.m_ScreenPosition).xy();
    Vector2f max = A.m_ScreenPosition.CWiseMax(B.m_ScreenPosition).CWiseMax(C.m_ScreenPosition).xy();

    if (max.y < minY ||
        min.y > maxY)
        return;

    // clamp min and max points to screen size so we don't calculate points that we don't see
    min = min.CWiseMin(Vector2f(m_ScreenSize.x - 1, maxY)).CWiseMax(Vector2f(0, minY));
    max = max.CWiseMin(Vector2f(m_ScreenSize.x - 1, maxY)).CWiseMax(Vector2f(0, minY));

    DrawLine(min, Vector2f{ max.x,min.y }, color, minY, maxY);
    DrawLine(Vector2f{ max.x,min.y }, max, color, minY, maxY);
    DrawLine(max, Vector2f{ min.x,max.y }, color, minY, maxY);
    DrawLine(Vector2f{ min.x,max.y }, min, color, minY, maxY);
}

void SoftwareRenderer::DrawLine(const TransformedVertex& VA, const TransformedVertex& VB, const Vector4f& color, int minY, int maxY)
{
    Vector2f A = VA.m_ScreenPosition.xy();
    Vector2f B = VB.m_ScreenPosition.xy();

    return DrawLine(A, B, color, minY, maxY);
}

void SoftwareRenderer::DrawLine(Vector2f A, Vector2f B, const Vector4f& color, int minY, int maxY)
{
    // Clip whole line against screen bounds
    if ((A.x < 0 && B.x < 0) ||
        (A.y < minY && B.y < minY) ||
        (A.x >= m_ScreenSize.x && B.x >= m_ScreenSize.x) ||
        (A.y > maxY && B.y > maxY))
        return;

    // Handle case when start end end point are on the same pixel
    if (int(A.x) == int(B.x) && int(A.y) == int(B.y))
    {
        PutPixel(int(A.x), int(A.y), Vector4f::ToARGB(color));
        return;
    }

    Vector2f dir = B - A;

    // Clip point A to minimum Y
    if (A.y < minY)
    {
        float t = (minY - A.y) / dir.y;
        A = A + dir * t;
    }
    // Clip point A to maximum Y
    else if (A.y > maxY)
    {
        float t = (maxY - A.y) / dir.y;
        A = A + dir * t;
    }

    // Clip point B to minimum Y
    if (B.y > maxY)
    {
        float t = (maxY - A.y) / dir.y;
        B = A + dir * t;
    }
    // Clip point B to maximum Y
    else if (B.y < minY)
    {
        float t = (minY - A.y) / dir.y;
        B = A + dir * t;
    }

    // y = ax + b
    float a = (B.y - A.y) / (B.x - A.x);
    float b = B.y - a * B.x;
    uint32_t intColor = Vector4f::ToARGB(color);

    if (abs(dir.x) >= abs(dir.y))
    {
        int startX = A.x;
        int endX = B.x;

        if (startX > endX)
        {
            startX = B.x;
            endX = A.x;
        }

        for (int x = startX; x < endX; ++x)
        {
            float y = a * x + b;
            PutPixel(x, y, intColor);
        }
    }
    else
    {
        float a = (B.x - A.x) / (B.y - A.y);
        float b = B.x - a * B.y;

        int startY = A.y;
        int endY = B.y;

        if (startY > endY)
        {
            startY = B.y;
            endY = A.y;
        }

        for (int y = startY; y < endY; ++y)
        {
            float x = a * y + b;
            PutPixel(x, y, intColor);
        }
    }
}

void SoftwareRenderer::SetModelMatrix(const Matrix4f& other)
{
    m_ModelMatrix = other;
    UpdateMVPMatrix();
}

void SoftwareRenderer::SetViewMatrix(const Matrix4f& other)
{
    m_ViewMatrix = other;
    Matrix4f inversedViewMatrix = m_ViewMatrix.Inversed();
    m_CameraPosition = Vector3f(inversedViewMatrix[12], inversedViewMatrix[13], inversedViewMatrix[14]);

    m_RenderParamsCPU.m_CameraPosition = Vector3f256C{ m_CameraPosition.x , m_CameraPosition.y , m_CameraPosition.z };
    m_RenderParamsSSE.m_CameraPosition = Vector3f128S{ m_CameraPosition.x , m_CameraPosition.y , m_CameraPosition.z };
    m_RenderParamsSSE8.m_CameraPosition = Vector3f128S8{ m_CameraPosition.x , m_CameraPosition.y , m_CameraPosition.z };
    m_RenderParamsAVX.m_CameraPosition = Vector3f256A{ m_CameraPosition.x , m_CameraPosition.y , m_CameraPosition.z };

    UpdateMVPMatrix();
}

void SoftwareRenderer::SetProjectionMatrix(const Matrix4f& other)
{
    m_ProjectionMatrix = other;
    UpdateMVPMatrix();
}

void SoftwareRenderer::SetTexture(shared_ptr<ITexture> texture)
{
    m_Texture = dynamic_pointer_cast<Texture>(texture);
    if (!m_Texture)
        m_Texture = m_DefaultTexture;
}

void SoftwareRenderer::SetWireFrameColor(const Vector4f& wireFrameColor)
{
    m_WireFrameColor = wireFrameColor;
}

void SoftwareRenderer::SetClearColor(const Vector4f& clearColor)
{
    m_ClearColor = Vector4f::ToARGB(clearColor);
}

void SoftwareRenderer::SetDiffuseColor(const Vector3f& diffuseColor)
{
    m_DiffuseColor = diffuseColor;
    m_RenderParamsCPU.m_DiffuseColor = Vector3f256C{ diffuseColor.x , diffuseColor.y , diffuseColor.z };
    m_RenderParamsSSE.m_DiffuseColor = Vector3f128S{ diffuseColor.x , diffuseColor.y , diffuseColor.z };
    m_RenderParamsSSE8.m_DiffuseColor = Vector3f128S8{ diffuseColor.x , diffuseColor.y , diffuseColor.z };
    m_RenderParamsAVX.m_DiffuseColor = Vector3f256A{ diffuseColor.x , diffuseColor.y , diffuseColor.z };
}

void SoftwareRenderer::SetAmbientColor(const Vector3f& ambientColor)
{
    m_AmbientColor = ambientColor;
    m_RenderParamsCPU.m_AmbientColor = Vector3f256C{ ambientColor.x , ambientColor.y , ambientColor.z };
    m_RenderParamsSSE.m_AmbientColor = Vector3f128S{ ambientColor.x , ambientColor.y , ambientColor.z };
    m_RenderParamsSSE8.m_AmbientColor = Vector3f128S8{ ambientColor.x , ambientColor.y , ambientColor.z };
    m_RenderParamsAVX.m_AmbientColor = Vector3f256A{ ambientColor.x , ambientColor.y , ambientColor.z };
}

void SoftwareRenderer::SetLightPosition(const Vector3f& lightPosition)
{
    m_LightPosition = lightPosition;
    m_RenderParamsCPU.m_LightPosition = Vector3f256C{ lightPosition.x , lightPosition.y , lightPosition.z };
    m_RenderParamsSSE.m_LightPosition = Vector3f128S{ lightPosition.x , lightPosition.y , lightPosition.z };
    m_RenderParamsSSE8.m_LightPosition = Vector3f128S8{ lightPosition.x , lightPosition.y , lightPosition.z };
    m_RenderParamsAVX.m_LightPosition = Vector3f256A{ lightPosition.x , lightPosition.y , lightPosition.z };
}

void SoftwareRenderer::SetDiffuseStrength(float diffuseStrength)
{
    m_DiffuseStrength = diffuseStrength;
}

void SoftwareRenderer::SetAmbientStrength(float ambientStrength)
{
    m_AmbientStrength = ambientStrength;
}

void SoftwareRenderer::SetSpecularStrength(float specularStrength)
{
    m_SpecularStrength = specularStrength;
}

void SoftwareRenderer::SetShininess(float shininess)
{
    m_Shininess = shininess;
}

void SoftwareRenderer::SetThreadsCount(uint8_t threadsCount)
{
    //if (threadsCount == 1)
    //    // no need to use thread pool for just 1 thread - execute work on main thread
    //    threadsCount = 0;

    if (!threadsCount)
        threadsCount = 1;

    if (m_ThreadsCount == threadsCount)
        return;

    m_ThreadsCount = threadsCount;
    //m_ThreadPool.SetThreadCount(m_ThreadsCount);
    m_TileThreadPool.SetThreadCount(m_ThreadsCount);
}

void SoftwareRenderer::SetColorizeThreads(bool colorizeThreads)
{
    m_ColorizeThreads = colorizeThreads;
}

void SoftwareRenderer::SetDrawWireframe(bool wireframe)
{
    m_DrawWireframe = wireframe;
}

void SoftwareRenderer::SetDrawBBoxes(bool drawBBoxes)
{
    m_DrawBBoxes = drawBBoxes;
}

void SoftwareRenderer::SetZWrite(bool zwrite)
{
    m_ZWrite = zwrite;
}

void SoftwareRenderer::SetZTest(bool ztest)
{
    m_ZTest = ztest;
}

void SoftwareRenderer::SetBlockMathMode(eBlockMathMode mathType)
{
    m_BlockMathMode = mathType;
}

void SoftwareRenderer::SetTileMode(eTileMode tileMode)
{
    if (m_TileMode == tileMode)
        return;

    switch (tileMode)
    {
    case eTileMode::Tile_32x32:
        m_TileSize = 32;
        break;
    case eTileMode::Tile_16x16:
        m_TileSize = 16;
        break;
    case eTileMode::Tile_8x8:
        m_TileSize = 8;
        break;
    }

    RecreateBuffers(m_TileSize, m_ScreenSize.x, m_ScreenSize.y);
}

void SoftwareRenderer::SetAlphaBlending(bool enable)
{
    m_AlphaBlend = enable;
}

void SoftwareRenderer::SetBackfaceCulling(bool enable)
{
    m_BackfaceCulling = enable;
}

int SoftwareRenderer::GetPixelsDrawn() const
{
    int pixels = 0;
    for (auto x : m_ZBuffer)
    {
        if (x != 1.0f)
            pixels++;
    }
    return pixels;
}
