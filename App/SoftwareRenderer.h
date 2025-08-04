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

struct DrawFunctionConfig
{
    static constexpr inline uint8_t Bits = 2;

    constexpr DrawFunctionConfig() = default;
    constexpr DrawFunctionConfig( uint8_t Value )
    {
        ZTest  = !!(Value & 0x01) ;
        ZWrite = !!(Value & 0x02) ;
    };

    constexpr uint8_t ToIndex()const noexcept
    {
        return (ZTest  ? 0x01 : 0)
             | (ZWrite ? 0x02 : 0);
    }

    bool ZTest     = false;
    bool ZWrite    = false;
};

enum class eDrawTriVersion : uint8_t
{
    DrawTriBaseline = 0,
    DrawTriv2   ,
    DrawTriv3   ,
    DrawTriv4   ,
    DrawTriv5   ,
    DrawTri     ,
};

template< typename T >
struct DrawFunctionConfigArray
{
    static constexpr inline auto SIZE = 1 << DrawFunctionConfig::Bits;

    constexpr DrawFunctionConfigArray()
    {
        uint8_t i = 0;
        for ( auto& El : m_Array )
            El.Config = DrawFunctionConfig{ i++ };
    }

    struct Element
    {
        DrawFunctionConfig Config;
        T                  Function = {};
    };
    Element m_Array[SIZE];
};

class SoftwareRenderer : public IRenderer
{
public:
    enum class eThreadTaskType : uint8_t
    {
        Unknown,
        Clear,
        RenderTile,
        ComposeTile,
    };

    struct TriangleData;
    struct ThreadTask;
    struct DrawTileData;

    SoftwareRenderer(int screenWidth, int screenHeight);
    ~SoftwareRenderer();

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
    void SetMathType(eMathType mathType)override;

    template< typename T >
    static T EdgeFunction(const Vector2<T>& A, const Vector2<T>& B, const Vector2<T>& C);
private:
    template< typename , DrawFunctionConfig >
    void DrawFilledTriangleBaseline(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int minY, int maxY, DrawStats& stats);

    template< typename MathT , DrawFunctionConfig >
    void DrawFilledTriangle_v2(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int minY, int maxY, DrawStats& stats);

    template< typename MathT , DrawFunctionConfig >
    void DrawFilledTriangle_v3(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int minY, int maxY, DrawStats& stats);

    template< typename MathT , DrawFunctionConfig Config >
    void DrawFilledTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int minY, int maxY, DrawStats& stats);


    template< eSimdType Type , bool Partial , int Elements = 8 >
    void DrawFulllTileImplSimd(const DrawTileData& TD, DrawStats* stats);

    template< bool Partial >
    void DrawTileImpl   (const DrawTileData& TD, DrawStats* stats);
    void DrawFullTile   (const DrawTileData& TD, DrawStats* stats);
    void DrawPartialTile(const DrawTileData& TD, DrawStats* stats);
    void DrawTile       (const DrawTileData& TD, DrawStats* stats);

    void DrawFilledTriangleT(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, DrawStats& stats);


    template< typename MathT , DrawFunctionConfig Config >
    void DrawFilledTriangles(const TransformedVertex* pVerts, size_t Count, const Vector4f& color, int minY, int maxY, DrawStats& stats);

    void DrawFilledTriangleWTF(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int minY, int maxY, DrawStats& stats);
    void DrawTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int MinY, int maxY);
    void DrawTriangleBoundingBox(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int MinY, int maxY);
    void DrawLine(const TransformedVertex& A, const TransformedVertex& B, const Vector4f& color, int MinY, int maxY);
    void DrawLine(Vector2f A, Vector2f B, const Vector4f& color, int MinY, int maxY);
    void PutPixel(int x, int y, uint32_t color);
    void PutPixelUnsafe(int x, int y, uint32_t color);
    void UpdateMVPMatrix();
    void DoRender(const vector<Vertex>& vertices, int MinY, int maxY, int threadID);

    struct ALIGN_FOR_AVX AlignedPixel : Vector4f{};

    struct TileInfo
    {
        using at_i16 = std::atomic<uint16_t>;

        Vector2si       TileIndex;
        Spinlock        Lock;
        mutable at_i16  DrawCount       = 0;
        Vector2i        TileScreenPos;
        Vector4f*       TileMem         = nullptr;
        float*          TileZMem        = nullptr;
    };

    inline TileInfo* GetTileInfo(const Vector2si& tileIndex)
    {
        return &m_TilesGrid[tileIndex.y * m_TilesGridSize.x + tileIndex.x];
    }

    struct Internal;

    Vector4f FragmentShader(const TransformedVertex& vertex);
    template< int Elements , eSimdType Type >
    Vector4<fsimd<Elements,Type>> FragmentShader(const SimdTransformedVertex<Elements,Type>& vertex);

    // 8 bit - one channel (8*4=32 - rgba)
    vector<uint32_t>    m_ScreenBuffer;
    vector<AlignedPixel>m_TilesBuffer;
    unique_ptr<TileInfo[]>m_TilesGrid;
    vector<float>       m_ZBuffer;
    Vector2si           m_TilesGridSize;
    Vector2si           m_LastTile;

    Vector4f            m_WireFrameColor = Vector4f(1, 1, 1, 1);
    Vector3f            m_DiffuseColor = Vector3f(1, 1, 1);
    Vector3f            m_AmbientColor = Vector3f(1, 1, 1);
    Vector3f            m_LightPosition = Vector3f(0, 0, -20);
    Vector3f            m_CameraPosition = Vector3f(0, 0, 0);
    Vector4f            m_ThreadColors[16];
    uint32_t            m_ClearColor = 0xFF000000;

    Vector3f256A        m_LightPositionSimd;
    Vector3f256A        m_DiffuseColorSimd;
    Vector3f256A        m_AmbientColorSimd;
    Vector3f256A        m_CameraPositionSimd;

    Matrix4f            m_ModelMatrix;
    Matrix4f            m_ViewMatrix;
    Matrix4f            m_ProjectionMatrix;
    Matrix4f            m_MVPMatrix;

    bool                m_DrawWireframe = false;
    bool                m_ColorizeThreads = false;
    bool                m_DrawBBoxes = false;
    bool                m_ZWrite = true;
    bool                m_ZTest = true;
    std::atomic<bool>   m_ShutingDown = false;
    eDrawTriVersion     m_DrawTriVersion = eDrawTriVersion::DrawTri;
    float               m_DiffuseStrength = 0.3f;
    float               m_AmbientStrength = 0.5f;
    float               m_SpecularStrength = 0.9f;
    float               m_Shininess = 32.0f;
    uint8_t             m_ThreadsCount = 0;

    atomic_int          m_FrameTriangles = 0;
    atomic_int          m_FrameTrianglesDrawn = 0;
    atomic_int          m_FramePixels = 0;
    atomic_int          m_FramePixelsDrawn = 0;
    atomic_int          m_FrameRasterTimeUS = 0;
    atomic_int          m_FrameTransformTimeUS = 0;
    atomic_int          m_FrameDrawTimeThreadUS = 0;
    atomic_int          m_FrameDrawTimeMainUS = 0;
    atomic_int          m_FillrateKP = 0;

    DrawStats           m_DrawStats;

    template< typename ... Args >
    TriangleData*       PushTriangleData( Args ... args );
    void                PushTile( DrawTileData data );

    vector<TriangleData> m_TrianglesData;
    vector<DrawTileData>    m_TilesData;

    std::atomic<TriangleData*>      m_pCurrentTriangleData = nullptr;
    std::atomic<DrawTileData*>      m_pCurrentTileData = nullptr;
    std::atomic<DrawTileData*>      m_pCurrentTileJob = nullptr;

    shared_ptr<Texture> m_Texture;
    shared_ptr<Texture> m_DefaultTexture;
    SimpleThreadPool    m_ThreadPool;
    SimpleThreadPool    m_TileThreadPool;

    int                 m_MathIndex = 0;
    MathCPU             m_MathCPU;
    MathSSE             m_MathSSE;
    MathAVX             m_MathAVX;
    const IMath*        m_MathArray[3] = { &m_MathCPU , &m_MathSSE , &m_MathAVX };
    //const IMath*        m_MathArray[3] = { &m_MathCPU , &m_MathSSE , &m_MathCPU };
    const IMath*        m_pSelectedMath = &m_MathCPU;
};
