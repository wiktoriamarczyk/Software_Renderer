/*
* Master’s thesis - Analysis of selected optimization techniques for a 3D software renderer
* Author: Wiktoria Marczyk
* Year: 2025
*/

#pragma once
#include "IRenderer.h"
#include "TransformedVertex.h"
#include "Texture.h"
#include "SimpleThreadPool.h"

#include <memory_resource>
#include "assimp/Compiler/pstdint.h"

#include "TransientAllocator.h"
#include "function_ref.h"
#include "RenderCommands.h"
#include "RenderCommandsBuffer.h"

struct TileInfo
{
    using at_i16 = atomic<uint16_t>;
    using at_CmdDrawTile = atomic<CommandRenderTile*>;

    Vector2si               m_TileIndex;
    uint32_t                m_TileMemOffset = 0;
    uint32_t                m_TileZOffset = 0;
    mutable at_i16          m_DrawCount = 0;
    Spinlock                m_Lock;
    mutable at_CmdDrawTile  m_pRenderTileCmd = nullptr;
};

struct TileData;

struct DrawFunctionConfig
{
    static constexpr inline uint8_t Bits = 2;

    constexpr DrawFunctionConfig() = default;

    constexpr DrawFunctionConfig(uint8_t Value)
    {
        m_ZTest = !!(Value & 0x01);
        m_ZWrite = !!(Value & 0x02);
    };

    constexpr uint8_t ToIndex()const noexcept
    {
        return (m_ZTest ? 0x01 : 0)
            | (m_ZWrite ? 0x02 : 0);
    }

    bool m_ZTest = false;
    bool m_ZWrite = false;
};

enum class eDrawTriVersion : uint8_t
{
    DrawTriBaseline = 0,
    DrawTriv2,
    DrawTriv3,
    DrawTriv4,
    DrawTriv5,
    DrawTri,
};

template< int Elements, eSimdType Type = eSimdType::None >
struct RenderParamsSimd
{
    Vector3<fsimd<Elements, Type>> m_LightPosition;
    Vector3<fsimd<Elements, Type>> m_DiffuseColor;
    Vector3<fsimd<Elements, Type>> m_AmbientColor;
    Vector3<fsimd<Elements, Type>> m_CameraPosition;
};

class SoftwareRenderer : public IRenderer
{
public:
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
    void SetBlockMathMode(eBlockMathMode mathType)override;
    virtual void SetTileMode(eTileMode tileMode)override;
    virtual void SetAlphaBlending(bool Enable)override;
    virtual void SetBackfaceCulling(bool Enable)override;
    virtual int GetPixelsDrawn() const override;

    template< typename T >
    static T EdgeFunction(const Vector2<T>& A, const Vector2<T>& B, const Vector2<T>& C);

    template< typename T >
    static Vector2<T> EdgeFunctionSeparate(const Vector2<T>& A, const Vector2<T>& B, const Vector2<T>& C);
private:
    struct RenderThreadData;

    template<typename, DrawFunctionConfig>
    void DrawFilledTriangleBaseline(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int minY, int maxY, DrawStats& stats);

    template<typename MathT, DrawFunctionConfig>
    void DrawFilledTriangle_v2(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int minY, int maxY, DrawStats& stats);

    template<typename MathT, DrawFunctionConfig>
    void DrawFilledTriangle_v3(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int minY, int maxY, DrawStats& stats);

    template<typename MathT, DrawFunctionConfig Config>
    void DrawFilledTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int minY, int maxY, DrawStats& stats);

    template<uint8_t TILE_SIZE, eSimdType Type, int Elements = 8>
    void DrawTileImplSimd(const CommandRenderTile& TD, RenderThreadData& data);

    template<uint8_t TILE_SIZE>
    void DrawTileImpl(const CommandRenderTile& TD, RenderThreadData& data);

    void DrawTile(const CommandRenderTile& TD, RenderThreadData& data);
    void GenerateTileJobs(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, DrawStats& stats, const PipelineSharedData* pPipelineSharedData, uint32_t tri_index, pmr::vector<const Command*>& outCommmands);

    template<typename MathT, DrawFunctionConfig Config>
    void DrawFilledTriangles(const TransformedVertex* pVerts, size_t Count, const Vector4f& color, int minY, int maxY, DrawStats& stats);

    void DrawTriangle(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int MinY, int maxY);
    void DrawTriangleBoundingBox(const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& color, int MinY, int maxY);
    void DrawLine(const TransformedVertex& A, const TransformedVertex& B, const Vector4f& color, int MinY, int maxY);
    void DrawLine(Vector2f A, Vector2f B, const Vector4f& color, int MinY, int maxY);
    void PutPixel(int x, int y, uint32_t color);
    void PutPixelUnsafe(int x, int y, uint32_t color);
    void UpdateMVPMatrix();
    void DoRender(const vector<Vertex>& vertices, int MinY, int maxY, int threadID);

    struct ALIGN_FOR_AVX AlignedPixel : Vector4f {};

    inline TileInfo* GetTileInfo(const Vector2si& tileIndex)
    {
        return &m_TilesGrid[tileIndex.y * m_TilesGridSize.x + tileIndex.x];
    }

    void RendererTaskWorker();
    void RecreateBuffers(uint8_t TileSize, int screenWidth, int screenHeight);
    void VertexAssemply(const CommandVertexAssemply& cmd);
    void ExecuteExitCommand();
    void ClearBuffers(const CommandClear& cmd);
    void Fill32BitBuffer(const CommandFill32BitBuffer& cmd);
    void VertexTransformAndClip(const CommandVertexTransformAndClip& cmd, RenderThreadData& data);
    void ProcessTriangles(const CommandProcessTriangles& cmd, RenderThreadData& data);
    void WaitForSync(const CommandSyncBarier& cmd);
    CommandBuffer* AllocTransientCommandBuffer() { return CommandBuffer::CreateCommandBuffer(m_TransientMemoryResource); }
    Vector4f FragmentShader(const TransformedVertex& vertex);
    template<int Elements, eSimdType Type>
    Vector4<fsimd<Elements, Type>> FragmentShader(const SimdTransformedVertex<Elements, Type>& vertex);

    // 8 bit - one channel (8*4=32 - rgba)
    vector<uint32_t>    m_ScreenBuffer;
    Vector2si           m_ScreenSize;
    Vector2si           m_PublicScreenSize;
    uint8_t             m_TileSize = 32;
    eTileMode           m_TileMode = eTileMode::Tile_32x32;

    vector<AlignedPixel> m_TilesBuffer;
    unique_ptr<TileInfo[]> m_TilesGrid;
    vector<float>       m_ZBuffer;
    Vector2si           m_TilesGridSize;
    Vector2si           m_LastTile;

    Vector4f            m_WireFrameColor = Vector4f(1, 1, 1, 1);
    Vector4f            m_VertexColor = Vector4f(1, 1, 1, 1);
    Vector3f            m_DiffuseColor = Vector3f(1, 1, 1);
    Vector3f            m_AmbientColor = Vector3f(1, 1, 1);
    Vector3f            m_LightPosition = Vector3f(0, 0, -20);
    Vector3f            m_CameraPosition = Vector3f(0, 0, 0);
    uint32_t            m_ClearColor = 0xFF000000;

    using RenderPrmCPU = RenderParamsSimd<8, eSimdType::CPU>;
    using RenderPrmSSE = RenderParamsSimd<4, eSimdType::SSE>;
    using RenderPrmSSE8 = RenderParamsSimd<8, eSimdType::SSE>;
    using RenderPrmAVX = RenderParamsSimd<8, eSimdType::AVX>;

    RenderPrmCPU        m_RenderParamsCPU;
    RenderPrmSSE        m_RenderParamsSSE;
    RenderPrmSSE8       m_RenderParamsSSE8;
    RenderPrmAVX        m_RenderParamsAVX;

    template<int Elements, eSimdType Type>
    auto& GetRenderParams()
    {
        if constexpr (Type == eSimdType::CPU && Elements == 8)
            return m_RenderParamsCPU;
        if constexpr (Type == eSimdType::SSE && Elements == 4)
            return m_RenderParamsSSE;
        else if constexpr (Type == eSimdType::SSE && Elements == 8)
            return m_RenderParamsSSE8;
        else if constexpr (Type == eSimdType::AVX)
            return m_RenderParamsAVX;
    }

    Matrix4f            m_ModelMatrix;
    Matrix4f            m_ViewMatrix;
    Matrix4f            m_ProjectionMatrix;
    Matrix4f            m_MVPMatrix;

    bool                m_DrawWireframe = false;
    bool                m_ColorizeThreads = false;
    bool                m_DrawBBoxes = false;
    bool                m_ZWrite = true;
    bool                m_ZTest = true;
    bool                m_AlphaBlend = false;
    bool                m_BackfaceCulling = false;
    std::atomic<bool>   m_ShutingDown = false;
    eDrawTriVersion     m_DrawTriVersion = eDrawTriVersion::DrawTri;
    float               m_DiffuseStrength = 0.3f;
    float               m_AmbientStrength = 0.5f;
    float               m_SpecularStrength = 0.9f;
    float               m_Shininess = 32.0f;
    uint8_t             m_ThreadsCount = 0;
    CommandBuffer*      m_pCommandBuffer = nullptr;

    atomic_int          m_FrameTriangles = 0;
    atomic_int          m_FrameTrianglesDrawn = 0;
    atomic_int          m_FramePixels = 0;
    atomic_int          m_FramePixelsDrawn = 0;
    atomic_int          m_FramePixelsCalcualted = 0;
    atomic_int          m_FrameRasterTimeUS = 0;
    atomic_int          m_FrameTransformTimeUS = 0;
    atomic_int          m_FrameDrawTimeThreadUS = 0;
    atomic_int          m_FrameDrawTimeMainUS = 0;
    atomic_int          m_FrameDrawsPerTile = 0;
    atomic_int          m_FillrateKP = 0;

    DrawStats           m_DrawStats;

    shared_ptr<Texture> m_Texture;
    shared_ptr<Texture> m_DefaultTexture;
    SimpleThreadPool    m_TileThreadPool;

    transient_memory_resource m_TransientMemoryResource;
    transient_allocator m_TransientAllocator{ m_TransientMemoryResource };

    eBlockMathMode      m_BlockMathMode = eBlockMathMode::AVXx256;
};