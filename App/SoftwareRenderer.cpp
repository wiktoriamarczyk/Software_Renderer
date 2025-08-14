/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#include "immintrin.h"

#include "SoftwareRenderer.h"
#include "TransformedVertex.h"
#include "VertexInterpolator.h"
#include <array>
#include <chrono>


bool g_useSimd               = true;
bool g_showTilesBoundry      = false;
bool g_showTilestype         = false;
bool g_showTriangleBoundry   = false;
bool g_showCornersClassify   = false;
bool g_showTilesGrid         = false;
bool g_ThreadPerTile         = true;
auto g_max_overdraw          = std::atomic<int>{0};

struct PipelineSharedData
{
    Plane          m_NearFrustumPlane;
    CommandBuffer* m_pProcessTrianglesCmdBuffer = nullptr;
    SyncBarrier*   m_pProcessTrianglesCmdBufferSync = nullptr;
    CommandBuffer* m_pRenderTilesCmdBuffer = nullptr;
    SyncBarrier*   m_pRenderTilesCmdBufferSync = nullptr;
    DrawConfig*    m_pDrawConfig = nullptr;
};

template< typename T >
struct EdgeFunctionRails
{
    EdgeFunctionRails( std::nullptr_t ){}
    EdgeFunctionRails( const Vector2<T>& A , const Vector2<T>& B , const Vector2<T>& C , const Vector2<T>& SP )
        : m_ABP_Stride{ A.y - B.y , B.x - A.x }
        , m_BCP_Stride{ B.y - C.y , C.x - B.x }
        , m_CAP_Stride{ C.y - A.y , A.x - C.x }

        , m_ABP_Start{ m_ABP_Stride.x * SP.x , m_ABP_Stride.y * SP.y - m_ABP_Stride.Dot( A ) }
        , m_BCP_Start{ m_BCP_Stride.x * SP.x , m_BCP_Stride.y * SP.y - m_BCP_Stride.Dot( B ) }
        , m_CAP_Start{ m_CAP_Stride.x * SP.x , m_CAP_Stride.y * SP.y - m_CAP_Stride.Dot( C ) }

        , m_SP{ SP.x , SP.y }
    {
    };

    Vector3<T> GetEdgeFunctionsXStride()const{
        return Vector3<T>( m_ABP_Stride.x , m_BCP_Stride.x , m_CAP_Stride.x );
    }
    Vector3<T> GetEdgeFunctionsYStride()const{
        return Vector3<T>( m_ABP_Stride.y , m_BCP_Stride.y , m_CAP_Stride.y );
    }

    Vector3<T> GetEdgeFunctionsXStart()const{
        return Vector3<T>( m_ABP_Start.x , m_BCP_Start.x , m_CAP_Start.x );
    }
    Vector3<T> GetEdgeFunctionsYStart()const{
        return Vector3<T>( m_ABP_Start.y , m_BCP_Start.y , m_CAP_Start.y );
    }

    struct Start
    {
        Vector3<T> x;
        Vector3<T> y;
    };

    Start GetStartFor( const Vector2<T>& P )const
    {
        auto PDiff = P - m_SP;
        Start Result;

        Result.x = GetEdgeFunctionsXStart() + GetEdgeFunctionsXStride() * PDiff.x;
        Result.y = GetEdgeFunctionsYStart() + GetEdgeFunctionsYStride() * PDiff.y;
        return Result;
    };

    Vector2<T> m_ABP_Stride;
    Vector2<T> m_BCP_Stride;
    Vector2<T> m_CAP_Stride;

    Vector2<T> m_ABP_Start;
    Vector2<T> m_BCP_Start;
    Vector2<T> m_CAP_Start;

    Vector2<T> m_SP;
};

enum eTileCoverage : uint8_t
{
    Undefined   = 0b000, // undefined state, should not be used
    Outside     = 0b001,
    Partial     = 0b011,
    Inside      = 0b111,
};

eTileCoverage operator|( eTileCoverage a , eTileCoverage b )
{
    return static_cast<eTileCoverage>( static_cast<uint8_t>(a) | static_cast<uint8_t>(b) );
}
eTileCoverage operator&( eTileCoverage a , eTileCoverage b )
{
    return static_cast<eTileCoverage>( static_cast<uint8_t>(a) & static_cast<uint8_t>(b) );
}

template< typename T >
inline T SoftwareRenderer::EdgeFunction(const Vector2<T>& A, const Vector2<T>& B, const Vector2<T>& C)
{
    return (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
}

inline void DrawStats::FinishDrawCallStats(Vector2<int> min, Vector2<int> max, int pixelsDrawn)
{
    m_FramePixels += (1+max.y-min.y)*(1+max.x-min.x);
    m_FramePixelsDrawn += pixelsDrawn;
    m_FrameTriangles++;
    m_FrameTrianglesDrawn++;
}

shared_ptr<ITexture> SoftwareRenderer::LoadTexture(const char* fileName) const
{
    if (!fileName || fileName[0]==0)
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
    m_pCommandBuffer->PushCommand<CommandClear>( m_ClearColor );

}

void SoftwareRenderer::ClearZBuffer()
{
    ZoneScoped;
    //std::fill(m_ZBuffer.begin(), m_ZBuffer.end(), 1.f);
    //return;
    m_pCommandBuffer->PushCommand<CommandClear>( std::nullopt , 1.0F );
}

inline Vector4f SoftwareRenderer::FragmentShader(const TransformedVertex& vertex)
{
    // Normalize the interpolated normal vector
    auto vertexNormal = vertex.m_Normal.FastNormalized();

    Vector4f sampledPixel = m_Texture->Sample(vertex.m_UV);
    //return sampledPixel * vertex.m_Color;

    Vector3f pointToLightDir = (m_LightPosition- vertex.m_WorldPosition).FastNormalized();

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
    Vector4f finalColor = Vector4f(sumOfLight,1.0f) * sampledPixel * vertex.m_Color;

    return finalColor;
}

template< int Elements , eSimdType Type >
inline Vector4<fsimd<Elements,Type>> SoftwareRenderer::FragmentShader(const SimdTransformedVertex<Elements,Type>& vertex)
{
    // Normalize the interpolated normal vector
    auto vertexNormal = vertex.m_Normal.FastNormalized();

    Vector4f256<Type> sampledPixel = m_Texture->Sample(vertex.m_UV);
    //return sampledPixel * vertex.m_Color;

    Vector3f256<Type> pointToLightDir = (m_LightPositionSimd- vertex.m_WorldPosition).FastNormalized();

    // ambient - light that is reflected from other objects
    Vector3f256<Type> ambient = m_AmbientColorSimd * m_AmbientStrength;

    // diffuse - light that is reflected from light source
    auto diffuseFactor = Math::Max(pointToLightDir.Dot(vertexNormal), f256<Type>::Zero );
    Vector3f256<Type> diffuse = m_DiffuseColorSimd * diffuseFactor * m_DiffuseStrength;

    // specular - light that is reflected from light source and is reflected in one direction
    // specular = specularStrength * specularColor * pow(max(dot(viewDir, reflectDir), 0.0), shininess)
    Vector3f256<Type> viewDir = (m_CameraPositionSimd - vertex.m_WorldPosition).FastNormalized();
    Vector3f256<Type> reflectDir = (pointToLightDir * -1).Reflect(vertexNormal);
    auto specularFactor = Math::Max(viewDir.Dot(reflectDir), f256<Type>::Zero).pow(m_Shininess);
    Vector3f256<Type> specular = m_DiffuseColorSimd * m_SpecularStrength * specularFactor;

    // final light color = (ambient + diffuse + specular) * modelColor
    Vector3f256<Type> sumOfLight = ambient + diffuse + specular;
    sumOfLight = sumOfLight.CWiseMin(Vector3f256<Type>(1, 1, 1));
    Vector4f256<Type> finalColor = Vector4f256<Type>(sumOfLight,1.0f) * sampledPixel * vertex.m_Color;

    return finalColor;
}
static constexpr int TILE_SIZE = 32;

static_assert(SCREEN_WIDTH  % TILE_SIZE == 0, "SCREEN_WIDTH must be divisible by TILE_SIZE");
static_assert(SCREEN_HEIGHT % TILE_SIZE == 0, "SCREEN_HEIGHT must be divisible by TILE_SIZE");

struct IntegrerPrecision
{
    int Bits           = 4;
    int Multiplier     = 1 << Bits;
    int Mask           = Multiplier - 1;
};
constexpr IntegrerPrecision Precision{ 0 };

struct ALIGN_FOR_AVX TriangleData
{
    TriangleData( std::nullptr_t )
        : m_Interpolator(nullptr)
        , m_EdgeFunctionRails(nullptr)
    {
    }
    TriangleData( const TransformedVertex& A, const TransformedVertex& B, const TransformedVertex& C, const Vector4f& Color , Vector2<int64_t> SPA , Vector2<int64_t> SPB , Vector2<int64_t> SPC , Vector2<int64_t> SP )
        : m_Interpolator(A, B, C, Color)
        , m_EdgeFunctionRails{ SPA , SPB , SPC , SP }
    {
    }

    VertexInterpolator         m_Interpolator;
    EdgeFunctionRails<int64_t> m_EdgeFunctionRails;
    float                      m_InvABC = 0.f;
    float                      m_MinZ = 0.f;
    float                      m_MaxZ = 0.f;
    uint32_t                   m_TriIndex = 0;
};

void transpose8Vec4f_to_Vec4f256(const Vector4f* in, Vector4f256A& out)
{
    __m256 tmp;

    __m256 row0 = _mm256_loadu_ps( in[0].Data() ); // A B C D  E F G H
    __m256 row1 = _mm256_loadu_ps( in[2].Data() ); // I J K L  M N O P
    __m256 row2 = _mm256_loadu_ps( in[4].Data() ); // Q R S T  U V W X
    __m256 row3 = _mm256_loadu_ps( in[6].Data() ); // Y Z 1 2  3 4 5 6


           tmp  = _mm256_permute2f128_ps( row0 ,row2 , 0x31 );
           row0 = _mm256_permute2f128_ps( row0 ,row2 , 0x20 ); // A B C D  Q R S T
           row2 = tmp;                                         // E F G H  U V W X
           tmp  = _mm256_permute2f128_ps( row1 ,row3 , 0x31 );
           row1 = _mm256_permute2f128_ps( row1 ,row3 , 0x20 ); // I J K L  Y Z 1 2
           row3 = tmp;                                         // M N O P  3 4 5 6

    // Unpack low/high
    __m256 t0 = _mm256_unpacklo_ps(row0, row2); // A E B F  Q U R V
    __m256 t1 = _mm256_unpackhi_ps(row0, row2); // C G D H  S W T X
    __m256 t2 = _mm256_unpacklo_ps(row1, row3); // I M J N  Y 3 Z 4
    __m256 t3 = _mm256_unpackhi_ps(row1, row3); // K O L P  1 5 2 6

    // Final permutes
    out.x.v =_mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0)); // A E I M Q U Y 3
    out.y.v =_mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2)); // B F J N R V Z 4
    out.z.v =_mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0)); // C G K O S W 1 5
    out.w.v =_mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2)); // D H L P T X 2 6
}



void transposeVec4f256_to_8Vec4f(const Vector4f256A& in, Vector4f* out)
{
    __m256 t0 = in.x.v;
    __m256 t1 = in.y.v;
    __m256 t2 = in.z.v;
    __m256 t3 = in.w.v;

    __m256 row0 = _mm256_unpacklo_ps(t0, t2);   // A E I M | Q U Y 3
                                                // C G K O | S W 1 5 -> row0 = A C E G | Q S U W
    __m256 row1 = _mm256_unpacklo_ps(t1, t3);   // B F J N | R V Z 4
                                                // D H L P | T X 2 6 -> row1 = B D F H | R T V X
    __m256 row2 = _mm256_unpackhi_ps(t0, t2);   // A E I M | Q U Y 3
                                                // C G K O | S W 1 5 -> row2 = I K M O | Y 1 3 5
    __m256 row3 = _mm256_unpackhi_ps(t1, t3);   // B F J N | R V Z 4
                                                // D H L P | T X 2 6 -> row3 = J L N P | Z 2 4 6

    t0 = _mm256_unpacklo_ps(row0, row1);        // A C E G | Q S U W
                                                // B D F H | R T V X t0 = -> A B C D | Q R S T
    t1 = _mm256_unpackhi_ps(row0, row1);        // A C E G | Q S U W
                                                // B D F H | R T V X t1 = -> E F G H | U V W X
    t2 = _mm256_unpacklo_ps(row2, row3);        // I K M O | Y 1 3 5
                                                // J L N P | Z 2 4 6 t2 = -> I J K L | Y Z 1 2
    t3 = _mm256_unpackhi_ps(row2, row3);        // I K M O | Y 1 3 5
                                                // J L N P | Z 2 4 6 t3 = -> M N O P | 3 4 5 6

    // row0                                     // A B C D | Q R S T
    row0 = _mm256_permute2f128_ps(t0, t1, 0x21);// Q R S T | E F G H

    // row2                                     // I J K L | Y Z 1 2
    row2 = _mm256_permute2f128_ps(t2, t3, 0x21);// Y Z 1 2 | M N O P


    __m256 vec1 = _mm256_blend_ps( t0 , row0 , 0b11110000 ); // A B C D | E F G H
    __m256 vec2 = _mm256_blend_ps( t2 , row2 , 0b11110000 ); // I J K L | M N O P
    __m256 vec3 = _mm256_blend_ps( row0 , t1 , 0b11110000 ); // Q R S T | U V W X
    __m256 vec4 = _mm256_blend_ps( row2 , t3 , 0b11110000 ); // Y Z 1 2 | 3 4 5 6

    _mm256_storeu_ps( out[0].Data() , vec1 );
    _mm256_storeu_ps( out[2].Data() , vec2 );
    _mm256_storeu_ps( out[4].Data() , vec3 );
    _mm256_storeu_ps( out[6].Data() , vec4 );
}

SoftwareRenderer::SoftwareRenderer(int screenWidth, int screenHeight)
{
    m_TransientMemoryResource.reset();
    m_ScreenBuffer.resize(screenWidth * screenHeight, 0);
    m_ZBuffer.resize(screenWidth * screenHeight, 0);

    m_TilesGridSize.x = (screenWidth  + TILE_SIZE-1) / TILE_SIZE;
    m_TilesGridSize.y = (screenHeight + TILE_SIZE-1) / TILE_SIZE;
    m_LastTile = Vector2si( m_TilesGridSize.x - 1 , m_TilesGridSize.y - 1 );
    m_TilesGrid.reset( new TileInfo[ m_TilesGridSize.x * m_TilesGridSize.y ] );

    constexpr auto TILE_PIXELS_COUNT = TILE_SIZE*TILE_SIZE;

    m_TilesBuffer.resize(m_TilesGridSize.x * m_TilesGridSize.y*TILE_PIXELS_COUNT);

    for( int y=0 ; y<m_TilesGridSize.y ; ++y )
    {
        for( int x=0 ; x<m_TilesGridSize.x ; ++x )
        {
            auto& Tile = m_TilesGrid[y * m_TilesGridSize.x + x];
            Tile.TileIndex     = Vector2si(x,y);
            Tile.TileScreenPos = Vector2i(x*TILE_SIZE, y*TILE_SIZE);
            Tile.TileMem       = m_TilesBuffer.data() + (y * m_TilesGridSize.x + x) * TILE_PIXELS_COUNT;
            Tile.TileZMem      = m_ZBuffer.data() +     (y * m_TilesGridSize.x + x) * TILE_PIXELS_COUNT;
            assert( Tile.TileZMem + TILE_PIXELS_COUNT <= m_ZBuffer.data() + m_ZBuffer.size() );
            assert( Tile.TileMem + TILE_PIXELS_COUNT <= m_TilesBuffer.data() + m_TilesBuffer.size() );
        }
    }

    m_ThreadColors[0]  = Vector4f(1.0f, 0.0f, 1.0f, 1.0f);
    m_ThreadColors[1]  = Vector4f(1.0f, 0.0f, 0.0f, 1.0f);
    m_ThreadColors[2]  = Vector4f(0.0f, 1.0f, 0.0f, 1.0f);
    m_ThreadColors[3]  = Vector4f(0.0f, 0.0f, 1.0f, 1.0f);
    m_ThreadColors[4]  = Vector4f(1.0f, 1.0f, 0.0f, 1.0f);
    m_ThreadColors[5]  = Vector4f(0.5f, 0.0f, 1.0f, 1.0f);
    m_ThreadColors[6]  = Vector4f(0.0f, 1.0f, 1.0f, 1.0f);
    m_ThreadColors[7]  = Vector4f(1.0f, 0.5f, 0.5f, 1.0f);
    m_ThreadColors[8]  = Vector4f(0.5f, 1.0f, 0.5f, 1.0f);
    m_ThreadColors[9]  = Vector4f(0.5f, 0.5f, 1.0f, 1.0f);
    m_ThreadColors[10] = Vector4f(1.0f, 1.0f, 0.5f, 1.0f);
    m_ThreadColors[11] = Vector4f(1.0f, 0.5f, 1.0f, 1.0f);

    m_ThreadColors[12] = Vector4f(0.2f, 0.5f, 1.0f, 1.0f);
    m_ThreadColors[13] = Vector4f(1.0f, 0.5f, 0.2f, 1.0f);
    m_ThreadColors[14] = Vector4f(0.5f, 0.2f, 1.0f, 1.0f);
    m_ThreadColors[15] = Vector4f(1.0f, 0.2f, 0.2f, 1.0f);

    m_DefaultTexture = std::make_shared<Texture>();
    m_DefaultTexture->CreateWhite4x4Tex();

    m_Texture = m_DefaultTexture;
    m_pCommandBuffer = AllocTransientCommandBuffer();

    m_TileThreadPool.SetThreadCount(16);
}

SoftwareRenderer::~SoftwareRenderer()
{
}

template< typename T >
FORCE_INLINE bool EdgeFunctionTest( const Vector3<T>& EdgeFunctionsValue )
{
    static_assert( std::is_signed_v<T> );

    if constexpr( std::is_integral_v<T> )
    {
        constexpr T NegativeTestBit = T(1) << (sizeof(T) * 8 - 1);

        return !!( (EdgeFunctionsValue.x | EdgeFunctionsValue.y | EdgeFunctionsValue.z ) & NegativeTestBit );
    }
    else
    {
        return (EdgeFunctionsValue.x | EdgeFunctionsValue.y | EdgeFunctionsValue.z ) < 0 ;
    }
}

template< bool Col , bool Cov , int Elements , eSimdType Simd >
void Vec4ToARGB_Simd( const simd<float,Elements,Simd>& Multiplier , const simd<int,Elements,Simd>& U8Masks , const Vector4f* pPixels , uint32_t* pColBuffer , const Vector4f* pCol = nullptr , const simd<int,Elements,Simd>* pCoverageMask = nullptr )
{
    using simdf = simd<float, Elements, Simd>;
    using simdi = simd<int, Elements, Simd>;

    const float* pFloats = pPixels->Data();

    simdf R{ pFloats+0*Elements , simd_alignment::AVX };
    simdf G{ pFloats+1*Elements , simd_alignment::AVX };
    simdf B{ pFloats+2*Elements , simd_alignment::AVX };
    simdf A{ pFloats+3*Elements , simd_alignment::AVX };

    R *= Multiplier;
    G *= Multiplier;
    B *= Multiplier;
    A *= Multiplier;

    if constexpr( Col )
    {
        R *= simdf(pCol->x);
        G *= simdf(pCol->y);
        B *= simdf(pCol->z);
        A *= simdf(pCol->w);
    }

    auto UR = ( R.static_cast_to<int>() & U8Masks );
    auto UG = ( G.static_cast_to<int>() & U8Masks ) << 8 ; // shift green to the left by 8 bits
    auto UB = ( B.static_cast_to<int>() & U8Masks ) << 16; // shift blue to the left by 16 bits
    auto UA = ( A.static_cast_to<int>() & U8Masks ) << 24; // shift alpha to the left by 24 bits

    simdi Piexl = UR | UG | UB | UA; // combine red, green, blue and alpha

    if constexpr( Cov )
        Piexl.store( (int*)pColBuffer , *pCoverageMask ); // store the pixel in the screen buffer using coverage mask
    else
        Piexl.store( (int*)pColBuffer ); // store the pixel in the screen buffer
}

void TileLineFToARGB( const Vector4f* pPixels , uint32_t* pColBuffer )
{
    f128S Multiplier = 255.0f; // set alpha to 255 for all pixels
    i128S U8Masks    = 0xFF; // masks for converting to ARGB

    for (int x = 0; x < TILE_SIZE; x+=4 , pPixels += 4 , pColBuffer += 4 )
    {
        Vec4ToARGB_Simd<false,false>( Multiplier , U8Masks , pPixels , pColBuffer );
    }
}

template< bool UseCoverage , int Elements , eSimdType Simd >
void TileLineFToARGB_Simd( const Vector4f* pPixels , uint32_t* pColBuffer , optional<Vector4f> Color , uint32_t Coverage )
{
    using simdf = simd<float, Elements, Simd>;
    using simdi = simd<int  , Elements, Simd>;

    simdf Multiplier =  255.0f; // set alpha to 255 for all pixels
    simdi U8Masks    =  0xFF;  // masks for converting to ARGB
    simdi Mask;

    const simdi* pMask = nullptr;

    if constexpr( UseCoverage )
    {
        if( !Coverage )
            return;

        if constexpr( Elements == 8 )
            Mask = simdi{ int(Coverage) } << simdi{ 7 , 6 , 5 , 4 , 3 , 2 , 1 , 0 } ;
        else
            Mask = simdi{ int(Coverage) } << simdi{ 3 , 2 , 1 , 0 } ;

        pMask = &Mask;
    }

    if( Color )
    {
        for (int x = 0; x < TILE_SIZE; x+=Elements , pPixels += Elements , pColBuffer += Elements )
        {
            Vec4ToARGB_Simd<true,UseCoverage>( Multiplier , U8Masks , pPixels , pColBuffer , &*Color , pMask );

            if constexpr( UseCoverage )
                Mask <<= Elements;
        }
    }
    else
    {
        for (int x = 0; x < TILE_SIZE; x+=Elements , pPixels += Elements , pColBuffer += Elements )
        {
            Vec4ToARGB_Simd<false,UseCoverage>( Multiplier , U8Masks , pPixels , pColBuffer , nullptr , pMask );

            if constexpr( UseCoverage )
                Mask <<= Elements;
        }
    }
}

template< bool Partial >
inline void SoftwareRenderer::DrawTileImpl(const CommandRenderTile& _InitTD, DrawStats* stats)
{
    ZoneScopedN( Partial ? "DrawTileImpl" : "DrawTileImplFull" );
    ZoneColor( 0xE0E000 );
    const auto pTileInfo    = _InitTD.TileInfo;

    // local storage for pixels and zbuffer - it seems to be faster than using TileMem directly
    ALIGN_FOR_AVX Vector4f  Pixels       [TILE_SIZE * TILE_SIZE];
    ALIGN_FOR_AVX float     ZBuffer      [TILE_SIZE * TILE_SIZE];
    ALIGN_FOR_AVX uint32_t  TileCoverage [TILE_SIZE] = {};

    // lock tile info to prevent concurrent access
    std::scoped_lock lock( pTileInfo->Lock );

    // copy tile memory to local storage
    memcpy( Pixels  , pTileInfo->TileMem  , sizeof(Pixels) );
    memcpy( ZBuffer , pTileInfo->TileZMem , sizeof(ZBuffer) );

    auto       pDrawCommand = g_ThreadPerTile ? pTileInfo->pRenderTileCmd.load( std::memory_order_relaxed ) : &_InitTD;
    auto       pixelsDrawn  = 0;
    int32_t    CommandIndex = 0;
    const auto TilePosition = pTileInfo->TileScreenPos;
    const int  StartY       = TilePosition.y;
    const int  EndY         = StartY + TILE_SIZE;

    for( ; pDrawCommand ; ++CommandIndex , pDrawCommand = pDrawCommand->pNext.load( std::memory_order_relaxed ) )
    {
    ZoneScopedN("Iteration");
    ZoneColor( CommandIndex>0 ? 0x00E000 : 0x0000E0 );

    const auto pTriangle    = pDrawCommand->Triangle;
    const auto EdgeStrideX  = pTriangle->m_EdgeFunctionRails.GetEdgeFunctionsXStride().ToVector3<int>();
    const auto EdgeStrideY  = pTriangle->m_EdgeFunctionRails.GetEdgeFunctionsYStride().ToVector3<int>();
    const auto _EdgeStart   = pTriangle->m_EdgeFunctionRails.GetStartFor( TilePosition.ToVector2<int64_t>() );
    const auto EdgeStartX   = _EdgeStart.x.ToVector3<int>();
    auto       EdgeStartY   = _EdgeStart.y.ToVector3<int>();
    const auto invABC       = pTriangle->m_InvABC;
    float*     pZBuffer     = ZBuffer;
    Vector4f*  pCurPixel    = Pixels;
    uint32_t*  pTileCoverage= TileCoverage;

    using NumericType = decltype(EdgeStartX.x);

    ALIGN_FOR_AVX Vector4f Pixels[TILE_SIZE * TILE_SIZE];
    Vector4f* pPixels = Pixels;

    ALIGN_FOR_AVX TransformedVertex interpolatedVertex;

    // loop through all pixels in tile square
    for (int y = StartY ; y < EndY; y++ , EdgeStartY += EdgeStrideY )
    {
        auto EdgeFunctions = EdgeStartX + EdgeStartY;

        for (int ix = 0; ix < TILE_SIZE; ix++ , EdgeFunctions += EdgeStrideX )
        {
            //if constexpr( Partial )
            if( !pDrawCommand->IsFullTile )
            {
                if( EdgeFunctionTest( EdgeFunctions ) )
                    continue; // outside triangle
            }

            Vector3f baricentricCoordinates = EdgeFunctions.ToVector3<float>() * invABC;

            pTriangle->m_Interpolator.InterpolateT<MathCPU>( Vector3f( baricentricCoordinates.y , baricentricCoordinates.z , baricentricCoordinates.x ) , interpolatedVertex);

            Vector4f col = Vector4f{ 1,1,1,1 };

            float& z = pZBuffer[ix];
            if (interpolatedVertex.m_ScreenPosition.z < z){
                if (m_ZWrite)
                    z = interpolatedVertex.m_ScreenPosition.z;
            }
            else if (m_ZTest){
                continue;
            }

            Vector4f finalColor = FragmentShader(interpolatedVertex);

            pCurPixel[ix] = finalColor*col;
            pixelsDrawn++;

            pTileCoverage[0] |= 1 << ix;
        }

        pCurPixel += TILE_SIZE;
        pZBuffer  += TILE_SIZE; // move to next row in Z-buffer
        pTileCoverage++;
    }

    if( !g_ThreadPerTile )
        break;
    }

    memcpy( pTileInfo->TileMem  , Pixels  , sizeof(Pixels)  );
    memcpy( pTileInfo->TileZMem , ZBuffer , sizeof(ZBuffer) );

    uint32_t* pTileCoverage = TileCoverage;
    Vector4f* pPixels       = Pixels;

    for (int y = StartY ; y < EndY; y++ )
    {
        auto pCurScreenPixel = m_ScreenBuffer.data() + (y * SCREEN_WIDTH + TilePosition.x);
        for( uint32_t x = 0 , cov = 1 ; x < TILE_SIZE ; ++x , cov <<= 1 )
        {
            if( pTileCoverage[0] & cov )
                pCurScreenPixel[x] = Vector4f::ToARGB( pPixels[x] );
        }
        pTileCoverage++;
        pPixels += TILE_SIZE; // move to next row in Pixels
    }



    if( stats )
        stats->m_FramePixelsDrawn += pixelsDrawn;
}

template< eSimdType Type , bool Partial , int Elements  >
inline void SoftwareRenderer::DrawTileImplSimd(const CommandRenderTile& _InitTD, DrawStats* stats)
{
    using f256t             = fsimd<Elements,Type>;
    using i256t             = isimd<Elements,Type>;
    using Vector3i256t      = Vector3<i256t>;
    using Vector4f256t      = Vector4<f256t>;
    using TransformedVertexT= SimdTransformedVertex<Elements,Type>;

    ZoneScopedN( Partial ? "DrawTileImplSimd" : "DrawTileImplSimdFull" );
    ZoneColor( 0xE0E000 );
    const auto pTileInfo    = _InitTD.TileInfo;

    // local storage for pixels and zbuffer - it seems to be faster than using TileMem directly
    ALIGN_FOR_AVX Vector4f  Pixels       [TILE_SIZE * TILE_SIZE];
    ALIGN_FOR_AVX float     ZBuffer      [TILE_SIZE * TILE_SIZE];
    ALIGN_FOR_AVX uint32_t  TileCoverage [TILE_SIZE] = {};

    // lock tile info to prevent concurrent access
    std::scoped_lock lock( pTileInfo->Lock );

    // copy tile memory to local storage
    memcpy( Pixels  , pTileInfo->TileMem  , sizeof(Pixels) );
    memcpy( ZBuffer , pTileInfo->TileZMem , sizeof(ZBuffer) );

    auto       pDrawCommand = g_ThreadPerTile ? pTileInfo->pRenderTileCmd.load( std::memory_order_relaxed ) : &_InitTD;
    auto       pixelsDrawn  = 0;
    int32_t    CommandIndex = 0;
    const auto TilePosition = pTileInfo->TileScreenPos;
    const int  StartY       = TilePosition.y;
    const int  EndY         = StartY + TILE_SIZE;

    for( ; pDrawCommand ; ++CommandIndex , pDrawCommand = pDrawCommand->pNext.load( std::memory_order_relaxed ) )
    {
    ZoneScopedN("Iteration");
    ZoneColor( CommandIndex>0 ? 0x00E000 : 0x0000E0 );

    constexpr int pack_size = f256t::elements_count;
    const auto pTriangle    = pDrawCommand->Triangle;
    const auto _EdgeStrideX = pTriangle->m_EdgeFunctionRails.GetEdgeFunctionsXStride().ToVector3<int>().Swizzle<1,2,0>();
    const auto _EdgeStrideY = pTriangle->m_EdgeFunctionRails.GetEdgeFunctionsYStride().ToVector3<int>().Swizzle<1,2,0>();
          auto EdgeStrideX  = Vector3i256t{ _EdgeStrideX.x , _EdgeStrideX.y , _EdgeStrideX.z };
    const auto EdgeStrideY  = Vector3i256t{ _EdgeStrideY.x , _EdgeStrideY.y , _EdgeStrideY.z };
    const auto _EdgeStart   = pTriangle->m_EdgeFunctionRails.GetStartFor( TilePosition.ToVector2<int64_t>() );
    const auto _EdgeStartX  = _EdgeStart.x.ToVector3<int>().Swizzle<1,2,0>();
    const auto _EdgeStartY  = _EdgeStart.y.ToVector3<int>().Swizzle<1,2,0>();
          auto EdgeStartX   = Vector3i256t{ _EdgeStartX.x , _EdgeStartX.y , _EdgeStartX.z };
            if constexpr( Elements == 8 )
               EdgeStartX  += EdgeStrideX*i256t{ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 };
            else
                EdgeStartX += EdgeStrideX*i256t{ 0 , 1 , 2 , 3 };
          auto EdgeStartY   = Vector3i256t{ _EdgeStartY.x , _EdgeStartY.y , _EdgeStartY.z };
    const auto invABC       = f256t{ pTriangle->m_InvABC };
    EdgeStrideX *= pack_size;

    TransformedVertexT      interpolatedVertex;
    Vector4f*               pCurPixel       = Pixels;
    uint32_t*               CoverageMask    = TileCoverage;
    float*                  pZBuffer        = ZBuffer;
    i256t                   write_mask;
    constexpr auto          pack_coverage_mask = (uint32_t(1)<<pack_size)-1;

    // loop through all pixels in tile square
    for (int y = StartY ; y < EndY; y++ , EdgeStartY += EdgeStrideY , CoverageMask++ )
    {
        // Calculate edge functions for the first pixel in the current line
        auto EdgeFunctions = EdgeStartX + EdgeStartY;

        uint32_t LineCoverageMask = 0;

        for (int ix = 0; ix < TILE_SIZE
            ; ix+=pack_size
            , EdgeFunctions += EdgeStrideX
            )
        {
            LineCoverageMask <<= pack_size;

            auto baricentricCoordinates = EdgeFunctions.ToVector3<f256t>() * invABC;

            // prepare coverage mask for bits from current pixel pack

            // initialize coverage with 1s - initially we assume that all pixels in current pack will be written
            uint32_t CurrentPackCoverageMask = 0xFFFFFFFF;

            if constexpr( Partial || 1 )
            {
                // compute edge functions for current pixel pack and generate mask with 1s for pixels that are inside triangle
                write_mask = (EdgeFunctions.x | EdgeFunctions.y | EdgeFunctions.z) >= i256t::Zero;

                // convert write_mask to 32-bit mask (8 least significant bits represent coverage for each pixel in the pack)
                CurrentPackCoverageMask = write_mask.to_mask_32();

                // perform edge function test for current pixel pack
                if( !CurrentPackCoverageMask )
                    // all pixels failed edge function test - all of them are outside triangle - we can skip further processing
                    continue;
            }
            else
            {
                // if we are drawing full tile, we assume that initially all pixels are inside triangle
                write_mask = i256t::AllBitsSet;
            }

            // interpolate z value
            pTriangle->m_Interpolator.InterpolateZ( baricentricCoordinates, interpolatedVertex );

            // load z-values for current pixel pack from Z-buffer
            f256t CurrentZValue( pZBuffer + ix , simd_alignment::AVX );

            if( m_ZTest )
            {
                // Perform Z-test on whole pixel pack:
                //      All pixels with z-value less than current z-value will generate 1 in write_mask, rest will generate 0
                write_mask &= ( interpolatedVertex.m_ScreenPosition.z < CurrentZValue ).reinterpret_cast_to<int>();

                // Select pixels that passed the Z-test - bit 1 will select second value, 0 will select first value
                CurrentZValue = CurrentZValue.select( interpolatedVertex.m_ScreenPosition.z , write_mask );

                if( m_ZWrite )
                    // store the current z-value back to the Z-buffer (we can write all pixels at once because we changed only pixels that passed the Z-test)
                    CurrentZValue.store( pZBuffer + ix , simd_alignment::AVX );

                // convert write_mask to 8 least significant bits representing coverage for each pixel in the pack
                CurrentPackCoverageMask &= write_mask.to_mask_32();
                if( !CurrentPackCoverageMask )
                    continue; // zest failed for all pixels - no need to interpolate color
            }
            else if( m_ZWrite )
            {
                // Only z-write is enabled, so we simply store the z-value for all pixels that passed edge function test
                CurrentZValue = CurrentZValue.select( interpolatedVertex.m_ScreenPosition.z , write_mask );

                CurrentZValue.store( pZBuffer + ix , simd_alignment::AVX );
            }

            // Add mask for current pixel pack to the coverage mask
            LineCoverageMask |= (CurrentPackCoverageMask & pack_coverage_mask);

            // Interpolate all other vertex attributes (UV, normal, etc.) for all pixels in the current pack
            pTriangle->m_Interpolator.InterpolateAllButZ( baricentricCoordinates, interpolatedVertex );

            // Call fragment shader to compute final color for all pixels in the current pack
            Vector4f256t finalColor = FragmentShader(interpolatedVertex);
            // store the final color in the local pixel buffer - we will write it to the screen buffer later
            finalColor.store( pCurPixel[ix].Data() , write_mask );

            //Vector4f256t finalColor = Vector4f256t(interpolatedVertex.m_UV * f256A{ 1.0f }, f256A{ 1.0f } , f256A{ 1.0f } );
        }

        pCurPixel += TILE_SIZE;
        pZBuffer  += TILE_SIZE;

        *CoverageMask |= LineCoverageMask; // accumulate coverage mask for the whole tile
    }

        if( !g_ThreadPerTile )
            break;
    }

    optional<Vector4f> White;
    if( m_ColorizeThreads )
    {
        if( auto ThreadId = SimpleThreadPool::GetThreadID() ; ThreadId>=0 && ThreadId < 16 )
        {
            auto col = m_ThreadColors[ThreadId];
            col = (col*0.3f) + 0.7;
            White = col;
        }
    }

    {
        Vector4f* pCurPixel       = Pixels;
        uint32_t* CoverageMask    = TileCoverage;
        for (int y = StartY ; y < EndY; y++)
        {
            TileLineFToARGB_Simd<true,8,eSimdType::AVX>( pCurPixel , m_ScreenBuffer.data() + y* SCREEN_WIDTH + TilePosition.x , White , *CoverageMask );
            pCurPixel += TILE_SIZE;
            CoverageMask++;
        }
        memcpy( pTileInfo->TileMem  , Pixels  , sizeof(Pixels)  );
        memcpy( pTileInfo->TileZMem , ZBuffer , sizeof(ZBuffer) );
    }


    // update max overdraw count
    auto old_max_overdraw = g_max_overdraw.load(std::memory_order_relaxed);
    for(;;)
    {
        CommandIndex = std::max( old_max_overdraw , CommandIndex );
        if( g_max_overdraw.compare_exchange_strong( old_max_overdraw , CommandIndex, std::memory_order_relaxed) )
            break;
    }

    // update pixels draw stats
    if( stats )
        stats->m_FramePixelsDrawn += pixelsDrawn;
}

void SoftwareRenderer::DrawPartialTile(const CommandRenderTile& TD, DrawStats* stats)
{
    if( g_useSimd )
        return DrawTileImplSimd<eSimdType::AVX,true,8>( TD, stats );
    else
        return DrawTileImpl<true>(TD, stats);
}
void SoftwareRenderer::DrawFullTile(const CommandRenderTile& TD, DrawStats* stats)
{
    if( g_useSimd )
        return DrawTileImplSimd<eSimdType::AVX,false,8>( TD, stats );
    else
        return DrawTileImpl<false>(TD, stats);
}
void SoftwareRenderer::DrawTile(const CommandRenderTile& TD, DrawStats* stats)
{
    if( TD.IsFullTile )
        DrawFullTile(TD, stats);
    else
        DrawPartialTile(TD, stats);
}

eTileCoverage ClassifyX( Vector2<int64_t> A , Vector2<int64_t> B , Vector2<int64_t> C , Vector2<int64_t> TilePos , const Vector2<int64_t> Close[3] , const Vector2<int64_t> Far[3] )
{
    //Vector2i TilePos = TileCell * TILE_SIZE * 16;

    int64_t Dists[6] =
    {
        SoftwareRenderer::EdgeFunction( A , B , TilePos + Close[0] ),
        SoftwareRenderer::EdgeFunction( A , B , TilePos + Far  [0] ),
        SoftwareRenderer::EdgeFunction( B , C , TilePos + Close[1] ),
        SoftwareRenderer::EdgeFunction( B , C , TilePos + Far  [1] ),
        SoftwareRenderer::EdgeFunction( C , A , TilePos + Close[2] ),
        SoftwareRenderer::EdgeFunction( C , A , TilePos + Far  [2] ),
    };


    eTileCoverage E0 = ( Dists[0] < 0 ? eTileCoverage(0x1) : eTileCoverage(0x3) )
                     | ( Dists[1] < 0 ? eTileCoverage(0x1) : eTileCoverage(0x5) );
    eTileCoverage E1 = ( Dists[2] < 0 ? eTileCoverage(0x1) : eTileCoverage(0x3) )
                     | ( Dists[3] < 0 ? eTileCoverage(0x1) : eTileCoverage(0x5) );
    eTileCoverage E2 = ( Dists[4] < 0 ? eTileCoverage(0x1) : eTileCoverage(0x3) )
                     | ( Dists[5] < 0 ? eTileCoverage(0x1) : eTileCoverage(0x5) );


    return E0 & E1 & E2;
}

inline void ImGuiAddLine(ImVec2 p1, ImVec2 p2, ImU32 col, float thickness = 1)
{
    p1.y = SCREEN_HEIGHT - p1.y;
    p2.y = SCREEN_HEIGHT - p2.y;

    ImGui::GetBackgroundDrawList()->AddLine(p1, p2, col, thickness);
}

inline void ImGuiAddRectFilled(ImVec2 p_min, ImVec2 p_max, ImU32 col, float rounding = 0.0f, ImDrawFlags flags = 0)
{
    p_min.y = SCREEN_HEIGHT - p_min.y;
    p_max.y = SCREEN_HEIGHT - p_max.y;

    ImGui::GetBackgroundDrawList()->AddRectFilled(p_min, p_max, col, rounding, flags);
}


inline void ImGuiAddRect(ImVec2 p_min, ImVec2 p_max, ImU32 col, float rounding = 0.0f, ImDrawFlags flags = 0, float thickness = 1.0f)
{
    p_min.y = SCREEN_HEIGHT - p_min.y;
    p_max.y = SCREEN_HEIGHT - p_max.y;
    ImGui::GetBackgroundDrawList()->AddRect(p_min, p_max, col, rounding, flags, thickness);
}

inline void ImGuiAddX(ImVec2 p_min, ImVec2 p_max, ImU32 col, float thickness = 1.0f)
{
    ImVec2 P[4] = { ImVec2{ p_min.x , p_min.y }
                  , ImVec2{ p_max.x , p_min.y }
                  , ImVec2{ p_min.x , p_max.y }
                  , ImVec2{ p_max.x , p_max.y } };

    ImGuiAddLine(P[0], P[3], col, thickness);
    ImGuiAddLine(P[1], P[2], col, thickness);
}

void SoftwareRenderer::GenerateTileJobs(const TransformedVertex& VA, const TransformedVertex& VB, const TransformedVertex& VC, const Vector4f& color, DrawStats& stats, const PipelineSharedData* pPipelineSharedData, uint32_t tri_index ,  pmr::vector<const Command*>& outCommmands )
{
    ZoneScoped;
    ZoneColor( 0x00E0E0 );
    // filling algorithm is working that way that we are going through all pixels in rectangle that is created by min and max points
    // and we are checking if pixel is inside triangle by using three lines and checking if pixel is on the same side of each line

    if( m_DrawBBoxes | m_DrawWireframe )
    {
        if (m_DrawWireframe)
            return DrawTriangle(VA, VB, VC, m_WireFrameColor, 0, SCREEN_HEIGHT);
        else
            return DrawTriangleBoundingBox( VA , VB, VC, m_WireFrameColor , 0 , SCREEN_HEIGHT );
    }

    Vector2 A = (VA.m_ScreenPosition.xy()*Precision.Multiplier).ToVector2<int64_t,eRoundMode::Round>();
    Vector2 B = (VB.m_ScreenPosition.xy()*Precision.Multiplier).ToVector2<int64_t,eRoundMode::Round>();
    Vector2 C = (VC.m_ScreenPosition.xy()*Precision.Multiplier).ToVector2<int64_t,eRoundMode::Round>();

    // clockwise order so we check if point is on the right side of line
    const auto ABC = EdgeFunction(A, B, C);

    // if our edge function (signed area x2) is negative, it's a back facing triangle and we can cull it
    bool isBackFacing = ABC <= 0;
    if (isBackFacing)
    {
        stats.m_FrameTriangles++;
        return;
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

    Vector2 min = A.CWiseMin(B).CWiseMin(C).CWiseMax(Vector2<int64_t>(0,0)).CWiseMin(Vector2<int64_t>(SCREEN_WIDTH*Precision.Multiplier, SCREEN_HEIGHT*Precision.Multiplier));
    Vector2 max = A.CWiseMax(B).CWiseMax(C).CWiseMax(Vector2<int64_t>(0,0)).CWiseMin(Vector2<int64_t>(SCREEN_WIDTH*Precision.Multiplier, SCREEN_HEIGHT*Precision.Multiplier));

    VertexInterpolator interpolator(VA, VB, VC);

    const auto StartP = ( min );

    constexpr auto PixelOffset = Vector2<int64_t>{ Precision.Multiplier , Precision.Multiplier }/2;

    TriangleData& Data = *m_TransientAllocator.allocate<TriangleData>( VA , VB , VC , Vector4f{1,1,1,1} , A , B , C , StartP );
    Data.m_InvABC = (1.0f / ABC);
    Data.m_TriIndex = tri_index;
    Data.m_MaxZ = Data.m_MinZ = VA.m_ScreenPosition.z;
    Data.m_MinZ = std::min( Data.m_MinZ , VB.m_ScreenPosition.z );
    Data.m_MaxZ = std::max( Data.m_MaxZ , VB.m_ScreenPosition.z );

    Data.m_MinZ = std::min( Data.m_MinZ , VC.m_ScreenPosition.z );
    Data.m_MaxZ = std::max( Data.m_MaxZ , VC.m_ScreenPosition.z );


    constexpr auto TILE_SIZE_M = TILE_SIZE * Precision.Multiplier;

    Vector2i MinTile;
    Vector2i MaxTile;

    {
        auto Min = min.ToVector2<int>() / Precision.Multiplier;
        auto Max = max.ToVector2<int>() / Precision.Multiplier;
        MinTile  =  Min             / TILE_SIZE;
        MaxTile  =((Max +TILE_SIZE) / TILE_SIZE).CWiseMin( m_LastTile.ToVector2<int>() );
    }

    struct Edge_t
    {
        Vector2<int64_t> A;
        Vector2<int64_t> B;
    };
    Vector2<int64_t> Closest [3];
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
    for( int i=0 ; auto& Edge : Edges )
    {
        auto A = Edge.A;
        auto B = Edge.B;
        auto AB = B - A;
        auto Shift = (A + AB/2) + (-AB.Rotated90())*10;

        uint8_t ClosestI   = 0;
        uint8_t FarthestI  = 0;
        int64_t ClosestD   = std::numeric_limits<int>::max();
        int64_t FarthestD  = std::numeric_limits<int>::min();

        for( uint8_t i = 0 ; i < 4 ; ++i )
        {
            auto P = Shift + Corners[i];

            int dist = SoftwareRenderer::EdgeFunction( B , A ,  P );
            if( dist < ClosestD )
            {
                ClosestD = dist;
                ClosestI = i;
            }
            if( dist > FarthestD )
            {
                FarthestD = dist;
                FarthestI = i;
            }
        }
        Closest [i  ] = Corners[ ClosestI  ] * TILE_SIZE_M;
        Farthest[i++] = Corners[ FarthestI ] * TILE_SIZE_M;
    }
    }

    if( g_showTriangleBoundry )
    {
        ImGuiAddLine( A , B , ImColor(255,0,64) , 3 );
        ImGuiAddLine( B , C , ImColor(0,255,64) , 3 );
        ImGuiAddLine( C , A , ImColor(64,0,255) , 3 );
    }

    if( g_showCornersClassify )
    {
        Vector2 TilePos = Vector2<int64_t>(4,m_TilesGridSize.y-4)*TILE_SIZE_M;

        ImGuiAddRect(TilePos, TilePos + TILE_SIZE, ImColor(255, 0, 0, 128) );

        ImGuiAddRect(TilePos+Closest[0], TilePos+Closest[0] + 8, ImColor(255, 0, 0, 255) , 0 , 0 , 3 );
        ImGuiAddRect(TilePos+Closest[1], TilePos+Closest[1] + 8, ImColor(0, 255, 0, 255) , 0 , 0 , 3 );
        ImGuiAddRect(TilePos+Closest[2], TilePos+Closest[2] + 8, ImColor(0, 0, 255, 255) , 0 , 0 , 3 );

        ImGuiAddX(TilePos+Farthest[0], TilePos+Farthest[0] + 8, ImColor(255, 0, 0, 255) , 2 );
        ImGuiAddX(TilePos+Farthest[1], TilePos+Farthest[1] + 8, ImColor(0, 255, 0, 255) , 2 );
        ImGuiAddX(TilePos+Farthest[2], TilePos+Farthest[2] + 8, ImColor(0, 0, 255, 255) , 2 );
    }

    CommandBuffer* pCommandBuffer = pPipelineSharedData ? pPipelineSharedData->m_pRenderTilesCmdBuffer : m_pCommandBuffer;

    transient_allocator alloc{ m_TransientMemoryResource };

    auto AddCommand = [&Data,&alloc,&outCommmands,tri_index]( const Vector2<int64_t>& TilePos , const TileInfo* pTileInfo , auto FullTile )
    {
        constexpr bool IsFullTile = FullTile();

        ZoneScopedN( IsFullTile ? "gen command full" : "gen command partial" );
        auto ID = pTileInfo->DrawCount++;

        CommandRenderTile& TD = *alloc.allocate<CommandRenderTile>();
        auto LogicPos  = TilePos.ToVector2<int>();
        TD.IsFullTile= IsFullTile;
        TD.TileDrawID= ID;
        TD.Triangle  = &Data;
        TD.TileInfo  = pTileInfo;

        CommandRenderTile* pCurTop = nullptr;

        if( g_ThreadPerTile )
        {
            pCurTop = pTileInfo->pRenderTileCmd.load( std::memory_order_relaxed );

            for(;;)
            {
                TD.pNext.store( pCurTop , std::memory_order_relaxed );
                if( !pTileInfo->pRenderTileCmd.compare_exchange_weak( pCurTop , &TD , std::memory_order_acq_rel ) )
                    continue;

                break;
            }
        }

        if( !pCurTop )
        {
            ZoneColor( 0x0000FF );
            outCommmands.push_back( &TD );
        }
        else
        {
            ZoneColor( 0x00FF00 );
        }

        //pCommandBuffer->PushCommand<CommandRenderTile>( TD );

        if( g_showTilestype )
        {
            if constexpr( IsFullTile )
                ImGuiAddRectFilled(TilePos, TilePos + TILE_SIZE, ImColor(64, 255, 0, 64) );
            else
                ImGuiAddRectFilled(TilePos, TilePos + TILE_SIZE, ImColor(0, 64, 255, 64) );
        }
    };

    {
        ZoneScopedN("Classify Tiles");
        for( int y = MinTile.y ; y <= MaxTile.y ; ++y )
        {
            for( int x = MinTile.x ; x <= MaxTile.x ; ++x )
            {
                const auto TileIndex = Vector2si( x , y );
                auto* pTileInfo = GetTileInfo( TileIndex );
                Vector2 TilePos = (TileIndex*TILE_SIZE_M).ToVector2<int64_t>();

                auto Classified = ClassifyX( A , B , C , TilePos , Closest , Farthest );

                if( g_showTilesBoundry )
                    ImGuiAddRect( TilePos , TilePos + TILE_SIZE, ImColor(32, 32, 32, 255) );

                if( Classified == eTileCoverage::Inside )
                {
                    AddCommand( TilePos , pTileInfo , std::true_type{} );
                }
                else if( Classified == eTileCoverage::Partial )
                {
                    AddCommand( TilePos , pTileInfo , std::false_type{} );
                }
            }
        }
    }
}

void SoftwareRenderer::BeginFrame()
{
    m_FramePixels = 0;
    m_FrameTriangles = 0;
    m_FramePixelsDrawn = 0;
    m_FrameTrianglesDrawn = 0;
    m_FrameDrawTimeMainUS = 0;
    m_FrameDrawTimeThreadUS = 0;
    m_FillrateKP = 0;

    m_FrameRasterTimeUS = 0;
    m_FrameTransformTimeUS = 0;
    g_max_overdraw = 0;

    for( int i=0 ; m_TilesGridSize.x * m_TilesGridSize.y > i ; ++i )
    {
        m_TilesGrid[i].DrawCount = 0;
        m_TilesGrid[i].pRenderTileCmd = nullptr;
    }
}

void SoftwareRenderer::EndFrame()
{
    m_TransientMemoryResource.reset();

    const int ThreadsDivide = ( m_ThreadPool.GetThreadCount() ? m_ThreadPool.GetThreadCount() : 1 );

    m_DrawStats.m_FramePixels         = m_FramePixels;
    m_DrawStats.m_FramePixelsDrawn    = m_FramePixelsDrawn;
    m_DrawStats.m_FrameTriangles      = m_FrameTriangles;
    m_DrawStats.m_FrameTrianglesDrawn = m_FrameTrianglesDrawn;
    m_DrawStats.m_DrawTimeUS          = m_FrameDrawTimeMainUS;
    m_DrawStats.m_DrawTimePerThreadUS = m_FrameDrawTimeThreadUS / ThreadsDivide;
    m_DrawStats.m_FillrateKP          = m_FillrateKP;
    m_DrawStats.m_RasterTimeUS        = m_FrameRasterTimeUS;
    m_DrawStats.m_RasterTimePerThreadUS=m_FrameRasterTimeUS / ThreadsDivide;
    m_DrawStats.m_TransformTimeUS     = m_FrameTransformTimeUS / ThreadsDivide;

    if( g_showTilesGrid )
    {
        for( int i=0 ; i<SCREEN_WIDTH; i+=TILE_SIZE )
            ImGui::GetBackgroundDrawList()->AddLine( ImVec2( i,0 ), ImVec2(i ,SCREEN_HEIGHT) , ImColor(255,128,255,128) );

        for( int i=0 ; i<SCREEN_HEIGHT; i+=TILE_SIZE )
            ImGui::GetBackgroundDrawList()->AddLine( ImVec2( 0,i ), ImVec2(SCREEN_WIDTH ,i) , ImColor(255,128,255,128) );
    }

    m_TransientMemoryResource.reset();
    m_pCommandBuffer = AllocTransientCommandBuffer();
}

void SoftwareRenderer::Render(const vector<Vertex>& vertices)
{
    m_pSelectedMath = m_MathArray[std::clamp(m_MathIndex,0,2)];

    ZoneScoped;
    const auto startTime = std::chrono::high_resolution_clock::now();

    auto pConfig = m_TransientAllocator.allocate<DrawConfig>();
    pConfig->m_Color = Vector4f{1,1,1,1};

    m_pCommandBuffer->AddSyncBarrier( "Pre VertexAssemply Sync" , m_ThreadsCount );
    m_pCommandBuffer->PushCommand<CommandVertexAssemply>( vertices , *pConfig );
    m_pCommandBuffer->AddSyncBarrier( "Post VertexAssemply Sync" , m_ThreadsCount );

    FrameMarkNamed( "Tasks launched" );
    m_TileThreadPool.LaunchTasks( { m_ThreadsCount , [this](){ RendererTaskWorker(); } } );

    for( int i=0 ; m_TilesGridSize.x * m_TilesGridSize.y > i ; ++i )
    {
        m_TilesGrid[i].DrawCount = 0;
        m_TilesGrid[i].pRenderTileCmd = nullptr;
    }

    const auto timeUS = std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::high_resolution_clock::now() - startTime).count();
    m_FrameDrawTimeMainUS += timeUS;

    m_pCommandBuffer = AllocTransientCommandBuffer();
}

void SoftwareRenderer::RendererTaskWorker()
{
    ZoneScoped;
    const auto pCommandBuffer = m_pCommandBuffer;
    pmr::vector<uint64_t> TmpBuffer{ &m_TransientMemoryResource };
    TmpBuffer.reserve( 2048 );
    for(;;)
    {
        auto pCommand = pCommandBuffer->GetNextCommand();
        if (!pCommand)
            return ExecuteExitCommand();

        switch( pCommand->GetCommandID() )
        {
        case eCommandID::ClearBuffers:           ClearBuffers            (*pCommand->static_cast_to<CommandClear>()                  );              continue;
        case eCommandID::Fill32BitBuffer:        Fill32BitBuffer         (*pCommand->static_cast_to<CommandFill32BitBuffer>()        );              continue;
        case eCommandID::SyncBarier:             WaitForSync             (*pCommand->static_cast_to<CommandSyncBarier>()             );              continue;
        case eCommandID::VertexAssemply:         VertexAssemply          (*pCommand->static_cast_to<CommandVertexAssemply>()         );              continue;
        case eCommandID::VertexTransformAndClip: VertexTransformAndClip  (*pCommand->static_cast_to<CommandVertexTransformAndClip>() );              continue;
        case eCommandID::ProcessTriangles:       ProcessTriangles        (*pCommand->static_cast_to<CommandProcessTriangles>()       );              continue;
        case eCommandID::RenderTile:             DrawTile                (*pCommand->static_cast_to<CommandRenderTile>()             ,&m_DrawStats); continue;
            ;
        default:
            assert( false && "Unknown command" );
            return;
        }
    }
}

void SoftwareRenderer::ClearBuffers(const CommandClear& cmd)
{
    ZoneScoped;
    if( !cmd.m_ClearColor && !cmd.m_ZValue )
        return;

    auto BUFFER_SUB_SIZE = m_ScreenBuffer.size()/8;//TILE_SIZE*TILE_SIZE*32;

    auto pWorkCmdBuffer = AllocTransientCommandBuffer();

    auto pSync = m_TransientAllocator.allocate<SyncBarrier>();
    pSync->m_Barrier.emplace( m_ThreadsCount );

    if( cmd.m_ClearColor )
    {
        span<uint32_t> BufferSpan(m_ScreenBuffer);

        for( int i = BUFFER_SUB_SIZE; i <BufferSpan.size(); i+=BUFFER_SUB_SIZE)
        {
            if( BufferSpan.size() < BUFFER_SUB_SIZE )
                break;

            auto SubSpan = BufferSpan.subspan(0, BUFFER_SUB_SIZE);
            BufferSpan = BufferSpan.subspan(SubSpan.size());

            pWorkCmdBuffer->PushCommand<CommandFill32BitBuffer>( SubSpan, *cmd.m_ClearColor);
        }

        if( !BufferSpan.empty() )
            pWorkCmdBuffer->PushCommand<CommandFill32BitBuffer>( BufferSpan, *cmd.m_ClearColor);
    }

    if( cmd.m_ZValue )
    {
        span<float> BufferSpan(m_ZBuffer);

        for( int i = BUFFER_SUB_SIZE; i <BufferSpan.size(); i+=BUFFER_SUB_SIZE)
        {
            if( BufferSpan.size() < BUFFER_SUB_SIZE )
                break;

            auto SubSpan = BufferSpan.subspan(0, BUFFER_SUB_SIZE);
            BufferSpan = BufferSpan.subspan(SubSpan.size());

            pWorkCmdBuffer->PushCommand<CommandFill32BitBuffer>( SubSpan, 1.0f );//*cmd.m_ZValue);
        }

        if( !BufferSpan.empty() )
            pWorkCmdBuffer->PushCommand<CommandFill32BitBuffer>( BufferSpan,1.0f );// *cmd.m_ZValue);
    }

    //pWorkCmdBuffer->AddSyncPoint( *pSync , m_ThreadsCount );

    m_pCommandBuffer->PushCommandBuffer( *pWorkCmdBuffer );
}

void SoftwareRenderer::Fill32BitBuffer( const CommandFill32BitBuffer& cmd )
{
    ZoneScoped;
    auto pBuffer = cmd.m_pBuffer;
    if( !pBuffer )
        return;

    if( cmd.m_Value.m_U8Array[0] == cmd.m_Value.m_U8Array[1] && cmd.m_Value.m_U8Array[1] == cmd.m_Value.m_U8Array[2] && cmd.m_Value.m_U8Array[2] && cmd.m_Value.m_U8Array[3] )
        memset( cmd.m_pBuffer , cmd.m_Value.m_U8Array[0] , cmd.m_ElementsCount * sizeof(uint32_t) );
    else
        std::fill_n( reinterpret_cast<uint32_t*>( cmd.m_pBuffer ) , cmd.m_ElementsCount , cmd.m_Value.m_UValue );
}

void SoftwareRenderer::VertexAssemply( const CommandVertexAssemply& cmd )
{
    ZoneScoped;
    auto vertices = cmd.m_Vertices;
    if( vertices.empty() )
        return;

    auto VerticesCount  = vertices.size();

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

        size_t i=0;
        for(  ; i < VerticesCount ; i+=TRIANGLES_PER_COMMAND*3 )
        {
            if( vertices.size() <  + TRIANGLES_PER_COMMAND*3 )
                break;

            auto SubSpan = vertices.subspan( 0 , TRIANGLES_PER_COMMAND*3 );
            vertices = vertices.subspan(SubSpan.size());

            pWorkCmdBuffer->PushCommand<CommandVertexTransformAndClip>( SubSpan , *pData , i/3 );
        }

        if( !vertices.empty() )
            pWorkCmdBuffer->PushCommand<CommandVertexTransformAndClip>( vertices , *pData , i/3 );
    }

    pVertexTransformAndClipSync->name = "VertexAssemply end";
    pWorkCmdBuffer->AddSyncPoint( *pVertexTransformAndClipSync , m_ThreadsCount );

    pVertexTransformAndClipSync->m_Barrier.emplace( m_ThreadsCount , triviall_function_ref{}.Assign( m_TransientMemoryResource , [=]
    {
        m_pCommandBuffer->PushCommandBuffer( *pData->m_pProcessTrianglesCmdBuffer );
        pData->m_pProcessTrianglesCmdBufferSync->name = "transform triangles end";
        m_pCommandBuffer->AddSyncPoint( *pData->m_pProcessTrianglesCmdBufferSync , m_ThreadsCount );
    }) );

    pData->m_pProcessTrianglesCmdBufferSync->m_Barrier.emplace( m_ThreadsCount , triviall_function_ref{}.Assign( m_TransientMemoryResource , [=]()
    {
        m_pCommandBuffer->PushCommandBuffer( *pData->m_pRenderTilesCmdBuffer );

        pData->m_pRenderTilesCmdBufferSync->name = "render triles end";
        m_pCommandBuffer->AddSyncPoint( *pData->m_pRenderTilesCmdBufferSync , m_ThreadsCount );
        m_pCommandBuffer->Finish();
    }));

    pData->m_pRenderTilesCmdBufferSync->m_Barrier.emplace( m_ThreadsCount );

    m_pCommandBuffer->PushCommandBuffer( *pWorkCmdBuffer );
}

void SoftwareRenderer::ExecuteExitCommand()
{
    ZoneScoped;
}

void SoftwareRenderer::VertexTransformAndClip( const CommandVertexTransformAndClip& cmd )
{
    ZoneScoped;
    ZoneColor( 0xE000E0 );
    auto vertices = cmd.m_Input;
    if( vertices.empty() )
        return;

    span<TransformedVertex> transformedVertices;

    {
        ZoneScopedN( "Clip" );
        ZoneColor( 0xE000E0 );
        vertices = ClipTriangles(cmd.m_pPipelineSharedData->m_NearFrustumPlane, 0.001f, vertices);
    }

    {
        ZoneScopedN( "Transform" );
        ZoneColor( 0xE000E0 );
        transformedVertices = m_TransientAllocator.allocate_array<TransformedVertex>(vertices.size());

        const auto startTime = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < vertices.size(); ++i)
            transformedVertices[i].ProjToScreen(vertices[i], m_ModelMatrix, m_MVPMatrix);

        m_FrameTransformTimeUS += std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::high_resolution_clock::now() - startTime).count();
    }

    //cmd.m_pPipelineSharedData->m_pProcessTrianglesCmdBuffer->PushCommandSync<CommandProcessTriangles>( cmd.m_pPipelineSharedData->m_pProcessTrianglesCmdBufferSync , transformedVertices , *cmd.m_pPipelineSharedData );
    cmd.m_pPipelineSharedData->m_pProcessTrianglesCmdBuffer->PushCommand<CommandProcessTriangles>( transformedVertices , *cmd.m_pPipelineSharedData , cmd.m_StartTriIndex );
}

void SoftwareRenderer::ProcessTriangles( const CommandProcessTriangles& cmd )
{
    ZoneScoped;
    ZoneColor( 0x00E0E0 );
    auto vertices = cmd.m_Vertices;
    if( vertices.empty() )
        return;

    std::array<uint8_t,(1024+16)*sizeof(void*)> Buffer;
    transient_memory_resource transient;
    pmr::monotonic_buffer_resource BufferedRes{ Buffer.data() , Buffer.size() , &transient };
    pmr::vector<const Command*> Commands{ &BufferedRes };
    Commands.reserve( 1024 );

    int tri = cmd.m_StartTriIndex;
    for (int i = 2; i < vertices.size(); i+=3, ++tri)
        GenerateTileJobs(vertices[i - 2], vertices[i - 1], vertices[i], cmd.m_pPipelineSharedData->m_pDrawConfig->m_Color , m_DrawStats,cmd.m_pPipelineSharedData , tri , Commands );

    auto pCmdBuf = CommandBuffer::CreateCommandBuffer( m_TransientMemoryResource , Commands , true );

    cmd.m_pPipelineSharedData->m_pRenderTilesCmdBuffer->PushCommandBuffer( *pCmdBuf );
}

void SoftwareRenderer::RenderDepthBuffer()
{
    ZoneScoped;
    for( int i= 0 ; i < m_ScreenBuffer.size() ; ++i )

    {
        uint32_t Col = std::clamp( int(255 * m_ZBuffer[i] ) , 0 , 255 );
        m_ScreenBuffer[i] = 0xFF000000 | (Col<<16) | (Col<<8) | (Col);
    }
}

void SoftwareRenderer::WaitForSync(const CommandSyncBarier& cmd)
{
    ZoneScoped;
    ZoneColor( 0xFF0000 );
    if( cmd.pAwaitSync )
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
        ZoneScopedN( "Transform" );
        const auto startTime = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < vertices.size(); ++i)
            transformedVertices[i].ProjToScreen(vertices[i], m_ModelMatrix, m_MVPMatrix);

        m_FrameTransformTimeUS += std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::high_resolution_clock::now() - startTime).count();
    }

    if (m_DrawWireframe || m_DrawBBoxes)
    {
        ZoneScopedN("Debug");
        const Vector4f color = m_ColorizeThreads ? m_ThreadColors[threadID] : m_WireFrameColor;

        for (int i = 0; i < vertices.size(); i += TRIANGLE_VERT_COUNT)
        {
            if (m_DrawWireframe)
                DrawTriangle(transformedVertices[i+0], transformedVertices[i+1], transformedVertices[i+2], color, minY, maxY);

            if (m_DrawBBoxes)
                DrawTriangleBoundingBox(transformedVertices[i+0], transformedVertices[i+1], transformedVertices[i+2], color, minY, maxY);
        }
    }
    else
    {
        ZoneScopedN("Draw");
        const auto startTime = std::chrono::high_resolution_clock::now();

        const Vector4f color = m_ColorizeThreads ? m_ThreadColors[threadID] : Vector4f(1.0f, 1.0f, 1.0f, 1.0f);


        DrawFunctionConfig c;
        c.ZTest  = m_ZTest;
        c.ZWrite = m_ZWrite;
        const auto Index = c.ToIndex();


        DrawFilledTriangles<MathSSE,DrawFunctionConfig{}>(transformedVertices.data(), transformedVertices.size(), color, minY, maxY, drawStats);

        m_FrameRasterTimeUS += std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::high_resolution_clock::now() - startTime).count();
    }

    const auto timeUS = std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::high_resolution_clock::now() - startTime).count();

    m_FrameTriangles        += drawStats.m_FrameTriangles;
    m_FrameTrianglesDrawn   += drawStats.m_FrameTrianglesDrawn;
    m_FramePixels           += drawStats.m_FramePixels;
    m_FramePixelsDrawn      += drawStats.m_FramePixelsDrawn;
    m_FillrateKP            += drawStats.m_FramePixelsDrawn * ( 1000.0f / timeUS );
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
    m_ScreenBuffer[y * SCREEN_WIDTH + x] = color;
}

inline void SoftwareRenderer::PutPixel(int x, int y, uint32_t color)
{
    if (x >= SCREEN_WIDTH || x <= 0 || y >= SCREEN_HEIGHT || y <= 0) {
        return;
    }
    m_ScreenBuffer[y * SCREEN_WIDTH + x] = color;
}

template< typename MathT , DrawFunctionConfig Config >
void SoftwareRenderer::DrawFilledTriangles(const TransformedVertex* pVerts, size_t Count, const Vector4f& color, int minY, int maxY, DrawStats& stats)
{
    switch( m_DrawTriVersion )
    {
    case eDrawTriVersion::DrawTriBaseline:
        for (size_t i = 0; i < Count; i += TRIANGLE_VERT_COUNT)
        {
            DrawFilledTriangleBaseline<MathT,Config>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
            //DrawFilledTriangle<MathT>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
        }
        return;

    case eDrawTriVersion::DrawTriv2:
        for (size_t i = 0; i < Count; i += TRIANGLE_VERT_COUNT)
        {
            DrawFilledTriangle_v2<MathT,Config>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
            //DrawFilledTriangle<MathT>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
        }
        return;

    case eDrawTriVersion::DrawTriv3:
        for (size_t i = 0; i < Count; i += TRIANGLE_VERT_COUNT)
        {
            DrawFilledTriangle_v3<MathT,Config>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
            //DrawFilledTriangle<MathT>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
        }
        return;


    default:
        for (size_t i = 0; i < Count; i += TRIANGLE_VERT_COUNT)
        {
            DrawFilledTriangle<MathT,Config>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
            //DrawFilledTriangle<MathT>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
        }
        return;
    }

}

template< typename , DrawFunctionConfig >
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
    Vector2i min = A.CWiseMin(B).CWiseMin(C).CWiseMin(Vector2f(SCREEN_WIDTH-1, maxY)).CWiseMax(Vector2f(0, minY)).ToVector2i();
    Vector2i max = A.CWiseMax(B).CWiseMax(C).CWiseMin(Vector2f(SCREEN_WIDTH-1, maxY)).CWiseMax(Vector2f(0, minY)).ToVector2i();

    const float invABC = 1.0f / ABC;

    VertexInterpolator interpolator(VA, VB, VC);
    TransformedVertex interpolatedVertex;

    int pixelsDrawn = 0;

    // loop through all pixels in rectangle
    for (int y = min.y; y <= max.y; ++y)
    {
        for (int x = min.x; x <= max.x; ++x)
        {
            const Vector2f P(x+0.5f, y+0.5f);
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
            Vector3f baricentricCoordinates = Vector3f( BCP, CAP , ABP) * invABC;
            interpolator.InterpolateZ(baricentricCoordinates, interpolatedVertex);

            float& z = m_ZBuffer[y * SCREEN_WIDTH + x];
            if (interpolatedVertex.m_ScreenPosition.z < z) {
                if (m_ZWrite)
                    z = interpolatedVertex.m_ScreenPosition.z;
            }
            else if (m_ZTest){
                continue;
            }


            interpolator.InterpolateAllButZ(baricentricCoordinates, interpolatedVertex);
            interpolatedVertex.m_Color = interpolatedVertex.m_Color * color;
            PutPixelUnsafe(x, y, FragmentShader(interpolatedVertex).ToARGB());
            pixelsDrawn++;
        }
    }

    stats.FinishDrawCallStats(min,max,pixelsDrawn);
}

template< typename MathT , DrawFunctionConfig >
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
    Vector2i min = A.CWiseMin(B).CWiseMin(C).CWiseMin(Vector2f(SCREEN_WIDTH-1, maxY)).CWiseMax(Vector2f(0, minY)).ToVector2i();
    Vector2i max = A.CWiseMax(B).CWiseMax(C).CWiseMin(Vector2f(SCREEN_WIDTH-1, maxY)).CWiseMax(Vector2f(0, minY)).ToVector2i();

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
            const Vector2f P(x+0.5f, y+0.5f);
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
            Vector3f baricentricCoordinates = Vector3f( BCP, CAP , ABP) * invABC;
            interpolator.Interpolate(mathInstance,baricentricCoordinates, interpolatedVertex);

            float& z = m_ZBuffer[y * SCREEN_WIDTH + x];
            if (interpolatedVertex.m_ScreenPosition.z < z) {
                if (m_ZWrite)
                    z = interpolatedVertex.m_ScreenPosition.z;
            }
            else if (m_ZTest){
                continue;
            }

            interpolatedVertex.m_Color = interpolatedVertex.m_Color * color;
            PutPixelUnsafe(x, y, FragmentShader(interpolatedVertex).ToARGB());
            pixelsDrawn++;
        }
    }

    stats.FinishDrawCallStats(min,max,pixelsDrawn);
}

struct ALIGN_FOR_AVX EdgeFunctionHelper1
{
    inline EdgeFunctionHelper1( Vector2f A , Vector2f B , Vector2f C )
        : PrecalculatedA{
          - (B.y - A.y) ,
            (B.x - A.x) ,
          - (C.y - B.y) ,
            (C.x - B.x) ,
          - (A.y - C.y) ,
            (A.x - C.x) ,
            0,
            0,
        }
        ,
        PrecalculatedB{
            - PrecalculatedA[0] * A.x ,
            - PrecalculatedA[1] * A.y ,
            - PrecalculatedA[2] * B.x ,
            - PrecalculatedA[3] * B.y ,
            - PrecalculatedA[4] * C.x ,
            - PrecalculatedA[5] * C.y ,
              0,
        }
    {}

    float PrecalculatedA[8];
    float PrecalculatedB[8];
};

struct ALIGN_FOR_AVX EdgeFunctionResult
{
    float ABP = 0;
    float BCP = 0;
    float CAP = 0;
    int   SKIP= 0;
};

template< typename MathT , DrawFunctionConfig >
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
    Vector2i min = A.CWiseMin(B).CWiseMin(C).CWiseMin(Vector2f(SCREEN_WIDTH-1, maxY)).CWiseMax(Vector2f(0, minY)).ToVector2i();
    Vector2i max = A.CWiseMax(B).CWiseMax(C).CWiseMin(Vector2f(SCREEN_WIDTH-1, maxY)).CWiseMax(Vector2f(0, minY)).ToVector2i();

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
            const Vector2f P(x+0.5f, y+0.5f);

            mathInstance.EdgeFunction3x( P , edgeHelper.PrecalculatedA , edgeHelper.PrecalculatedB , &(edgeResult.ABP) );

            if (edgeResult.ABP < 0 || edgeResult.BCP < 0 || edgeResult.CAP < 0)
                continue;

            Vector3f baricentricCoordinates = Vector3f( edgeResult.BCP, edgeResult.CAP , edgeResult.ABP) * invABC;
            interpolator.Interpolate(mathInstance,baricentricCoordinates, interpolatedVertex);

            float& z = m_ZBuffer[y * SCREEN_WIDTH + x];
            if (interpolatedVertex.m_ScreenPosition.z < z) {
                if (m_ZWrite)
                    z = interpolatedVertex.m_ScreenPosition.z;
            }
            else if (m_ZTest){
                continue;
            }

            interpolatedVertex.m_Color = interpolatedVertex.m_Color * color;
            PutPixelUnsafe(x, y, FragmentShader(interpolatedVertex).ToARGB());
            pixelsDrawn++;
        }
    }

    stats.FinishDrawCallStats(min,max,pixelsDrawn);
}

template< typename MathT , DrawFunctionConfig Config >
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
    min = min.CWiseMin(Vector2<int64_t>(SCREEN_WIDTH-1, maxY)).CWiseMax(Vector2<int64_t>(0, minY));
    max = max.CWiseMin(Vector2<int64_t>(SCREEN_WIDTH-1, maxY)).CWiseMax(Vector2<int64_t>(0, minY));

    const float invABC = 1.0f / ABC;

    VertexInterpolator interpolator(VA, VB, VC);
    TransformedVertex interpolatedVertex;

    int pixelsDrawn = 0;

    Vector4f baricentricCoordinates;

    const auto StartP = ( min );


          auto CAP_Stride = Vector2( C.y - A.y , A.x - C.x );
          auto BCP_Stride = Vector2( B.y - C.y , C.x - B.x );
          auto ABP_Stride = Vector2( A.y - B.y , B.x - A.x );

    const auto CAP_Start  = CAP_Stride * (StartP - Vector2( C.x , C.y ) );
    const auto BCP_Start  = BCP_Stride * (StartP - Vector2( B.x , B.y ) );
    const auto ABP_Start  = ABP_Stride * (StartP - Vector2( A.x , A.y ) );


    auto ABP_Y = ABP_Start.y;
    auto BCP_Y = BCP_Start.y;
    auto CAP_Y = CAP_Start.y;

    // loop through all pixels in rectangle
    for (auto ScrrenY = min.y; ScrrenY <= max.y; ScrrenY++, ABP_Y+= ABP_Stride.y, BCP_Y+=BCP_Stride.y, CAP_Y+=CAP_Stride.y)
    {
        int linearpos = ScrrenY * SCREEN_WIDTH + min.x;
        uint32_t LineIndex = 0;

        auto ABP_X = ABP_Start.x;
        auto BCP_X = BCP_Start.x;
        auto CAP_X = CAP_Start.x;

        for (auto ScrrenX = min.x; ScrrenX <= max.x; ScrrenX++, ++linearpos, ABP_X += ABP_Stride.x, BCP_X += BCP_Stride.x, CAP_X += CAP_Stride.x)
        {
            auto ABP = (ABP_X + ABP_Y);
            auto BCP = (BCP_X + BCP_Y);
            auto CAP = (CAP_X + CAP_Y);

            if ((ABP | BCP |CAP) < 0)
                continue;

            // if pixel is inside triangle, draw it

            // dividing edge function values by ABC will give us barycentric coordinates - how much each vertex contributes to final color in point P
            Vector3f baricentricCoordinates = Vector3f( ABP , BCP , CAP ) * invABC;

            interpolator.InterpolateT<MathT>( Vector3f( baricentricCoordinates.y , baricentricCoordinates.z , baricentricCoordinates.x ) , interpolatedVertex);

            float& z = m_ZBuffer[linearpos];
            if (interpolatedVertex.m_ScreenPosition.z < z) {
                if (m_ZWrite)
                    z = interpolatedVertex.m_ScreenPosition.z;
            }
            else if (m_ZTest){
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
    Vector2f min = A.m_ScreenPosition.CWiseMin( B.m_ScreenPosition ).CWiseMin( C.m_ScreenPosition ).xy();
    Vector2f max = A.m_ScreenPosition.CWiseMax( B.m_ScreenPosition ).CWiseMax( C.m_ScreenPosition ).xy();

    if( max.y < minY ||
        min.y > maxY )
        return;

    // clamp min and max points to screen size so we don't calculate points that we don't see
    min = min.CWiseMin(Vector2f(SCREEN_WIDTH-1, maxY)).CWiseMax(Vector2f(0, minY));
    max = max.CWiseMin(Vector2f(SCREEN_WIDTH-1, maxY)).CWiseMax(Vector2f(0, minY));

    DrawLine(min , Vector2f{max.x,min.y}, color, minY, maxY);
    DrawLine(Vector2f{max.x,min.y} , max, color, minY, maxY);
    DrawLine(max , Vector2f{min.x,max.y}, color, minY, maxY);
    DrawLine(Vector2f{min.x,max.y} , min, color, minY, maxY);
}

void SoftwareRenderer::DrawLine(const TransformedVertex& VA, const TransformedVertex& VB, const Vector4f& color, int minY, int maxY)
{
    Vector2f A = VA.m_ScreenPosition.xy();
    Vector2f B = VB.m_ScreenPosition.xy();

    return DrawLine(A, B, color, minY,  maxY);
}

void SoftwareRenderer::DrawLine(Vector2f A, Vector2f B, const Vector4f& color, int minY, int maxY)
{
    // Clip whole line against screen bounds
    if ((A.x < 0 && B.x < 0) ||
        (A.y < minY && B.y < minY) ||
        (A.x >= SCREEN_WIDTH  && B.x >= SCREEN_WIDTH) ||
        (A.y > maxY && B.y > maxY) )
        return;

    // Handle case when start end end point are on the same pixel
    if (int(A.x) == int(B.x) && int(A.y) == int(B.y))
    {
        PutPixel( int(A.x) , int(A.y) , Vector4f::ToARGB(color) );
        return;
    }

    Vector2f dir = B - A;

    // Clip point A to minimum Y
    if (A.y < minY)
    {
        float t = ( minY - A.y ) / dir.y;
        A = A + dir * t;
    }
    // Clip point A to maximum Y
    else if (A.y > maxY)
    {
        float t = ( maxY - A.y ) / dir.y;
        A = A + dir * t;
    }

    // Clip point B to minimum Y
    if (B.y > maxY)
    {
        float t = ( maxY - A.y ) / dir.y;
        B = A + dir * t;
    }
    // Clip point B to maximum Y
    else if (B.y < minY)
    {
        float t = ( minY - A.y ) / dir.y;
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

        if (startX > endX) {
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

        if (startY > endY) {
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
    m_CameraPosition    = Vector3f(inversedViewMatrix[12], inversedViewMatrix[13], inversedViewMatrix[14]);
    m_CameraPositionSimd= Vector3f256A{ m_CameraPosition.x , m_CameraPosition.y , m_CameraPosition.z };
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
    m_DiffuseColor      = diffuseColor;
    m_DiffuseColorSimd  = Vector3f256A{ diffuseColor.x , diffuseColor.y , diffuseColor.z };
}

void SoftwareRenderer::SetAmbientColor(const Vector3f& ambientColor)
{
    m_AmbientColor      = ambientColor;
    m_AmbientColorSimd  = Vector3f256A{ ambientColor.x , ambientColor.y , ambientColor.z };
}

void SoftwareRenderer::SetLightPosition(const Vector3f& lightPosition)
{
    m_LightPosition     = lightPosition;
    m_LightPositionSimd = Vector3f256A{ lightPosition.x , lightPosition.y , lightPosition.z };
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

    if( !threadsCount )
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

void SoftwareRenderer::SetMathType(eMathType mathType)
{
    m_MathIndex = static_cast<uint8_t>(mathType);
}