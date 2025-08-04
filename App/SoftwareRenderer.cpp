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


int g_selected_tri = 0;
thread_local size_t g_tri_index = 0;

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


    {
        int blozksizee = m_ScreenBuffer.size()/16;

        vector<function<void()>> Tasks;
        Tasks.reserve(m_TileThreadPool.GetThreadCount());

        for( int i=0 ; i<16 ; ++i )
        {
            Tasks.push_back( [this,blozksizee,i]()
            {
                ZoneScopedN( "ClearScreenTask");
                std::fill(m_ScreenBuffer.data()+i*blozksizee, m_ScreenBuffer.data()+(i+1)*blozksizee, m_ClearColor);
            } );
        }

        // launch tasks in the thread pool
        m_TileThreadPool.LaunchTasks( std::move(Tasks) );
    }
}

void SoftwareRenderer::ClearZBuffer()
{
    ZoneScoped;
    //std::fill(m_ZBuffer.begin(), m_ZBuffer.end(), 1.f);

    {
        int blozksizee = m_ZBuffer.size()/16;

        vector<function<void()>> Tasks;
        Tasks.reserve(m_TileThreadPool.GetThreadCount());

        for( int i=0 ; i<16 ; ++i )
        {
            Tasks.push_back( [this,blozksizee,i]()
            {
                ZoneScopedN( "ClearScreenTask");
                std::fill(m_ZBuffer.data()+i*blozksizee, m_ZBuffer.data()+(i+1)*blozksizee, 1.0f);
            } );
        }

        // launch tasks in the thread pool
        m_TileThreadPool.LaunchTasks( std::move(Tasks) );
    }
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

struct ALIGN_FOR_AVX SoftwareRenderer::TriangleData
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
};

struct SoftwareRenderer::ThreadTask
{
    eThreadTaskType TaskType = eThreadTaskType::Unknown;
};

struct SoftwareRenderer::DrawTileData : ThreadTask
{
    bool                IsFullTile = false;
    uint16_t            TileDrawID = 0;
    Vector2i            ScreenPos;
    Vector2i            LogicPos;
    const TriangleData* Triangle;
    const TileInfo*     TileInfo;
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
    //
    //Vector4f test[8] =
    //{
    //    { 10 , 11 , 12 , 13 },
    //    { 20 , 21 , 22 , 23 },
    //    { 30 , 31 , 32 , 33 },
    //    { 40 , 41 , 42 , 43 },
    //    { 50 , 51 , 52 , 53 },
    //    { 60 , 61 , 62 , 63 },
    //    { 70 , 71 , 72 , 73 },
    //    { 80 , 81 , 82 , 83 },
    //};
    //Vector4f test2[8] = {};

    //Vector4f256A vec256;

    //transpose8Vec4f_to_Vec4f256(test, vec256);
    //transposeVec4f256_to_8Vec4f(vec256, test2);

    m_TrianglesData.resize(100,nullptr);
    m_TilesData.resize( ((screenWidth+TILE_SIZE-1)/TILE_SIZE) * ((screenHeight+TILE_SIZE-1)/TILE_SIZE) * 2 );

    m_TileThreadPool.SetThreadCount(16);
}

SoftwareRenderer::~SoftwareRenderer()
{
}

template< typename ... Args >
SoftwareRenderer::TriangleData* SoftwareRenderer::PushTriangleData( Args ... args )
{
    TriangleData* pCurData = m_pCurrentTriangleData.load( std::memory_order_relaxed );
    for( ;; )
    {
        if( m_pCurrentTriangleData.compare_exchange_strong( pCurData , pCurData + 1 ) )
            break;
    }

    new(pCurData) TriangleData( std::forward<Args>(args)...);
    return pCurData;
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
inline void SoftwareRenderer::DrawTileImpl(const DrawTileData& TD, DrawStats* stats)
{
    ZoneScoped;
    const auto pTriangle    = TD.Triangle;
    const auto TilePosition = TD.ScreenPos;

    // lock tile info to prevent concurrent access
    std::scoped_lock lock( TD.TileInfo->Lock );

    const auto EdgeStrideX  = pTriangle->m_EdgeFunctionRails.GetEdgeFunctionsXStride().ToVector3<int>();
    const auto EdgeStrideY  = pTriangle->m_EdgeFunctionRails.GetEdgeFunctionsYStride().ToVector3<int>();
    const auto _EdgeStart   = pTriangle->m_EdgeFunctionRails.GetStartFor( TilePosition.ToVector2<int64_t>() );
    const auto EdgeStartX   = _EdgeStart.x.ToVector3<int>();
    auto       EdgeStartY   = _EdgeStart.y.ToVector3<int>();
    const auto invABC       = pTriangle->m_InvABC;
    const int  StartY       = TilePosition.y;
    const int  EndY         = StartY + TILE_SIZE;
    float*     pZBuffer     = TD.TileInfo->TileZMem;

    auto       pixelsDrawn  = 0;

    using NumericType = decltype(EdgeStartX.x);

    ALIGN_FOR_AVX Vector4f Pixels[TILE_SIZE * TILE_SIZE];
    Vector4f* pPixels = Pixels;

    ALIGN_FOR_AVX TransformedVertex interpolatedVertex;

    // loop through all pixels in tile square
    for (int y = StartY ; y < EndY; y++ , EdgeStartY += EdgeStrideY )
    {
        uint32_t* pColBuffer = m_ScreenBuffer.data() + y* SCREEN_WIDTH + TilePosition.x;

        auto EdgeFunctions = EdgeStartX + EdgeStartY;

        for (int ix = 0; ix < TILE_SIZE; ix++ , EdgeFunctions += EdgeStrideX )
        {
            if constexpr( Partial )
            {
                if( EdgeFunctionTest( EdgeFunctions ) )
                    continue; // outside triangle
            }

            Vector3f baricentricCoordinates = EdgeFunctions.ToVector3<float>() * invABC;

            pTriangle->m_Interpolator.InterpolateT<MathCPU>( Vector3f( baricentricCoordinates.y , baricentricCoordinates.z , baricentricCoordinates.x ) , interpolatedVertex);

            float& z = pZBuffer[ix];
            if (interpolatedVertex.m_ScreenPosition.z < z){
                if (m_ZWrite)
                    z = interpolatedVertex.m_ScreenPosition.z;
            }
            else if (m_ZTest){
                continue;
            }

            Vector4f finalColor = FragmentShader(interpolatedVertex);
            //finalColor = Vector4f{ interpolatedVertex.m_UV , 1.0f , 1.0f };

            if constexpr( Partial )
            {
                pColBuffer[ix] = Vector4f::ToARGB(finalColor);
                pixelsDrawn++;
            }
            else
            {
                *pPixels++ = finalColor;
            }
        }
        pZBuffer += TILE_SIZE; // move to next row in Z-buffer
    }

    if constexpr( !Partial )
    {
        pPixels = Pixels;
        for (int y = StartY ; y < EndY; y++)
        {
            TileLineFToARGB( pPixels , m_ScreenBuffer.data() + y* SCREEN_WIDTH + TilePosition.x );
            pPixels += TILE_SIZE;
        }
    }

    if( stats )
        stats->m_FramePixelsDrawn += pixelsDrawn;
}

template< eSimdType Type , bool Partial , int Elements  >
inline void SoftwareRenderer::DrawFulllTileImplSimd(const DrawTileData& TD, DrawStats* stats)
{
    using f256t             = fsimd<Elements,Type>;
    using i256t             = isimd<Elements,Type>;
    using Vector3i256t      = Vector3<i256t>;
    using Vector4f256t      = Vector4<f256t>;
    using TransformedVertexT= SimdTransformedVertex<Elements,Type>;


    ZoneScoped;
    const auto pTriangle    = TD.Triangle;
    const auto pTileInfo    = TD.TileInfo;
    constexpr int pack_size = f256t::elements_count;

    // lock tile info to prevent concurrent access
    std::scoped_lock lock( pTileInfo->Lock );



    const auto TilePosition = TD.ScreenPos;
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
    const int  StartY       = TilePosition.y;
    const int  EndY         = StartY + TILE_SIZE;
    auto       pixelsDrawn  = 0;
    EdgeStrideX *= pack_size;


    // local pixels storage - it seems to be faster than using TileMem directly
    ALIGN_FOR_AVX Vector4f  Pixels       [TILE_SIZE * TILE_SIZE];
    ALIGN_FOR_AVX uint32_t  TileCoverage [TILE_SIZE] = {};
    TransformedVertexT      interpolatedVertex;
    Vector4f*               pCurPixel       = Pixels;
    uint32_t*               CoverageMask    = TileCoverage;
    float*                  pZBuffer        = pTileInfo->TileZMem;
    i256t                   write_mask;

    // loop through all pixels in tile square
    for (int y = StartY ; y < EndY; y++ , EdgeStartY += EdgeStrideY )
    {
        // Calculate edge functions for the first pixel in the current line
        auto EdgeFunctions = EdgeStartX + EdgeStartY;

        for (int ix = 0; ix < TILE_SIZE; ix+=pack_size , EdgeFunctions += EdgeStrideX )
        {
            auto baricentricCoordinates = EdgeFunctions.ToVector3<f256t>() * invABC;

            // prepare coverage mask for bits from current pixel pack
            *CoverageMask <<= pack_size;
            // initialize coverage with 1s - initially we assume that all pixels in current pack will be written
            uint32_t CurrentPackCoverageMask = 0xFFFFFFFF;

            if constexpr( Partial )
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
            *CoverageMask |= CurrentPackCoverageMask;

            // Interpolate all other vertex attributes (UV, normal, etc.) for all pixels in the current pack
            pTriangle->m_Interpolator.InterpolateAllButZ( baricentricCoordinates, interpolatedVertex );


            // Call fragment shader to compute final color for all pixels in the current pack
            Vector4f256t finalColor = FragmentShader(interpolatedVertex);
            // store the final color in the local pixel buffer - we will write it to the screen buffer later
            finalColor.store( pCurPixel[ix].Data() , simd_alignment::AVX );

            //Vector4f256t finalColor = Vector4f256t(interpolatedVertex.m_UV * f256A{ 1.0f }, f256A{ 1.0f } , f256A{ 1.0f } );
        }

        pCurPixel += TILE_SIZE;
        pZBuffer  += TILE_SIZE;
        CoverageMask++;
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
        pCurPixel    = Pixels;
        CoverageMask = TileCoverage;
        for (int y = StartY ; y < EndY; y++)
        {
            TileLineFToARGB_Simd<true,8,eSimdType::AVX>( pCurPixel , m_ScreenBuffer.data() + y* SCREEN_WIDTH + TilePosition.x , White , *CoverageMask );
            pCurPixel += TILE_SIZE;
            CoverageMask++;
        }
    }

    if( stats )
        stats->m_FramePixelsDrawn += pixelsDrawn;
}

void SoftwareRenderer::PushTile( DrawTileData data )
{
    DrawTileData* pCurData = m_pCurrentTileData.load( std::memory_order_relaxed );
    for( ;; )
    {
        if( m_pCurrentTileData.compare_exchange_strong( pCurData , pCurData + 1 ) )
            break;
    }

    *pCurData = data;

    //if( data.IsFullTile  )
    //    DrawFullTile(data,nullptr);
    //else
    //    DrawPartialTile(data,nullptr);
}

bool g_useSimd = true;

void SoftwareRenderer::DrawPartialTile(const DrawTileData& TD, DrawStats* stats)
{
    if( g_useSimd )
        return DrawFulllTileImplSimd<eSimdType::AVX,true,8>( TD, stats );
    else
        return DrawTileImpl<true>(TD, stats);
}
void SoftwareRenderer::DrawFullTile(const DrawTileData& TD, DrawStats* stats)
{
    if( g_useSimd )
        return DrawFulllTileImplSimd<eSimdType::AVX,false,8>( TD, stats );
    else
        return DrawTileImpl<false>(TD, stats);
}
void SoftwareRenderer::DrawTile(const DrawTileData& TD, DrawStats* stats)
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

bool g_showTilesBoundry      = false;
bool g_showTilestype         = false;
bool g_showTriangleBoundry   = false;
bool g_showCornersClassify   = false;

void SoftwareRenderer::DrawFilledTriangleT(const TransformedVertex& VA, const TransformedVertex& VB, const TransformedVertex& VC, const Vector4f& color, DrawStats& stats)
{
    ZoneScoped;
    // filling algorithm is working that way that we are going through all pixels in rectangle that is created by min and max points
    // and we are checking if pixel is inside triangle by using three lines and checking if pixel is on the same side of each line

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

    //min2.x = (min2.x + MultiplierMask) & (~MultiplierMask);
    //min2.y = (min2.y + MultiplierMask) & (~MultiplierMask);
    //max2.x = (max2.x + MultiplierMask) & (~MultiplierMask);
    //max2.y = (max2.y + MultiplierMask) & (~MultiplierMask);

    VertexInterpolator interpolator(VA, VB, VC);

    const auto StartP = ( min );//+ PixelOffset );

    constexpr auto PixelOffset = Vector2<int64_t>{ Precision.Multiplier , Precision.Multiplier }/2;

    TriangleData& Data = *PushTriangleData( VA , VB , VC , Vector4f{1,1,1,1} , A , B , C , StartP );
    Data.m_InvABC = (1.0f / ABC);

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
                auto ID = pTileInfo->DrawCount++;

                DrawTileData TD;
                TD.LogicPos  = TilePos.ToVector2<int>();
                TD.ScreenPos = (TD.LogicPos / Precision.Multiplier);
                TD.Triangle  = &Data;
                TD.IsFullTile= true;
                TD.TileInfo  = pTileInfo;
                TD.TileDrawID= ID;
                PushTile(TD);
                //DrawFullTile( TD , stats );
                if( g_showTilestype )
                    ImGuiAddRectFilled(TilePos, TilePos + TILE_SIZE, ImColor(64, 255, 0, 64) );
            }
            else if( Classified == eTileCoverage::Partial )
            {
                auto ID = pTileInfo->DrawCount++;

                DrawTileData TD;
                TD.LogicPos  = TilePos.ToVector2<int>();
                TD.ScreenPos = (TD.LogicPos / Precision.Multiplier);
                TD.Triangle  = &Data;
                TD.IsFullTile= false;
                TD.TileInfo  = pTileInfo;
                TD.TileDrawID= ID;
                PushTile(TD);
                //DrawPartialTile( TD , stats );
                if( g_showTilestype )
                    ImGuiAddRectFilled(TilePos, TilePos + TILE_SIZE, ImColor(0, 64, 255, 64) );
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

    for( int i=0 ; m_TilesGridSize.x * m_TilesGridSize.y > i ; ++i )
    {
        m_TilesGrid[i].DrawCount = 0;
    }
}

void SoftwareRenderer::EndFrame()
{
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

    for( int i=0 ; i<SCREEN_WIDTH; i+=64 )
        ImGui::GetBackgroundDrawList()->AddLine( ImVec2( i,0 ), ImVec2(i ,SCREEN_HEIGHT) , ImColor(255,128,255,128) );

    for( int i=0 ; i<SCREEN_HEIGHT; i+=64 )
        ImGui::GetBackgroundDrawList()->AddLine( ImVec2( 0,i ), ImVec2(SCREEN_WIDTH ,i) , ImColor(255,128,255,128) );
}

void SoftwareRenderer::Render(const vector<Vertex>& vertices)
{
    m_pCurrentTriangleData.store( m_TrianglesData.data() );
    m_pCurrentTileData.store( m_TilesData.data() );

    m_pSelectedMath = m_MathArray[std::clamp(m_MathIndex,0,2)];
    //m_pSelectedMath->log();

    ZoneScoped;
    const auto startTime = std::chrono::high_resolution_clock::now();
    int threadsCount = m_ThreadPool.GetThreadCount();
    if (threadsCount > 0)
    {
        int linesPerThread = SCREEN_HEIGHT / threadsCount;
        int lineStyartY = 0;
        int lineEndY = linesPerThread;

        vector<function<void()>> tasks(threadsCount);
        for (int i = 0; i < threadsCount; ++i)
        {
            if (i + 1 == threadsCount)
                lineEndY = SCREEN_HEIGHT - 1;
            tasks[i] = [this,&vertices, lineStyartY, lineEndY,i]
            {
                ZoneScopedN("Render Task");
                DoRender(vertices, lineStyartY, lineEndY,i);
            };
            lineStyartY = lineEndY+1;
            lineEndY = lineStyartY+linesPerThread;
        }

        m_ThreadPool.LaunchTasks(std::move(tasks));
    }
    else
    {
        ZoneScopedN("Preprocess triangles");
        DoRender(vertices, 0, SCREEN_HEIGHT - 1,0);
    }
    const auto timeUS = std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::high_resolution_clock::now() - startTime).count();
    m_FrameDrawTimeMainUS += timeUS;

    auto pTileDataEnd = m_pCurrentTileData.load();
    m_pCurrentTileData = m_TilesData.data();

    {
        ZoneScopedN("Render Tiles");
        // set current tile job pointer to the start of tiles data
        m_pCurrentTileJob = m_TilesData.data();

        auto Task = [this,&pTileDataEnd]()
        {
            ZoneScopedN("Render task");
            for(;;)
            {
                // get current tile to render
                auto pTileData = m_pCurrentTileJob.load(std::memory_order_relaxed);
                if( !pTileData || pTileData >= pTileDataEnd )
                    return; // all tiles are processed - exit task

                // increment current tile job pointer to signal that we are processing this tile
                if( !m_pCurrentTileJob.compare_exchange_strong( pTileData , pTileData + 1 ) )
                    continue;

                // process tile
                //if( pTileData->TaskType != eThreadTaskType::ComposeTile )
                    DrawTile(*pTileData,nullptr);
            };
        };

        // create a vector of tasks for each thread in the thread pool
        vector<function<void()>> Tasks( m_TileThreadPool.GetThreadCount() , function<void()>(Task) );

        // launch tasks in the thread pool
        m_TileThreadPool.LaunchTasks( std::move(Tasks) );
    }
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

struct SoftwareRenderer::Internal
{
    using DrawFuncT = void (SoftwareRenderer::*)(const TransformedVertex* pVerts, size_t Count, const Vector4f& color, int minY, int maxY, DrawStats& stats);

    template< typename MathT , uint8_t FunctionIndex >
    static constexpr DrawFuncT GenerateDrawFunction()
    {
        constexpr DrawFunctionConfig Config{ FunctionIndex };
        return &SoftwareRenderer::DrawFilledTriangles<MathT,Config>;
    };


    static constexpr inline size_t Funcs = 1 << DrawFunctionConfig::Bits;

    struct Array
    {
        DrawFuncT m_Array[Funcs] = {};

        constexpr DrawFuncT operator[](size_t index) const
        {
            return m_Array[index];
        }
    };

    template< typename MathT , size_t ... FunctionIndices >
    static constexpr Array GenerateDrawFunctionsArrayHelper( std::index_sequence< FunctionIndices... > )
    {
        return Array{ GenerateDrawFunction<MathT,FunctionIndices>()... };
    };

    template< typename MathT >
    static constexpr Array GenerateDrawFunctionsArray()
    {
        return GenerateDrawFunctionsArrayHelper<MathT>(std::make_index_sequence<Funcs>() );
    };


    template< typename MathT >
    static constexpr inline Array DrawFunctions = []{ return GenerateDrawFunctionsArray<MathT>(); }();
};

void SoftwareRenderer::DoRender(const vector<Vertex>& inVertices, int minY, int maxY, int threadID)
{
    ZoneScoped;
    Plane nearFrustumPlane;
    m_MVPMatrix.GetFrustumNearPlane(nearFrustumPlane);

    const auto startTime = std::chrono::high_resolution_clock::now();

    const vector<Vertex>& vertices = ClipTriangles(nearFrustumPlane, 0.001f, inVertices);
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

        ////printf(".");


        ///constexpr auto pFunc = &SoftwareRenderer::DrawFilledTriangles<MathCPU,DrawFunctionConfig{}>;
        ///
        /////DrawFilledTriangles<MathSSE,DrawFunctionConfig{}>( transformedVertices.data(), transformedVertices.size(), color, minY, maxY, drawStats );
        ///(this->*pFunc)(transformedVertices.data(), transformedVertices.size(), color, minY, maxY, drawStats);

        //DrawFilledTriangles<MathSSE,DrawFunctionConfig{}>(transformedVertices.data(), transformedVertices.size(), color, minY, maxY, drawStats);;

        //printf(".");

        //if( m_MathIndex == 1 )
        //    (this->*Internal::DrawFunctions<MathSSE>[Index])(transformedVertices.data(), transformedVertices.size(), color, minY, maxY, drawStats);
        ////else if( m_MathIndex == 2 )
        ////    (this->*Internal::DrawFunctions<MathAVX>[Index])(transformedVertices.data(), transformedVertices.size(), color, minY, maxY, drawStats);
        //else
            (this->*Internal::DrawFunctions<MathCPU>[Index])(transformedVertices.data(), transformedVertices.size(), color, minY, maxY, drawStats);

        //if( m_MathIndex == 1 )
        //    DrawFilledTriangles<MathSSE,DrawFunctionConfig{}>(transformedVertices.data(), transformedVertices.size(), color, minY, maxY, drawStats);
        //else if( m_MathIndex == 2 )
        //    DrawFilledTriangles<MathAVX,DrawFunctionConfig{}>(transformedVertices.data(), transformedVertices.size(), color, minY, maxY, drawStats);
        //else
        //    DrawFilledTriangles<MathCPU,DrawFunctionConfig{}>(transformedVertices.data(), transformedVertices.size(), color, minY, maxY, drawStats);

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
            g_tri_index = i;
            DrawFilledTriangleBaseline<MathT,Config>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
            //DrawFilledTriangle<MathT>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
        }
        return;

    case eDrawTriVersion::DrawTriv2:
        for (size_t i = 0; i < Count; i += TRIANGLE_VERT_COUNT)
        {
            g_tri_index = i;
            DrawFilledTriangle_v2<MathT,Config>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
            //DrawFilledTriangle<MathT>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
        }
        return;

    case eDrawTriVersion::DrawTriv3:
        for (size_t i = 0; i < Count; i += TRIANGLE_VERT_COUNT)
        {
            g_tri_index = i;
            DrawFilledTriangle_v3<MathT,Config>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
            //DrawFilledTriangle<MathT>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
        }
        return;


    default:
        for (size_t i = 0; i < Count; i += TRIANGLE_VERT_COUNT)
        {
            g_tri_index = i;
            DrawFilledTriangle<MathT,Config>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
            //DrawFilledTriangle<MathT>(pVerts[i+0], pVerts[i+1], pVerts[i+2], color, minY, maxY, stats);
        }
        return;
    }

}

constexpr int impl_version = 3;


void aligntest( const void* ptr )
{
    size_t pos = reinterpret_cast<size_t>(ptr);
    if( pos % AVX_ALIGN != 0 )
    {
        exit(0);
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

int NextPowerOfTwo(int x)
{
    if (x <= 0)
        return 1;
    --x;
    for (int i = 1; i < sizeof(int) * 8; i <<= 1)
        x |= x >> i;
    return x + 1;
}

//template< typename T >
//struct line2
//{
//    constexpr line2() = default;
//
//    constexpr inline line2(Vector2<T> start, Vector2<T> end)
//    {
//        A = start;
//        B = end;
//        AB = B - A;
//
//        Normal = AB.Rotated90();
//        Normal.Normalize();
//
//        D = - (Normal.Dot(start));
//    }
//
//    constexpr inline float DistanceToPoint(const Vector2<T> &v) const
//    {
//        return Normal.Dot(v) + D;
//    }
//    constexpr auto MidPoint() const
//    {
//        return A + AB / 2;
//    }
//
//    Vector2<T> A;
//    Vector2<T> B;
//    Vector2<T> AB;
//    float      D = 0; // distance from origin to line
//    Vector2<T> Normal;
//};
//
//using line2f = line2<float>;
//using line2i = line2<int>;
//
//template< typename T >
//line2( Vector2<T> start, Vector2<T> end ) -> line2<T>;


template< typename MathT , DrawFunctionConfig Config >
void SoftwareRenderer::DrawFilledTriangle(const TransformedVertex& VA, const TransformedVertex& VB, const TransformedVertex& VC, const Vector4f& color, int minY, int maxY, DrawStats& stats)
{
    ZoneScoped;

    //if( g_tri_index == size_t(g_selected_tri) )
        return DrawFilledTriangleT(VA, VB, VC, color, stats);

    // filling algorithm is working that way that we are going through all pixels in rectangle that is created by min and max points
    // and we are checking if pixel is inside triangle by using three lines and checking if pixel is on the same side of each line

    Vector2f A = VA.m_ScreenPosition.xy();//.ToVector2i().ToVector2f();
    Vector2f B = VB.m_ScreenPosition.xy();//.ToVector2i().ToVector2f();
    Vector2f C = VC.m_ScreenPosition.xy();//.ToVector2i().ToVector2f();

    // clockwise order so we check if point is on the right side of line
    const float ABC = EdgeFunction(A, B, C);

    // if our edge function (signed area x2) is negative, it's a back facing triangle and we can cull it
    bool isBackFacing = ABC <= 0;
    if (isBackFacing)
    {
        stats.m_FrameTriangles++;
        return;
    }

    //if( g_tri_index == size_t(g_selected_tri) )
    //{
    //    TileTriangle( A , B , C );
    //}

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
    Vector2i min = A.CWiseMin(B).CWiseMin(C).ToVector2i();
    Vector2i max = A.CWiseMax(B).CWiseMax(C).ToVector2i();

    // clamp min and max points to screen size so we don't calculate points that we don't see
    min = min.CWiseMin(Vector2i(SCREEN_WIDTH-1, maxY)).CWiseMax(Vector2i(0, minY));
    max = max.CWiseMin(Vector2i(SCREEN_WIDTH-1, maxY)).CWiseMax(Vector2i(0, minY));

    const float invABC = 1.0f / ABC;

    VertexInterpolator interpolator(VA, VB, VC);
    TransformedVertex interpolatedVertex;

    int pixelsDrawn = 0;

    struct ALIGN_FOR_AVX EdgeResult_t
    {
        float ABP = 0;
        float BCP = 0;
        float CAP = 0;
        int   SKIP= 0;
      //float _[5] = {0};
    };
    ALIGN_FOR_AVX float Arg0[8] =
    {
        - (B.y - A.y) ,
          (B.x - A.x) ,
        - (C.y - B.y) ,
          (C.x - B.x) ,
        - (A.y - C.y) ,
          (A.x - C.x) ,
          0,
          0,
    };
    ALIGN_FOR_AVX float Arg2[8] =
    {
        - Arg0[0] * A.x ,
        - Arg0[1] * A.y ,
        - Arg0[2] * B.x ,
        - Arg0[3] * B.y ,
        - Arg0[4] * C.x ,
        - Arg0[5] * C.y ,
          0,
    };
    //ALIGN_FOR_AVX float Arg1[8] = {};
    EdgeResult_t EdgeResult;
    EdgeResult8_t EdgeResult8;
    ALIGN_FOR_AVX Vector4f baricentricCoordinates;

    MathT math;
    //VertexInterpolator::InterpolatedSource tmpA;

    const int PrecisionBits = 10;
    const int Multiplier    = 1 << PrecisionBits;
    const int MultiplierMask= Multiplier - 1;

    Vector2 A2 = (VA.m_ScreenPosition.xy()*Multiplier).ToVector2<int64_t,eRoundMode::Round>();
    Vector2 B2 = (VB.m_ScreenPosition.xy()*Multiplier).ToVector2<int64_t,eRoundMode::Round>();
    Vector2 C2 = (VC.m_ScreenPosition.xy()*Multiplier).ToVector2<int64_t,eRoundMode::Round>();
    Vector2 min2 = A2.CWiseMin(B2).CWiseMin(C2);
    Vector2 max2 = A2.CWiseMax(B2).CWiseMax(C2);
    min2 = min2.CWiseMax(Vector2<int64_t>(0, minY*Multiplier));
    max2 = max2.CWiseMax(Vector2<int64_t>(0, minY*Multiplier));

    min2.x = (min2.x + MultiplierMask) & (~MultiplierMask);
    min2.y = (min2.y + MultiplierMask) & (~MultiplierMask);
    max2.x = (max2.x + MultiplierMask) & (~MultiplierMask);
    max2.y = (max2.y + MultiplierMask) & (~MultiplierMask);

    min2 = min2.CWiseMin(Vector2<int64_t>(SCREEN_WIDTH*Multiplier-1, maxY*Multiplier));
    max2 = max2.CWiseMin(Vector2<int64_t>(SCREEN_WIDTH*Multiplier-1, maxY*Multiplier));

    // clockwise order so we check if point is on the right side of line
    const auto ABC2 = EdgeFunction(A2, B2, C2);
    //const float invABC2 = 1.0f / ABC2;

    constexpr auto PixelOffset = Vector2<int64_t>{ Multiplier , Multiplier }/2;

    const auto StartP = ( min2 + PixelOffset );


          auto CAP_Stride = Vector2( C2.y - A2.y , A2.x - C2.x );
          auto BCP_Stride = Vector2( B2.y - C2.y , C2.x - B2.x );
          auto ABP_Stride = Vector2( A2.y - B2.y , B2.x - A2.x );

    const auto CAP_Start  = CAP_Stride * (StartP - Vector2( C2.x , C2.y ) );
    const auto BCP_Start  = BCP_Stride * (StartP - Vector2( B2.x , B2.y ) );
    const auto ABP_Start  = ABP_Stride * (StartP - Vector2( A2.x , A2.y ) );

    CAP_Stride = CAP_Stride * Multiplier;
    BCP_Stride = BCP_Stride * Multiplier;
    ABP_Stride = ABP_Stride * Multiplier;


    auto ABP_Y = ABP_Start.y;
    auto BCP_Y = BCP_Start.y;
    auto CAP_Y = CAP_Start.y;

    // loop through all pixels in rectangle
    for (auto ScrrenY = min2.y; ScrrenY <= max2.y; ScrrenY+=Multiplier, ABP_Y+= ABP_Stride.y, BCP_Y+=BCP_Stride.y, CAP_Y+=CAP_Stride.y)
    {
        int linearpos = (ScrrenY>>PrecisionBits) * SCREEN_WIDTH + min.x;
        uint32_t LineIndex = 0;

        auto ABP_X = ABP_Start.x;
        auto BCP_X = BCP_Start.x;
        auto CAP_X = CAP_Start.x;

        for (auto ScrrenX = min2.x; ScrrenX <= max2.x; ScrrenX+=Multiplier, ++linearpos, ABP_X += ABP_Stride.x, BCP_X += BCP_Stride.x, CAP_X += CAP_Stride.x)
        {
            ALIGN_FOR_AVX const Vector2f P( (ScrrenX>>PrecisionBits)+0.5f, (ScrrenY>>PrecisionBits)+0.5f );


            //const float CAP = (A.x - C.x) * (P.y - C.y)
            //                - (A.y - C.y) * (P.x - C.x);
            //const float ABP = (B.x - A.x) * (P.y - A.y)
            //                - (B.y - A.y) * (P.x - A.x);
            //const float BCP = (C.x - B.x) * (P.y - B.y)
            //                - (C.y - B.y) * (P.x - B.x);

            //if constexpr( impl_version==0 )
            //{
                m_pSelectedMath->EdgeFunction3x( P , Arg0 , Arg2 , &(EdgeResult.ABP) );
                //math.EdgeFunction3x( P , Arg0 , Arg2 , &EdgeResult.ABP );

            //    if (EdgeResult.ABP < 0)
            //        continue;
            //    if (EdgeResult.BCP < 0)
            //        continue;
            //    if (EdgeResult.CAP < 0)
            //        continue;
            //}
            //else
                if constexpr( impl_version==1 )
            {
                //aligntest( &P );
                //aligntest( Arg0 );
                //aligntest( Arg2 );
                //aligntest( &EdgeResult );

                if( math.EdgeFunction3xToBool( P , Arg0 , Arg2 , &EdgeResult.ABP ) )
                    continue;
            }
            else if constexpr( impl_version==2 )
            {
                auto Index = LineIndex%8;
                if( Index==0 )
                    m_pSelectedMath->EdgeFunction3xToBoolx8( P , Arg0 , Arg2 , EdgeResult8.data() );

                if( EdgeResult8.SKIP[ Index ] )
                    continue;

                EdgeResult.ABP = EdgeResult8.ABP[ Index ];
                EdgeResult.BCP = EdgeResult8.BCP[ Index ];
                EdgeResult.CAP = EdgeResult8.CAP[ Index ];
            }
            else if constexpr( impl_version==3 )
            {
                EdgeResult_t EdgeResult;

                auto ABP = (ABP_X + ABP_Y);
                auto BCP = (BCP_X + BCP_Y);
                auto CAP = (CAP_X + CAP_Y);

                if (ABP < 0)
                    continue;
                if (BCP < 0)
                    continue;
                if (CAP < 0)
                    continue;

                EdgeResult.ABP = ABP;
                EdgeResult.BCP = BCP;
                EdgeResult.CAP = CAP;
            }

            // if pixel is inside triangle, draw it

            // dividing edge function values by ABC will give us barycentric coordinates - how much each vertex contributes to final color in point P

            //math.MultiplyVec4ByScalar( &EdgeResult.ABP , invABC2 , &baricentricCoordinates.x );
            //Vector3f baricentricCoordinates = Vector3f( EdgeResult.BCP, EdgeResult.CAP , EdgeResult.ABP) * invABC;

            Vector3f baricentricCoordinates = Vector3f( EdgeResult.ABP , EdgeResult.BCP , EdgeResult.CAP ) * invABC;

            interpolator.InterpolateT<MathT>( Vector3f( baricentricCoordinates.y , baricentricCoordinates.z , baricentricCoordinates.x ) , interpolatedVertex);

            //float& z = m_ZBuffer[y * SCREEN_WIDTH + x];

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
            PutPixelUnsafe(ScrrenX>>PrecisionBits, ScrrenY>>PrecisionBits, Vector4f::ToARGB(finalColor));
            pixelsDrawn++;
        }
    }

    stats.FinishDrawCallStats(min,max,pixelsDrawn);
}

void SoftwareRenderer::DrawFilledTriangleWTF(const TransformedVertex& VA, const TransformedVertex& VB, const TransformedVertex& VC, const Vector4f& color, int minY, int maxY, DrawStats& stats)
{
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

    // min and max points of rectangle
    Vector2f min = A.CWiseMin(B).CWiseMin(C);
    Vector2f max = A.CWiseMax(B).CWiseMax(C);

    // clamp min and max points to screen size so we don't calculate points that we don't see
    min = min.CWiseMin(Vector2f(SCREEN_WIDTH-1, maxY)).CWiseMax(Vector2f(0, minY));
    max = max.CWiseMin(Vector2f(SCREEN_WIDTH-1, maxY)).CWiseMax(Vector2f(0, minY));

    const float invABC = 1.0f / ABC;

    VertexInterpolator interpolator(VA, VB, VC);
    TransformedVertex interpolatedVertex;

    int pixelsDrawn = 0;


    ALIGN_FOR_AVX Vector2f PABC [4] = { A, B, C };
    ALIGN_FOR_AVX float    PREF[8] =
    {
        - (B.y - A.y) ,
          (B.x - A.x) ,
        - (C.y - B.y) ,
          (C.x - B.x) ,
        - (A.y - C.y) ,
          (A.x - C.x) ,
    };
    ALIGN_FOR_AVX float SUFF[8] =
    {
        - PREF[0] * PABC[0].x ,
        - PREF[1] * PABC[0].y ,
        - PREF[2] * PABC[1].x ,
        - PREF[3] * PABC[1].y ,
        - PREF[4] * PABC[2].x ,
        - PREF[5] * PABC[2].y ,
    };
    ALIGN_FOR_AVX float PS[8] = {};
    //PABC

    // loop through all pixels in rectangle
    for (int y = min.y; y <= max.y; ++y)
    {
        for (int x = min.x; x <= max.x; ++x)
        {
            const Vector2f P(x+0.5f, y+0.5f);
            // calculate value of edge function for each line

            PS[0] = P.x;
            PS[1] = P.y;
            PS[2] = P.x;
            PS[3] = P.y;
            PS[4] = P.x;
            PS[5] = P.y;

            m_pSelectedMath->MulAddVec8( PREF , PS , SUFF , PS );

            const float ABP = PS[0] + PS[1];
            const float BCP = PS[2] + PS[3];
            const float CAP = PS[4] + PS[5];

            if (ABP < 0)
                continue;
            if (BCP < 0)
                continue;
            if (CAP < 0)
                continue;
            // if pixel is inside triangle, draw it
            //
            // dividing edge function values by ABC will give us barycentric coordinates - how much each vertex contributes to final color in point P
            Vector3f baricentricCoordinates = Vector3f( BCP, CAP , ABP) * invABC;
            //interpolator.InterpolateZ(baricentricCoordinates, interpolatedVertex);

            interpolator.Interpolate(*m_pSelectedMath, baricentricCoordinates, interpolatedVertex);

            float& z = m_ZBuffer[y * SCREEN_WIDTH + x];
            if (interpolatedVertex.m_ScreenPosition.z < z) {
                if (m_ZWrite)
                    z = interpolatedVertex.m_ScreenPosition.z;
            }
            else if (m_ZTest){
                continue;
            }

            //interpolator.InterpolateAllButZ(baricentricCoordinates, interpolatedVertex);
            interpolatedVertex.m_Color = interpolatedVertex.m_Color * color;
            Vector4f finalColor = FragmentShader(interpolatedVertex);
            PutPixelUnsafe(x, y, Vector4f::ToARGB(finalColor));
            pixelsDrawn++;
        }
    }

    stats.m_FramePixels += (1+max.y-min.y)*(1+max.x-min.x);
    stats.m_FramePixelsDrawn += pixelsDrawn;
    stats.m_FrameTriangles++;
    stats.m_FrameTrianglesDrawn++;
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
    if (threadsCount == 1)
        // no need to use thread pool for just 1 thread - execute work on main thread
        threadsCount = 0;

    if (m_ThreadsCount == threadsCount)
        return;

    m_ThreadsCount = threadsCount;
    m_ThreadPool.SetThreadCount(m_ThreadsCount);
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