/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include <SFML/Graphics.hpp>
#include "../imgui/imgui.h"
#include "../imgui/imgui-SFML.h"
#include <tracy/Tracy.hpp>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <chrono>
#include <string>
#include <filesystem>
#include <functional>
#include <future>
#include <optional>
#include <atomic>
#include <semaphore>
#include <numbers>
#include <bit>
#include <span>
#include <barrier>

using std::vector;
using std::string;
using std::shared_ptr;
using std::make_shared;
using std::unique_ptr;
using std::make_unique;
using std::function;
using std::filesystem::current_path;
using namespace std::filesystem;
using std::filesystem::exists;
using std::promise;
using std::future;
using std::optional;
using std::atomic_bool;
using std::atomic_int;
using std::counting_semaphore;
using std::max;
using std::thread;
using std::array;
using std::span;
using std::atomic;

namespace pmr
{
    using namespace std::pmr;
}

//const int SCREEN_WIDTH  = 1920*2;
//const int SCREEN_HEIGHT = 1024*2;
const int SCREEN_WIDTH  = 1920;
const int SCREEN_HEIGHT = 1024;
//const int SCREEN_WIDTH  = 640;
//const int SCREEN_HEIGHT = 480;
const int MAX_MODEL_TRIANGLES = 200'000;
const int MAX_TEXTURE_SIZE = 4096;
const int FULL_ANGLE = 360;
const float PI = std::numbers::pi;
const int TRIANGLE_VERT_COUNT = 3;
const string INIT_TEXTURE_PATH = "../Data/Checkerboard.png";
inline const char* MODEL_FORMATS = ".fbx,.glb,.gltf,.blend,.obj";
inline const char* TEXTURE_FORMATS = ".png,.jpg,.jpeg,.bmp";

template< typename T1 , typename T2 , typename ... A >
concept HasConstructFromArray = requires( const T2* p , uint32_t index , const A& ... args ){ { T1::construct_from_array( index , p , args... ) }; };

template< typename T1 , typename T2 , typename ... A >
concept HasLoadFromArray = requires( T1& t1 , const T2* p , uint32_t index , const A& ... args ){ { t1.load_from_array( index , p , args... ) }; };

template< typename T1 , typename T2 , typename ... A >
concept HasStoreToArray = requires( const T1& t1 , T2* p , uint32_t index , const A& ... args ){ { t1.store_to_array( index , p , args... ) }; };

template< typename T >
class Vector2;

struct DrawStats
{
    int m_FrameTriangles         = 0;
    int m_FrameTrianglesDrawn    = 0;
    int m_FrameDrawsPerTile      = 0;
    int m_FramePixels            = 0;
    int m_FramePixelsDrawn       = 0;
    int m_FramePixelsCalcualted  = 0;

    int m_RasterTimeUS           = 0;
    int m_RasterTimePerThreadUS  = 0;
    int m_TransformTimeUS        = 0;
    int m_TransformTimePerThreadUS=0;

    int m_DrawTimeUS             = 0;
    int m_DrawTimePerThreadUS    = 0;
    int m_FillrateKP             = 0;
    int m_DT                     = 0;

    inline void FinishDrawCallStats(Vector2<int> min, Vector2<int> max, int pixelsDrawn);
};

template< typename T , typename D >
struct DependentTypeT{
    using Type = T;
};

template< typename T , typename D >
using DependentType = typename DependentTypeT<T,D>::Type;

enum class eRoundMode : uint8_t
{
    Floor = 0,
    Round = 1,
    Ceil  = 2
};

#define AVX_ALIGN 32
#define ALIGN_FOR_AVX alignas(AVX_ALIGN)

#define SSE_ALIGN 16
#define ALIGN_FOR_SSE alignas(SSE_ALIGN)

// detect clang
#if defined( __clang__ ) || defined( __GNUC__ )
    #define FORCE_INLINE inline __attribute__((always_inline))
#else
    #define FORCE_INLINE __forceinline
#endif

// detect clang
#if defined( __clang__ ) || defined( __GNUC__ )
    #define FINAVX2 __attribute__((target("avx2")))
#else
    #define FINAVX2 __forceinline
#endif
//#define ALIGN_FOR_AVX

namespace Math
{

template< typename T >
inline void Rsqrt( const T& val , T& out )
{
    if constexpr( requires{ val.rsqrt(); } )
        out = val.rsqrt();
    else if constexpr( std::same_as<T,float> )
        _mm_store_ss( &out , _mm_rsqrt_ss( _mm_set_ss( val ) ) );
    else if constexpr( std::same_as<T,double> )
        _mm_store_sd( &out , _mm_rsqrt_sd( _mm_set_sd( val ) ) );
    else
        out = sqrt( 1 / val );
}

template< typename T >
inline void Min( const T& A , const T& B , T& out )
{
    if constexpr( requires{ { A.min( B ) } -> std::same_as<T>; } )
        out = A.min( B );
    else
        out = std::min( A , B );
}

template< typename T >
inline T Min( const T& A , const T& B )
{
    T result;
    Min( A , B , result );
    return result;
}

template< typename T >
inline void Max( const T& A , const T& B , T& out )
{
    if constexpr( requires{ { A.max( B ) } -> std::same_as<T>; } )
        out = A.max( B );
    else
        out = std::max( A , B );
}

template< typename T >
inline T Max( const T& A , const T& B )
{
    T result;
    Max( A , B , result );
    return result;
}

}