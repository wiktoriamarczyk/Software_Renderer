/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once

#include "Common.h"

__m128 emu_mm_log_ps(const __m128& d);
__m128 emu_mm_exp_ps(const __m128& d);
__m128 emu_mm_pow_ps(const __m128& x, const __m128& y);
__m128 emu_mm_pow_ps(const __m128& x, float y);

__m256 emu_mm256_log_ps(const __m256& d);
__m256 emu_mm256_exp_ps(const __m256& d);
__m256 emu_mm256_pow_ps(const __m256& x, const __m256& y);
__m256 emu_mm256_pow_ps(const __m256& x, float y);

//****************************************************************
//  SIMD types and data alignment
//****************************************************************

enum class eSimdType
{
    None    = 0,
    SSE     = 1,
    AVX     = 2,
};

enum class eDataAlignment
{
    None    = 0,
    Align16 = 16,
    Align32 = 32,

    SSE     = Align16,
    AVX     = Align32,
};

template< eDataAlignment Alignment >
using DataAlignmentTagT = std::integral_constant<eDataAlignment, Alignment>;

using DataAlignmentNoneTag = DataAlignmentTagT<eDataAlignment::None>;
using DataAlignmentSSETag  = DataAlignmentTagT<eDataAlignment::SSE>;
using DataAlignmentAVXTag  = DataAlignmentTagT<eDataAlignment::AVX>;

namespace simd_alignment
{
    constexpr DataAlignmentNoneTag None;
    constexpr DataAlignmentSSETag  SSE;
    constexpr DataAlignmentAVXTag  AVX;
}

template< typename T >
consteval auto GetDataAlignmentTagFor()
{
    return DataAlignmentTagT< eDataAlignment(alignof(T)) >{};
}

namespace _details
{
    template< int Size >
    constexpr inline void* MAX_UINT = nullptr;
    template<>
    constexpr inline uint8_t MAX_UINT<1> = 0xFF;
    template<>
    constexpr inline uint16_t MAX_UINT<2> = 0xFFFF;
    template<>
    constexpr inline uint32_t MAX_UINT<4> = 0xFFFFFFFF;
    template<>
    constexpr inline uint64_t MAX_UINT<8> = 0xFFFFFFFFFFFFFFFF;

    template< typename T >
    constexpr float ALL_BITS_ONE = std::bit_cast<T>( MAX_UINT<sizeof(T)> );
    template< typename T >
    constexpr float ALL_BITS_ZERO = std::bit_cast<T>( 0 );
}

//****************************************************************
// SIMD CPU data type
//****************************************************************

template< typename T , int Elements >
struct simd_data_type
{
    T v[Elements] = {};
    FORCE_INLINE constexpr       T& operator[](int i)       noexcept { return v[i]; }
    FORCE_INLINE constexpr const T& operator[](int i) const noexcept { return v[i]; }
};


template< typename T , eSimdType Type , int Elements >
struct simd_type_maping;

template< typename T , int Elements >
struct simd_type_maping< T , eSimdType::None , Elements >
{
    using type = simd_data_type< T, Elements>;

    FORCE_INLINE static void add(const type& A , const type& B, type& R ) noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = A.v[i] + B.v[i];
    }
    FORCE_INLINE static void add(const type& A , const T& B , type& R ) noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = A.v[i] + B;
    }

    FORCE_INLINE static void sub(const type& A , const type& B, type& R ) noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = A.v[i] - B.v[i];
    }
    FORCE_INLINE static void sub(const type& A , const T& B, type& R ) noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = A.v[i] - B;
    }

    FORCE_INLINE static void mul(const type& A , const type& B, type& R ) noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = A.v[i] * B.v[i];
    }

    FORCE_INLINE static void mul(const type& A , const T& B, type& R ) noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = A.v[i] * B;
    }

    FORCE_INLINE static void div(const type& A , const type& B, type& R ) noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = A.v[i] / B.v[i];
    }

    FORCE_INLINE static void AND(const type& A , const type& B, type& R ) noexcept requires( std::is_integral_v<T> )
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = A.v[i] | B.v[i];
    }

    FORCE_INLINE static void OR(const type& A , const type& B, type& R ) noexcept requires( std::is_integral_v<T> )
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = A.v[i] | B.v[i];
    }
    FORCE_INLINE static void NOT(const type& A, type& R)noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = ~A.v[i];
    }

    FORCE_INLINE static void div(const type& A , const T& B, type& R ) noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = A.v[i] / B;
    }

    FORCE_INLINE static void shl(const type& A , int32_t B , type& R ) noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = A.v[i] << B;
    }

    FORCE_INLINE static void shr(const type& A , int32_t B , type& R ) noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = A.v[i] << B;
    }

    FORCE_INLINE static void set(float A , type& R ) noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = A;
    }

    FORCE_INLINE static void set(float A , float B , float C , float D , type& R ) noexcept
    {
        R = _mm_set_ps( A , B , C , D );
    }

    FORCE_INLINE static void load( const float* value , type& R , auto align_tag )
    {
        memccpy( R.v, value, 0, sizeof(type));
    }
    FORCE_INLINE static void store( const type& value , float* R , auto align_tag )
    {
        memccpy( R, value.v, 0, sizeof(type));
    }

    FORCE_INLINE static void rsqrt( const type& Value , type& R )
    {
        Math::Rsqrt( Value , R );
    }
    FORCE_INLINE static void min(const type& A, const type& B, type& R)
    {
        R = std::min( A , B );
    }
    FORCE_INLINE static void max(const type& A, const type& B, type& R)
    {
        R = std::max( A , B );
    }
    FORCE_INLINE static void cmp_le(const type& A, const type& B, type& R)noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = (A.v[i] <= B.v[i]) ? _details::ALL_BITS_ONE<T> : _details::ALL_BITS_ZERO<T>;
    }
    FORCE_INLINE static void cmp_lt(const type& A, const type& B, type& R)noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = (A.v[i] < B.v[i]) ? _details::ALL_BITS_ONE<T> : _details::ALL_BITS_ZERO<T>;
    }
    FORCE_INLINE static void cmp_ge(const type& A, const type& B, type& R)noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = (A.v[i] >= B.v[i]) ? _details::ALL_BITS_ONE<T> : _details::ALL_BITS_ZERO<T>;
    }
    FORCE_INLINE static void cmp_gt(const type& A, const type& B, type& R)noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = (A.v[i] > B.v[i]) ? _details::ALL_BITS_ONE<T> : _details::ALL_BITS_ZERO<T>;
    }
    FORCE_INLINE static void cmp_eq(const type& A, const type& B, type& R)noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = (A.v[i] == B.v[i]) ? _details::ALL_BITS_ONE<T> : _details::ALL_BITS_ZERO<T>;
    }
    FORCE_INLINE static void cmp_neq(const type& A, const type& B, type& R)noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R.v[i] = (A.v[i] != B.v[i]) ? _details::ALL_BITS_ONE<T> : _details::ALL_BITS_ZERO<T>;
    }

    template< typename T2 >
    FORCE_INLINE static void r_cast(const type& A , simd_data_type<T2,Elements>& R ) noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R[i] = std::bit_cast<T2>( A[i] );
    }

    template< typename T2 >
    FORCE_INLINE static void s_cast(const type& A , simd_data_type<T2,Elements>& R ) noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R[i] = static_cast<T2>( A[i] );
    }

    FORCE_INLINE static void log(const type& A , type & R ) noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R[i] = logf( A[i] );
    }
    FORCE_INLINE static void exp(const type& A , type & R ) noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R[i] = expf( A[i] );
    }
    FORCE_INLINE static void pow(const type& A , const type& B , type & R ) noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R[i] = powf( A[i] , B[i] );
    }
    FORCE_INLINE static void pow(const type& A , float B , type & R ) noexcept
    {
        for( int i = 0; i < Elements; ++i )
            R[i] = powf( A[i] , B );
    }
};

//****************************************************************
// SIMD SSE data type - float
//****************************************************************


template<>
struct simd_type_maping< float , eSimdType::SSE , 4 >
{
    using type = __m128;
    FORCE_INLINE static void add    (const type& A , const type& B, type& R ) noexcept          { R = _mm_add_ps( A , B );                  }
    FORCE_INLINE static void add    (const type& A , float       B, type& R ) noexcept          { R = _mm_add_ps( A , _mm_set1_ps( B ) );   }
    FORCE_INLINE static void sub    (const type& A , const type& B, type& R ) noexcept          { R = _mm_sub_ps( A , B );                  }
    FORCE_INLINE static void sub    (const type& A , float       B, type& R ) noexcept          { R = _mm_sub_ps( A , _mm_set1_ps( B ) );   }
    FORCE_INLINE static void mul    (const type& A , const type& B, type& R ) noexcept          { R = _mm_mul_ps( A , B );                  }
    FORCE_INLINE static void mul    (const type& A , float       B, type& R ) noexcept          { R = _mm_mul_ps( A , _mm_set1_ps( B ) );   }
    FORCE_INLINE static void div    (const type& A , const type& B, type& R ) noexcept          { R = _mm_div_ps( A , B  );                 }
    FORCE_INLINE static void div    (const type& A , float       B, type& R ) noexcept          { R = _mm_div_ps( A , _mm_set1_ps( B ) );   }
    FORCE_INLINE static void min    (const type& A , const type& B, type& R ) noexcept          { R = _mm_min_ps( A , B  );                 }
    FORCE_INLINE static void max    (const type& A , const type& B, type& R ) noexcept          { R = _mm_max_ps( A , B  );                 }
    FORCE_INLINE static void cmp_le (const type& A, const type& B, type& R)noexcept             { R = _mm_cmple_ps( A , B  );               }
    FORCE_INLINE static void cmp_lt (const type& A, const type& B, type& R)noexcept             { R = _mm_cmplt_ps( A , B  );               }
    FORCE_INLINE static void cmp_ge (const type& A, const type& B, type& R)noexcept             { R = _mm_cmpge_ps( A , B  );               }
    FORCE_INLINE static void cmp_gt (const type& A, const type& B, type& R)noexcept             { R = _mm_cmpgt_ps( A , B  );               }
    FORCE_INLINE static void cmp_eq (const type& A, const type& B, type& R)noexcept             { R = _mm_cmpeq_ps( A , B  );               }
    FORCE_INLINE static void cmp_neq(const type& A, const type& B, type& R)noexcept             { R = _mm_cmpneq_ps( A , B  );              }
    FORCE_INLINE static void AND    (const type& A, const type& B, type& R ) noexcept           { R = _mm_and_ps( A , B );                  }
    FORCE_INLINE static void OR     (const type& A, const type& B, type& R ) noexcept           { R = _mm_or_ps ( A , B );                  }
    FORCE_INLINE static void rsqrt  (const type& A , type& R )noexcept                          { R = _mm_rsqrt_ps( A );                    }
    FORCE_INLINE static void set    (float A , type& R ) noexcept                               { R = _mm_set1_ps( A );                     }
    FORCE_INLINE static void set    (float A , float B , float C , float D , type& R ) noexcept { R = _mm_set_ps( D , C , B , A );          }
    FORCE_INLINE static auto r_cast (const type& A, __m128i& R)                                 { R = _mm_castps_si128(A);                  }
    FORCE_INLINE static auto r_cast (const type& A, __m128d& R)                                 { R = _mm_castps_pd(A);                     }
    FORCE_INLINE static auto r_cast (const type& A, __m128& R)                                  { R = A;                                    }
    FORCE_INLINE static void s_cast (const type& A , __m128i& R ) noexcept                      { R = _mm_cvttps_epi32(A);                  }
    FORCE_INLINE static void s_cast (const type& A , __m128d& R ) noexcept                      { R = _mm_cvtps_pd(A);                      }
    FORCE_INLINE static void s_cast (const type& A , __m128 & R ) noexcept                      { R = A;                                    }
    FORCE_INLINE static void log    (const type& A , type & R ) noexcept                        { R = emu_mm_log_ps(A);                     }
    FORCE_INLINE static void exp    (const type& A , type & R ) noexcept                        { R = emu_mm_exp_ps(A);                     }
    FORCE_INLINE static void pow    (const type& A , const type& B , type & R ) noexcept        { R = emu_mm_pow_ps(A,B);                   }
    FORCE_INLINE static void pow    (const type& A , float B , type & R ) noexcept              { R = emu_mm_pow_ps(A,B);                   }

    //FORCE_INLINE static void pow    (const type& A , const type& B, type& R ) noexcept          { R = _mm_getexp_ps (y*log(x)) }
    FORCE_INLINE static void load   (const float* value , type& R , auto align_tag )
    {
        if constexpr( align_tag() >= eDataAlignment::SSE )
            R = _mm_load_ps(value);
        else
            R = _mm_loadu_ps(value);
    }
    FORCE_INLINE static void store( const type& value , float* R , auto align_tag )
    {
        if constexpr( align_tag() >= eDataAlignment::SSE )
            _mm_store_ps( R , value );
        else
            _mm_storeu_ps( R , value );
    }
};

template<>
struct simd_type_maping< float , eSimdType::SSE ,8 >
{
    using type = simd_data_type<__m128,2>;
    FORCE_INLINE static void add    (const type& A , const type& B, type& R ) noexcept          { R.v[0] = _mm_add_ps( A.v[0] , B.v[0] );
                                                                                                  R.v[1] = _mm_add_ps( A.v[1] , B.v[1] );       }
    FORCE_INLINE static void add    (const type& A , float       B, type& R ) noexcept          { auto SB = _mm_set1_ps( B );
                                                                                                  R.v[0] = _mm_add_ps( A.v[0] , SB );
                                                                                                  R.v[1] = _mm_add_ps( A.v[1] , SB );           }
    FORCE_INLINE static void sub    (const type& A , const type& B, type& R ) noexcept          { R.v[0] = _mm_sub_ps( A.v[0] , B.v[0] );
                                                                                                  R.v[1] = _mm_sub_ps( A.v[1] , B.v[1] );       }
    FORCE_INLINE static void sub    (const type& A , float       B, type& R ) noexcept          { auto SB = _mm_set1_ps( B );
                                                                                                  R.v[0] = _mm_sub_ps( A.v[0] , SB );
                                                                                                  R.v[1] = _mm_sub_ps( A.v[1] , SB );           }
    FORCE_INLINE static void mul    (const type& A , const type& B, type& R ) noexcept          { R.v[0] = _mm_mul_ps( A.v[0] , B.v[0] );
                                                                                                  R.v[1] = _mm_mul_ps( A.v[1] , B.v[1] );       }
    FORCE_INLINE static void mul    (const type& A , float       B, type& R ) noexcept          { auto SB = _mm_set1_ps( B );
                                                                                                  R.v[0] = _mm_mul_ps( A.v[0] , SB );
                                                                                                  R.v[1] = _mm_mul_ps( A.v[1] , SB );           }
    FORCE_INLINE static void div    (const type& A , const type& B, type& R ) noexcept          { R.v[0] = _mm_div_ps( A.v[0] , B.v[0] );
                                                                                                  R.v[1] = _mm_div_ps( A.v[1] , B.v[1] );       }
    FORCE_INLINE static void div    (const type& A , float       B, type& R ) noexcept          { auto SB = _mm_set1_ps( B );
                                                                                                  R.v[0] = _mm_div_ps( A.v[0] , SB );
                                                                                                  R.v[1] = _mm_div_ps( A.v[1] , SB );           }
    FORCE_INLINE static void min    (const type& A , const type& B, type& R ) noexcept          { R.v[0] = _mm_min_ps( A.v[0] , B.v[0] );
                                                                                                  R.v[1] = _mm_min_ps( A.v[1] , B.v[1] );       }
    FORCE_INLINE static void max    (const type& A , const type& B, type& R ) noexcept          { R.v[0] = _mm_max_ps( A.v[0] , B.v[0] );
                                                                                                  R.v[1] = _mm_max_ps( A.v[1] , B.v[1] );       }
    FORCE_INLINE static void rsqrt  (const type& A , type& R )noexcept                          { R.v[0] = _mm_rsqrt_ps( A.v[0] );
                                                                                                  R.v[1] = _mm_rsqrt_ps( A.v[1] );              }
    FORCE_INLINE static void set    ( float A , type& R ) noexcept                              { R.v[0] = _mm_set1_ps( A );
                                                                                                  R.v[1] = R.v[0];                              }
    FORCE_INLINE static void set    ( float A , float B , float C , float D , float A2 , float B2 , float C2 , float D2 , type& R ) noexcept{
                                                                                                  R.v[0] = _mm_set_ps( D  , C  , B  , A  );
                                                                                                  R.v[1] = _mm_set_ps( D2 , C2 , B2 , A2 );     }
    FORCE_INLINE static auto r_cast (const type& A, simd_data_type<__m128i,2>& R)               { R[0] = _mm_castps_si128(A[0]);
                                                                                                  R[1] = _mm_castps_si128(A[1]);                }
    FORCE_INLINE static auto r_cast (const type& A, simd_data_type<__m128d,2>& R)               { R[0] = _mm_castps_pd(A[0]);
                                                                                                  R[1] = _mm_castps_pd(A[1]);                   }
    FORCE_INLINE static auto r_cast (const type& A, simd_data_type<__m128 ,2>& R)               { R = A;                                        }
    FORCE_INLINE static void s_cast (const type& A, simd_data_type<__m128i,2>& R)noexcept       { R[0] = _mm_cvttps_epi32(A[0]);
                                                                                                  R[1] = _mm_cvttps_epi32(A[1]);                }
    FORCE_INLINE static void s_cast (const type& A, simd_data_type<__m128d,2>& R)noexcept       { R[0] = _mm_cvtps_pd(A[0]);
                                                                                                  R[1] = _mm_cvtps_pd(A[1]);                    }
    FORCE_INLINE static void s_cast (const type& A, simd_data_type<__m128 ,2>& R)noexcept       { R = A;                                        }

    FORCE_INLINE static void load( const float* value , type& R , auto align_tag )
    {
        if constexpr( align_tag() >= eDataAlignment::SSE )
        {
            R.v[0] = _mm_load_ps(value + 0);
            R.v[1] = _mm_load_ps(value + 4);
        }
        else
        {
            R.v[0] = _mm_loadu_ps(value + 0);
            R.v[1] = _mm_loadu_ps(value + 4);
        }
    }
    FORCE_INLINE static void store( const type& value , float* R , auto align_tag )
    {
        if constexpr( align_tag() >= eDataAlignment::SSE )
        {
            _mm_store_ps( R + 0 , value.v[0] );
            _mm_store_ps( R + 4 , value.v[1] );
        }
        else
        {
            _mm_storeu_ps( R + 0 , value.v[0] );
            _mm_storeu_ps( R + 4 , value.v[1] );
        }
    }
};

//****************************************************************
// SIMD SSE data type - double
//****************************************************************

template<>
struct simd_type_maping< double , eSimdType::SSE , 2 >
{
    using type = __m128d;
    FORCE_INLINE static void add    (const type& A , const type& B, type& R ) noexcept  { R = _mm_add_pd( A , B );                  }
    FORCE_INLINE static void add    (const type& A , double B     , type& R ) noexcept  { R = _mm_add_pd( A , _mm_set1_pd( B ) );   }
    FORCE_INLINE static void sub    (const type& A , const type& B, type& R ) noexcept  { R = _mm_sub_pd( A , B );                  }
    FORCE_INLINE static void sub    (const type& A , double B     , type& R ) noexcept  { R = _mm_sub_pd( A , _mm_set1_pd( B ) );   }
    FORCE_INLINE static void mul    (const type& A , const type& B, type& R ) noexcept  { R = _mm_mul_pd( A , B );                  }
    FORCE_INLINE static void mul    (const type& A , double B     , type& R ) noexcept  { R = _mm_mul_pd( A , _mm_set1_pd( B ) );   }
    FORCE_INLINE static void div    (const type& A , const type& B, type& R ) noexcept  { R = _mm_div_pd( A , B  );                 }
    FORCE_INLINE static void div    (const type& A , double B     , type& R ) noexcept  { R = _mm_div_pd( A , _mm_set1_pd( B ) );   }
    FORCE_INLINE static void min    (const type& A , const type& B, type& R ) noexcept  { R = _mm_min_pd( A , B );                  }
    FORCE_INLINE static void max    (const type& A , const type& B, type& R ) noexcept  { R = _mm_max_pd( A , B );                  }
    FORCE_INLINE static void cmp_le (const type& A, const type& B, type& R)noexcept     { R =  _mm_cmple_pd( A , B  );              }
    FORCE_INLINE static void cmp_lt (const type& A, const type& B, type& R)noexcept     { R =  _mm_cmplt_pd( A , B  );              }
    FORCE_INLINE static void cmp_ge (const type& A, const type& B, type& R)noexcept     { R =  _mm_cmpge_pd( A , B  );              }
    FORCE_INLINE static void cmp_gt (const type& A, const type& B, type& R)noexcept     { R =  _mm_cmpgt_pd( A , B  );              }
    FORCE_INLINE static void cmp_eq (const type& A, const type& B, type& R)noexcept     { R =  _mm_cmpeq_pd( A , B  );              }
    FORCE_INLINE static void cmp_neq(const type& A, const type& B, type& R)noexcept     { R = _mm_cmpneq_pd( A , B  );              }
    FORCE_INLINE static void set    (double A , type& R ) noexcept                      { R = _mm_set1_pd( A );                     }
    FORCE_INLINE static void set    (double A , double B , type& R ) noexcept           { R = _mm_set_pd( B , A );                  }
    FORCE_INLINE static auto r_cast (const type& A, __m128i& R)                         { R = _mm_castpd_si128(A);                  }
    FORCE_INLINE static auto r_cast (const type& A, __m128d& R)                         { R = A;                                    }
    FORCE_INLINE static auto r_cast (const type& A, __m128& R)                          { R = _mm_castpd_ps(A);                     }
    FORCE_INLINE static void s_cast (const type& A , __m128i& R ) noexcept              { R = _mm_cvtpd_epi32(A);                   }
    FORCE_INLINE static void s_cast (const type& A , __m128d& R ) noexcept              { R = A;                                    }
    FORCE_INLINE static void s_cast (const type& A , __m128 & R ) noexcept              { R = _mm_cvtpd_ps(A);                      }

    FORCE_INLINE static void load( const double* value , type& R , auto align_tag )
    {
        if constexpr( align_tag() >= eDataAlignment::SSE )
            R = _mm_load_pd(value);
        else
            R = _mm_loadu_pd(value);
    }
    FORCE_INLINE static void store( const type& value , double* R , auto align_tag )
    {
        if constexpr( align_tag() >= eDataAlignment::SSE )
            _mm_store_pd( R , value );
        else
            _mm_storeu_pd( R , value );
    }
};

//****************************************************************
// SIMD SSE data type - int32_t
//****************************************************************

template<>
struct simd_type_maping< int32_t , eSimdType::SSE , 4 >
{
    using type = __m128i;
    FORCE_INLINE static void add    (const type& A, const type& B, type& R ) noexcept                   { R = _mm_add_epi32( A , B );                   }
    FORCE_INLINE static void add    (const type& A, int32_t B    , type& R ) noexcept                   { R = _mm_add_epi32( A , _mm_set1_epi32( B ) ); }
    FORCE_INLINE static void sub    (const type& A, const type& B, type& R ) noexcept                   { R = _mm_sub_epi32( A , B );                   }
    FORCE_INLINE static void sub    (const type& A, int32_t B    , type& R ) noexcept                   { R = _mm_sub_epi32( A , _mm_set1_epi32( B ) ); }
    FORCE_INLINE static void mul    (const type& A, const type& B, type& R ) noexcept                   { R = _mm_mul_epi32( A , B );                   }
    FORCE_INLINE static void mul    (const type& A, int32_t B    , type& R ) noexcept                   { R = _mm_mul_epi32( A , _mm_set1_epi32( B ) ); }
    FORCE_INLINE static void min    (const type& A , const type& B, type& R ) noexcept                  { R = _mm_min_epi32( A , B  );                  }
    FORCE_INLINE static void max    (const type& A , const type& B, type& R ) noexcept                  { R = _mm_max_epi32( A , B  );                  }
    FORCE_INLINE static void cmp_le (const type& A, const type& B, type& R)noexcept                 { R =  _mm_cmpeq_epi32( _mm_cmpgt_epi32( B , A  ) , _mm_setzero_si128() ); }
    FORCE_INLINE static void cmp_lt (const type& A, const type& B, type& R)noexcept                 { R =  _mm_cmplt_epi32( A , B  );               }
    FORCE_INLINE static void cmp_ge (const type& A, const type& B, type& R)noexcept                 { R =  _mm_cmpeq_epi32( _mm_cmplt_epi32( B , A  ) , _mm_setzero_si128() ); }
    FORCE_INLINE static void cmp_gt (const type& A, const type& B, type& R)noexcept                 { R =  _mm_cmpgt_epi32( A , B  );               }
    FORCE_INLINE static void cmp_eq (const type& A, const type& B, type& R)noexcept                 { R =  _mm_cmpeq_epi32( A , B  );               }
    FORCE_INLINE static void cmp_neq(const type& A, const type& B, type& R)noexcept                 { R =  _mm_cmpeq_epi32( _mm_cmpeq_epi32( A , B ) , _mm_setzero_si128() ); }
    FORCE_INLINE static void AND    (const type& A, const type& B, type& R ) noexcept                   { R = _mm_and_epi32( A , B );                   }
    FORCE_INLINE static void OR     (const type& A, const type& B, type& R ) noexcept                   { R = _mm_or_epi32 ( A , B );                   }
    FORCE_INLINE static void NOT    (const type& A, type& R                ) noexcept                   { R = _mm_xor_epi32 ( A , _mm_set1_epi32(-1) ); }
    FORCE_INLINE static void shl    (const type& A, int32_t B    , type& R ) noexcept                   { R = _mm_slli_epi32( A , B );                  }
    FORCE_INLINE static void shr    (const type& A, int32_t B    , type& R ) noexcept                   { R = _mm_srai_epi32( A , B );                  }
    FORCE_INLINE static void set    (int32_t A , type& R ) noexcept                                     { R = _mm_set1_epi32( A );                      }
    FORCE_INLINE static void set    (int32_t A , int32_t B , int32_t C , int32_t D , type& R ) noexcept { R = _mm_set_epi32( D , C , B , A );           }
    FORCE_INLINE static auto r_cast (const type& A, __m128i& R)                                         { R = A;                                        }
    FORCE_INLINE static auto r_cast (const type& A, __m128d& R)                                         { R = _mm_castsi128_pd(A);                      }
    FORCE_INLINE static auto r_cast (const type& A, __m128& R)                                          { R = _mm_castsi128_ps(A);                      }
    FORCE_INLINE static void s_cast (const type& A , __m128i& R ) noexcept                              { R = A;                                        }
    FORCE_INLINE static void s_cast (const type& A , __m128d& R ) noexcept                              { R = _mm_cvtepi32_pd(A);                       }
    FORCE_INLINE static void s_cast (const type& A , __m128 & R ) noexcept                              { R = _mm_cvtepi32_ps(A);                       }

    FORCE_INLINE static void load( const int32_t* value , type& R , auto align_tag )
    {
        if constexpr( align_tag() >= eDataAlignment::SSE )
            R = _mm_load_si128( (__m128i*)value);
        else
            R = _mm_loadu_epi32(value);
    }
    FORCE_INLINE static void store( const type& value , int32_t* R , auto align_tag )
    {
        if constexpr( align_tag() >= eDataAlignment::SSE )
            _mm_store_si128( (__m128i*)R , value );
        else
            _mm_storeu_epi32( R , value );
    }
};

template<>
struct simd_type_maping< int32_t , eSimdType::SSE , 8 >
{
    using type = simd_data_type<__m128i,2>;
    FORCE_INLINE static void add(const type& A , const type& B, type& R ) noexcept          { R.v[0]  = _mm_add_epi32( A.v[0] , B.v[0] );
                                                                                              R.v[1]  = _mm_add_epi32( A.v[1] , B.v[1] );       }
    FORCE_INLINE static void add(const type& A , int32_t     B, type& R ) noexcept          { auto SB = _mm_set1_epi32( B );
                                                                                              R.v[0]  = _mm_add_epi32( A.v[0] , SB );
                                                                                              R.v[1]  = _mm_add_epi32( A.v[1] , SB );           }
    FORCE_INLINE static void sub(const type& A , const type& B, type& R ) noexcept          { R.v[0]  = _mm_sub_epi32( A.v[0] , B.v[0] );
                                                                                              R.v[1]  = _mm_sub_epi32( A.v[1] , B.v[1] );       }
    FORCE_INLINE static void sub(const type& A , int32_t     B, type& R ) noexcept          { auto SB = _mm_set1_epi32( B );
                                                                                              R.v[0]  = _mm_sub_epi32( A.v[0] , SB );
                                                                                              R.v[1]  = _mm_sub_epi32( A.v[1] , SB );           }
    FORCE_INLINE static void mul(const type& A , const type& B, type& R ) noexcept          { R.v[0]  = _mm_mul_epi32( A.v[0] , B.v[0] );
                                                                                              R.v[1]  = _mm_mul_epi32( A.v[1] , B.v[1] );       }
    FORCE_INLINE static void mul(const type& A , int32_t     B, type& R ) noexcept          { auto SB = _mm_set1_epi32( B );
                                                                                              R.v[0]  = _mm_mul_epi32( A.v[0] , SB );
                                                                                              R.v[1]  = _mm_mul_epi32( A.v[1] , SB );           }
    FORCE_INLINE static void min    (const type& A , const type& B, type& R ) noexcept      { R.v[0]  = _mm_min_epi32( A.v[0] , B.v[0] );
                                                                                              R.v[1]  = _mm_min_epi32( A.v[1] , B.v[1] );       }
    FORCE_INLINE static void max    (const type& A , const type& B, type& R ) noexcept      { R.v[0]  = _mm_max_epi32( A.v[0] , B.v[0] );
                                                                                              R.v[1]  = _mm_max_epi32( A.v[1] , B.v[1] );       }

    FORCE_INLINE static void cmp_le (const type& A, const type& B, type& R)noexcept         { auto ZERO = _mm_setzero_si128();
                                                                                              R.v[0]  = _mm_cmpeq_epi32( _mm_cmpgt_epi32( B.v[0] , A.v[0] ) , ZERO );
                                                                                              R.v[1]  = _mm_cmpeq_epi32( _mm_cmpgt_epi32( B.v[1] , A.v[1] ) , ZERO );  }
    FORCE_INLINE static void cmp_lt (const type& A, const type& B, type& R)noexcept         { R.v[0]  = _mm_cmplt_epi32( A.v[0] , B.v[0]  );
                                                                                              R.v[1]  = _mm_cmplt_epi32( A.v[1] , B.v[1]  );               }
    FORCE_INLINE static void cmp_ge (const type& A, const type& B, type& R)noexcept         { auto ZERO = _mm_setzero_si128();
                                                                                              R.v[0] =  _mm_cmpeq_epi32( _mm_cmplt_epi32( B.v[0] , A.v[0] ) , ZERO );
                                                                                              R.v[1] =  _mm_cmpeq_epi32( _mm_cmplt_epi32( B.v[1] , A.v[1] ) , ZERO ); }
    FORCE_INLINE static void cmp_gt (const type& A, const type& B, type& R)noexcept         { R.v[0]  = _mm_cmpgt_epi32( A.v[0] , B.v[0] );
                                                                                              R.v[1]  = _mm_cmpgt_epi32( A.v[1] , B.v[1] );                 }
    FORCE_INLINE static void cmp_eq (const type& A, const type& B, type& R)noexcept         { R.v[0]  = _mm_cmpeq_epi32( A.v[0] , B.v[0] );
                                                                                              R.v[1]  = _mm_cmpeq_epi32( A.v[1] , B.v[1] );                  }
    FORCE_INLINE static void cmp_neq(const type& A, const type& B, type& R)noexcept         { auto ONE = _mm_set1_epi32(-1);
                                                                                              R.v[0]  = _mm_xor_epi32( _mm_cmpeq_epi32( A.v[0] , B.v[0] ) , ONE );
                                                                                              R.v[1]  = _mm_xor_epi32( _mm_cmpeq_epi32( A.v[1] , B.v[1] ) , ONE ); }
    FORCE_INLINE static void AND(const type& A, const type& B, type& R ) noexcept           { R.v[0]  = _mm_and_epi32( A.v[0] , B.v[0] );
                                                                                              R.v[1]  = _mm_and_epi32( A.v[1] , B.v[1] );       }
    FORCE_INLINE static void OR (const type& A, const type& B, type& R ) noexcept           { R.v[0]  = _mm_or_epi32( A.v[0] , B.v[0] );
                                                                                              R.v[1]  = _mm_or_epi32( A.v[1] , B.v[1] );        }
    FORCE_INLINE static void NOT(const type& A, type& R                ) noexcept           { auto ONE = _mm_set1_epi32(-1);
                                                                                              R.v[0] = _mm_xor_epi32 ( A.v[0] , ONE );
                                                                                              R.v[1] = _mm_xor_epi32 ( A.v[1] , ONE );          }
    FORCE_INLINE static void set( int32_t A , type& R ) noexcept                            { R.v[0]  = _mm_set1_epi32( A );
                                                                                              R.v[1]  = R.v[0];                                 }
    FORCE_INLINE static void shl(const type& A , int32_t B    , type& R ) noexcept          { R.v[0]  = _mm_slli_epi32( A.v[0] , B );
                                                                                              R.v[1]  = _mm_slli_epi32( A.v[1] , B );           }
    FORCE_INLINE static void shr(const type& A , int32_t B    , type& R ) noexcept          { R.v[0]  = _mm_srai_epi32( A.v[0] , B );
                                                                                              R.v[1]  = _mm_srai_epi32( A.v[1] , B );           }
    FORCE_INLINE static void set( int32_t A , int32_t B , int32_t C , int32_t D , int32_t A2 , int32_t B2 , int32_t C2 , int32_t D2 , type& R ) noexcept{
                                                                                              R.v[0]  = _mm_set_epi32( D  , C  , B  , A  );
                                                                                              R.v[1]  = _mm_set_epi32( D2 , C2 , B2 , A2 );     }
    FORCE_INLINE static auto r_cast (const type& A, simd_data_type<__m128i,2>& R)           { R = A;                                            }
    FORCE_INLINE static auto r_cast (const type& A, simd_data_type<__m128d,2>& R)           { R[0] = _mm_castsi128_pd(A[0]);
                                                                                              R[1] = _mm_castsi128_pd(A[1]);                    }
    FORCE_INLINE static auto r_cast (const type& A, simd_data_type<__m128 ,2>& R)           { R[0] = _mm_castsi128_ps(A[0]);
                                                                                              R[1] = _mm_castsi128_ps(A[1]);                    }
    FORCE_INLINE static void s_cast (const type& A, simd_data_type<__m128i,2>& R)noexcept   { R = A;                                            }
    FORCE_INLINE static void s_cast (const type& A, simd_data_type<__m128d,2>& R)noexcept   { R[0] = _mm_cvtepi32_pd(A[0]);
                                                                                              R[0] = _mm_cvtepi32_pd(A[0]);                     }
    FORCE_INLINE static void s_cast (const type& A, simd_data_type<__m128 ,2>& R)noexcept   { R[0] = _mm_cvtepi32_ps(A[0]);
                                                                                              R[0] = _mm_cvtepi32_ps(A[0]);                     }

    FORCE_INLINE static void load( const int32_t* value , type& R , auto align_tag )
    {
        if constexpr( align_tag() >= eDataAlignment::SSE )
        {
            R.v[0] = _mm_load_si128((__m128i*)value + 0);
            R.v[1] = _mm_load_si128((__m128i*)value + 1);
        }
        else
        {
            R.v[0] = _mm_loadu_epi32(value + 0);
            R.v[1] = _mm_loadu_epi32(value + 4);
        }
    }
    FORCE_INLINE static void store( const type& value , int32_t* R , auto align_tag )
    {
        if constexpr( align_tag() >= eDataAlignment::SSE )
        {
            _mm_store_si128( (__m128i*)R + 0 , value.v[0] );
            _mm_store_si128( (__m128i*)R + 1 , value.v[1] );
        }
        else
        {
            _mm_storeu_epi32( R + 0 , value.v[0] );
            _mm_storeu_epi32( R + 4 , value.v[1] );
        }
    }
};

//****************************************************************
// SIMD AVX data type - float
//****************************************************************

template<>
struct simd_type_maping< float , eSimdType::AVX , 8 >
{
    using type = __m256;
    FORCE_INLINE static void add    (const type& A , const type& B, type& R ) noexcept          { R = _mm256_add_ps( A , B );                   }
    FORCE_INLINE static void add    (const type& A , float B      , type& R ) noexcept          { R = _mm256_add_ps( A , _mm256_set1_ps( B ) ); }
    FORCE_INLINE static void sub    (const type& A , const type& B, type& R ) noexcept          { R = _mm256_sub_ps( A , B );                   }
    FORCE_INLINE static void sub    (const type& A , float B      , type& R ) noexcept          { R = _mm256_sub_ps( A , _mm256_set1_ps( B ) ); }
    FORCE_INLINE static void mul    (const type& A , const type& B, type& R ) noexcept          { R = _mm256_mul_ps( A , B );                   }
    FORCE_INLINE static void mul    (const type& A , float B      , type& R ) noexcept          { R = _mm256_mul_ps( A , _mm256_set1_ps( B ) ); }
    FORCE_INLINE static void div    (const type& A , const type& B, type& R ) noexcept          { R = _mm256_div_ps( A , B  );                  }
    FORCE_INLINE static void div    (const type& A , float B      , type& R ) noexcept          { R = _mm256_div_ps( A , _mm256_set1_ps( B ) ); }
    FORCE_INLINE static void min    (const type& A , const type& B, type& R ) noexcept          { R = _mm256_min_ps( A , B  );                  }
    FORCE_INLINE static void max    (const type& A , const type& B, type& R ) noexcept          { R = _mm256_max_ps( A , B  );                  }
    FORCE_INLINE static void cmp_le (const type& A, const type& B, type& R)noexcept             { R =  _mm256_cmp_ps( A , B , _CMP_LE_OQ );     }
    FORCE_INLINE static void cmp_lt (const type& A, const type& B, type& R)noexcept             { R =  _mm256_cmp_ps( A , B , _CMP_LT_OQ );     }
    FORCE_INLINE static void cmp_ge (const type& A, const type& B, type& R)noexcept             { R =  _mm256_cmp_ps( A , B , _CMP_GE_OQ );     }
    FORCE_INLINE static void cmp_gt (const type& A, const type& B, type& R)noexcept             { R =  _mm256_cmp_ps( A , B , _CMP_GT_OQ );     }
    FORCE_INLINE static void cmp_eq (const type& A, const type& B, type& R)noexcept             { R =  _mm256_cmp_ps( A , B , _CMP_EQ_OQ );     }
    FORCE_INLINE static void cmp_neq(const type& A, const type& B, type& R)noexcept             { R =  _mm256_cmp_ps( A , B , _CMP_NEQ_OQ );    }
    FINAVX2      static void AND    (const type& A, const type& B, type& R ) noexcept           { R = _mm256_and_ps ( A , B );                  }
    FORCE_INLINE static void OR     (const type& A, const type& B, type& R ) noexcept           { R = _mm256_or_ps  ( A , B );                  }
    FORCE_INLINE static void rsqrt  (const type& A , type& R )noexcept                          { R = _mm256_rsqrt_ps( A );                     }
    FORCE_INLINE static void set    (float A , type& R ) noexcept                               { R = _mm256_set1_ps( A );                      }
    FORCE_INLINE static void set    (float A , float B , float C , float D , float A2 , float B2 , float C2 , float D2 , type& R ) noexcept { R = _mm256_set_ps( D2 , C2 , B2 , A2 , D , C , B , A ); }
    FORCE_INLINE static void log    (const type& A , type & R ) noexcept                        { R = emu_mm256_log_ps(A);                      }
    FORCE_INLINE static void exp    (const type& A , type & R ) noexcept                        { R = emu_mm256_exp_ps(A);                      }
    FORCE_INLINE static void pow    (const type& A , const type& B , type & R ) noexcept        { R = emu_mm256_pow_ps(A,B);                    }
    FORCE_INLINE static void pow    (const type& A , float B , type & R ) noexcept              { R = emu_mm256_pow_ps(A,B);                    }

    FORCE_INLINE static auto r_cast (const type& A, __m256i& R)                                 { R = _mm256_castps_si256(A);                   }
    FORCE_INLINE static auto r_cast (const type& A, __m256d& R)                                 { R = _mm256_castps_pd(A);                      }
    FORCE_INLINE static auto r_cast (const type& A, __m256 & R)                                 { R = A;                                        }
    FORCE_INLINE static void s_cast (const type& A, __m256i& R) noexcept                        { R = _mm256_cvttps_epi32(A);                   }
  //FORCE_INLINE static void s_cast (const type& A, __m256d& R) noexcept                        { R = _mm256_cvtps_pd(A);                       }
    FORCE_INLINE static void s_cast (const type& A, __m256 & R) noexcept                        { R = A;                                        }

    FORCE_INLINE static void load   ( const float* value , type& R , auto align_tag )
    {
        if constexpr( align_tag() >= eDataAlignment::AVX )
            R = _mm256_load_ps(value);
        else
            R = _mm256_loadu_ps(value);
    }
    template< eDataAlignment A >
    FORCE_INLINE static void store( const type& value , float* R , DataAlignmentTagT<A> align_tag )
    {
        if constexpr( A >= eDataAlignment::AVX )
            _mm256_store_ps( R , value );
        else
            _mm256_storeu_ps( R , value );
    }
    FORCE_INLINE static void store( const type& value , float* R , __m256i Mask )
    {
        _mm256_maskstore_ps( R , Mask ,  value );
    }
};

template<>
struct simd_type_maping< int32_t , eSimdType::AVX , 8 >
{
    using type = __m256i;
    FINAVX2      static void add    (const type& A, const type& B, type& R ) noexcept                   { R = _mm256_add_epi32( A , B );                        }
    FINAVX2      static void add    (const type& A, int32_t B    , type& R ) noexcept                   { R = _mm256_add_epi32( A , _mm256_set1_epi32( B ) );   }
    FINAVX2      static void sub    (const type& A, const type& B, type& R ) noexcept                   { R = _mm256_sub_epi32( A , B );                        }
    FINAVX2      static void sub    (const type& A, int32_t B    , type& R ) noexcept                   { R = _mm256_sub_epi32( A , _mm256_set1_epi32( B ) );   }
    FINAVX2      static void mul    (const type& A, const type& B, type& R ) noexcept                   { R = _mm256_mullo_epi32( A , B );                      }
    FINAVX2      static void mul    (const type& A, int32_t B    , type& R ) noexcept                   { R = _mm256_mullo_epi32( A , _mm256_set1_epi32( B ) ); }
    FORCE_INLINE static void min    (const type& A ,const type& B, type& R ) noexcept                   { R = _mm256_min_epi32( A , B  );                       }
    FORCE_INLINE static void max    (const type& A ,const type& B, type& R ) noexcept                   { R = _mm256_max_epi32( A , B  );                       }
    FORCE_INLINE static void cmp_le (const type& A, const type& B, type& R)noexcept                     { R = _mm256_xor_epi32( _mm256_cmpgt_epi32( A , B ) , _mm256_set1_epi32(-1) ); }
    FORCE_INLINE static void cmp_lt (const type& A, const type& B, type& R)noexcept                     { R = _mm256_cmpgt_epi32( B , A  );              }
    FORCE_INLINE static void cmp_ge (const type& A, const type& B, type& R)noexcept                     { R = _mm256_andnot_si256(_mm256_cmpgt_epi32( B , A ), _mm256_set1_epi32(-1)); };
    FORCE_INLINE static void cmp_gt (const type& A, const type& B, type& R)noexcept                     { R = _mm256_cmpgt_epi32( A , B  );              }
    FORCE_INLINE static void cmp_eq (const type& A, const type& B, type& R)noexcept                     { R = _mm256_cmpeq_epi32( A , B  );              }
    FORCE_INLINE static void cmp_neq(const type& A, const type& B, type& R)noexcept                     { R = _mm256_cmpeq_epi32( _mm256_cmpeq_epi32( A , B ) , _mm256_setzero_si256() ); }
    FINAVX2      static void AND    (const type& A, const type& B, type& R ) noexcept                   { R = _mm256_and_si256 ( A , B );                       }
    FORCE_INLINE static void OR     (const type& A, const type& B, type& R ) noexcept                   { R = _mm256_or_si256 ( A , B );                        }
    FORCE_INLINE static void shl    (const type& A, int32_t B    , type& R ) noexcept                   { R = _mm256_slli_epi32( A , B );                       }
    FORCE_INLINE static void shr    (const type& A, int32_t B    , type& R ) noexcept                   { R = _mm256_srai_epi32( A , B );                       }
    FORCE_INLINE static void set    (int32_t A , type& R ) noexcept                                     { R = _mm256_set1_epi32( A );                           }
    FORCE_INLINE static void set    (int32_t A , int32_t B , int32_t C , int32_t D , int32_t A2 , int32_t B2 , int32_t C2 , int32_t D2 , type& R ) noexcept { R = _mm256_set_epi32( D2 , C2 , B2 , A2 , D , C , B , A ); }

    FORCE_INLINE static auto r_cast (const type& A, __m256i& R)                                         { R = A;                                                }
    FORCE_INLINE static auto r_cast (const type& A, __m256d& R)                                         { R = _mm256_castsi256_pd(A);                           }
    FORCE_INLINE static auto r_cast (const type& A, __m256 & R)                                         { R = _mm256_castsi256_ps(A);                           }
    FORCE_INLINE static void s_cast (const type& A, __m256i& R) noexcept                                { R = A;                                                }
  //FORCE_INLINE static void s_cast (const type& A, __m256d& R) noexcept                                { R = _mm256_cvtepi32_pd(A);                            }
    FORCE_INLINE static void s_cast (const type& A, __m256 & R) noexcept                                { R = _mm256_cvtepi32_ps(A);                            }

    FORCE_INLINE static void load   ( const int32_t* value , type& R , auto align_tag )
    {
        if constexpr( align_tag() >= eDataAlignment::SSE )
            R = _mm256_load_si256((__m256i*)value);
        else
            R = _mm256_loadu_epi32(value);
    }
    FORCE_INLINE static void store( const type& value , int32_t* R , auto align_tag )
    {
        if constexpr( align_tag() >= eDataAlignment::SSE )
            _mm256_store_si256( (__m256i*)R , value );
        else
            _mm256_storeu_si256( (__m256i*)R , value );
    }
};

//****************************************************************
// SIMD implementation
//****************************************************************

template< typename T , int Elements , eSimdType Simd = eSimdType::None >
class simd_impl
{
    using sse_map_t = simd_type_maping<T,Simd,Elements>;
    using sse_t = typename sse_map_t::type;
public:
    using value_type = T;

    FORCE_INLINE simd_impl() = default;
    FORCE_INLINE simd_impl(const sse_t& v)  noexcept
        : v(v)
    {}
    template< typename T2 >
    simd_impl(const simd_impl<T2,Elements,Simd>& value )  noexcept
        : simd_impl( value.static_cast_to<T>() )
    {}
    template< typename T2 >
    FORCE_INLINE simd_impl<T2,Elements,Simd> static_cast_to()const noexcept
    {
        simd_impl<T2,Elements,Simd> Result;
        sse_map_t::s_cast( v , Result.v );
        return Result;
    }
    template< typename T2 >
    FORCE_INLINE simd_impl<T2,Elements,Simd> reinterpret_cast_to()const noexcept
    {
        simd_impl<T2,Elements,Simd> Result;
        sse_map_t::r_cast( v , Result.v );
        return Result;
    }
    FORCE_INLINE simd_impl(T x, T y, T z, T w) noexcept requires(std::is_same_v<T,T> && Elements==4)
    {
        sse_map_t::set( x , y , z , w , v );
    }
    FORCE_INLINE simd_impl(T x, T y, T z, T w , T x2, T y2, T z2, T w2) noexcept requires(std::is_same_v<T,T> && Elements==8)
    {
        sse_map_t::set( x , y , z , w , x2 , y2 , z2 , w2 , v );
    }
    FORCE_INLINE simd_impl(double x, double y) noexcept requires(std::is_same_v<T,double>)
    {
        sse_map_t::set( x , y , v );
    }
    FORCE_INLINE simd_impl(T value) noexcept requires(std::is_same_v<T,T>)
    {
        sse_map_t::set( value , v );
    }
    FORCE_INLINE simd_impl(double value) noexcept requires(std::is_same_v<T,double>)
    {
        sse_map_t::set( value , v );
    }

    template< eDataAlignment Alignment = eDataAlignment::None >
    FORCE_INLINE simd_impl(const T* value , DataAlignmentTagT<Alignment> t = {} ) requires(std::is_same_v<T,T>)
    {
        sse_map_t::load( value, v , t );
    }

    template< eDataAlignment Alignment = eDataAlignment::None >
    FORCE_INLINE static simd_impl construct_from_array(uint32_t index , const T*& pArray , DataAlignmentTagT<Alignment> t = {} ) requires(std::is_same_v<T,T>)
    {
        simd_impl result;
        result.load_from_array( index , pArray , t );
        return result;
    }

    FORCE_INLINE void set( T x, T y, T z, T w ) noexcept requires(std::is_same_v<T,T>)
    {
        sse_map_t::set( x , y , z , w , v );
    }
    FORCE_INLINE void set( double x, double y ) noexcept requires(std::is_same_v<T,double>)
    {
        sse_map_t::set( x , y , v );
    }
    FORCE_INLINE void set( T value ) noexcept requires(std::is_same_v<T,T>)
    {
        sse_map_t::set( value , v );
    }
    FORCE_INLINE void set( double value ) noexcept requires(std::is_same_v<T,double>)
    {
        sse_map_t::set( value , v );
    }
    FORCE_INLINE void set( const sse_t& value ) noexcept
    {
        v = value;
    }
    template< eDataAlignment Alignment = eDataAlignment::None >
    FORCE_INLINE void load(const T* value , DataAlignmentTagT<Alignment> t = {} ) requires(std::is_same_v<T,T>)
    {
        sse_map_t::load( value , v , t );
    }

    template< eDataAlignment Alignment = eDataAlignment::None >
    FORCE_INLINE void load_from_array(uint32_t Index, const T*& pArray , DataAlignmentTagT<Alignment> t = {} ) requires(std::is_same_v<T,T>)
    {
        sse_map_t::load( pArray , v , t );
        pArray += Elements;
    }

    template< eDataAlignment Alignment = eDataAlignment::None >
    FORCE_INLINE void store(T* value , DataAlignmentTagT<Alignment> t = {} ) const requires(std::is_same_v<T,T>)
    {
        sse_map_t::store( v , value , t );
    }

    template< eDataAlignment Alignment = eDataAlignment::None >
    FORCE_INLINE void store_to_array(uint32_t Index, T*& pArray , DataAlignmentTagT<Alignment> t = {} ) const requires(std::is_same_v<T,T>)
    {
        sse_map_t::store( v , pArray , t );
        pArray += Elements;
    }

    FORCE_INLINE void store_to_array(uint32_t Index, T*& pArray , const simd_impl<int,Elements,Simd>& Mask ) const requires(std::is_same_v<T,T>)
    {
        sse_map_t::store( v , pArray , Mask.v );
        pArray += Elements;
    }

    FORCE_INLINE simd_impl rsqrt()const noexcept
    {
        simd_impl result;
        sse_map_t::rsqrt(v, result.v);
        return result;
    }
    FORCE_INLINE simd_impl min( const simd_impl& value )const noexcept
    {
        simd_impl result;
        sse_map_t::min(v, value.v, result.v);
        return result;
    }
    FORCE_INLINE simd_impl max( const simd_impl& value )const noexcept
    {
        simd_impl result;
        sse_map_t::max(v, value.v, result.v);
        return result;
    }
    FORCE_INLINE simd_impl clamp( const simd_impl& min , const simd_impl& max )const noexcept
    {
        simd_impl result;
        sse_map_t::max(v, min.v, result.v);
        sse_map_t::min(v, max.v, result.v);
        return result;
    }
    FORCE_INLINE simd_impl operator+(const simd_impl& other) const noexcept
    {
        simd_impl result;
        sse_map_t::add(v, other.v, result.v);
        return result;
    }
    FORCE_INLINE simd_impl& operator+=(const simd_impl& other) noexcept
    {
        sse_map_t::add(v, other.v, v);
        return *this;
    }
    FORCE_INLINE simd_impl operator-(const simd_impl& other) const noexcept
    {
        simd_impl result;
        sse_map_t::sub(v, other.v, result.v);
        return result;
    }
    FORCE_INLINE simd_impl& operator-=(const simd_impl& other) noexcept
    {
        sse_map_t::sub(v, other.v, v);
        return *this;
    }
    FORCE_INLINE simd_impl operator*(const simd_impl& other) const noexcept
    {
        simd_impl result;
        sse_map_t::mul(v, other.v, result.v);
        return result;
    }

    FORCE_INLINE simd_impl operator<(const simd_impl& other)const noexcept
    {
        simd_impl result;
        sse_map_t::cmp_lt(v, other.v, result.v);
        return result;
    }
    FORCE_INLINE simd_impl operator<=(const simd_impl& other)const noexcept
    {
        simd_impl result;
        sse_map_t::cmp_le(v, other.v, result.v);
        return result;
    }
    FORCE_INLINE simd_impl operator>(const simd_impl& other)const noexcept
    {
        simd_impl result;
        sse_map_t::cmp_gt(v, other.v, result.v);
        return result;
    }
    FORCE_INLINE simd_impl operator>=(const simd_impl& other)const noexcept
    {
        simd_impl result;
        sse_map_t::cmp_ge(v, other.v, result.v);
        return result;
    }
    FORCE_INLINE simd_impl operator==(const simd_impl& other)const noexcept
    {
        simd_impl result;
        sse_map_t::cmp_eq(v, other.v, result.v);
        return result;
    }
    FORCE_INLINE simd_impl operator!=(const simd_impl& other)const noexcept
    {
        simd_impl result;
        sse_map_t::cmp_neq(v, other.v, result.v);
        return result;
    }

    simd_impl& operator*=(const simd_impl& other) noexcept
    {
        sse_map_t::mul(v, other.v, v);
        return *this;
    }
    FORCE_INLINE simd_impl operator<<(int32_t other) const noexcept
    {
        simd_impl result;
        sse_map_t::shl(v, other, result.v);
        return result;
    }
    FORCE_INLINE simd_impl& operator<<=(int32_t other) noexcept
    {
        sse_map_t::shl(v, other, v);
        return *this;
    }
    FORCE_INLINE simd_impl operator>>(int32_t other) const noexcept
    {
        simd_impl result;
        sse_map_t::shr(v, other, result.v);
        return result;
    }
    FORCE_INLINE simd_impl& operator>>=(int32_t other) noexcept
    {
        sse_map_t::shr(v, other, v);
        return *this;
    }
    FORCE_INLINE simd_impl operator*(T value) const noexcept
    {
        simd_impl result;
        sse_map_t::mul(v, value, result.v);
        return result;
    }
    FORCE_INLINE simd_impl operator/(T value) const noexcept requires(std::is_floating_point_v<T>)
    {
        simd_impl result;
        sse_map_t::div(v, value, result.v);
        return result;
    }
    FORCE_INLINE friend simd_impl operator/(T A, const simd_impl& B) noexcept requires(std::is_floating_point_v<T>)
    {
        simd_impl result{ A };
        result /= B;
        return result;
    }
    FORCE_INLINE simd_impl operator/(const simd_impl& other) const noexcept
    {
        simd_impl result;
        sse_map_t::div(v, other.v, result.v);
        return result;
    }

    FORCE_INLINE simd_impl operator%(const simd_impl& other) const noexcept requires(std::is_integral_v<T>)
    {
        using fsimd = simd_impl<float,Elements,Simd>;
        auto fthis  = fsimd(*this);
        auto fother = fsimd(other);
        return *this - ( simd_impl(fthis/fother)*other );
    }
    FORCE_INLINE simd_impl& operator/=(const simd_impl& other) noexcept
    {
        sse_map_t::div(v, other.v, v);
        return *this;
    }

    FORCE_INLINE simd_impl operator&(const simd_impl& other) const noexcept
    {
        simd_impl result;
        sse_map_t::AND(v, other.v, result.v);
        return result;
    }

    FORCE_INLINE simd_impl operator|(const simd_impl& other) const noexcept
    {
        simd_impl result;
        sse_map_t::OR(v, other.v, result.v);
        return result;
    }

    FORCE_INLINE simd_impl log()const noexcept
    {
        simd_impl result;
        sse_map_t::log(v, result.v);
        return result;
    }

    FORCE_INLINE simd_impl exp()const noexcept
    {
        simd_impl result;
        sse_map_t::exp(v, result.v);
        return result;
    }

    FORCE_INLINE simd_impl pow( const simd_impl& other )const noexcept
    {
        simd_impl result;
        sse_map_t::pow(v, other.v, result.v);
        return result;
    }

    FORCE_INLINE simd_impl pow( float other )const noexcept
    {
        simd_impl result;
        sse_map_t::pow(v, other, result.v);
        return result;
    }


    sse_t v = {};

    static const simd_impl Zero          ;
    static const simd_impl One           ;
    static const simd_impl Two           ;
    static const simd_impl PI            ;
    static const simd_impl sqrt_2        ;
    static const simd_impl inv_sqrt_2    ;


    //v4sf log_ps(v4sf x) {
    //#ifdef USE_SSE2
    //  v4si emm0;
    //#else
    //  v2si mm0, mm1;
    //#endif
    //  v4sf one = *(v4sf*)_ps_1;

    //  v4sf invalid_mask = _mm_cmple_ps(x, _mm_setzero_ps());

    //  x = _mm_max_ps(x, *(v4sf*)_ps_min_norm_pos);  /* cut off denormalized stuff */

    //#ifndef USE_SSE2
    //  /* part 1: x = frexpf(x, &e); */
    //  COPY_XMM_TO_MM(x, mm0, mm1);
    //  mm0 = _mm_srli_pi32(mm0, 23);
    //  mm1 = _mm_srli_pi32(mm1, 23);
    //#else
    //  emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);
    //#endif
    //  /* keep only the fractional part */
    //  x = _mm_and_ps(x, *(v4sf*)_ps_inv_mant_mask);
    //  x = _mm_or_ps(x, *(v4sf*)_ps_0p5);

    //#ifndef USE_SSE2
    //  /* now e=mm0:mm1 contain the really base-2 exponent */
    //  mm0 = _mm_sub_pi32(mm0, *(v2si*)_pi32_0x7f);
    //  mm1 = _mm_sub_pi32(mm1, *(v2si*)_pi32_0x7f);
    //  v4sf e = _mm_cvtpi32x2_ps(mm0, mm1);
    //  _mm_empty(); /* bye bye mmx */
    //#else
    //  emm0 = _mm_sub_epi32(emm0, *(v4si*)_pi32_0x7f);
    //  v4sf e = _mm_cvtepi32_ps(emm0);
    //#endif

    //  e = _mm_add_ps(e, one);

    //  /* part2:
    //     if( x < SQRTHF ) {
    //       e -= 1;
    //       x = x + x - 1.0;
    //     } else { x = x - 1.0; }
    //  */
    //  v4sf mask = _mm_cmplt_ps(x, *(v4sf*)_ps_cephes_SQRTHF);
    //  v4sf tmp = _mm_and_ps(x, mask);
    //  x = _mm_sub_ps(x, one);
    //  e = _mm_sub_ps(e, _mm_and_ps(one, mask));
    //  x = _mm_add_ps(x, tmp);


    //  v4sf z = _mm_mul_ps(x,x);

    //  v4sf y = *(v4sf*)_ps_cephes_log_p0;
    //  y = _mm_mul_ps(y, x);
    //  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p1);
    //  y = _mm_mul_ps(y, x);
    //  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p2);
    //  y = _mm_mul_ps(y, x);
    //  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p3);
    //  y = _mm_mul_ps(y, x);
    //  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p4);
    //  y = _mm_mul_ps(y, x);
    //  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p5);
    //  y = _mm_mul_ps(y, x);
    //  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p6);
    //  y = _mm_mul_ps(y, x);
    //  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p7);
    //  y = _mm_mul_ps(y, x);
    //  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_log_p8);
    //  y = _mm_mul_ps(y, x);

    //  y = _mm_mul_ps(y, z);


    //  tmp = _mm_mul_ps(e, *(v4sf*)_ps_cephes_log_q1);
    //  y = _mm_add_ps(y, tmp);


    //  tmp = _mm_mul_ps(z, *(v4sf*)_ps_0p5);
    //  y = _mm_sub_ps(y, tmp);

    //  tmp = _mm_mul_ps(e, *(v4sf*)_ps_cephes_log_q2);
    //  x = _mm_add_ps(x, y);
    //  x = _mm_add_ps(x, tmp);
    //  x = _mm_or_ps(x, invalid_mask); // negative arg will be NAN
    //  return x;
    //}


    //static inline simd_impl gmx_simdcall frexp(simd_impl value, SimdFInt32* exponent)
    //{
    //    svbool_t        pg           = svptrue_b32();
    //    const svint32_t exponentMask = svdup_n_s32(0x7F800000);
    //    const svint32_t mantissaMask = svdup_n_s32(0x807FFFFF);
    //    const svint32_t exponentBias = svdup_n_s32(126); // add 1 to make our definition identical to frexp()
    //    const svfloat32_t half = svdup_n_f32(0.5f);
    //    svint32_t         iExponent;

    //    iExponent = svand_s32_x(pg, svreinterpret_s32_f32(value.simdInternal_), exponentMask);
    //    iExponent = svsub_s32_x(
    //            pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, svreinterpret_u32_s32(iExponent), 23)), exponentBias);

    //    svfloat32_t result = svreinterpret_f32_s32(
    //            svorr_s32_x(pg,
    //                        svand_s32_x(pg, svreinterpret_s32_f32(value.simdInternal_), mantissaMask),
    //                        svreinterpret_s32_f32(half)));

    //    if (opt == MathOptimization::Safe)
    //    {
    //        svbool_t valueIsZero = svcmpeq_n_f32(pg, value.simdInternal_, 0.0F);
    //        iExponent            = svsel_s32(valueIsZero, svdup_n_s32(0), iExponent);
    //        result               = svsel_f32(valueIsZero, value.simdInternal_, result);
    //    }

    //    exponent->simdInternal_ = iExponent;
    //    return { result };
    //}

    //static inline simd_impl log()const
    //{
    //    const simd_impl corr(static_cast<T>(0.693147180559945286226764));
    //    const simd_impl CL9 (static_cast<T>(0.2371599674224853515625  ));
    //    const simd_impl CL7 (static_cast<T>(0.28527900576591491699218 ));
    //    const simd_impl CL5 (static_cast<T>(0.400005519390106201171875));
    //    const simd_impl CL3 (static_cast<T>(0.666666567325592041015625));
    //    const simd_impl CL1 (2.0);
    //    simd_impl       fExp, x2, p;
    //    SimdFBool       m;
    //    SimdFInt32      iExp;

    //    // For log(), the argument cannot be 0, so use the faster version of frexp()
    //    x    = frexp<MathOptimization::Unsafe>(x, &iExp);
    //    fExp = cvtI2R(iExp);

    //    m = x < invsqrt2;
    //    // Adjust to non-IEEE format for x<1/sqrt(2): exponent -= 1, mantissa *= 2.0
    //    fExp = fExp - selectByMask(one, m);
    //    x    = x * blend(one, two, m);

    //    x  = (x - one) * inv(x + one);
    //    x2 = x * x;

    //    p = fma(CL9, x2, CL7);
    //    p = fma(p, x2, CL5);
    //    p = fma(p, x2, CL3);
    //    p = fma(p, x2, CL1);
    //    p = fma(p, x, corr * fExp);

    //    return p;
    //}
};

template< typename T , int Elements , eSimdType Simd >
const simd_impl<T,Elements,Simd> simd_impl<T,Elements,Simd>::Zero          = { static_cast<T>(0) };
template< typename T , int Elements , eSimdType Simd >
const simd_impl<T,Elements,Simd> simd_impl<T,Elements,Simd>::One           = { static_cast<T>(1) };
template< typename T , int Elements , eSimdType Simd >
const simd_impl<T,Elements,Simd> simd_impl<T,Elements,Simd>::Two           = { static_cast<T>(2) };
template< typename T , int Elements , eSimdType Simd >
const simd_impl<T,Elements,Simd> simd_impl<T,Elements,Simd>::PI            = { static_cast<T>( std::numbers::pi_v<double> ) };
template< typename T , int Elements , eSimdType Simd >
const simd_impl<T,Elements,Simd> simd_impl<T,Elements,Simd>::sqrt_2        = { static_cast<T>( 1.4142135623730951 ) };
template< typename T , int Elements , eSimdType Simd >
const simd_impl<T,Elements,Simd> simd_impl<T,Elements,Simd>::inv_sqrt_2    = { static_cast<T>( 1 / 1.4142135623730951 ) };


//****************************************************************
// SIMD types
//****************************************************************

template< typename T , int Elements , eSimdType Simd = eSimdType::None >
using simd = simd_impl<T, Elements , Simd>;

template< typename T , int Elements = 4 >
using simd_cpu = simd_impl<T, Elements , eSimdType::None>;

template< typename T , int Elements = 4 >
using simd_sse = simd_impl<T, Elements , eSimdType::SSE>;

template< typename T , int Elements = 8 >
using simd_avx = simd_impl<T, Elements , eSimdType::AVX>;

template< int Elements , eSimdType Simd = eSimdType::None >
using fsimd = simd_impl< float, Elements , Simd>;

template< eSimdType Simd = eSimdType::None >
using f128   = simd<float, 4, Simd>;
template< eSimdType Simd = eSimdType::None >
using f256   = simd<float, 8, Simd>;

template< eSimdType Simd = eSimdType::None >
using i128   = simd<int, 4, Simd>;
template< eSimdType Simd = eSimdType::None >
using i256   = simd<int, 8, Simd>;

using f128S  = f128<eSimdType::SSE>;
using f256S  = f256<eSimdType::SSE>;
using f256A  = f256<eSimdType::AVX>;

using i128S  = i128<eSimdType::SSE>;
using i256S = i256<eSimdType::SSE>;
using i256A  = i256<eSimdType::AVX>;