/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#include "Simd.h"

const __m128 c_v4f_Inf         =    _mm_set1_ps(-logf(0.0f));
const __m128 c_v4f_InfMinus    =    _mm_set1_ps( logf(0.0f));
const __m256 c_v8f_Inf         = _mm256_set1_ps(-logf(0.0f));
const __m256 c_v8f_InfMinus    = _mm256_set1_ps( logf(0.0f));

#define _v4f_is_ninf(x)               _mm_cmpeq_ps(x, c_v4f_InfMinus)
#define _v8f_is_ninf(x)              _mm256_cmp_ps(x, c_v8f_InfMinus,_CMP_EQ_OQ)
#define _v4f_is_pinf(x)               _mm_cmpeq_ps(x, c_v4f_Inf)
#define _v8f_is_pinf(x)              _mm256_cmp_ps(x, c_v8f_Inf,_CMP_EQ_OQ)
#define _v4f_negatei(x)              _mm_sub_epi32(   _mm_setzero_si128(), x)
#define _v8f_negatei(x)           _mm256_sub_epi32(_mm256_setzero_si256(), x)
#define _v4f_vselecti(mask, x, y)    _mm_blendv_ps(y, x,    _mm_castsi128_ps(mask))
#define _v8f_vselecti(mask, x, y) _mm256_blendv_ps(y, x, _mm256_castsi256_ps(mask))
#define _v4f_vselect(mask, x, y)     _mm_blendv_ps(y, x, mask)
#define _v8f_vselect(mask, x, y)  _mm256_blendv_ps(y, x, mask)

#define _v4f_iselect(mask, x, y)     _mm_castps_si128(   _mm_blendv_ps(   _mm_castsi128_ps(y),    _mm_castsi128_ps(x), mask))
#define _v8f_iselect(mask, x, y)  _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(y), _mm256_castsi256_ps(x), mask))

#    define _mm_exp_ps emu_mm_exp_ps
#    define _mm_log_ps emu_mm_log_ps

namespace constants
{

template< typename T >
constexpr float Cf_PI4_A = 0.78515625f;
template< typename T >
constexpr float Cf_PI4_B = 0.00024187564849853515625f;
template< typename T >
constexpr float Cf_PI4_C = 3.7747668102383613586e-08f;
template< typename T >
constexpr float Cf_PI4_D = 1.2816720341285448015e-12f;
template< typename T >
constexpr float c_f[] = {0.31830988618379067154f        , 0.00282363896258175373077393f , -0.0159569028764963150024414f, 0.0425049886107444763183594f, -0.0748900920152664184570312f,
                         0.106347933411598205566406f    , -0.142027363181114196777344f  , 0.199926957488059997558594f, -0.333331018686294555664062f, 1.57079632679489661923f, 5.421010862427522E-20f,
                         1.8446744073709552E19f         , -Cf_PI4_A<T> * 4.0f           , -Cf_PI4_B<T> * 4.0f, -Cf_PI4_C<T> * 4.0f, -Cf_PI4_D<T> * 4.0f, 2.6083159809786593541503e-06f, -0.0001981069071916863322258f,
                         0.00833307858556509017944336f  , -0.166666597127914428710938f  , -Cf_PI4_A<T> * 2.0f, -Cf_PI4_B<T> * 2.0f, -Cf_PI4_C<T> * 2.0f, -Cf_PI4_D<T> * 2.0f, 0.63661977236758134308f,
                        -0.000195169282960705459117889f , 0.00833215750753879547119141f , -0.166666537523269653320312f, -2.71811842367242206819355e-07f, 2.47990446951007470488548e-05f,
                        -0.00138888787478208541870117f  , 0.0416666641831398010253906f  , -0.5f, 1.0f, 0.00927245803177356719970703f, 0.00331984995864331722259521f, 0.0242998078465461730957031f,
                         0.0534495301544666290283203f   , 0.133383005857467651367188f   , 0.333331853151321411132812f, 0.78539816339744830962f, -1.0f, 0.5f, 3.14159265358979323846f, 0.7071f,
                         0.2371599674224853515625f      , 0.285279005765914916992188f   , 0.400005519390106201171875f, 0.666666567325592041015625f, 2.0f, 0.693147180559945286226764f,
                         1.442695040888963407359924681001892137426645954152985934135449406931f, -0.693145751953125f, -1.428606765330187045e-06f, 0.00136324646882712841033936f,
                         0.00836596917361021041870117f  , 0.416710823774337768554688f , 0.166665524244308471679688f, 0.499999850988388061523438f};

}

__m128 _v4f_ldexp(const __m128& x, const __m128i& q)
{
    constexpr auto c_f = constants::c_f<float>;

    __m128i m = _mm_srai_epi32(q, 31);
    m = _mm_slli_epi32(_mm_sub_epi32(_mm_srai_epi32(_mm_add_epi32(m, q), 6), m), 4);
    __m128i t = _mm_sub_epi32(q, _mm_slli_epi32(m, 2));
    m = _mm_add_epi32(m, _mm_set1_epi32(0x7f));
    m = _mm_and_si128(_mm_cmpgt_epi32(m, _mm_setzero_si128()), m);
    __m128i n = _mm_cmpgt_epi32(m, _mm_set1_epi32(0xff));
    m = _mm_or_si128(_mm_andnot_si128(n, m), _mm_and_si128(n, _mm_set1_epi32(0xff)));
    __m128 u = _mm_castsi128_ps(_mm_slli_epi32(m, 23));
    __m128 r = _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(_mm_mul_ps(x, u), u), u), u);
    u = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(t, _mm_set1_epi32(0x7f)), 23));

    return _mm_mul_ps(r, u);
}

__m256 _v8f_ldexp(const __m256& x, const __m256i& q)
{
    constexpr auto c_f = constants::c_f<float>;

    __m256i m = _mm256_srai_epi32(q, 31);
    m = _mm256_slli_epi32(_mm256_sub_epi32(_mm256_srai_epi32(_mm256_add_epi32(m, q), 6), m), 4);
    __m256i t = _mm256_sub_epi32(q, _mm256_slli_epi32(m, 2));
    m = _mm256_add_epi32(m, _mm256_set1_epi32(0x7f));
    m = _mm256_and_si256(_mm256_cmpgt_epi32(m, _mm256_setzero_si256()), m);
    __m256i n = _mm256_cmpgt_epi32(m, _mm256_set1_epi32(0xff));
    m = _mm256_or_si256(_mm256_andnot_si256(n, m), _mm256_and_si256(n, _mm256_set1_epi32(0xff)));
    __m256 u = _mm256_castsi256_ps(_mm256_slli_epi32(m, 23));
    __m256 r = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(x, u), u), u), u);
    u = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(t, _mm256_set1_epi32(0x7f)), 23));

    return _mm256_mul_ps(r, u);
}

__m128i _v4f_logbp1(const __m128& d)
{
    constexpr auto c_f = constants::c_f<float>;

    __m128 m = _mm_cmplt_ps(d, _mm_broadcast_ss(c_f + 10));
    __m128 r = _v4f_vselect(m, _mm_mul_ps(_mm_broadcast_ss(c_f + 11), d), d);
    __m128i q = _mm_and_si128(_mm_srli_epi32(_mm_castps_si128(r), 23), _mm_set1_epi32(0xff));
    q = _mm_sub_epi32(q, _v4f_iselect(m, _mm_set1_epi32(64 + 0x7e), _mm_set1_epi32(0x7e)));

    return q;
}

__m256i _v8f_logbp1(const __m256& d)
{
    constexpr auto c_f = constants::c_f<float>;

    __m256 m = _mm256_cmp_ps(d, _mm256_broadcast_ss(c_f + 10), _CMP_LT_OQ);
    __m256 r = _v8f_vselect(m, _mm256_mul_ps(_mm256_broadcast_ss(c_f + 11), d), d);
    __m256i q = _mm256_and_si256(_mm256_srli_epi32(_mm256_castps_si256(r), 23), _mm256_set1_epi32(0xff));
    q = _mm256_sub_epi32(q, _v8f_iselect(m, _mm256_set1_epi32(64 + 0x7e), _mm256_set1_epi32(0x7e)));

    return q;
}

__m128 emu_mm_log_ps(const __m128& d)
{
    constexpr auto c_f = constants::c_f<float>;

    __m128 x = _mm_mul_ps(d, _mm_broadcast_ss(c_f + 44));
    __m128i e = _v4f_logbp1(x);
    __m128 m = _v4f_ldexp(d, _v4f_negatei(e));
    __m128 r = x;

    x = _mm_div_ps(_mm_add_ps(_mm_broadcast_ss(c_f + 41), m), _mm_add_ps(_mm_broadcast_ss(c_f + 33), m));
    __m128 x2 = _mm_mul_ps(x, x);

    __m128 t = _mm_broadcast_ss(c_f + 45);
    t = _mm_fmadd_ps(t, x2, _mm_broadcast_ss(c_f + 46));
    t = _mm_fmadd_ps(t, x2, _mm_broadcast_ss(c_f + 47));
    t = _mm_fmadd_ps(t, x2, _mm_broadcast_ss(c_f + 48));
    t = _mm_fmadd_ps(t, x2, _mm_broadcast_ss(c_f + 49));

    x = _mm_fmadd_ps(x, t, _mm_mul_ps(_mm_broadcast_ss(c_f + 50), _mm_cvtepi32_ps(e)));
    x = _v4f_vselect(_v4f_is_pinf(r), c_v4f_Inf, x);

    x = _mm_or_ps(_mm_cmpgt_ps(_mm_setzero_ps(), r), x);
    x = _v4f_vselect(_mm_cmpeq_ps(r, _mm_setzero_ps()), c_v4f_InfMinus, x);

    return x;
}

__m256 emu_mm256_log_ps(const __m256& d)
{
    constexpr auto c_f = constants::c_f<float>;

    __m256 x = _mm256_mul_ps(d, _mm256_broadcast_ss(c_f + 44));
    __m256i e = _v8f_logbp1(x);
    __m256 m = _v8f_ldexp(d, _v8f_negatei(e));
    __m256 r = x;

    x = _mm256_div_ps(_mm256_add_ps(_mm256_broadcast_ss(c_f + 41), m), _mm256_add_ps(_mm256_broadcast_ss(c_f + 33), m));
    __m256 x2 = _mm256_mul_ps(x, x);

    __m256 t = _mm256_broadcast_ss(c_f + 45);
    t = _mm256_fmadd_ps(t, x2, _mm256_broadcast_ss(c_f + 46));
    t = _mm256_fmadd_ps(t, x2, _mm256_broadcast_ss(c_f + 47));
    t = _mm256_fmadd_ps(t, x2, _mm256_broadcast_ss(c_f + 48));
    t = _mm256_fmadd_ps(t, x2, _mm256_broadcast_ss(c_f + 49));

    x = _mm256_fmadd_ps(x, t, _mm256_mul_ps(_mm256_broadcast_ss(c_f + 50), _mm256_cvtepi32_ps(e)));
    x = _v8f_vselect(_v8f_is_pinf(r), c_v8f_Inf, x);

    x = _mm256_or_ps(_mm256_cmp_ps(_mm256_setzero_ps(), r,_CMP_GT_OQ), x);
    x = _v8f_vselect(_mm256_cmp_ps(r, _mm256_setzero_ps(),_CMP_EQ_OQ), c_v8f_InfMinus, x);

    return x;
}

__m128 emu_mm_exp_ps(const __m128& d)
{
    constexpr auto c_f = constants::c_f<float>;

    __m128i q = _mm_cvtps_epi32(_mm_mul_ps(d, _mm_broadcast_ss(c_f + 51)));

    __m128 s = _mm_fmadd_ps(_mm_cvtepi32_ps(q), _mm_broadcast_ss(c_f + 52), d);
    s = _mm_fmadd_ps(_mm_cvtepi32_ps(q), _mm_broadcast_ss(c_f + 53), s);

    __m128 u = _mm_broadcast_ss(c_f + 54);
    u = _mm_fmadd_ps(u, s, _mm_broadcast_ss(c_f + 55));
    u = _mm_fmadd_ps(u, s, _mm_broadcast_ss(c_f + 56));
    u = _mm_fmadd_ps(u, s, _mm_broadcast_ss(c_f + 57));
    u = _mm_fmadd_ps(u, s, _mm_broadcast_ss(c_f + 58));

    u = _mm_add_ps(_mm_broadcast_ss(c_f + 33), _mm_fmadd_ps(_mm_mul_ps(s, s), u, s));
    u = _v4f_ldexp(u, q);

    u = _mm_andnot_ps(_v4f_is_ninf(d), u);

    return u;
}

__m256 emu_mm256_exp_ps(const __m256& d)
{
    constexpr auto c_f = constants::c_f<float>;

    __m256i q = _mm256_cvtps_epi32(_mm256_mul_ps(d, _mm256_broadcast_ss(c_f + 51)));

    __m256 s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q), _mm256_broadcast_ss(c_f + 52), d);
    s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q), _mm256_broadcast_ss(c_f + 53), s);

    __m256 u = _mm256_broadcast_ss(c_f + 54);
    u = _mm256_fmadd_ps(u, s, _mm256_broadcast_ss(c_f + 55));
    u = _mm256_fmadd_ps(u, s, _mm256_broadcast_ss(c_f + 56));
    u = _mm256_fmadd_ps(u, s, _mm256_broadcast_ss(c_f + 57));
    u = _mm256_fmadd_ps(u, s, _mm256_broadcast_ss(c_f + 58));

    u = _mm256_add_ps(_mm256_broadcast_ss(c_f + 33), _mm256_fmadd_ps(_mm256_mul_ps(s, s), u, s));
    u = _v8f_ldexp(u, q);

    u = _mm256_andnot_ps(_v8f_is_ninf(d), u);

    return u;
}

__m128 emu_mm_pow_ps(const __m128& x, const __m128& y)
{
    return emu_mm_exp_ps(_mm_mul_ps(emu_mm_log_ps(x), y));
}

__m128 emu_mm_pow_ps(const __m128& x, float y)
{
    return emu_mm_exp_ps(_mm_mul_ps(emu_mm_log_ps(x), _mm_set1_ps(y)));
}

__m256 emu_mm256_pow_ps(const __m256& x, const __m256& y)
{
    return emu_mm256_exp_ps(_mm256_mul_ps(emu_mm256_log_ps(x), y));
}

__m256 emu_mm256_pow_ps(const __m256& x, float y)
{
    return emu_mm256_exp_ps(_mm256_mul_ps(emu_mm256_log_ps(x), _mm256_set1_ps(y)));
}