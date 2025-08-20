/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#include "Simd.h"
#include <bit>

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


static auto generate_mm256_compress_lookup()
{
    struct mm256_compress_lookup
    {
        __m256i data[256];
    };
    mm256_compress_lookup result;

    for (int m = 0; m < 256; m++)
    {
        int pos = 0;
        int idx[8] = {};

        for (int i = 0; i < 8; i++)
        {
            if (m & (1 << i))
                idx[pos++] = i;
        }

        result.data[m] = _mm256_set_epi32(idx[7], idx[6], idx[5], idx[4], idx[3], idx[2], idx[1], idx[0]);
    }

    return result;
}

static auto generate_mm256_decompress_lookup()
{
    struct mm256_decompress_lookup
    {
        __m256i data[256];
    };
    mm256_decompress_lookup result;

    for (int m = 0; m < 256; m++)
    {
        int pos = 0;
        int idx[8] = {};

        for (int i = 0; i < 8; i++)
        {
            if (m & (1 << i))
                idx[i] = pos++;
        }

        result.data[m] = _mm256_set_epi32(idx[7], idx[6], idx[5], idx[4], idx[3], idx[2], idx[1], idx[0]);
    }

    return result;
}

static auto generate_mm_compress_lookup()
{
    struct mm_compress_lookup
    {
        __m128i data[16];
    };
    mm_compress_lookup result;

    for (int m = 0; m < 16; m++)
    {
        int pos = 0;
        int idx[4] = {};

        for (int i = 0; i < 4; i++)
        {
            if (m & (1 << i))
                idx[pos++] = i;
        }

        result.data[m] = _mm_set_epi32(idx[3], idx[2], idx[1], idx[0]);
    }

    return result;
}

static auto generate_mm_decompress_lookup()
{
    struct mm_decompress_lookup
    {
        __m128i data[16];
    };
    mm_decompress_lookup result;

    for (int m = 0; m < 16; m++)
    {
        int pos = 0;
        int idx[4] = {};

        for (int i = 0; i < 4; i++)
        {
            if (m & (1 << i))
                idx[i] = pos++;
        }

        result.data[m] = _mm_set_epi32(idx[3], idx[2], idx[1], idx[0]);
    }

    return result;
}

const auto mm256_compress_lookup    = generate_mm256_compress_lookup();
const auto mm256_decompress_lookup  = generate_mm256_decompress_lookup();
const auto mm_compress_lookup       = generate_mm_compress_lookup();
const auto mm_decompress_lookup     = generate_mm_decompress_lookup();

//****************************************************************
//
//****************************************************************

int emu_mm256_mask_compressstoreu_ps(float *dst, __m256 mask, __m256 data)
{
    const auto maskbits = _mm256_movemask_ps(mask);
    const auto skipbits = 8-std::popcount(uint32_t(maskbits));

    __m256 permuted = _mm256_permutevar8x32_ps(data, mm256_compress_lookup.data[maskbits] );

    ALIGN_FOR_AVX constexpr uint32_t init_mask[8] =
    {
        uint32_t(0b11111111) << 24 , uint32_t(0b11111110) << 24 , uint32_t(0b11111100) << 24 , uint32_t(0b11111000) << 24 ,
        uint32_t(0b11110000) << 24 , uint32_t(0b11100000) << 24 , uint32_t(0b11000000) << 24 , uint32_t(0b10000000) << 24 ,
    };

    auto write_mask = i256A{(const int32_t*)init_mask , simd_alignment::AVX} << skipbits;

    _mm256_maskstore_ps(dst, write_mask.v , permuted);

    return maskbits;
}

int emu_mm256_mask_compressstoreu_ps_x4( float* dstA, float* dstB, float* dstC, float* dstD, __m256 mask, const __m256* dataA, const __m256* dataB, const __m256* dataC, const __m256* dataD)
{
    const auto maskbits     = _mm256_movemask_ps(mask);
    const auto skipbits     = 8-std::popcount(uint32_t(maskbits));
    const auto permute_ctrl = mm256_compress_lookup.data[maskbits];

    __m256 permuted[4];

    ALIGN_FOR_AVX constexpr uint32_t init_mask[8] =
    {
        uint32_t(0b11111111) << 24 , uint32_t(0b11111110) << 24 , uint32_t(0b11111100) << 24 , uint32_t(0b11111000) << 24 ,
        uint32_t(0b11110000) << 24 , uint32_t(0b11100000) << 24 , uint32_t(0b11000000) << 24 , uint32_t(0b10000000) << 24 ,
    };

    permuted[0] = _mm256_permutevar8x32_ps(*dataA, permute_ctrl );
    permuted[1] = _mm256_permutevar8x32_ps(*dataB, permute_ctrl );
    permuted[2] = _mm256_permutevar8x32_ps(*dataC, permute_ctrl );
    permuted[3] = _mm256_permutevar8x32_ps(*dataD, permute_ctrl );

    auto write_mask = i256A{(const int32_t*)init_mask , simd_alignment::AVX} << skipbits;

    _mm256_maskstore_ps(dstA, write_mask.v , permuted[0]);
    _mm256_maskstore_ps(dstB, write_mask.v , permuted[1]);
    _mm256_maskstore_ps(dstC, write_mask.v , permuted[2]);
    _mm256_maskstore_ps(dstD, write_mask.v , permuted[3]);

    return maskbits;
}

int emu_mm256_mask_compressstoreu_ps_ov(float *dst, __m256 mask, __m256 data)
{
    const auto maskbits = _mm256_movemask_ps(mask);

    __m256 permuted = _mm256_permutevar8x32_ps(data, mm256_compress_lookup.data[maskbits] );

    _mm256_storeu_ps(dst, permuted);

    return maskbits;
}

int emu_mm256_mask_compressstoreu_ps_x4_ov( float* dstA, float* dstB, float* dstC, float* dstD, __m256 mask, const __m256* dataA, const __m256* dataB, const __m256* dataC, const __m256* dataD)
{
    const auto maskbits     = _mm256_movemask_ps(mask);
    const auto permute_ctrl = mm256_compress_lookup.data[maskbits];

    __m256 permuted[4];

    permuted[0] = _mm256_permutevar8x32_ps(*dataA, permute_ctrl );
    permuted[1] = _mm256_permutevar8x32_ps(*dataB, permute_ctrl );
    permuted[2] = _mm256_permutevar8x32_ps(*dataC, permute_ctrl );
    permuted[3] = _mm256_permutevar8x32_ps(*dataD, permute_ctrl );

    _mm256_storeu_ps(dstA, permuted[0]);
    _mm256_storeu_ps(dstB, permuted[1]);
    _mm256_storeu_ps(dstC, permuted[2]);
    _mm256_storeu_ps(dstD, permuted[3]);

    return maskbits;
}

__m256 emu_mm256_mask_expandloadu_ps(__m256 src, __m256 mask, const void* mem, int& mask_bits)
{
    mask_bits = _mm256_movemask_ps(mask);
    auto data = _mm256_permutevar8x32_ps(_mm256_loadu_ps((const float*)mem), mm256_decompress_lookup.data[mask_bits] );

    return _mm256_blendv_ps(src, data, mask);
}


void emu_mm256_mask_expandloadu_ps_x4      (__m256* dstA,__m256* dstB,__m256* dstC,__m256* dstD, __m256 mask, const float* memA, const float* memB, const float* memC, const float* memD, int& mask_bits)
{
    mask_bits = _mm256_movemask_ps(mask);
    const auto permute_ctrl = mm256_decompress_lookup.data[mask_bits];

    dstA[0] = _mm256_loadu_ps(memA);
    dstB[0] = _mm256_loadu_ps(memB);
    dstC[0] = _mm256_loadu_ps(memC);
    dstD[0] = _mm256_loadu_ps(memD);

    dstA[0] = _mm256_permutevar8x32_ps( dstA[0] , permute_ctrl );
    dstB[0] = _mm256_permutevar8x32_ps( dstB[0] , permute_ctrl );
    dstC[0] = _mm256_permutevar8x32_ps( dstC[0] , permute_ctrl );
    dstD[0] = _mm256_permutevar8x32_ps( dstD[0] , permute_ctrl );
}
void _mm256_transpose_ARGBx8_to_Ax8Rx8Gx8Bx8_ps( __m256& inout0, __m256& inout1, __m256& inout2, __m256& inout3)
{
    __m256 tmp;

    // inout0 ->  A B C D  E F G H
    // inout1 ->  I J K L  M N O P
    // inout2 ->  Q R S T  U V W X
    // inout3 ->  Y Z 1 2  3 4 5 6

    tmp     = _mm256_permute2f128_ps( inout0 ,inout2 , 0x31 );
    inout0  = _mm256_permute2f128_ps( inout0 ,inout2 , 0x20 );  // A B C D  Q R S T
    inout2  = tmp;                                              // E F G H  U V W X
    tmp     = _mm256_permute2f128_ps( inout1 ,inout3 , 0x31 );
    inout1  = _mm256_permute2f128_ps( inout1 ,inout3 , 0x20 );  // I J K L  Y Z 1 2
    inout3  = tmp;                                              // M N O P  3 4 5 6

    // Unpack low/high
    __m256 t0 = _mm256_unpacklo_ps(inout0, inout2); // A E B F  Q U R V
    __m256 t1 = _mm256_unpackhi_ps(inout0, inout2); // C G D H  S W T X
    __m256 t2 = _mm256_unpacklo_ps(inout1, inout3); // I M J N  Y 3 Z 4
    __m256 t3 = _mm256_unpackhi_ps(inout1, inout3); // K O L P  1 5 2 6

    // Final permutes
    inout0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0)); // A E I M Q U Y 3
    inout1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2)); // B F J N R V Z 4
    inout2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0)); // C G K O S W 1 5
    inout3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2)); // D H L P T X 2 6
}

void _mm256_transpose_ARGBx8_to_Ax8Rx8Gx8Bx8_epi32( __m256i& inout0, __m256i& inout1, __m256i& inout2, __m256i& inout3)
{
    __m256i tmp;

    // inout0 ->  A B C D  E F G H
    // inout1 ->  I J K L  M N O P
    // inout2 ->  Q R S T  U V W X
    // inout3 ->  Y Z 1 2  3 4 5 6

    tmp     = _mm256_permute2f128_si256( inout0 ,inout2 , 0x31 );
    inout0  = _mm256_permute2f128_si256( inout0 ,inout2 , 0x20 );   // A B C D  Q R S T
    inout2  = tmp;                                                  // E F G H  U V W X
    tmp     = _mm256_permute2f128_si256( inout1 ,inout3 , 0x31 );
    inout1  = _mm256_permute2f128_si256( inout1 ,inout3 , 0x20 );   // I J K L  Y Z 1 2
    inout3  = tmp;                                                  // M N O P  3 4 5 6

    // Unpack low/high
    __m256 t0 = _mm256_castsi256_ps( _mm256_unpacklo_epi32(inout0, inout2) ); // A E B F  Q U R V
    __m256 t1 = _mm256_castsi256_ps( _mm256_unpackhi_epi32(inout0, inout2) ); // C G D H  S W T X
    __m256 t2 = _mm256_castsi256_ps( _mm256_unpacklo_epi32(inout1, inout3) ); // I M J N  Y 3 Z 4
    __m256 t3 = _mm256_castsi256_ps( _mm256_unpackhi_epi32(inout1, inout3) ); // K O L P  1 5 2 6

    // Final permutes
    inout0 = _mm256_castps_si256( _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0)) ); // A E I M Q U Y 3
    inout1 = _mm256_castps_si256( _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2)) ); // B F J N R V Z 4
    inout2 = _mm256_castps_si256( _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0)) ); // C G K O S W 1 5
    inout3 = _mm256_castps_si256( _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2)) ); // D H L P T X 2 6
}

//****************************************************************
//
//****************************************************************

int emu_mm_mask_compressstoreu_ps(float *dst, __m128 mask, __m128 data)
{
    const auto maskbits = static_cast<uint8_t>( _mm_movemask_ps(mask) );
    const auto skipbits = 4-std::popcount(uint32_t(maskbits));

    __m128 permuted = _mm_permutevar_ps(data, mm_compress_lookup.data[maskbits] );

    ALIGN_FOR_SSE constexpr uint32_t init_mask[4] =
    {
        uint32_t(0b11110000) << 24 , uint32_t(0b11100000) << 24 , uint32_t(0b11000000) << 24 , uint32_t(0b10000000) << 24 ,
    };

    auto write_mask = i128S{(const int32_t*)init_mask , simd_alignment::SSE} << skipbits;

    _mm_maskstore_ps(dst, write_mask.v , permuted);

    return maskbits;
}

int emu_mm_mask_compressstoreu_ps_x4( float* dstA, float* dstB, float* dstC, float* dstD, __m128 mask, const __m128* dataA, const __m128* dataB, const __m128* dataC, const __m128* dataD)
{
    const auto maskbits     = _mm_movemask_ps(mask);
    const auto skipbits     = 8-std::popcount(uint32_t(maskbits));
    const auto permute_ctrl = mm_compress_lookup.data[maskbits];

    __m128 permuted[4];

    ALIGN_FOR_SSE constexpr uint32_t init_mask[4] =
    {
        uint32_t(0b11110000) << 24 , uint32_t(0b11100000) << 24 , uint32_t(0b11000000) << 24 , uint32_t(0b10000000) << 24 ,
    };

    permuted[0] = _mm_permutevar_ps(*dataA, permute_ctrl );
    permuted[1] = _mm_permutevar_ps(*dataB, permute_ctrl );
    permuted[2] = _mm_permutevar_ps(*dataC, permute_ctrl );
    permuted[3] = _mm_permutevar_ps(*dataD, permute_ctrl );

    auto write_mask = i128S{(const int32_t*)init_mask , simd_alignment::SSE} << skipbits;

    _mm_maskstore_ps(dstA, write_mask.v , permuted[0]);
    _mm_maskstore_ps(dstB, write_mask.v , permuted[1]);
    _mm_maskstore_ps(dstC, write_mask.v , permuted[2]);
    _mm_maskstore_ps(dstD, write_mask.v , permuted[3]);

    return maskbits;
}

int emu_mm_mask_compressstoreu_ps_x4_ov( float* dstA, float* dstB, float* dstC, float* dstD, __m128 mask, const __m128* dataA, const __m128* dataB, const __m128* dataC, const __m128* dataD)
{
    const auto maskbits     = _mm_movemask_ps(mask);
    const auto permute_ctrl = mm_compress_lookup.data[maskbits];

    __m128 permuted[4];

    permuted[0] = _mm_permutevar_ps(*dataA, permute_ctrl );
    permuted[1] = _mm_permutevar_ps(*dataB, permute_ctrl );
    permuted[2] = _mm_permutevar_ps(*dataC, permute_ctrl );
    permuted[3] = _mm_permutevar_ps(*dataD, permute_ctrl );

    _mm_storeu_ps(dstA, permuted[0]);
    _mm_storeu_ps(dstB, permuted[1]);
    _mm_storeu_ps(dstC, permuted[2]);
    _mm_storeu_ps(dstD, permuted[3]);

    return maskbits;
}

int emu_mm_mask_compressstoreu_ps_ov(float *dst, __m128 mask, __m128 data)
{
    const auto maskbits = static_cast<uint8_t>( _mm_movemask_ps(mask) );

    __m128 permuted = _mm_permutevar_ps(data, mm_compress_lookup.data[maskbits] );

    _mm_storeu_ps(dst , permuted);

    return maskbits;
}

void emu_mm_mask_expandloadu_ps_x4(__m128* dstA,__m128* dstB,__m128* dstC,__m128* dstD, __m128 mask, const float* memA, const float* memB, const float* memC, const float* memD, int& mask_bits)
{
    mask_bits = _mm_movemask_ps(mask);
    const auto permute_ctrl = mm_decompress_lookup.data[mask_bits];

    dstA[0] = _mm_loadu_ps(memA);
    dstB[0] = _mm_loadu_ps(memB);
    dstC[0] = _mm_loadu_ps(memC);
    dstD[0] = _mm_loadu_ps(memD);

    dstA[0] = _mm_permutevar_ps( dstA[0] , permute_ctrl );
    dstB[0] = _mm_permutevar_ps( dstB[0] , permute_ctrl );
    dstC[0] = _mm_permutevar_ps( dstC[0] , permute_ctrl );
    dstD[0] = _mm_permutevar_ps( dstD[0] , permute_ctrl );
}

__m128 emu_mm_mask_expandloadu_ps(__m128 src, __m128 mask, const void* mem, int& mask_bits)
{
    mask_bits = _mm_movemask_ps(mask);
    auto data = _mm_permutevar_ps(_mm_loadu_ps((const float*)mem), mm_decompress_lookup.data[mask_bits] );

    return _mm_blendv_ps(src, data, mask);
}

void _mm_transpose_ARGBx4_to_Ax4Rx4Gx4Bx4_ps( __m128& inout0, __m128& inout1, __m128& inout2, __m128& inout3)
{
    __m256 tmp;

    // inout0 ->  A B C D
    // inout1 ->  E F G H
    // inout2 ->  I J K L
    // inout3 ->  M N O P

    // krok 1: rozplatanie parami
    __m128 t0 = _mm_unpacklo_ps(inout0, inout1); // A0 A1 R0 R1  | A E B F
    __m128 t1 = _mm_unpackhi_ps(inout0, inout1); // G0 G1 B0 B1  | C G D H
    __m128 t2 = _mm_unpacklo_ps(inout2, inout3); // A2 A3 R2 R3  | I M J N
    __m128 t3 = _mm_unpackhi_ps(inout2, inout3); // G2 G3 B2 B3  | K O L P

    // krok 2: finalna transpozycja
    inout0 = _mm_movelh_ps(t0, t2); // A E I M
    inout1 = _mm_movehl_ps(t2, t0); // B F J N
    inout2 = _mm_movelh_ps(t1, t3); // C G K O
    inout3 = _mm_movehl_ps(t3, t1); // D H L P
}

void _mm_transpose_ARGBx4_to_Ax4Rx4Gx4Bx4_epi32( __m128i& inout0, __m128i& inout1, __m128i& inout2, __m128i& inout3)
{
    __m256 tmp;

    // inout0 ->  A B C D
    // inout1 ->  E F G H
    // inout2 ->  I J K L
    // inout3 ->  M N O P

    // krok 1: rozplatanie parami
    __m128 t0 = _mm_castsi128_ps( _mm_unpacklo_epi32(inout0, inout1) ); // A0 A1 R0 R1  | A E B F
    __m128 t1 = _mm_castsi128_ps( _mm_unpackhi_epi32(inout0, inout1) ); // G0 G1 B0 B1  | C G D H
    __m128 t2 = _mm_castsi128_ps( _mm_unpacklo_epi32(inout2, inout3) ); // A2 A3 R2 R3  | I M J N
    __m128 t3 = _mm_castsi128_ps( _mm_unpackhi_epi32(inout2, inout3) ); // G2 G3 B2 B3  | K O L P

    // krok 2: finalna transpozycja
    inout0 = _mm_castps_si128( _mm_movelh_ps(t0, t2) ); // A E I M
    inout1 = _mm_castps_si128( _mm_movehl_ps(t2, t0) ); // B F J N
    inout2 = _mm_castps_si128( _mm_movelh_ps(t1, t3) ); // C G K O
    inout3 = _mm_castps_si128( _mm_movehl_ps(t3, t1) ); // D H L P
}

void _mm_transpose_ARGBx8_to_Ax8Rx8Gx8Bx8_ps( __m128& inout0, __m128& inout1, __m128& inout2, __m128& inout3, __m128& inout4, __m128& inout5, __m128& inout6, __m128& inout7)
{
    _mm_transpose_ARGBx4_to_Ax4Rx4Gx4Bx4_ps( inout0, inout1, inout2, inout3 );
    _mm_transpose_ARGBx4_to_Ax4Rx4Gx4Bx4_ps( inout4, inout5, inout6, inout7 );

    __m128 tmp0, tmp1, tmp2, tmp3;

    tmp1 = inout1;
    tmp2 = inout2;
    tmp3 = inout3;

    inout1 = inout4;
    inout2 = tmp1;
    inout3 = inout5;

    inout4 = tmp2;
    inout5 = inout6;
    inout6 = tmp3;

}

void _mm_transpose_ARGBx8_to_Ax8Rx8Gx8Bx8_epi32( __m128i& inout0, __m128i& inout1, __m128i& inout2, __m128i& inout3, __m128i& inout4, __m128i& inout5, __m128i& inout6, __m128i& inout7)
{
    _mm_transpose_ARGBx4_to_Ax4Rx4Gx4Bx4_epi32( inout0, inout1, inout2, inout3 );
    _mm_transpose_ARGBx4_to_Ax4Rx4Gx4Bx4_epi32( inout4, inout5, inout6, inout7 );

    __m128i tmp0, tmp1, tmp2, tmp3;

    tmp1 = inout1;
    tmp2 = inout2;
    tmp3 = inout3;

    inout1 = inout4;
    inout2 = tmp1;
    inout3 = inout5;

    inout4 = tmp2;
    inout5 = inout6;
    inout6 = tmp3;

}


int test1 = []()
{
    ALIGN_FOR_AVX float data[8] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
    ALIGN_FOR_AVX int   ctrl[8] = {    0,   -1,   -1,   -1,    0,   -1,   -1,    0 };

    __m256 a    = _mm256_loadu_ps(data);
    __m256 mask = _mm256_castsi256_ps( _mm256_loadu_si256((const __m256i*)ctrl) );

    float dst[8] = {-1,-1,-1,-1,-1,-1,-1,-1};

    int count1 = emu_mm256_mask_compressstoreu_ps(dst, mask,a);

    int bits = -1;
    __m256 expanded = emu_mm256_mask_expandloadu_ps(_mm256_setzero_ps(), mask, dst,bits);

    float dst2[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    _mm256_storeu_ps(dst2, expanded);

    {
        ALIGN_FOR_AVX float src[16]     = { 9.0f, 7.0f, 5.0f, 3.0f, 2.0f, 4.0f, 6.0f, 8.0f
                                          , 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 7.0f, 8.0f };

                      float stream[16]  = { -1,-1,-1,-1,-1,-1,-1,-1 ,-1,-1,-1,-1,-1,-1,-1,-1 };
        ALIGN_FOR_AVX float dst[16]     = { -1,-1,-1,-1,-1,-1,-1,-1 ,-1,-1,-1,-1,-1,-1,-1,-1 };

        auto pStream = stream;

        f256A A{ src , simd_alignment::AVX };
        f256A B{ 5.0f };
        auto Mask = A >= B;
        pStream += std::popcount( (uint32_t) A.compressed_store( pStream , Mask ) );

        A.load( src+8 , simd_alignment::AVX );
        f256A Mask2 = A < B;

        pStream += std::popcount( (uint32_t) A.compressed_store( pStream , Mask2 ) );

        pStream = stream;

        f256A C;

        pStream += std::popcount( (uint32_t) C.expand_load( pStream , Mask ) );
        C.store( dst , Mask );

        pStream += std::popcount( (uint32_t) C.expand_load( pStream , Mask2 ) );
        C.store( dst+8 , Mask2 );


        int i=0;
    }


    return 0;
}();

int test2 = []()
{
    ALIGN_FOR_SSE float data[4] = {  1.0f,  2.0f,  3.0f,  4.0f };
    ALIGN_FOR_SSE int   ctrl[4] = {    -1,     0,     -1,   -1 };

    __m128 a    = _mm_loadu_ps(data);
    __m128 mask = _mm_castsi128_ps( _mm_loadu_si128((const __m128i*)ctrl) );

    float dst[4] = {-1,-1,-1,-1};

    int count1 = emu_mm_mask_compressstoreu_ps(dst, mask,a);

    int bits = -1;
    __m128 expanded = emu_mm_mask_expandloadu_ps(_mm_setzero_ps(), mask, dst,bits);

    float dst2[4] = {-1,-1,-1,-1};
    _mm_storeu_ps(dst2, expanded);

    {
        ALIGN_FOR_SSE float in[16] =
        {
            1.0f , 2.0f , 3.0f , 4.0f ,
            5.0f , 6.0f , 7.0f , 8.0f ,
            9.0f ,10.0f ,11.0f ,12.0f ,
            13.0f,14.0f, 15.0f, 16.0f ,
        };
        ALIGN_FOR_SSE float out[16] = {};

        __m128 regs[4];
        regs[0] = _mm_load_ps( in + 0 );
        regs[1] = _mm_load_ps( in + 4 );
        regs[2] = _mm_load_ps( in + 8 );
        regs[3] = _mm_load_ps( in + 12 );

        _mm_transpose_ARGBx4_to_Ax4Rx4Gx4Bx4_ps( regs[0], regs[1], regs[2], regs[3] );

        _mm_store_ps( out + 0, regs[0] );
        _mm_store_ps( out + 4, regs[1] );
        _mm_store_ps( out + 8, regs[2] );
        _mm_store_ps( out + 12, regs[3] );

        int i=0;
    }
    {
        ALIGN_FOR_AVX float in[32] =
        {
            1.0f , 11.0f , 21.0f , 31.0f   ,   2.0f , 12.0f , 22.0f , 32.0f ,
            3.0f , 13.0f , 23.0f , 33.0f   ,   4.0f , 14.0f , 24.0f , 34.0f ,
            5.0f , 15.0f , 25.0f , 35.0f   ,   6.0f , 16.0f , 26.0f , 36.0f ,
            7.0f , 17.0f , 27.0f , 37.0f   ,   8.0f , 18.0f , 28.0f , 38.0f ,
        };
        ALIGN_FOR_AVX float out1[32] = {};
        ALIGN_FOR_AVX float out2[32] = {};

        __m128 regs1[8];
        __m256 regs2[4];

        regs1[0] = _mm_load_ps( in +  0 );
        regs1[1] = _mm_load_ps( in +  4 );
        regs1[2] = _mm_load_ps( in +  8 );
        regs1[3] = _mm_load_ps( in + 12 );
        regs1[4] = _mm_load_ps( in + 16 );
        regs1[5] = _mm_load_ps( in + 20 );
        regs1[6] = _mm_load_ps( in + 24 );
        regs1[7] = _mm_load_ps( in + 28 );

        _mm_transpose_ARGBx8_to_Ax8Rx8Gx8Bx8_ps( regs1[0], regs1[1], regs1[2], regs1[3] , regs1[4], regs1[5], regs1[6], regs1[7] );

        _mm_store_ps( out1 +  0 , regs1[0] );
        _mm_store_ps( out1 +  4 , regs1[1] );
        _mm_store_ps( out1 +  8 , regs1[2] );
        _mm_store_ps( out1 + 12 , regs1[3] );
        _mm_store_ps( out1 + 16 , regs1[4] );
        _mm_store_ps( out1 + 20 , regs1[5] );
        _mm_store_ps( out1 + 24 , regs1[6] );
        _mm_store_ps( out1 + 28 , regs1[7] );

        regs2[0] = _mm256_load_ps( in +  0 );
        regs2[1] = _mm256_load_ps( in +  8 );
        regs2[2] = _mm256_load_ps( in + 16 );
        regs2[3] = _mm256_load_ps( in + 24 );

        _mm256_transpose_ARGBx8_to_Ax8Rx8Gx8Bx8_ps( regs2[0], regs2[1], regs2[2], regs2[3] );

        _mm256_store_ps( out2 +  0 , regs2[0] );
        _mm256_store_ps( out2 +  8 , regs2[1] );
        _mm256_store_ps( out2 + 16 , regs2[2] );
        _mm256_store_ps( out2 + 24 , regs2[3] );

        int o=0;
    }
    {
        ALIGN_FOR_AVX float in[8] =
        {
             1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f
        };
        ALIGN_FOR_AVX float out1[8] = {};
        ALIGN_FOR_AVX float out2[8] = {};
        ALIGN_FOR_AVX float out3[8] = {};

        f256A reg_1 { in     , simd_alignment::AVX };
        f256S reg_2 { in     , simd_alignment::SSE };
        f128S reg_3A{ in     , simd_alignment::SSE };
        f128S reg_3B{ in + 4 , simd_alignment::SSE };

        auto R1 = reg_1   > 3.0f;
        auto R2 = reg_2   > 3.0f;
        auto R3A= reg_3A  > 3.0f;
        auto R3B = reg_3B > 3.0f;

        R1.store( out1 , R1 );
        R2.store( out2 , R2 );
        R3A.store( out3 , R3A );
        R3B.store( out3 + 4 , R3B );

        auto R1m = R1.static_cast_to<int>().to_mask_32();
        auto R2m = R2.static_cast_to<int>().to_mask_32();
        auto R3Am = R3A.static_cast_to<int>().to_mask_32();
        auto R3Bm = R3B.static_cast_to<int>().to_mask_32();

        int i=0;
    }

    return 0;
}();

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