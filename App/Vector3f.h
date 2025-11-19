/*
* Master’s thesis - Analysis of selected optimization techniques for a 3D software renderer
* Author: Wiktoria Marczyk
* Year: 2025
*/


#pragma once
#include "Common.h"
#include "Simd.h"

class Matrix4f;

inline void FastNormalize3( const float lpInput[3], float lpOutput[3])
{
    __m128 vInputA = _mm_load_ss(lpInput+0); // load input vector (x, y, z, a)
    __m128 vInputB = _mm_load_ss(lpInput+1); // load input vector (x, y, z, a)
    __m128 vInputC = _mm_load_ss(lpInput+2); // load input vector (x, y, z, a)

    __m128 vInputSqA = _mm_mul_ss(vInputA,vInputA); // square the input values
    __m128 vInputSqB = _mm_mul_ss(vInputB,vInputB); // square the input values
    __m128 vInputSqC = _mm_mul_ss(vInputC,vInputC); // square the input values

    vInputSqA = _mm_add_ss(vInputSqA, vInputSqB);
    vInputSqA = _mm_add_ss(vInputSqA, vInputSqC);

    vInputSqA = _mm_rsqrt_ss(vInputSqA); // compute the inverse sqrt

    _mm_store_ss( lpOutput+0 , _mm_mul_ss(vInputSqA, vInputA) );
    _mm_store_ss( lpOutput+1 , _mm_mul_ss(vInputSqA, vInputB) );
    _mm_store_ss( lpOutput+2 , _mm_mul_ss(vInputSqA, vInputC) );
}

template<typename T>
class Vector3;

inline void FastNormalize( Vector3<float>& v );

template<typename T>
class Vector3
{
public:
    Vector3()=default;
    Vector3(T x, T y, T z);

    template< typename U , typename ... A >
        requires HasConstructFrom<T,U,A...>
    Vector3( const U& Array , const A& ... args )
        : x{ Array[ 0 ] , args... }
        , y{ Array[ 1 ] , args... }
        , z{ Array[ 2 ] , args... }
    {}

    template< typename T2, eRoundMode RM = eRoundMode::Floor>
    constexpr Vector3<T2> ToVector3()const
    {
        if constexpr(!std::is_floating_point_v<T2>)
            return Vector3<T2>( static_cast<T2>(x), static_cast<T2>(y), static_cast<T2>(z));
        else if constexpr(RM == eRoundMode::Floor )
            return Vector3<T2>( static_cast<T2>(x), static_cast<T2>(y), static_cast<T2>(z));
        else if constexpr(RM == eRoundMode::Round )
            return Vector3<T2>( round(x), round(y), round(z));
        else
            return Vector3<T2>( ceil(x), ceil(y), ceil(z));
    }

    template<typename U , typename ... A>
        requires HasLoadFrom<T,U,A...>
    void load(const U& Array , const A& ... args)
    {
        x.load( Array[0] , args... );
        y.load( Array[1] , args... );
        z.load( Array[2] , args... );
    }

    template<typename U, typename ... A>
        requires HasStoreTo<T,U,A...>
    void store(U& Array , const A& ... args)const
    {
        x.store( Array[0] , args... );
        y.store( Array[1] , args... );
        z.store( Array[2] , args... );
    }

    Vector3 operator+(const Vector3& other)const;
    void    operator+=(const Vector3& other);
    Vector3 operator-(const Vector3& other)const;
    Vector3 operator*(const Vector3& other)const;
    void    operator*=(const Vector3& other);
    Vector3 operator*(const T& value)const;
    void    operator*=(const T& value);
    Vector3 operator/(const T& value)const;

    friend Vector3 operator*(T value, const Vector3& v)
    {
        return Vector3(v.x * value, v.y * value, v.z * value);
    }

          T* data()       { return &x; }
    const T* data() const { return &x; }

    template< int xp , int yp , int zp >
    Vector3 Swizzle()const
    {
        auto* pData = data();
        return Vector3( pData[xp], pData[yp], pData[zp] );
    }

    T GetLength()const;
    T Dot(const Vector3& other)const;
    T MaxComponent()const;
    Vector3& Normalize();
    Vector3 Normalized()const;
    Vector3 Multiplied(const Matrix4f& m) const;
    Vector3 Transformed(const Matrix4f& m) const;
    Vector3 TransformedVec(const Matrix4f& m) const;
    Vector3 Cross(const Vector3& other)const;
    Vector3 CWiseMin(const Vector3& other)const;
    Vector3 CWiseMax(const Vector3& other)const;
    Vector3 CWiseAbs(const Vector3& other)const;
    Vector3 Reflect(const Vector3& normal)const;

    inline Vector3& FastNormalize()
    {
        auto sumsq = x * x + y * y + z * z;
        Math::Rsqrt(sumsq,sumsq);
        x *= sumsq;
        y *= sumsq;
        z *= sumsq;
        return *this;
    }

    inline Vector3 FastNormalized()const
    {
        Vector3 result = *this;
        result.FastNormalize();
        return result;
    }

    T x{};
    T y{};
    T z{};
};

inline void FastNormalize( Vector3<float>& v )
{
    float sumsq = v.x * v.x + v.y * v.y + v.z * v.z;
    auto mm_sum = _mm_set_ss(sumsq); // set the sum of squares to the first element of the vector
    mm_sum = _mm_rsqrt_ss(mm_sum); // compute the inverse sqrt
    _mm_store_ss( &sumsq, mm_sum); // store the result back to sumsq

    v.x *= sumsq;
    v.y *= sumsq;
    v.z *= sumsq;
}

template< typename T >
inline Vector3<T>::Vector3(T x, T y, T z)
{
    this->x = x;
    this->y = y;
    this->z = z;
}

template< typename T >
inline Vector3<T> Vector3<T>::operator+(const Vector3& other)const
{
    return Vector3(x + other.x, y + other.y, z + other.z);
}

template< typename T >
inline void Vector3<T>::operator+=(const Vector3& other)
{
    this->x += other.x;
    this->y += other.y;
    this->z += other.z;
}

template< typename T >
inline Vector3<T> Vector3<T>::operator*(const Vector3& other)const
{
    return Vector3(x * other.x, y * other.y, z * other.z);
}

template< typename T >
inline void Vector3<T>::operator*=(const Vector3& other)
{
    this->x *= other.x;
    this->y *= other.y;
    this->z *= other.z;
}

template< typename T >
inline Vector3<T> Vector3<T>::operator-(const Vector3& other)const
{
    return Vector3(x - other.x, y - other.y, z - other.z);
}

template< typename T >
inline Vector3<T> Vector3<T>::operator*(const T& value)const
{
    return Vector3(x * value, y * value, z * value);
}
template< typename T >
inline void Vector3<T>::operator*=(const T& value)
{
    this->x *= value;
    this->y *= value;
    this->z *= value;
}

template< typename T >
inline Vector3<T> Vector3<T>::operator/(const T& value)const
{
    return Vector3(x / value, y / value, z / value);
}

template< typename T >
T Vector3<T>::GetLength()const
{
    return sqrt(x * x + y * y + z * z);
}

template< typename T >
Vector3<T>& Vector3<T>::Normalize()
{
    T length = GetLength();
    if (length == 0.0f)
    {
        // Handle the case where the vector is zero-length
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
        return *this;
    }
    else
    {
        T invLength = 1.0f / length;
        x *= invLength;
        y *= invLength;
        z *= invLength;
    }

    return *this;
}


template< typename T >
Vector3<T> Vector3<T>::Normalized()const
{
    T length = GetLength();
    if (length == 0.0f)
    {
        return Vector3(0.0f, 0.0f, 0.0f); // Return zero vector if length is zero
    }
    else
    {
        T invLength = 1.0f / length;
        return Vector3(x * invLength, y * invLength, z * invLength);
    }
}

template< typename T >
T Vector3<T>::Dot(const Vector3& other)const
{
    return x * other.x + y * other.y + z * other.z;
}

template< typename T >
T Vector3<T>::MaxComponent() const
{
    return Math::Max(Math::Max(x, y), z);
}

template< typename T >
Vector3<T> Vector3<T>::Cross(const Vector3& other)const
{
    return Vector3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
}

template< typename T >
Vector3<T> Vector3<T>::CWiseMin(const Vector3& other) const
{
    return Vector3(Math::Min(x, other.x), Math::Min(y, other.y), Math::Min(z, other.z));
}

template< typename T >
Vector3<T> Vector3<T>::CWiseMax(const Vector3& other) const
{
    return Vector3(Math::Max(x, other.x), Math::Max(y, other.y), Math::Max(z, other.z));
}

template< typename T >
Vector3<T> Vector3<T>::CWiseAbs(const Vector3& other) const
{
    return Vector3(std::abs(x), std::abs(y), std::abs(z));
}

template< typename T >
Vector3<T> Vector3<T>::Reflect(const Vector3& normal) const
{
    return *this - normal * 2.f * Dot(normal);
}

using Vector3f = Vector3<float>;
using Vector3i = Vector3<int>;

//template< typename T , int Rank , eSimdType Type = eSimdType::None >
//struct SimdVector3
//{
//    simd<T,Rank,Type> x = {};
//    simd<T,Rank,Type> y = {};
//    simd<T,Rank,Type> z = {};
//};

template< eSimdType Type = eSimdType::SSE >
using Vector3f128 = Vector3< f128<Type> >;

template< eSimdType Type = eSimdType::SSE >
using Vector3f256 = Vector3< f256<Type> >;

template< eSimdType Type = eSimdType::SSE >
using Vector3i256 = Vector3< i256<Type> >;

using Vector3f128S = Vector3f128<eSimdType::SSE>;
using Vector3f128S8= Vector3f256<eSimdType::SSE>;
using Vector3f256A = Vector3f256<eSimdType::AVX>;
using Vector3f256C = Vector3f256<eSimdType::CPU>;
using Vector3i256A = Vector3i256<eSimdType::AVX>;