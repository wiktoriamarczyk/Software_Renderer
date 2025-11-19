/*
* Master’s thesis - Analysis of selected optimization techniques for a 3D software renderer
* Author: Wiktoria Marczyk
* Year: 2025
*/

#pragma once
#include "Common.h"
#include "Vector3f.h"
#include "Simd.h"

class Matrix4f;

template< typename T >
class Vector4
{
public:
    constexpr Vector4() = default;
    constexpr Vector4(const Vector2<T>& v,T z, T w);
    constexpr Vector4(const Vector3<T>& v,T w);
    constexpr Vector4(T x, T y, T z, T w);

    template< typename U , typename ... A >
        requires HasConstructFrom<T,U,A...>
    constexpr Vector4( const U& Array , const A& ... args )
        : x{ Array[0] , args... }
        , y{ Array[1] , args... }
        , z{ Array[2] , args... }
        , w{ Array[3] , args... }
    {}

    constexpr Vector4<T> operator+(Vector4<T> other)const;
    constexpr Vector4<T> operator-(Vector4<T> other)const;
    constexpr Vector4<T> operator*(Vector4<T> other)const;
    constexpr Vector4<T> operator*(T value)const;
    constexpr Vector4<T> operator+(T value)const;
    constexpr Vector4<T> operator/(T value)const;

    constexpr friend Vector4<T> operator*(T value, const Vector4& v)
    {
        return Vector4(v.x * value, v.y * value, v.z * value, v.w * value);
    }

    template< typename U , typename ... A >
        requires HasLoadFrom<T,U,A...>
    void load( const U& Array , const A& ... args )
    {
        x.load( Array[0] , args... );
        y.load( Array[1] , args... );
        z.load( Array[2] , args... );
        w.load( Array[3] , args... );
    }

    template< typename U , typename ... A >
        requires HasStoreTo<T,U,A...>
    void store( U& Array , const A& ... args )const
    {
        x.store( Array[0] , args... );
        y.store( Array[1] , args... );
        z.store( Array[2] , args... );
        w.store( Array[3] , args... );
    }

    T GetLength()const;
    T Dot(const Vector4& other)const;
    Vector4& Normalize();
    Vector4<T> Normalized()const;
    Vector4<T> Transformed(const Matrix4f& m) const;
    Vector4<T> CWiseMin(const Vector4& other)const;
    Vector4<T> CWiseMax(const Vector4& other)const;

          Vector2<T>& xy();
    const Vector2<T>& xy()const;

          Vector3<T>& xyz();
    const Vector3<T>& xyz()const;

          T* Data() { return &x; }
    const T* Data() const { return &x; }

          T* data()       { return &x; }
    const T* data() const { return &x; }

    constexpr static uint32_t ToARGB(const Vector4& color);
    constexpr inline uint32_t ToARGB()const{ return ToARGB(*this); }
    constexpr static Vector4<T> FromARGB(uint32_t color);

    T x{};
    T y{};
    T z{};
    T w{};
};

template< typename T >
constexpr inline Vector4<T>::Vector4(T x, T y, T z, T w)
{
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
}

template< typename T >
constexpr inline Vector4<T>::Vector4(const Vector2<T>& v, T z, T w)
{
    this->x = v.x;
    this->y = v.y;
    this->z = z;
    this->w = w;
}

template< typename T >
constexpr inline Vector4<T>::Vector4(const Vector3<T>& v, T w)
{
    this->x = v.x;
    this->y = v.y;
    this->z = v.z;
    this->w = w;
}

template<typename T>
constexpr inline Vector4<T> Vector4<T>::operator+(Vector4<T> other)const
{
    return Vector4(x + other.x, y + other.y, z + other.z, w + other.w);
}

template<typename T>
constexpr inline Vector4<T> Vector4<T>::operator-(Vector4<T> other)const
{
    return Vector4(x - other.x, y - other.y, z - other.z, w - other.w);
}

template<typename T>
constexpr inline Vector4<T> Vector4<T>::operator*(Vector4<T> other) const
{
    return Vector4(x * other.x, y * other.y, z * other.z, w * other.w);
}

template<typename T>
constexpr inline Vector4<T> Vector4<T>::operator*(T value)const
{
    return Vector4(x * value, y * value, z * value, w * value);
}

template< typename T >
constexpr inline Vector4<T> Vector4<T>::operator+(T value)const
{
    return Vector4(x + value, y + value, z + value, w + value);
}

template< typename T >
constexpr inline Vector4<T> Vector4<T>::operator/(T value)const
{
    return Vector4(x / value, y / value, z / value, w / value);
}

template<typename T>
inline const Vector3<T>& Vector4<T>::xyz() const
{
    return reinterpret_cast<const Vector3<T>&>(*this);
}

template<typename T>
inline Vector3<T>& Vector4<T>::xyz()
{
    return reinterpret_cast<Vector3<T>&>(*this);
}

template<typename T>
inline Vector2<T>& Vector4<T>::xy()
{
    return reinterpret_cast<Vector2<T>&>(*this);
}

template<typename T>
inline const Vector2<T>& Vector4<T>::xy()const
{
    return reinterpret_cast<const Vector2<T>&>(*this);
}

template<typename T>
constexpr inline uint32_t Vector4<T>::ToARGB(const Vector4& color)
{
    return (uint32_t)(color.w * 255) << 24 | (uint32_t)(color.z * 255) << 16 | (uint32_t)(color.y * 255) << 8 | (uint32_t)(color.x * 255);
}

template<typename T>
T Vector4<T>::GetLength()const
{
    return sqrt(x * x + y * y + z * z + w * w);
}

template<typename T>
Vector4<T>& Vector4<T>::Normalize()
{
    auto length = GetLength();
    x = x / length;
    y = y / length;
    z = z / length;
    w = w / length;

    return *this;
}

template<typename T>
Vector4<T> Vector4<T>::Normalized()const
{
    return Vector4<T>(x / GetLength(), y / GetLength(), z / GetLength(), w / GetLength());
}

template<typename T>
T Vector4<T>::Dot(const Vector4<T>& other)const
{
    return x * other.x + y * other.y + z * other.z + w * other.w;
}

template<typename T>
Vector4<T> Vector4<T>::CWiseMin(const Vector4<T>& other) const
{
    return Vector4<T>(std::min(x, other.x), std::min(y, other.y), std::min(z, other.z), std::min(w, other.w));
}

template<typename T>
Vector4<T> Vector4<T>::CWiseMax(const Vector4<T>& other) const
{
    return Vector4<T>(std::max(x, other.x), std::max(y, other.y), std::max(z, other.z), std::max(w, other.w));
}


template<typename T>
constexpr Vector4<T> Vector4<T>::FromARGB(uint32_t color)
{
    return Vector4((T)(color & 0xFF), (T)((color >> 8) & 0xFF), (T)((color >> 16) & 0xFF), (T)((color >> 24))) * 1.0f / 255.0f;
}

using Vector4f = Vector4<float>;

template<eSimdType Type = eSimdType::SSE>
using Vector4f128 = Vector4< f128<Type>>;

template<eSimdType Type = eSimdType::SSE>
using Vector4f256 = Vector4< f256<Type>>;

using Vector4f128A = Vector4f128<eSimdType::AVX>;
using Vector4f256A = Vector4f256<eSimdType::AVX>;