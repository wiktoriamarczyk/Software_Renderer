/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Common.h"
#include "Simd.h"

class Matrix4f;
template< typename T >
class Vector3;


template< typename T >
class Vector2 {
public:
    constexpr Vector2() = default;
    constexpr Vector2(T x, T y);
              Vector2(Vector3<T> vector);

    template< typename U , typename ... A >
        requires HasConstructFrom<T,U,A...>
    Vector2( const U& Array , const A& ... args )
        : x{ Array[ 0 ] , args... }
        , y{ Array[ 1 ] , args... }
    {}

    template< typename T2 , eRoundMode RM = eRoundMode::Floor >
    constexpr Vector2<T2> ToVector2()const
    {
        if constexpr( !std::is_floating_point_v<T2> )
            return Vector2<T2>( static_cast<T2>(x) , static_cast<T2>(y) );
        else if constexpr( RM == eRoundMode::Floor )
            return Vector2<T2>( static_cast<T2>(x), static_cast<T2>(y) );
        else if constexpr( RM == eRoundMode::Round )
            return Vector2<T2>( round(x) , round(y) );
        else
            return Vector2<T2>( ceil(x) , ceil(y) );
    }

    template< typename U , typename ... A >
        requires HasLoadFrom<T,U,A...>
    void load( const U& Array , const A& ... args )
    {
        x.load( Array[0] , args... );
        y.load( Array[1] , args... );
    }

    template< typename U , typename ... A >
        requires HasStoreTo<T,U,A...>
    void store( U& Array , const A& ... args )const
    {
        x.store( Array[0] , args... );
        y.store( Array[1] , args... );
    }

    constexpr Vector2<int> ToVector2i() const requires(std::is_floating_point<T>::value)
    {
        return ToVector2<int>();
    }
    template< eRoundMode RM = eRoundMode::Floor >
    constexpr Vector2<float> ToVector2f() const requires(std::is_integral<T>::value)
    {
        return ToVector2<float, RM>();
    }


    constexpr bool operator==(const Vector2& other)const
    {
        return x == other.x && y == other.y;
    }
    constexpr Vector2 operator+(Vector2 other)const;
    constexpr void    operator+=(Vector2 other);
    constexpr Vector2 operator-(Vector2 other)const;
    constexpr Vector2 operator*(Vector2 other)const;
    constexpr Vector2 operator/(Vector2 other)const;
    constexpr Vector2 operator+(T value)const;
    constexpr Vector2 operator-(T value)const;
    constexpr Vector2 operator*(T value)const;
    constexpr Vector2 operator/(T value)const;
    constexpr Vector2 operator-()const;

    constexpr friend Vector2 operator*(T value, const Vector2& v)
    {
        return Vector2(v.x * value, v.y * value);
    }

    constexpr Vector2 Rotated90()const noexcept
    {
        return Vector2(-y, x);
    }

    constexpr operator ImVec2() const noexcept
    {
        return ImVec2(x, y);
    }

    T GetLength()const;
    constexpr T Dot(const Vector2& other)const;
    Vector2& Normalize();
    Vector2 Normalized()const;
    constexpr Vector2 CWiseMin(const Vector2& other)const;
    constexpr Vector2 CWiseMax(const Vector2& other)const;

    T* Data() { return &x; }
    const T* Data() const { return &x; }

    T x{};
    T y{};
};

template< typename T >
constexpr inline Vector2<T>::Vector2(T x, T y)
{
    this->x = x;
    this->y = y;
}

template< typename T >
constexpr inline Vector2<T> Vector2<T>::operator+(Vector2 other)const
{
    return Vector2(x + other.x, y + other.y);
}

template< typename T >
constexpr inline void Vector2<T>::operator+=(Vector2 other)
{
    this->x += other.x;
    this->y += other.y;
}

template< typename T >
constexpr inline Vector2<T> Vector2<T>::operator-(Vector2 other)const
{
    return Vector2(x - other.x, y - other.y);
}

template< typename T >
constexpr inline Vector2<T> Vector2<T>::operator*(Vector2 other)const
{
    return Vector2(x * other.x, y * other.y);
}

template< typename T >
constexpr inline Vector2<T> Vector2<T>::operator/(Vector2 other)const
{
    return Vector2(x / other.x, y / other.y);
}

template< typename T >
constexpr inline Vector2<T> Vector2<T>::operator+(T value)const
{
    return Vector2(x + value, y + value);
}

template< typename T >
constexpr inline Vector2<T> Vector2<T>::operator-(T value)const
{
    return Vector2(x - value, y - value);
}

template< typename T >
constexpr inline Vector2<T> Vector2<T>::operator*(T value)const
{
    return Vector2(x * value, y * value);
}

template< typename T >
constexpr inline Vector2<T> Vector2<T>::operator/(T value)const
{
    return Vector2(x / value, y / value);
}

template< typename T >
constexpr inline Vector2<T> Vector2<T>::operator-()const
{
    return Vector2(-x,-y);
}

template< typename T >
Vector2<T>::Vector2(Vector3<T> vector)
{
    x = vector.x;
    y = vector.y;
}

template< typename T >
T Vector2<T>::GetLength()const
{
    return T(sqrt(x * x + y * y));
}

template< typename T >
Vector2<T>& Vector2<T>::Normalize()
{
    float length = GetLength();
    x = x / length;
    y = y / length;

    return *this;
}

template< typename T >
Vector2<T> Vector2<T>::Normalized()const
{
    return Vector2(x / GetLength(), y / GetLength());
}


template< typename T >
constexpr T Vector2<T>::Dot(const Vector2& other)const
{
    return x * other.x + y * other.y;
}

template< typename T >
constexpr Vector2<T> Vector2<T>::CWiseMin(const Vector2& other) const
{
    return Vector2(std::min(x, other.x), std::min(y, other.y));
}

template< typename T >
constexpr Vector2<T> Vector2<T>::CWiseMax(const Vector2& other) const
{
    return Vector2(std::max(x, other.x), std::max(y, other.y));
}

using Vector2f = Vector2<float>;
using Vector2i = Vector2<int>;
using Vector2si= Vector2<int16_t>;

template< eSimdType Type = eSimdType::SSE >
using Vector2f128 = Vector2< f128<Type> >;

template< eSimdType Type = eSimdType::SSE >
using Vector2f256 = Vector2< f256<Type> >;
