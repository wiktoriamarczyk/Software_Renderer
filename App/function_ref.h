/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Common.h"
#include "TransientAllocator.h"

struct triviall_function_ref
{
    using FuncT = void(*)(void*);
public:
    triviall_function_ref() = default;
    template< typename T >
        requires( std::is_trivially_copy_constructible_v<T> && std::is_trivially_destructible_v<T> && std::invocable<T> )
    triviall_function_ref& Assign( transient_memory_resource& A , const T& Func )
    {
        pData = A.allocate(sizeof(T), alignof(T));
        memcpy(pData, &Func, sizeof(T));
        pFunc = [](void* pData)
        {
            auto& func = *static_cast<T*>(pData);
            func();
        };
        return *this;
    }
    void operator()()const noexcept
    {
        if( pFunc )
            pFunc(pData);
    }
    bool IsValid()const noexcept
    {
        return pFunc != nullptr;
    }
    void reset() noexcept
    {
        pFunc = nullptr;
        pData = nullptr;
    }
private:
    FuncT pFunc = nullptr;
    void* pData = nullptr;
};