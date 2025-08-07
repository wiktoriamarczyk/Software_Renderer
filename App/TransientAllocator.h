/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once

#include "Common.h"
#include <memory_resource>

extern atomic<size_t> g_memory_resource_mem;

struct TransientMemoryAllocator
{
    TransientMemoryAllocator();
    ~TransientMemoryAllocator();
    void* do_allocate(std::size_t bytes, std::size_t alignment, std::pmr::synchronized_pool_resource& upstream );
    void reset();
private:
    struct Page;

    vector<unique_ptr<Page>>    m_Pages;
    vector<unique_ptr<Page>>    m_EmptyPages;
    Page*                       m_pPage = nullptr;
};

struct transient_memory_resource : public std::pmr::memory_resource
{
    void* do_allocate(std::size_t bytes, std::size_t alignment) override;
    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {}
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override{ return false; }

    void reset();
private:
    struct AllocSlot;
    struct AllocSlot;
    struct Context;
    static Context& GetContext();

    std::pmr::synchronized_pool_resource m_Fallback;
};

struct transient_allocator
{
    transient_allocator( transient_memory_resource& res ) noexcept
        : m_pResource( &res )
    {}

    template< typename T , typename ... ARGS >
    T* allocate( ARGS&& ... args )
    {
        return new(m_pResource->allocate(sizeof(T), alignof(T))) T(std::forward<ARGS>(args)...);
    }

    template< typename T , typename ... ARGS >
    span<T> allocate_array( size_t Count , ARGS&& ... args)
    {
        T* pMem = reinterpret_cast<T*>( m_pResource->allocate(sizeof(T)*Count, alignof(T)) );
        for( size_t i = 0 ; i < Count ; ++i )
            new(pMem+i) T(std::forward<ARGS>(args)...);

        return { pMem , Count };
    }
private:
    transient_memory_resource* m_pResource = nullptr;
};