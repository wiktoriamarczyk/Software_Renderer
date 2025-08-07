/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#include "TransientAllocator.h"
#include "Math.h"

atomic<size_t> g_memory_resource_mem;

struct alignas(64) TransientMemoryAllocator::Page
{
    constexpr static inline auto SIZE = 256*1024;

    Page();
    ~Page();
    void* allocate( size_t bytes, size_t alignment , std::pmr::synchronized_pool_resource& upstream );
    void reset();

    array<uint8_t, SIZE>    m_Buffer;
    uint8_t*                m_pCurPos   = m_Buffer.data();
    uint32_t                m_FreeLeft  = SIZE;
    const uint8_t*const     m_pEnd      = m_Buffer.data() + m_Buffer.size();
};


TransientMemoryAllocator::Page::Page()
{
    g_memory_resource_mem.fetch_add( sizeof(Page) , std::memory_order_relaxed );
}

TransientMemoryAllocator::Page::~Page()
{
    g_memory_resource_mem.fetch_sub( sizeof(Page) , std::memory_order_relaxed );
}

void TransientMemoryAllocator::Page::reset()
{
    m_pCurPos = m_Buffer.data();
    m_FreeLeft = SIZE;
}

void* TransientMemoryAllocator::Page::allocate( size_t bytes, size_t alignment , std::pmr::synchronized_pool_resource& upstream )
{
    static_assert( IsPowerOfTwo( AVX_ALIGN ) );

    alignment = Granulate<size_t>(alignment, AVX_ALIGN);
    bytes = Granulate<size_t>(bytes, alignment);
    if( alignment > 64 || bytes > SIZE/4 )
        return upstream.allocate(bytes, alignment);

    if( m_pCurPos + bytes > m_pEnd )
        return nullptr;

    auto pMem = m_pCurPos;
    m_pCurPos += bytes;
    return pMem;
}

TransientMemoryAllocator::TransientMemoryAllocator()
{
    m_Pages.reserve(64);
    m_Pages.reserve(64);

    m_Pages.push_back(make_unique<Page>());
    m_pPage = m_Pages.back().get();
}

TransientMemoryAllocator::~TransientMemoryAllocator()
{
}

void* TransientMemoryAllocator::do_allocate(std::size_t bytes, std::size_t alignment, std::pmr::synchronized_pool_resource& upstream )
{
    ZoneScoped;
    auto pMem = m_pPage->allocate(bytes, alignment, upstream);
    if( pMem )
        return pMem;

    if( m_EmptyPages.empty() )
    {
        ZoneScopedN("Allocate new page");
        m_Pages.push_back(make_unique<Page>());
        m_pPage = m_Pages.back().get();
    }
    else
    {
        m_Pages.push_back(std::move(m_EmptyPages.back()));
        m_EmptyPages.pop_back();
        m_pPage = m_Pages.back().get();
    }

    return m_pPage->allocate(bytes, alignment,upstream);
}

void TransientMemoryAllocator::reset()
{
    m_pPage = nullptr;

    for( auto& pPage : m_Pages )
    {
        pPage->reset();
        m_EmptyPages.push_back(std::move(pPage));
    }

    m_Pages.clear();

    if( m_EmptyPages.empty() )
        return;

    m_Pages.push_back( std::move( m_EmptyPages.back() ) );
    m_EmptyPages.pop_back();

    m_pPage = m_Pages.back().get();
};


struct transient_memory_resource::Context
{
    vector<AllocSlot*> m_AllocSlots;
    std::recursive_mutex m_AllocSlotsMutex;
};

struct transient_memory_resource::AllocSlot
{
    AllocSlot()
    {
        auto& ctx = GetContext();
        std::scoped_lock lock(ctx.m_AllocSlotsMutex);
        ctx.m_AllocSlots.push_back(this);
    }
    ~AllocSlot()
    {
        auto& ctx = GetContext();
        std::scoped_lock lock(ctx.m_AllocSlotsMutex);
        auto it = std::find(ctx.m_AllocSlots.begin(), ctx.m_AllocSlots.end(), this);
        if (it != ctx.m_AllocSlots.end())
            ctx.m_AllocSlots.erase(it);
    }

    TransientMemoryAllocator m_pAlloc;

    static thread_local optional<AllocSlot> s_Allocator;
};

thread_local constinit optional<transient_memory_resource::AllocSlot> transient_memory_resource::AllocSlot::s_Allocator;

void* transient_memory_resource::do_allocate(std::size_t bytes, std::size_t alignment)
{
    ZoneScoped;
    if( !AllocSlot::s_Allocator )
        AllocSlot::s_Allocator.emplace();

    return AllocSlot::s_Allocator->m_pAlloc.do_allocate(bytes, alignment, m_Fallback);
}

void transient_memory_resource::reset()
{
    ZoneScoped;
    auto& ctx = GetContext();

    std::scoped_lock lock(ctx.m_AllocSlotsMutex);

    for( auto& pSlot : ctx.m_AllocSlots )
        pSlot->m_pAlloc.reset();
};

transient_memory_resource::Context& transient_memory_resource::GetContext()
{
    static Context ctx;
    return ctx;
};