/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#include "TransientAllocator.h"
#include "Math.h"

atomic<size_t> g_memory_resource_mem;

struct TransientMemoryAllocator::Page
{
    void* allocate( size_t bytes, size_t alignment , std::pmr::synchronized_pool_resource& upstream );

    constexpr Page()=default;
    virtual ~Page();
    void reset();

    struct PageImpl;
protected:
    Page( uint8_t* pMem , uint32_t size );
private:
    uint8_t*                m_pCurPos   = nullptr;
    uint32_t                m_FreeLeft  = 0;
    uint8_t*const           m_pStartPos = nullptr;
    const uint32_t          m_MaxSize   = 0xFFFFFF;
    const uint8_t*const     m_pEnd      = nullptr;
};

TransientMemoryAllocator::Page::Page( uint8_t* pMem , uint32_t size )
    : m_pCurPos( pMem )
    , m_FreeLeft( size )
    , m_pStartPos( pMem )
    , m_MaxSize( size )
    , m_pEnd( pMem + size )
{
    g_memory_resource_mem.fetch_add( m_MaxSize , std::memory_order_relaxed );
}

struct TransientMemoryAllocator::Page::PageImpl : Page
{
    constexpr static inline auto SIZE = 512*1024;

    PageImpl()
        : Page( m_Buffer , SIZE )
    {}
private:
    alignas(64) uint8_t m_Buffer[SIZE];
};

TransientMemoryAllocator::Page::~Page()
{
    if( m_pStartPos )
        g_memory_resource_mem.fetch_sub( m_MaxSize , std::memory_order_relaxed );
}

void TransientMemoryAllocator::Page::reset()
{
    m_pCurPos = m_pStartPos;
    m_FreeLeft = m_MaxSize;
}

FORCE_INLINE void* TransientMemoryAllocator::Page::allocate( size_t bytes, size_t alignment , std::pmr::synchronized_pool_resource& upstream )
{
    ZoneScoped;
    static_assert( IsPowerOfTwo( AVX_ALIGN ) );

    alignment = Granulate<size_t>(alignment, AVX_ALIGN);
    bytes = Granulate<size_t>(bytes, alignment);
    if( alignment > 64 || bytes > m_MaxSize/4 )
        return upstream.allocate(bytes, alignment);

    if( m_pCurPos + bytes > m_pEnd )
        return nullptr;

    auto pMem = m_pCurPos;
    m_pCurPos += bytes;
    return pMem;
}

struct TransientMemoryAllocator::Storage
{
    Storage();
    ~Storage();
    void FreeSlot( TLSlot& slot );
    static Storage* RegisterSlot( TLSlot& slot );
    Page* GetNextFreePage();
    void reset();
    static Storage& Get()
    {
        static Storage storage;
        return storage;
    }
private:
    vector<unique_ptr<Page>>    m_Pages;
    vector<unique_ptr<Page>>    m_EmptyPages;
    vector<TLSlot*>             m_Slots;
    std::recursive_mutex        m_Mutex;
};

struct TransientMemoryAllocator::TLSlot
{
    friend class Storage;

    constexpr TLSlot()
    {
        m_pPage = &s_EmptyPage;
    }

    FORCE_INLINE void* allocate( size_t bytes, size_t alignment , std::pmr::synchronized_pool_resource& upstream )
    {
        auto pMem = m_pPage->allocate( bytes, alignment, upstream );
        if( pMem )
            return pMem;

        if( !m_pStorage )
            m_pStorage = Storage::RegisterSlot(*this);

        if( m_pStorage )
        {
            if( auto pNewPage = m_pStorage->GetNextFreePage() )
                m_pPage = pNewPage;
        }

        pMem = m_pPage->allocate( bytes, alignment, upstream );
        return pMem;
    }

    void reset()
    {
        m_pPage = &s_EmptyPage;
    }

    ~TLSlot()
    {
        if( m_pPage && m_pStorage )
            m_pStorage->FreeSlot(*this);
    }

private:
    Page*       m_pPage     = nullptr;
    Storage*    m_pStorage  = nullptr;
    static inline Page s_EmptyPage ;
};

constinit thread_local TransientMemoryAllocator::TLSlot TransientMemoryAllocator::s_TLSlot;

TransientMemoryAllocator::Storage::Storage()
{
    m_Pages.reserve(64);
    m_EmptyPages.reserve(64);
}

TransientMemoryAllocator::Storage::~Storage()
{
    std::scoped_lock lock(m_Mutex);

    for( auto& TLSlot : m_Slots )
    {
        if( TLSlot->m_pStorage == this )
        {
            TLSlot->m_pStorage = nullptr;
            TLSlot->reset();
        }
    }
}

void TransientMemoryAllocator::Storage::FreeSlot( TLSlot& slot )
{
    if( !slot.m_pStorage )
        return;

    ZoneScoped;

    std::scoped_lock lock(m_Mutex);

    m_Slots.erase(std::remove(m_Slots.begin(), m_Slots.end(), &slot), m_Slots.end());
    slot.m_pStorage = nullptr;
}

TransientMemoryAllocator::Storage* TransientMemoryAllocator::Storage::RegisterSlot( TLSlot& slot )
{
    if( slot.m_pStorage )
        return slot.m_pStorage;

    ZoneScoped;

    auto& Storage = Get();
    std::scoped_lock lock(Storage.m_Mutex);

    Storage.m_Slots.push_back(&slot);
    slot.m_pStorage = &Storage;
    return slot.m_pStorage;
}

TransientMemoryAllocator::Page* TransientMemoryAllocator::Storage::GetNextFreePage()
{
    ZoneScoped;
    std::scoped_lock lock(m_Mutex);

    Page* pPage = nullptr;

    if( m_EmptyPages.empty() )
    {
        ZoneScopedN("Allocate new page");
        m_Pages.push_back(make_unique<Page::PageImpl>());
        pPage = m_Pages.back().get();
    }
    else
    {
        m_Pages.push_back(std::move(m_EmptyPages.back()));
        m_EmptyPages.pop_back();
        pPage = m_Pages.back().get();
    }
    return pPage;
};

void TransientMemoryAllocator::Storage::reset()
{
    ZoneScoped;
    std::scoped_lock lock(m_Mutex);

    for( auto& pSlot : m_Slots )
        pSlot->reset();

    for( auto& pPage : m_Pages )
    {
        pPage->reset();
        m_EmptyPages.push_back(std::move(pPage));
    }

    m_Pages.clear();
}

FORCE_INLINE void* TransientMemoryAllocator::do_allocate(std::size_t bytes, std::size_t alignment, std::pmr::synchronized_pool_resource& upstream )
{
    return s_TLSlot.allocate(bytes, alignment, upstream);
}

void TransientMemoryAllocator::reset()
{
    ZoneScoped;
    auto& Storage = Storage::Get();
    Storage.reset();
};

void* transient_memory_resource::do_allocate(std::size_t bytes, std::size_t alignment)
{
    return TransientMemoryAllocator::do_allocate(bytes, alignment, m_Fallback);
}

void transient_memory_resource::reset()
{
    ZoneScoped;
    TransientMemoryAllocator::reset();
    m_Fallback.release();
};

monotonic_stack_unsynchronized_memory_resource::monotonic_stack_unsynchronized_memory_resource( uint32_t Size , uint32_t BaseAlign , std::pmr::memory_resource& Upstream )
    : m_Upstream(Upstream)
{
    m_pStart = static_cast<uint8_t*>( Upstream.allocate(Size, BaseAlign) );
    m_pEnd = m_pStart + Size;
    m_pCurPos = m_pStart;

    m_Size     = static_cast<size_t>(Size);
    m_Alignment = static_cast<uint8_t>(BaseAlign);
}

monotonic_stack_unsynchronized_memory_resource::~monotonic_stack_unsynchronized_memory_resource()
{
    ZoneScoped;
    if( m_pStart )
        m_Upstream.deallocate(m_pStart, m_Size, m_Alignment);
}

void* monotonic_stack_unsynchronized_memory_resource::do_allocate(std::size_t bytes, std::size_t alignment)
{
    ZoneScoped;

    auto pMem = std::align( alignment , bytes , m_pCurPos, m_MemLeft );
    if( !pMem )
        return m_Upstream.allocate(bytes, alignment);

    return pMem;
}

void monotonic_stack_unsynchronized_memory_resource::do_deallocate(void* p, std::size_t bytes, std::size_t alignment)
{
    ZoneScoped;
    if( p < m_pStart || p >= m_pEnd )
        // If the pointer is outside the allocated range, delegate to upstream
        return m_Upstream.deallocate(p, bytes, alignment);

    m_pCurPos = static_cast<uint8_t*>(p);
    m_MemLeft = m_pEnd - static_cast<uint8_t*>( m_pCurPos );
}

bool monotonic_stack_unsynchronized_memory_resource::do_is_equal(const std::pmr::memory_resource& other) const noexcept
{
    return &other == this;
}