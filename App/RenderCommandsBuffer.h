/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "RenderCommands.h"
#include "SimpleThreadPool.h"

struct CommandBuffer
{
    const Command* GetNextCommand() noexcept;

    template< std::derived_from<Command> T , typename ... ARGS >
    void PushCommand( ARGS&& ... args ) noexcept;
    bool PushCommandBuffer( CommandBuffer& Cmd )noexcept;
    void AddSyncBarrier( const char* Name , uint8_t Count = 1 );
    void AddSyncBarrier( const char* Name , triviall_function_ref Callabck , uint8_t Count = 1 );
    void AddSyncPoint( ISyncBarier& Sync , uint8_t Count = 1 );
    static CommandBuffer* CreateCommandBuffer( transient_memory_resource& Res ) noexcept;
protected:
    friend class transient_allocator;
    struct Context;

    CommandBuffer( transient_memory_resource& Res );

    span<const Command*> GetCommands() const noexcept;
    void PushCommandImpl( Command* pCmd ) noexcept;
    bool PushCommands( span<const Command*> CommandsToAdd ) noexcept;
    bool CheckGrow( Context*& pContext );
    void GrowCommandBuffer( Context*& pContext , size_t ExtraSize = 0 ) noexcept;

    using CommandAllocator      = pmr::polymorphic_allocator<const Command*>;

    struct Context
    {
    atomic<const Command**>     m_pCurReadCommand  = nullptr;
    atomic<const Command**>     m_pCurWriteCommand = nullptr;
    const Command**             m_pFirstCommand    = nullptr;
    const Command**             m_pEndCommand      = nullptr;
    };

    atomic<Context*>            m_pContext          = &m_BaseContext;
    transient_memory_resource&  m_MemoryResource;
    CommandAllocator            m_CmdAllocator{ &m_MemoryResource };
    pmr::vector<const Command*> m_Commands;
    Spinlock                    m_CmdBufSpinLock;
    transient_allocator         m_Allocator        { m_MemoryResource };
    Context                     m_BaseContext;

};

inline span<const Command*> CommandBuffer::GetCommands() const noexcept
{
    auto pContext = m_pContext.load(std::memory_order_acquire);
    return { pContext->m_pFirstCommand , pContext->m_pCurWriteCommand.load() };
}

inline void CommandBuffer::PushCommandImpl( Command* pCmd ) noexcept
{
    auto pContext = m_pContext.load( std::memory_order_acquire );
    auto pCur = pContext->m_pCurWriteCommand.load( std::memory_order_relaxed );
    for(;;)
    {
        if( pContext->m_pCurWriteCommand == pContext->m_pEndCommand )
        {
            GrowCommandBuffer( pContext );
            continue;
        }

        if( pContext->m_pCurWriteCommand.compare_exchange_weak( pCur , pCur+1 , std::memory_order_acq_rel ) )
            break;
    }

    *pCur = pCmd;
}

inline bool CommandBuffer::PushCommands( span<const Command*> CommandsToAdd ) noexcept
{
    auto pContext = m_pContext.load( std::memory_order_acquire );

    for(;;)
    {
        const Command** pCur = pContext->m_pCurWriteCommand.load( std::memory_order_relaxed );
        if( pCur+CommandsToAdd.size() == pContext->m_pEndCommand )
        {
            GrowCommandBuffer( pContext , CommandsToAdd.size() );
            continue;
        }

        if( !pContext->m_pCurWriteCommand.compare_exchange_weak( pCur , pCur+CommandsToAdd.size() , std::memory_order_acq_rel ) )
            continue;

        memcpy( pCur , CommandsToAdd.data() , CommandsToAdd.size_bytes() );
        return true;
    }
};

inline const Command* CommandBuffer::GetNextCommand() noexcept
{
    auto pContext = m_pContext.load( std::memory_order_acquire );
    for( ;; )
    {
        auto pRead = pContext->m_pCurReadCommand.load( std::memory_order_relaxed );

        if( pRead == pContext->m_pEndCommand )
        {
            if( CheckGrow( pContext ) )
                continue;

            return nullptr;
        }

        // try to acquire the next command
        if( !pContext->m_pCurReadCommand.compare_exchange_weak( pRead , pRead+1 , std::memory_order_acq_rel ) )
            // failed to acquire, try again
            continue;

        return *pRead;
    }
}

inline bool CommandBuffer::PushCommandBuffer( CommandBuffer& Cmd ) noexcept
{
    if( &Cmd == this )
        return false; // cannot push self

    span<const Command*> CommandsToAdd = Cmd.GetCommands();

    return PushCommands( CommandsToAdd );
};

template< std::derived_from<Command> T , typename ... ARGS >
inline void CommandBuffer::PushCommand( ARGS&& ... args ) noexcept
{
    auto pCommand = m_Allocator.allocate<T>( std::forward<ARGS>(args)... );
    return PushCommandImpl( pCommand );
}

inline void CommandBuffer::AddSyncBarrier( const char* Name , uint8_t Count )
{
    auto pSync = m_Allocator.allocate<SyncBarrier>();
    pSync->name = Name;
    pSync->m_Barrier.emplace(Count);

    AddSyncPoint( *pSync , Count );
}

inline void CommandBuffer::AddSyncBarrier( const char* Name , triviall_function_ref Callabck , uint8_t Count )
{
    auto pSync = m_Allocator.allocate<SyncBarrier>();
    pSync->name = Name;
    pSync->m_Barrier.emplace(Count, std::move(Callabck));

    AddSyncPoint( *pSync , Count );
}

inline void CommandBuffer::AddSyncPoint( ISyncBarier& Sync , uint8_t Count )
{
    if( Count == 1 )
    {
        PushCommand<CommandSyncBarier>(Sync);
    }
    else if( Count > 1 )
    {
        auto Commands = m_Allocator.allocate_array<CommandSyncBarier>( Count , Sync );
        auto CommandPtrs = m_Allocator.allocate_array<const Command*>( Count , nullptr );

        for(  size_t i = 0 ; i < Count ; ++i )
            CommandPtrs[i] = &Commands[i];


        PushCommands( CommandPtrs );
    }
}