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

    void PushCommand( span<const Command*const> ) noexcept;

    void PushCommandBuffer( CommandBuffer& Buf );

    void AddSyncBarrier( const char* Name , uint8_t Count = 1 );
    void AddSyncBarrier( const char* Name , triviall_function_ref Callabck , uint8_t Count = 1 );
    void AddSyncPoint( ISyncBarier& Sync , uint8_t Count = 1 );
    static CommandBuffer* CreateCommandBuffer( transient_memory_resource& Res ) noexcept;
    static CommandBuffer* CreateCommandBuffer( transient_memory_resource& Res , span<const Command*const> Commands , bool ExactSize ) noexcept;
    void Finish()
    {
        PushCommandImpl( EncodedCommandPtr{ nullptr , eCommandPtrKind::End } );
    }
protected:
    friend class transient_allocator;
    struct Context;

    CommandBuffer( transient_memory_resource& Res );
    CommandBuffer( transient_memory_resource& Res ,span<const Command*const> Commands, bool ExactSize);

    void PushCommandImpl( eEncodedCommandPtr pCmd ) noexcept;

    void ReserveComands( span<eEncodedCommandPtr*> Res ) noexcept;

    using CommandAllocator      = pmr::polymorphic_allocator<const Command*>;

    transient_memory_resource&  m_MemoryResource;
    CommandAllocator            m_CmdAllocator{ &m_MemoryResource };
    Spinlock                    m_CmdBufSpinLock;
    transient_allocator         m_Allocator        { m_MemoryResource };

    bool HandleReadNonStandardCommnad();
    void HandleNoWriteSpace( const eEncodedCommandPtr* );

    atomic<eEncodedCommandPtr*> m_pCurReadCommand  = nullptr;
    atomic<eEncodedCommandPtr*> m_pCurWriteCommand = nullptr;
    eEncodedCommandPtr*         m_pFirstCommand    = nullptr;
    span<eEncodedCommandPtr>    m_Commands;
};


inline void CommandBuffer::PushCommandImpl( eEncodedCommandPtr pCmd )noexcept//  Command* pCmd ) noexcept
{
    eEncodedCommandPtr* pCur = nullptr;
    for(;;)
    {
        pCur = m_pCurWriteCommand.load( std::memory_order_relaxed );

        if( *pCur != eEncodedCommandPtr::Invalid )
        {
            HandleNoWriteSpace( pCur );
            continue;
        }

        if( m_pCurWriteCommand.compare_exchange_weak( pCur , pCur+1 , std::memory_order_acq_rel ) )
            break;
    }

    *pCur = pCmd;
}

inline void CommandBuffer::ReserveComands( span<eEncodedCommandPtr*> Res ) noexcept
{
    eEncodedCommandPtr** pBegin = Res.data();
    eEncodedCommandPtr** pEnd = pBegin + Res.size();

    for(;pBegin<pEnd;)
    {
        auto pCur = m_pCurWriteCommand.load( std::memory_order_relaxed );

        if( *pCur != eEncodedCommandPtr::Invalid )
        {
            HandleNoWriteSpace( pCur );
            continue;
        }

        if( !m_pCurWriteCommand.compare_exchange_weak( pCur , pCur+1 , std::memory_order_acq_rel ) )
            continue;

        pBegin[0] = pCur;
        pBegin++;
    }
}

inline void CommandBuffer::PushCommand( span<const Command*const> Commands ) noexcept
{
    if( Commands.empty() )
        return;

    eEncodedCommandPtr* TmpBuffer[100];
    for( ; Commands.size() ; )
    {
        size_t Count = std::min( Commands.size() , size_t(100) );
        auto pBuffer = span{ TmpBuffer , Count };

        ReserveComands( pBuffer );

        for( size_t i = 0 ; i < Count ; ++i )
            *TmpBuffer[i] = EncodedCommandPtr{ Commands[i] , eCommandPtrKind::Standard };

        Commands = Commands.subspan(Count);
    }
}

inline void CommandBuffer::PushCommandBuffer( CommandBuffer& Cmd )
{
    if( &Cmd == this )
        return; // cannot push self

    eEncodedCommandPtr* pCur = nullptr;
    for(;;)
    {
        pCur = m_pCurWriteCommand.load( std::memory_order_relaxed );

        if( pCur[0] != eEncodedCommandPtr::Invalid )
        {
            HandleNoWriteSpace( pCur );
            continue;
        }

        if( m_pCurWriteCommand.compare_exchange_weak( pCur , pCur+1 , std::memory_order_acq_rel ) )
            break;
    }


    auto pJumpCmd = m_Allocator.allocate<CommandReadJump>( *Cmd.m_pFirstCommand );
    auto pReturnCmd = m_Allocator.allocate<CommandReadJump>( pCur[1] );

    pCur[0] = EncodedCommandPtr{ pJumpCmd , eCommandPtrKind::Special };
    Cmd.m_pCurWriteCommand.load()[0] = EncodedCommandPtr{ pReturnCmd , eCommandPtrKind::Special };
}

inline const Command* CommandBuffer::GetNextCommand() noexcept
{
    for( ;; )
    {
        eEncodedCommandPtr* pRead = m_pCurReadCommand.load( std::memory_order_relaxed );
        EncodedCommandPtr Ins{ *pRead };

        if( Ins.GetKind() != eCommandPtrKind::Standard )
        {
            if( HandleReadNonStandardCommnad() )
                continue;

            return nullptr;
        }

        // try to acquire the next command
        if( !m_pCurReadCommand.compare_exchange_weak( pRead , pRead+1 , std::memory_order_acq_rel ) )
            // failed to acquire, try again
            continue;

        return Ins.GetCommand();
    }
}


template< std::derived_from<Command> T , typename ... ARGS >
inline void CommandBuffer::PushCommand( ARGS&& ... args ) noexcept
{
    auto pCommand = m_Allocator.allocate<T>( std::forward<ARGS>(args)... );
    return PushCommandImpl( EncodedCommandPtr{ pCommand , eCommandPtrKind::Standard } );
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
    for( uint8_t i = 0 ; i < Count ; ++i )
    {
        PushCommand<CommandSyncBarier>(Sync);
    }
}