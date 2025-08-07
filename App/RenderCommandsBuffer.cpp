/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "RenderCommandsBuffer.h"

bool CommandBuffer::CheckGrow( Context*& pContext )
{
    ZoneScoped;
    std::scoped_lock lock( m_CmdBufSpinLock );

    auto pCurContext = m_pContext.load();
    if( pContext == pCurContext )
        return false;

    pContext = pCurContext;
    return true;
}

void CommandBuffer::GrowCommandBuffer( Context*& pContext , size_t ExtraSize ) noexcept
{
    ZoneScoped;
    std::scoped_lock lock( m_CmdBufSpinLock );

    const auto WriteOffset = (pContext->m_pCurWriteCommand.load()-m_Commands.data());
    const auto ReadOffset  = (pContext->m_pCurReadCommand.load() -m_Commands.data());

    auto pNewContext = m_Allocator.allocate<Context>();

    ExtraSize += m_Commands.size();
    if( ExtraSize < m_Commands.size()*2 )
        ExtraSize = m_Commands.size()*2;

    m_Commands.resize(ExtraSize);

    pNewContext->m_pFirstCommand     = m_Commands.data();
    pNewContext->m_pEndCommand       = pNewContext->m_pFirstCommand + m_Commands.size() - 1;
    pNewContext->m_pCurWriteCommand  = pNewContext->m_pFirstCommand + WriteOffset;
    pNewContext->m_pCurReadCommand   = pNewContext->m_pFirstCommand + ReadOffset;

    pContext = pNewContext;
    m_pContext = pNewContext;
}

CommandBuffer::CommandBuffer( transient_memory_resource& Res )
    : m_MemoryResource  (Res)
    , m_Commands        (m_CmdAllocator)
{
    ZoneScoped;
    m_Commands.resize(1024*8);
    memset( m_Commands.data(), 0, sizeof(m_Commands[0]) * m_Commands.size() );
    m_BaseContext.m_pFirstCommand    = m_Commands.data();
    m_BaseContext.m_pCurReadCommand  = m_BaseContext.m_pFirstCommand;
    m_BaseContext.m_pCurWriteCommand = m_BaseContext.m_pFirstCommand;
    m_BaseContext.m_pEndCommand      = m_BaseContext.m_pFirstCommand + m_Commands.size() - 1;
}

CommandBuffer* CommandBuffer::CreateCommandBuffer( transient_memory_resource& Res ) noexcept
{
    transient_allocator Allocator{ Res };
    return Allocator.allocate<CommandBuffer>( Res );
}