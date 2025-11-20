/*
* Master’s thesis - Analysis of selected optimization techniques for a 3D software renderer
* Author: Wiktoria Marczyk
* Year: 2025
*/

#pragma once
#include "RenderCommandsBuffer.h"
#include "RenderCommands.h"

CommandBuffer::CommandBuffer(transient_memory_resource& Res)
    : m_MemoryResource(Res)
{
    ZoneScoped;
    m_Commands = m_Allocator.allocate_array<eEncodedCommandPtr>(1024 * 8, eEncodedCommandPtr::Invalid);

    m_pFirstCommand = m_Commands.data();
    m_Commands.back() = EncodedCommandPtr{ nullptr , eCommandPtrKind::End };
    m_pCurReadCommand = m_pFirstCommand;
    m_pCurWriteCommand = m_pFirstCommand;
}

CommandBuffer::CommandBuffer(transient_memory_resource& Res, span<const Command* const> Commands, bool ExactSize)
    : m_MemoryResource(Res)
{
    auto Size = ExactSize ? std::max<size_t>(16, Commands.size() + 1) : std::min<size_t>(1024 * 8, Commands.size() + 1);

    ZoneScoped;
    m_Commands = m_Allocator.allocate_array<eEncodedCommandPtr>(Size, eEncodedCommandPtr::Invalid);
    for (size_t i = 0; i < Commands.size(); ++i)
        m_Commands[i] = EncodedCommandPtr{ Commands[i] , eCommandPtrKind::Standard };

    m_pFirstCommand = m_Commands.data();
    m_Commands.back() = EncodedCommandPtr{ nullptr , eCommandPtrKind::End };
    m_pCurReadCommand = m_pFirstCommand;
    m_pCurWriteCommand = m_pFirstCommand + Commands.size();
}

CommandBuffer* CommandBuffer::CreateCommandBuffer(transient_memory_resource& Res) noexcept
{
    transient_allocator Allocator{ Res };
    return Allocator.allocate<CommandBuffer>(Res);
}

CommandBuffer* CommandBuffer::CreateCommandBuffer(transient_memory_resource& Res, span<const Command* const> Commands, bool ExactSize) noexcept
{
    transient_allocator Allocator{ Res };
    return Allocator.allocate<CommandBuffer>(Res, Commands, ExactSize);
}

bool CommandBuffer::HandleReadNonStandardCommnad()
{
    ZoneScoped;
    ZoneColor(0xFF6A00);
    std::scoped_lock lock(m_CmdBufSpinLock);

    EncodedCommandPtr Commmand{ *m_pCurReadCommand.load() };

    switch (Commmand.GetKind())
    {
    case eCommandPtrKind::Null:
    {
        ZoneScopedN("NoCommand");
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // no command, wait
        return true; // no command
    }

    case eCommandPtrKind::Standard:
        return true; // standard command, no special handling

    case eCommandPtrKind::Special:
    {
        ZoneScopedN("Special");
        auto pCommand = Commmand.GetCommand();
        assert(pCommand);
        if (!pCommand)
            return false;

        switch (pCommand->m_CommandID)
        {

        case eCommandID::CmdReadJump:
        {
            ZoneScopedN("CmdReadJump");
            auto pCmd = pCommand->static_cast_to<CommandReadJump>();
            assert(pCmd);

            m_pCurReadCommand = pCmd->pCmd;
            return true;
        }

        default:
            return false;
        }

        return false;
    }

    case eCommandPtrKind::End:
        return false; // end of commands
    default:
        return false;
    }

}

void CommandBuffer::HandleNoWriteSpace(const eEncodedCommandPtr* pLastCmd)
{
    ZoneScoped;
    std::scoped_lock lock(m_CmdBufSpinLock);

    auto pCur = m_pCurWriteCommand.load();
    if (pCur != pLastCmd)
        return; // value has changed since we checked, we need to check again

    EncodedCommandPtr Commmand{ *pCur };

    switch (Commmand.GetKind())
    {
    case eCommandPtrKind::Null:
        return; // there is space - return

    case eCommandPtrKind::Standard:
    {
        auto pwtf = Commmand.GetCommand();
        throw 1;
    }

    case eCommandPtrKind::Special:
    {
        auto pCommand = Commmand.GetCommand();
        assert(pCommand);
        if (!pCommand)
            throw 1;

        switch (pCommand->m_CommandID)
        {
        case eCommandID::CmdReadJump:   return; // switch command, no special handling
        default:                        throw 1; // unknown command
        }
    }

    case eCommandPtrKind::End:
    {
        auto pNewCmdBuf = CreateCommandBuffer(m_MemoryResource);
        auto pSwitchCmd = m_Allocator.allocate<CommandReadJump>(*pNewCmdBuf->m_pFirstCommand);

        m_pCurWriteCommand = pNewCmdBuf->m_pFirstCommand;

        pCur[0] = EncodedCommandPtr{ pSwitchCmd , eCommandPtrKind::Special };
        return;
    }
    }
}