/*
* Master’s thesis - Analysis of selected optimization techniques for a 3D software renderer
* Author: Wiktoria Marczyk
* Year: 2025
*/

#pragma once
#include "Common.h"
#include "TransformedVertex.h"
#include "function_ref.h"

struct CommandBuffer;
struct TriangleData;
struct TileInfo;

enum class eCommandID : uint8_t
{
    NoCommand,
    RenderTile,
    VertexAssemply,
    VertexTransformAndClip,
    ProcessTriangles,
    ClearBuffers,
    Fill32BitBuffer,
    SyncBarier,
    CmdReadJump,
};

struct DrawControl
{
    bool m_IsFullTile : 1 = false;
    bool m_ZTest : 1 = false;
    bool m_ZWrite : 1 = false;
    bool m_AlphaBlend : 1 = false;
};

struct DrawConfig
{
    Vector4f    m_Color;
    DrawControl m_DrawControl;
};

struct ISyncBarier
{
    virtual void Wait()const noexcept = 0;
};

struct SyncBarrier : ISyncBarier
{
    mutable optional<std::barrier<triviall_function_ref>> m_Barrier;
    const char* m_Name = "";

    virtual void Wait()const noexcept override
    {
        ZoneScoped;
        ZoneName(m_Name, strlen(m_Name));
        m_Barrier->arrive_and_wait();
        m_Barrier.reset();
    }
};

struct PipelineSharedData;

enum class eCommandPtrKind : uint8_t
{
    Null = 0,
    Standard = 1,
    Special = 2,
    End = 3,
};

enum class _eEncodedCommandPtr : size_t
{
    Invalid = 0,
};

struct eEncodedCommandPtr
{
    static const eEncodedCommandPtr Invalid;

    constexpr auto operator<=>(const eEncodedCommandPtr& other)const noexcept = default;

    _eEncodedCommandPtr val = _eEncodedCommandPtr::Invalid;
};

inline constexpr const eEncodedCommandPtr eEncodedCommandPtr::Invalid = eEncodedCommandPtr{ _eEncodedCommandPtr::Invalid };

#define COMMAND_ALIGN alignas(8)

struct Command
{
    eCommandID m_CommandID = eCommandID::NoCommand;

    eCommandID GetCommandID()const noexcept
    {
        return m_CommandID;
    }

    template< std::derived_from<Command> T >
    const T* static_cast_to()const
    {
        if (m_CommandID != T::COMMAND_ID)
        {
            assert(false && "Wrong command type");
            return nullptr; // wrong type
        }
        return static_cast<const T*>(this);
    }
protected:
    Command(eCommandID ID)
        : m_CommandID(ID)
    {
    }
};

struct EncodedCommandPtr
{
    static constexpr inline auto Bits = sizeof(size_t) * 8;
    static constexpr inline auto DataBits = 2;
    static constexpr inline auto DataMask = (1 << DataBits) - 1;
    static constexpr inline auto RequiredCmdAlign = 1 << DataBits;

    EncodedCommandPtr(const Command* pCmd, eCommandPtrKind Kind)
    {
        size_t Ptr = reinterpret_cast<size_t>(pCmd);
        assert(Ptr % RequiredCmdAlign == 0 && "Command pointer must be aligned to 8 bytes");

        m_Separated.m_Ptr = Ptr >> DataBits;
        m_Separated.m_Data = size_t(Kind) & DataMask;
    };
    EncodedCommandPtr(eEncodedCommandPtr Encoded)
        : m_Encoded(Encoded)
    {
    }
    EncodedCommandPtr(const EncodedCommandPtr& Encoded)
        : m_Encoded(Encoded.m_Encoded)
    {
    }

    const Command* GetCommand()const noexcept
    {
        return reinterpret_cast<const Command*>(m_Separated.m_Ptr << DataBits);
    }

    eCommandPtrKind GetKind()const noexcept
    {
        return static_cast<eCommandPtrKind>(m_Separated.m_Data);
    }

    eEncodedCommandPtr GetEncoded()const noexcept
    {
        return m_Encoded;
    }
    operator eEncodedCommandPtr()const noexcept
    {
        return GetEncoded();
    }

    eEncodedCommandPtr Next()const noexcept
    {
        EncodedCommandPtr next = *this;
        next.m_Separated.m_Data += sizeof(const Command*);
        return next.m_Encoded;
    }

    eEncodedCommandPtr GetEncodedCommand()const noexcept
    {
        return m_Encoded;
    }

    struct SeparatedCommand
    {
        size_t m_Ptr : Bits - DataBits = 0;
        size_t m_Data : DataBits = 0;
    };

    union
    {
        eEncodedCommandPtr  m_Encoded = eEncodedCommandPtr{};
        SeparatedCommand    m_Separated;
    };
};

struct COMMAND_ALIGN CommandFill32BitBuffer : Command
{
    static constexpr auto COMMAND_ID = eCommandID::Fill32BitBuffer;

    CommandFill32BitBuffer(span<float> Buffer, float Val)
        : Command(COMMAND_ID)
        , m_pBuffer{ Buffer.data() }
        , m_ElementsCount{ static_cast<uint32_t>(Buffer.size()) }
        , m_Value{ .m_FValue{ Val} }
    {
    }
    CommandFill32BitBuffer(span<uint32_t> Buffer, uint32_t Val)
        : Command(COMMAND_ID)
        , m_pBuffer{ Buffer.data() }
        , m_ElementsCount{ static_cast<uint32_t>(Buffer.size()) }
        , m_Value{ .m_UValue{ Val } }
    {
    }

    void* m_pBuffer = nullptr;
    uint32_t m_ElementsCount = 0;
    union
    {
        uint32_t m_UValue = 0.0f;
        uint8_t  m_U8Array[4];
        float    m_FValue;
    } m_Value;
};

struct COMMAND_ALIGN CommandClear : Command
{
    static constexpr auto COMMAND_ID = eCommandID::ClearBuffers;

    CommandClear(optional<uint32_t> ClearColor = 0, optional<float> ZValue = 1.0f)
        : Command(COMMAND_ID)
        , m_ClearColor(ClearColor)
        , m_ZValue(ZValue)
    {
    }
    optional<uint32_t>  m_ClearColor;
    optional<float>     m_ZValue;
};

struct COMMAND_ALIGN  CommandVertexAssemply : Command
{
    static constexpr auto COMMAND_ID = eCommandID::VertexAssemply;

    CommandVertexAssemply(span<const Vertex> Input, DrawConfig& Config)
        : Command(COMMAND_ID)
        , m_Vertices(Input)
        , m_pConfig(&Config)
    {
    }
    span<const Vertex> m_Vertices;
    DrawConfig* m_pConfig = nullptr;
};

struct COMMAND_ALIGN  CommandVertexTransformAndClip : Command
{
    static constexpr auto COMMAND_ID = eCommandID::VertexTransformAndClip;

    CommandVertexTransformAndClip(span<const Vertex> Input, const PipelineSharedData& Data, uint32_t StartTriIndex)
        : Command(COMMAND_ID)
        , m_StartTriIndex(StartTriIndex)
        , m_Input(Input)
        , m_pPipelineSharedData(&Data)
    {
    }
    uint32_t m_StartTriIndex = 0;
    span<const Vertex> m_Input;
    const  PipelineSharedData* m_pPipelineSharedData = nullptr;
};

struct COMMAND_ALIGN  CommandProcessTriangles : Command
{
    static constexpr auto COMMAND_ID = eCommandID::ProcessTriangles;

    CommandProcessTriangles(span<const TransformedVertex> Input, const PipelineSharedData& Data, uint32_t StartTriIndex)
        : Command(COMMAND_ID)
        , m_StartTriIndex(StartTriIndex)
        , m_Vertices(Input)
        , m_pPipelineSharedData(&Data)
    {
    }
    uint32_t m_StartTriIndex = 0;
    span<const TransformedVertex> m_Vertices;
    const  PipelineSharedData* m_pPipelineSharedData = nullptr;
};

struct COMMAND_ALIGN  CommandSyncBarier : Command
{
    static constexpr auto COMMAND_ID = eCommandID::SyncBarier;

    CommandSyncBarier(ISyncBarier& Sync)
        : Command(COMMAND_ID)
        , pAwaitSync(&Sync)
    {
    }

    ISyncBarier* pAwaitSync = nullptr;
};

struct COMMAND_ALIGN  CommandReadJump : Command
{
    static constexpr auto COMMAND_ID = eCommandID::CmdReadJump;

    CommandReadJump(eEncodedCommandPtr& Cmd)
        : Command(COMMAND_ID)
        , pCmd{ &Cmd }
    {
    }

    eEncodedCommandPtr* pCmd = nullptr;
};

struct COMMAND_ALIGN CommandRenderTile : Command
{
    static constexpr auto COMMAND_ID = eCommandID::RenderTile;

    CommandRenderTile()
        : Command(COMMAND_ID)
    {
    }
    DrawControl         DrawControl;
    uint32_t            TileDrawID = 0;
    const TriangleData* Triangle;
    const TileInfo* TileInfo;
    atomic<CommandRenderTile*> pNext = nullptr;
};
