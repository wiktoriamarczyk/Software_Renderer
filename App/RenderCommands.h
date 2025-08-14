/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
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
    AppendCmdBuf,

    SwitchCmdBuf,
};

struct DrawConfig
{
    Vector4f m_Color;
};

struct ISyncBarier
{
    virtual void Wait()const noexcept = 0;
};

struct SyncBarrier : ISyncBarier
{
    mutable optional<std::barrier<triviall_function_ref>> m_Barrier;
    const char* name = "";

    virtual void Wait()const noexcept override
    {
        ZoneScoped;
        ZoneName(name, strlen(name));
        m_Barrier->arrive_and_wait();
        m_Barrier.reset();
    }
};

struct PipelineSharedData;

enum class eCommandPtrKind : uint8_t
{
    Null       = 0,
    Standard   = 1,
    Special    = 2,
    End        = 3,
};

enum class eEncodedCommandPtr : size_t
{
    Invalid = 0,
};

struct alignas(4) Command
{
    eCommandID m_CommandID = eCommandID::NoCommand;

    eCommandID GetCommandID()const noexcept
    {
        return m_CommandID;
    }

    template< std::derived_from<Command> T >
    const T* static_cast_to()const
    {
        if( m_CommandID != T::COMMAND_ID )
        {
            assert( false && "Wrong command type" );
            return nullptr; // wrong type
        }
        return static_cast<const T*>(this);
    }
protected:
    Command( eCommandID ID )
        : m_CommandID( ID )
    {}
};

struct EncodedCommandPtr
{
    static constexpr inline auto Bits = sizeof(size_t)*8;
    static constexpr inline auto DataBits = 2;
    static constexpr inline auto DataMask = (1 << DataBits) - 1;
    static constexpr inline auto RequiredCmdAlign = 1 << DataBits;

    EncodedCommandPtr( const Command* pCmd , eCommandPtrKind Kind )
    {
        size_t Ptr = reinterpret_cast<size_t>(pCmd);
        assert( Ptr % RequiredCmdAlign == 0 && "Command pointer must be aligned to 8 bytes" );

        m_Separated.m_Ptr  = Ptr >> DataBits;
        m_Separated.m_Data = size_t(Kind) & DataMask;
    };
    EncodedCommandPtr( eEncodedCommandPtr Encoded )
        : m_Enoded(Encoded)
    {}
    EncodedCommandPtr( const EncodedCommandPtr& Encoded )
        : m_Enoded(Encoded.m_Enoded)
    {}

    const Command* GetCommand()const noexcept
    {
        return reinterpret_cast<const Command*>(m_Separated.m_Ptr << DataBits);
    }

    eCommandPtrKind GetKind()const noexcept
    {
        return static_cast<eCommandPtrKind>( m_Separated.m_Data );
    }
    eEncodedCommandPtr GetEncoded()const noexcept
    {
        return m_Enoded;
    }
    operator eEncodedCommandPtr()const noexcept
    {
        return GetEncoded();
    }

    eEncodedCommandPtr Next()const noexcept
    {
        EncodedCommandPtr next = *this;
        next.m_Separated.m_Data += sizeof(const Command*);
        return next.m_Enoded;
    }

    eEncodedCommandPtr GetEncodedCommand()const noexcept
    {
        return m_Enoded;
    }

    struct SeparatedCommand
    {
    size_t m_Ptr    : Bits - DataBits   = 0;
    size_t m_Data   : DataBits          = 0;
    };

    union
    {
        eEncodedCommandPtr  m_Enoded = eEncodedCommandPtr::Invalid;
        SeparatedCommand    m_Separated;
    };
};

struct CommandFill32BitBuffer : Command
{
    static constexpr auto COMMAND_ID = eCommandID::Fill32BitBuffer;

    CommandFill32BitBuffer( span<float> Buffer , float Val )
        : Command(COMMAND_ID)
        , m_pBuffer{Buffer.data()}
        , m_ElementsCount{static_cast<uint32_t>(Buffer.size())}
        , m_Value{ .m_FValue{ Val} }
    {}
    CommandFill32BitBuffer( span<uint32_t> Buffer , uint32_t Val )
        : Command(COMMAND_ID)
        , m_pBuffer{Buffer.data()}
        , m_ElementsCount{static_cast<uint32_t>(Buffer.size())}
        , m_Value{ .m_UValue{ Val } }
    {}

    void*    m_pBuffer = nullptr;
    uint32_t m_ElementsCount = 0;
    union
    {
        uint32_t m_UValue = 0.0f;
        uint8_t  m_U8Array[4];
        float    m_FValue;
    } m_Value;
};

struct CommandClear : Command
{
    static constexpr auto COMMAND_ID = eCommandID::ClearBuffers;

    CommandClear( optional<uint32_t> ClearColor = 0 , optional<float> ZValue = 1.0f )
        : Command(COMMAND_ID)
        , m_ClearColor(ClearColor)
        , m_ZValue(ZValue)
    {}
    optional<uint32_t>  m_ClearColor;
    optional<float>     m_ZValue    ;
};

struct CommandVertexAssemply : Command
{
    static constexpr auto COMMAND_ID = eCommandID::VertexAssemply;

    CommandVertexAssemply( span<const Vertex> Input , DrawConfig& Config )
        : Command(COMMAND_ID)
        , m_Vertices(Input)
        , m_pConfig(&Config)
    {}
    span<const Vertex> m_Vertices;
    DrawConfig* m_pConfig = nullptr;
};

struct CommandVertexTransformAndClip : Command
{
    static constexpr auto COMMAND_ID = eCommandID::VertexTransformAndClip;

    CommandVertexTransformAndClip( span<const Vertex> Input , const  PipelineSharedData& Data )
        : Command(COMMAND_ID)
        , m_Input(Input)
        , m_pPipelineSharedData(&Data)
    {}
    span<const Vertex> m_Input;
    const  PipelineSharedData* m_pPipelineSharedData = nullptr;
};

struct CommandProcessTriangles : Command
{
    static constexpr auto COMMAND_ID = eCommandID::ProcessTriangles;

    CommandProcessTriangles( span<const TransformedVertex> Input , const PipelineSharedData& Data )
        : Command(COMMAND_ID)
        , m_Vertices(Input)
        , m_pPipelineSharedData(&Data)
    {}
    span<const TransformedVertex> m_Vertices;
    const  PipelineSharedData* m_pPipelineSharedData = nullptr;
};

struct CommandSyncBarier : Command
{
    static constexpr auto COMMAND_ID = eCommandID::SyncBarier;

    CommandSyncBarier( ISyncBarier& Sync )
        : Command(COMMAND_ID)
        , pAwaitSync(&Sync)
    {}

    ISyncBarier* pAwaitSync = nullptr;
};

struct CommandAppendCommmandBufffer : Command
{
    static constexpr auto COMMAND_ID = eCommandID::AppendCmdBuf;

    CommandAppendCommmandBufffer( CommandBuffer& Src , CommandBuffer& Dst , bool AtEnd = true )
        : Command(COMMAND_ID)
        , pSrc(&Src)
        , pDst(&Dst)
        , m_AtEnd(AtEnd)
    {}

    bool m_AtEnd = true;
    CommandBuffer* pSrc = nullptr;
    CommandBuffer* pDst = nullptr;
};

struct CommmandReadJump : Command
{
    static constexpr auto COMMAND_ID = eCommandID::SwitchCmdBuf;

    CommmandReadJump( eEncodedCommandPtr& Cmd )
        : Command(COMMAND_ID)
        , pCmd{ &Cmd }
    {}

    eEncodedCommandPtr* pCmd = nullptr;
};

struct CommandRenderTile : Command
{
    static constexpr auto COMMAND_ID = eCommandID::RenderTile;

    CommandRenderTile()
        : Command(COMMAND_ID)
    {}
    bool                IsFullTile = false;
    uint16_t            TileDrawID = 0;
    Vector2i            ScreenPos;
    const TriangleData* Triangle;
    const TileInfo*     TileInfo;
};
