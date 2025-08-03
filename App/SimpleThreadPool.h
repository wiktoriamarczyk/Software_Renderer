/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Common.h"

class SimpleThreadPool
{
public:
    using TaskFunc = function<void()>;

    SimpleThreadPool();
    ~SimpleThreadPool();
    void SetThreadCount(uint8_t count);
    uint8_t GetThreadCount()const;
    void LaunchTasks(vector<TaskFunc> tasks);
    static int32_t GetThreadID();
private:
    struct Task
    {
        promise<void>   m_FinishPromise;
        TaskFunc        m_Func;
    };

    void Worker();
    optional<Task> AcquireTask();

    atomic_bool m_Finlizing;
    counting_semaphore<> m_NewTaskSemaphore;
    vector<Task> m_Tasks;
    std::mutex m_TasksCS;
    atomic_int m_ThreadCount = 0;
    std::mutex m_IDsCS;
    vector<unique_ptr<int32_t>> m_FreeThreadIds;
};

class Spinlock
{
public:
    constexpr Spinlock()noexcept = default;
    constexpr ~Spinlock()noexcept = default;

    void lock()const noexcept
    {
        lock( std::memory_order_acquire );
    }
    void lockSeqCst()const noexcept
    {
        lock( std::memory_order_seq_cst );
    }

    void unlock()const noexcept
    {
        m_Lock.store( false , std::memory_order_release );
    }
private:
    void lock( std::memory_order LockOrder )const noexcept
    {
        for (;;)
        {
            if( !m_Lock.exchange( true , LockOrder ) )
                return;

            while( m_Lock.load( std::memory_order_relaxed ) )
                std::this_thread::yield();
        }
    }
    mutable std::atomic<bool> m_Lock = 0;
};