#include "SimpleThreadPool.h"

SimpleThreadPool::SimpleThreadPool()
    : m_NewTaskSemaphore(0)
{
    m_Finlizing = false;
}

SimpleThreadPool::~SimpleThreadPool()
{
    {
        std::unique_lock lock(m_TasksCS);
        if (m_ThreadCount <= 0)
            return;

        m_Finlizing = true;
        m_NewTaskSemaphore.release(m_ThreadCount);
    }

    while (m_ThreadCount)
    {
    }
}

void SimpleThreadPool::SetThreadCount(uint8_t Count)
{
    {
        std::unique_lock lock(m_TasksCS);
        int CurCount = m_ThreadCount;
        if (Count == CurCount)
            return;

        if (Count < CurCount)
        {
            int TasksToKill = CurCount - Count;
            for (int i = 0; i < TasksToKill; ++i)
                m_Tasks.push_back({});

            m_NewTaskSemaphore.release(TasksToKill);
        }
        else
        {
            int TasksToSpawn = Count - CurCount;
            for (int i = 0; i < TasksToSpawn; ++i)
                thread([this] { Worker(); }).detach();

        }
    }

    while (m_ThreadCount != Count)
    {
    }
}

void SimpleThreadPool::Worker()
{
    m_ThreadCount++;
    while (true)
    {
        m_NewTaskSemaphore.acquire();
        if (m_Finlizing)
            break;

        optional<Task> Task = AcquireTask();
        if (!Task || !Task->m_Func)
            break;

        Task->m_Func();
        Task->m_FinishPromise.set_value();
    }
    m_ThreadCount--;
}

void SimpleThreadPool::LaunchTasks(vector<TaskFunc> TaskFuncs)
{
    vector<future<void>> TasksAwaiters;
    {
        std::unique_lock lock(m_TasksCS);
        if (m_ThreadCount <= 0)
            return;

        for (auto& Func : TaskFuncs)
        {
            if (!Func)
                continue;

            m_Tasks.push_back(Task{ {}, std::move(Func) });
            TasksAwaiters.push_back(m_Tasks.back().m_FinishPromise.get_future());
        }

        m_NewTaskSemaphore.release(TasksAwaiters.size());
    }

    for (auto& Awaiter : TasksAwaiters)
    {
        Awaiter.get();
    }
}


optional<SimpleThreadPool::Task> SimpleThreadPool::AcquireTask()
{
    std::unique_lock lock(m_TasksCS);

    optional<SimpleThreadPool::Task> Result;
    if (m_Tasks.empty())
        return Result;

    Result = std::move(m_Tasks.front());
    m_Tasks.erase(m_Tasks.begin());
    return Result;
}

 uint8_t SimpleThreadPool::GetThreadCount()const
 {
     return m_ThreadCount;
 }