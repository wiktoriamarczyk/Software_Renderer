/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

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

void SimpleThreadPool::SetThreadCount(uint8_t count)
{
    {
        std::unique_lock lock(m_TasksCS);
        int curCount = m_ThreadCount;
        if (count == curCount)
            return;

        if (count < curCount)
        {
            int tasksToKill = curCount - count;
            for (int i = 0; i < tasksToKill; ++i)
                m_Tasks.push_back({});

            m_NewTaskSemaphore.release(tasksToKill);
        }
        else
        {
            int tasksToSpawn = count - curCount;
            for (int i = 0; i < tasksToSpawn; ++i)
                thread([this] { Worker(); }).detach();

        }
    }

    while (m_ThreadCount != count)
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
    vector<future<void>> tasksAwaiters;
    {
        std::unique_lock lock(m_TasksCS);
        if (m_ThreadCount <= 0)
            return;

        for (auto& Func : TaskFuncs)
        {
            if (!Func)
                continue;

            m_Tasks.push_back(Task{ {}, std::move(Func) });
            tasksAwaiters.push_back(m_Tasks.back().m_FinishPromise.get_future());
        }

        m_NewTaskSemaphore.release(tasksAwaiters.size());
    }

    for (auto& Awaiter : tasksAwaiters)
    {
        Awaiter.get();
    }
}


optional<SimpleThreadPool::Task> SimpleThreadPool::AcquireTask()
{
    std::unique_lock lock(m_TasksCS);

    optional<SimpleThreadPool::Task> result;
    if (m_Tasks.empty())
        return result;

    result = std::move(m_Tasks.front());
    m_Tasks.erase(m_Tasks.begin());
    return result;
}

 uint8_t SimpleThreadPool::GetThreadCount()const
 {
     return m_ThreadCount;
 }