#pragma once

#include "Common.h"

#include <functional>
#include <future>
#include <semaphore>

class SimpleThreadPool
{
public:
    using TaskFunc = function<void()>;

    SimpleThreadPool();
    ~SimpleThreadPool();
    void SetThreadCount(uint8_t Count);
    uint8_t GetThreadCount()const;
    void LaunchTasks(vector<TaskFunc> tasks);
public:
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
};