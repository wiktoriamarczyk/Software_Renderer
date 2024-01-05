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
};