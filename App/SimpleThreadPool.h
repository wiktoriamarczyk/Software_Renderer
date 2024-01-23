/*
* Engineering thesis - Software-based 3D Graphics Renderer
* Author: Wiktoria Marczyk
* Year: 2024
*/

#pragma once
#include "Common.h"

/**
* Klasa reprezentuj¹ca pulê w¹tków.
*/
class SimpleThreadPool
{
public:
    /**
    * Funkcja wykonywana przez w¹tek.
    */
    using TaskFunc = function<void()>;

    SimpleThreadPool();
    ~SimpleThreadPool();
    /**
    * Ustawia liczbê w¹tków.
    * @param count liczba w¹tków
    */
    void SetThreadCount(uint8_t count);
    /**
    * Zwraca liczbê w¹tków.
    * @return liczba w¹tków
    */
    uint8_t GetThreadCount()const;
    /**
    * Uruchamia zadania dla wszystkich dostêpnych w¹tków.
    * @param tasks zadania
    */
    void LaunchTasks(vector<TaskFunc> tasks);
private:
    /**
    * Struktura reprezentuj¹ca zadanie.
    */
    struct Task
    {
        promise<void>   m_FinishPromise; ///< obiekt do sygnalizacji zakoñczenia zadania
        TaskFunc        m_Func; ///< funkcja zadania
    };

    /**
    * Funkcja przetwarzaj¹ca zadania w ramach w¹tku.
    */
    void Worker();
    /**
    * Pobiera zadanie.
    * @return zadanie
    */
    optional<Task> AcquireTask();

    atomic_bool m_Finlizing; ///< flaga informuj¹ca o zakoñczeniu pracy
    counting_semaphore<> m_NewTaskSemaphore; ///< semafor do sygnalizacji nowego zadania
    vector<Task> m_Tasks; ///< zadania
    std::mutex m_TasksCS; ///< sekcja krytyczna dla zadañ
    atomic_int m_ThreadCount = 0; ///< liczba w¹tków
};