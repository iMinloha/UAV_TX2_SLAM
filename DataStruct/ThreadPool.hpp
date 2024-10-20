#ifndef DEEPVISION_THREADPOOL_HPP
#define DEEPVISION_THREADPOOL_HPP

#include "thread"

// 任务函数
typedef void (*Task)(void * arg);

// 线程池类
class ThreadPool {
public:
    // 构造函数
    ThreadPool(int thread_num);
    ~ThreadPool();
    // 添加任务
    void AddTask(Task task, void *arg);
    // 停止所有线程
    void StopAll();
private:
    // 线程循环
    void ThreadLoop();
    // 获取一个任务
    Task GetOneTask();
    // 线程函数
    static void *ThreadFunc(void *arg);
private:
    struct ThreadTask {
        // 任务函数
        Task task;
        // 参数
        void *arg;
    };
    // 线程数
    int thread_num;
    // 池状态
    bool stop_all;
    // 互斥锁
    pthread_mutex_t mutex;
    // 条件变量
    pthread_cond_t cond;
    // 线程数组
    std::thread *threads;
    // 任务队列
    ThreadTask *task_queue;
    // 队头
    int queue_head;
    // 队尾
    int queue_tail;
    // 队列大小
    int queue_size;
};




#endif //DEEPVISION_THREADPOOL_HPP
