#ifndef DEEPVISION_THREADPOOL_HPP
#define DEEPVISION_THREADPOOL_HPP

#include "thread"

// ������
typedef void (*Task)(void * arg);

// �̳߳���
class ThreadPool {
public:
    // ���캯��
    ThreadPool(int thread_num);
    ~ThreadPool();
    // �������
    void AddTask(Task task, void *arg);
    // ֹͣ�����߳�
    void StopAll();
private:
    // �߳�ѭ��
    void ThreadLoop();
    // ��ȡһ������
    Task GetOneTask();
    // �̺߳���
    static void *ThreadFunc(void *arg);
private:
    struct ThreadTask {
        // ������
        Task task;
        // ����
        void *arg;
    };
    // �߳���
    int thread_num;
    // ��״̬
    bool stop_all;
    // ������
    pthread_mutex_t mutex;
    // ��������
    pthread_cond_t cond;
    // �߳�����
    std::thread *threads;
    // �������
    ThreadTask *task_queue;
    // ��ͷ
    int queue_head;
    // ��β
    int queue_tail;
    // ���д�С
    int queue_size;
};




#endif //DEEPVISION_THREADPOOL_HPP
