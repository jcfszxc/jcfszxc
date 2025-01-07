#ifndef WORKFLOW_H
#define WORKFLOW_H

#include "config_reader.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include "config_reader.h"
#include <sched.h>
#include <pthread.h>
#include <thread>  // Include this for sleep functionality
#include <chrono>
#include "rknn_api.h"
#include "object_detection.h"

#include <librtmp/rtmp.h>  // RTMP library

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/imgutils.h>
    #include <rk_mpi.h>
    #include <libavutil/hwcontext_drm.h>
    #include <libdrm/drm_fourcc.h>
}



class Workflow {
public:
    Workflow();

    // 启动工作流
    void run();

    // 将解码的AVFrame存入队列
    void store_frame(AVFrame* frame);

private:
    Config config_;

    // 内部任务函数
    void loadConfiguration();  // 重命名为 loadConfiguration
    void task1();
    void task2();
    void task3();
    void task4();

    void print_queue_lengths();

    std::queue<AVFrame*> visual_queue_;
    std::mutex visual_queue_mutex_;
    std::condition_variable visual_queue_cv_;

    std::queue<AVFrame*> detection_queue_;
    std::mutex detection_queue_mutex_;
    std::condition_variable detection_queue_cv_;

    std::mutex rga_mutex;

    AVFrame* pull_stream(const char* stream_url);  // 确保 pull_stream 是成员函数
    std::vector<detect_result_group_t> detection_results_;  // 存储 task2 中的检测结果
    std::mutex detection_results_mutex_;
    std::condition_variable detection_results_cv_;

    std::queue<cv::Mat> shared_image_queue;
    std::mutex image_queue_mutex;
    std::condition_variable image_queue_cv;

};

#endif // WORKFLOW_H
