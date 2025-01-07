#include <iostream>
#include "workflow.h"

void bind_to_cpu(int cpu_id) {
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(cpu_id, &cpu_set);

    pthread_t current_thread = pthread_self();
    if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpu_set) != 0) {
        std::cerr << "Failed to bind thread to CPU " << cpu_id << std::endl;
    } else {
        std::cout << "Successfully bound thread to CPU " << cpu_id << std::endl;
    }
}


// 构造函数
Workflow::Workflow() {}

// 启动工作流
void Workflow::run() {
    std::cout << "Workflow is starting with configuration loading..." << std::endl;

    // 加载配置
    loadConfiguration();

    // 执行任务
    std::thread task1_thread(&Workflow::task1, this);
    std::thread task2_thread(&Workflow::task2, this);
    std::thread task3_thread(&Workflow::task3, this);
    std::thread task4_thread(&Workflow::task4, this);
    
    task1_thread.join();
    task2_thread.join();
    task3_thread.join();
    task4_thread.join();

    std::cout << "Workflow completed." << std::endl;
}

// 加载配置函数
void Workflow::loadConfiguration() {
    config_ = loadConfig("config.json");  // 这里调用的是 config_reader.h 中的 loadConfig 函数
}

void Workflow::store_frame(AVFrame* frame) {
    // 深拷贝 AVFrame，避免复用问题
    AVFrame* visual_frame = av_frame_alloc();
    av_frame_ref(visual_frame, frame);

    AVFrame* detection_frame = av_frame_alloc();
    av_frame_ref(detection_frame, frame);


    // 锁定并推入可视化队列
    {
        std::lock_guard<std::mutex> lock(visual_queue_mutex_);
        
        // 如果队列已满，丢弃旧帧
        if (!visual_queue_.empty()) {
            AVFrame* old_frame = visual_queue_.front();
            av_frame_unref(old_frame);
            av_frame_free(&old_frame);
            visual_queue_.pop();
        }

        visual_queue_.push(visual_frame);
    }
    visual_queue_cv_.notify_one();

    // 锁定并推入检测队列
    {
        std::lock_guard<std::mutex> lock(detection_queue_mutex_);
        
        // 如果队列已满，丢弃旧帧
        if (!detection_queue_.empty()) {
            AVFrame* old_frame = detection_queue_.front();
            av_frame_unref(old_frame);
            av_frame_free(&old_frame);
            detection_queue_.pop();
        }

        detection_queue_.push(detection_frame);
    }
    detection_queue_cv_.notify_one();
}

void Workflow::print_queue_lengths() {
    // Lock and print the visual_queue_ length
    {
        std::lock_guard<std::mutex> lock(visual_queue_mutex_);
        std::cout << "visual_queue_ length: " << visual_queue_.size() << std::endl;
    }

    // Lock and print the detection_queue_ length
    {
        std::lock_guard<std::mutex> lock(detection_queue_mutex_);
        std::cout << "detection_queue_ length: " << detection_queue_.size() << std::endl;
    }

    // Lock and print the shared_image_queue length
    {
        std::lock_guard<std::mutex> lock(image_queue_mutex);
        std::cout << "shared_image_queue length: " << shared_image_queue.size() << std::endl;
    }

    // Lock and print the detection_results_ length
    {
        std::lock_guard<std::mutex> lock(detection_results_mutex_);
        std::cout << "detection_results_ length: " << detection_results_.size() << std::endl;
    }
}


AVFrame* Workflow::pull_stream(const char* stream_url) {
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    // AVCodec* codec = nullptr;
    AVFrame* frame = av_frame_alloc();
    AVPacket packet;

    // Initialize FFmpeg network module
    avformat_network_init();

    // Open RTSP stream
    if (avformat_open_input(&fmt_ctx, stream_url, nullptr, nullptr) != 0) {
        std::cerr << "Could not open input stream." << std::endl;
        av_frame_free(&frame);  // 防止内存泄漏
        return nullptr;
    }

    // Find stream information
    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        std::cerr << "Could not find stream information." << std::endl;
        av_frame_free(&frame);  // 防止内存泄漏
        return nullptr;
    }

    // Find video stream index
    int video_stream_index = -1;
    for (unsigned i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }
    if (video_stream_index == -1) {
        std::cerr << "Could not find a video stream." << std::endl;
        av_frame_free(&frame);  // 防止内存泄漏
        return nullptr;
    }

    // Find RKMpp hardware decoder
    // codec = avcodec_find_decoder_by_name("h264_rkmpp");
    const AVCodec* codec = avcodec_find_decoder_by_name("h264_rkmpp");
    if (!codec) {
        std::cerr << "Could not find RKMpp hardware-accelerated decoder (h264_rkmpp)." << std::endl;
        av_frame_free(&frame);  // 防止内存泄漏
        return nullptr;
    }

    // Create codec context
    codec_ctx = avcodec_alloc_context3(codec);
    if (avcodec_parameters_to_context(codec_ctx, fmt_ctx->streams[video_stream_index]->codecpar) < 0) {
        std::cerr << "Could not copy codec context." << std::endl;
        avcodec_free_context(&codec_ctx);
        av_frame_free(&frame);
        return nullptr;
    }

    // Set hardware accelerated format for RKMpp
    // codec_ctx->get_format = [](AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) {
    codec_ctx->get_format = [](AVCodecContext* /* ctx */, const enum AVPixelFormat* pix_fmts) {
        while (*pix_fmts != AV_PIX_FMT_NONE) {
            if (*pix_fmts == AV_PIX_FMT_DRM_PRIME || *pix_fmts == AV_PIX_FMT_YUV420P) {
                return *pix_fmts;  // Use DRM format for hardware acceleration
            }
            pix_fmts++;
        }
        return AV_PIX_FMT_YUV420P;  // Fallback to software decoding if no hardware-accelerated formats are available
    };

    // Set lenient decoding options
    codec_ctx->flags2 |= AV_CODEC_FLAG2_CHUNKS | AV_CODEC_FLAG2_FAST;
    codec_ctx->skip_frame = AVDISCARD_BIDIR;
    codec_ctx->err_recognition = AV_EF_CAREFUL | AV_EF_COMPLIANT | AV_EF_AGGRESSIVE;

    // Open codec
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        std::cerr << "Could not open codec." << std::endl;
        avcodec_free_context(&codec_ctx);
        av_frame_free(&frame);
        return nullptr;
    }

    // Seek to the nearest keyframe
    avformat_seek_file(fmt_ctx, video_stream_index, 0, 0, 0, AVSEEK_FLAG_BACKWARD);

    // Get the frame rate from the stream (do this once, outside the loop)
    // AVRational frame_rate = fmt_ctx->streams[video_stream_index]->r_frame_rate;
    // double fps = av_q2d(frame_rate);
    // std::cout << "Frame rate: " << fps << " frames per second" << std::endl;

    // Continuously read and decode frames
    while (true) {
        if (av_read_frame(fmt_ctx, &packet) >= 0) {
            if (packet.stream_index == video_stream_index) {
                // Check if PTS and DTS are valid
                if (packet.pts == AV_NOPTS_VALUE || packet.dts == AV_NOPTS_VALUE) {
                    av_packet_unref(&packet);
                    continue;
                }

                // Send packet to decoder
                int ret = avcodec_send_packet(codec_ctx, &packet);
                if (ret == 0) {
                    while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                        store_frame(frame);

                        // // Print PTS or DTS based frame timing and frame rate
                        // static int64_t last_pts = AV_NOPTS_VALUE;
                        // if (last_pts != AV_NOPTS_VALUE) {
                        //     double frame_duration = (frame->pts - last_pts) * av_q2d(fmt_ctx->streams[video_stream_index]->time_base);
                        //     double current_fps = 1.0 / frame_duration;
                        //     std::cout << "Current FPS: " << current_fps << std::endl;
                        //     // print_queue_lengths();

                        // }
                        // last_pts = frame->pts;

                    }
                } else if (ret == AVERROR_INVALIDDATA) {
                    std::cerr << "Invalid data, skipping frame." << std::endl;
                } else {
                    std::cerr << "Failed to send packet to decoder, error code: " << ret << std::endl;
                }
            }
            av_packet_unref(&packet);
        } else {
            std::cerr << "Failed to read packet from stream." << std::endl;
            break;  // Break loop on failure to read from stream
        }
    }

    // Free resources
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);

    return frame;
}


// task1 仅负责调用 pull_stream 并管理整个任务流程
void Workflow::task1() {
    std::cout << "Executing task 1: MPP hardware encoding and stream pulling..." << std::endl;

    bind_to_cpu(0);

    const char* stream_url = config_.videoDevice.c_str();

    // 只需要调用 pull_stream，它会负责存储帧
    AVFrame* frame = pull_stream(stream_url);
    if (!frame) {
        std::cerr << "Failed to pull stream from RTSP source." << std::endl;
        return;
    }

    std::cout << "Task 1 completed: Stream pulled and encoded successfully." << std::endl;
}


// 添加计时函数
#define TIME_POINT_START(name) auto start_##name = std::chrono::high_resolution_clock::now();
#define TIME_POINT_END(name, label) { \
    auto end_##name = std::chrono::high_resolution_clock::now(); \
    std::chrono::duration<double> diff_##name = end_##name - start_##name; \
    std::cerr << label << " took " << diff_##name.count() << " seconds" << std::endl; \
}
// Function to calculate the distance from a point to the center of the image
float distanceToCenter(int x, int y, int centerX, int centerY) {
    return std::sqrt(std::pow(x - centerX, 2) + std::pow(y - centerY, 2));
}

// Function to get the center point of a bounding box
void getBoxCenter(const BOX_RECT& box, int& centerX, int& centerY) {
    centerX = (box.left + box.right) / 2;
    centerY = (box.top + box.bottom) / 2;
}

void Workflow::task2() {
    std::cout << "Executing task 2: Processing stored frames..." << std::endl;

    bind_to_cpu(1);

    // Initialize RKNN context and detection parameters
    std::cout << "Initializing RKNN context and detection parameters..." << std::endl;
    
    rknn_input_output_num io_num;
    rknn_context ctx = initializeRKNN(config_.modelPath, io_num);
    std::cout << "RKNN context initialized. Input num: " << io_num.n_input << ", Output num: " << io_num.n_output << std::endl;

    // Query and set up tensor attributes
    std::vector<rknn_tensor_attr> input_attrs(io_num.n_input);
    std::vector<rknn_tensor_attr> output_attrs(io_num.n_output);
    queryAndDumpTensorAttrs(ctx, io_num, input_attrs, output_attrs);

    std::cout << "Tensor attributes queried successfully." << std::endl;

    // Set up model input parameters
    int channel = 3;
    int model_height = input_attrs[0].dims[2];
    int model_width = input_attrs[0].dims[1];
    printf("model input height=%d, width=%d, channel=%d\n", model_height, model_width, channel);

    // Create detection parameters
    DetectionParams params = createDetectionParams(ctx, io_num, input_attrs, output_attrs,
                                                   model_width, model_height, config_.boxThresh, config_.nmsThresh);

    // Print detection thresholds
    std::cout << "Detection thresholds - Box Threshold: " << config_.boxThresh << ", NMS Threshold: " << config_.nmsThresh << std::endl;
    
    std::cout << "Detection parameters created." << std::endl;

    // Initialize RGA images
    rga_info_t srcRga, dstRga;

    // 初始化源图像和目标图像的 RGA 信息
    memset(&srcRga, 0, sizeof(rga_info_t));
    memset(&dstRga, 0, sizeof(rga_info_t));

    // int frame_count = 0;  // 计数器，用于统计帧数
    // auto start_time = std::chrono::steady_clock::now();  // 记录起始时间

    while (true) {
        std::unique_lock<std::mutex> lock(detection_queue_mutex_);

        // 等待直到有新帧加入队列
        detection_queue_cv_.wait(lock, [this]() { return !detection_queue_.empty(); });

        // 从队列中取出一帧
        AVFrame* frame = detection_queue_.front();
        detection_queue_.pop();
        lock.unlock();

        // // 计算帧率
        // frame_count++;
        // auto current_time = std::chrono::steady_clock::now();
        // std::chrono::duration<double> elapsed_seconds = current_time - start_time;

        // // 每隔一秒钟打印一次帧率
        // if (elapsed_seconds.count() >= 1.0) {
        //     double fps = frame_count / elapsed_seconds.count();  // 计算帧率
        //     std::cout << "Current FPS: " << fps << " frames per second" << std::endl;

        //     // 重置计数器和起始时间
        //     frame_count = 0;
        //     start_time = current_time;
        // }



        if (!frame || frame->format != AV_PIX_FMT_DRM_PRIME) {
            std::cerr << "Frame is not in DRM_PRIME format" << std::endl;
        }

        AVDRMFrameDescriptor* drm_frame_desc = (AVDRMFrameDescriptor*)frame->data[0];
        if (!drm_frame_desc) {
            std::cerr << "Failed to retrieve DRM frame descriptor" << std::endl;
        }

        // Get the prime fd (file descriptor) for the buffer
        int prime_fd = drm_frame_desc->objects[0].fd;
        if (prime_fd < 0) {
            std::cerr << "Invalid DRM PRIME FD" << std::endl;
        }

        // Get the size of the buffer (you can obtain this from the plane or object)
        uint32_t size = drm_frame_desc->objects[0].size;

        // Map the DRM buffer into memory
        void* mapped_memory = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, prime_fd, 0);
        if (mapped_memory == MAP_FAILED) {
            munmap(mapped_memory, size);  // 释放映射的DRM内存
            std::cerr << "Failed to map DRM buffer" << std::endl;
        }

        // Get width, height and strides from the frame
        int width = frame->width;
        int height = frame->height;
        // int stride = drm_frame_desc->layers[0].planes[0].pitch;
                
        // std::cout << "Frame width: " << width << ", height: " << height << ", format: " << frame->format << std::endl;

        int src_format;
        if (drm_frame_desc->layers[0].format == DRM_FORMAT_NV12) {
            src_format = RK_FORMAT_YCbCr_420_SP;  // NV12对应的RGA格式
        } else if (drm_frame_desc->layers[0].format == DRM_FORMAT_YUV420) {
            src_format = RK_FORMAT_YCbCr_420_P;  // YUV420P对应的RGA格式
        } else {
            std::cerr << "Unsupported DRM format for RGA" << std::endl;
            munmap(mapped_memory, size);
            return;
        }

        // 2. 配置源图像信息
        srcRga.fd = -1;
        srcRga.virAddr = mapped_memory;  // 设置源图像的虚拟地址
        srcRga.mmuFlag = 1;
        rga_set_rect(&srcRga.rect, 0, 0, width, height, width, height, src_format);

        // 3. 配置目标图像信息

        // 分配用于缩放后的目标图像缓冲区
        uint8_t* dst_data = (uint8_t*)malloc(model_width * model_height * channel);
        if (dst_data == NULL) {
            std::cerr << "Failed to allocate memory for dst_data" << std::endl;
            return;
        }

        dstRga.fd = -1;
        dstRga.virAddr = dst_data;  // 设置目标图像的虚拟地址
        dstRga.mmuFlag = 1;
        rga_set_rect(&dstRga.rect, 0, 0, model_width, model_height, model_width, model_height, RK_FORMAT_BGR_888);

        // Lock the mutex before calling c_RkRgaBlit
        {
            std::lock_guard<std::mutex> rga_lock(rga_mutex);
            c_RkRgaBlit(&srcRga, &dstRga, NULL);
        }

        // Prepare input for RKNN
        rknn_input inputs[1];
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = params.width * params.height * 3;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].pass_through = 0;
        inputs[0].buf = dst_data;

        rknn_inputs_set(params.ctx, params.io_num.n_input, inputs);

        // Prepare output for RKNN
        rknn_output outputs[params.io_num.n_output];
        memset(outputs, 0, sizeof(outputs));
        for (uint32_t i = 0; i < params.io_num.n_output; ++i)
        {
            outputs[i].index = i;
            outputs[i].want_float = 0;
        }

        // Run inference
        rknn_run(params.ctx, NULL);
        rknn_outputs_get(params.ctx, params.io_num.n_output, outputs, NULL);

        // Prepare for post-processing
        detect_result_group_t detect_result_group;
        std::vector<float> out_scales;
        std::vector<int32_t> out_zps;
        for (uint32_t i = 0; i < params.io_num.n_output; ++i)
        {
            out_scales.push_back(params.output_attrs[i].scale);
            out_zps.push_back(params.output_attrs[i].zp);
        }

        BOX_RECT pads;
        memset(&pads, 0, sizeof(BOX_RECT));
        float scale_w = params.width / static_cast<float>(width);
        float scale_h = params.height / static_cast<float>(height);

        // Perform post-processing
        post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, params.height, params.width,
                     params.box_conf_threshold, params.nms_threshold, pads, scale_w, scale_h, out_zps, out_scales,
                     &detect_result_group);

        // // Print detection thresholds
        // std::cout << "Detection thresholds - Box Threshold: " << params.box_conf_threshold << ", NMS Threshold: " << params.nms_threshold << std::endl;
        
        // Release RKNN output memory
        rknn_outputs_release(params.ctx, params.io_num.n_output, outputs);
        
        // Filter the detection results to keep only the closest box to the center
        if (detect_result_group.count > 0) {
            int imgCenterX = params.width / 2;
            int imgCenterY = params.height / 2;
            float minDistance = std::numeric_limits<float>::max();
            int closestBoxIndex = 0;

            for (int i = 0; i < detect_result_group.count; ++i) {
                int boxCenterX, boxCenterY;
                getBoxCenter(detect_result_group.results[i].box, boxCenterX, boxCenterY);
                float distance = distanceToCenter(boxCenterX, boxCenterY, imgCenterX, imgCenterY);

                if (distance < minDistance) {
                    minDistance = distance;
                    closestBoxIndex = i;
                }
            }

            // Keep only the closest box
            if (closestBoxIndex != 0) {
                detect_result_group.results[0] = detect_result_group.results[closestBoxIndex];
            }
            detect_result_group.count = 1;
        }

        // 存储检测结果
        // detection_results_.push_back(detect_result_group);
                
        // 锁定并推入检测结果队列
        {
            std::lock_guard<std::mutex> lock(detection_results_mutex_);

            // 如果队列不为空，丢弃旧的检测结果
            if (!detection_results_.empty()) {
                detection_results_.pop_back();  // 移除旧的检测结果
            }

            // 插入新的检测结果
            detection_results_.push_back(detect_result_group);
        }
        detection_results_cv_.notify_one();  // 通知等待线程

        free(dst_data);  // Add this after you are done using dst_data

        // Unmap memory after processing
        munmap(mapped_memory, size);

        // 假设处理完后释放帧资源
        av_frame_unref(frame);
        av_frame_free(&frame);

    }
}


// 执行任务 3
void Workflow::task3() {
    std::cout << "Executing task 3..." << std::endl;

    bind_to_cpu(2);

    // Initialize RGA images
    rga_info_t srcRga, dstRga;

    // 初始化源图像和目标图像的 RGA 信息
    memset(&srcRga, 0, sizeof(rga_info_t));
    memset(&dstRga, 0, sizeof(rga_info_t));
    
    detect_result_group_t local_detect_result;  // 本地存储最新的检测结果

    // 使用 config_ 进行操作
    while (true) {
        // 读取最新的检测结果
        {
            std::lock_guard<std::mutex> detection_lock(detection_results_mutex_);
            if (!detection_results_.empty()) {
                local_detect_result = detection_results_.back();  // 复制最新的检测结果
            }
        }

        std::unique_lock<std::mutex> lock(visual_queue_mutex_);

        // 等待直到有新帧加入队列
        visual_queue_cv_.wait(lock, [this]() { return !visual_queue_.empty(); });

        // 从队列中取出一帧
        AVFrame* frame = visual_queue_.front();
        visual_queue_.pop();
        lock.unlock();

        if (!frame || frame->format != AV_PIX_FMT_DRM_PRIME) {
            std::cerr << "Frame is not in DRM_PRIME format" << std::endl;
        }

        AVDRMFrameDescriptor* drm_frame_desc = (AVDRMFrameDescriptor*)frame->data[0];
        if (!drm_frame_desc) {
            std::cerr << "Failed to retrieve DRM frame descriptor" << std::endl;
        }

        // Get the prime fd (file descriptor) for the buffer
        int prime_fd = drm_frame_desc->objects[0].fd;
        if (prime_fd < 0) {
            std::cerr << "Invalid DRM PRIME FD" << std::endl;
        }

        // Get the size of the buffer (you can obtain this from the plane or object)
        uint32_t size = drm_frame_desc->objects[0].size;

        // Map the DRM buffer into memory
        void* mapped_memory = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, prime_fd, 0);
        if (mapped_memory == MAP_FAILED) {
            munmap(mapped_memory, size);  // 释放映射的DRM内存
            std::cerr << "Failed to map DRM buffer" << std::endl;
        }

        // Get width, height and strides from the frame
        int width = frame->width;
        int height = frame->height;
        // int stride = drm_frame_desc->layers[0].planes[0].pitch;

        int src_format;
        if (drm_frame_desc->layers[0].format == DRM_FORMAT_NV12) {
            src_format = RK_FORMAT_YCbCr_420_SP;  // NV12对应的RGA格式
        } else if (drm_frame_desc->layers[0].format == DRM_FORMAT_YUV420) {
            src_format = RK_FORMAT_YCbCr_420_P;  // YUV420P对应的RGA格式
        } else {
            std::cerr << "Unsupported DRM format for RGA" << std::endl;
            munmap(mapped_memory, size);
            return;
        }

        // 2. 配置源图像信息
        srcRga.fd = -1;
        srcRga.virAddr = mapped_memory;  // 设置源图像的虚拟地址
        srcRga.mmuFlag = 1;
        rga_set_rect(&srcRga.rect, 0, 0, width, height, width, height, src_format);

        // 3. 配置目标图像信息
        // 分配用于缩放后的目标图像缓冲区
        uint8_t* dst_data = (uint8_t*)malloc(width * height * 3);
        if (dst_data == NULL) {
            free(dst_data);  // 释放目标图像缓冲区的内存
            std::cerr << "Failed to allocate memory for dst_data" << std::endl;
            return;
        }

        dstRga.fd = -1;
        dstRga.virAddr = dst_data;  // 设置目标图像的虚拟地址
        dstRga.mmuFlag = 1;
        rga_set_rect(&dstRga.rect, 0, 0, width, height, width, height, RK_FORMAT_BGR_888);


        // Lock the mutex before calling c_RkRgaBlit
        {
            std::lock_guard<std::mutex> rga_lock(rga_mutex);
            c_RkRgaBlit(&srcRga, &dstRga, NULL);
        }


        // 4. 使用 OpenCV 将 dst_data 转换为 Mat 并绘制检测框
        cv::Mat dst_img(height, width, CV_8UC3, dst_data);

        for (int i = 0; i < local_detect_result.count; ++i) {
            detect_result_t result = local_detect_result.results[i];

            // 计算检测框的位置
            cv::Rect bbox(result.box.left, result.box.top, 
                          result.box.right - result.box.left, 
                          result.box.bottom - result.box.top);

            // 在目标图像上绘制矩形框
            cv::rectangle(dst_img, bbox, cv::Scalar(0, 255, 0), 2);  // 绿色框，线宽为 2
        }

        // // 5. 将处理后的图像保存为 PNG 文件
        // cv::imwrite("test.png", dst_img);

        // // 将处理后的图像加入共享队列
        // {
        //     std::lock_guard<std::mutex> lock(image_queue_mutex);
        //     shared_image_queue.push(dst_img.clone());  // 将 dst_img 拷贝加入队列
        // }
        // image_queue_cv.notify_one();  // 通知 task4 有新图像
        
        // 设置队列的最大长度
        const size_t max_queue_size = 3;

        // 将处理后的图像加入共享队列
        {
            std::lock_guard<std::mutex> lock(image_queue_mutex);

            // 如果队列已满，丢弃旧帧
            if (shared_image_queue.size() >= max_queue_size) {
                shared_image_queue.pop();  // 丢弃队列中的第一个元素（旧图像）
            }

            // 将处理后的图像 dst_img 拷贝加入队列
            shared_image_queue.push(dst_img.clone());  
        }

        // 通知 task4 有新图像
        image_queue_cv.notify_one();

        // Unmap memory after processing
        munmap(mapped_memory, size);

        // 假设处理完后释放帧资源
        av_frame_unref(frame);
        av_frame_free(&frame);
        
        // 释放目标图像的缓冲区
        free(dst_data);
    }
}


void Workflow::task4() {
    std::cout << "Executing task 4: Video encoding and streaming with Rockchip hardware acceleration..." << std::endl;

    // Initialize FFmpeg library
    avformat_network_init();

    // Set up the output format context for RTMP streaming
    AVFormatContext* fmt_ctx = nullptr;
    avformat_alloc_output_context2(&fmt_ctx, nullptr, "flv", "rtmp://localhost/live/stream");

    if (!fmt_ctx) {
        std::cerr << "Failed to create output context for RTMP stream." << std::endl;
        return;
    }

    // Find H.264 encoder
    const AVCodec* codec = avcodec_find_encoder_by_name("h264_rkmpp");
    if (!codec) {
        std::cerr << "Could not find Rockchip hardware encoder (h264_rkmpp)." << std::endl;
        return;
    }

    // Create a video stream
    AVStream* video_stream = avformat_new_stream(fmt_ctx, nullptr);
    if (!video_stream) {
        std::cerr << "Failed to create video stream." << std::endl;
        return;
    }

    // Create codec context for the encoder
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        std::cerr << "Failed to allocate codec context." << std::endl;
        return;
    }

    // Set codec parameters for hardware-accelerated H.264 encoding
    codec_ctx->width = 1280;
    codec_ctx->height = 960;
    // codec_ctx->width = config_.width;   // Load width from config
    // codec_ctx->height = config_.height; // Load height from config
    codec_ctx->time_base = AVRational{1, 25};  // 25 fps
    codec_ctx->framerate = AVRational{25, 1};
    codec_ctx->pix_fmt = AV_PIX_FMT_NV12;  // Change to NV12 for hardware encoding
    codec_ctx->bit_rate = 400000;  // Set bitrate
    

    if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    // Open the encoder
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        std::cerr << "Failed to open encoder." << std::endl;
        return;
    }

    // Copy the codec parameters to the stream
    if (avcodec_parameters_from_context(video_stream->codecpar, codec_ctx) < 0) {
        std::cerr << "Failed to copy codec parameters to the video stream." << std::endl;
        return;
    }

    // Open the output file for streaming
    if (!(fmt_ctx->flags & AVFMT_NOFILE)) {
        if (avio_open(&fmt_ctx->pb, "rtmp://localhost/live/stream", AVIO_FLAG_WRITE) < 0) {
            std::cerr << "Failed to open RTMP stream." << std::endl;
            return;
        }
    }

    // Write the stream header
    if (avformat_write_header(fmt_ctx, nullptr) < 0) {
        std::cerr << "Failed to write RTMP stream header." << std::endl;
        return;
    }

    // Create AVFrame for encoding
    AVFrame* frame = av_frame_alloc();
    frame->format = codec_ctx->pix_fmt;
    frame->width = codec_ctx->width;
    frame->height = codec_ctx->height;

    // Allocate image buffer for the frame
    if (av_image_alloc(frame->data, frame->linesize, frame->width, frame->height, codec_ctx->pix_fmt, 32) < 0) {
        std::cerr << "Failed to allocate image buffer." << std::endl;
        return;
    }

    // Create a packet for storing encoded data
    AVPacket* packet = av_packet_alloc();
    if (!packet) {
        std::cerr << "Failed to allocate packet." << std::endl;
        return;
    }

    // Initialize RGA images
    rga_info_t srcRga, dstRga;

    // 初始化源图像和目标图像的 RGA 信息
    memset(&srcRga, 0, sizeof(rga_info_t));
    memset(&dstRga, 0, sizeof(rga_info_t));
    
    // Frame generation and encoding loop
    int frame_index = 0;
    while (true) {

        // 从共享队列中获取处理好的图像
        cv::Mat image;
        {
            std::unique_lock<std::mutex> lock(image_queue_mutex);
            image_queue_cv.wait(lock, [this] { return !shared_image_queue.empty(); });  // 捕获 this 指针
            image = shared_image_queue.front();
            shared_image_queue.pop();
        }

        // 2. 配置源图像信息
        srcRga.fd = -1;
        srcRga.virAddr = image.data;  // 设置源图像的虚拟地址
        srcRga.mmuFlag = 1;
        rga_set_rect(&srcRga.rect, 0, 0, image.cols, image.rows, image.cols, image.rows, RK_FORMAT_BGR_888);

        // 3. 配置目标图像信息

        // 分配用于缩放后的目标图像缓冲区
        uint8_t* dst_data = (uint8_t*)malloc(frame->width * frame->height * 3 / 2);
        if (dst_data == NULL) {
            std::cerr << "Failed to allocate memory for dst_data" << std::endl;
            return;
        }

        dstRga.fd = -1;
        dstRga.virAddr = dst_data;  // 设置目标图像的虚拟地址
        dstRga.mmuFlag = 1;
        rga_set_rect(&dstRga.rect, 0, 0, frame->width, frame->height, frame->width, frame->height, RK_FORMAT_YCbCr_420_SP);

        // Lock the mutex before calling c_RkRgaBlit
        {
            std::lock_guard<std::mutex> rga_lock(rga_mutex);
            c_RkRgaBlit(&srcRga, &dstRga, NULL);
        }

        // // 将dst_data的数据复制到image，而不是创建新的dst_img
        // image = cv::Mat(frame->height, frame->width, CV_8UC3, dst_data);

        // After RGA conversion, copy the data directly to AVFrame
        frame->data[0] = dst_data;  // Y plane
        frame->data[1] = dst_data + frame->width * frame->height;  // UV plane (interleaved U and V)


        // for (int y = 0; y < image.rows; ++y) {
        //     for (int x = 0; x < image.cols; ++x) {
        //         cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
        //         frame->data[0][y * frame->linesize[0] + x] = static_cast<uint8_t>(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);  // Y plane
        //         if (y % 2 == 0 && x % 2 == 0) {
        //             frame->data[1][(y / 2) * frame->linesize[1] + x] = static_cast<uint8_t>(-0.169 * pixel[2] - 0.331 * pixel[1] + 0.5 * pixel[0] + 128);  // U plane
        //             frame->data[1][(y / 2) * frame->linesize[1] + x + 1] = static_cast<uint8_t>(0.5 * pixel[2] - 0.419 * pixel[1] - 0.081 * pixel[0] + 128);  // V plane
        //         }
        //     }
        // }

        // Set frame properties
        frame->pts = frame_index++;

        // Send the frame to the encoder
        if (avcodec_send_frame(codec_ctx, frame) < 0) {
            std::cerr << "Error sending frame to encoder." << std::endl;
            break;
        }

        // Receive the encoded packet
        while (avcodec_receive_packet(codec_ctx, packet) == 0) {
            packet->stream_index = video_stream->index;
            packet->pts = av_rescale_q(packet->pts, codec_ctx->time_base, video_stream->time_base);
            packet->dts = av_rescale_q(packet->dts, codec_ctx->time_base, video_stream->time_base);

            // Write the encoded packet to the RTMP stream
            if (av_interleaved_write_frame(fmt_ctx, packet) < 0) {
                std::cerr << "Error writing packet to RTMP stream." << std::endl;
                break;
            }
            av_packet_unref(packet);
        }

        // // 释放目标图像的缓冲区
        free(dst_data);

    }

    // Free resources and close the stream
    // Free image buffer allocated by av_image_alloc
    av_freep(&frame->data[0]);

    // Free packet
    av_packet_free(&packet);

    // Free frame
    av_frame_free(&frame);

    // Close the encoder context
    avcodec_free_context(&codec_ctx);

    // Close the RTMP stream if opened
    if (fmt_ctx && !(fmt_ctx->flags & AVFMT_NOFILE)) {
        avio_closep(&fmt_ctx->pb);
    }

    // Free the format context
    avformat_free_context(fmt_ctx);

    // av_packet_free(&packet);
    // av_frame_free(&frame);
    // avcodec_free_context(&codec_ctx);
    // avformat_free_context(fmt_ctx);

    std::cout << "RTMP streaming finished." << std::endl;
}
