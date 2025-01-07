#ifndef OBJECT_DETECTION_H
#define OBJECT_DETECTION_H

#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <stdio.h>
#include <stdlib.h>
#include "postprocess.h"
#include "tensor_utils.h"

// Struct to hold detection results
struct DetectionResult
{
    std::string label;
    float confidence;
    cv::Rect bbox;
};

// Struct to hold detection parameters
struct DetectionParams
{
    rknn_context ctx;
    rknn_input_output_num io_num;
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    int width;
    int height;
    float box_conf_threshold;
    float nms_threshold;
};

// Global variables for thread-safe queue operations
extern std::queue<cv::Mat> detectionQueue;
extern std::mutex queueMutex;
extern std::condition_variable detectionCondition;
extern std::atomic<bool> stopDetection;
extern std::queue<std::vector<DetectionResult>> resultQueue;
extern std::condition_variable resultCondition;

// Main object detection function
void runObjectDetection(const DetectionParams &params);

#ifdef __cplusplus
extern "C"
{
#endif

// Utility functions for loading data and model
unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
unsigned char *load_model(const char *filename, int *model_size);

// RKNN initialization and query functions
rknn_context initializeRKNN(const std::string &modelPath, rknn_input_output_num &io_num);
void queryAndDumpTensorAttrs(rknn_context ctx, rknn_input_output_num io_num,
                             std::vector<rknn_tensor_attr> &input_attrs,
                             std::vector<rknn_tensor_attr> &output_attrs);

// Function to create detection parameters
DetectionParams createDetectionParams(rknn_context ctx, rknn_input_output_num io_num,
                                      const std::vector<rknn_tensor_attr> &input_attrs,
                                      const std::vector<rknn_tensor_attr> &output_attrs,
                                      int width, int height, float box_conf_threshold, float nms_threshold);

#ifdef __cplusplus
}
#endif

#endif // OBJECT_DETECTION_H