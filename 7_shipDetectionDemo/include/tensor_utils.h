#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include "rga/im2d.h"
#include "rga/RgaApi.h"

// Function to print tensor attributes
void dump_tensor_attr(rknn_tensor_attr *attr);

// Function to resize image using RGA
int resize_rga(rga_buffer_t &src, rga_buffer_t &dst, const cv::Mat &image, cv::Mat &resized_image, const cv::Size &target_size);

// Utility functions for string representations
const char* get_format_string(rknn_tensor_format fmt);
const char* get_type_string(rknn_tensor_type type);
const char* get_qnt_type_string(rknn_tensor_qnt_type qnt_type);

#endif // TENSOR_UTILS_H