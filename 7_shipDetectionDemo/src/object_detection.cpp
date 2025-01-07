#include "object_detection.h"
#include <iostream>
#include <cmath>

// Global variables for thread-safe queue operations
std::queue<cv::Mat> detectionQueue;
std::mutex queueMutex;
std::condition_variable detectionCondition;
std::atomic<bool> stopDetection(false);
std::queue<std::vector<DetectionResult>> resultQueue;
std::condition_variable resultCondition;


// Function to calculate distance from center
float distanceFromCenter(const cv::Rect& bbox, int frameWidth, int frameHeight) {
    float centerX = frameWidth / 2.0f;
    float centerY = frameHeight / 2.0f;
    float bboxCenterX = bbox.x + bbox.width / 2.0f;
    float bboxCenterY = bbox.y + bbox.height / 2.0f;
    return std::sqrt(std::pow(bboxCenterX - centerX, 2) + std::pow(bboxCenterY - centerY, 2));
}

// New function to check if bbox is too close to edges
bool isTooCloseToEdge(const cv::Rect& bbox, int frameWidth, int frameHeight, float edgeThreshold) {
    return (bbox.x < edgeThreshold * frameWidth ||
            bbox.y < edgeThreshold * frameHeight ||
            (bbox.x + bbox.width) > (1 - edgeThreshold) * frameWidth ||
            (bbox.y + bbox.height) > (1 - edgeThreshold) * frameHeight);
}

void runObjectDetection(const DetectionParams &params)
{
    while (!stopDetection)
    {

        // Wait for a frame to be available in the queue
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            detectionCondition.wait(lock, []
                                    { return !detectionQueue.empty() || stopDetection; });
            if (stopDetection)
                break;
            frame = detectionQueue.front();
            detectionQueue.pop();
        }

        // Check if the frame is empty or invalid
        if (frame.empty() || frame.cols == 0 || frame.rows == 0)
        {
            std::cerr << "Received an empty or invalid frame. Skipping detection." << std::endl;
            continue;
        }

        // Ensure the frame size matches the expected input size
        if (frame.cols != params.width || frame.rows != params.height)
        {
            std::cerr << "Frame size mismatch. Expected " << params.width << "x" << params.height
                      << ", got " << frame.cols << "x" << frame.rows << ". Resizing." << std::endl;
            cv::resize(frame, frame, cv::Size(params.width, params.height));
        }

        // Prepare input for RKNN
        rknn_input inputs[1];
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = params.width * params.height * 3;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].pass_through = 0;
        inputs[0].buf = frame.data;

        int ret = rknn_inputs_set(params.ctx, params.io_num.n_input, inputs);
        if (ret < 0)
        {
            std::cerr << "rknn_inputs_set error. Skipping this frame." << std::endl;
            continue;
        }
        
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
        float scale_w = params.width / static_cast<float>(frame.cols);
        float scale_h = params.height / static_cast<float>(frame.rows);

        // Perform post-processing
        post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, params.height, params.width,
                     params.box_conf_threshold, params.nms_threshold, pads, scale_w, scale_h, out_zps, out_scales,
                     &detect_result_group);

        // Release RKNN output memory
        rknn_outputs_release(params.ctx, params.io_num.n_output, outputs);

        // Find the detection result closest to center
        DetectionResult nearestResult;
        float minDistance = std::numeric_limits<float>::max();
        float edgeThreshold = 0.1f;

        for (int i = 0; i < detect_result_group.count; i++)
        {
            detect_result_t det_result = detect_result_group.results[i];
            cv::Rect bbox(det_result.box.left, det_result.box.top,
                          det_result.box.right - det_result.box.left,
                          det_result.box.bottom - det_result.box.top);
            
            float distance = distanceFromCenter(bbox, frame.cols, frame.rows);

            // Check if the bounding box is too close to the edge
            if (isTooCloseToEdge(bbox, frame.cols, frame.rows, edgeThreshold)) {
                continue; // Skip this detection
            }

            if (distance < minDistance)
            {
                minDistance = distance;
                nearestResult.label = det_result.name;
                nearestResult.confidence = det_result.prop;
                nearestResult.bbox = bbox;
            }
        }

        // Add only the nearest result to the output queue
        std::vector<DetectionResult> detectionResults;
        if (detect_result_group.count > 0)  // Check if we found any detection
        {
            detectionResults.push_back(nearestResult);
        }

        // Add results to the output queue
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            resultQueue.push(detectionResults);
        }
        resultCondition.notify_one();
    }
}

// Helper function to load data from file
unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

// Load model from file
unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

// Initialize RKNN context
rknn_context initializeRKNN(const std::string& modelPath, rknn_input_output_num& io_num) {
    rknn_context ctx;
    int ret;
    
    printf("Loading model...\n");
    int model_data_size = 0;
    unsigned char *model_data = load_model(modelPath.c_str(), &model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        exit(-1);
    }
    
    // Query SDK version
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        printf("rknn_query error ret=%d\n", ret);
        exit(-1);
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);
    
    // Query input/output numbers
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_query error ret=%d\n", ret);
        exit(-1);
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
    
    return ctx;
}

// Query and dump tensor attributes
void queryAndDumpTensorAttrs(rknn_context ctx, rknn_input_output_num io_num, 
                             std::vector<rknn_tensor_attr>& input_attrs, 
                             std::vector<rknn_tensor_attr>& output_attrs) {
    int ret;
    // Query input tensor attributes
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_query error ret=%d\n", ret);
            exit(-1);
        }
        dump_tensor_attr(&(input_attrs[i]));
    }
    
    // Query output tensor attributes
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
    }
}

// Create detection parameters
DetectionParams createDetectionParams(rknn_context ctx, rknn_input_output_num io_num,
                                      const std::vector<rknn_tensor_attr>& input_attrs,
                                      const std::vector<rknn_tensor_attr>& output_attrs,
                                      int width, int height, float box_conf_threshold, float nms_threshold) {
    DetectionParams params;
    params.ctx = ctx;
    params.io_num = io_num;
    params.input_attrs = input_attrs;
    params.output_attrs = output_attrs;
    params.width = width;
    params.height = height;
    params.box_conf_threshold = box_conf_threshold;
    params.nms_threshold = nms_threshold;
    return params;
}