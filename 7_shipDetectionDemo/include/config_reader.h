#ifndef CONFIG_READER_H
#define CONFIG_READER_H

#include <string>

// Configuration structure
struct Config {
    bool useRTSP;
    std::string rtspUrl;
    std::string videoDevice;
    std::string modelPath;
    float boxThresh;
    float nmsThresh;
    int width=960;
    int height=640;
};

// Function declarations
std::string readConfigFile(const std::string &filename);
Config parseConfig(const std::string &jsonContent);
Config loadConfig(const std::string& configPath);

#endif // CONFIG_READER_H