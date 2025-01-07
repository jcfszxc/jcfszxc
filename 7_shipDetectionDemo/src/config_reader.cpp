#include "config_reader.h"
#include <iostream>
#include <fstream>
#include <stdexcept>

// Read the content of the configuration file
std::string readConfigFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open configuration file: " + filename);
    }

    return std::string((std::istreambuf_iterator<char>(file)),
                       (std::istreambuf_iterator<char>()));
}

// Parse the JSON format configuration content
Config parseConfig(const std::string &jsonContent)
{
    Config config;

    // Parse each configuration item
    size_t pos, start, end;
    std::string value;

    // video_device
    pos = jsonContent.find("\"video_device\"");
    if (pos == std::string::npos)
        throw std::runtime_error("'video_device' key not found in configuration file");
    pos = jsonContent.find(':', pos);
    start = jsonContent.find('"', pos);
    end = jsonContent.find('"', start + 1);
    config.videoDevice = jsonContent.substr(start + 1, end - start - 1);
    
    // model_path
    pos = jsonContent.find("\"model_path\"");
    if (pos == std::string::npos)
        throw std::runtime_error("'model_path' key not found in configuration file");
    pos = jsonContent.find(':', pos);
    start = jsonContent.find('"', pos);
    end = jsonContent.find('"', start + 1);
    config.modelPath = jsonContent.substr(start + 1, end - start - 1);

    // BOX_THRESH
    pos = jsonContent.find("\"BOX_THRESH\"");
    if (pos == std::string::npos)
        throw std::runtime_error("'BOX_THRESH' key not found in configuration file");
    pos = jsonContent.find(':', pos);
    start = jsonContent.find_first_not_of(" \t\n\r", pos + 1);
    end = jsonContent.find_first_of(",}", start);
    config.boxThresh = std::stof(jsonContent.substr(start, end - start));

    // NMS_THRESH
    pos = jsonContent.find("\"NMS_THRESH\"");
    if (pos == std::string::npos)
        throw std::runtime_error("'NMS_THRESH' key not found in configuration file");
    pos = jsonContent.find(':', pos);
    start = jsonContent.find_first_not_of(" \t\n\r", pos + 1);
    end = jsonContent.find_first_of(",}", start);
    config.nmsThresh = std::stof(jsonContent.substr(start, end - start));

    // width
    pos = jsonContent.find("\"width\"");
    if (pos == std::string::npos)
        throw std::runtime_error("'width' key not found in configuration file");
    pos = jsonContent.find(':', pos);
    start = jsonContent.find_first_not_of(" \t\n\r", pos + 1);
    end = jsonContent.find_first_of(",}", start);
    config.width = std::stoi(jsonContent.substr(start, end - start));

    // height
    pos = jsonContent.find("\"height\"");
    if (pos == std::string::npos)
        throw std::runtime_error("'height' key not found in configuration file");
    pos = jsonContent.find(':', pos);
    start = jsonContent.find_first_not_of(" \t\n\r", pos + 1);
    end = jsonContent.find_first_of(",}", start);
    config.height = std::stoi(jsonContent.substr(start, end - start));

    // width
    pos = jsonContent.find("\"width\"");
    if (pos == std::string::npos)
        throw std::runtime_error("'width' key not found in configuration file");
    pos = jsonContent.find(':', pos);
    start = jsonContent.find_first_not_of(" \t\n\r", pos + 1);
    end = jsonContent.find_first_of(",}", start);
    config.width = std::stoi(jsonContent.substr(start, end - start));

    // height
    pos = jsonContent.find("\"height\"");
    if (pos == std::string::npos)
        throw std::runtime_error("'height' key not found in configuration file");
    pos = jsonContent.find(':', pos);
    start = jsonContent.find_first_not_of(" \t\n\r", pos + 1);
    end = jsonContent.find_first_of(",}", start);
    config.height = std::stoi(jsonContent.substr(start, end - start));

    // Output parsed configuration
    std::cout << "Parsed configuration: " << std::endl;
    std::cout << "video_device: " << config.videoDevice << std::endl;
    std::cout << "model_path: " << config.modelPath << std::endl;
    std::cout << "BOX_THRESH: " << config.boxThresh << std::endl;
    std::cout << "NMS_THRESH: " << config.nmsThresh << std::endl;
    std::cout << "width: " << config.width << std::endl;
    std::cout << "height: " << config.height << std::endl;


    return config;
}

// Load and parse the configuration file
Config loadConfig(const std::string &configPath)
{
    try
    {
        std::string jsonContent = readConfigFile(configPath);
        return parseConfig(jsonContent);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        exit(-1);
    }
}