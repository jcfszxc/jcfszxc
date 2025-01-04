#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/28 21:53
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : logger.py
# @Description   : 

import logging
import os
from pathlib import Path
from src.config import Config

def get_logger(name: str) -> logging.Logger:
    """获取日志记录器"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        # 确保日志目录存在
        Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # 创建文件处理器
        log_file = Config.LOG_DIR / f"{name.replace('.','-')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
        logger.addHandler(file_handler)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
        logger.addHandler(console_handler)
    
    return logger