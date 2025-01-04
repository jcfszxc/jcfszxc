#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/28 21:51
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : config.py
# @Description   : 

import os
from pathlib import Path

class Config:
    """应用程序配置类"""
    
    # 应用基础配置
    APP_NAME = "Task Manager"
    APP_VERSION = "1.0.0"
    
    # 路径配置
    BASE_DIR = Path(__file__).parent.parent
    DATABASE_DIR = BASE_DIR / "data"
    DATABASE_PATH = DATABASE_DIR / "tasks.db"
    LOG_DIR = BASE_DIR / "logs"
    
    # 数据库配置
    DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def init_app(cls):
        """初始化应用程序配置"""
        # 创建必要的目录
        cls.DATABASE_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)