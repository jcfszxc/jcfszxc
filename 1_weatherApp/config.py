#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/29 19:58
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : config.py
# @Description   : 

import os
from dotenv import load_dotenv

load_dotenv()

# OpenWeatherMap API配置
API_KEY = os.getenv('WEATHER_API_KEY', 'cd6ecd53961857ef29e1cf4cc3167620')  # 请替换为您的API密钥
BASE_URL = "http://api.openweathermap.org/data/2.5"

# 默认城市列表
DEFAULT_CITIES = [
    {"name": "北京", "code": "beijing,cn"},
    {"name": "上海", "code": "shanghai,cn"},
    {"name": "广州", "code": "guangzhou,cn"}
]