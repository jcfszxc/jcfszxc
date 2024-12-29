#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/29 19:59
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : api.py
# @Description   : 

import requests
from datetime import datetime
from typing import List, Dict, Optional
from .data_models import WeatherData, ForecastData
from config import API_KEY, BASE_URL

class WeatherAPI:
    """天气API调用类"""
    
    @staticmethod
    def kelvin_to_celsius(kelvin: float) -> float:
        """将开尔文温度转换为摄氏度"""
        return kelvin - 273.15
    
    @classmethod
    def get_current_weather(cls, city_code: str) -> Optional[WeatherData]:
        """获取当前天气数据"""
        try:
            url = f"{BASE_URL}/weather"
            params = {
                "q": city_code,
                "appid": API_KEY
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return WeatherData(
                temperature=cls.kelvin_to_celsius(data['main']['temp']),
                humidity=data['main']['humidity'],
                wind_speed=data['wind']['speed'],
                description=data['weather'][0]['description'],
                icon=data['weather'][0]['icon'],
                timestamp=datetime.now()
            )
        except Exception as e:
            print(f"Error fetching current weather: {e}")
            return None

    @classmethod
    def get_forecast(cls, city_code: str) -> List[ForecastData]:
        """获取天气预报数据"""
        try:
            url = f"{BASE_URL}/forecast"
            params = {
                "q": city_code,
                "appid": API_KEY
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            forecasts = []
            
            for item in data['list'][:5]:  # 获取5天预报
                forecast = ForecastData(
                    date=datetime.fromtimestamp(item['dt']),
                    temp_min=cls.kelvin_to_celsius(item['main']['temp_min']),
                    temp_max=cls.kelvin_to_celsius(item['main']['temp_max']),
                    description=item['weather'][0]['description'],
                    icon=item['weather'][0]['icon']
                )
                forecasts.append(forecast)
            
            return forecasts
        except Exception as e:
            print(f"Error fetching forecast: {e}")
            return []