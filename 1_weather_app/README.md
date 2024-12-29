# Weather App

一个简单的天气预报桌面应用，使用Python和PyQt6开发。

## 功能特点

- 实时天气显示
- 未来5天天气预报
- 多城市切换
- 自动更新天气数据
- 清爽现代的用户界面

## 安装说明

1. 克隆项目到本地：
```bash
git clone <repository-url>
cd weather-app
```

2. 创建并激活虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 配置API密钥：
   - 在 [OpenWeatherMap](https://openweathermap.org/api) 注册并获取API密钥
   - 创建 `.env` 文件并添加以下内容：
   ```
   WEATHER_API_KEY=your_api_key_here
   ```

5. 运行应用：
```bash
python main.py
```

## 项目结构

```
weather_app/
│  README.md
│  requirements.txt
│  main.py
│  config.py
│
├─src/
│  ├─ui/          # 界面相关
│  ├─core/        # 核心功能
│  └─resources/   # 资源文件
│
└─tests/          # 单元测试
```

## 使用说明

1. 启动应用后，从下拉菜单选择城市
2. 点击"刷新"按钮手动更新天气数据
3. 应用会每5分钟自动更新一次数据

## 注意事项

- 请确保已安装所有依赖
- 需要有效的OpenWeatherMap API密钥
- 确保网络连接