# Video Frame Extractor (视频帧提取器)

一个基于Python的视频帧提取工具，提供图形用户界面，支持批量处理视频文件并提取关键帧。

## 功能特性

- 📹 支持多种视频格式（mp4, avi, h264）
- 🖼️ 智能帧提取，避免冗余
- 📁 保持原始目录结构
- 🚀 多线程并行处理
- 📊 实时进度显示
- ⏹️ 支持随时取消任务

## 系统要求

- Python 3.7+
- OpenCV
- PyQt6

## 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/video-frame-extractor.git
cd video-frame-extractor
```

2. 创建虚拟环境（可选但推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 启动程序：
```bash
python main.py
```

2. 在界面上选择：
   - 输入视频目录：包含需要处理的视频文件的文件夹
   - 输出图片目录：提取的帧将保存在此目录下

3. 点击"开始处理"按钮开始提取帧

4. 处理过程中可以：
   - 查看实时进度
   - 随时取消处理
   - 查看处理统计信息

## 项目结构

```
video_frame_extractor/
├── requirements.txt
├── main.py
├── src/
│   ├── __init__.py
│   ├── core/           # 核心处理逻辑
│   │   ├── __init__.py
│   │   ├── video_processor.py
│   │   └── utils.py
│   └── ui/            # 用户界面相关
│       ├── __init__.py
│       ├── main_window.py
│       └── processing_dialog.py
```

## TODO列表

### 界面优化
- [ ] 添加深色主题支持
- [ ] 优化进度条显示效果
- [ ] 添加处理任务队列管理界面
- [ ] 添加视频预览窗口
- [ ] 多任务并行处理可视化（显示每个任务的单独进度）

### 功能增强
- [ ] 添加帧采样频率设置
  - [ ] 支持固定间隔采样
  - [ ] 支持基于场景变化的智能采样
  - [ ] 提供预设采样方案
- [ ] 图片保存设置
  - [ ] 自定义默认保存路径
  - [ ] 支持多种图片格式（JPEG、PNG、WEBP等）
  - [ ] 图片质量参数调整
- [ ] 批处理增强
  - [ ] 任务优先级设置
  - [ ] 自定义并行任务数
  - [ ] 处理资源占用监控
- [ ] 导出处理报告
  - [ ] 详细的处理统计信息
  - [ ] 错误日志记录
  - [ ] 处理时间分析

### 性能优化
- [ ] 内存使用优化
- [ ] GPU加速支持
- [ ] 大文件处理优化

### 其他功能
- [ ] 添加配置文件支持
- [ ] 多语言支持
- [ ] 自动更新检查
- [ ] 处理结果预览
- [ ] 批处理任务的保存和恢复

## 贡献指南

欢迎提交Pull Request或Issue来帮助改进这个项目！

## 许可证

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 作者

[jcfszxc]

## 更新日志

### v1.0.0 (2024-01-02)
- 初始版本发布
- 基本的视频帧提取功能
- 图形用户界面
- 多线程处理支持