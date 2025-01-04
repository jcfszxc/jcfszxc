# Task Manager

一个基于PyQt6的个人任务管理应用。

## 功能特点

- 任务管理：创建、编辑、删除任务
- 优先级设置：为任务设置优先级（低、中、高）
- 截止日期：设置任务截止日期
- 状态追踪：跟踪任务完成状态
- 数据持久化：使用SQLite数据库存储任务数据
- 日志记录：完整的日志记录功能

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/task-manager.git
cd task-manager
```

2. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 安装项目：
```bash
pip install -e .
```

## 使用说明

1. 启动应用：
```bash
task-manager
```

2. 使用界面：
- 点击"添加任务"按钮创建新任务
- 在任务列表中管理现有任务
- 使用右侧按钮编辑、完成或删除任务

## 项目结构

```
task_manager/
├── docs/                 # 文档
├── src/                 # 源代码
│   ├── models/         # 数据模型
│   ├── views/          # 视图
│   ├── controllers/    # 控制器
│   └── utils/          # 工具函数
├── tests/              # 测试文件
└── requirements.txt    # 项目依赖
```

## 开发说明

1. 运行测试：
```bash
pytest
```

2. 代码格式化：
```bash
black src tests
```

3. 静态类型检查：
```bash
mypy src
```

## 贡献指南

1. Fork 项目
2. 创建特性分支：`git checkout -b feature/AmazingFeature`
3. 提交更改：`git commit -m 'Add some AmazingFeature'`
4. 推送分支：`git push origin feature/AmazingFeature`
5. 提交 Pull Request

## 许可证

本项目基于 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

作者 - [@jcfszxc](https://github.com/jcfszxc)

项目链接: [https://github.com/jcfszxc/jcfszxc](https://github.com/jcfszxc/jcfszxc)