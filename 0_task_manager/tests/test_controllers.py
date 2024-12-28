import pytest
from datetime import datetime
from src.controllers.task_controller import TaskController
from src.constants import Priority, TaskStatus

@pytest.fixture
def task_controller():
    """任务控制器fixture"""
    controller = TaskController()
    yield controller

def test_task_creation(task_controller):
    """测试创建任务"""
    task = task_controller.create_task(
        name="测试任务",
        priority=Priority.HIGH,
        description="测试描述"
    )
    
    assert task.id is not None
    assert task.name == "测试任务"
    assert task.priority == Priority.HIGH
    assert task.description == "测试描述"

def test_task_retrieval(task_controller):
    """测试获取任务"""
    # 创建测试任务
    task = task_controller.create_task("测试任务", Priority.MEDIUM)
    
    # 获取所有任务
    tasks = task_controller.get_all_tasks()
    assert len(tasks) > 0
    
    # 根据ID获取任务
    retrieved_task = task_controller.get_task_by_id(task.id)
    assert retrieved_task is not None
    assert retrieved_task.name == "测试任务"

def test_task_update(task_controller):
    """测试更新任务"""
    # 创建测试任务
    task = task_controller.create_task("原始任务", Priority.LOW)
    
    # 更新任务
    task.name = "更新后的任务"
    task.priority = Priority.HIGH
    task_controller.update_task(task)
    
    # 验证更新
    updated_task = task_controller.get_task_by_id(task.id)
    assert updated_task.name == "更新后的任务"
    assert updated_task.priority == Priority.HIGH

def test_task_deletion(task_controller):
    """测试删除任务"""
    # 创建测试任务
    task = task_controller.create_task("要删除的任务", Priority.MEDIUM)
    
    # 删除任务
    task_controller.delete_task(task.id)
    
    # 验证删除
    deleted_task = task_controller.get_task_by_id(task.id)
    assert deleted_task is None