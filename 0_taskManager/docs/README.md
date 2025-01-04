# Task Manager

A Python-based task management system with command-line interface.

## Features

- Create, read, update, and delete tasks
- Set task priorities and due dates
- Mark tasks as complete/incomplete
- Validate task data
- Command-line interface

## Installation

```bash
pip install -r requirements.txt
python setup.py install
```

## Quick Start

```python
from src.controllers.task_controller import TaskController
from src.models.task import Task

# Initialize controller
controller = TaskController()

# Create a new task
task = controller.create_task(
    name="Complete project documentation",
    priority="high",
    due_date="2024-12-31"
)

# Mark task as complete
controller.update_task(task.id, status="completed")
```

## API Documentation

### Models

#### Task

```python
class Task:
    def __init__(self, name, priority="medium", due_date=None):
        self.id = uuid.uuid4()
        self.name = name
        self.priority = priority
        self.due_date = due_date
        self.status = "pending"
        self.created_at = datetime.now()
```

Attributes:
- `id`: Unique identifier (UUID)
- `name`: Task name (string)
- `priority`: Priority level ("low", "medium", "high")
- `due_date`: Due date (datetime)
- `status`: Current status ("pending", "completed")
- `created_at`: Creation timestamp

### Controllers

#### TaskController

Main methods:
- `create_task(name, priority, due_date)`: Creates a new task
- `get_task(task_id)`: Retrieves a task by ID
- `update_task(task_id, **kwargs)`: Updates task attributes
- `delete_task(task_id)`: Deletes a task
- `list_tasks()`: Lists all tasks

### Validators

- `validate_name(name)`: Validates task name
- `validate_priority(priority)`: Validates priority level
- `validate_status(status)`: Validates task status
- `validate_date(date)`: Validates due date format

## Examples

### Basic Task Management

```python
from src.controllers.task_controller import TaskController

def main():
    # Initialize controller
    controller = TaskController()
    
    # Create tasks
    task1 = controller.create_task(
        name="Write project proposal",
        priority="high",
        due_date="2024-12-30"
    )
    
    task2 = controller.create_task(
        name="Schedule team meeting",
        priority="medium",
        due_date="2024-12-29"
    )
    
    # List all tasks
    tasks = controller.list_tasks()
    for task in tasks:
        print(f"Task: {task.name}, Priority: {task.priority}")
    
    # Update task
    controller.update_task(task1.id, status="completed")
    
    # Delete task
    controller.delete_task(task2.id)

if __name__ == "__main__":
    main()
```

### Task Filtering and Sorting

```python
def filter_high_priority_tasks(controller):
    tasks = controller.list_tasks()
    high_priority = [task for task in tasks if task.priority == "high"]
    return high_priority

def get_overdue_tasks(controller):
    tasks = controller.list_tasks()
    today = datetime.now()
    overdue = [task for task in tasks 
              if task.due_date and task.due_date < today 
              and task.status != "completed"]
    return overdue

def sort_tasks_by_due_date(controller):
    tasks = controller.list_tasks()
    return sorted(tasks, key=lambda x: x.due_date or datetime.max)
```

### Error Handling

```python
from src.utils.validators import ValidationError

def safe_create_task(controller, name, priority, due_date):
    try:
        task = controller.create_task(name, priority, due_date)
        print(f"Successfully created task: {task.name}")
        return task
    except ValidationError as e:
        print(f"Validation error: {str(e)}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.