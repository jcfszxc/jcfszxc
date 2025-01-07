#include <iostream>
#include "workflow.h"

int main() {
    // 创建并运行工作流
    Workflow workflow;  // 正确的对象实例化，没有括号
    workflow.run();     // 调用 run 方法

    return 0;
}
