from typing import Tuple, Callable
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import Qt

class PointPairWidget(QPushButton):
    """特征点对显示组件"""
    
    STYLE_TEMPLATE = """
        QPushButton {
            background-color: #f0f0f0;
            border: 2px solid #c0c0c0;
            border-radius: 5px;
            padding: 5px;
            text-align: center;  /* 添加文本居中对齐 */
        }
        QPushButton:hover {
            background-color: #e0e0e0;
        }
        QPushButton[selected="true"] {
            background-color: #90EE90;
            border: 2px solid #008000;
        }
    """
    
    def __init__(
        self,
        index: int,
        visible_point: Tuple[float, float],
        ir_point: Tuple[float, float],
        click_handler: Callable[[int], None]
    ):
        """
        Args:
            index: 点对索引
            visible_point: 可见光图像上的点坐标
            ir_point: 红外图像上的点坐标
            click_handler: 点击事件处理函数
        """
        # 创建按钮文本,显示索引和坐标
        button_text = (
            f"点对 {index + 1}\n"
            f"可见光: ({int(visible_point[0])}, {int(visible_point[1])})\n"
            f"红外: ({int(ir_point[0])}, {int(ir_point[1])})"
        )
        super().__init__(button_text)
        
        self.setMinimumSize(120, 80)  # 增加按钮大小以容纳更多文本
        self.setStyleSheet(self.STYLE_TEMPLATE)
        
        tooltip = (
            f"可见光: ({int(visible_point[0])}, {int(visible_point[1])})\n"
            f"红外: ({int(ir_point[0])}, {int(ir_point[1])})"
        )
        self.setToolTip(tooltip)
        
        self.clicked.connect(lambda: click_handler(index))