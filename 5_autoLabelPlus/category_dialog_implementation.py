from PyQt5.QtWidgets import QDialog, QApplication, QMessageBox
from PyQt5.QtCore import Qt
import sys

# 导入生成的UI类
from category_dialog import Ui_CategoryDialog

class CategoryDialog(QDialog, Ui_CategoryDialog):
    def __init__(self, parent=None):
        super(CategoryDialog, self).__init__(parent)
        self.setupUi(self)
        
        # 连接信号和槽
        self.buttonBox.accepted.connect(self.validate_and_accept)
        self.buttonBox.rejected.connect(self.reject)
        self.autoLabelCheckBox.stateChanged.connect(self.on_auto_label_changed)
        
        # 添加列表项点击信号连接
        self.categoryList.itemClicked.connect(self.on_item_clicked)
        # 添加双击信号连接
        self.categoryList.itemDoubleClicked.connect(self.on_item_double_clicked)
        
        # 初始化列表
        self.init_category_list()
        
    def init_category_list(self):
        """初始化类别列表"""
        categories = ["person", "car", "bicycle", "dog", "cat"]
        self.categoryList.clear()
        self.categoryList.addItems(categories)
    
    def on_item_clicked(self, item):
        """列表项被点击时的处理"""
        self.lineEdit.setText(item.text())
    
    def on_item_double_clicked(self, item):
        """列表项被双击时的处理"""
        self.lineEdit.setText(item.text())
        self.validate_and_accept()
                
    def on_auto_label_changed(self, state):
        """自动标注复选框状态变化时的处理"""
        is_checked = state == Qt.Checked
        print(f"Auto label is {'enabled' if is_checked else 'disabled'}")
        
    def get_selected_category(self):
        """获取选中的类别"""
        return self.lineEdit.text()
        
    def validate_and_accept(self):
        """验证输入并决定是否接受对话框"""
        if not self.lineEdit.text().strip():
            QMessageBox.warning(self, "输入验证", "请输入或选择一个类别！")
            return
        self.accept()

def main():
    app = QApplication(sys.argv)
    dialog = CategoryDialog()
    
    if dialog.exec_() == QDialog.Accepted:
        selected_category = dialog.get_selected_category()
        auto_label_enabled = dialog.autoLabelCheckBox.isChecked()
        print(f"Selected category: {selected_category}")
        print(f"Auto label enabled: {auto_label_enabled}")
    
if __name__ == "__main__":
    main()