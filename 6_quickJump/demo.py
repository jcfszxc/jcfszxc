import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from pathlib import Path
from tkinter.font import Font

class ProjectManager:
    def __init__(self):
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("项目管理器")
        self.root.geometry("800x500")  # 增加窗口尺寸以获得更好的视觉效果
        
        # 配置主题颜色
        self.colors = {
            'bg': '#f0f0f0',
            'button_bg': '#2196F3',
            'button_fg': 'white',
            'hover_bg': '#1976D2',
            'list_bg': 'white',
            'list_select': '#E3F2FD'
        }
        
        # 设置窗口样式
        self.root.configure(bg=self.colors['bg'])
        self.style = ttk.Style()
        self.style.configure('Modern.TButton', 
                           padding=10, 
                           font=('Microsoft YaHei UI', 10))
        
        # 初始化项目数据
        self.projects = self.load_projects()
        
        # 创建自定义字体
        self.title_font = Font(family='Microsoft YaHei UI', size=12, weight='bold')
        self.list_font = Font(family='Microsoft YaHei UI', size=10)
        
        # 创建界面
        self.create_widgets()
        
        # 绑定快捷键
        self.bind_shortcuts()
        

    def create_widgets(self):
        """创建美化后的界面组件"""
        # 创建主框架，添加内边距
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建标题标签
        title_label = tk.Label(main_frame, 
                             text="项目管理器", 
                             font=self.title_font, 
                             bg=self.colors['bg'])
        title_label.pack(pady=(0, 10))
        
        # 创建左侧的项目列表框架
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # 创建搜索框
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.filter_projects)
        search_frame = ttk.Frame(list_frame)
        search_frame.pack(fill=tk.X, pady=(0, 5))
        
        search_entry = ttk.Entry(search_frame, 
                               textvariable=self.search_var, 
                               font=self.list_font)
        search_entry.pack(fill=tk.X)
        search_entry.insert(0, "搜索项目...")
        search_entry.bind('<FocusIn>', lambda e: search_entry.delete(0, tk.END) 
                         if search_entry.get() == "搜索项目..." else None)
        
        # 创建项目列表
        self.project_listbox = tk.Listbox(list_frame, 
                                        font=self.list_font,
                                        selectmode=tk.SINGLE,
                                        activestyle='none',
                                        bg=self.colors['list_bg'],
                                        selectbackground=self.colors['list_select'],
                                        selectforeground='black',
                                        relief=tk.FLAT)
        self.project_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 美化滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 配置滚动条和列表框的联动
        self.project_listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.configure(command=self.project_listbox.yview)
        
        # 创建右侧的按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建美化后的按钮
        buttons = [
            ("添加项目", self.add_project, "Ctrl+N"),
            ("打开项目", self.open_project, "Ctrl+O"),
            ("删除项目", self.delete_project, "Delete")
        ]
        
        for text, command, shortcut in buttons:
            button_frame = ttk.Frame(button_frame)
            button_frame.pack(pady=5)
            
            btn = ttk.Button(button_frame, 
                           text=text, 
                           command=command, 
                           style='Modern.TButton',
                           width=15)
            btn.pack()
            
            # 添加快捷键提示
            shortcut_label = tk.Label(button_frame, 
                                    text=shortcut, 
                                    font=('Microsoft YaHei UI', 8),
                                    fg='gray',
                                    bg=self.colors['bg'])
            shortcut_label.pack()
        
        # 添加状态栏
        self.status_var = tk.StringVar()
        status_bar = tk.Label(self.root, 
                            textvariable=self.status_var, 
                            bd=1, 
                            relief=tk.SUNKEN, 
                            anchor=tk.W,
                            font=('Microsoft YaHei UI', 9))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 绑定双击事件
        self.project_listbox.bind('<Double-Button-1>', lambda e: self.open_project())
        
        # 更新项目列表显示
        self.update_project_list()
        
        # 更新项目列表显示
        self.update_project_list()


    def filter_projects(self, *args):
        """根据搜索框内容过滤项目列表"""
        search_text = self.search_var.get().lower()
        if search_text == "搜索项目...":
            return
            
        self.project_listbox.delete(0, tk.END)
        for project_name in sorted(self.projects.keys()):
            if search_text in project_name.lower():
                self.project_listbox.insert(tk.END, project_name)

    def bind_shortcuts(self):
        """绑定快捷键"""
        self.root.bind('<Control-n>', lambda e: self.add_project())
        self.root.bind('<Control-o>', lambda e: self.open_project())
        self.root.bind('<Delete>', lambda e: self.delete_project())
        
    def add_project(self):
        """添加新项目"""
        # 打开文件夹选择对话框
        folder_path = filedialog.askdirectory(title="选择项目文件夹")
        if folder_path:
            # 获取文件夹名作为项目名
            project_name = os.path.basename(folder_path)
            self.projects[project_name] = folder_path
            self.save_projects()
            self.update_project_list()
    
    def open_project(self):
        """打开选中的项目"""
        selection = self.project_listbox.curselection()
        if not selection:
            messagebox.showwarning("提示", "请先选择一个项目")
            return
            
        project_name = self.project_listbox.get(selection[0])
        folder_path = self.projects.get(project_name)
        
        if folder_path and os.path.exists(folder_path):
            # 根据操作系统打开文件夹
            if os.name == 'nt':  # Windows
                os.startfile(folder_path)
            else:  # macOS 和 Linux
                os.system(f'open "{folder_path}"')
        else:
            messagebox.showerror("错误", "项目文件夹不存在")
            
    def delete_project(self):
        """删除选中的项目"""
        selection = self.project_listbox.curselection()
        if not selection:
            messagebox.showwarning("提示", "请先选择一个项目")
            return
            
        project_name = self.project_listbox.get(selection[0])
        if messagebox.askyesno("确认", f"确定要删除项目 {project_name} 吗？"):
            del self.projects[project_name]
            self.save_projects()
            self.update_project_list()
    
    def update_project_list(self):
        """更新项目列表显示"""
        self.project_listbox.delete(0, tk.END)
        for project_name in sorted(self.projects.keys()):
            self.project_listbox.insert(tk.END, project_name)
    
    def load_projects(self):
        """从配置文件加载项目数据"""
        config_path = Path.home() / '.project_manager.json'
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def save_projects(self):
        """保存项目数据到配置文件"""
        config_path = Path.home() / '.project_manager.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.projects, f, ensure_ascii=False, indent=2)
    
    def run(self):
        """运行应用"""
        self.root.mainloop()

if __name__ == '__main__':
    app = ProjectManager()
    app.run()