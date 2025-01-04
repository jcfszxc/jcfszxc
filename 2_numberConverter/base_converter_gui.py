import tkinter as tk
from tkinter import ttk, messagebox
import re

class NumberConverter:
    """整合的数值转换器，支持进制转换和编码转换"""
    
    @staticmethod
    def to_decimal(value: str, base: int, is_complement: bool = False, bits: int = 16) -> int:
        """将任意进制数转换为十进制，支持补码解析和负数处理"""
        if not value:
            return 0
            
        try:
            if not is_complement:
                # 处理带符号的输入
                if value.startswith('-'):
                    return -int(value[1:], base)
                return int(value, base)
            else:
                # 补码转十进制
                if len(value) < bits:  # 自动补齐位数
                    value = '0' * (bits - len(value)) + value
                if value[0] == '1':  # 负数
                    return int(value, 2) - (1 << bits)
                return int(value, 2)  # 正数
        except ValueError:
            raise ValueError(f"无效的{base}进制数：{value}")

    @staticmethod
    def to_binary(decimal: int, bits: int = 16, get_complement: bool = False) -> str:
        """将十进制数转换为指定位数的二进制，可选补码表示"""
        if get_complement and decimal < 0:
            # 负数的补码表示
            return format((1 << bits) + decimal, f'0{bits}b')
        else:
            # 正数或原码表示
            return format(decimal & ((1 << bits) - 1), f'0{bits}b')

    @staticmethod
    def get_original_code(decimal: int, bits: int = 16) -> str:
        """获取原码表示"""
        if decimal >= 0:
            return '0' + format(decimal, f'0{bits-1}b')[-bits+1:]
        else:
            return '1' + format(abs(decimal), f'0{bits-1}b')[-bits+1:]

    @staticmethod
    def get_inverse_code(decimal: int, bits: int = 16) -> str:
        """获取反码表示"""
        if decimal >= 0:
            return NumberConverter.get_original_code(decimal, bits)
        else:
            original = NumberConverter.get_original_code(decimal, bits)
            return original[0] + ''.join('1' if b == '0' else '0' for b in original[1:])

class ConverterGUI:
    """集成的进制与编码转换器界面"""
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("进制与编码转换器")
        self.window.geometry("800x600")
        self.window.resizable(True, True)
        
        # 设置样式
        style = ttk.Style()
        style.configure('TLabel', font=('Arial', 10))
        style.configure('TEntry', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 10))
        
        self._init_ui()
        
    def _init_ui(self):
        """初始化用户界面"""
        main_frame = ttk.Frame(self.window, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 输入区域
        input_frame = ttk.LabelFrame(main_frame, text="输入", padding="10")
        input_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # 数值输入
        ttk.Label(input_frame, text="数值:").grid(row=0, column=0, padx=5, pady=5)
        self.input_value = ttk.Entry(input_frame, width=40)
        self.input_value.grid(row=0, column=1, columnspan=2, padx=5, pady=5)
        
        # 输入进制选择
        ttk.Label(input_frame, text="输入格式:").grid(row=1, column=0, padx=5, pady=5)
        self.input_format = ttk.Combobox(input_frame, width=37,
                                       values=["二进制原码", "二进制反码", "二进制补码",
                                              "八进制", "十进制", "十六进制"])
        self.input_format.grid(row=1, column=1, columnspan=2, padx=5, pady=5)
        self.input_format.set("十进制")
        
        # 位数选择
        ttk.Label(input_frame, text="位数:").grid(row=2, column=0, padx=5, pady=5)
        self.bits = ttk.Combobox(input_frame, width=10, values=["8", "16", "32"])
        self.bits.grid(row=2, column=1, padx=5, pady=5)
        self.bits.set("16")
        
        # 转换按钮
        convert_button = ttk.Button(main_frame, text="转换", command=self._convert)
        convert_button.grid(row=1, column=0, columnspan=2, pady=20)
        
        # 结果区域
        result_frame = ttk.LabelFrame(main_frame, text="结果", padding="10")
        result_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # 各种结果显示
        results = [
            ("二进制原码:", "original"),
            ("二进制反码:", "inverse"),
            ("二进制补码:", "complement"),
            ("八进制:", "octal"),
            ("十进制:", "decimal"),
            ("十六进制:", "hex")
        ]
        
        self.results = {}
        for i, (label, name) in enumerate(results):
            ttk.Label(result_frame, text=label).grid(row=i, column=0, padx=5, pady=5)
            result = ttk.Entry(result_frame, width=50, state='readonly')
            result.grid(row=i, column=1, padx=5, pady=5)
            self.results[name] = result
            
        # 复制按钮
        for i, (_, name) in enumerate(results):
            copy_btn = ttk.Button(result_frame, text="复制",
                                command=lambda n=name: self._copy_result(n))
            copy_btn.grid(row=i, column=2, padx=5, pady=5)
        
        # 清除按钮
        clear_button = ttk.Button(main_frame, text="清除", command=self._clear)
        clear_button.grid(row=3, column=0, columnspan=2, pady=10)
        
        # 绑定回车键
        self.window.bind('<Return>', lambda e: self._convert())
        
    def _get_base_from_format(self, format_str: str) -> tuple:
        """从格式字符串中获取进制和编码类型"""
        format_mapping = {
            "二进制原码": (2, "original"),
            "二进制反码": (2, "inverse"),
            "二进制补码": (2, "complement"),
            "八进制": (8, None),
            "十进制": (10, None),
            "十六进制": (16, None)
        }
        return format_mapping.get(format_str, (10, None))
        
    def _convert(self):
        """执行转换"""
        try:
            # 获取输入值
            input_value = self.input_value.get().strip()
            if not input_value:
                messagebox.showwarning("警告", "请输入要转换的数值！")
                return
                
            # 获取位数和格式
            bits = int(self.bits.get())
            base, code_type = self._get_base_from_format(self.input_format.get())
            
            # 转换为十进制
            if code_type:  # 如果是编码格式
                is_complement = (code_type == "complement")
                decimal = NumberConverter.to_decimal(input_value, 2, is_complement, bits)
                if code_type == "inverse":  # 反码需要特殊处理
                    if input_value[0] == '1':  # 负数
                        decimal = -(int(''.join('1' if b == '0' else '0' 
                                   for b in input_value[1:]), 2) + 1)
            else:
                # 处理十六进制负数的特殊情况
                if base == 16 and len(input_value) == bits//4 and int(input_value[0], 16) >= 8:
                    # 如果最高位大于等于8，说明是负数的十六进制表示
                    decimal = NumberConverter.to_decimal('1' + '0'*(bits-1), 2, True, bits) + \
                            int(input_value, base)
                else:
                    decimal = NumberConverter.to_decimal(input_value, base)
            
            # 检查范围
            max_value = (1 << (bits - 1)) - 1
            min_value = -(1 << (bits - 1))
            if not (min_value <= decimal <= max_value):
                messagebox.showerror("错误", 
                                   f"{bits}位有符号数的范围是 [{min_value}, {max_value}]")
                return
            
            # 更新各种结果
            converter = NumberConverter()
            results = {
                'original': converter.get_original_code(decimal, bits),
                'inverse': converter.get_inverse_code(decimal, bits),
                'complement': converter.to_binary(decimal, bits, True),
                'decimal': str(decimal),
                'octal': ('-' if decimal < 0 else '') + oct(abs(decimal))[2:],
                'hex': ('-' if decimal < 0 else '') + hex(abs(decimal))[2:].upper()
            }
            
            for name, value in results.items():
                entry = self.results[name]
                entry.configure(state='normal')
                entry.delete(0, tk.END)
                entry.insert(0, value)
                entry.configure(state='readonly')
                
        except ValueError as e:
            messagebox.showerror("错误", str(e))
        except Exception as e:
            messagebox.showerror("错误", f"转换过程中出现错误：{str(e)}")
            
    def _copy_result(self, result_name: str):
        """复制结果到剪贴板"""
        value = self.results[result_name].get()
        self.window.clipboard_clear()
        self.window.clipboard_append(value)
        
    def _clear(self):
        """清除所有输入和输出"""
        self.input_value.delete(0, tk.END)
        for result in self.results.values():
            result.configure(state='normal')
            result.delete(0, tk.END)
            result.configure(state='readonly')
            
    def run(self):
        """运行应用程序"""
        self.window.mainloop()

if __name__ == "__main__":
    app = ConverterGUI()
    app.run()