#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2024/12/28 21:51
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : database.py
# @Description   : 

import sqlite3
from typing import List, Optional
from contextlib import contextmanager
from src.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

class Database:
    """数据库管理类"""
    
    def __init__(self):
        self.db_path = Config.DATABASE_PATH
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()
    
    def init_database(self):
        """初始化数据库表结构"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    due_date TEXT,
                    completed_at TEXT
                )
            ''')
            conn.commit()
            logger.info("Database initialized successfully")