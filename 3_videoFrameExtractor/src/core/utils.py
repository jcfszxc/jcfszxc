#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/01/02 12:31
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : utils.py
# @Description   : 

import random
import string

def generate_random_string(length=8):
    """Generate a random string of fixed length"""
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(length))