#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time          : 2025/01/02 12:32
# @Author        : jcfszxc
# @Email         : jcfszxc.ai@gmail.com
# @File          : video_processor.py
# @Description   : 

import os
import cv2
from concurrent.futures import ThreadPoolExecutor
from .utils import generate_random_string
from PyQt6.QtCore import QObject, pyqtSignal

class VideoProcessor(QObject):
    progress_updated = pyqtSignal(str, int, int)  # video_path, current, total
    processing_finished = pyqtSignal(int, int)    # total_saved, total_errors
    
    def __init__(self):
        super().__init__()
        self._should_stop = False
    
    def stop_processing(self):
        self._should_stop = True
    
    def process_video(self, video_path, output_dir):
        """Process a single video file"""
        if self._should_stop:
            return 0, 0
            
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(2, int(fps / 2))
        
        frame_count = 0
        saved_count = 0
        error_count = 0
        
        while not self._should_stop:
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    random_suffix = generate_random_string()
                    output_filename = os.path.join(
                        output_dir, 
                        f"frame_{saved_count:04d}_{random_suffix}.jpg"
                    )
                    cv2.imwrite(output_filename, frame)
                    saved_count += 1
                    
                    # Emit progress signal
                    self.progress_updated.emit(
                        os.path.basename(video_path),
                        frame_count,
                        total_frames
                    )
                
                frame_count += 1
            except cv2.error as e:
                error_count += 1
                print(f"Error processing frame {frame_count} in {video_path}: {str(e)}")
                cap.grab()
        
        cap.release()
        return saved_count, error_count
    
    def process_directory(self, input_dir, output_base_dir):
        """Process all videos in the directory"""
        self._should_stop = False
        total_saved = 0
        total_errors = 0
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.endswith(('.mp4', '.avi', '.h264')):
                        video_path = os.path.join(root, file)
                        relative_path = os.path.relpath(root, input_dir)
                        output_dir = os.path.join(
                            output_base_dir, 
                            relative_path, 
                            os.path.splitext(file)[0]
                        )
                        futures.append(
                            executor.submit(self.process_video, video_path, output_dir)
                        )
            
            for future in futures:
                if not self._should_stop:
                    saved, errors = future.result()
                    total_saved += saved
                    total_errors += errors
        
        self.processing_finished.emit(total_saved, total_errors)
        return total_saved, total_errors