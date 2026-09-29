"""
视频录制工具类
用于VLM视频推理，支持多路口、多方向的视频录制
基于ImageSaver扩展，将连续帧保存为视频文件
"""
import os
import cv2
import json
import numpy as np
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from collections import defaultdict


def convert_rgb_to_bgr(image):
    """将 RGB 图像转换为 BGR 格式（OpenCV 使用）"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def add_direction_label(image, direction: str):
    """
    在图片上叠加方向标签
    
    Args:
        image: BGR 格式的图片（OpenCV）
        direction: 方向字符串 ('N', 'S', 'E', 'W')
    
    Returns:
        添加了标签的图片
    """
    labeled_image = image.copy()
    height, width = labeled_image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 1
    
    direction_full = {
        'N': 'NORTH',
        'S': 'SOUTH',
        'E': 'EAST',
        'W': 'WEST'
    }
    text = direction_full.get(direction, direction)
    
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = 12
    y = text_height + 12
    
    overlay = labeled_image.copy()
    cv2.rectangle(overlay, 
                  (x - 4, y - text_height - 4),
                  (x + text_width + 4, y + baseline + 4),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, labeled_image, 0.4, 0, labeled_image)
    cv2.putText(labeled_image, text, (x, y), font, font_scale, (0, 255, 255), thickness)
    
    return labeled_image


class VideoRecorder:
    """
    视频录制器，用于VLM视频推理
    支持多路口、多方向同时录制
    """
    
    def __init__(self,
                 scenario: str,
                 direction_mapping: Dict[str, Dict[str, str]] = None,
                 direction_mapping_path: str = None,
                 base_dir: str = None,
                 session_id: str = None,
                 fps: int = 10,
                 codec: str = 'mp4v',
                 resolution: Tuple[int, int] = None,
                 add_labels: bool = True,
                 verbose: bool = True):
        """
        初始化视频录制器
        
        Args:
            scenario: 场景名称 ('jinan', 'hangzhou', 'newyork', 'test_2x2')
            direction_mapping: 方向映射字典 {tls_id: {cam_idx: direction}}
            direction_mapping_path: 方向映射JSON文件路径
            base_dir: 基础目录
            session_id: 会话ID，默认为当前时间
            fps: 视频帧率，默认10fps
            codec: 视频编码器，默认'mp4v'
            resolution: 视频分辨率 (width, height)，None表示自动检测
            add_labels: 是否在视频中添加方向标签
            verbose: 是否打印详细信息
        """
        self.scenario = scenario
        
        # Implementation note.
        if direction_mapping is not None:
            self.direction_mapping = direction_mapping
        elif direction_mapping_path is not None:
            self.direction_mapping = self._load_direction_mapping(direction_mapping_path)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.normpath(os.path.join(script_dir, '..'))
            default_path = os.path.join(project_root, 'output', f'direction_mapping_{scenario}.json')
            self.direction_mapping = self._load_direction_mapping(default_path)
        
        # Implementation note.
        if base_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.normpath(os.path.join(script_dir, '..'))
            scenario_name = scenario.capitalize()
            base_dir = os.path.join(project_root, 'TransSimHub', 'examples', scenario_name)
        self.base_dir = base_dir
        
        if session_id is None:
            session_id = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        self.session_id = session_id
        
        if self.session_id == "":
            self.session_dir = self.base_dir
        else:
            self.session_dir = os.path.join(self.base_dir, self.session_id)
        
        # Implementation note.
        self.images_dir = os.path.join(self.session_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Implementation note.
        self.fps = fps
        self.codec = codec
        self.resolution = resolution
        self.add_labels = add_labels
        self.verbose = verbose
        
        # Implementation note.
        self.video_writers: Dict[Tuple[str, str], cv2.VideoWriter] = {}
        
        # Implementation note.
        self.frame_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Implementation note.
        self.total_frames = 0
        self.active_videos = 0
        
        if verbose:
            print(f"[VideoRecorder] 初始化完成")
            print(f"  场景: {scenario}")
            print(f"  保存目录: {self.images_dir}")
            print(f"  路口数量: {len(self.direction_mapping)}")
            print(f"  帧率: {fps} fps")
            print(f"  编码器: {codec}")
    
    def _load_direction_mapping(self, path: str) -> Dict[str, Dict[str, str]]:
        """加载方向映射文件"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"方向映射文件不存在: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_direction(self, tls_id: str, cam_idx: int) -> str:
        """获取摄像头对应的方向"""
        if tls_id in self.direction_mapping:
            return self.direction_mapping[tls_id].get(str(cam_idx), str(cam_idx))
        return str(cam_idx)
    
    def _get_video_writer(self, 
                          tls_id: str, 
                          direction: str, 
                          image_shape: Tuple[int, int, int],
                          step_num: int = None) -> cv2.VideoWriter:
        """
        获取或创建VideoWriter
        
        Args:
            tls_id: 路口ID
            direction: 方向
            image_shape: 图像形状 (height, width, channels)
            step_num: 步数（用于确定目录）
        
        Returns:
            cv2.VideoWriter对象
        """
        key = (tls_id, direction)
        
        if key in self.video_writers:
            return self.video_writers[key]
        
        # Implementation note.
        height, width = image_shape[:2]
        if self.resolution is not None:
            width, height = self.resolution
        
        # Implementation note.
        step_dir = os.path.join(self.images_dir, f"step_{step_num:04d}" if step_num is not None else "videos")
        tls_dir = os.path.join(step_dir, tls_id)
        os.makedirs(tls_dir, exist_ok=True)
        
        video_filename = f"{direction}.mp4"
        video_path = os.path.join(tls_dir, video_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
        
        if not writer.isOpened():
            raise RuntimeError(f"无法创建视频文件: {video_path}")
        
        self.video_writers[key] = writer
        self.active_videos += 1
        
        if self.verbose:
            print(f"[VideoRecorder] 创建视频: {video_filename} ({width}x{height} @ {self.fps}fps)")
        
        return writer
    
    def add_frame(self, 
                  tls_id: str, 
                  cam_idx: int, 
                  image: np.ndarray,
                  step_num: int = None) -> bool:
        """
        添加一帧到对应的视频
        
        Args:
            tls_id: 路口ID
            cam_idx: 摄像头索引
            image: 图像数据 (RGB格式)
        
        Returns:
            是否成功添加
        """
        try:
            direction = self.get_direction(tls_id, cam_idx)
            
            # Implementation note.
            if image.dtype != np.uint8:
                if image.dtype in [np.float32, np.float64]:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Implementation note.
            bgr_image = convert_rgb_to_bgr(image)
            
            # Implementation note.
            if self.add_labels:
                bgr_image = add_direction_label(bgr_image, direction)
            
            # Implementation note.
            if self.resolution is not None:
                bgr_image = cv2.resize(bgr_image, self.resolution)
            
            # Implementation note.
            writer = self._get_video_writer(tls_id, direction, bgr_image.shape, step_num)
            
            # Implementation note.
            writer.write(bgr_image)
            
            # Implementation note.
            key = (tls_id, direction)
            self.frame_counts[key] += 1
            self.total_frames += 1
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"[VideoRecorder] 添加帧失败: {tls_id}_{cam_idx}, 错误: {e}")
            return False
    
    def add_step_frames(self, 
                        step: int,
                        sensor_data: Dict,
                        tls_ids: List[str] = None,
                        num_cameras: int = 4) -> int:
        """
        添加一个step的所有帧
        
        Args:
            sensor_data: 传感器数据字典 {tls_id_cam_idx: {'junction_front_all': image}}
            tls_ids: 要录制的路口ID列表，默认为所有路口
            num_cameras: 每个路口的摄像头数量
        
        Returns:
            成功添加的帧数
        """
        if tls_ids is None:
            tls_ids = list(self.direction_mapping.keys())
        
        success_count = 0
        
        for tls_id in tls_ids:
            for cam_idx in range(num_cameras):
                sensor_key = f"{tls_id}_{cam_idx}"
                
                if sensor_key not in sensor_data:
                    continue
                
                image_data = sensor_data[sensor_key]
                
                # Implementation note.
                if isinstance(image_data, dict):
                    image = image_data.get('junction_front_all')
                else:
                    image = image_data
                
                if image is None:
                    continue
                
                # Implementation note.
                if self.add_frame(tls_id, cam_idx, image, step):
                    success_count += 1
        
        return success_count
    
    def finalize(self) -> Dict[str, str]:
        """
        完成录制，释放所有VideoWriter
        
        Returns:
            视频文件路径字典 {(tls_id, direction): video_path}
        """
        video_paths = {}
        
        for (tls_id, direction), writer in self.video_writers.items():
            writer.release()
            
            # Implementation note.
            # Implementation note.
            video_filename = f"{direction}.mp4"
            video_paths[f"{tls_id}_{direction}"] = f"videos stored in step directories under {self.images_dir}"
            
            frame_count = self.frame_counts[(tls_id, direction)]
            duration = frame_count / self.fps
            
            if self.verbose:
                print(f"[VideoRecorder] 完成: {video_filename} ({frame_count} 帧, {duration:.1f}秒)")
        
        self.video_writers.clear()
        
        if self.verbose:
            print(f"[VideoRecorder] 录制完成，共 {len(video_paths)} 个视频文件")
            print(f"  总帧数: {self.total_frames}")
            print(f"  保存目录: {self.images_dir}")
        
        return video_paths
    
    def get_stats(self) -> Dict:
        """获取录制统计信息"""
        return {
            'session_dir': self.session_dir,
            'images_dir': self.images_dir,  # Implementation note.
            'total_frames': self.total_frames,
            'active_videos': self.active_videos,
            'frame_counts': dict(self.frame_counts),
            'fps': self.fps,
            'codec': self.codec
        }
    
    def __repr__(self):
        return f"VideoRecorder(scenario={self.scenario}, videos={self.active_videos}, frames={self.total_frames})"
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出，自动完成录制"""
        self.finalize()


if __name__ == '__main__':
    import numpy as np
    
    print("=" * 60)
    print("测试: VideoRecorder 视频录制")
    print("=" * 60)
    
    # Implementation note.
    test_mapping = {
        "intersection_0_0": {"0": "N", "1": "E", "2": "W", "3": "S"},
        "intersection_0_1": {"0": "N", "1": "E", "2": "W", "3": "S"}
    }
    
    # Implementation note.
    with VideoRecorder(
        scenario='test_2x2',
        direction_mapping=test_mapping,
        session_id='test_video',
        fps=10,
        verbose=True
    ) as recorder:
        
        # Implementation note.
        for step in range(30):
            fake_sensor_data = {}
            
            # Implementation note.
            for tls_id in test_mapping.keys():
                for cam_idx in range(4):
                    # Implementation note.
                    fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
                    fake_image[:, :, 0] = (step * 8) % 256  # Implementation note.
                    fake_image[:, :, 1] = 128
                    fake_image[:, :, 2] = 255 - (step * 8) % 256  # Implementation note.
                    
                    fake_sensor_data[f"{tls_id}_{cam_idx}"] = {
                        'junction_front_all': fake_image
                    }
            
            # Implementation note.
            added = recorder.add_step_frames(step, fake_sensor_data)
            
            if step % 10 == 0:
                print(f"  Step {step}: 添加了 {added} 帧")
        
        print("\n录制统计:")
        stats = recorder.get_stats()
        for key, value in stats.items():
            if key != 'frame_counts':
                print(f"  {key}: {value}")
    
    print("\n✅ 测试完成！请查看生成的视频文件。")
