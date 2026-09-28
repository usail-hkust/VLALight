"""
图片保存工具类
用于VLM训练数据收集，按照路口和方向保存摄像头图片
"""
import os
import json
import cv2
from datetime import datetime
from typing import Dict, Optional, List

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
    import numpy as np
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

def crop_and_resize(image, left_crop: float = 0.40, right_crop: float = 0.30, scale_mode: str = 'fit_width'):
    """
    对图像进行非对称裁剪并等比缩放
    
    Args:
        image: BGR 格式的图片（OpenCV）
        left_crop: 左侧裁剪比例（0.40 = 裁掉左侧40%）
        right_crop: 右侧裁剪比例（0.30 = 裁掉右侧30%）
        scale_mode: 缩放模式 ('fit_width', 'fit_height', 'none')
    
    Returns:
        裁剪并缩放后的图片
    """
    h, w = image.shape[:2]
    
    # Implementation note.
    left_px = int(w * left_crop)
    right_px = int(w * (1 - right_crop))
    
    if right_px <= left_px:
        return image  # Implementation note.
    
    cropped = image[:, left_px:right_px]
    crop_h, crop_w = cropped.shape[:2]
    
    # Implementation note.
    if scale_mode == 'fit_width':
        # Implementation note.
        scale_factor = w / crop_w
        new_h = int(crop_h * scale_factor)
        resized = cv2.resize(cropped, (w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return resized
    elif scale_mode == 'fit_height':
        # Implementation note.
        return cropped
    else:  # 'none'
        return cropped

class ImageSaver:
    """图片保存器，管理VLM训练图片的保存"""
    
    def __init__(self, 
                 scenario: str,
                 direction_mapping: Dict[str, Dict[str, str]] = None,
                 direction_mapping_path: str = None,
                 base_dir: str = None,
                 session_id: str = None,
                 batch_size: int = None,
                 verbose: bool = True,
                 enable_preprocess: bool = True,
                 left_crop: float = 0.40,
                 right_crop: float = 0.30,
                 scale_mode: str = 'fit_width',
                 strict: bool = False,
                 output_width: Optional[int] = None,
                 output_height: Optional[int] = None):
        """
        初始化图片保存器
        
        Args:
            scenario: 场景名称 ('jinan', 'hangzhou', 'newyork')
            direction_mapping: 方向映射字典 {tls_id: {cam_idx: direction}}
            direction_mapping_path: 方向映射JSON文件路径（与direction_mapping二选一）
            base_dir: 基础目录，默认为 TransSimHub/examples/{Scenario}
            session_id: 会话ID，默认为当前时间
            batch_size: 流式保存时每批保存的路口数量，None表示一次性保存所有路口
            verbose: 是否打印初始化信息
            enable_preprocess: 是否启用图像预处理（裁剪+缩放）
            left_crop: 左侧裁剪比例
            right_crop: 右侧裁剪比例
            scale_mode: 缩放模式
            output_width: 保存图片的目标宽度；None 表示保持原有输出尺寸
            output_height: 保存图片的目标高度；None 表示保持原有输出尺寸
        """
        self.scenario = scenario
        self.enable_preprocess = enable_preprocess
        self.left_crop = left_crop
        self.right_crop = right_crop
        self.scale_mode = scale_mode
        self.strict = strict
        if (output_width is None) != (output_height is None):
            raise ValueError("output_width and output_height must be provided together")
        if output_width is not None and (int(output_width) <= 0 or int(output_height) <= 0):
            raise ValueError("output_width and output_height must be positive")
        self.output_width = int(output_width) if output_width is not None else None
        self.output_height = int(output_height) if output_height is not None else None
        if direction_mapping is not None:
            self.direction_mapping = direction_mapping
        elif direction_mapping_path is not None:
            self.direction_mapping = self._load_direction_mapping(direction_mapping_path)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.normpath(os.path.join(script_dir, '..'))
            default_path = os.path.join(project_root, 'output', f'direction_mapping_{scenario}.json')
            self.direction_mapping = self._load_direction_mapping(default_path)
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
            os.makedirs(self.session_dir, exist_ok=True)
        else:
            self.session_dir = os.path.join(self.base_dir, self.session_id)
            os.makedirs(self.session_dir, exist_ok=True)
        self.batch_size = batch_size
        self.tls_ids_list = list(self.direction_mapping.keys())
        self.current_batch_index = 0
        self.saved_count = 0
        self.step_count = 0
        if verbose:
            print(f"[ImageSaver] 初始化完成")
            print(f"  场景: {scenario}")
            print(f"  保存目录: {self.session_dir}")
            print(f"  路口数量: {len(self.direction_mapping)}")
            if batch_size:
                total_batches = (len(self.tls_ids_list) + batch_size - 1) // batch_size
                print(f"  流式保存: 每批 {batch_size} 个路口，共 {total_batches} 批")
    
    def _load_direction_mapping(self, path: str) -> Dict[str, Dict[str, str]]:
        """加载方向映射文件"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"方向映射文件不存在: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _resize_for_save(self, image):
        """按可选的保存尺寸缩放图片，默认保持原有尺寸逻辑。"""
        if self.output_width is None:
            return image
        source_h, source_w = image.shape[:2]
        if source_w == self.output_width and source_h == self.output_height:
            return image
        interpolation = (
            cv2.INTER_AREA
            if self.output_width <= source_w and self.output_height <= source_h
            else cv2.INTER_LINEAR
        )
        return cv2.resize(
            image,
            (self.output_width, self.output_height),
            interpolation=interpolation,
        )
    
    def get_direction(self, tls_id: str, cam_idx: int) -> str:
        """
        获取摄像头对应的方向
        
        Args:
            tls_id: 路口ID
            cam_idx: 摄像头索引 (0-3)
        
        Returns:
            方向字符串 ('N', 'S', 'E', 'W') 或索引字符串
        """
        if tls_id in self.direction_mapping:
            return self.direction_mapping[tls_id].get(str(cam_idx), str(cam_idx))
        return str(cam_idx)
    
    def save_step_images(self, 
                         step: int, 
                         sensor_data: Dict, 
                         tls_ids: List[str] = None,
                         num_cameras: int = 4) -> Dict[str, List[str]]:
        """
        保存一个step的所有路口图片（串行保存）
        
        Args:
            step: 当前步数
            sensor_data: 传感器数据字典，格式为 {tls_id_cam_idx: {'junction_front_all': image}}
            tls_ids: 要保存的路口ID列表，默认为direction_mapping中的所有路口
            num_cameras: 每个路口的摄像头数量，默认4
        
        Returns:
            保存的图片路径字典 {tls_id: [path1, path2, ...]}
        """
        images_dir = os.path.join(self.session_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        step_dir = os.path.join(images_dir, f"step_{step:04d}")
        os.makedirs(step_dir, exist_ok=True)
        if tls_ids is None:
            tls_ids = list(self.direction_mapping.keys())
        saved_paths = {}
        for tls_id in tls_ids:
            tls_dir = os.path.join(step_dir, tls_id)
            os.makedirs(tls_dir, exist_ok=True)
            saved_paths[tls_id] = []
            for cam_idx in range(num_cameras):
                sensor_key = f"{tls_id}_{cam_idx}"
                if sensor_key not in sensor_data:
                    if self.strict:
                        raise KeyError(f"missing sensor key: {sensor_key}")
                    continue
                image_data = sensor_data[sensor_key]
                if isinstance(image_data, dict):
                    image = image_data.get('junction_front_all')
                else:
                    image = image_data
                if image is None:
                    if self.strict:
                        raise ValueError(f"empty image for sensor key: {sensor_key}")
                    continue
                direction = self.get_direction(tls_id, cam_idx)
                image_path = os.path.join(tls_dir, f"{direction}.jpg")
                try:
                    import numpy as np
                    
                    # Implementation note.
                    if image.dtype != np.uint8:
                        if image.dtype in [np.float32, np.float64]:
                            if image.max() <= 1.0:
                                image = (image * 255).astype(np.uint8)
                            else:
                                image = image.astype(np.uint8)
                        else:
                            image = image.astype(np.uint8)
                    
                    bgr_image = convert_rgb_to_bgr(image)
                    
                    # Implementation note.
                    if self.enable_preprocess:
                        bgr_image = crop_and_resize(bgr_image, self.left_crop, self.right_crop, self.scale_mode)
                    
                    labeled_image = add_direction_label(bgr_image, direction)
                    labeled_image = self._resize_for_save(labeled_image)
                    
                    # Implementation note.
                    success, encoded_image = cv2.imencode('.jpg', labeled_image)
                    if success:
                        with open(image_path, 'wb') as f:
                            f.write(encoded_image.tobytes())
                        saved_paths[tls_id].append(image_path)
                        self.saved_count += 1
                    elif self.strict:
                        raise RuntimeError(f"cv2.imencode failed for {image_path}")
                except Exception as e:
                    if self.strict:
                        raise RuntimeError(f"failed to save image {image_path}: {e}") from e
                    pass  # Implementation note.
        self.step_count += 1
        return saved_paths
    
    def save_single_junction(self,
                             step: int,
                             tls_id: str,
                             sensor_data: Dict,
                             num_cameras: int = 4) -> List[str]:
        """
        保存单个路口的图片
        
        Args:
            step: 当前步数
            tls_id: 路口ID
            sensor_data: 传感器数据字典
            num_cameras: 摄像头数量
        
        Returns:
            保存的图片路径列表
        """
        images_dir = os.path.join(self.session_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        step_dir = os.path.join(images_dir, f"step_{step:04d}")
        tls_dir = os.path.join(step_dir, tls_id)
        os.makedirs(tls_dir, exist_ok=True)
        saved_paths = []
        for cam_idx in range(num_cameras):
            sensor_key = f"{tls_id}_{cam_idx}"
            if sensor_key not in sensor_data:
                continue
            image_data = sensor_data[sensor_key]
            if isinstance(image_data, dict):
                image = image_data.get('junction_front_all')
            else:
                image = image_data
            if image is None:
                continue
            direction = self.get_direction(tls_id, cam_idx)
            image_path = os.path.join(tls_dir, f"{direction}.jpg")
            bgr_image = convert_rgb_to_bgr(image)
            
            # Implementation note.
            if self.enable_preprocess:
                bgr_image = crop_and_resize(bgr_image, self.left_crop, self.right_crop, self.scale_mode)
            
            labeled_image = add_direction_label(bgr_image, direction)
            labeled_image = self._resize_for_save(labeled_image)
            cv2.imwrite(image_path, labeled_image)
            saved_paths.append(image_path)
            self.saved_count += 1
        return saved_paths
    
    def save_step_images_streaming(self,
                                    step: int,
                                    sensor_data: Dict,
                                    num_cameras: int = 4) -> Dict[str, List[str]]:
        """
        流式保存一个step的部分路口图片（每次只保存一批路口）
        
        Args:
            step: 当前步数
            sensor_data: 传感器数据字典
            num_cameras: 每个路口的摄像头数量
        
        Returns:
            保存的图片路径字典 {tls_id: [path1, path2, ...]}
        """
        if self.batch_size is None:
            return self.save_step_images(step, sensor_data, num_cameras=num_cameras)
        start_idx = self.current_batch_index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.tls_ids_list))
        batch_tls_ids = self.tls_ids_list[start_idx:end_idx]
        total_batches = (len(self.tls_ids_list) + self.batch_size - 1) // self.batch_size
        self.current_batch_index = (self.current_batch_index + 1) % total_batches
        return self.save_step_images(step, sensor_data, tls_ids=batch_tls_ids, num_cameras=num_cameras)
    
    def get_current_batch_info(self) -> Dict:
        """获取当前批次信息"""
        if self.batch_size is None:
            return {'mode': 'all', 'batch_size': len(self.tls_ids_list)}
        total_batches = (len(self.tls_ids_list) + self.batch_size - 1) // self.batch_size
        return {
            'mode': 'streaming',
            'batch_size': self.batch_size,
            'current_batch': self.current_batch_index,
            'total_batches': total_batches
        }
    
    def get_image_path(self, step: int, tls_id: str, direction: str) -> str:
        """
        获取指定图片的路径（用于后续读取）
        
        Args:
            step: 步数
            tls_id: 路口ID
            direction: 方向 ('N', 'S', 'E', 'W')
        
        Returns:
            图片路径
        """
        return os.path.join(self.session_dir, f"step_{step:04d}", tls_id, f"{direction}.jpg")
    
    def get_stats(self) -> Dict:
        """获取保存统计信息"""
        return {
            'session_dir': self.session_dir,
            'step_count': self.step_count,
            'saved_count': self.saved_count,
            'junction_count': len(self.direction_mapping)
        }
    
    def __repr__(self):
        return f"ImageSaver(scenario={self.scenario}, session={self.session_id}, saved={self.saved_count})"

if __name__ == '__main__':
    import numpy as np
    print("=" * 50)
    print("测试1: 普通保存模式")
    print("=" * 50)
    saver1 = ImageSaver(scenario='jinan')
    fake_sensor_data = {}
    for tls_id in list(saver1.direction_mapping.keys())[:4]:
        for cam_idx in range(4):
            fake_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            fake_sensor_data[f"{tls_id}_{cam_idx}"] = {
                'junction_front_all': fake_image
            }
    saved = saver1.save_step_images(step=0, sensor_data=fake_sensor_data)
    print(f"保存了 {sum(len(v) for v in saved.values())} 张图片")
    print("\n" + "=" * 50)
    print("测试2: 流式保存模式 (batch_size=2)")
    print("=" * 50)
    saver2 = ImageSaver(scenario='jinan', batch_size=2)
    for step in range(4):
        saved = saver2.save_step_images_streaming(step=step, sensor_data=fake_sensor_data)
        batch_info = saver2.get_current_batch_info()
        tls_saved = list(saved.keys())
        print(f"Step {step}: 保存路口 {tls_saved}, 下一批次: {batch_info['current_batch']}/{batch_info['total_batches']}")
    print(f"\n统计: {saver2.get_stats()}")
