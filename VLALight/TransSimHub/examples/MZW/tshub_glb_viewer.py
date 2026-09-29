'''
使用TransSimHub的3D渲染系统查看GLB文件
直接利用项目的Panda3D引擎和窗口系统
'''
import os
import sys
import math
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tshub.tshub_env3d.vis3d_renderer._showbase_instance import _ShowBaseInstance
from tshub.tshub_env3d.vis3d_utils.colors import Colors
from panda3d.core import *

class TSHubGLBViewer:
    def __init__(self, glb_path, window_title="TSHub GLB Viewer"):
        self.glb_path = glb_path
        self.window_title = window_title
        self.showbase = None
        self.model_np = None
        self.camera_distance = 100
        self.camera_height = 50
        self.rotation_speed = 30  # 度/秒
        
    def initialize_renderer(self):
        """初始化TransSimHub的3D渲染器"""
        print("🎮 初始化TransSimHub 3D渲染器...")
        
        # 设置渲染模式为onscreen以显示窗口
        _ShowBaseInstance._render_mode = "onscreen"
        
        # 创建ShowBase实例（单例模式）
        self.showbase = _ShowBaseInstance()
        
        # 设置背景颜色
        self.showbase.setBackgroundColor(*Colors.LightBlue.value)
        
        # 设置基本光照
        self.setup_lighting()
        
        # 设置摄像机
        self.setup_camera()
        
        print("✅ 渲染器初始化完成")
        
    def setup_lighting(self):
        """设置光照"""
        # 环境光
        ambient_light = AmbientLight('ambient_light')
        ambient_light.setColor((0.4, 0.4, 0.4, 1))
        ambient_light_np = self.showbase.render.attachNewNode(ambient_light)
        self.showbase.render.setLight(ambient_light_np)
        
        # 方向光
        directional_light = DirectionalLight('directional_light')
        directional_light.setColor((0.8, 0.8, 0.8, 1))
        directional_light.setDirection((-1, -1, -1))
        directional_light_np = self.showbase.render.attachNewNode(directional_light)
        self.showbase.render.setLight(directional_light_np)
        
        # 点光源
        point_light = PointLight('point_light')
        point_light.setColor((0.6, 0.6, 0.6, 1))
        point_light_np = self.showbase.render.attachNewNode(point_light)
        point_light_np.setPos(50, 50, 100)
        self.showbase.render.setLight(point_light_np)
        
    def setup_camera(self):
        """设置摄像机"""
        # 设置摄像机初始位置
        self.showbase.camera.setPos(self.camera_distance, -self.camera_distance, self.camera_height)
        self.showbase.camera.lookAt(0, 0, 0)
        
        # 设置视野
        self.showbase.camLens.setFov(60)
        
    def load_glb_model(self):
        """加载GLB模型"""
        if not os.path.exists(self.glb_path):
            print(f"❌ 文件不存在: {self.glb_path}")
            return False
            
        print(f"📁 加载GLB模型: {self.glb_path}")
        
        try:
            # 使用Panda3D加载GLB模型
            self.model_np = self.showbase.loader.loadModel(self.glb_path)
            
            if self.model_np:
                # 将模型添加到场景
                self.model_np.reparentTo(self.showbase.render)
                
                # 获取模型边界
                bounds = self.model_np.getBounds()
                if bounds:
                    center = bounds.getCenter()
                    radius = bounds.getRadius()
                    
                    print(f"✅ 模型加载成功")
                    print(f"📊 模型中心: ({center.x:.2f}, {center.y:.2f}, {center.z:.2f})")
                    print(f"📊 模型半径: {radius:.2f}")
                    
                    # 根据模型大小调整摄像机
                    self.adjust_camera_for_model(center, radius)
                    
                    # 设置模型材质（可选）
                    self.setup_model_material()
                    
                    return True
                else:
                    print("⚠️ 无法获取模型边界")
                    return True
            else:
                print("❌ 模型加载失败")
                return False
                
        except Exception as e:
            print(f"❌ 加载异常: {e}")
            return False
    
    def adjust_camera_for_model(self, center, radius):
        """根据模型大小调整摄像机"""
        # 计算合适的摄像机距离
        self.camera_distance = radius * 3
        self.camera_height = radius * 1.5
        
        # 更新摄像机位置
        angle = 45 * (math.pi / 180)  # 45度角
        cam_x = center.x + self.camera_distance * math.cos(angle)
        cam_y = center.y + self.camera_distance * math.sin(angle)
        cam_z = center.z + self.camera_height
        
        self.showbase.camera.setPos(cam_x, cam_y, cam_z)
        self.showbase.camera.lookAt(center.x, center.y, center.z)
        
        print(f"📷 摄像机调整到: ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f})")
        
    def setup_model_material(self):
        """设置模型材质"""
        if self.model_np:
            # 启用自动法线
            self.model_np.setRenderModeWireframe()
            self.model_np.clearRenderMode()
            
            # 设置材质属性
            material = Material()
            material.setShininess(32.0)
            material.setAmbient((0.2, 0.2, 0.2, 1))
            material.setDiffuse((0.8, 0.8, 0.8, 1))
            material.setSpecular((1.0, 1.0, 1.0, 1))
            self.model_np.setMaterial(material)
            
    def setup_camera_controls(self):
        """Setup camera controls"""
        # Camera rotation task
        self.showbase.taskMgr.add(self.rotate_camera_task, "rotate_camera")
        
        # Keyboard controls
        self.showbase.accept("arrow_left", self.rotate_left)
        self.showbase.accept("arrow_right", self.rotate_right)
        self.showbase.accept("arrow_up", self.zoom_in)
        self.showbase.accept("arrow_down", self.zoom_out)
        self.showbase.accept("space", self.reset_camera)
        self.showbase.accept("w", self.toggle_wireframe)
        
        # Mouse controls
        self.showbase.accept("mouse1", self.start_mouse_look)
        self.showbase.accept("mouse1-up", self.stop_mouse_look)
        
        print("Controls:")
        print("  Left/Right Arrow : Rotate")
        print("  Up/Down Arrow    : Zoom")
        print("  Space            : Reset Camera")
        print("  W                : Toggle Wireframe")
        print("  Mouse Drag       : Free Rotation")
        print("  ESC              : Exit")
        
    def rotate_camera_task(self, task):
        """自动旋转摄像机任务"""
        if hasattr(self, 'auto_rotate') and self.auto_rotate:
            angle = task.time * self.rotation_speed * (math.pi / 180)
            
            if self.model_np:
                bounds = self.model_np.getBounds()
                if bounds:
                    center = bounds.getCenter()
                    cam_x = center.x + self.camera_distance * math.cos(angle)
                    cam_y = center.y + self.camera_distance * math.sin(angle)
                    cam_z = center.z + self.camera_height
                    
                    self.showbase.camera.setPos(cam_x, cam_y, cam_z)
                    self.showbase.camera.lookAt(center.x, center.y, center.z)
        
        return task.cont
    
    def rotate_left(self):
        """向左旋转"""
        self.manual_rotate(-10)
    
    def rotate_right(self):
        """向右旋转"""
        self.manual_rotate(10)
        
    def manual_rotate(self, angle_deg):
        """手动旋转摄像机"""
        if self.model_np:
            bounds = self.model_np.getBounds()
            if bounds:
                center = bounds.getCenter()
                current_pos = self.showbase.camera.getPos()
                
                # 计算当前角度
                dx = current_pos.x - center.x
                dy = current_pos.y - center.y
                current_angle = math.atan2(dy, dx)
                
                # 新角度
                new_angle = current_angle + angle_deg * (math.pi / 180)
                
                # 新位置
                cam_x = center.x + self.camera_distance * math.cos(new_angle)
                cam_y = center.y + self.camera_distance * math.sin(new_angle)
                cam_z = current_pos.z
                
                self.showbase.camera.setPos(cam_x, cam_y, cam_z)
                self.showbase.camera.lookAt(center.x, center.y, center.z)
    
    def zoom_in(self):
        """拉近"""
        self.camera_distance *= 0.9
        self.update_camera_distance()
    
    def zoom_out(self):
        """拉远"""
        self.camera_distance *= 1.1
        self.update_camera_distance()
        
    def update_camera_distance(self):
        """更新摄像机距离"""
        if self.model_np:
            bounds = self.model_np.getBounds()
            if bounds:
                center = bounds.getCenter()
                current_pos = self.showbase.camera.getPos()
                
                # 保持当前角度，只改变距离
                dx = current_pos.x - center.x
                dy = current_pos.y - center.y
                angle = math.atan2(dy, dx)
                
                cam_x = center.x + self.camera_distance * math.cos(angle)
                cam_y = center.y + self.camera_distance * math.sin(angle)
                cam_z = current_pos.z
                
                self.showbase.camera.setPos(cam_x, cam_y, cam_z)
    
    def reset_camera(self):
        """重置摄像机"""
        if self.model_np:
            bounds = self.model_np.getBounds()
            if bounds:
                center = bounds.getCenter()
                radius = bounds.getRadius()
                self.adjust_camera_for_model(center, radius)
                
    def toggle_wireframe(self):
        """Toggle wireframe mode"""
        if self.model_np:
            if hasattr(self, 'wireframe_mode') and self.wireframe_mode:
                self.model_np.clearRenderMode()
                self.wireframe_mode = False
                print("Switched to Solid Mode")
            else:
                self.model_np.setRenderModeWireframe()
                self.wireframe_mode = True
                print("Switched to Wireframe Mode")
    
    def start_mouse_look(self):
        """开始鼠标控制"""
        # 这里可以添加鼠标拖拽控制逻辑
        pass
    
    def stop_mouse_look(self):
        """停止鼠标控制"""
        pass
    
    def add_info_display(self):
        """Add information display"""
        # Create text display
        from direct.gui.OnscreenText import OnscreenText
        
        info_text = f"GLB File: {os.path.basename(self.glb_path)}"
        
        OnscreenText(
            text=info_text,
            pos=(-1.3, 0.9),
            scale=0.06,
            fg=(1, 1, 1, 1),
            bg=(0, 0, 0, 0.5),
            align=TextNode.ALeft
        )
        
        controls_text = """Controls:
Arrow Keys: Rotate/Zoom
Space: Reset  W: Wireframe
Mouse: Drag to Rotate"""
        
        OnscreenText(
            text=controls_text,
            pos=(-1.3, -0.7),
            scale=0.05,
            fg=(1, 1, 1, 1),
            bg=(0, 0, 0, 0.5),
            align=TextNode.ALeft
        )
    
    def run(self, auto_rotate=True):
        """Run the viewer"""
        print("Starting TSHub GLB Viewer...")
        
        # Initialize renderer
        self.initialize_renderer()
        
        # Load model
        if not self.load_glb_model():
            print("Failed to load model, exiting")
            return
        
        # Setup controls
        self.auto_rotate = auto_rotate
        self.setup_camera_controls()
        
        # Add info display
        self.add_info_display()
        
        print("GLB Viewer Started!")
        
        # Run main loop
        try:
            self.showbase.run()
        except KeyboardInterrupt:
            print("\nUser interrupted, exiting viewer")
        except Exception as e:
            print(f"Runtime error: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='TransSimHub GLB 3D Viewer')
    parser.add_argument('glb_file', nargs='?', help='Path to GLB file')
    parser.add_argument('--no-rotate', action='store_true', help='Disable auto rotation')
    
    args = parser.parse_args()
    
    # Find GLB file
    if not args.glb_file:
        candidates = [
            './3d_assets/buildings.glb',
            './3d_assets/map.glb',
            './buildings.glb'
        ]
        
        for candidate in candidates:
            if os.path.exists(candidate):
                args.glb_file = candidate
                break
        
        if not args.glb_file:
            print("Please specify a GLB file path")
            print("Usage: python tshub_glb_viewer.py <glb_file>")
            return
    
    # Create and run viewer
    viewer = TSHubGLBViewer(args.glb_file)
    viewer.run(auto_rotate=not args.no_rotate)

if __name__ == '__main__':
    main()
