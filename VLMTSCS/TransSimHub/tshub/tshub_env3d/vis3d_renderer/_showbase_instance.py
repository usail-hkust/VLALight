'''
# Anonymous project source
@Date: 2024-07-03 23:43:42
@Description: 继承 ShowBase, Panda3D 的主界面
LastEditTime: 2025-07-28 22:25:17
'''
from ...utils.get_abs_path import get_abs_path
current_file_path = get_abs_path(__file__)

import ctypes
import os
import sys

# Keep GLVND out of the training process-wide LD_PRELOAD.  Ray rendering
# actors inherit the vendor path and load the dispatch libraries globally
# here, before Panda3D plugins are imported.
if os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"):
    sys.setdlopenflags(os.RTLD_NOW | os.RTLD_GLOBAL)
    for _glvnd_library in (
        "libGLdispatch.so.0",
        "libEGL.so.1",
        "libOpenGL.so.0",
        "libGLX.so.0",
        "libGL.so.1",
    ):
        ctypes.CDLL(_glvnd_library, mode=os.RTLD_NOW | os.RTLD_GLOBAL)

import simplepbr
from loguru import logger
from threading import Lock
from direct.showbase.ShowBase import ShowBase

from panda3d.core import (
    NodePath,
    Shader,
    Filename,
    loadPrcFileData,
)

from .base_render import DEBUG_MODE, BACKEND_LITERALS

class _ShowBaseInstance(ShowBase):
    """Wraps a singleton instance of ShowBase from Panda3D.
    """
    _debug_mode: DEBUG_MODE = DEBUG_MODE.WARNING
    _rendering_backend: BACKEND_LITERALS = "p3headlessgl" # pandagl, p3headlessgl
    _render_mode: str = "onscreen" # onscreen or offscreen

    @classmethod
    def load_config(cls, key, value) -> None:
        """Helper method to load configuration.
        """
        loadPrcFileData("", f"{key} {value}")
        
    def __new__(cls, use_render_pipeline=False):
        # Singleton pattern:  ensure only 1 ShowBase instance
        if "__it__" not in cls.__dict__:
            if cls._debug_mode <= DEBUG_MODE.INFO:
                cls.load_config("gl-debug", "#t") # 启用 OpenGL 的调试模式
                cls.load_config("want-pstats", "1") # 启用 Panda3D 的性能统计工具 PStats
            
            # 设置渲染后端，例如"pandagl"，由类属性决定
            cls.load_config("load-display", cls._rendering_backend)
            if cls._rendering_backend == "p3headlessgl":
                egl_device_index = os.environ.get("PANDA3D_EGL_DEVICE_INDEX", "").strip()
                if egl_device_index:
                    cls.load_config("egl-device-index", egl_device_index)
            cls.load_config("window-title", "TSHub 3D") # 设置窗口标题
            
            # TinyDisplay cannot decode the sRGB vehicle textures.  A
            # p3headlessgl request must therefore fail instead of silently
            # producing corrupted training images with a software fallback.
            aux_displays = [] if cls._rendering_backend == "p3headlessgl" else [
                "pandagl", "pandadx9", "pandadx8",
                "pandagles", "pandagles2", "p3headlessgl", "p3tinydisplay"
            ]
            for display in aux_displays:
                cls.load_config("aux-display", display)

            # Load other configurations
            configs = {
                "sync-video": "false", # 禁用垂直同步，否则渲染速率会被限制为屏幕的刷新率
                "model-cache-compressed-textures": "1", # 启用模型缓存中的压缩纹理，这可以减少内存使用，提高性能。
                "audio-library-name": "null", # 禁用音频库，不处理音频输出
                "notify-level": cls._debug_mode.name.lower(), # 设置通知级别
                "default-directnotify-level": cls._debug_mode.name.lower(), # 设置默认的直接通知级别
                "print-pipe-types": "false", # 禁止打印管道类型信息
                # "show-buffers": "#t", # 开启 Panda3D 的缓冲区可视化功能
            }
            if cls._rendering_backend == "p3headlessgl":
                configs["framebuffer-multisample"] = "0"
                configs["multisamples"] = "0"
            else:
                configs["framebuffer-multisample"] = "1"
                configs["multisamples"] = "8"
                configs["threading-model"] = "Cull/Draw"
            for key, value in configs.items():
                cls.load_config(key, value)
                
        it = cls.__dict__.get("__it__")
        if it is None:
            cls.__it__ = it = object.__new__(cls)
            it.init()
        return it

    def __init__(self) -> None:
        """单例模式 (singleton pattern), 使用 init() 而不是这里的 __init__()
        """
        pass

    def init(self) -> None:
        """Initializer for the purposes of maintaining a singleton of this class.
        """
        self._render_lock = Lock()
        try:
            # There can be only 1 ShowBase instance at a time.
            if _ShowBaseInstance._render_mode == "offscreen":
                super().__init__(windowType="offscreen") # 此时是没有界面的
            elif _ShowBaseInstance._render_mode == "onscreen":
                super().__init__() # 开启可视化界面
            self._validate_rendering_backend()
            # 暂时禁用 simplepbr 进行调试
            # simplepbr.init(
            #     msaa_samples=16,
            #     use_hardware_skinning=True,
            #     use_normal_maps=True,
            #     use_330=False
            # ) # https://github.com/Moguri/panda3d-simplepbr

            self.setBackgroundColor(255, 255, 255, 1) # 设置背景颜色, (0,0,0) 是黑色
            self.setFrameRateMeter(True) # 是否显示 FPS
            logger.info("SIM: 初始化 ShowBase 实例")
        except Exception as e:
            raise e

    def _validate_rendering_backend(self) -> None:
        if self.__class__._rendering_backend != "p3headlessgl":
            return

        pipe_name = self.pipe.getType().getName() if self.pipe is not None else ""
        if pipe_name != "eglGraphicsPipe":
            raise RuntimeError(
                "p3headlessgl requested, but Panda3D selected "
                f"{pipe_name or 'no graphics pipe'}; refusing TinyDisplay fallback"
            )

        gsg = self.win.getGsg() if self.win is not None else None
        vendor = gsg.getDriverVendor().strip() if gsg is not None else ""
        renderer = gsg.getDriverRenderer().strip() if gsg is not None else ""
        version = gsg.getDriverVersion().strip() if gsg is not None else ""
        if not vendor or not renderer or not version:
            raise RuntimeError(
                "p3headlessgl created an EGL output without a valid OpenGL "
                "context; check the Ray worker GLVND/EGL environment"
            )

        requested_device = os.environ.get("PANDA3D_EGL_DEVICE_INDEX", "").strip()
        actual_devices = self._nvml_process_gpu_indices()
        if requested_device and actual_devices:
            requested_index = int(requested_device)
            if requested_index not in actual_devices:
                raise RuntimeError(
                    "Panda3D EGL device mismatch: "
                    f"requested={requested_index}, actual={actual_devices}, pid={os.getpid()}"
                )

        logger.info(
            "SIM: Headless OpenGL ready: pipe={}, vendor={}, renderer={}, version={}, "
            "egl_device_index={}, nvml_gpu_indices={}",
            pipe_name,
            vendor,
            renderer,
            version,
            requested_device or "default",
            actual_devices or "unavailable",
        )

    @staticmethod
    def _nvml_process_gpu_indices() -> list[int]:
        """Return physical GPUs currently holding this renderer process."""
        try:
            import pynvml

            pynvml.nvmlInit()
            pid = os.getpid()
            matches = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                processes = []
                for getter_name in (
                    "nvmlDeviceGetGraphicsRunningProcesses_v3",
                    "nvmlDeviceGetGraphicsRunningProcesses",
                    "nvmlDeviceGetComputeRunningProcesses_v3",
                    "nvmlDeviceGetComputeRunningProcesses",
                ):
                    getter = getattr(pynvml, getter_name, None)
                    if not callable(getter):
                        continue
                    try:
                        processes.extend(getter(handle))
                    except pynvml.NVMLError:
                        continue
                if any(int(process.pid) == pid for process in processes):
                    matches.append(index)
            return matches
        except (ImportError, RuntimeError, ValueError):
            return []
        except Exception as exc:
            logger.warning("SIM: NVML renderer placement check unavailable: {}", exc)
            return []
        
    # #################################
    # 下面两个 method 用于调整 class 的参数
    # #################################
    @classmethod
    def set_render_mode(cls, render_mode: str) -> None:
        """Sets the render mode.
        """
        cls._render_mode = render_mode
              
    @classmethod
    def set_rendering_verbosity(cls, debug_mode: DEBUG_MODE) -> None:
        """Set rendering debug information verbosity.
        """
        cls._debug_mode = debug_mode
        cls.load_config("notify-level", cls._debug_mode.name.lower())
        cls.load_config("default-directnotify-level", cls._debug_mode.name.lower())

    @classmethod
    def set_rendering_backend(
        cls,
        rendering_backend: BACKEND_LITERALS,
    ) -> None:
        """Sets the rendering backend.
        """
        if "__it__" not in cls.__dict__:
            cls._rendering_backend = rendering_backend
        else:
            if cls._rendering_backend != rendering_backend:
                logger.warning("SIM: Cannot apply rendering backend after setup.")

    # ##################
    # 关于 showbase 的删除
    # ##################
    def destroy(self) -> None:
        """Destroy this renderer and clean up all remaining resources.
        """
        super().destroy()
        self.__class__.__it__ = None

    def __del__(self) -> None:
        try:
            self.destroy()
        except BaseException:
            pass
    
    # #############
    # 设置 SIM ROOT
    # #############
    def setup_sim_root(self, simid: str):
        """Creates the simulation root node in the scene graph.
        """
        root_np = NodePath(simid)
        # 根节点放在 render 上面
        with self._render_lock:
            root_np.reparentTo(self.render)
        if self.__class__._rendering_backend == "p3headlessgl":
            logger.info("SIM: Skip root GLSL shader for p3headlessgl backend.")
            return root_np
        
        # 使用 Panda3D 的 Filename 类处理路径
        import os
        vertex_abs = os.path.normpath(current_file_path("../_assets_3d/shader/unlit_shader.vert"))
        fragment_abs = os.path.normpath(current_file_path("../_assets_3d/shader/unlit_shader.frag"))
        
        # 转换为 Panda3D Filename 对象
        vertex_path = Filename.fromOsSpecific(vertex_abs)
        fragment_path = Filename.fromOsSpecific(fragment_abs)
        
        unlit_shader = Shader.load(
            Shader.SL_GLSL,
            vertex=vertex_path,
            fragment=fragment_path,
        )
        
        if unlit_shader is None:
            logger.error("SIM: Shader 加载失败: %s", vertex_path)
        
        root_np.setShader(unlit_shader, priority=10)

        return root_np
