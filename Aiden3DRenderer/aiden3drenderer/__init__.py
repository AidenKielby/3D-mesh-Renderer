"""
Aiden3DRenderer - A lightweight 3D wireframe renderer

Example usage:
    from aiden3drenderer import Renderer3D, register_shape
    
    renderer = Renderer3D()
    renderer.run()
"""

__version__ = "1.13.1"
__author__ = "Aiden"

from .renderer import Renderer3D, register_shape, renderer_type, object_type, register_renderer_event
from .camera import Camera
from . import physics
from . import obj_loader
from . import bounding_box
from .object_type import object_type
from . import dae_loader
from .video_renderer import VideoRenderer3D, VideoRendererObject
from .button import Button
from .entity import Entity
from .custom_shader import CustomShader
from .material import Material
from .math_shape import MathShape

__all__ = [
    "Renderer3D",
    "register_shape",
    "renderer_type",
    "object_type",
    "register_renderer_event",
    "Camera",
    "physics",
    "obj_loader",
    "bounding_box",
    "object_type",
    "dae_loader",
    "VideoRenderer3D",
    "VideoRendererObject",
    "Button",
    "Entity",
    "CustomShader",
    "Material",
    "MathShape"
]
