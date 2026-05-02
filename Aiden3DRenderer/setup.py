from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).resolve().parent
readme_path = here / "README.md"
if not readme_path.exists():
    readme_path = here.parent / "README.md"

with readme_path.open("r", encoding="utf-8") as fh:
    long_description = fh.read()

mac_requires = [
    "pyobjc-framework-Metal",
    "pyobjc-framework-Cocoa",
    "pyobjc-framework-Quartz",
]

setup(
    name="aiden3drenderer",
    version="1.13.1",
    author="Aiden",
    author_email="headstone.yt@gmail.com",
    description="A real-time 3D function visualizer with a plug-and-play GPU pipeline—write simple compute shaders to create custom effects without dealing with complex rendering internals.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AidenKielby/3D-mesh-Renderer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Graphics :: 3D Rendering",
        "License :: OSI Approved :: APACHE 2.0 License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    keywords=[
        "3d", "renderer", "wireframe", "pygame", "mesh",
        "3d-rendering", "procedural", "terrain", "physics",
        "opengl", "rasterizer", "projection", "camera",
        "computer-graphics", "3d-engine"
    ],
    python_requires=">=3.11",
    install_requires=[
        "pygame>=2.6.0",
        "numpy>=2.4.2",
        "pillow>=12.1.0",
        "lxml>=6.0.2"
    ],
    extras_require={
        "dev": ["pytest>=7.0", "black", "flake8"],
        "shadergraph": ["dearpygui>=1.0.0"],
        "mac-raster": mac_requires,
        "raster": ["moderngl>=5.12.0"],
        "video-renderer": ["opencv-python>=4.13.0.90"]
    },
    package_data={"aiden3drenderer": ["fonts/*.ttf", "fonts/*.png", "Demo/*"]},
    entry_points={
        "console_scripts": [
            "aiden3d-demo=aiden3drenderer.Demo.silly_skull:demo",
            "inverted-aiden3d-demo=aiden3drenderer.Demo.silly_skull:demo_inv",
            "aiden3d-mac=aiden3drenderer.Demo.silly_skull:demo_mac",
            "shader-graph=aiden3drenderer.ShaderGraph.gui:run",
        ],
    },
    include_package_data=True,

)