## Prerequisites:
* aiden3drenderer installed
* a ".obj" file to use

## The fun part:
to start, import the neccesarry features from aiden3drenderer:
```python
from aiden3drenderer import Renderer3D, obj_loader, Entity
```

create the rest of the code as done in the "first_file" lesson.
the file should now look like this:
```python
from aiden3drenderer import Renderer3D, obj_loader, Entity

renderer = Renderer3D()

renderer.run()
```

next, anywhere before the run function is called, but after the renderer is defined, create an obj object:
```python
obj = obj_loader.get_obj("obj_filepath.obj", renderer.add_texture_for_raster("texture_filepath.png"), scale=4)
```
this code loads the obj, and saves the 3d model itself, the textures for rasterization and other things like the scale

then, add the obj to the renderer:
```python
renderer.add_obj(obj)
```

your program should now look like this:
```python
from aiden3drenderer import Renderer3D, obj_loader, Entity

renderer = Renderer3D()

obj = obj_loader.get_obj("obj_filepath.obj", renderer.add_texture_for_raster("texture_filepath.png"), scale=4)
renderer.add_obj(obj)

renderer.run()
```

now all thats left is to run the program!
