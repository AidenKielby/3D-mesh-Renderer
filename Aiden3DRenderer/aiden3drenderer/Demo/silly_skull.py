import math
import numpy as np
from importlib.resources import files, as_file
from aiden3drenderer import Renderer3D, obj_loader, renderer_type, Entity, bounding_box, CustomShader, Material

skull_script = """
dist_to_player = (
    renderer.camera.position[0] - entity.position[0],
    renderer.camera.position[1] - entity.position[1],
    renderer.camera.position[2] - entity.position[2],
)
entity.velocity = [dist_to_player[0] / 2, dist_to_player[1] / 2, dist_to_player[2] / 2]
"""

dammage_shader = """
#version 430
layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 1) uniform image2D destTex;
layout(binding = 0) uniform sampler2D srcTex;
uniform sampler2D damageImage;
uniform float distance;

void main(){
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dims = imageSize(destTex);

    if (px.x >= dims.x || px.y >= dims.y) {
        return;
    }

    vec2 uv = (vec2(px) + vec2(0.5)) / vec2(dims);

    vec4 srcCol = texelFetch(srcTex, px, 0);
    vec4 damageCol = texture(damageImage, uv);

    vec4 outCol = srcCol;
    if (distance < 1.0) {
        outCol = mix(srcCol, damageCol, damageCol.a);
    }

    imageStore(destTex, px, outCol);
}
"""

invert_colors_shader = """
#version 430
layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 1) uniform image2D destTex;
uniform sampler2D srcTex;

void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dims = imageSize(destTex);

    if (px.x >= dims.x || px.y >= dims.y) {
        return;
    }

    vec4 srcCol = texelFetch(srcTex, px, 0);

    imageStore(destTex, px, vec4(1-srcCol.x, 1-srcCol.y, 1-srcCol.z, srcCol.w));
}
"""

def demo_inv():
    renderer = Renderer3D(600, 600, "Skull following you and yeah", True)
    renderer.set_render_type(renderer_type.RASTERIZE)
    renderer.using_obj_filetype_format = True

    # Resolve demo assets from package resources so the demo works when installed
    demo_pkg = 'aiden3drenderer.Demo'
    with as_file(files(demo_pkg).joinpath('skull.obj')) as skull_obj_path, \
         as_file(files(demo_pkg).joinpath('skull.png')) as skull_png_path, \
         as_file(files(demo_pkg).joinpath('damage_image.png')) as damage_image_path:

        skull_material = Material("skull_mat", skull_png_path, None)

        obj = obj_loader.get_obj(str(skull_obj_path), skull_material, scale=4)

        skull_entity = Entity(obj, renderer, bounding_box=bounding_box.get_bounding_box(obj[0]))
        skull_entity.add_script(skull_script)
        skull_entity.add_entity_variable("dist_to_player", (0,0,0))

        renderer.add_entity(skull_entity)

        shader = CustomShader(dammage_shader, renderer.ctx)

        tex_binding = 3
        _ = shader.add_texture(str(damage_image_path), tex_binding, "damageImage")

    # dynamic distance updates every frame via renderer.shaders input
    renderer.shaders.append({
        'shader': shader,
        'inputs': [
            ("distance", lambda: float(np.linalg.norm(np.array(renderer.entities[0].variables["dist_to_player"]))))
        ],
    })

    color_inv = CustomShader(invert_colors_shader, renderer.ctx)

    renderer.shaders.append({
        'shader': color_inv,
        'inputs': [],
    })

    renderer.run()

def demo():
    renderer = Renderer3D(600, 600, "Skull following you and yeah", True)
    renderer.set_render_type(renderer_type.MESH)
    renderer.using_obj_filetype_format = True

    # Resolve demo assets from package resources so the demo works when installed
    demo_pkg = 'aiden3drenderer.Demo'
    with as_file(files(demo_pkg).joinpath('skull.obj')) as skull_obj_path, \
         as_file(files(demo_pkg).joinpath('skull.png')) as skull_png_path, \
         as_file(files(demo_pkg).joinpath('damage_image.png')) as damage_image_path:

        skull_material = Material("skull_mat", skull_png_path, None)

        obj = obj_loader.get_obj(str(skull_obj_path), skull_material, scale=4)

        skull_entity = Entity(obj, renderer, bounding_box=bounding_box.get_bounding_box(obj[0]))
        skull_entity.add_script(skull_script)
        skull_entity.add_entity_variable("dist_to_player", (0,0,0))

        renderer.add_entity(skull_entity)

        shader = CustomShader(dammage_shader, renderer.ctx)

        tex_binding = 4
        _ = shader.add_texture(str(damage_image_path), tex_binding, "damageImage")

    # dynamic distance updates every frame via renderer.shaders input
    renderer.shaders.append({
        'shader': shader,
        'inputs': [
            ("distance", lambda: float(np.linalg.norm(np.array(renderer.entities[0].variables["dist_to_player"])))),
        ],
    })

    while True:
        renderer.loopable_run()

def demo_mac():
    renderer = Renderer3D(600, 600, "Skull following you and yeah", True)
    renderer.render_type = renderer_type.MESH
    renderer.using_obj_filetype_format = True

    # Resolve demo assets from package resources so the demo works when installed
    demo_pkg = 'aiden3drenderer.Demo'
    with as_file(files(demo_pkg).joinpath('skull.obj')) as skull_obj_path, \
         as_file(files(demo_pkg).joinpath('skull.png')) as skull_png_path:
        skull_material = Material("skull_mat", skull_png_path, None)

        obj = obj_loader.get_obj(str(skull_obj_path), skull_material, scale=4)

        skull_entity = Entity(obj, renderer, bounding_box=bounding_box.get_bounding_box(obj[0]))
        skull_entity.add_script(skull_script)
        skull_entity.add_entity_variable("dist_to_player", (0,0,0))

        renderer.add_entity(skull_entity)

    renderer.run()

if __name__ == "__main__":
    demo()