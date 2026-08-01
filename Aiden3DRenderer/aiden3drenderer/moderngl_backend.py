from .renderer_backend import RenderBackend
from .custom_shader import CustomShader
from .object_type import object_type
from .renderer import renderer_type

import numpy as np
import moderngl
from PIL import Image
import pygame
import sys

compute_shader_for_rasterization = """
#version 430

layout(local_size_x = 16, local_size_y = 16) in;

struct Triangle {
    vec2 pos1;    // 0  - 8  bytes
    vec2 pos2;    // 8  - 16 bytes 
    vec2 pos3;    // 16 - 24 bytes 
    
    float d1;     // 24 - 28 bytes (Depth for p1)
    float d2;     // 28 - 32 bytes (Depth for p2)
    float d3;     // 32 - 36 bytes (Depth for p3)
    
    float light_mult;   // 36 - 40 

    vec2 uv1;    // 40 - 48
    vec2 uv2;    // 48 - 56
    vec2 uv3;    // 56 - 64

    float is_skybox;   // 64 - 68
    float texture_index; // 68 - 72
    float pad1; // 72 - 76
    float pad2; // 76 - 80
};

layout(std430, binding = 0) buffer triangle_data {
    Triangle tris[];
};

layout(rgba32f, binding = 0) uniform image2D destTex;

layout(binding = 1) uniform sampler2DArray inTex;
layout(binding = 2) uniform sampler2D skyTex;
uniform uint tri_count;

uniform bool depthView;
uniform bool heatMap;

shared Triangle local_tris[256];

float dot(vec2 p1, vec2 p3, vec2 p2) {
    vec2 tri_vec = p2-p1;
    vec2 other_vec = p3 - p1;

    return tri_vec.x * other_vec.y - tri_vec.y * other_vec.x;
}   

bool is_point_in_tri(vec2 p0, vec2 p1, vec2 p2, vec2 point) {
    float d1 = dot(p0, p1, point);
    float d2 = dot(p1, p2, point);
    float d3 = dot(p2, p0, point);

    bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    
    return !(has_neg && has_pos);
}  

float tri_area(vec2 p1, vec2 p2, vec2 p3){
    vec2 a = p2 - p1;
    vec2 b = p3 - p1;
    // The magnitude of the cross product of two vectors 
    // is the area of the parallelogram they form. 
    // Half of that is the triangle area.
    return 0.5 * abs(a.x * b.y - a.y * b.x);
}

float cross2d(vec2 a, vec2 b) {
    return a.x * b.y - a.y * b.x;
}

float depth_in_tri(vec2 p0, vec2 p1, vec2 p2, vec2 point, vec3 depths) {
    vec2 v0 = p1 - p0;
    vec2 v1 = p2 - p0;
    vec2 v2 = point - p0;

    // Total area (technically double the area, but ratios remain the same)
    float total = cross2d(v0, v1);

    // Prevent division by zero for degenerate triangles
    if (abs(total) < 0.00001) {
        return 1e38; // GLSL equivalent of infinity
    }

    // Barycentric coordinates (weights)
    // We use the vectors from the vertices to the point
    float w1 = cross2d(v2, v1) / total;
    float w2 = cross2d(v0, v2) / total;
    float w0 = 1.0 - w1 - w2;

    // Interpolate depth using the weights
    // depths.x = depth at p0, depths.y = depth at p1, depths.z = depth at p2
    float depth = w0 * depths.x + w1 * depths.y + w2 * depths.z;

    return depth;
}

void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dims = imageSize(destTex);
    bool in_b = pixel_coords.x < dims.x && pixel_coords.y < dims.y;

    vec2 p_center = vec2(pixel_coords) + 0.5;

    float best_depth = 1e38;
    vec3 best_color = imageLoad(destTex, pixel_coords).rgb;

    uint num_tris = min(tri_count, uint(tris.length()));
    uint local_id = gl_LocalInvocationIndex; // 0 to 255

    // LOOP IN CHUNKS OF 256
    for (uint i = 0; i < num_tris; i += 256) {
        if (i + local_id < num_tris) {
            local_tris[local_id] = tris[i + local_id];
        }
        
        barrier(); // Wait for all threads to finish loading
        uint limit = min(256, num_tris - i);
        if (in_b) {
            for (uint j = 0; j < limit; j++) {
                float minx = min(min(local_tris[j].pos1.x, local_tris[j].pos2.x), local_tris[j].pos3.x);
                float maxx = max(max(local_tris[j].pos1.x, local_tris[j].pos2.x), local_tris[j].pos3.x);
                float miny = min(min(local_tris[j].pos1.y, local_tris[j].pos2.y), local_tris[j].pos3.y);
                float maxy = max(max(local_tris[j].pos1.y, local_tris[j].pos2.y), local_tris[j].pos3.y);
                if (p_center.x < minx || p_center.x > maxx ||
                    p_center.y < miny || p_center.y > maxy) {
                    continue;
                }
                if (is_point_in_tri(local_tris[j].pos1, local_tris[j].pos2, local_tris[j].pos3, p_center)) {
                    vec3 inv_z = vec3(
                        1.0 / local_tris[j].d1,
                        1.0 / local_tris[j].d2,
                        1.0 / local_tris[j].d3
                    );

                    float one_over_z = depth_in_tri(
                        local_tris[j].pos1,
                        local_tris[j].pos2,
                        local_tris[j].pos3,
                        p_center,
                        inv_z
                    );

                float d = 1.0 / one_over_z;
                    
                    if (d < best_depth) {
                        best_depth = d;
                        if (depthView){
                            float c = -pow(2, (-abs(d) * 0.75))+1;
                            best_color = vec3(c, c, c);
                        }
                        else if (heatMap){
                            float t = clamp(d * 0.35, 0.0, 1.0);
                            best_color = mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), t);
                        }
                        else{
                            if (local_tris[j].uv1.x < 0.0 || local_tris[j].uv2.x < 0.0 || local_tris[j].uv3.x < 0.0) {
                                float c = -pow(2, (-abs(d) * 0.75))+1;
                                best_color = vec3(c, c, c);
                            }
                            else{
                                vec3 ws = vec3(1.0/local_tris[j].d1, 1.0/local_tris[j].d2, 1.0/local_tris[j].d3);

                                vec3 us = vec3(local_tris[j].uv1.x * ws.x, local_tris[j].uv2.x * ws.y, local_tris[j].uv3.x * ws.z);
                                vec3 vs = vec3(local_tris[j].uv1.y * ws.x, local_tris[j].uv2.y * ws.y, local_tris[j].uv3.y * ws.z);

                                float u_over_w = depth_in_tri(local_tris[j].pos1, local_tris[j].pos2, local_tris[j].pos3, p_center, us);
                                float v_over_w = depth_in_tri(local_tris[j].pos1, local_tris[j].pos2, local_tris[j].pos3, p_center, vs);
                                float one_over_w = depth_in_tri(local_tris[j].pos1, local_tris[j].pos2, local_tris[j].pos3, p_center, ws);

                                vec2 uv = vec2(u_over_w / one_over_w, 1.0 - (v_over_w / one_over_w));

                                vec4 color = vec4(1.0);
                                vec3 real_col = vec3(1.0);

                                if (local_tris[j].is_skybox == 1){
                                    color = texture(skyTex, uv);
                                    real_col = vec3(color.x, color.y, color.z);
                                    real_col = best_color * (1-color.w) + real_col * color.w;
                                }
                                else{
                                    uv = fract(uv);
                                    color = texture(inTex, vec3(uv, local_tris[j].texture_index));
                                    real_col = vec3(color.x, color.y, color.z);
                                    real_col = best_color * (1-color.w) + real_col * color.w;
                                }

                                best_color = vec3(real_col.x * local_tris[j].light_mult, real_col.y * local_tris[j].light_mult, real_col.z * local_tris[j].light_mult);
                            }
                            
                        }
                    }
                }
            }
        }
        
        barrier(); // Wait before loading the next chunk
    }

    if (in_b){
        imageStore(destTex, pixel_coords, vec4(best_color, 1.0));
    }

}

"""

glsl_frag_shader = """
#version 330

uniform sampler2D tex;

in vec2 uv;
out vec4 fragColor;

void main() {
    fragColor = texture(tex, vec2(uv.x, 1-uv.y));
}
"""

glsl_vert_shader = """
#version 330

in vec2 in_pos;
out vec2 uv;

void main() {
    uv = (in_pos + 1.0) * 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

tri_dtype = np.dtype([
    ('pos', 'f4', (3, 2)),    # 3 positions (x, y)
    ('depths', 'f4', 3),      # 3 depth values
    ('light_mult', 'f4', 1),
    ('uv', 'f4', (3, 2)),     # 3 positions (u, v)
    ('is_skybox', 'f4', 1),
    ('texture_index', 'f4', 1),
    ('pad1', 'f4', 1),
    ('pad2', 'f4', 1),
])

class ModernGLBackend(RenderBackend):
    def setup(self, width, height, rasterize, renderer):
        self.renderer = renderer
        self.width = width
        self.height = height
        self.resizable_window = getattr(renderer, "resizable_window", True)
        self.texture_layers = getattr(renderer, "texture_layers", [])
        self.textures = getattr(renderer, "textures", {})
        self.last_texture_array = getattr(renderer, "last_texture_array", None)
        self.last_size = getattr(renderer, "last_size", 1)
        self.shaders = getattr(renderer, "shaders", [])
        if rasterize == True:
            self.ctx = moderngl.create_context()
        else:
            self.ctx = moderngl.create_context(standalone=True)
        self.compute_shader_container = CustomShader(compute_shader_for_rasterization, self.ctx)
        self.compute_shader = self.compute_shader_container.compute_shader
        self.compute_shader["depthView"].value = False
        self.compute_shader["heatMap"].value = False

        self.blit_prog = self.ctx.program(
            vertex_shader=glsl_vert_shader,
            fragment_shader=glsl_frag_shader
        )
        self.blit_prog["tex"].value = 0

        self.blit_vbo = self.ctx.buffer(np.array([
            -1.0, -1.0,
            1.0, -1.0,
            -1.0,  1.0,
            1.0,  1.0,
        ], dtype="f4"))

        self.blit_vao = self.ctx.simple_vertex_array(self.blit_prog, self.blit_vbo, "in_pos")

        self.disable_finish_call = False # when True, increases performance, but might lead to artifacts!
        self.tri_buffer = self.ctx.buffer(reserve=tri_dtype.itemsize * 10000)

        self.texture_path = None
        self.skybox_texture_path = None

        self.texture = None
        self.skybox_texture = None

        self.render_distance = 20
        self.rasterization_size = (width//2, height//2)
        self.rasterization_size = (width // 2, height // 2)

        rw = self.rasterization_size[0] + (16 - self.rasterization_size[0] % 16) % 16
        rh = self.rasterization_size[1] + (16 - self.rasterization_size[1] % 16) % 16
        self.rasterization_size = (rw, rh)

        self.output_tex = self.ctx.texture((rw, rh), 4, dtype='f4')
        self.alt = self.ctx.texture((rw, rh), 4, dtype='f4')
        self._output_clear_rgba = np.ones((rh, rw, 4), dtype=np.float32)
        self.upscaled_surface = pygame.Surface((renderer.width, renderer.height))

        self.output_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.alt.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.raster_half_w = self.rasterization_size[0] // 2
        self.raster_half_h = self.rasterization_size[1] // 2

        self.last_present_tex = self.output_tex
        self._bind_triangle_data_buffer()
        self._sync_renderer_state()

    def _sync_renderer_state(self):
        if not self.renderer:
            return
        r = self.renderer
        r.ctx = self.ctx
        r.compute_shader_container = self.compute_shader_container
        r.compute_shader = self.compute_shader
        r.tri_buffer = self.tri_buffer
        r.output_tex = self.output_tex
        r.alt = self.alt
        r.rasterization_size = self.rasterization_size
        r.raster_half_w = self.raster_half_w
        r.raster_half_h = self.raster_half_h
        r.disable_finish_call = self.disable_finish_call
        r.last_present_tex = self.last_present_tex
        r._output_clear_rgba = self._output_clear_rgba
        r.upscaled_surface = self.upscaled_surface
        r.texture = self.texture
        r.skybox_texture = self.skybox_texture
        r.texture_path = self.texture_path
        r.skybox_texture_path = self.skybox_texture_path
        r.textures = self.textures
        r.texture_layers = self.texture_layers
        r.last_texture_array = self.last_texture_array
        r.last_size = self.last_size
        r.render_distance = self.render_distance

    def _bind_triangle_data_buffer(self):
        cs = self.compute_shader_container
        if cs is None:
            return
        tri_binding = None
        for b in cs.buffers:
            if b[0] == "triangle_data":
                tri_binding = b[4]
                break
        try:
            if tri_binding is not None:
                self.tri_buffer.bind_to_storage_buffer(tri_binding)
                cs.buffer_objects["triangle_data"] = self.tri_buffer
            else:
                cs.set_buffer("triangle_data", 10000, element_size=tri_dtype.itemsize)
        except Exception:
            pass

    def set_render_type(self, type: renderer_type, screen):
        self.render_type = type
        if type == renderer_type.RASTERIZE:
            self.raster_selected = True
            width = self.renderer.width
            height = self.renderer.height
            if self.renderer.resizable_window:
                self.renderer.screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF, pygame.RESIZABLE)
            else:
                self.renderer.screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
            self.ctx = moderngl.create_context()
            self.disable_finish_call = False # when True, increases performance, but might lead to artifacts!
            self.tri_buffer = self.ctx.buffer(reserve=tri_dtype.itemsize * 10000)

            rw = self.rasterization_size[0] + (16 - self.rasterization_size[0] % 16) % 16
            rh = self.rasterization_size[1] + (16 - self.rasterization_size[1] % 16) % 16
            self.rasterization_size = (rw, rh)

            self.output_tex = self.ctx.texture((rw, rh), 4, dtype='f4')
            self.alt = self.ctx.texture((rw, rh), 4, dtype='f4')
            self._output_clear_rgba = np.ones((rh, rw, 4), dtype=np.float32)
            self.upscaled_surface = pygame.Surface((width, height)).convert()

            self.raster_half_w = self.rasterization_size[0] // 2
            self.raster_half_h = self.rasterization_size[1] // 2
            self.compute_shader_container = CustomShader(compute_shader_for_rasterization, self.ctx)
            self.compute_shader = self.compute_shader_container.compute_shader
            self.compute_shader["depthView"].value = False
            self.compute_shader["heatMap"].value = False

            self.output_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.alt.filter = (moderngl.NEAREST, moderngl.NEAREST)

            self.blit_prog = self.ctx.program(
                vertex_shader=glsl_vert_shader,
                fragment_shader=glsl_frag_shader
            )
            self.blit_prog["tex"].value = 0

            self.blit_vbo = self.ctx.buffer(np.array([
                -1.0, -1.0,
                1.0, -1.0,
                -1.0,  1.0,
                1.0,  1.0,
            ], dtype="f4"))

            self.blit_vao = self.ctx.simple_vertex_array(self.blit_prog, self.blit_vbo, "in_pos")

            self._bind_triangle_data_buffer()
            self.rebuild_textures()
            self.rebuild_shaders()
            self._sync_renderer_state()
        else:
            self.renderer.screen = pygame.display.set_mode((self.renderer.width, self.renderer.height))
            self._sync_renderer_state()

    def set_rasterization_size(self, size: tuple[int, int]):
        width, height = size
        width = width + (16 - width % 16) % 16
        height = height + (16 - height % 16) % 16
        self.rasterization_size = (width, height)

        self.raster_half_w = self.rasterization_size[0] // 2
        self.raster_half_h = self.rasterization_size[1] // 2

        if sys.platform != "darwin":
            if self.output_tex is not None:
                self.output_tex.release()
            self.output_tex = self.ctx.texture((width, height), 4, dtype='f4')

            if self.alt is not None:
                self.alt.release()
            self.alt = self.ctx.texture((width, height), 4, dtype='f4')

            self.output_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.alt.filter = (moderngl.NEAREST, moderngl.NEAREST)

            self._output_clear_rgba = np.ones((height, width, 4), dtype=np.float32)
        self._sync_renderer_state()

    def set_texture_for_raster(self, img_path):
        if sys.platform != "darwin":
            if self.texture is not None:
                self.texture.release()

            self.texture_path = img_path
            img = Image.open(self.texture_path).convert("RGBA")
            img_data = np.array(img, dtype='u1')

            self.texture_layers.clear()
            self.texture_layers.append(img_data)
            self.last_texture_array = img_data
            self.last_size = 1

            array_data = np.stack(self.texture_layers, axis=0)  # (layers, h, w, 4)
            self.texture = self.ctx.texture_array(
                size=(img.size[0], img.size[1], self.last_size),
                components=4,
                data=array_data.tobytes()
            )
            self.texture.use(location=1)
            self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.texture.repeat_x = False
            self.texture.repeat_y = False
            self.compute_shader["inTex"].value = 1

            self.textures.clear()
            self.textures[img_path] = len(self.texture_layers) - 1
            self._sync_renderer_state()
            return len(self.texture_layers) - 1

    def add_texture_for_raster(self, img_path):
        if sys.platform != "darwin":
            if not self.texture_layers:
                return self.set_texture_for_raster(img_path) 
            
            if img_path in self.textures:
                return self.textures[img_path]

            img = Image.open(img_path).convert("RGBA")
            img_data = np.array(img, dtype='u1')

            base_h, base_w, _ = self.texture_layers[0].shape
            h, w, _ = img_data.shape
            if (h, w) != (base_h, base_w):
                img = img.resize((base_w, base_h), Image.Resampling.NEAREST)
                img_data = np.array(img, dtype='u1')

            self.texture_layers.append(img_data)
            self.last_size = len(self.texture_layers)

            if self.texture is not None:
                self.texture.release()

            array_data = np.stack(self.texture_layers, axis=0)
            h, w = self.texture_layers[0].shape[:2]
            self.last_size = len(self.texture_layers)
            self.last_texture_array = img_data

            if self.texture is not None:
                self.texture.release()

            array_data = np.stack(self.texture_layers, axis=0)  # (layers, h, w, 4)
            self.texture = self.ctx.texture_array(
                size=(w, h, self.last_size),
                components=4,
                data=array_data.tobytes()
            )
            self.texture.use(location=1)
            self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.texture.repeat_x = False
            self.texture.repeat_y = False
            self.compute_shader["inTex"].value = 1

            self.textures[img_path] = len(self.texture_layers) - 1
            self._sync_renderer_state()
            return len(self.texture_layers) -1
    
    def toggle_depth_view(self, b: bool):
        self.depth_view_enabled = b
        if sys.platform != "darwin":
            self.compute_shader["depthView"].value = b
    
    def toggle_heat_map(self, b: bool):
        self.heat_map_enabled = b
        if sys.platform != "darwin":
            self.compute_shader["heatMap"].value = b

    def rebuild_shaders(self):
        for entryI in range(len(self.shaders)):
            entry = self.shaders[entryI]
            shader: CustomShader = entry.get('shader')
            if shader is None:
                continue
            # new shader
            newShader = CustomShader(shader.shader_code, self.ctx)
            # update the textures for the new context and shader
            for tex in shader.texture_info:
                newShader.add_texture(tex[0], tex[1], tex[2])
            # update the buffers too
            for b in shader.buffers:
                buffer_name = b[0]
                binding = b[4]
                if buffer_name in shader.buffer_objects:
                    old_buf = shader.buffer_objects[buffer_name]
                    
                    new_buf = self.ctx.buffer(data=old_buf.read())
                    new_buf.bind_to_storage_buffer(binding)
                    newShader.buffer_objects[buffer_name] = new_buf
            self.shaders[entryI]['shader'] = newShader
    
    def rasterize(self, all_tris): #make it work
        data = np.zeros(len(all_tris), dtype=tri_dtype)
        for i, (depths, tri, uv1, uv2, uv3, light_m, is_skybox, tri_tex_index) in enumerate(all_tris):
            p0, p1, p2 = tri
            data[i]['pos'] = ((p0[0], p0[1]), (p1[0], p1[1]), (p2[0], p2[1]))
            data[i]['depths'] = depths
            if not is_skybox:
                if None in (uv1, uv2, uv3) or self.texture == None:
                    data[i]['uv'] = ((-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0))
                    data[i]['light_mult'] = 1
                    data[i]["is_skybox"] = 0
                    data[i]["texture_index"] = tri_tex_index
                else:
                    data[i]['uv'] = (uv1, uv2, uv3)
                    data[i]['light_mult'] = light_m
                    data[i]["is_skybox"] = 0
                    data[i]["texture_index"] = tri_tex_index
            else:
                if None in (uv1, uv2, uv3) or self.skybox_texture == None:
                    data[i]['uv'] = ((-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0))
                    data[i]['light_mult'] = 1
                    data[i]["is_skybox"] = 0
                else:
                    data[i]['uv'] = (uv1, uv2, uv3)
                    data[i]['light_mult'] = 1
                    data[i]["is_skybox"] = 1
            

        self.tri_buffer.write(data.tobytes())
        self.tri_buffer.bind_to_storage_buffer(0, offset=0, size=data.nbytes)

        self.output_tex.bind_to_image(0, read=False, write=True)

        self.compute_shader['tri_count'].value = len(all_tris)

        self.output_tex.write(self._output_clear_rgba.tobytes())
        self.alt.write(self._output_clear_rgba.tobytes())
        self.compute_shader.run((self.rasterization_size[0] + 15) // 16, (self.rasterization_size[1] + 15) // 16)

        if not self.disable_finish_call:
            try:
                self.ctx.finish()
            except Exception:
                pass

        self.last_present_tex = self.output_tex
        
        # n is number of triangles processed
        last_binding = self.run_compute_shaders(len(all_tris))

        if last_binding == 0:
            self.output_tex.use(location=0)
        else:
            self.alt.use(location=0)
        
        self.ctx.memory_barrier()

        self.ctx.screen.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        self.blit_vao.render(moderngl.TRIANGLE_STRIP)
        self._sync_renderer_state()
    
    def rebuild_textures(self):
        if not self.texture_layers:
            return
        if self.texture is not None:
            self.texture.release()

        array_data = np.stack(self.texture_layers, axis=0)
        h, w = self.texture_layers[0].shape[:2]
        self.last_size = len(self.texture_layers)

        if self.texture is not None:
            self.texture.release()

        array_data = np.stack(self.texture_layers, axis=0)  # (layers, h, w, 4)
        self.texture = self.ctx.texture_array(
            size=(w, h, self.last_size),
            components=4,
            data=array_data.tobytes()
        )
        self.texture.use(location=1)
        self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.texture.repeat_x = False
        self.texture.repeat_y = False
        self.compute_shader["inTex"].value = 1

        if self.skybox_texture_path:
            self.generate_cross_type_cubemap_skybox(20, self.skybox_texture_path)
        self._sync_renderer_state()

    def run_compute_shaders(self, tri_count):
        if sys.platform == 'darwin':
            return

        last_output_binding = 0

        for entry in self.shaders:
            shader = entry.get('shader')
            if shader is None:
                continue

            inputs = entry.get('inputs', [])
            for inp in inputs:
                if isinstance(inp, tuple) and len(inp) >= 2 and isinstance(inp[0], str):
                    uname = inp[0]
                    getter = inp[1]
                    val = getter() if callable(getter) else getter
                    try:
                        shader.compute_shader[uname].value = val
                    except Exception:
                        shader.compute_shader[uname] = val

            for b in shader.buffers:
                name = b[0]
                binding = b[4]
                if name == 'triangle_data' and name not in shader.buffer_objects:
                    try:
                        self.tri_buffer.bind_to_storage_buffer(binding)
                        shader.buffer_objects[name] = self.tri_buffer
                    except Exception:
                        pass

            if last_output_binding == 1:
                try:
                    self.output_tex.bind_to_image(1, read=False, write=True)
                except Exception:
                    pass
            else:
                try:
                    self.alt.bind_to_image(1, read=False, write=True)
                except Exception:
                    pass

            if last_output_binding == 1:
                try:
                    self.alt.use(location=0)
                    try:
                        shader.compute_shader['srcTex'].value = 0
                    except Exception:
                        pass
                except Exception:
                    pass
            else:
                try:
                    self.output_tex.use(location=last_output_binding)
                    try:
                        shader.compute_shader['srcTex'].value = last_output_binding
                    except Exception:
                        pass
                except Exception:
                    pass

            try:
                shader.compute_shader['tri_count'].value = int(tri_count)
            except Exception:
                pass

            try:
                groups_x = max(1, (self.rasterization_size[0] + 15) // 16)
                groups_y = max(1, (self.rasterization_size[1] + 15) // 16)
                shader.compute_shader.run(groups_x, groups_y, 1)
                if not self.disable_finish_call:
                    try:
                        self.ctx.finish()
                    except Exception:
                        pass
            except Exception:
                pass

            if last_output_binding == 1:
                self.last_present_tex = self.output_tex
            else:
                self.last_present_tex = self.alt

            last_output_binding = (last_output_binding + 1) % 2

        return last_output_binding

    def capture_pause_snapshot(self):
        if self.render_type != renderer_type.RASTERIZE:
            return
        self.ctx.finish()
        rw, rh = self.rasterization_size
        raw_data = self.last_present_tex.read()
        img_array = np.frombuffer(raw_data, dtype='f4').reshape((rh, rw, 4))
        img_uint8 = (np.clip(img_array, 0.0, 1.0) * 255).astype('uint8')
        img_uint8[..., 3] = 255
        img_uint8 = img_uint8[..., [2, 1, 0, 3]]

        image_surface = pygame.image.frombuffer(img_uint8.tobytes(), (self.rasterization_size[0], self.rasterization_size[1]), 'RGBA')
        if self.upscaled_surface.get_size() != (self.renderer.width, self.renderer.height):
            self.upscaled_surface = pygame.Surface((self.renderer.width, self.renderer.height)).convert()
        pygame.transform.scale(image_surface, (self.renderer.width, self.renderer.height), self.upscaled_surface)
        self.renderer.pause_img = image_surface
        self._sync_renderer_state()

    def generate_cubemap_skybox(self, radius: int, texture_path, left_uvs, right_uvs, top_uvs, bottom_uvs, forward_uvs, backward_uvs):
        self.render_distance = radius
        self.renderer.render_distance = radius
        verts = np.array([(-1, -1, -1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (1, 1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, 1)])
        verts = verts * radius
        faces = [(0, 3, 2), (2, 5, 3), (1, 4, 6), (6, 4, 7), (0, 1, 2), (2, 1, 4), (3, 5, 6), (6, 5, 7), (0, 6, 1), (0, 3, 6), (2, 4, 5), (5, 4, 7)]

        uvs = [
            left_uvs[1], left_uvs[3], left_uvs[0], left_uvs[2],
            right_uvs[0], right_uvs[1], right_uvs[2], right_uvs[3],
            backward_uvs[0], backward_uvs[1], backward_uvs[2], backward_uvs[3],
            forward_uvs[0], forward_uvs[1], forward_uvs[2], forward_uvs[3],
            bottom_uvs[0], bottom_uvs[1], bottom_uvs[2], bottom_uvs[3],
            top_uvs[0], top_uvs[1], top_uvs[2], top_uvs[3],
        ]

        uv_faces = [
            (0, 2, 1), (1, 3, 2),
            (4, 6, 5), (5, 6, 7),
            (9, 8, 11), (11, 8, 10),
            (12, 14, 13), (13, 14, 15),
            (16, 19, 17), (16, 18, 19),
            (22, 23, 20), (20, 23, 21),
        ]

        if self.skybox_texture is not None:
            self.skybox_texture.release()
        self.skybox_texture_path = texture_path
        img = Image.open(self.skybox_texture_path).convert("RGBA")
        img_data = np.array(img, dtype='u1')

        self.skybox_texture = self.ctx.texture(img.size, 4, img_data.tobytes())
        self.skybox_texture.use(location=2)
        self.skybox_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.skybox_texture.repeat_x = False
        self.skybox_texture.repeat_y = False
        self.compute_shader["skyTex"].value = 2

        self.renderer.vertices_faces_list.append([verts.tolist(), faces, uvs, uv_faces, object_type.SKYBOX, 0])
        self._sync_renderer_state()

    def generate_cross_type_cubemap_skybox(self, radius: int, img_path):
        img_w, img_h = Image.open(img_path).size
        eps_x = 1.0 / img_w
        eps_y = 1.0 / img_h

        self.generate_cubemap_skybox(
            radius,
            img_path,
            ((0.75 - eps_x, 1 / 3 + eps_y), (0.5 + eps_x, 1 / 3 + eps_y), (0.75 - eps_x, 2 / 3 - eps_y), (0.5 + eps_x, 2 / 3 - eps_y)),
            ((0.25 - eps_x, 1 / 3 + eps_y), (0 + eps_x, 1 / 3 + eps_y), (0.25 - eps_x, 2 / 3 - eps_y), (0 + eps_x, 2 / 3 - eps_y)),
            ((0.5 - eps_x, 1 - eps_y), (0.25 + eps_x, 1 - eps_y), (0.5 - eps_x, 2 / 3 + eps_y), (0.25 + eps_x, 2 / 3 + eps_y)),
            ((0.5 - eps_x, 1 / 3 - eps_y), (0.25 + eps_x, 1 / 3 - eps_y), (0.5 - eps_x, 0 + eps_y), (0.25 + eps_x, 0 + eps_y)),
            ((0.75 + eps_x, 1 / 3 + eps_y), (1 - eps_x, 1 / 3 + eps_y), (0.75 + eps_x, 2 / 3 - eps_y), (1 - eps_x, 2 / 3 - eps_y)),
            ((0.25 + eps_x, 1 / 3 + eps_y), (0.5 - eps_x, 1 / 3 + eps_y), (0.25 + eps_x, 2 / 3 - eps_y), (0.5 - eps_x, 2 / 3 - eps_y)),
        )
            
