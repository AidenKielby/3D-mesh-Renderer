"""
Main 3D renderer class
"""
import math
import pygame
from pygame import QUIT
import sys
import importlib
from importlib import resources
import numpy as np
if sys.platform != "darwin":
    import moderngl
else:
    import objc
    from Metal import *
    from Quartz import *

# Optional moderngl import on macOS for compatibility.
if sys.platform == "darwin":
    try:
        import moderngl as _moderngl
        moderngl = _moderngl
        _MAC_HAS_MODERNGL = True
    except Exception:
        _MAC_HAS_MODERNGL = False

from PIL import Image
from enum import Enum

from .material import Material
from . import bounding_box
from .object_type import object_type
from .camera import Camera
from .button import Button
from .entity import Entity
from .custom_shader import CustomShader

CUSTOM_SHAPES = {}
CUSTOM_EVENTS = {}

def register_shape(name: str, key=None, is_animated: bool = False, color: tuple[int] = None):
    def decorator(func):
        CUSTOM_SHAPES[name] = {
            'function': func,
            'is_animated': is_animated,
            'key': key,
            'color': color
        }
        return func
    return decorator

def register_renderer_event(key):
    def decorator(func):
        if key in CUSTOM_EVENTS:
            CUSTOM_EVENTS[key].append(func)
        else:
            CUSTOM_EVENTS[key] = [func]
        return func
    return decorator

class renderer_type(Enum):
    RASTERIZE = "rasterize"
    POLYGON_FILL = "polygon_fill"
    MESH = "mesh"

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
                    vec3 ds = vec3(local_tris[j].d1, local_tris[j].d2, local_tris[j].d3);
                    float d = depth_in_tri(local_tris[j].pos1, local_tris[j].pos2, local_tris[j].pos3, p_center, ds);
                    
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

metal_compute_shader = """
#pragma clang diagnostic ignored "-Wmissing-prototypes"

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Globals
{
    uint tri_count;
    uint depthView;
    uint heatMap;
};

struct Triangle
{
    float2 pos1;
    float2 pos2;
    float2 pos3;
    float d1;
    float d2;
    float d3;
    float light_mult;
    float2 uv1;
    float2 uv2;
    float2 uv3;
    float is_skybox;
    float texture_index;
    float pad1;
    float pad2;
};

struct triangle_data
{
    Triangle tris[1];
};


constant uint3 gl_WorkGroupSize [[maybe_unused]] = uint3(16u, 16u, 1u);

static inline __attribute__((always_inline))
float _dot(thread const float2& p1, thread const float2& p3, thread const float2& p2)
{
    float2 tri_vec = p2 - p1;
    float2 other_vec = p3 - p1;
    return (tri_vec.x * other_vec.y) - (tri_vec.y * other_vec.x);
}

static inline __attribute__((always_inline))
bool is_point_in_tri(thread const float2& p0, thread const float2& p1, thread const float2& p2, thread const float2& point)
{
    float2 param = p0;
    float2 param_1 = p1;
    float2 param_2 = point;
    float d1 = _dot(param, param_1, param_2);
    float2 param_3 = p1;
    float2 param_4 = p2;
    float2 param_5 = point;
    float d2 = _dot(param_3, param_4, param_5);
    float2 param_6 = p2;
    float2 param_7 = p0;
    float2 param_8 = point;
    float d3 = _dot(param_6, param_7, param_8);
    bool has_neg = ((d1 < 0.0) || (d2 < 0.0)) || (d3 < 0.0);
    bool has_pos = ((d1 > 0.0) || (d2 > 0.0)) || (d3 > 0.0);
    return !(has_neg && has_pos);
}

static inline __attribute__((always_inline))
float cross2d(thread const float2& a, thread const float2& b)
{
    return (a.x * b.y) - (a.y * b.x);
}

static inline __attribute__((always_inline))
float depth_in_tri(thread const float2& p0, thread const float2& p1, thread const float2& p2, thread const float2& point, thread const float3& depths)
{
    float2 v0 = p1 - p0;
    float2 v1 = p2 - p0;
    float2 v2 = point - p0;
    float2 param = v0;
    float2 param_1 = v1;
    float total = cross2d(param, param_1);
    if (abs(total) < 9.9999997473787516355514526367188e-06)
    {
        return 9.9999996802856924650656260769173e+37;
    }
    float2 param_2 = v2;
    float2 param_3 = v1;
    float w1 = cross2d(param_2, param_3) / total;
    float2 param_4 = v0;
    float2 param_5 = v2;
    float w2 = cross2d(param_4, param_5) / total;
    float w0 = (1.0 - w1) - w2;
    float depth = ((w0 * depths.x) + (w1 * depths.y)) + (w2 * depths.z);
    return depth;
}

kernel void main0(constant uint* spvBufferSizeConstants [[buffer(25)]], constant Globals& globals [[buffer(0)]], device triangle_data& _251 [[buffer(1)]], texture2d<float, access::read_write> destTex [[texture(0)]], texture2d<float> skyTex [[texture(1)]], texture2d_array<float> inTex [[texture(2)]], sampler skyTexSmplr [[sampler(0)]], sampler inTexSmplr [[sampler(1)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]], uint gl_LocalInvocationIndex [[thread_index_in_threadgroup]])
{
    threadgroup Triangle local_tris[256];
    constant uint& _251BufferSize = spvBufferSizeConstants[1];
    int2 pixel_coords = int2(gl_GlobalInvocationID.xy);
    int2 dims = int2(destTex.get_width(), destTex.get_height());
    bool _216 = pixel_coords.x < dims.x;
    bool _224;
    if (_216)
    {
        _224 = pixel_coords.y < dims.y;
    }
    else
    {
        _224 = _216;
    }
    bool in_b = _224;
    float2 p_center = float2(pixel_coords) + float2(0.5);
    float best_depth = 9.9999996802856924650656260769173e+37;
    float3 best_color = destTex.read(uint2(pixel_coords)).xyz;
    uint num_tris = min(globals.tri_count, uint(int((_251BufferSize - 0) / 80)));
    uint local_id = gl_LocalInvocationIndex;
    for (uint i = 0u; i < num_tris; i += 256u)
    {
        if ((i + local_id) < num_tris)
        {
            uint _284 = i + local_id;
            local_tris[local_id].pos1 = _251.tris[_284].pos1;
            local_tris[local_id].pos2 = _251.tris[_284].pos2;
            local_tris[local_id].pos3 = _251.tris[_284].pos3;
            local_tris[local_id].d1 = _251.tris[_284].d1;
            local_tris[local_id].d2 = _251.tris[_284].d2;
            local_tris[local_id].d3 = _251.tris[_284].d3;
            local_tris[local_id].light_mult = _251.tris[_284].light_mult;
            local_tris[local_id].uv1 = _251.tris[_284].uv1;
            local_tris[local_id].uv2 = _251.tris[_284].uv2;
            local_tris[local_id].uv3 = _251.tris[_284].uv3;
            local_tris[local_id].is_skybox = _251.tris[_284].is_skybox;
            local_tris[local_id].texture_index = _251.tris[_284].texture_index;
            local_tris[local_id].pad1 = _251.tris[_284].pad1;
            local_tris[local_id].pad2 = _251.tris[_284].pad2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint limit = min(256u, (num_tris - i));
        if (in_b)
        {
            for (uint j = 0u; j < limit; j++)
            {
                float minx = fast::min(fast::min(local_tris[j].pos1.x, local_tris[j].pos2.x), local_tris[j].pos3.x);
                float maxx = fast::max(fast::max(local_tris[j].pos1.x, local_tris[j].pos2.x), local_tris[j].pos3.x);
                float miny = fast::min(fast::min(local_tris[j].pos1.y, local_tris[j].pos2.y), local_tris[j].pos3.y);
                float maxy = fast::max(fast::max(local_tris[j].pos1.y, local_tris[j].pos2.y), local_tris[j].pos3.y);
                bool _402 = p_center.x < minx;
                bool _410;
                if (!_402)
                {
                    _410 = p_center.x > maxx;
                }
                else
                {
                    _410 = _402;
                }
                bool _418;
                if (!_410)
                {
                    _418 = p_center.y < miny;
                }
                else
                {
                    _418 = _410;
                }
                bool _426;
                if (!_418)
                {
                    _426 = p_center.y > maxy;
                }
                else
                {
                    _426 = _418;
                }
                if (_426)
                {
                    continue;
                }
                float2 param = local_tris[j].pos1;
                float2 param_1 = local_tris[j].pos2;
                float2 param_2 = local_tris[j].pos3;
                float2 param_3 = p_center;
                if (is_point_in_tri(param, param_1, param_2, param_3))
                {
                    float3 ds = float3(local_tris[j].d1, local_tris[j].d2, local_tris[j].d3);
                    float2 param_4 = local_tris[j].pos1;
                    float2 param_5 = local_tris[j].pos2;
                    float2 param_6 = local_tris[j].pos3;
                    float2 param_7 = p_center;
                    float3 param_8 = ds;
                    float d = depth_in_tri(param_4, param_5, param_6, param_7, param_8);
                    if (d < best_depth)
                    {
                        best_depth = d;
                        if (globals.depthView != 0u)
                        {
                            float c = (-pow(2.0, (-abs(d)) * 0.75)) + 1.0;
                            best_color = float3(c);
                        }
                        else
                        {
                            if (globals.heatMap != 0u)
                            {
                                float t = fast::clamp(d * 0.3499999940395355224609375, 0.0, 1.0);
                                best_color = mix(float3(0.0, 0.0, 1.0), float3(1.0, 0.0, 0.0), t);
                            }
                            else
                            {
                                bool _519 = (local_tris[j].uv1)[0u] < 0.0;
                                bool _527;
                                if (!_519)
                                {
                                    _527 = (local_tris[j].uv2)[0u] < 0.0;
                                }
                                else
                                {
                                    _527 = _519;
                                }
                                bool _535;
                                if (!_527)
                                {
                                    _535 = (local_tris[j].uv3)[0u] < 0.0;
                                }
                                else
                                {
                                    _535 = _527;
                                }
                                if (_535)
                                {
                                    float c_1 = (-pow(2.0, (-abs(d)) * 0.75)) + 1.0;
                                    best_color = float3(c_1);
                                }
                                else
                                {
                                    float3 ws = float3(1.0 / local_tris[j].d1, 1.0 / local_tris[j].d2, 1.0 / local_tris[j].d3);
                                    float3 us = float3((local_tris[j].uv1)[0u] * ws.x, (local_tris[j].uv2)[0u] * ws.y, (local_tris[j].uv3)[0u] * ws.z);
                                    float3 vs = float3((local_tris[j].uv1)[1u] * ws.x, (local_tris[j].uv2)[1u] * ws.y, (local_tris[j].uv3)[1u] * ws.z);
                                    float2 param_9 = local_tris[j].pos1;
                                    float2 param_10 = local_tris[j].pos2;
                                    float2 param_11 = local_tris[j].pos3;
                                    float2 param_12 = p_center;
                                    float3 param_13 = us;
                                    float u_over_w = depth_in_tri(param_9, param_10, param_11, param_12, param_13);
                                    float2 param_14 = local_tris[j].pos1;
                                    float2 param_15 = local_tris[j].pos2;
                                    float2 param_16 = local_tris[j].pos3;
                                    float2 param_17 = p_center;
                                    float3 param_18 = vs;
                                    float v_over_w = depth_in_tri(param_14, param_15, param_16, param_17, param_18);
                                    float2 param_19 = local_tris[j].pos1;
                                    float2 param_20 = local_tris[j].pos2;
                                    float2 param_21 = local_tris[j].pos3;
                                    float2 param_22 = p_center;
                                    float3 param_23 = ws;
                                    float one_over_w = depth_in_tri(param_19, param_20, param_21, param_22, param_23);
                                    float2 uv = float2(u_over_w / one_over_w, 1.0 - (v_over_w / one_over_w));
                                    float4 color = float4(1.0);
                                    float3 real_col = float3(1.0);
                                    if (local_tris[j].is_skybox > 0.5)
                                    {
                                        color = skyTex.sample(skyTexSmplr, uv, level(0.0));
                                        real_col = float3(color.x, color.y, color.z);
                                        real_col = (best_color * (1.0 - color.w)) + (real_col * color.w);
                                    }
                                    else
                                    {
                                        uint slice = uint(local_tris[j].texture_index);
                                        color = inTex.sample(inTexSmplr, uv, slice);
                                        real_col = float3(color.x, color.y, color.z);
                                        real_col = (best_color * (1.0 - color.w)) + (real_col * color.w);
                                    }
                                    best_color = float3(real_col.x * local_tris[j].light_mult, real_col.y * local_tris[j].light_mult, real_col.z * local_tris[j].light_mult);
                                }
                            }
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (in_b)
    {
        destTex.write(float4(best_color, 1.0), uint2(pixel_coords));
    }
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

class Renderer3D:
    
    def __init__(self, width=1000, height=1000, title="Aiden 3D Renderer", load_default_shapes: bool = True, resizable_window: bool = True):
        pygame.init()
        width = width + (16 - width % 16) % 16
        height = height + (16 - height % 16) % 16
        self.width = width
        self.height = height
        self.half_w = width // 2
        self.half_h = height // 2
        pygame.display.set_caption(title)
        
        self.camera = Camera()
        self.clock = pygame.time.Clock()
        
        self.current_shape = "mountain"
        self.last_shape = None
        self.animation_time = 0
        self.grid_coords = None
        self.needs_regen = False
        self.shapes = [self.current_shape]
        self.projected = None

        self.is_starting = True
        self.resizable_window = resizable_window

        self.render_type = renderer_type.MESH
        if self.render_type == renderer_type.RASTERIZE:
            if resizable_window:
                self.screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF, pygame.RESIZABLE)
            else:
                self.screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        else:
            if resizable_window:
                self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
            else:
                self.screen = pygame.display.set_mode((width, height))
        self.depth_view_enabled = False
        self.heat_map_enabled = False
        self.show_debug_fps = False
        self.backface_culling_enabled = True
        self.front_face_ccw = False
        self.triangle_base_color_1= (150, 0, 150)
        self.triangle_base_color_2 = (50, 0, 50)
        self.triangle_color_list_1= []
        self.triangle_color_list_2 = []

        self.grid_coords_list = []
        self.vertices_faces_list = []
        self.textures = {} # name, index
        self.projected_vertices_faces_list = []
        self.using_obj_filetype_format = False
        self.projections_list = []

        self.is_using_default_shapes = load_default_shapes
        self._default_shape_names = set()
        self._default_shapes_loaded = False

        self.last_texture_array = None
        self.last_size = 1
        self.texture_layers = []

        self.lighting_strictness = 0.5

        self.last_time = pygame.time.get_ticks()
        self.time = self.last_time
        self.delta_time = 0.1

        self.bounding_boxes = []

        if load_default_shapes:
            before = set(CUSTOM_SHAPES.keys())
            try:
                mod = importlib.import_module('.shapes', package=__package__)
                importlib.reload(mod)
            except Exception:
                mod = None
            after = set(CUSTOM_SHAPES.keys())
            self._default_shape_names = after - before
            self._default_shapes_loaded = bool(self._default_shape_names)

        if self.render_type == renderer_type.RASTERIZE:
            self.ctx = moderngl.create_context()
        else:
            self.ctx = moderngl.create_context(standalone=True)
        
        if sys.platform != "darwin":
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
        else:
            self.compute_shader = None
        
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
        self.upscaled_surface = pygame.Surface((self.width, self.height)).convert()

        self.output_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.alt.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.raster_half_w = self.rasterization_size[0] // 2
        self.raster_half_h = self.rasterization_size[1] // 2

        self.entities: list[Entity] = []

        font_res = resources.files("aiden3drenderer").joinpath("fonts/not_a_font_but_whatever.png")
        try:
            with resources.as_file(font_res) as font_path:
                pathImg = str(font_path)
                self.shape_material = Material("shapeMat", pathImg, None)
                self.shape_material = self.add_material(self.shape_material)
        except Exception:
            try:
                pathImg = str(font_res)
                self.shape_material = Material("shapeMat", pathImg, None)
                self.shape_material = self.add_material(self.shape_material)
            except Exception:
                self.shape_material = Material("shapeMat", None, None)
                self.shape_material = self.add_material(self.shape_material)

        self.last_present_tex = self.output_tex
        self.pause_img = None
        self.raster_selected = False

        def exit_button():
            pygame.quit()
            sys.exit()

        def resume_button():
            self.show_pause_menu = False
            self.show_settings_menu = False

        def open_settings_menu():
            self.show_settings_menu = True

        def back_to_pause_menu():
            self.show_settings_menu = False

        def set_render_mesh():
            self.raster_selected = False
            if sys.platform != "darwin":
                self.set_render_type(renderer_type.MESH)
            else:
                self.mac_set_render_type(renderer_type.MESH)

        def set_render_fill():
            self.raster_selected = False
            if sys.platform != "darwin":
                self.set_render_type(renderer_type.RASTERIZE)
            else:
                self.mac_set_render_type(renderer_type.RASTERIZE)

        def set_render_raster():
            if sys.platform != "darwin":
                self.set_render_type(renderer_type.RASTERIZE)
            else:
                self.mac_set_render_type(renderer_type.RASTERIZE)

        def toggle_depth_setting():
            self.depth_view_enabled = not self.depth_view_enabled
            self.toggle_depth_view(self.depth_view_enabled)

        def toggle_heat_setting():
            self.heat_map_enabled = not self.heat_map_enabled
            self.toggle_heat_map(self.heat_map_enabled)

        def decrease_fov_setting():
            self.camera.fov = max(30, self.camera.fov - 5)

        def increase_fov_setting():
            self.camera.fov = min(170, self.camera.fov + 5)

        def decrease_lighting_setting():
            self.lighting_strictness = max(0.0, self.lighting_strictness - 0.05)

        def increase_lighting_setting():
            self.lighting_strictness = min(1.0, self.lighting_strictness + 0.05)

        def toggle_default_shapes_setting():
            self.using_obj_filetype_format = not self.using_obj_filetype_format

        def toggle_debug_fps_setting():
            self.show_debug_fps = not self.show_debug_fps

        self.menu_w = width * 2 / 3
        self.menu_x = width / 6
        self.btn_h = height * 1 / 14
        self.y0 = height * 1 / 6
        self.gap = height * 1 / 40

        button_colors = {
            "border": (230, 230, 230),
            "text": (245, 245, 245),
            "resume": (58, 146, 92),
            "mesh": (62, 88, 148),
            "fill": (101, 72, 150),
            "raster": (152, 98, 44),
            "depth": (54, 127, 127),
            "heat": (148, 68, 68),
            "settings": (88, 110, 146),
            "fov": (120, 92, 164),
            "light": (90, 136, 84),
            "shapes": (122, 104, 70),
            "quit": (168, 56, 56),
        }

        settings_col_w = (self.menu_w - self.gap) / 2
        settings_x_left = self.menu_x
        settings_x_right = self.menu_x + settings_col_w + self.gap

        self.resume_button = Button(
            self.screen, (self.menu_w, self.btn_h), (self.menu_x, self.y0), resume_button,
            border_color=button_colors["border"], color=button_colors["resume"],
            text="Resume", text_color=button_colors["text"]
        )
        self.settings_button = Button(
            self.screen, (self.menu_w, self.btn_h), (self.menu_x, self.y0 + (self.btn_h + self.gap) * 1), open_settings_menu,
            border_color=button_colors["border"], color=button_colors["settings"],
            text="Settings", text_color=button_colors["text"]
        )

        self.mesh_button = Button(
            self.screen, (settings_col_w, self.btn_h), (settings_x_left, self.y0 + (self.btn_h + self.gap) * 0), set_render_mesh,
            border_color=button_colors["border"], color=button_colors["mesh"],
            text="Renderer: Mesh", text_color=button_colors["text"]
        )
        self.fill_button = Button(
            self.screen, (settings_col_w, self.btn_h), (settings_x_right, self.y0 + (self.btn_h + self.gap) * 0), set_render_fill,
            border_color=button_colors["border"], color=button_colors["fill"],
            text="Renderer: Fill", text_color=button_colors["text"]
        )
        self.raster_button = Button(
            self.screen, (settings_col_w, self.btn_h), (settings_x_left, self.y0 + (self.btn_h + self.gap) * 1), set_render_raster,
            border_color=button_colors["border"], color=button_colors["raster"],
            text="Renderer: Raster", text_color=button_colors["text"]
        )
        self.depth_button = Button(
            self.screen, (settings_col_w, self.btn_h), (settings_x_left, self.y0 + (self.btn_h + self.gap) * 2), toggle_depth_setting,
            border_color=button_colors["border"], color=button_colors["depth"],
            text="Depth View: OFF", text_color=button_colors["text"]
        )
        self.heat_button = Button(
            self.screen, (settings_col_w, self.btn_h), (settings_x_right, self.y0 + (self.btn_h + self.gap) * 1), toggle_heat_setting,
            border_color=button_colors["border"], color=button_colors["heat"],
            text="Heat Map: OFF", text_color=button_colors["text"]
        )

        self.fov_down_button = Button(
            self.screen, (settings_col_w, self.btn_h), (settings_x_left, self.y0 + (self.btn_h + self.gap) * 3), decrease_fov_setting,
            border_color=button_colors["border"], color=button_colors["fov"],
            text="FOV -", text_color=button_colors["text"]
        )
        self.fov_up_button = Button(
            self.screen, (settings_col_w, self.btn_h), (settings_x_right, self.y0 + (self.btn_h + self.gap) * 3), increase_fov_setting,
            border_color=button_colors["border"], color=button_colors["fov"],
            text="FOV +", text_color=button_colors["text"]
        )

        self.light_down_button = Button(
            self.screen, (settings_col_w, self.btn_h), (settings_x_left, self.y0 + (self.btn_h + self.gap) * 4), decrease_lighting_setting,
            border_color=button_colors["border"], color=button_colors["light"],
            text="Lighting -", text_color=button_colors["text"]
        )
        self.light_up_button = Button(
            self.screen, (settings_col_w, self.btn_h), (settings_x_right, self.y0 + (self.btn_h + self.gap) * 4), increase_lighting_setting,
            border_color=button_colors["border"], color=button_colors["light"],
            text="Lighting +", text_color=button_colors["text"]
        )

        self.default_shapes_button = Button(
            self.screen, (settings_col_w, self.btn_h), (settings_x_right, self.y0 + (self.btn_h + self.gap) * 2), toggle_default_shapes_setting,
            border_color=button_colors["border"], color=button_colors["shapes"],
            text="Default Shapes: ON", text_color=button_colors["text"]
        )

        self.debug_fps_button = Button(
            self.screen, (settings_col_w, self.btn_h), (settings_x_left, self.y0 + (self.btn_h + self.gap) * 5), toggle_debug_fps_setting,
            border_color=button_colors["border"], color=button_colors["settings"],
            text="Debug FPS: OFF", text_color=button_colors["text"]
        )

        self.settings_back_button = Button(
            self.screen, (self.menu_w, self.btn_h), (self.menu_x, height - self.btn_h - height * 1 / 14), back_to_pause_menu,
            border_color=button_colors["border"], color=button_colors["settings"],
            text="Back", text_color=button_colors["text"]
        )

        # Keep quit button at the bottom of the pause menu.
        self.exit_button = Button(
            self.screen,
            (self.menu_w, self.btn_h),
            (self.menu_x, height - self.btn_h - height * 1 / 14),
            exit_button,
            border_color=button_colors["border"],
            color=button_colors["quit"],
            text="Quit",
            text_color=button_colors["text"]
        )

        self.main_pause_buttons = [
            self.resume_button,
            self.settings_button,
            self.exit_button,
        ]

        self.settings_buttons = [
            self.mesh_button,
            self.fill_button,
            self.raster_button,
            self.depth_button,
            self.heat_button,
            self.fov_down_button,
            self.fov_up_button,
            self.light_down_button,
            self.light_up_button,
            self.default_shapes_button,
            self.debug_fps_button,
            self.settings_back_button,
        ]

        self.pause_buttons = self.main_pause_buttons + self.settings_buttons

        self.pause_title_font = pygame.font.Font(None, int(height * 0.08))
        self.pause_info_font = pygame.font.Font(None, int(height * 0.04))
        self.debug_fps_font = pygame.font.Font(None, int(height * 0.03))

        self.show_pause_menu = False
        self.show_settings_menu = False

        # order matters!!! first shader 0, then shader 1, etc.
        # Use a list of dicts: { 'shader': CustomShader, 'inputs': [path,...] }
        self.shaders = []

        if sys.platform != "darwin" and hasattr(self, 'compute_shader_container') and self.compute_shader_container:
            cs = self.compute_shader_container
            # Attempt to bind the renderer's tri buffer to the shader's triangle_data binding
            tri_binding = None
            for b in cs.buffers:
                if b[0] == 'triangle_data':
                    tri_binding = b[4]
                    break
            try:
                if tri_binding is not None:
                    self.tri_buffer.bind_to_storage_buffer(tri_binding)
                    cs.buffer_objects['triangle_data'] = self.tri_buffer
                else:
                    cs.set_buffer('triangle_data', 10000, element_size=tri_dtype.itemsize)
            except Exception:
                pass

            #self.shaders.append({'shader': cs, 'inputs': []})

        if sys.platform == "darwin":
            self.mac_enable_raster()

        self.rasterization_mult = 0.5

    def add_obj(self, obj, bounding_box=None):
        self.vertices_faces_list.append(obj)
        if bounding_box is not None:
            self.bounding_boxes.append(bounding_box)
        mat: Material = obj[5]
        mat = self.add_material(mat)
        obj[5] = mat

    def add_material(self, material: Material):
        material.texture_index = self.add_texture_for_raster(material.texture_path)
        return material

    def add_entity(self, entity: Entity):
        entity.model[5] = self.add_material(entity.model[5])
        self.entities.append(entity)

    def _resize(self, w: int, h: int):
        """Handle window resize: update dims, fonts, display surface, upscaled surface, and button layout."""
        self.width = int(w)
        self.height = int(h)
        self.half_w = self.width // 2
        self.half_h = self.height // 2

        # Update fonts sized to the new window
        pygame.font.init()
        self.pause_title_font = pygame.font.Font(None, int(self.height * 0.08))
        self.pause_info_font = pygame.font.Font(None, int(self.height * 0.04))
        self.debug_fps_font = pygame.font.Font(None, int(self.height * 0.03))

        # Recompute menu layout metrics
        self.menu_w = self.width * 2 / 3
        self.menu_x = self.width / 6
        self.btn_h = self.height * 1 / 14
        self.y0 = self.height * 1 / 6
        self.gap = self.height * 1 / 40

        # Update the display surface to the new size (preserve resizable flag)
        try:
            flags = self.screen.get_flags()
            if flags & pygame.RESIZABLE:
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
            else:
                self.screen = pygame.display.set_mode((self.width, self.height))
        except Exception:
            self.screen = pygame.display.set_mode((self.width, self.height))

        # Ensure upscaled surface matches the window size
        self.upscaled_surface = pygame.Surface((self.width, self.height)).convert()

        # If buttons exist, update their rects and fonts
        if hasattr(self, 'resume_button'):
            settings_col_w = (self.menu_w - self.gap) / 2
            settings_x_left = self.menu_x
            settings_x_right = self.menu_x + settings_col_w + self.gap

            try:
                self.resume_button.set_rect((self.menu_w, self.btn_h), (self.menu_x, self.y0))
                self.settings_button.set_rect((self.menu_w, self.btn_h), (self.menu_x, self.y0 + (self.btn_h + self.gap) * 1))

                self.mesh_button.set_rect((settings_col_w, self.btn_h), (settings_x_left, self.y0 + (self.btn_h + self.gap) * 0))
                self.fill_button.set_rect((settings_col_w, self.btn_h), (settings_x_right, self.y0 + (self.btn_h + self.gap) * 0))
                #self.raster_button.set_rect((settings_col_w, self.btn_h), (settings_x_left, self.y0 + (self.btn_h + self.gap) * 1))
                self.depth_button.set_rect((settings_col_w, self.btn_h), (settings_x_left, self.y0 + (self.btn_h + self.gap) * 2))
                self.heat_button.set_rect((settings_col_w, self.btn_h), (settings_x_right, self.y0 + (self.btn_h + self.gap) * 1))

                self.fov_down_button.set_rect((settings_col_w, self.btn_h), (settings_x_left, self.y0 + (self.btn_h + self.gap) * 3))
                self.fov_up_button.set_rect((settings_col_w, self.btn_h), (settings_x_right, self.y0 + (self.btn_h + self.gap) * 3))

                self.light_down_button.set_rect((settings_col_w, self.btn_h), (settings_x_left, self.y0 + (self.btn_h + self.gap) * 4))
                self.light_up_button.set_rect((settings_col_w, self.btn_h), (settings_x_right, self.y0 + (self.btn_h + self.gap) * 4))

                self.default_shapes_button.set_rect((settings_col_w, self.btn_h), (settings_x_right, self.y0 + (self.btn_h + self.gap) * 2))
                self.debug_fps_button.set_rect((settings_col_w, self.btn_h), (settings_x_left, self.y0 + (self.btn_h + self.gap) * 5))

                self.settings_back_button.set_rect((self.menu_w, self.btn_h), (self.menu_x, self.height - self.btn_h - self.height * 1 / 14))
                self.exit_button.set_rect((self.menu_w, self.btn_h), (self.menu_x, self.height - self.btn_h - self.height * 1 / 14))
            except Exception:
                # Defensive: if any button missing, ignore and continue
                pass
        self.set_rasterization_size((int(self.width*self.rasterization_mult), int(self.height*self.rasterization_mult)))
    
    def set_rasterization_downsize(self, size_mult: float):
        self.rasterization_mult = size_mult
        self.set_rasterization_size((int(self.width*self.rasterization_mult), int(self.height*self.rasterization_mult)))

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

    def toggle_depth_view(self, b: bool):
        self.depth_view_enabled = b
        if sys.platform != "darwin":
            self.compute_shader["depthView"].value = b
    
    def toggle_heat_map(self, b: bool):
        self.heat_map_enabled = b
        if sys.platform != "darwin":
            self.compute_shader["heatMap"].value = b

    def draw_pause_menu(self):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((20, 20, 20, 170))
        self.screen.blit(overlay, (0, 0))

        title = self.pause_title_font.render("Paused", True, (255, 255, 255))
        self.screen.blit(title, title.get_rect(center=(self.half_w, self.height * 0.1)))

        current_renderer = f"Current Renderer: {self.render_type.value}"
        info = self.pause_info_font.render(current_renderer, True, (240, 240, 240))
        self.screen.blit(info, info.get_rect(center=(self.half_w, self.height * 0.78)))

        for button in self.main_pause_buttons:
            button.draw()

    def draw_settings_menu(self):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((20, 20, 20, 170))
        self.screen.blit(overlay, (0, 0))

        title = self.pause_title_font.render("Settings", True, (255, 255, 255))
        self.screen.blit(title, title.get_rect(center=(self.half_w, self.height * 0.1)))

        # Update button labels (use set_text so fonts are refit for current button sizes)
        self.depth_button.set_text(f"Depth View: {'ON' if self.depth_view_enabled else 'OFF'}")
        self.heat_button.set_text(f"Heat Map: {'ON' if self.heat_map_enabled else 'OFF'}")
        self.default_shapes_button.set_text(f"OBJ Mode: {'ON' if self.using_obj_filetype_format else 'OFF'}")
        self.debug_fps_button.set_text(f"Debug FPS: {'ON' if self.show_debug_fps else 'OFF'}")

        info_line_1 = self.pause_info_font.render(f"FOV: {self.camera.fov:.0f}", True, (240, 240, 240))
        info_line_2 = self.pause_info_font.render(f"Lighting Strictness: {self.lighting_strictness:.2f}", True, (240, 240, 240))
        self.screen.blit(info_line_1, info_line_1.get_rect(center=(self.half_w, self.height * 0.75)))
        self.screen.blit(info_line_2, info_line_2.get_rect(center=(self.half_w, self.height * 0.80)))

        for button in self.settings_buttons:
            button.draw()

    def draw_debug_fps(self):
        if not self.show_debug_fps:
            return

        fps = self.clock.get_fps()
        fps_text = self.debug_fps_font.render(f"FPS: {fps:.1f}", True, (20, 20, 20))
        bg_rect = fps_text.get_rect(topleft=(10, 10)).inflate(12, 8)
        pygame.draw.rect(self.screen, (255, 255, 255), bg_rect)
        pygame.draw.rect(self.screen, (20, 20, 20), bg_rect, 1)
        self.screen.blit(fps_text, fps_text.get_rect(center=bg_rect.center))

    def run_compute_shaders(self, tri_count):
        if sys.platform == 'darwin':
            return

        last_output_binding = 0

        for entry in self.shaders:
            shader = entry.get('shader')
            if shader is None:
                continue

            # Allow shader entries to include tuple inputs for dynamic uniform updates.
            # Format: ('uniform_name', value_or_callable)
            # These are applied each frame before attaching textures / running the shader.
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

            # Ensure triangle_data buffer is bound
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

            # Also make the renderer output available as a sampler for shaders that sample the
            # current framebuffer (bound to texture unit 2).
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

            last_output_binding = (last_output_binding+1)%2

        return last_output_binding

    def capture_pause_snapshot(self):
        if self.render_type == renderer_type.RASTERIZE:
            self.ctx.finish()
            rw, rh = self.rasterization_size
            raw_data = self.last_present_tex.read()
            img_array = np.frombuffer(raw_data, dtype='f4').reshape((rh, rw, 4))
            img_uint8 = (np.clip(img_array, 0.0, 1.0) * 255).astype('uint8')
            img_uint8[..., 3] = 255
            img_uint8 = img_uint8[..., [2, 1, 0, 3]] 

            image_surface = pygame.image.frombuffer(img_uint8.tobytes(), (self.rasterization_size[0], self.rasterization_size[1]), 'RGBA')
            # Ensure the destination surface matches requested size (fixes resize bug)
            if self.upscaled_surface.get_size() != (self.width, self.height):
                self.upscaled_surface = pygame.Surface((self.width, self.height)).convert()
            pygame.transform.scale(image_surface, (self.width, self.height), self.upscaled_surface)
            self.pause_img = image_surface

    def signed_area_2d(self, p0, p1, p2):
        return ((p1[0] - p0[0]) * (p2[1] - p0[1])) - ((p1[1] - p0[1]) * (p2[0] - p0[0]))

    def is_backface_projected(self, p0, p1, p2):
        if not self.backface_culling_enabled:
            return False
        area = self.signed_area_2d(p0, p1, p2)
        if abs(area) < 1e-8:
            return True
        if self.front_face_ccw:
            return area <= 0
        return area >= 0

    def set_render_type(self, type: renderer_type):
        self.render_type = type
        if type == renderer_type.RASTERIZE:
            self.raster_selected = True
            if self.resizable_window:
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.OPENGL | pygame.DOUBLEBUF, pygame.RESIZABLE)
            else:
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.OPENGL | pygame.DOUBLEBUF)
            self.ctx = moderngl.create_context()
            self.disable_finish_call = False # when True, increases performance, but might lead to artifacts!
            self.tri_buffer = self.ctx.buffer(reserve=tri_dtype.itemsize * 10000)

            rw = self.rasterization_size[0] + (16 - self.rasterization_size[0] % 16) % 16
            rh = self.rasterization_size[1] + (16 - self.rasterization_size[1] % 16) % 16
            self.rasterization_size = (rw, rh)

            self.output_tex = self.ctx.texture((rw, rh), 4, dtype='f4')
            self.alt = self.ctx.texture((rw, rh), 4, dtype='f4')
            self._output_clear_rgba = np.ones((rh, rw, 4), dtype=np.float32)
            self.upscaled_surface = pygame.Surface((self.width, self.height)).convert()

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

            self.rebuild_textures()
            self.rebuild_shaders()
        else:
            self.screen = pygame.display.set_mode((self.width, self.height))

    def set_texture_for_raster(self, img_path):
        if sys.platform != "darwin":
            if self.texture is not None:
                self.texture.release()

            self.texture_path = img_path
            img = Image.open(self.texture_path).convert("RGBA")
            img_data = np.array(img, dtype='u1')

            self.texture_layers = [img_data]
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

            self.textures = {}
            self.textures[img_path] = len(self.texture_layers) - 1
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
            return len(self.texture_layers) -1
    
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

    def rebuild_textures(self):
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
        
    def smooth_fadeout(self, dist):
        return 0.5*(1+math.cos((1/self.render_distance)*math.pi*min(abs(dist), self.render_distance)))

    def generate_cubemap_skybox(self, radius: int, texture_path, left_uvs, right_uvs, top_uvs, bottom_uvs, forward_uvs, backward_uvs):
        self.render_distance = radius
        #uv inputs go like: op left uv, top right uv, bottom left uv, bottom right uv
        if sys.platform != "darwin":
            verts = np.array([(-1,-1,-1), (1,-1,-1), (-1,1,-1), (-1,-1,1), (1,1,-1), (-1,1,1), (1,-1,1), (1,1,1)])
            verts = verts * radius
            faces = [(0,3,2), (2,5,3), (1,4,6), (6,4,7), (0,1,2), (2,1,4), (3,5,6), (6,5,7), (0,6,1), (0,3,6), (2,4,5), (5,4,7)]

            uvs = [
                # left (0-3)
                left_uvs[1], left_uvs[3], left_uvs[0], left_uvs[2],
                # right (4-7)
                right_uvs[0], right_uvs[1], right_uvs[2], right_uvs[3],
                # backward (8-11)
                backward_uvs[0], backward_uvs[1], backward_uvs[2], backward_uvs[3],
                # forward (12-15)
                forward_uvs[0], forward_uvs[1], forward_uvs[2], forward_uvs[3],
                # bottom (16-19)
                bottom_uvs[0], bottom_uvs[1], bottom_uvs[2], bottom_uvs[3],
                # top (20-23)
                top_uvs[0], top_uvs[1], top_uvs[2], top_uvs[3],
            ]

            uv_faces = [
                (0, 2, 1), (1, 3, 2),    # left
                (4, 6, 5), (5, 6, 7),    # right
                (9, 8, 11), (11, 8, 10), # backward
                (12, 14, 13), (13, 14, 15), # forward
                (16, 19, 17), (16, 18, 19), # bottom
                (22, 23, 20), (20, 23, 21), # top
            ]

            if self.skybox_texture is not None:
                self.skybox_texture.release()
            self.skybox_texture_path = texture_path
            img = Image.open(self.skybox_texture_path).convert("RGBA")
            img_data = np.array(img, dtype='u1')

            self.skybox_texture = self.ctx.texture(img.size, 4, img_data.tobytes())
            self.skybox_texture.use(location=2)  # bind to texture unit 0
            self.skybox_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.skybox_texture.repeat_x = False
            self.skybox_texture.repeat_y = False
            self.compute_shader["skyTex"].value = 2 

            self.vertices_faces_list.append([verts.tolist(),faces,uvs,uv_faces, object_type.SKYBOX, 0])

    def generate_sprite_bilboard(self, material, pos=(0,0,0), size=1):
        if sys.platform != "darwin":
            verts = [pos] 
            faces = [(0, 0, 0)] 

            uvs = [(1,1), (1,0), (0,1), (0,0)]
            uv_faces = [(0,2,1), (3,1,2)]

            material = self.add_material(material)
            self.vertices_faces_list.append([verts, faces, uvs, uv_faces, object_type.BILLBOARD, material, size])
    
    def generate_cross_type_cubemap_skybox(self, radius: int, img_path):
        img_w, img_h = Image.open(img_path).size
        eps_x = 1.0 / img_w
        eps_y = 1.0 / img_h

        self.generate_cubemap_skybox(radius, img_path,
            # right: 
            ((0.75-eps_x,   1/3+eps_y), (0.5+eps_x,     1/3+eps_y), (0.75-eps_x,   2/3-eps_y), (0.5+eps_x,     2/3-eps_y)),
            # left:
            ((0.25-eps_x,   1/3+eps_y), (0+eps_x,       1/3+eps_y), (0.25-eps_x,   2/3-eps_y), (0+eps_x,       2/3-eps_y)),
            # top: 
            ((0.5-eps_x,    1-eps_y),   (0.25+eps_x,    1-eps_y),   (0.5-eps_x,    2/3+eps_y), (0.25+eps_x,    2/3+eps_y)),
            # bottom:
            ((0.5-eps_x,     1/3-eps_y), (0.25+eps_x,   1/3-eps_y), (0.5-eps_x,     0+eps_y),   (0.25+eps_x,   0+eps_y)),
            # forward:
            ((0.75+eps_x,    1/3+eps_y), (1-eps_x,      1/3+eps_y), (0.75+eps_x,    2/3-eps_y), (1-eps_x,      2/3-eps_y)),
            # back: 
            ((0.25+eps_x,    1/3+eps_y), (0.5-eps_x,    1/3+eps_y), (0.25+eps_x,    2/3-eps_y), (0.5-eps_x,    2/3-eps_y)),
        )

    def normalize(self, v):
        length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        if length == 0:
            return (0, 0, 0)
        return (v[0]/length, v[1]/length, v[2]/length)

    def cross(self, a, b):
        return (
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        )

    def dot(self, a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    def set_starting_shape(self, shape: str):
        self.current_shape = shape
        self.shapes = [self.current_shape]

    def set_use_default_shapes(self, use: bool):
        if use == self.is_using_default_shapes:
            return
        self.is_using_default_shapes = use
        if use:
            before = set(CUSTOM_SHAPES.keys())
            try:
                mod = importlib.import_module('.shapes', package=__package__)
                importlib.reload(mod)
            except Exception:
                return
            after = set(CUSTOM_SHAPES.keys())
            added = after - before
            self._default_shape_names.update(added)
            self._default_shapes_loaded = bool(self._default_shape_names)
        else:
            for name in list(self._default_shape_names):
                if name in CUSTOM_SHAPES:
                    del CUSTOM_SHAPES[name]
            self._default_shape_names.clear()
            self._default_shapes_loaded = False

    def shape_to_verticies_faces(self, matrix):
        faces = []
        uv = [(0, 0), (0, 1), (1, 0), (1, 1)]
        uv_faces = []

        if matrix is None or len(matrix) == 0 or len(matrix[0]) == 0:
            return [[], faces, uv, uv_faces, False, 0]

        h = len(matrix)
        w = len(matrix[0])

        def idx(r, c):
            return r * w + c

        for r in range(h):
            row = matrix[r]
            for c in range(len(row)):
                point = row[c]
                if point is None:
                    continue

                if c < w - 1 and r < h - 1:
                    p1 = matrix[r][c + 1]
                    p2 = matrix[r + 1][c]
                    if p1 is not None and p2 is not None:
                        faces.append((idx(r, c), idx(r, c + 1), idx(r + 1, c)))
                        uv_faces.append((0, 1, 2))

                if c > 0 and r > 0:
                    p1 = matrix[r][c - 1]
                    p2 = matrix[r - 1][c]
                    if p1 is not None and p2 is not None:
                        faces.append((idx(r, c), idx(r, c - 1), idx(r - 1, c)))
                        uv_faces.append((3, 2, 1))

        verts = [p for row in matrix for p in row]
        obj = [verts, faces, uv, uv_faces, object_type.OBJ, self.shape_material]
        self.bounding_boxes.append(bounding_box.get_bounding_box(verts))
        return obj


    def mesh_to_matrix(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
        if X.shape != Y.shape or X.shape != Z.shape:
            raise ValueError("X, Y, Z must have the same shape")

        rows = []
        for i in range(X.shape[0]):
            row = []
            for j in range(X.shape[1]):
                # (x, height, z) where height is Y in matplotlib terms
                row.append((float(X[i, j]), float(Z[i, j]), float(Y[i, j])))
            rows.append(row)
        return rows


    def plot_surface_from_mesh(self, renderer, X: list, Y: list, Z: list,
                            name: str = "surface", key=None, color: tuple = (120, 180, 220)):
        """Register a static surface shape from NumPy arrays and set it active."""
        matrix = self.mesh_to_matrix(np.array(X), np.array(Y), np.array(Z))

        @register_shape(name, key=key, is_animated=False, color=color)
        def _surface():
            return matrix

        renderer.set_starting_shape(name)

    def project_3d_to_2d(self, matrix, fov, camera_pos, camera_facing):
        projected = []
        #print(len(self.grid_coords_list))
        if matrix is None:
            return None
        # Cache trig values and projection factor per-call to avoid repeated math calls
        cos_y = math.cos(camera_facing[1])
        sin_y = math.sin(camera_facing[1])
        cos_x = math.cos(camera_facing[0])
        sin_x = math.sin(camera_facing[0])
        cos_z = math.cos(camera_facing[2])
        sin_z = math.sin(camera_facing[2])
        f = 1 / math.tan(fov / 2)
        for xIdx in range(len(matrix)):
            xList = matrix[xIdx]
            row = []
            for yIdx in range(len(xList)):
                point = xList[yIdx]

                if point is None:
                    row.append(None)
                    continue

                x = point[0]
                y = point[1]
                z = point[2]

                x -= camera_pos[0]
                y -= camera_pos[1]
                z -= camera_pos[2]

                x1 = x * cos_y + z * sin_y
                y1 = y
                z1 = -x * sin_y + z * cos_y

                x2 = x1
                y2 = y1 * cos_x - z1 * sin_x
                z2 = y1 * sin_x + z1 * cos_x

                x3 = x2 * cos_z - y2 * sin_z
                y3 = x2 * sin_z + y2 * cos_z
                z3 = z2

                if z3 <= 0.001:
                    row.append(None)
                    continue

                dd_x = (x3 * f) / -z3
                dd_y = (y3 * f) / -z3

                if self.render_type == renderer_type.RASTERIZE:
                    half_w, half_h = self.raster_half_w, self.raster_half_h
                else:
                    half_w, half_h = self.half_w, self.half_h

                px = dd_x * half_h + half_w
                py = dd_y * half_h + half_h

                """margin = 5000
                if px < -margin or px > self.width + margin or py < -margin or py > self.height + margin:
                    row.append(None)
                    continue"""
                
                if self.render_type == renderer_type.MESH:
                    row.append((px, py))
                elif self.render_type == renderer_type.POLYGON_FILL:
                    row.append((px, py, z3))
                elif self.render_type == renderer_type.RASTERIZE:
                    row.append((px, py, z3))
            projected.append(row)

        return projected
    
    def project_3d_to_2d_flat(self, inList, fov, camera_pos, camera_facing, obj_type):
        projected = []
        is_skybox = obj_type == object_type.SKYBOX
        is_billboard = obj_type == object_type.BILLBOARD
        #print(len(self.grid_coords_list))
        if inList is None:
            return None
        # Cache trig values and projection factor per-call to avoid repeated math calls
        cos_y = math.cos(camera_facing[1])
        sin_y = math.sin(camera_facing[1])
        cos_x = math.cos(camera_facing[0])
        sin_x = math.sin(camera_facing[0])
        cos_z = math.cos(camera_facing[2])
        sin_z = math.sin(camera_facing[2])
        f = 1 / math.tan(fov / 2)
        for xIdx in range(len(inList)):
            point = inList[xIdx]

            if point is None:
                projected.append(None)
                continue

            x = point[0]
            y = point[1]
            z = point[2]

            if not is_skybox:
                x -= camera_pos[0]
                y -= camera_pos[1]
                z -= camera_pos[2]

            if not is_billboard:
                x1 = x * cos_y + z * sin_y
                y1 = y
                z1 = -x * sin_y + z * cos_y

                x2 = x1
                y2 = y1 * cos_x - z1 * sin_x
                z2 = y1 * sin_x + z1 * cos_x

                x3 = x2 * cos_z - y2 * sin_z
                y3 = x2 * sin_z + y2 * cos_z
                z3 = z2
            else:
                x3 = x
                y3 = y
                z3 = z

            if z3 <= 0.001:
                projected.append(None)
                continue

            dd_x = (x3 * f) / -z3
            dd_y = (y3 * f) / -z3

            if self.render_type == renderer_type.RASTERIZE:
                half_w, half_h = self.raster_half_w, self.raster_half_h
            else:
                half_w, half_h = self.half_w, self.half_h

            px = dd_x * half_h + half_w
            py = dd_y * half_h + half_h

            """margin = 1000
            if px < -margin or px > self.width + margin or py < -margin or py > self.height + margin:
                projected.append(None)
                continue"""
            
            if self.render_type == renderer_type.MESH:
                projected.append((px, py))
            elif self.render_type == renderer_type.POLYGON_FILL:
                projected.append((px, py, z3))
            elif self.render_type == renderer_type.RASTERIZE:
                projected.append((px, py, z3))

        return projected

    def render_wireframe(self, matrix):
        if self.render_type == renderer_type.MESH:
            if matrix is None:
                return
            for xIdx in range(len(matrix)):
                xList = matrix[xIdx]
                for yIdx in range(len(xList)):
                    point = xList[yIdx]
                    if point is not None:
                        points = []

                        try:
                            if xIdx < len(matrix) - 1 and matrix[xIdx+1][yIdx] is not None:
                                points.append(matrix[xIdx+1][yIdx])
                            if yIdx < len(xList) - 1 and matrix[xIdx][yIdx+1] is not None:
                                points.append(matrix[xIdx][yIdx+1])
                        except IndexError:
                            continue

                        for p in points:
                            pygame.draw.line(self.screen, (0, 0, 0), point, p, 2)
        elif self.render_type == renderer_type.POLYGON_FILL:
            all_tris = []
            for matI in range(len(matrix)):
                mat = matrix[matI]
                if mat is None:
                    continue
                tris = []

                #print(len(self.triangle_color_list_1))

                col1 = self.triangle_color_list_1[matI] if self.triangle_color_list_1[matI] is not None else self.triangle_base_color_1
                col2 = self.triangle_color_list_2[matI] if self.triangle_color_list_2[matI] is not None else self.triangle_base_color_2
                
                for xIdx in range(len(mat)):
                    xList = mat[xIdx]
                    for yIdx in range(len(xList)):
                        point = xList[yIdx]
                        if point is not None:

                            if xIdx < len(mat) - 1 and yIdx < len(xList) - 1:
                                p1 = mat[xIdx][yIdx + 1]
                                p2 = mat[xIdx + 1][yIdx]
                                if p1 is not None and p2 is not None:
                                    """if self.is_backface_projected(point, p1, p2):
                                        continue"""
                                    #pygame.draw.polygon(screen, (150, 0, 150), [point, p1, p2], 0)
                                    d1 = (point[2] + p1[2] + p2[2]) / 3 if len(point) > 2 else 0
                                    tris.append((d1, (point, p1, p2), col1))

                            if xIdx > 0 and yIdx > 0:
                                p1 = mat[xIdx][yIdx - 1]
                                p2 = mat[xIdx - 1][yIdx]
                                if p1 is not None and p2 is not None:
                                    """if self.is_backface_projected(point, p1, p2):
                                        continue"""
                                    #pygame.draw.polygon(screen, (50, 0, 50), [point, p1, p2], 0)
                                    d1 = (point[2] + p1[2] + p2[2]) / 3 if len(point) > 2 else 0
                                    tris.append((d1, (point, p1, p2), col2))
                all_tris.extend(tris)
            
            all_tris.sort(key=lambda t: t[0], reverse=True)

            for _, tri, col in all_tris:
                pygame.draw.polygon(
                    self.screen,
                    col,
                    [(tri[0][0], tri[0][1]), (tri[1][0], tri[1][1]), (tri[2][0], tri[2][1])],
                    0,
                )

    def smax_poly(self, a, b, k):
        h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
        return self.mix(b, a, h) + k * h * (1.0 - h)
    
    def smax_exp(self, a, b, k):
        return np.log(np.exp(k * a) + np.exp(k * b)) / k

    def mix(self, a, b, t):
        return a * (1.0 - t) + b * t
    
    def interp_vert(self, v0, v1, t):
        return (
            v0[0] + t * (v1[0] - v0[0]),
            v0[1] + t * (v1[1] - v0[1]),
            v0[2] + t * (v1[2] - v0[2]),
        )
    def interp_uv(self, uv0, uv1, t):
        return (
            uv0[0] + t * (uv1[0] - uv0[0]),
            uv0[1] + t * (uv1[1] - uv0[1]),
        )
    def t_near(self, v_in, v_out, near):
        denom = v_out[2] - v_in[2]
        if abs(denom) < 1e-10:
            return 0.0
        return (near - v_in[2]) / denom
    
    def clip_triangle_near(self, verts_3d, uvs, near=0.1):
        if verts_3d is None or uvs is None or len(verts_3d) != 3 or len(uvs) != 3:
            return []

        def winding(v0, v1, v2):
            return (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])

        def orient_like_source(tri_verts, tri_uvs, source_w):
            tri_w = winding(tri_verts[0], tri_verts[1], tri_verts[2])
            if abs(source_w) > 1e-12 and tri_w * source_w < 0:
                tri_verts = [tri_verts[0], tri_verts[2], tri_verts[1]]
                tri_uvs = [tri_uvs[0], tri_uvs[2], tri_uvs[1]]
            return tri_verts, tri_uvs

        source_w = winding(verts_3d[0], verts_3d[1], verts_3d[2])

        inside_idx = [i for i, v in enumerate(verts_3d) if v[2] >= near]
        outside_idx = [i for i, v in enumerate(verts_3d) if v[2] < near]

        if len(inside_idx) == 3:
            return [(verts_3d, uvs)]

        if len(inside_idx) == 0:
            return []

        if len(inside_idx) == 1:
            i0 = inside_idx[0]
            o0, o1 = outside_idx[0], outside_idx[1]

            v0, uv0 = verts_3d[i0], uvs[i0]
            v1, uv1 = verts_3d[o0], uvs[o0]
            v2, uv2 = verts_3d[o1], uvs[o1]

            t1 = self.t_near(v0, v1, near)
            t2 = self.t_near(v0, v2, near)

            p1 = self.interp_vert(v0, v1, t1)
            puv1 = self.interp_uv(uv0, uv1, t1)
            p2 = self.interp_vert(v0, v2, t2)
            puv2 = self.interp_uv(uv0, uv2, t2)

            tri_verts, tri_uvs = orient_like_source([v0, p1, p2], [uv0, puv1, puv2], source_w)
            return [(tri_verts, tri_uvs)]

        if len(inside_idx) == 2:
            i0, i1 = inside_idx[0], inside_idx[1]
            o = outside_idx[0]

            v0, uv0 = verts_3d[i0], uvs[i0]
            v1, uv1 = verts_3d[i1], uvs[i1]
            v2, uv2 = verts_3d[o], uvs[o]

            t1 = self.t_near(v0, v2, near)
            t2 = self.t_near(v1, v2, near)

            p1 = self.interp_vert(v0, v2, t1)
            puv1 = self.interp_uv(uv0, uv2, t1)
            p2 = self.interp_vert(v1, v2, t2)
            puv2 = self.interp_uv(uv1, uv2, t2)

            tri1_verts, tri1_uvs = orient_like_source([v0, v1, p1], [uv0, uv1, puv1], source_w)
            tri2_verts, tri2_uvs = orient_like_source([v1, p2, p1], [uv1, puv2, puv1], source_w)

            return [(tri1_verts, tri1_uvs), (tri2_verts, tri2_uvs)]

        return []
    
    def cam(self, pt, is_skybox):
        if not is_skybox:
            x, y, z = pt[0]-self.camera.position[0], pt[1]-self.camera.position[1], pt[2]-self.camera.position[2]
        else:
            x, y, z = pt[0], pt[1], pt[2]
        cy, sy = math.cos(self.camera.rotation[1]), math.sin(self.camera.rotation[1])
        cx, sx = math.cos(self.camera.rotation[0]), math.sin(self.camera.rotation[0])
        cz, sz = math.cos(self.camera.rotation[2]), math.sin(self.camera.rotation[2])
        x1 = x*cy + z*sy;  z1 = -x*sy + z*cy
        x2 = x1;           y2 = y*cx - z1*sx;  z2 = y*sx + z1*cx
        x3 = x2*cz - y2*sz; y3 = x2*sz + y2*cz
        return (x3, y3, z2)

    def render_shape_from_obj_format(self, matrix, texture_p):
        #print(matrix[0][0])
        if self.render_type == renderer_type.RASTERIZE:

            cy, sy = math.cos(self.camera.rotation[1]), math.sin(self.camera.rotation[1])
            cx, sx = math.cos(self.camera.rotation[0]), math.sin(self.camera.rotation[0])
            cz, sz = math.cos(self.camera.rotation[2]), math.sin(self.camera.rotation[2])

            cam_right = ( cy*cz - sy*sx*sz,  -cx*sz,  sy*cz + cy*sx*sz)
            cam_up    = ( cy*sz + sy*sx*cz,   cx*cz,  sy*sz - cy*sx*cz)

            all_tris = []
            fov_rad = math.radians(self.camera.fov)

            for matI in range(len(matrix)):
                mat = matrix[matI]
                if mat is None:
                    continue

                vertices, faces, uv, uv_faces, obj_type, material = mat
                
                is_skybox = obj_type == object_type.SKYBOX
                if not is_skybox:
                    texture_index = material.texture_index
                else:
                    texture_index = material
                is_billboard = obj_type == object_type.BILLBOARD

                if is_billboard:
                    size = self.vertices_faces_list[matI][6]
                    cx_pos = self.vertices_faces_list[matI][0][0]

                    hs = size * 0.5
                    tr = (cx_pos[0] + cam_right[0]*hs + cam_up[0]*hs,
                        cx_pos[1] + cam_right[1]*hs + cam_up[1]*hs,
                        cx_pos[2] + cam_right[2]*hs + cam_up[2]*hs)
                    br = (cx_pos[0] + cam_right[0]*hs - cam_up[0]*hs,
                        cx_pos[1] + cam_right[1]*hs - cam_up[1]*hs,
                        cx_pos[2] + cam_right[2]*hs - cam_up[2]*hs)
                    tl = (cx_pos[0] - cam_right[0]*hs + cam_up[0]*hs,
                        cx_pos[1] - cam_right[1]*hs + cam_up[1]*hs,
                        cx_pos[2] - cam_right[2]*hs + cam_up[2]*hs)
                    bl = (cx_pos[0] - cam_right[0]*hs - cam_up[0]*hs,
                        cx_pos[1] - cam_right[1]*hs - cam_up[1]*hs,
                        cx_pos[2] - cam_right[2]*hs - cam_up[2]*hs)

                    def proj_pt(p):
                        c = self.cam(p, False)
                        if c[2] <= 0.001:
                            return None
                        f = 1.0 / math.tan(fov_rad / 2)
                        return (
                            (c[0] * f / -c[2]) * self.raster_half_h + self.raster_half_w,
                            (c[1] * f / -c[2]) * self.raster_half_h + self.raster_half_h,
                            c[2]
                        )

                    pp = [proj_pt(v) for v in [tr, br, tl, bl]]
                    if None in pp:
                        continue 
                    bill_uvs = [(1,1), (1,0), (0,1), (0,0)]
                    bill_tris = [
                        (0, 2, 1), 
                        (3, 1, 2), 
                    ]
                    bill_uv_faces = [(0, 2, 1), (3, 1, 2)]

                    for fi, (f0, f1, f2) in enumerate(bill_tris):
                        p0, p1, p2 = pp[f0], pp[f1], pp[f2]
                        uvi = bill_uv_faces[fi]
                        u0, u1, u2 = bill_uvs[uvi[0]], bill_uvs[uvi[1]], bill_uvs[uvi[2]]
                        all_tris.append((
                            (p0[2], p1[2], p2[2]),
                            (p0, p1, p2),
                            u0, u1, u2,
                            1.0, False, texture_index
                        ))
                    continue

                if self.using_obj_filetype_format and not is_billboard:
                    unprojected_verticies, *_ = self.vertices_faces_list[matI]

                for faceI in range(len(faces)):
                    face = faces[faceI]

                    if self.using_obj_filetype_format and not is_billboard:
                        up0 = unprojected_verticies[face[0]]
                        up1 = unprojected_verticies[face[1]]
                        up2 = unprojected_verticies[face[2]]
                        if None in (up0, up1, up2):
                            continue

                    uv_face = uv_faces[faceI]
                    uv0 = uv[uv_face[0]]
                    uv1 = uv[uv_face[1]]
                    uv2 = uv[uv_face[2]]
                    
                    if self.using_obj_filetype_format:
                        unprojected_normal = self.normalT_camera_space((up0, up1, up2))
                        light_dir = np.array([0, 1, 0])
                        light_m = max(self.lighting_strictness, np.dot(light_dir, np.array(unprojected_normal)))

                        cam0, cam1, cam2 = self.cam(up0, is_skybox), self.cam(up1, is_skybox), self.cam(up2, is_skybox)

                        clipped = self.clip_triangle_near([cam0, cam1, cam2], [uv0, uv1, uv2], near=0.001)

                        for clipped_verts, clipped_uvs in clipped:
                            def proj(v):
                                f = 1.0 / math.tan(fov_rad / 2)
                                return (
                                    (v[0] * f / -v[2]) * self.raster_half_h + self.raster_half_w,
                                    (v[1] * f / -v[2]) * self.raster_half_h + self.raster_half_h,
                                    v[2]
                                )
                            pp0, pp1, pp2 = proj(clipped_verts[0]), proj(clipped_verts[1]), proj(clipped_verts[2])
                            if not is_skybox:
                                if self.is_backface_projected(pp0, pp1, pp2):
                                    continue
                            if is_skybox:
                                all_tris.append((
                                    (pp0[2], pp1[2], pp2[2]),
                                    (pp0, pp1, pp2),
                                    clipped_uvs[0], clipped_uvs[1], clipped_uvs[2],
                                    light_m, is_skybox, texture_index
                                ))
                            else:
                                if not (abs(pp0[2]) > self.render_distance and abs(pp1[2]) > self.render_distance and abs(pp2[2]) > self.render_distance):
                                    all_tris.append((
                                        (pp0[2], pp1[2], pp2[2]),
                                        (pp0, pp1, pp2),
                                        clipped_uvs[0], clipped_uvs[1], clipped_uvs[2],
                                        light_m, is_skybox, texture_index
                                    ))
                    else:
                        p0 = vertices[face[0]]
                        p1 = vertices[face[1]]
                        p2 = vertices[face[2]]
                        if None in (p0, p1, p2):
                            continue
                        if not is_skybox:
                            cam_dir = self.cam((0, 0, 1), True)  # forward in cam space
                        # compute face normal in camera space for backface culling
                        """unprojected_normal = self.normalT_camera_space((p0, p1, p2))
                        if np.dot(unprojected_normal, [cam_dir[0], cam_dir[1], cam_dir[2]]) > 0:
                            continue"""
                        if not (p0[2] < 0 or p1[2] < 0 or p2[2] < 0):
                            all_tris.append((
                                (p0[2], p1[2], p2[2]),
                                (p0, p1, p2),
                                uv0, uv1, uv2,
                                1, is_skybox, texture_index
                            ))

            n = len(all_tris)
            if n == 0:
                return

            data = np.zeros(n, dtype=tri_dtype)
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

            self.compute_shader['tri_count'].value = n

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
            last_binding = self.run_compute_shaders(n)

            if last_binding == 0:
                self.output_tex.use(location=0)
            else:
                self.alt.use(location=0)
            
            self.ctx.memory_barrier()

            self.ctx.screen.use()
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)

            self.blit_vao.render(moderngl.TRIANGLE_STRIP)


        elif self.render_type == renderer_type.POLYGON_FILL:
            all_tris = []
            s = True
            fov_rad = math.radians(self.camera.fov)
            for matI in range(len(matrix)):
                mat = matrix[matI]
                if mat is None:
                    continue

                vertices, faces, uv, uv_faces, obj_type, texture_index = mat
                is_skybox = obj_type == object_type.SKYBOX
                if self.using_obj_filetype_format:
                    unprojected_verticies, *_ = self.vertices_faces_list[matI]

                for faceI in range(len(faces)):
                    face = faces[faceI]

                    if self.using_obj_filetype_format:
                        up0 = unprojected_verticies[face[0]]
                        up1 = unprojected_verticies[face[1]]
                        up2 = unprojected_verticies[face[2]]
                        if None in (up0, up1, up2):
                            continue

                    uv_face = uv_faces[faceI]
                    uv0 = uv[uv_face[0]]
                    uv1 = uv[uv_face[1]]
                    uv2 = uv[uv_face[2]]
                    
                    if self.using_obj_filetype_format:
                        unprojected_normal = self.normalT_camera_space((up0, up1, up2))

                        if unprojected_normal[0] > 0:
                            col = (255, 0, 0)
                        elif unprojected_normal[0] < 0:
                            col = (155, 0, 0)
                        elif unprojected_normal[1] > 0:
                            col = (0, 255, 0)
                        elif unprojected_normal[1] < 0:
                            col = (0, 155, 0)
                        elif unprojected_normal[2] > 0:
                            col = (0, 0, 255)
                        elif unprojected_normal[2] < 0:
                            col = (0, 0, 155)

                        cam0, cam1, cam2 = self.cam(up0, is_skybox), self.cam(up1, is_skybox), self.cam(up2, is_skybox)

                        clipped = self.clip_triangle_near([cam0, cam1, cam2], [uv0, uv1, uv2], near=0.1)

                        for clipped_verts, clipped_uvs in clipped:
                            def proj(v):
                                f = 1.0 / math.tan(fov_rad / 2)
                                return (
                                    (v[0] * f / -v[2]) * self.half_h + self.half_w,
                                    (v[1] * f / -v[2]) * self.half_h + self.half_h,
                                    v[2]
                                )
                            pp0, pp1, pp2 = proj(clipped_verts[0]), proj(clipped_verts[1]), proj(clipped_verts[2])
                            if not is_skybox:
                                if self.is_backface_projected(pp0, pp1, pp2):
                                    continue
                            if is_skybox:
                                depth = (pp0[2] + pp1[2] + pp2[2]) / 3.0
                                all_tris.append((
                                    depth,
                                    (pp0, pp1, pp2),
                                    col
                                ))
                            else:
                                if not (abs(pp0[2]) > self.render_distance and abs(pp1[2]) > self.render_distance and abs(pp2[2]) > self.render_distance):
                                    depth = (pp0[2] + pp1[2] + pp2[2]) / 3.0
                                    all_tris.append((
                                        depth,
                                        (pp0, pp1, pp2),
                                        col
                                    ))
                    else:
                        p0 = vertices[face[0]]
                        p1 = vertices[face[1]]
                        p2 = vertices[face[2]]
                        if None in (p0, p1, p2):
                            continue
                        if not is_skybox:
                            if self.is_backface_projected(p0, p1, p2):
                                continue
                        depth = (p0[2] + p1[2] + p2[2]) / 3.0
                        if not (p0[2] < 0 or p1[2] < 0 or p2[2] < 0):
                            all_tris.append((
                                depth,
                                (p0, p1, p2),
                                col
                            ))

            all_tris.sort(key=lambda t: t[0], reverse=True)
            for _, tri, col in all_tris:
                pygame.draw.polygon(self.screen, col, [(v[0], v[1]) for v in tri], 0)
        elif self.render_type == renderer_type.MESH:
            fov_rad = math.radians(self.camera.fov)
            for matI in range(len(matrix)):
                mat = matrix[matI]
                if mat is None:
                    continue

                vertices, faces, uv, uv_faces, obj_type, texture_index = mat
                is_skybox = obj_type == object_type.SKYBOX
                if self.using_obj_filetype_format:
                    unprojected_verticies, *_ = self.vertices_faces_list[matI]

                for faceI in range(len(faces)):
                    face = faces[faceI]

                    if self.using_obj_filetype_format:
                        up0 = unprojected_verticies[face[0]]
                        up1 = unprojected_verticies[face[1]]
                        up2 = unprojected_verticies[face[2]]
                        if None in (up0, up1, up2):
                            continue

                    uv_face = uv_faces[faceI]
                    uv0 = uv[uv_face[0]]
                    uv1 = uv[uv_face[1]]
                    uv2 = uv[uv_face[2]]
                    
                    if self.using_obj_filetype_format:
                        unprojected_normal = self.normalT_camera_space((up0, up1, up2))

                        cam0, cam1, cam2 = self.cam(up0, is_skybox), self.cam(up1, is_skybox), self.cam(up2, is_skybox)

                        clipped = self.clip_triangle_near([cam0, cam1, cam2], [uv0, uv1, uv2], near=0.1)

                        for clipped_verts, clipped_uvs in clipped:
                            def proj(v):
                                f = 1.0 / math.tan(fov_rad / 2)
                                return (
                                    (v[0] * f / -v[2]) * self.half_h + self.half_w,
                                    (v[1] * f / -v[2]) * self.half_h + self.half_h,
                                    v[2]
                                )
                            pp0, pp1, pp2 = proj(clipped_verts[0]), proj(clipped_verts[1]), proj(clipped_verts[2])
                            if not is_skybox:
                                if self.is_backface_projected(pp0, pp1, pp2):
                                    continue
                            if not is_skybox:
                                if not (abs(pp0[2]) > self.render_distance and abs(pp1[2]) > self.render_distance and abs(pp2[2]) > self.render_distance):
                                    pygame.draw.line(self.screen, (0, 0, 0), (pp0[0],pp0[1]), (pp1[0],pp1[1]), 2)
                                    pygame.draw.line(self.screen, (0, 0, 0), (pp1[0],pp1[1]), (pp2[0],pp2[1]), 2)
                                    pygame.draw.line(self.screen, (0, 0, 0), (pp2[0],pp2[1]), (pp0[0],pp0[1]), 2)
                            else:
                                pygame.draw.line(self.screen, (0, 0, 0), (pp0[0],pp0[1]), (pp1[0],pp1[1]), 2)
                                pygame.draw.line(self.screen, (0, 0, 0), (pp1[0],pp1[1]), (pp2[0],pp2[1]), 2)
                                pygame.draw.line(self.screen, (0, 0, 0), (pp2[0],pp2[1]), (pp0[0],pp0[1]), 2)
                    else:
                        p0 = vertices[face[0]]
                        p1 = vertices[face[1]]
                        p2 = vertices[face[2]]
                        if None in (p0, p1, p2):
                            continue
                        if not is_skybox:
                            if self.is_backface_projected(p0, p1, p2):
                                continue
                        if not (p0[2] < 0 or p1[2] < 0 or p2[2] < 0):
                            pygame.draw.line(self.screen, (0, 0, 0), (p0[0],p0[1]), (p1[0],p1[1]), 2)
                            pygame.draw.line(self.screen, (0, 0, 0), (p1[0],p1[1]), (p2[0],p2[1]), 2)
                            pygame.draw.line(self.screen, (0, 0, 0), (p2[0],p2[1]), (p0[0],p0[1]), 2)

    


    def normalize(self, v):
        return v / np.linalg.norm(v)

    def depth_in_tri(self, tri, point, depths):
        t_p0 = tri[0]
        t_p1 = tri[1]
        t_p2 = tri[2]
        # Compute barycentric weights using triangle sub-areas, then interpolate depth.
        # weight for p0 = area(p1,p2,P) / area(p0,p1,p2)
        a0 = self.tri_area(t_p1, t_p2, point)
        a1 = self.tri_area(t_p2, t_p0, point)
        a2 = self.tri_area(t_p0, t_p1, point)

        total = self.tri_area(t_p0, t_p1, t_p2)
        if total == 0:
            return math.inf

        w0 = a0 / total
        w1 = a1 / total
        w2 = a2 / total

        depth = w0 * depths[0] + w1 * depths[1] + w2 * depths[2]

        return depth

    def tri_area(self, p1, p2, p3):
        x1 = p1[0]
        x2 = p2[0]
        x3 = p3[0]
        y1 = p1[1]
        y2 = p2[1]
        y3 = p3[1]
        return 0.5 * abs(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))

    def dot(self, p1, p2, p3):
        tri_vec = (p2[0] - p1[0], p2[1] - p1[1])
        other_vec = (p3[0] - p1[0], p3[1] - p1[1])

        rotated_vec = (tri_vec[1], -tri_vec[0])

        return other_vec[0] * rotated_vec[0] + other_vec[1] * rotated_vec[1]

    
    def is_point_inside_triangle(self, p0, p1, p2, point):
        #print(p0,p1,p2)
        d1 = self.dot(p0, p1, point)
        d2 = self.dot(p1, p2, point)
        d3 = self.dot(p2, p0, point)

        if (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0):
            return True
        
        return False

    def to_cam_space(self, point):
        x = point[0]
        y = point[1]
        z = point[2]

        x -= self.camera.position[0]
        y -= self.camera.position[1]
        z -= self.camera.position[2]

        x1 = x * math.cos(self.camera.rotation[1]) + z * math.sin(self.camera.rotation[1])
        y1 = y
        z1 = -x * math.sin(self.camera.rotation[1]) + z * math.cos(self.camera.rotation[1])

        x2 = x1
        y2 = y1 * math.cos(self.camera.rotation[0]) - z1 * math.sin(self.camera.rotation[0])
        z2 = y1 * math.sin(self.camera.rotation[0]) + z1 * math.cos(self.camera.rotation[0])

        x3 = x2 * math.cos(self.camera.rotation[2]) - y2 * math.sin(self.camera.rotation[2])
        y3 = x2 * math.sin(self.camera.rotation[2]) + y2 * math.cos(self.camera.rotation[2])
        z3 = z2

        if z3 <= 0.1:
            return None

        return (x3, y3, z3)

    def normalT_camera_space(self, tri):
        p0, p1, p2 = tri
        edge1 = (p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2])
        edge2 = (p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2])
        nx = edge1[1]*edge2[2] - edge1[2]*edge2[1]
        ny = edge1[2]*edge2[0] - edge1[0]*edge2[2]
        nz = edge1[0]*edge2[1] - edge1[1]*edge2[0]
        length = math.sqrt(nx*nx + ny*ny + nz*nz)
        if length == 0:
            return (0, 0, 0)
        return (nx/length, ny/length, nz/length)
    
    def generate_shape(self, shape_name, time=0):
        if shape_name in CUSTOM_SHAPES:
            shape_info = CUSTOM_SHAPES[shape_name]
            func = shape_info['function']
            color = shape_info['color']

            if color is not None:
                self.triangle_color_list_2.append((color[0]*0.75, color[1]*0.75, color[2]*0.75))
            else:
                self.triangle_color_list_2.append(None)

            self.triangle_color_list_1.append(color)
            
            
            if shape_info['is_animated']:
                return func(time=time), True
            else:
                return func(), False
        
        return None, False
    
    def generate_shape_from_key_press(self, pressedKeys, time=0):
        shapesList = []
        for shape, value in CUSTOM_SHAPES.items():
            key = value['key']
            if key is not None:
                if pressedKeys[key]:
                    shapesList.append(shape)
        
        if len(shapesList) >= 1:
            self.shapes = shapesList
        
    def min_z(self, lvl1):
        values = [
            t[2]
            for lvl2 in lvl1
            for t in lvl2
            if t is not None
        ]
        return max(values) if values else float("-inf")

    def run(self):
        running = True
        while running:
            self.time = pygame.time.get_ticks()
            self.delta_time = (self.time-self.last_time)/1000
            self.last_time = self.time
            ent_indexes = []
            for e in self.entities:
                e.update()
                ent_indexes.append(len(self.vertices_faces_list))
                self.vertices_faces_list.append(e.get_entity())

            self.screen.fill((255, 255, 255))
            self.clock.tick(60)
            self.animation_time += 0.01
            # Precompute FOV radians once per frame
            fov_rad = math.radians(self.camera.fov)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False

                elif event.type == pygame.VIDEORESIZE:
                    self._resize(event.w, event.h)
                    
                self.camera.handle_mouse_events(event)
                if self.show_pause_menu:
                    active_buttons = self.settings_buttons if self.show_settings_menu else self.main_pause_buttons
                    for button in active_buttons:
                        button.update(pygame.mouse.get_pos(), event)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    if not self.show_pause_menu:
                        self.show_pause_menu = True
                    elif self.show_settings_menu:
                        self.show_settings_menu = False
                    else:
                        self.show_pause_menu = False
                        self.show_settings_menu = False
            
            keys = pygame.key.get_pressed()
            
            self.camera.update(keys)

            # go through the custom events and do the function associated with it
            for key_need_press in CUSTOM_EVENTS:
                if keys[key_need_press]:
                    for func in CUSTOM_EVENTS[key_need_press]:
                        func()
            
            self.generate_shape_from_key_press(keys, self.animation_time)
            if not self.using_obj_filetype_format:
                for i in range(len(self.shapes)):
                    shape_name = self.shapes[i]
                    #print(shape_name)
                    self.grid_coords_list.append(self.generate_shape(shape_name, self.animation_time))
                if self.render_type != renderer_type.RASTERIZE:
                    for i in range(len(self.grid_coords_list)):
                        self.grid_coords = self.grid_coords_list[i]
                        if self.grid_coords:
                            projected = self.project_3d_to_2d(
                                self.grid_coords[0],
                                fov_rad,
                                tuple(self.camera.position),
                                tuple(self.camera.rotation)
                            )
                            self.projections_list.append(projected)

                    if self.render_type == renderer_type.POLYGON_FILL:
                        self.render_wireframe(self.projections_list)
                    elif self.render_type == renderer_type.MESH:
                        for proj in self.projections_list:
                            self.render_wireframe(proj)
                else:
                    if self.grid_coords_list is not None:
                        vfl = [
                            self.shape_to_verticies_faces(proj[0])
                            for proj in self.grid_coords_list
                        ]

                        for i in range(len(vfl)):
                            projected = self.project_3d_to_2d_flat(
                                    vfl[i][0],
                                    fov_rad,
                                    tuple(self.camera.position),
                                    tuple(self.camera.rotation),
                                    vfl[i][4]
                                )
                            self.projected_vertices_faces_list.append([projected, vfl[i][1], vfl[i][2], vfl[i][3], vfl[i][4], vfl[i][5]])
                        #if not self.is_mesh:
                        self.render_shape_from_obj_format(self.projected_vertices_faces_list, self.texture_path)
            else:
                for i in range(len(self.vertices_faces_list)):
                    projected = self.project_3d_to_2d_flat(
                            self.vertices_faces_list[i][0],
                            fov_rad,
                            tuple(self.camera.position),
                            tuple(self.camera.rotation),
                            self.vertices_faces_list[i][4]
                        )
                    self.projected_vertices_faces_list.append([projected, self.vertices_faces_list[i][1], self.vertices_faces_list[i][2], self.vertices_faces_list[i][3], self.vertices_faces_list[i][4], self.vertices_faces_list[i][5]])
                #if not self.is_mesh:
                self.render_shape_from_obj_format(self.projected_vertices_faces_list, self.texture_path)

            self.grid_coords_list = []
            self.triangle_color_list_1 = []
            self.triangle_color_list_2 = []
            self.projections_list = []
            self.projected_vertices_faces_list = []

            if self.show_pause_menu:
                if self.pause_img is None and self.raster_selected:
                    self.capture_pause_snapshot()
                    if sys.platform != "darwin":
                        self.set_render_type(renderer_type.POLYGON_FILL)
                    else:
                        self.mac_set_render_type(renderer_type.POLYGON_FILL)
                if self.pause_img is not None and self.raster_selected:
                    self.screen.blit(self.upscaled_surface, (0, 0))
            else:
                if self.raster_selected and self.pause_img is not None:
                    if sys.platform != "darwin":
                        self.set_render_type(renderer_type.RASTERIZE)
                    else:
                        self.mac_set_render_type(renderer_type.RASTERIZE)
                self.pause_img = None

            #print(self.pause_img)
            
            for button in self.pause_buttons:
                button.toggled = False
            if self.show_pause_menu:
                active_buttons = self.settings_buttons if self.show_settings_menu else self.main_pause_buttons
                for button in active_buttons:
                    button.toggled = True
                if self.show_settings_menu:
                    self.draw_settings_menu()
                else:
                    self.draw_pause_menu()

            self.draw_debug_fps()

            pygame.display.flip()

            for i in ent_indexes:
                del self.vertices_faces_list[i]
        
        pygame.quit()

    def loopable_run(self):
        self.time = pygame.time.get_ticks()
        self.delta_time = (self.time-self.last_time)/1000
        self.last_time = self.time
        ent_indexes = []
        for e in self.entities:
            e.update()
            ent_indexes.append(len(self.vertices_faces_list))
            self.vertices_faces_list.append(e.get_entity())

        self.screen.fill((255, 255, 255))
        self.clock.tick(60)
        self.animation_time += 0.01
        # Precompute FOV radians once per frame
        fov_rad = math.radians(self.camera.fov)
        
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                self._resize(event.w, event.h)
                
            self.camera.handle_mouse_events(event)
            if self.show_pause_menu:
                active_buttons = self.settings_buttons if self.show_settings_menu else self.main_pause_buttons
                for button in active_buttons:
                    button.update(pygame.mouse.get_pos(), event)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                if not self.show_pause_menu:
                    self.show_pause_menu = True
                elif self.show_settings_menu:
                    self.show_settings_menu = False
                else:
                    self.show_pause_menu = False
                    self.show_settings_menu = False
        
        keys = pygame.key.get_pressed()
        
        self.camera.update(keys)
        # go through the custom events and do the function associated with it
        for key_need_press in CUSTOM_EVENTS:
            if keys[key_need_press]:
                for func in CUSTOM_EVENTS[key_need_press]:
                    func()
            
        
        self.generate_shape_from_key_press(keys, self.animation_time)
        if not self.using_obj_filetype_format:
            for i in range(len(self.shapes)):
                shape_name = self.shapes[i]
                #print(shape_name)
                self.grid_coords_list.append(self.generate_shape(shape_name, self.animation_time))
            if self.render_type != renderer_type.RASTERIZE:
                for i in range(len(self.grid_coords_list)):
                    self.grid_coords = self.grid_coords_list[i]
                    if self.grid_coords:
                        projected = self.project_3d_to_2d(
                            self.grid_coords[0],
                            fov_rad,
                            tuple(self.camera.position),
                            tuple(self.camera.rotation)
                        )
                        self.projections_list.append(projected)

                if self.render_type == renderer_type.POLYGON_FILL:
                    self.render_wireframe(self.projections_list)
                elif self.render_type == renderer_type.MESH:
                    for proj in self.projections_list:
                        self.render_wireframe(proj)
            else:
                if self.grid_coords_list is not None:
                    vfl = [
                        self.shape_to_verticies_faces(proj[0])
                        for proj in self.grid_coords_list
                    ]

                    for i in range(len(vfl)):
                        projected = self.project_3d_to_2d_flat(
                                vfl[i][0],
                                fov_rad,
                                tuple(self.camera.position),
                                tuple(self.camera.rotation),
                                vfl[i][4]
                            )
                        self.projected_vertices_faces_list.append([projected, vfl[i][1], vfl[i][2], vfl[i][3], vfl[i][4], vfl[i][5]])
                    #if not self.is_mesh:
                    self.render_shape_from_obj_format(self.projected_vertices_faces_list, self.texture_path)
        else:
            for i in range(len(self.vertices_faces_list)):
                projected = self.project_3d_to_2d_flat(
                        self.vertices_faces_list[i][0],
                        fov_rad,
                        tuple(self.camera.position),
                        tuple(self.camera.rotation),
                        self.vertices_faces_list[i][4]
                    )
                self.projected_vertices_faces_list.append([projected, self.vertices_faces_list[i][1], self.vertices_faces_list[i][2], self.vertices_faces_list[i][3], self.vertices_faces_list[i][4], self.vertices_faces_list[i][5]])
            #if not self.is_mesh:
            self.render_shape_from_obj_format(self.projected_vertices_faces_list, self.texture_path)

        self.grid_coords_list = []
        self.triangle_color_list_1 = []
        self.triangle_color_list_2 = []
        self.projections_list = []
        self.projected_vertices_faces_list = []
        
        for button in self.pause_buttons:
            button.toggled = False
        if self.show_pause_menu:
            active_buttons = self.settings_buttons if self.show_settings_menu else self.main_pause_buttons
            for button in active_buttons:
                button.toggled = True
            if self.show_settings_menu:
                self.draw_settings_menu()
            else:
                self.draw_pause_menu()

        self.draw_debug_fps()

        pygame.display.flip()

        for i in ent_indexes:
            del self.vertices_faces_list[i]

    def mac_barycentric(self, p, a, b, c):
        v0 = (b[0] - a[0], b[1] - a[1])
        v1 = (c[0] - a[0], c[1] - a[1])
        v2 = (p[0] - a[0], p[1] - a[1])
        denom = v0[0] * v1[1] - v0[1] * v1[0]
        if abs(denom) < 1e-10:
            return None
        w1 = (v2[0] * v1[1] - v2[1] * v1[0]) / denom
        w2 = (v0[0] * v2[1] - v0[1] * v2[0]) / denom
        w0 = 1.0 - w1 - w2
        return (w0, w1, w2)

    def mac_sample_texture(self, u, v, is_skybox, texture_index):
        if is_skybox:
            tex = getattr(self, "_mac_skybox_texture", None)
        else:
            layers = getattr(self, "_mac_texture_layers", [])
            if texture_index is None or texture_index < 0 or texture_index >= len(layers):
                tex = None
            else:
                tex = layers[int(texture_index)]
        if tex is None:
            return None

        h, w = tex.shape[:2]
        uu = np.clip(u, 0.0, 1.0)
        vv = np.clip(v, 0.0, 1.0)
        x = min(w - 1, max(0, int(uu * (w - 1))))
        y = min(h - 1, max(0, int(vv * (h - 1))))
        px = tex[y, x]
        if px.dtype != np.float32:
            return px.astype(np.float32) / 255.0
        return px

    def mac_set_rasterization_size(self, size: tuple[int, int]):
        width, height = size
        width = width + (16 - width % 16) % 16
        height = height + (16 - height % 16) % 16
        self.rasterization_size = (width, height)
        self.raster_half_w = self.rasterization_size[0] // 2
        self.raster_half_h = self.rasterization_size[1] // 2
        self._mac_output = np.ones((height, width, 4), dtype=np.float32)
        self._mac_depth = np.full((height, width), np.inf, dtype=np.float32)
        self._output_clear_rgba = np.ones((height, width, 4), dtype=np.float32)

    def mac_toggle_depth_view(self, b: bool):
        self.depth_view_enabled = b

    def mac_toggle_heat_map(self, b: bool):
        self.heat_map_enabled = b

    def mac_set_texture_for_raster(self, img_path):
        if img_path is None:
            return None
        img = Image.open(img_path).convert("RGBA")
        img_data = np.array(img, dtype='u1')
        self._mac_texture_layers = [img_data]
        self._mac_textures = {img_path: 0}
        self.texture_layers = self._mac_texture_layers
        self.textures = self._mac_textures
        return 0

    def mac_add_texture_for_raster(self, img_path):
        if img_path is None:
            return None
        if not getattr(self, "_mac_texture_layers", []):
            return self.mac_set_texture_for_raster(img_path)
        if img_path in self._mac_textures:
            return self._mac_textures[img_path]

        img = Image.open(img_path).convert("RGBA")
        img_data = np.array(img, dtype='u1')

        base_h, base_w, _ = self._mac_texture_layers[0].shape
        h, w, _ = img_data.shape
        if (h, w) != (base_h, base_w):
            img = img.resize((base_w, base_h), Image.Resampling.NEAREST)
            img_data = np.array(img, dtype='u1')

        self._mac_texture_layers.append(img_data)
        idx = len(self._mac_texture_layers) - 1
        self._mac_textures[img_path] = idx
        self.texture_layers = self._mac_texture_layers
        self.textures = self._mac_textures
        return idx

    def mac_rebuild_textures(self):
        # CPU textures are stored in-memory; nothing to rebuild.
        return

    def mac_generate_cubemap_skybox(self, radius: int, texture_path, left_uvs, right_uvs, top_uvs, bottom_uvs, forward_uvs, backward_uvs):
        self.render_distance = radius
        verts = np.array([(-1,-1,-1), (1,-1,-1), (-1,1,-1), (-1,-1,1), (1,1,-1), (-1,1,1), (1,-1,1), (1,1,1)])
        verts = verts * radius
        faces = [(0,3,2), (2,5,3), (1,4,6), (6,4,7), (0,1,2), (2,1,4), (3,5,6), (6,5,7), (0,6,1), (0,3,6), (2,4,5), (5,4,7)]

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

        self.skybox_texture_path = texture_path
        img = Image.open(self.skybox_texture_path).convert("RGBA")
        self._mac_skybox_texture = np.array(img, dtype='u1')
        self.vertices_faces_list.append([verts.tolist(), faces, uvs, uv_faces, object_type.SKYBOX, 0])

    def mac_generate_cross_type_cubemap_skybox(self, radius: int, img_path):
        img_w, img_h = Image.open(img_path).size
        eps_x = 1.0 / img_w
        eps_y = 1.0 / img_h
        self.mac_generate_cubemap_skybox(radius, img_path,
            ((0.75-eps_x,   1/3+eps_y), (0.5+eps_x,     1/3+eps_y), (0.75-eps_x,   2/3-eps_y), (0.5+eps_x,     2/3-eps_y)),
            ((0.25-eps_x,   1/3+eps_y), (0+eps_x,       1/3+eps_y), (0.25-eps_x,   2/3-eps_y), (0+eps_x,       2/3-eps_y)),
            ((0.5-eps_x,    1-eps_y),   (0.25+eps_x,    1-eps_y),   (0.5-eps_x,    2/3+eps_y), (0.25+eps_x,    2/3+eps_y)),
            ((0.5-eps_x,     1/3-eps_y), (0.25+eps_x,   1/3-eps_y), (0.5-eps_x,     0+eps_y),   (0.25+eps_x,   0+eps_y)),
            ((0.75+eps_x,    1/3+eps_y), (1-eps_x,      1/3+eps_y), (0.75+eps_x,    2/3-eps_y), (1-eps_x,      2/3-eps_y)),
            ((0.25+eps_x,    1/3+eps_y), (0.5-eps_x,    1/3+eps_y), (0.25+eps_x,    2/3-eps_y), (0.5-eps_x,    2/3-eps_y)),
        )

    def mac_capture_pause_snapshot(self):
        if self._mac_last_surface is None:
            return
        if self.upscaled_surface.get_size() != (self.width, self.height):
            self.upscaled_surface = pygame.Surface((self.width, self.height)).convert()
        pygame.transform.scale(self._mac_last_surface, (self.width, self.height), self.upscaled_surface)
        self.pause_img = self._mac_last_surface

    def mac_collect_tris(self, matrix):
        cy, sy = math.cos(self.camera.rotation[1]), math.sin(self.camera.rotation[1])
        cx, sx = math.cos(self.camera.rotation[0]), math.sin(self.camera.rotation[0])
        cz, sz = math.cos(self.camera.rotation[2]), math.sin(self.camera.rotation[2])

        cam_right = ( cy*cz - sy*sx*sz,  -cx*sz,  sy*cz + cy*sx*sz)
        cam_up    = ( cy*sz + sy*sx*cz,   cx*cz,  sy*sz - cy*sx*cz)

        all_tris = []
        fov_rad = math.radians(self.camera.fov)

        for matI in range(len(matrix)):
            mat = matrix[matI]
            if mat is None:
                continue

            vertices, faces, uv, uv_faces, obj_type, material = mat

            is_skybox = obj_type == object_type.SKYBOX
            if not is_skybox:
                texture_index = material.texture_index if hasattr(material, "texture_index") else material
            else:
                texture_index = material
            is_billboard = obj_type == object_type.BILLBOARD

            if is_billboard:
                size = self.vertices_faces_list[matI][6]
                cx_pos = self.vertices_faces_list[matI][0][0]

                hs = size * 0.5
                tr = (cx_pos[0] + cam_right[0]*hs + cam_up[0]*hs,
                    cx_pos[1] + cam_right[1]*hs + cam_up[1]*hs,
                    cx_pos[2] + cam_right[2]*hs + cam_up[2]*hs)
                br = (cx_pos[0] + cam_right[0]*hs - cam_up[0]*hs,
                    cx_pos[1] + cam_right[1]*hs - cam_up[1]*hs,
                    cx_pos[2] + cam_right[2]*hs - cam_up[2]*hs)
                tl = (cx_pos[0] - cam_right[0]*hs + cam_up[0]*hs,
                    cx_pos[1] - cam_right[1]*hs + cam_up[1]*hs,
                    cx_pos[2] - cam_right[2]*hs + cam_up[2]*hs)
                bl = (cx_pos[0] - cam_right[0]*hs - cam_up[0]*hs,
                    cx_pos[1] - cam_right[1]*hs - cam_up[1]*hs,
                    cx_pos[2] - cam_right[2]*hs - cam_up[2]*hs)

                def proj_pt(p):
                    c = self.cam(p, False)
                    if c[2] <= 0.001:
                        return None
                    f = 1.0 / math.tan(fov_rad / 2)
                    return (
                        (c[0] * f / -c[2]) * self.raster_half_h + self.raster_half_w,
                        (c[1] * f / -c[2]) * self.raster_half_h + self.raster_half_h,
                        c[2]
                    )

                pp = [proj_pt(v) for v in [tr, br, tl, bl]]
                if None in pp:
                    continue
                bill_uvs = [(1,1), (1,0), (0,1), (0,0)]
                bill_tris = [(0, 2, 1), (3, 1, 2)]
                bill_uv_faces = [(0, 2, 1), (3, 1, 2)]

                for fi, (f0, f1, f2) in enumerate(bill_tris):
                    p0, p1, p2 = pp[f0], pp[f1], pp[f2]
                    uvi = bill_uv_faces[fi]
                    u0, u1, u2 = bill_uvs[uvi[0]], bill_uvs[uvi[1]], bill_uvs[uvi[2]]
                    all_tris.append(((p0[2], p1[2], p2[2]), (p0, p1, p2), u0, u1, u2, 1.0, False, texture_index))
                continue

            if self.using_obj_filetype_format and not is_billboard:
                unprojected_verticies, *_ = self.vertices_faces_list[matI]

            for faceI in range(len(faces)):
                face = faces[faceI]

                if self.using_obj_filetype_format and not is_billboard:
                    up0 = unprojected_verticies[face[0]]
                    up1 = unprojected_verticies[face[1]]
                    up2 = unprojected_verticies[face[2]]
                    if None in (up0, up1, up2):
                        continue

                uv_face = uv_faces[faceI]
                uv0 = uv[uv_face[0]]
                uv1 = uv[uv_face[1]]
                uv2 = uv[uv_face[2]]

                if self.using_obj_filetype_format:
                    unprojected_normal = self.normalT_camera_space((up0, up1, up2))
                    light_dir = np.array([0, 1, 0])
                    light_m = max(self.lighting_strictness, np.dot(light_dir, np.array(unprojected_normal)))

                    cam0, cam1, cam2 = self.cam(up0, is_skybox), self.cam(up1, is_skybox), self.cam(up2, is_skybox)

                    clipped = self.clip_triangle_near([cam0, cam1, cam2], [uv0, uv1, uv2], near=0.001)

                    for clipped_verts, clipped_uvs in clipped:
                        def proj(v):
                            f = 1.0 / math.tan(fov_rad / 2)
                            return (
                                (v[0] * f / -v[2]) * self.raster_half_h + self.raster_half_w,
                                (v[1] * f / -v[2]) * self.raster_half_h + self.raster_half_h,
                                v[2]
                            )
                        pp0, pp1, pp2 = proj(clipped_verts[0]), proj(clipped_verts[1]), proj(clipped_verts[2])
                        if not is_skybox:
                            if self.is_backface_projected(pp0, pp1, pp2):
                                continue
                        if is_skybox:
                            all_tris.append(((pp0[2], pp1[2], pp2[2]), (pp0, pp1, pp2), clipped_uvs[0], clipped_uvs[1], clipped_uvs[2], light_m, is_skybox, texture_index))
                        else:
                            if not (abs(pp0[2]) > self.render_distance and abs(pp1[2]) > self.render_distance and abs(pp2[2]) > self.render_distance):
                                all_tris.append(((pp0[2], pp1[2], pp2[2]), (pp0, pp1, pp2), clipped_uvs[0], clipped_uvs[1], clipped_uvs[2], light_m, is_skybox, texture_index))
                else:
                    p0 = vertices[face[0]]
                    p1 = vertices[face[1]]
                    p2 = vertices[face[2]]
                    if None in (p0, p1, p2):
                        continue
                    if not (p0[2] < 0 or p1[2] < 0 or p2[2] < 0):
                        all_tris.append(((p0[2], p1[2], p2[2]), (p0, p1, p2), uv0, uv1, uv2, 1.0, is_skybox, texture_index))

        return all_tris

    def mac_rasterize_tris(self, all_tris):
        h, w = self.rasterization_size[1], self.rasterization_size[0]
        color = self._mac_output
        depth = self._mac_depth

        color[:] = 1.0
        depth[:] = np.inf

        for depths, tri, uv1, uv2, uv3, light_m, is_skybox, tri_tex_index in all_tris:
            p0, p1, p2 = tri
            x0, y0 = p0[0], p0[1]
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]

            minx = max(0, int(min(x0, x1, x2)))
            maxx = min(w - 1, int(max(x0, x1, x2)) + 1)
            miny = max(0, int(min(y0, y1, y2)))
            maxy = min(h - 1, int(max(y0, y1, y2)) + 1)

            if maxx < minx or maxy < miny:
                continue

            inv_d0 = 1.0 / depths[0] if depths[0] != 0 else 0.0
            inv_d1 = 1.0 / depths[1] if depths[1] != 0 else 0.0
            inv_d2 = 1.0 / depths[2] if depths[2] != 0 else 0.0

            # Vectorize barycentrics
            yy, xx = np.mgrid[miny:maxy+1, minx:maxx+1]
            px = xx.astype(np.float32) + 0.5
            py = yy.astype(np.float32) + 0.5

            v0x, v0y = x1 - x0, y1 - y0
            v1x, v1y = x2 - x0, y2 - y0
            v2x, v2y = px - x0, py - y0
            denom = v0x * v1y - v0y * v1x
            if abs(denom) < 1e-10:
                continue
            w1 = (v2x * v1y - v2y * v1x) / denom
            w2 = (v0x * v2y - v0y * v2x) / denom
            w0 = 1.0 - w1 - w2

            valid = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not valid.any():
                continue

            d = w0 * depths[0] + w1 * depths[1] + w2 * depths[2]
            d_valid = d < depth[yy, xx]
            final_mask = valid & d_valid
            if not final_mask.any():
                continue

            depth[yy[final_mask], xx[final_mask]] = d[final_mask]

            if self.depth_view_enabled:
                cc = -np.power(2, (-np.abs(d[final_mask]) * 0.75)) + 1
                color[yy[final_mask], xx[final_mask], :3] = cc[:, np.newaxis]
                color[yy[final_mask], xx[final_mask], 3] = 1.0
            elif self.heat_map_enabled:
                t = np.clip(d[final_mask] * 0.35, 0.0, 1.0)
                color[yy[final_mask], xx[final_mask], :3] = np.column_stack([1.0 * t, np.zeros_like(t), 1.0 - t])
                color[yy[final_mask], xx[final_mask], 3] = 1.0
            elif uv1[0] < 0.0 or uv2[0] < 0.0 or uv3[0] < 0.0:
                cc = -np.power(2, (-np.abs(d[final_mask]) * 0.75)) + 1
                color[yy[final_mask], xx[final_mask], :3] = cc[:, np.newaxis]
                color[yy[final_mask], xx[final_mask], 3] = 1.0
            else:
                u_num = uv1[0] * inv_d0 * w0 + uv2[0] * inv_d1 * w1 + uv3[0] * inv_d2 * w2
                v_num = uv1[1] * inv_d0 * w0 + uv2[1] * inv_d1 * w1 + uv3[1] * inv_d2 * w2
                w_num = inv_d0 * w0 + inv_d1 * w1 + inv_d2 * w2
                w_valid = w_num != 0
                final_mask = final_mask & w_valid
                if final_mask.any():
                    u = u_num[final_mask] / w_num[final_mask]
                    v = 1.0 - (v_num[final_mask] / w_num[final_mask])

                    if is_skybox:
                        tex = getattr(self, "_mac_skybox_texture", None)
                    else:
                        layers = getattr(self, "_mac_texture_layers", [])
                        tex = None
                        if tri_tex_index is not None and 0 <= int(tri_tex_index) < len(layers):
                            tex = layers[int(tri_tex_index)]

                    if tex is None:
                        cc = -np.power(2, (-np.abs(d[final_mask]) * 0.75)) + 1
                        color[yy[final_mask], xx[final_mask], :3] = cc[:, np.newaxis]
                        color[yy[final_mask], xx[final_mask], 3] = 1.0
                    else:
                        th, tw = tex.shape[:2]
                        uu = np.clip(u, 0.0, 1.0)
                        vv = np.clip(v, 0.0, 1.0)
                        tx = np.clip((uu * (tw - 1)).astype(np.int32), 0, tw - 1)
                        ty = np.clip((vv * (th - 1)).astype(np.int32), 0, th - 1)
                        texel = tex[ty, tx]
                        if texel.dtype != np.float32:
                            texel = texel.astype(np.float32) / 255.0
                        alpha = texel[:, 3:4]
                        base = color[yy[final_mask], xx[final_mask], :3]
                        rgb = texel[:, :3]
                        blended = base * (1.0 - alpha) + rgb * alpha
                        blended = blended * light_m
                        color[yy[final_mask], xx[final_mask], :3] = blended
                        color[yy[final_mask], xx[final_mask], 3] = 1.0

        return color

    def mac_render_shape_from_obj_format(self, matrix, texture_p):
        if self.render_type != renderer_type.RASTERIZE:
            return ORIG_RENDER_SHAPE_FROM_OBJ_FORMAT(self, matrix, texture_p)

        all_tris = self.mac_collect_tris(matrix)
        if not all_tris:
            return

        color = self.mac_rasterize_tris(all_tris)
        img_uint8 = (np.clip(color, 0.0, 1.0) * 255).astype('uint8')
        surface = pygame.image.frombuffer(img_uint8.tobytes(), (self.rasterization_size[0], self.rasterization_size[1]), 'RGBA')
        if self.upscaled_surface.get_size() != (self.width, self.height):
            self.upscaled_surface = pygame.Surface((self.width, self.height)).convert()
        pygame.transform.scale(surface, (self.width, self.height), self.upscaled_surface)
        self.screen.blit(self.upscaled_surface, (0, 0))
        self._mac_last_surface = surface

    def mac_set_render_type(self, type: renderer_type):
        self.render_type = type
        if type == renderer_type.RASTERIZE:
            self.raster_selected = True
            if self.resizable_window:
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
            else:
                self.screen = pygame.display.set_mode((self.width, self.height))
            self.set_rasterization_size((int(self.width * self.rasterization_mult), int(self.height * self.rasterization_mult)))
        else:
            self.screen = pygame.display.set_mode((self.width, self.height))

    def mac_enable_raster(self):
        if getattr(self, "_mac_raster_enabled", False):
            return

        self._mac_raster_enabled = True
        self._mac_texture_layers = []
        self._mac_textures = {}
        self._mac_skybox_texture = None
        self._mac_output = None
        self._mac_depth = None
        self._mac_last_surface = None

        self.set_render_type = self.mac_set_render_type
        self.set_rasterization_size = self.mac_set_rasterization_size
        self.set_texture_for_raster = self.mac_set_texture_for_raster
        self.add_texture_for_raster = self.mac_add_texture_for_raster
        self.rebuild_textures = self.mac_rebuild_textures
        self.generate_cubemap_skybox = self.mac_generate_cubemap_skybox
        self.generate_cross_type_cubemap_skybox = self.mac_generate_cross_type_cubemap_skybox
        self.toggle_depth_view = self.mac_toggle_depth_view
        self.toggle_heat_map = self.mac_toggle_heat_map
        self.capture_pause_snapshot = self.mac_capture_pause_snapshot
        self.render_shape_from_obj_format = self.mac_render_shape_from_obj_format

        if hasattr(self, "raster_button"):
            self.raster_button.function = lambda: self.set_render_type(renderer_type.RASTERIZE)

ORIG_RENDER_SHAPE_FROM_OBJ_FORMAT = Renderer3D.render_shape_from_obj_format

def main():
    renderer = Renderer3D()
    renderer.run()


if __name__ == "__main__":
    main()