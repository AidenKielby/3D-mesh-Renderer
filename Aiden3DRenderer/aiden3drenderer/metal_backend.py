from .renderer_backend import RenderBackend
from .object_type import object_type
from .renderer import renderer_type

import numpy as np
import pygame
from PIL import Image

try:
    import objc
    from Metal import (
        MTLCreateSystemDefaultDevice,
        MTLTextureDescriptor,
        MTLTextureType2DArray,
        MTLPixelFormatRGBA32Float,
        MTLPixelFormatRGBA8Unorm,
        MTLTextureUsageShaderRead,
        MTLTextureUsageShaderWrite,
        MTLStorageModeShared,
        MTLResourceStorageModeShared,
        MTLSamplerDescriptor,
        MTLSamplerMinMagFilterNearest,
        MTLSamplerAddressModeClampToEdge,
        MTLRegionMake2D,
        MTLSizeMake,
    )
except Exception:
    objc = None
    MTLCreateSystemDefaultDevice = None

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
    ("pos", "f4", (3, 2)),
    ("depths", "f4", 3),
    ("light_mult", "f4", 1),
    ("uv", "f4", (3, 2)),
    ("is_skybox", "f4", 1),
    ("texture_index", "f4", 1),
    ("pad1", "f4", 1),
    ("pad2", "f4", 1),
])

class MetalBackend(RenderBackend):
    def __init__(self):
        self.renderer = None
        self.device = None
        self.command_queue = None
        self.pipeline_state = None
        self.sampler_state = None
        self.metal_ready = False
        self.use_cpu_fallback = False
        self.output_tex = None
        self.alt = None
        self.texture = None
        self.skybox_texture = None
        self.skybox_image = None
        self.default_texture = None
        self.default_skybox_texture = None
        self.last_surface = None

        self.texture_layers = []
        self.textures = {}
        self.last_texture_array = None
        self.last_size = 1

        self.rasterization_size = (0, 0)
        self.raster_half_w = 0
        self.raster_half_h = 0
        self.render_distance = 20
        self.output_clear_rgba = None
        self.upscaled_surface = None
        self.depth_view_enabled = False
        self.heat_map_enabled = False
        self.texture_path = None
        self.skybox_texture_path = None
        self.cpu_output = None
        self.cpu_depth = None

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

        self.device = MTLCreateSystemDefaultDevice() if MTLCreateSystemDefaultDevice else None
        if self.device is None:
            self.use_cpu_fallback = True
            self.metal_ready = False
        else:
            try:
                self.command_queue = self.device.newCommandQueue()
                self.pipeline_state = self.build_pipeline()
                self.sampler_state = self.create_sampler_state()
                self.metal_ready = True
            except Exception as exc:
                print(f"Metal setup failed, using CPU fallback: {exc}")
                self.metal_ready = False
                self.use_cpu_fallback = True

        self.set_rasterization_size((width // 2, height // 2))
        if self.device is not None:
            self.default_texture = self.create_texture_array(1, 1, 1)
            self.default_skybox_texture = self.create_texture_2d(1, 1)
            white = np.array([[[255, 255, 255, 255]]], dtype="u1")
            region = MTLRegionMake2D(0, 0, 1, 1)
            self.default_texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage_(
                region,
                0,
                0,
                white.tobytes(),
                4,
                4,
            )
            self.default_skybox_texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
                region,
                0,
                white.tobytes(),
                4,
            )
        self.last_surface = None
        self.sync_renderer_state()

    def sync_renderer_state(self):
        if not self.renderer:
            return
        r = self.renderer
        r.ctx = None
        r.compute_shader_container = None
        r.compute_shader = None
        r.tri_buffer = None
        r.output_tex = self.output_tex
        r.alt = self.alt
        r.rasterization_size = self.rasterization_size
        r.raster_half_w = self.raster_half_w
        r.raster_half_h = self.raster_half_h
        r.disable_finish_call = False
        r.last_present_tex = self.output_tex
        r._output_clear_rgba = self.output_clear_rgba
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

    def build_pipeline(self):
        library, error = self.device.newLibraryWithSource_options_error_(metal_compute_shader, None, None)
        if error:
            raise RuntimeError(str(error))
        fn = library.newFunctionWithName_("main0")
        pipeline, error = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if error:
            raise RuntimeError(str(error))
        return pipeline

    def create_sampler_state(self):
        desc = MTLSamplerDescriptor.alloc().init()
        desc.setMinFilter_(MTLSamplerMinMagFilterNearest)
        desc.setMagFilter_(MTLSamplerMinMagFilterNearest)
        desc.setSAddressMode_(MTLSamplerAddressModeClampToEdge)
        desc.setTAddressMode_(MTLSamplerAddressModeClampToEdge)
        return self.device.newSamplerStateWithDescriptor_(desc)

    def create_output_texture(self, width, height):
        desc = MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            MTLPixelFormatRGBA32Float,
            int(width),
            int(height),
            False,
        )
        desc.setStorageMode_(MTLStorageModeShared)
        desc.setUsage_(MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite)
        return self.device.newTextureWithDescriptor_(desc)

    def create_texture_array(self, width, height, layers):
        desc = MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            MTLPixelFormatRGBA8Unorm,
            int(width),
            int(height),
            False,
        )
        desc.setTextureType_(MTLTextureType2DArray)
        desc.setArrayLength_(int(layers))
        desc.setStorageMode_(MTLStorageModeShared)
        desc.setUsage_(MTLTextureUsageShaderRead)
        return self.device.newTextureWithDescriptor_(desc)

    def create_texture_2d(self, width, height):
        desc = MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            MTLPixelFormatRGBA8Unorm,
            int(width),
            int(height),
            False,
        )
        desc.setStorageMode_(MTLStorageModeShared)
        desc.setUsage_(MTLTextureUsageShaderRead)
        return self.device.newTextureWithDescriptor_(desc)

    def set_render_type(self, type: renderer_type, screen=None):
        self.render_type = type
        if type == renderer_type.RASTERIZE:
            self.raster_selected = True
            if self.resizable_window:
                self.renderer.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
            else:
                self.renderer.screen = pygame.display.set_mode((self.width, self.height))
            scale = getattr(self.renderer, "rasterization_mult", 0.5)
            self.set_rasterization_size((int(self.width * scale), int(self.height * scale)))
        else:
            self.renderer.screen = pygame.display.set_mode((self.width, self.height))
        self.sync_renderer_state()

    def set_rasterization_size(self, size: tuple[int, int]):
        width, height = size
        width = width + (16 - width % 16) % 16
        height = height + (16 - height % 16) % 16
        self.rasterization_size = (width, height)
        self.raster_half_w = width // 2
        self.raster_half_h = height // 2

        if self.device is not None:
            self.output_tex = self.create_output_texture(width, height)
            self.alt = self.create_output_texture(width, height)

        self.output_clear_rgba = np.ones((height, width, 4), dtype=np.float32)
        self.cpu_output = np.ones((height, width, 4), dtype=np.float32)
        self.cpu_depth = np.full((height, width), np.inf, dtype=np.float32)

        if self.renderer:
            self.upscaled_surface = pygame.Surface((self.renderer.width, self.renderer.height))
        self.sync_renderer_state()

    def set_texture_for_raster(self, img_path):
        if img_path is None:
            return None

        self.texture_path = img_path
        img = Image.open(self.texture_path).convert("RGBA")
        img_data = np.array(img, dtype="u1")

        self.texture_layers.clear()
        self.texture_layers.append(img_data)
        self.last_texture_array = img_data
        self.last_size = 1

        if self.metal_ready:
            self.texture = self.create_texture_array(img.size[0], img.size[1], self.last_size)
            region = MTLRegionMake2D(0, 0, img.size[0], img.size[1])
            bytes_per_row = img.size[0] * 4
            bytes_per_image = bytes_per_row * img.size[1]
            self.texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage_(
                region,
                0,
                0,
                img_data.tobytes(),
                bytes_per_row,
                bytes_per_image,
            )

        self.textures.clear()
        self.textures[img_path] = 0
        self.sync_renderer_state()
        return 0

    def add_texture_for_raster(self, img_path):
        if img_path is None:
            return None

        if not self.texture_layers:
            return self.set_texture_for_raster(img_path)

        if img_path in self.textures:
            return self.textures[img_path]

        img = Image.open(img_path).convert("RGBA")
        img_data = np.array(img, dtype="u1")

        base_h, base_w, _ = self.texture_layers[0].shape
        h, w, _ = img_data.shape
        if (h, w) != (base_h, base_w):
            img = img.resize((base_w, base_h), Image.Resampling.NEAREST)
            img_data = np.array(img, dtype="u1")

        self.texture_layers.append(img_data)
        self.last_size = len(self.texture_layers)
        self.last_texture_array = img_data

        if self.metal_ready:
            self.texture = self.create_texture_array(base_w, base_h, self.last_size)
            region = MTLRegionMake2D(0, 0, base_w, base_h)
            bytes_per_row = base_w * 4
            bytes_per_image = bytes_per_row * base_h
            for idx, layer in enumerate(self.texture_layers):
                self.texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage_(
                    region,
                    0,
                    idx,
                    layer.tobytes(),
                    bytes_per_row,
                    bytes_per_image,
                )

        idx = len(self.texture_layers) - 1
        self.textures[img_path] = idx
        self.sync_renderer_state()
        return idx

    def rebuild_textures(self):
        if not self.texture_layers or not self.metal_ready:
            return
        h, w = self.texture_layers[0].shape[:2]
        self.texture = self.create_texture_array(w, h, len(self.texture_layers))
        region = MTLRegionMake2D(0, 0, w, h)
        bytes_per_row = w * 4
        bytes_per_image = bytes_per_row * h
        for idx, layer in enumerate(self.texture_layers):
            self.texture.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage_(
                region,
                0,
                idx,
                layer.tobytes(),
                bytes_per_row,
                bytes_per_image,
            )
        if self.skybox_texture_path:
            self.generate_cross_type_cubemap_skybox(20, self.skybox_texture_path)
        self.sync_renderer_state()

    def rebuild_shaders(self):
        return

    def toggle_depth_view(self, b: bool):
        self.depth_view_enabled = b

    def toggle_heat_map(self, b: bool):
        self.heat_map_enabled = b

    def rasterize(self, all_tris):
        if not all_tris:
            return
        if self.metal_ready:
            try:
                self.rasterize_metal(all_tris)
                return
            except Exception as exc:
                print(f"Metal rasterize failed, using CPU fallback: {exc}")
                self.metal_ready = False
                self.use_cpu_fallback = True
        self.rasterize_cpu(all_tris)

    def rasterize_metal(self, all_tris):
        data = np.zeros(len(all_tris), dtype=tri_dtype)
        for i, (depths, tri, uv1, uv2, uv3, light_m, is_skybox, tri_tex_index) in enumerate(all_tris):
            p0, p1, p2 = tri
            data[i]["pos"] = ((p0[0], p0[1]), (p1[0], p1[1]), (p2[0], p2[1]))
            data[i]["depths"] = depths

            if tri_tex_index is None:
                tri_tex_index = 0

            if not is_skybox:
                if None in (uv1, uv2, uv3) or self.texture is None:
                    data[i]["uv"] = ((-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0))
                    data[i]["light_mult"] = 1
                    data[i]["is_skybox"] = 0
                    data[i]["texture_index"] = tri_tex_index
                else:
                    data[i]["uv"] = (uv1, uv2, uv3)
                    data[i]["light_mult"] = light_m
                    data[i]["is_skybox"] = 0
                    data[i]["texture_index"] = tri_tex_index
            else:
                if None in (uv1, uv2, uv3) or self.skybox_texture is None:
                    data[i]["uv"] = ((-1.0, -1.0), (-1.0, -1.0), (-1.0, -1.0))
                    data[i]["light_mult"] = 1
                    data[i]["is_skybox"] = 0
                else:
                    data[i]["uv"] = (uv1, uv2, uv3)
                    data[i]["light_mult"] = 1
                    data[i]["is_skybox"] = 1

        if self.output_tex is None:
            return

        region = MTLRegionMake2D(0, 0, self.rasterization_size[0], self.rasterization_size[1])
        bytes_per_row = self.rasterization_size[0] * 4 * 4
        self.output_tex.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            region,
            0,
            self.output_clear_rgba.tobytes(),
            bytes_per_row,
        )

        tri_bytes = data.tobytes()
        tri_buffer = self.device.newBufferWithBytes_length_options_(
            tri_bytes,
            len(tri_bytes),
            MTLResourceStorageModeShared,
        )
        globals_data = np.array([
            len(all_tris),
            1 if self.depth_view_enabled else 0,
            1 if self.heat_map_enabled else 0,
            0,
        ], dtype=np.uint32)
        globals_buffer = self.device.newBufferWithBytes_length_options_(
            globals_data.tobytes(),
            globals_data.nbytes,
            MTLResourceStorageModeShared,
        )
        size_data = np.array([0, len(tri_bytes)], dtype=np.uint32)
        size_buffer = self.device.newBufferWithBytes_length_options_(
            size_data.tobytes(),
            size_data.nbytes,
            MTLResourceStorageModeShared,
        )

        command_buffer = self.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pipeline_state)
        encoder.setBuffer_offset_atIndex_(globals_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(tri_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(size_buffer, 0, 25)

        encoder.setTexture_atIndex_(self.output_tex, 0)
        encoder.setTexture_atIndex_(self.skybox_texture or self.default_skybox_texture, 1)
        encoder.setTexture_atIndex_(self.texture or self.default_texture, 2)
        encoder.setSamplerState_atIndex_(self.sampler_state, 0)
        encoder.setSamplerState_atIndex_(self.sampler_state, 1)

        tg = MTLSizeMake(16, 16, 1)
        groups_x = (self.rasterization_size[0] + 15) // 16
        groups_y = (self.rasterization_size[1] + 15) // 16
        grid = MTLSizeMake(groups_x, groups_y, 1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid, tg)
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        self.last_present_tex = self.output_tex
        self.run_compute_shaders(len(all_tris))

        tex_to_present = self.last_present_tex or self.output_tex
        raw = np.empty((self.rasterization_size[1], self.rasterization_size[0], 4), dtype=np.float32)
        tex_to_present.getBytes_bytesPerRow_fromRegion_mipmapLevel_(
            raw,
            bytes_per_row,
            region,
            0,
        )

        img_uint8 = (np.clip(raw, 0.0, 1.0) * 255).astype("uint8")
        surface = pygame.image.frombuffer(
            img_uint8.tobytes(),
            (self.rasterization_size[0], self.rasterization_size[1]),
            "BGRA",
        )
        if self.upscaled_surface.get_size() != (self.renderer.width, self.renderer.height):
            self.upscaled_surface = pygame.Surface((self.renderer.width, self.renderer.height)).convert()
        pygame.transform.scale(surface, (self.renderer.width, self.renderer.height), self.upscaled_surface)
        self.renderer.screen.blit(self.upscaled_surface, (0, 0))
        self.last_surface = surface
        self.sync_renderer_state()

    def rasterize_cpu(self, all_tris):
        h, w = self.rasterization_size[1], self.rasterization_size[0]
        color = self.cpu_output
        depth = self.cpu_depth

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

            yy, xx = np.mgrid[miny:maxy + 1, minx:maxx + 1]
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
                        tex = self.skybox_image
                    else:
                        tex = None
                        if tri_tex_index is not None and 0 <= int(tri_tex_index) < len(self.texture_layers):
                            tex = self.texture_layers[int(tri_tex_index)]

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

        img_uint8 = (np.clip(color, 0.0, 1.0) * 255).astype("uint8")
        surface = pygame.image.frombuffer(
            img_uint8.tobytes(),
            (self.rasterization_size[0], self.rasterization_size[1]),
            "RGBA",
        )
        if self.upscaled_surface.get_size() != (self.renderer.width, self.renderer.height):
            self.upscaled_surface = pygame.Surface((self.renderer.width, self.renderer.height)).convert()
        pygame.transform.scale(surface, (self.renderer.width, self.renderer.height), self.upscaled_surface)
        self.renderer.screen.blit(self.upscaled_surface, (0, 0))
        self.last_surface = surface
        self.sync_renderer_state()

    def capture_pause_snapshot(self):
        if self.last_surface is None:
            return
        if self.upscaled_surface.get_size() != (self.renderer.width, self.renderer.height):
            self.upscaled_surface = pygame.Surface((self.renderer.width, self.renderer.height)).convert()
        pygame.transform.scale(self.last_surface, (self.renderer.width, self.renderer.height), self.upscaled_surface)
        self.renderer.pause_img = self.last_surface
        self.sync_renderer_state()

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

        self.skybox_texture_path = texture_path
        img = Image.open(self.skybox_texture_path).convert("RGBA")
        self.skybox_image = np.array(img, dtype="u1")

        if self.metal_ready:
            self.skybox_texture = self.create_texture_2d(img.size[0], img.size[1])
            region = MTLRegionMake2D(0, 0, img.size[0], img.size[1])
            bytes_per_row = img.size[0] * 4
            self.skybox_texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
                region,
                0,
                self.skybox_image.tobytes(),
                bytes_per_row,
            )

        self.renderer.vertices_faces_list.append([verts.tolist(), faces, uvs, uv_faces, object_type.SKYBOX, 0])
        self.sync_renderer_state()

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

    def run_compute_shaders(self, tri_count):
        if not self.shaders:
            return 0

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
                        shader.compute_shader[uname] = val
                    except Exception:
                        pass

            if last_output_binding == 1:
                src_tex = self.alt
                dest_tex = self.output_tex
            else:
                src_tex = self.output_tex
                dest_tex = self.alt

            if src_tex is None or dest_tex is None:
                continue

            shader.textures["srcTex"] = src_tex
            shader.textures["destTex"] = dest_tex

            try:
                shader.compute_shader["tri_count"] = int(tri_count)
            except Exception:
                pass

            groups_x = max(1, (self.rasterization_size[0] + 15) // 16)
            groups_y = max(1, (self.rasterization_size[1] + 15) // 16)
            try:
                shader.compute_shader.run(groups_x, groups_y, 1)
            except Exception:
                pass

            self.last_present_tex = dest_tex
            last_output_binding = (last_output_binding + 1) % 2

        return last_output_binding
