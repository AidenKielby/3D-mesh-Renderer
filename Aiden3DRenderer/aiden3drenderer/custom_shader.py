import sys
import re
import struct
import ctypes
import numpy as np
from PIL import Image

if sys.platform != "darwin":
    import moderngl
    MTLCreateSystemDefaultDevice = None
    MTLTextureDescriptor = None
    MTLTextureType2DArray = None
    MTLPixelFormatRGBA32Float = None
    MTLPixelFormatRGBA8Unorm = None
    MTLTextureUsageShaderRead = None
    MTLTextureUsageShaderWrite = None
    MTLStorageModeShared = None
    MTLResourceStorageModeShared = None
    MTLSamplerDescriptor = None
    MTLSamplerMinMagFilterNearest = None
    MTLSamplerAddressModeClampToEdge = None
    MTLRegionMake2D = None
    MTLSizeMake = None
else:
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

glsl_type_to_bytes = {
    "float": 4,
    "vec2" : 8,
    "vec3": 16, #padded for std430
    "vec4": 16
}

metal_type_layout = {
    "float": (4, 4),
    "int": (4, 4),
    "uint": (4, 4),
    "float2": (8, 8),
    "int2": (8, 8),
    "uint2": (8, 8),
    "float3": (16, 16),
    "int3": (16, 16),
    "uint3": (16, 16),
    "float4": (16, 16),
    "int4": (16, 16),
    "uint4": (16, 16),
}

# the layout for dstTex MUST be binding = 1!
# so that would look like this: "layout(rgba32f, binding = 1) uniform image2D destTex;"
# Also! Binding for srcTex is to be not defined or set to 0!
class MetalUniformProxy:
    def __init__(self, shader, name: str):
        self.shader = shader
        self.name = name

    @property
    def value(self):
        return self.shader.uniform_values.get(self.name)

    @value.setter
    def value(self, val):
        self.shader.write_to_uniform(self.name, val)


class MetalComputeShader:
    def __init__(self, shader):
        self.shader = shader

    def __getitem__(self, key: str):
        return MetalUniformProxy(self.shader, key)

    def __setitem__(self, key: str, value):
        self.shader.write_to_uniform(key, value)

    def run(self, x, y, z):
        self.shader.dispatch(x, y, z)


class CustomShader:
    def __init__(self, shader_code: str, context=None):
        self.shader_code = shader_code
        self.buffer_objects = {}
        self.textures = {}
        self.texture_info = []
        self.is_metal = sys.platform == "darwin"

        if not self.is_metal:
            self.ctx = context or moderngl.create_context(standalone=True)
            self.compute_shader = self.ctx.compute_shader(shader_code)
            self.buffers = self.get_buffers()
            self.uniforms = self.get_uniforms()
            self.buffer_sizes = {}
            return

        self.ctx = None
        self.device = MTLCreateSystemDefaultDevice() if MTLCreateSystemDefaultDevice else None
        if self.device is None:
            raise RuntimeError("Metal is not available on this system.")

        self.command_queue = self.device.newCommandQueue()
        self.pipeline_state = self.build_pipeline()
        self.sampler_state = self.create_sampler_state()
        self.threadgroup_size = self.parse_threadgroup_size(shader_code)

        parsed = self.parse_metal_resources(shader_code)
        self.buffer_bindings = parsed[0]
        self.texture_bindings = parsed[1]
        self.sampler_bindings = parsed[2]
        self.uniform_layout = parsed[3]
        self.uniform_binding = parsed[4]
        self.uniform_var_name = parsed[5]

        self.uniform_values = {}
        self.uniform_buffer = None
        self.uniform_buffer_size = self.uniform_layout.get("__size__", 0)
        self.uniform_dirty = True
        self.buffer_sizes = {}

        self.default_texture = self.create_default_texture()
        self.compute_shader = MetalComputeShader(self)
        self.buffers = []
        self.uniforms = self.get_uniforms()

    def get_buffers(self):
        if self.is_metal:
            return []
        bufs = []
        code_l = self.shader_code.split("\n")
        i = 0
        while i < len(code_l):
            line = code_l[i].split('//')[0].strip() 
            if not line:
                i += 1
                continue

            match = re.search(r"binding\s*=\s*(\d+)", line)
            binding = int(match.group(1)) if match else 0

            line = re.sub(r'layout\s*\(.*?\)\s*', '', line)

            if line.startswith("buffer "):
                tokens = line.split()
                buffer_name = tokens[1]

                i += 1
                while i < len(code_l):
                    inner = code_l[i].split('//')[0].strip()
                    if inner and inner != '}':
                        inner_tokens = inner.split()
                        var_type = inner_tokens[0]
                        var_name = inner_tokens[1].replace(';','')
                        is_list = '[' in inner_tokens[1]
                        bufs.append([buffer_name, var_type, var_name, is_list, binding])
                        break
                    i += 1
            i += 1
        return bufs

    def add_texture(self, texture_path: str, location: int, texture_name: str, verbose: bool = False):
        if self.is_metal:
            try:
                image = Image.open(texture_path)
            except FileNotFoundError:
                raise FileNotFoundError("Error: Image file not found.")

            self.texture_info.append((texture_path, location, texture_name))
            image = image.convert("RGBA")
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = np.array(image, dtype="u1")
            w, h = image.size

            texture = self.create_texture(w, h)
            region = MTLRegionMake2D(0, 0, w, h)
            bytes_per_row = w * 4
            texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
                region,
                0,
                img_data.tobytes(),
                bytes_per_row,
            )

            self.textures[texture_name] = texture
            if texture_name not in self.texture_bindings:
                self.texture_bindings[texture_name] = int(location)
            if verbose:
                print(f"texture: {texture_name} using binding {location}")
            return texture

        try:
            image = Image.open(texture_path)
        except FileNotFoundError:
            raise FileNotFoundError("Error: Image file not found.")
        
        self.texture_info.append((texture_path, location, texture_name))
        image = image.convert('RGBA')
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image_data = image.tobytes('raw', 'RGBA')
        size = image.size

        texture = self.ctx.texture(
            size=size,
            components=4,
            data=image_data
        )

        # disable wrap and use nearest sampling for exact framebuffer copies
        try:
            texture.repeat_x = False
            texture.repeat_y = False
        except Exception:
            pass

        try:
            texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        except Exception:
            pass

        self.textures[texture_name] = texture

        texture.use(location=location)
        self.compute_shader[texture_name] = location
        if verbose:
            print(f"texture: {texture_name} using binding {location}")

        return texture
    
    

    def get_uniforms(self):
        if self.is_metal:
            unis = []
            binding = self.uniform_binding if self.uniform_binding is not None else 0
            for name, info in self.uniform_layout.items():
                if name == "__size__":
                    continue
                unis.append([info[1], name, binding])
            return unis
        unis = []
        code_l = self.shader_code.split("\n")
        for line in code_l:
            line = line.split('//')[0].strip()
            if not line:
                continue
            
            match = re.search(r"binding\s*=\s*(\d+)", line)
            binding = int(match.group(1)) if match else 0

            line = re.sub(r'layout\s*\(.*?\)\s*', '', line)

            if "uniform " in line:
                tokens = line.split()
                idx = tokens.index("uniform")
                type_name = tokens[idx+1]
                var_name = tokens[idx+2].replace(';','')
                unis.append([type_name, var_name, binding])
        return unis
    
    def set_buffer(self, buffer_name: str, element_count, element_size: int = None):
        if self.is_metal:
            binding = self.buffer_bindings.get(buffer_name)
            if binding is None:
                raise NameError(f"Buffer name '{buffer_name}' not found in Metal bindings!")
            if element_size is not None:
                size = int(element_count) * int(element_size)
            else:
                size = int(element_count) * 4
            buf = self.device.newBufferWithLength_options_(size, MTLResourceStorageModeShared)
            self.buffer_objects[buffer_name] = buf
            self.buffer_sizes[buffer_name] = size
            return
        for b in self.buffers:
            name = b[0]
            if name == buffer_name:
                var_type = b[1]
                binding = b[4]
                if element_size is not None:
                    size = element_count * element_size
                else:
                    size = element_count * glsl_type_to_bytes.get(var_type, 4)  # default to 4
                buf = self.ctx.buffer(reserve=size)
                buf.bind_to_storage_buffer(binding)
                self.buffer_objects[buffer_name] = buf
                return
        raise NameError(f"Buffer name '{buffer_name}' not found!")

    
    def write_to_buffer(self, buffer_name: str, data_bytes):
        if self.is_metal:
            if buffer_name not in self.buffer_bindings:
                raise NameError(f"Buffer name '{buffer_name}' not found in Metal bindings!")
            data = data_bytes if isinstance(data_bytes, (bytes, bytearray)) else bytes(data_bytes)
            buf = self.device.newBufferWithBytes_length_options_(
                data,
                len(data),
                MTLResourceStorageModeShared,
            )
            self.buffer_objects[buffer_name] = buf
            self.buffer_sizes[buffer_name] = len(data)
            return
        buf = self.buffer_objects.get(buffer_name)
        if not buf:
            raise NameError(f"Buffer name '{buffer_name}' not allocated yet!")
        buf.write(data_bytes)
    
    def write_to_uniform(self, uniform_name: str, data_bytes):
        if self.is_metal:
            if uniform_name not in self.uniform_layout:
                raise NameError(f"Uniform '{uniform_name}' not found!")
            self.uniform_values[uniform_name] = data_bytes
            self.uniform_dirty = True
            return
        for u in self.uniforms:
            # uniforms stored as [type_name, var_name, binding]
            name = u[1]
            if name == uniform_name:
                self.compute_shader[uniform_name] = data_bytes
                return
        raise NameError(f"Uniform '{uniform_name}' not found!")
    
    def read_from_buffer(self, buffer_name: str, num_elements, element_type='vec3'):
        if self.is_metal:
            buf = self.buffer_objects.get(buffer_name)
            if buf is None:
                raise NameError(f"Buffer '{buffer_name}' not found!")

            stride_map = {
                "float": 1,
                "vec2": 2,
                "vec3": 4,
                "vec4": 4,
            }
            stride = stride_map.get(element_type, 1)
            size = int(num_elements) * int(stride) * 4
            size = self.buffer_sizes.get(buffer_name, size)

            ptr = buf.contents()
            if hasattr(ptr, "as_buffer"):
                data_bytes = ptr.as_buffer(size)
                data = bytes(data_bytes)
            else:
                data = ctypes.string_at(int(ptr), size)

            data_array = np.frombuffer(data, dtype="f4").reshape(num_elements, stride)
            if element_type == "vec3":
                data_array = data_array[:, :3]
            elif element_type == "vec2":
                data_array = data_array[:, :2]
            return data_array
        if buffer_name not in self.buffer_objects:
            raise NameError(f"Buffer '{buffer_name}' not found!")

        data_bytes = self.buffer_objects[buffer_name].read()
        
        stride_map = {
            "float": 1,
            "vec2": 2,
            "vec3": 4, 
            "vec4": 4
        }

        stride = stride_map.get(element_type, 1)
        data_array = np.frombuffer(data_bytes, dtype='f4').reshape(num_elements, stride)

        if element_type == 'vec3':
            data_array = data_array[:, :3]
        elif element_type == 'vec2':
            data_array = data_array[:, :2]

        return data_array

    def build_pipeline(self):
        library, error = self.device.newLibraryWithSource_options_error_(self.shader_code, None, None)
        if error:
            raise RuntimeError(str(error))
        fn = library.newFunctionWithName_("main0")
        if fn is None:
            raise RuntimeError("Metal shader must define a kernel named 'main0'.")
        pipeline, error = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if error:
            raise RuntimeError(str(error))
        return pipeline

    def create_sampler_state(self):
        desc = MTLSamplerDescriptor.alloc().init()
        desc.minFilter = MTLSamplerMinMagFilterNearest
        desc.magFilter = MTLSamplerMinMagFilterNearest
        desc.sAddressMode = MTLSamplerAddressModeClampToEdge
        desc.tAddressMode = MTLSamplerAddressModeClampToEdge
        return self.device.newSamplerStateWithDescriptor_(desc)

    def create_texture(self, width, height):
        desc = MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            MTLPixelFormatRGBA8Unorm,
            int(width),
            int(height),
            False,
        )
        desc.storageMode = MTLStorageModeShared
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite
        return self.device.newTextureWithDescriptor_(desc)

    def create_default_texture(self):
        texture = self.create_texture(1, 1)
        region = MTLRegionMake2D(0, 0, 1, 1)
        white = np.array([[[255, 255, 255, 255]]], dtype="u1")
        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            region,
            0,
            white.tobytes(),
            4,
        )
        return texture

    def parse_threadgroup_size(self, code: str):
        match = re.search(r"gl_WorkGroupSize\s*\[\[maybe_unused\]\]\s*=\s*uint3\((\d+)u?,\s*(\d+)u?,\s*(\d+)u?\)", code)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return (16, 16, 1)

    def parse_metal_resources(self, code: str):
        clean = re.sub(r"//.*", "", code)

        struct_defs = {}
        struct_re = re.compile(r"struct\s+(\w+)\s*\{([^}]*)\};", re.S)
        for name, body in struct_re.findall(clean):
            fields = []
            for line in body.split(";"):
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                if len(tokens) < 2:
                    continue
                field_type = tokens[0]
                field_name = tokens[1].replace(";", "")
                if "[" in field_name:
                    field_name = field_name.split("[")[0]
                fields.append((field_type, field_name))
            struct_defs[name] = fields

        buffer_bindings = {}
        for name, binding in re.findall(r"\b(\w+)\s*\[\[buffer\((\d+)\)\]\]", clean):
            buffer_bindings[name] = int(binding)

        texture_bindings = {}
        for name, binding in re.findall(r"\b(\w+)\s*\[\[texture\((\d+)\)\]\]", clean):
            texture_bindings[name] = int(binding)

        sampler_bindings = {}
        for name, binding in re.findall(r"\b(\w+)\s*\[\[sampler\((\d+)\)\]\]", clean):
            sampler_bindings[name] = int(binding)

        uniform_layout = {"__size__": 0}
        uniform_binding = None
        uniform_var_name = None

        match = re.search(r"constant\s+(\w+)\s*&\s*(\w+)\s*\[\[buffer\((\d+)\)\]\]", clean)
        if match:
            struct_name, var_name, binding = match.group(1), match.group(2), int(match.group(3))
            uniform_binding = binding
            uniform_var_name = var_name
            fields = struct_defs.get(struct_name, [])
            offset = 0
            for field_type, field_name in fields:
                size_align = metal_type_layout.get(field_type)
                if size_align is None:
                    continue
                size, align = size_align
                if offset % align != 0:
                    offset += align - (offset % align)
                uniform_layout[field_name] = (offset, field_type)
                offset += size
            if offset % 16 != 0:
                offset += 16 - (offset % 16)
            uniform_layout["__size__"] = offset

        return buffer_bindings, texture_bindings, sampler_bindings, uniform_layout, uniform_binding, uniform_var_name

    def build_uniform_buffer(self):
        if not self.uniform_layout or self.uniform_layout.get("__size__", 0) == 0:
            return None
        size = self.uniform_layout["__size__"]
        data = bytearray(size)
        for name, info in self.uniform_layout.items():
            if name == "__size__":
                continue
            offset, type_name = info
            if name not in self.uniform_values:
                continue
            raw = self.uniform_values[name]
            packed = self.pack_uniform_value(type_name, raw)
            if packed is None:
                continue
            data[offset:offset + len(packed)] = packed
        self.uniform_dirty = False
        return self.device.newBufferWithBytes_length_options_(bytes(data), len(data), MTLResourceStorageModeShared)

    def pack_uniform_value(self, type_name: str, value):
        if type_name in ("float", "half"):
            return struct.pack("<f", float(value))
        if type_name == "int":
            return struct.pack("<i", int(value))
        if type_name == "uint":
            return struct.pack("<I", int(value))
        if type_name in ("float2", "float3", "float4"):
            vals = list(value) if isinstance(value, (list, tuple, np.ndarray)) else [float(value)]
            count = int(type_name[-1])
            while len(vals) < count:
                vals.append(0.0)
            vals = vals[:count]
            return struct.pack("<" + "f" * count, *[float(v) for v in vals])
        if type_name in ("int2", "int3", "int4"):
            vals = list(value) if isinstance(value, (list, tuple, np.ndarray)) else [int(value)]
            count = int(type_name[-1])
            while len(vals) < count:
                vals.append(0)
            vals = vals[:count]
            return struct.pack("<" + "i" * count, *[int(v) for v in vals])
        if type_name in ("uint2", "uint3", "uint4"):
            vals = list(value) if isinstance(value, (list, tuple, np.ndarray)) else [int(value)]
            count = int(type_name[-1])
            while len(vals) < count:
                vals.append(0)
            vals = vals[:count]
            return struct.pack("<" + "I" * count, *[int(v) for v in vals])
        return None

    def dispatch(self, groups_x, groups_y, groups_z):
        if not self.is_metal:
            return

        command_buffer = self.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pipeline_state)

        if self.uniform_binding is not None:
            uniform_buffer = self.build_uniform_buffer()
            if uniform_buffer is not None:
                encoder.setBuffer_offset_atIndex_(uniform_buffer, 0, self.uniform_binding)

        for name, binding in self.buffer_bindings.items():
            if name == self.uniform_var_name:
                continue
            buf = self.buffer_objects.get(name)
            if buf is not None:
                encoder.setBuffer_offset_atIndex_(buf, 0, binding)

        for name, binding in self.texture_bindings.items():
            tex = self.textures.get(name, self.default_texture)
            encoder.setTexture_atIndex_(tex, binding)

        for name, binding in self.sampler_bindings.items():
            encoder.setSamplerState_atIndex_(self.sampler_state, binding)

        tg = MTLSizeMake(self.threadgroup_size[0], self.threadgroup_size[1], self.threadgroup_size[2])
        grid = MTLSizeMake(int(groups_x), int(groups_y), int(groups_z))
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid, tg)
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()