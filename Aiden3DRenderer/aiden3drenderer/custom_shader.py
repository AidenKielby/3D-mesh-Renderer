import sys
if sys.platform != "darwin":
    import moderngl
    import re
    import numpy as np
    from PIL import Image

glsl_type_to_bytes = {
    "float": 4,
    "vec2" : 8,
    "vec3": 16, #padded for std430
    "vec4": 16
}

# the layout for dstTex MUST be binding = 1!
# so that would look like this: "layout(rgba32f, binding = 1) uniform image2D destTex;"
# Also! Binding for srcTex is to be not defined or set to 0!
class CustomShader:
    def __init__(self, shader_code: str, context=None):
        self.shader_code = shader_code

        self.ctx = context or moderngl.create_context(standalone=True)
        
        self.compute_shader = self.ctx.compute_shader(shader_code)

        self.buffers = self.get_buffers()
        self.uniforms = self.get_uniforms()

        self.buffer_objects = {}
        self.textures = {}
        self.texture_info = []

    def get_buffers(self):
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
        buf = self.buffer_objects.get(buffer_name)
        if not buf:
            raise NameError(f"Buffer name '{buffer_name}' not allocated yet!")
        buf.write(data_bytes)
    
    def write_to_uniform(self, uniform_name: str, data_bytes):
        for u in self.uniforms:
            # uniforms stored as [type_name, var_name, binding]
            name = u[1]
            if name == uniform_name:
                self.compute_shader[uniform_name] = data_bytes
                return
        raise NameError(f"Uniform '{uniform_name}' not found!")
    
    def read_from_buffer(self, buffer_name: str, num_elements, element_type='vec3'):
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
