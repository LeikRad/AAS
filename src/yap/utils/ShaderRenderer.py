import moderngl
import numpy as np
from PIL import Image
import os

class ShaderRenderer:
    def __init__(self, width=256, height=256):
        self.width = width
        self.height = height
        
        # Create a standalone OpenGL context (no window needed)
        self.ctx = moderngl.create_standalone_context()
        self.texture = None
        
        # Determine shader directory
        shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')
        
        # Load vertex shader from file
        vertex_path = os.path.join(shader_dir, 'vertex.glsl')
        if os.path.exists(vertex_path):
            with open(vertex_path, 'r') as f:
                self.vertex_shader = f.read()
        else:
            # Fallback to inline shader
            self.vertex_shader = """
            #version 330
            in vec2 a_position;
            out vec2 v_coord;

            void main() {
            	vec2 xpos = a_position * vec2(2, 2) - vec2(1, 1);
            	gl_Position = vec4(xpos, 0, 1);
            	v_coord = a_position;
            }
            """
        
        # Load fragment shader from file or use inline
        fragment_path = os.path.join(shader_dir, 'fragment.glsl')
        if os.path.exists(fragment_path):
            with open(fragment_path, 'r') as f:
                self.fragment_shader = f.read()
        else:
            self.fragment_shader = self._default_fragment_shader()
        
        # Create shader program
        self.program = self.ctx.program(
            vertex_shader=self.vertex_shader,
            fragment_shader=self.fragment_shader
        )
        
        # Create a simple 2D quad with coordinates (0,0) to (1,1)
        # Vertex shader will scale to NDC (-1,-1) to (1,1)
        vertices = np.array([
            # Position (x, y) in 0-1 range
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,
        ], dtype='f4')
        
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.simple_vertex_array(
            self.program,
            self.vbo,
            'a_position'
        )
        
        # Create framebuffer for off-screen rendering
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)]
        )
    
    def _default_fragment_shader(self):
        return """
        #version 330
        in vec2 v_coord;
        layout (location = 0, index = 0) out vec4 o_color;
        uniform sampler2D tx;
        void main() {
        	vec4 t = texture(tx, v_coord);
        	float clr = t.x;
        	float ch = t.y;
        	ch /= clr;
        	clr *= 4096.0;
        //	if (clr != 0.0)
        //		clr = 1.0 + (log(clr) / 10.0);
        	o_color = vec4(clr * (1.0 - ch), clr/2.0, clr * ch, 0);
        }
        """
    
    def upload_texture(self, texture_array):
        """
        Upload a 256x256x2 numpy array (RG32F format) to GPU
        
        Args:
            texture_array: numpy array of shape (256, 256, 2) with dtype float32
        """
        if texture_array.shape != (256, 256, 2):
            raise ValueError("Texture array must be 256x256x2")
        
        if texture_array.dtype != np.float32:
            texture_array = texture_array.astype(np.float32)
        
        # Create or update the texture
        if self.texture:
            self.texture.release()
        
        # Create RG32F texture
        self.texture = self.ctx.texture(
            (self.width, self.height),
            components=2,
            dtype='f4'
        )
        
        # Upload data
        self.texture.write(texture_array.tobytes())
        
        # Set texture parameters
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture.repeat_x = False
        self.texture.repeat_y = False
    
    def render(self):
        """Render the texture using the shader"""
        if not self.texture:
            raise ValueError("No texture uploaded. Call upload_texture() first.")
        
        # Bind framebuffer for off-screen rendering
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        # Bind texture
        self.texture.use(0)
        self.program['tx'] = 0
        
        # Render
        self.vao.render()
    
    def save_image(self, filename='output.png'):
        """Save the rendered result to an image file"""
        self.render()
        
        # Read pixels from framebuffer
        data = self.fbo.read(components=4)
        
        # Convert to PIL Image
        img = Image.frombytes('RGBA', (self.width, self.height), data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)  # Flip vertically
        
        # Convert to RGB and save
        img = img.convert('RGB')
        img.save(filename)
        
        return img
    
    def get_raw_texture_data(self):
        """Get the raw RG32F texture data as numpy array"""
        if not self.texture:
            raise ValueError("No texture uploaded.")
        
        data = self.texture.read()
        texture_array = np.frombuffer(data, dtype=np.float32)
        texture_array = texture_array.reshape((self.height, self.width, 2))
        
        return texture_array
    
    def cleanup(self):
        """Release OpenGL resources"""
        if self.texture:
            self.texture.release()
        self.fbo.release()
        self.vao.release()
        self.vbo.release()
        self.program.release()
        self.ctx.release()
    
    def __del__(self):
        """Cleanup on object destruction"""
        try:
            self.cleanup()
        except:
            pass
