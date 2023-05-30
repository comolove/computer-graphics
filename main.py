from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np
import os

g_cam_azimuth = 0.
g_cam_elevation = 0.

g_cam_before_scail = 1.
g_cam_scail = 1.

g_cam_xoffset = 0.
g_cam_yoffset = 0.
g_cam_before_xpos = 0.
g_cam_before_ypos = 0.

g_cam_viewport_mode = 1

g_cam_distance = np.sqrt(67500)

g_cam_camera_point = glm.vec3(150,150,150)
g_cam_looking_point = glm.vec3(0.0,0.0,0.0)
g_cam_up_vector = glm.vec3(0,1,0)

g_mode = 0

vao_obj = 0
v_num_obj = 0

cat = 0
guns = []
hearts = []

vao_cat = 0
vao_gun = 0
vao_heart = 0

v_num_cat = 0
v_num_gun = 0
v_num_heart = 0

polygon_mode = 0

CW = glm.vec4(glm.normalize(g_cam_camera_point - g_cam_looking_point),1)
CU = glm.vec4(glm.normalize(glm.cross(g_cam_up_vector,CW.xyz)),1)
CV = glm.vec4(glm.normalize(glm.cross(CW.xyz,CU.xyz)),1)
P = glm.perspective(45, 1, .1, 600)

g_vertex_shader_src_attribute = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;
uniform mat3 color;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
    
}
'''

g_vertex_shader_src_color_uniform = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 

out vec3 vout_surface_pos;
out vec3 vout_normal;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(transpose(inverse(M))) * vin_normal);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
}
'''

g_fragment_shader_src_pong = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;

out vec4 FragColor;

uniform vec3 view_pos;
uniform vec3 light_pos;
uniform vec3 color;

void main()
{
    // light and material properties
    vec3 light_color = vec3(1,1,1);
    vec3 material_color = color;
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient = 0.3*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;

    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular = light_color;  // for non-metal material

    // ambient
    vec3 ambient = light_ambient * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);

    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;

    vec3 ecolor = ambient + diffuse + specular;
    FragColor = vec4(ecolor, 1.);
}
'''

class Node:
    def __init__(self, parent, shape_transform, color, v_num):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = shape_transform
        self.color = color

        # vertex cnt
        self.v_num = v_num

    def set_transform(self, transform):
        self.transform = transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.transform
        else:
            self.global_transform = self.transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform
    def get_shape_transform(self):
        return self.shape_transform
    def get_color(self):
        return self.color

def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------
    
    # vertex shader 
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)    # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source) # provide shader source code
    glCompileShader(vertex_shader)                      # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())
        
    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)    # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source) # provide shader source code
    glCompileShader(fragment_shader)                        # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()               # create an empty program object
    glAttachShader(shader_program, vertex_shader)    # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program

def calculate_uvw():
    global g_cam_up_vector, g_cam_looking_point, g_cam_camera_point, CU, CV, CW
    CW = glm.vec4(glm.normalize(g_cam_camera_point - g_cam_looking_point),0)
    CU = glm.vec4(glm.normalize(glm.cross(g_cam_up_vector,CW.xyz)),0)
    CV = glm.vec4(glm.normalize(glm.cross(CW.xyz,CU.xyz)),0)


def key_callback(window, key, scancode, action, mods):
    global P, g_cam_viewport_mode, g_mode, vao_obj, g_cam_distance, g_cam_camera_point, polygon_mode
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    if key==GLFW_KEY_V and action==GLFW_PRESS:
        if g_cam_viewport_mode==1:
            P = glm.ortho(-400,400,-400,400,-400,400)
            g_cam_viewport_mode = 0
        else:
            P = glm.perspective(45, 1, .1, 600)
            g_cam_viewport_mode = 1
    if key==GLFW_KEY_H:
        g_mode = 1
        vao_obj = 0
    if key==GLFW_KEY_Z and action==GLFW_PRESS:
        if polygon_mode == 0:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            polygon_mode = 1
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            polygon_mode = 0


def mouse_cursor_callback(window, xpos, ypos):
    global g_cam_before_xpos, g_cam_before_ypos, g_cam_azimuth, g_cam_elevation, g_cam_xoffset, g_cam_yoffset, g_cam_camera_point, g_cam_looking_point, g_cam_up_vector, g_cam_distance
    x_offset = g_cam_before_xpos - xpos
    y_offset = g_cam_before_ypos - ypos
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS):
        g_cam_azimuth =  x_offset*np.radians(0.3)
        g_cam_elevation = -y_offset*np.radians(0.3)

        g_cam_before_xpos = xpos
        g_cam_before_ypos = ypos

        g_cam_camera_point += (2*g_cam_distance*np.sin(g_cam_azimuth/2))*glm.vec3((glm.rotate(g_cam_azimuth/2,g_cam_up_vector)*CU))
        g_cam_camera_point = g_cam_looking_point + g_cam_distance*glm.normalize(g_cam_camera_point - g_cam_looking_point)
        calculate_uvw()

        g_cam_camera_point += (2*g_cam_distance*np.sin(g_cam_elevation/2))*glm.vec3((glm.rotate(g_cam_elevation/2,CU.xyz)*CV))
        g_cam_camera_point = g_cam_looking_point + g_cam_distance*glm.normalize(g_cam_camera_point - g_cam_looking_point)
        calculate_uvw()

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS):
        g_cam_xoffset = x_offset/10
        g_cam_yoffset = -y_offset/10
        g_cam_before_xpos = xpos
        g_cam_before_ypos = ypos

def mouse_button_callback(window, button, action, mod):
    global g_cam_before_xpos, g_cam_before_ypos, g_cam_elevation, g_cam_azimuth
    if button==GLFW_MOUSE_BUTTON_LEFT:
        if action==GLFW_PRESS:
            g_cam_before_xpos = glfwGetCursorPos(window)[0]
            g_cam_before_ypos = glfwGetCursorPos(window)[1]
    if button==GLFW_MOUSE_BUTTON_RIGHT:
        if action==GLFW_PRESS:
            g_cam_before_xpos = glfwGetCursorPos(window)[0]
            g_cam_before_ypos = glfwGetCursorPos(window)[1]
            
    
def scroll_callback(window, xpos, ypos):
    global g_cam_scail, g_cam_camera_point, g_cam_before_scail, g_cam_distance, g_cam_looking_point
    g_cam_before_scail = g_cam_scail
    if ypos < 0 and g_cam_distance > np.radians(100):
        g_cam_scail -= np.radians(100)
    if ypos > 0:
        g_cam_scail += np.radians(100)
    g_cam_camera_point += (g_cam_scail - g_cam_before_scail)*CW.xyz
    g_cam_distance = np.sqrt(glm.length2(g_cam_camera_point - g_cam_looking_point))
    calculate_uvw()

def draw_node(vao, node, VP, MVP_loc, color_loc):
    MVP = VP * node.get_global_transform() * node.get_shape_transform()
    color = node.get_color()

    glBindVertexArray(vao)
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniform3f(color_loc, color.r, color.g, color.b)
    glDrawElements(GL_TRIANGLES, node.v_num, GL_UNSIGNED_INT, None)

def prepare_vao_obj(vertices, faces, normals, normal_index):
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # create and activate EBO (element buffer object)
    EBO = glGenBuffers(1)   # create a buffer object ID and store it to EBO variable
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)  # activate EBO as an element buffer object

    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes + normals.nbytes, None, GL_STATIC_DRAW)

    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes + normal_index.nbytes, None, GL_STATIC_DRAW)

    # copy vertex data to VBO
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices.ptr) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # copy index data to EBO
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, faces.nbytes, faces.ptr) # allocate GPU memory for and copy index data to the currently bound element buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    glBufferSubData(GL_ARRAY_BUFFER, vertices.nbytes, normals.nbytes, normals.ptr)

    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, normal_index.nbytes, normal_index.ptr)

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * glm.sizeof(glm.float32), ctypes.c_void_p(vertices.nbytes))
    glEnableVertexAttribArray(1)

    return VAO

def parse_obj(path,scail = 30):
    global v_num_gun, v_num_cat, v_num_heart, v_num_obj
    with open(path, 'r') as f:
            
        vertices = np.array([])
        faces = np.array([])
        normals = np.array([])
        normal_index = np.array([])
        index = []

        v_num = 0
        face3_num = 0
        face4_num = 0
        faceM_num = 0

        for line in f:
            components = line.strip().split()

            if len(components) == 0:
                continue
            elif components[0] == 'v':
                vertex = [float(components[1])*scail, float(components[2])*scail, float(components[3])*scail]
                vertices = np.append(vertices,vertex)

            elif components[0] == 'vn':
                normal = [float(components[1]), float(components[2]), float(components[3])]
                normals = np.append(normals,normal)

            elif components[0] == 'f':
                face = [int(x.split('/')[0]) for x in components[1:]]
                    
                if len(face) == 3:
                    face3_num += 1
                elif len(face) == 4:
                    face4_num += 1
                else :
                    faceM_num += 1
                f1 = face[0] - 1
                f2 = face[1] - 1
                f3 = face[2] - 1
                faces = np.append(faces,[f1,f2,f3])
                v_num += 3
                for i in range(3,len(face)):
                    f2 = f3
                    f3 = face[i] - 1
                    faces = np.append(faces,[f1,f2,f3])
                    v_num += 3

                if '/' not in components[1] or ('/' in components[1] and len(components[1].split('/')) <= 2):
                    continue
                if '//' in components[1]:
                    index = [int(x.split('//')[1]) for x in components[1:]]
                elif '/' in components[1] and len(components[1].split('/')) > 2:
                    index = [int(x.split('/')[2]) for x in components[1:]]
                i1 = index[0]
                i2 = index[1]
                i3 = index[2]
                normal_index = np.append(normal_index,[i1,i2,i3])
                for i in range(3,len(index)):
                    i2 = i3
                    i3 = index[i] 
                    normal_index = np.append(normal_index,[i1,i2,i3])
        

        if(scail == 30):
            v_num_obj = v_num

            segments = path.split("\\")
            last_segment = segments[-1]

            print(last_segment)
            print(face3_num + face4_num + faceM_num)
            print(face3_num)
            print(face4_num)
            print(faceM_num)
        elif(scail == 0.3):
            v_num_cat = v_num
        elif(scail == 100):
            v_num_gun = v_num
        elif(scail == 3):
            v_num_heart = v_num
        
        normal_index = normal_index + (len(vertices) / 3 - 1)

        vertices = vertices.astype(np.float32)
        faces = faces.astype(np.uint32)
        normals = normals.astype(np.float32)
        normal_index = normal_index.astype(np.uint32)

        vertices = glm.array(vertices)
        faces = glm.array(faces)
        normals = glm.array(normals)
        normal_index = glm.array(normal_index)

        return prepare_vao_obj(vertices,faces,normals,normal_index)

def drop_callback(window, paths):
    global vao_obj
    for path in paths:
        vao_obj = parse_obj(path)

def prepare_vao_hierarchical_model():
    global cat, guns, hearts, vao_cat, vao_gun, vao_heart
    root_path = "C:\\Users\\jungj\\Desktop\\computer-graphics"
    cat_path = os.path.join(root_path,"cat\\cat.obj")
    gun_path = os.path.join(root_path,"gun\\Gun.obj")
    heart_path = os.path.join(root_path,"heart\\12190_Heart_v1_L3.obj")
    vao_cat = parse_obj(cat_path,0.3)
    vao_gun = parse_obj(gun_path,100)
    vao_heart = parse_obj(heart_path,3)

    # create a hirarchical model - Node(parent, shape_transform, color)
    cat = Node(None, glm.identity(glm.mat4), glm.vec3(1,0.5,0), v_num_cat)
    for i in range(0,2):
        guns.append(Node(cat, glm.translate((0,50,40*((-1)**i))) * glm.rotate(glm.mat4(1.),glm.radians(180 + 90*(-1)**(i+1)),glm.vec3(0,1,0)) , glm.vec3(.6,.3,0), v_num_gun))
        for j in range(0,2):
            hearts.append(Node(guns[i],glm.translate((50*((-1)**j),30,100*((-1)**i))) * glm.rotate(glm.mat4(1.),glm.radians(270),glm.vec3(1,0,0)),glm.vec3(1,0,0),v_num_heart))

def prepare_vao_triangle():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         100, 100, -100, 0.0, 1.0, 1.0,
         100, -100, -100, 0.0, 1.0, 1.0,
         -100, 100, -100, 0.0, 1.0, 1.0,
         -100, 100, -100, 0.0, 1.0, 1.0,
         100, -100, -100, 0.0, 1.0, 1.0,
         -100, -100, -100, 0.0, 1.0, 1.0,

         100, 100, -100, 0.0, 1.0, 0.0,
         100, 100, 100, 0.0, 1.0, 0.0,
         100, -100, -100, 0.0, 1.0, 0.0,
         100, 100, 100, 0.0, 1.0, 0.0,
         100, -100, 100, 0.0, 1.0, 0.0,
         100, -100, -100, 0.0, 1.0, 0.0,

         100, 100, 100, 0.0, 0.0, 1.0,
         100, 100, -100, 0.0, 0.0, 1.0,
         -100, 100, -100, 0.0, 0.0, 1.0,
         -100, 100, -100, 0.0, 0.0, 1.0,
         100, 100, 100, 0.0, 0.0, 1.0,
         -100, 100, 100, 0.0, 0.0, 1.0,

         -100, -100, 100, 1.0, 0.0, 0.0,
         -100, 100, 100, 1.0, 0.0, 0.0,
         100, -100, 100, 1.0, 0.0, 0.0,
         100, -100, 100, 1.0, 0.0, 0.0,
         -100, 100, 100, 1.0, 0.0, 0.0,
         100, 100, 100, 1.0, 0.0, 0.0,

         -100, -100, 100, 1.0, 0.0, 1.0,
         -100, -100, -100, 1.0, 0.0, 1.0,
         -100, 100, 100, 1.0, 0.0, 1.0,
         -100, -100, -100, 1.0, 0.0, 1.0,
         -100, 100, -100, 1.0, 0.0, 1.0,
         -100, 100, 100, 1.0, 0.0, 1.0,

         -100, -100, -100, 1.0, 1.0, 0.0,
         -100, -100, 100, 1.0, 1.0, 0.0,
         100, -100, 100, 1.0, 1.0, 0.0,
         100, -100, 100, 1.0, 1.0, 0.0,
         -100, -100, -100, 1.0, 1.0, 0.0,
         100, -100, -100, 1.0, 1.0, 0.0,
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = np.array([])
    vertices = np.append(vertices,[0,0,0,1,0,0,300,0,0,1,0,0])
    vertices = np.append(vertices,[0,0,0,0,1,0,0,300,0,0,1,0])
    vertices = np.append(vertices,[0,0,0,0,0,1,0,0,300,0,0,1])
    vertices = np.append(vertices,[0,0,0,1,1,1,0,0,-1.5,1,1,1])
    vertices = np.append(vertices,[0,0,0,1,1,1,-1.5,0,0,1,1,1])
    vertices = np.append(vertices,[0,0,0,1,1,1,0,0,-300,1,1,1])
    vertices = np.append(vertices,[0,0,0,1,1,1,-300,0,0,1,1,1])

    for i in range(1,10):
        vertices = np.append(vertices,[i*30,0,300,1,1,1])
        vertices = np.append(vertices,[i*30,0,-300,1,1,1])
        vertices = np.append(vertices,[-i*30,0,300,1,1,1])
        vertices = np.append(vertices,[-i*30,0,-300,1,1,1])
        vertices = np.append(vertices,[300,0,i*30,1,1,1])
        vertices = np.append(vertices,[-300,0,i*30,1,1,1])
        vertices = np.append(vertices,[300,0,-i*30,1,1,1])
        vertices = np.append(vertices,[-300,0,-i*30,1,1,1])

    vertices = glm.array(vertices.astype(np.float32))
    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def main():
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(1000, 1000, '2019060682', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetKeyCallback(window, key_callback)
    glfwSetCursorPosCallback(window, mouse_cursor_callback)
    glfwSetMouseButtonCallback(window, mouse_button_callback)
    glfwSetScrollCallback(window, scroll_callback)
    glfwSetDropCallback(window, drop_callback)

    # load shaders
    shader_for_frame = load_shaders(g_vertex_shader_src_attribute, g_fragment_shader_src)
    shader_for_obj = load_shaders(g_vertex_shader_src_color_uniform, g_fragment_shader_src_pong)

    # get uniform locations
    MVP_loc_frame = glGetUniformLocation(shader_for_frame, 'MVP')
    MVP_loc_obj = glGetUniformLocation(shader_for_obj, 'MVP')
    M_loc_obj = glGetUniformLocation(shader_for_obj, 'M')
    color_loc_obj = glGetUniformLocation(shader_for_obj, 'color')
    view_pos_loc_obj = glGetUniformLocation(shader_for_obj, 'view_pos')
    light_pos_loc_obj = glGetUniformLocation(shader_for_obj, 'light_pos')
    
    # prepare vaos
    vao_triangle = prepare_vao_triangle()
    vao_frame = prepare_vao_frame()
    prepare_vao_hierarchical_model()

    global g_cam_camera_point, g_cam_looking_point, g_cam_up_vector, g_cam_azimuth, g_cam_elevation, g_cam_xoffset, g_cam_yoffset, CU, CV, CW, P

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_for_frame)

        # view matrix
        # rotate camera position with g_cam_ang / move camera up & down with g_cam_height
        g_cam_camera_point += (g_cam_xoffset)*CU.xyz + (g_cam_yoffset)*CV.xyz
        g_cam_looking_point += (g_cam_xoffset)*CU.xyz + (g_cam_yoffset)*CV.xyz
        calculate_uvw()

        V = glm.lookAt(g_cam_camera_point, g_cam_looking_point, g_cam_up_vector)

        I = glm.mat4()

        MVP = P*V*I
        glUniformMatrix4fv(MVP_loc_frame, 1, GL_FALSE, glm.value_ptr(MVP))

        # draw current frame
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 88)

        glUseProgram(shader_for_obj)

        glUniformMatrix4fv(M_loc_obj, 1, GL_FALSE, glm.value_ptr(I))

        t = glfwGetTime()
        T = glm.translate(glm.vec3(50*np.sin(t), .2, 0.))
        R = glm.rotate(glm.mat4(1.0), glm.radians(45*np.cos(t)), glm.vec3(0,1,0))

        # current frame: P*V
        MVP = P*V*I
        glUniformMatrix4fv(MVP_loc_obj, 1, GL_FALSE, glm.value_ptr(MVP))

        glUniform3f(view_pos_loc_obj, g_cam_looking_point.x, g_cam_looking_point.y, g_cam_looking_point.z)
        glUniform3f(light_pos_loc_obj,g_cam_looking_point.x*np.sin(t),g_cam_looking_point.y,g_cam_looking_point.z*np.cos(t))

        cat.set_transform(T)

        for i in range(0,len(guns)):
            guns[i].set_transform(R)
        
        for i in range(0,len(hearts)):
            hearts[i].set_transform(glm.translate(glm.vec3(50*np.sin(t)*(-1)**i, .2, 0.)))

        cat.update_tree_global_transform()


        if (vao_obj != 0):
            glBindVertexArray(vao_obj)
            glUniform3f(color_loc_obj, .5, .5, .5)
            glDrawElements(GL_TRIANGLES, v_num_obj, GL_UNSIGNED_INT, None)
        elif g_mode != 0:
            # glBindVertexArray(vao_cat)
            # glDrawElements(GL_TRIANGLES, v_num_cat, GL_UNSIGNED_INT, None)
            draw_node(vao_cat,cat,MVP,MVP_loc_obj,color_loc_obj)

            # glBindVertexArray(vao_gun)
            # glDrawElements(GL_TRIANGLES, v_num_gun, GL_UNSIGNED_INT, None)
            for i in range(0,len(guns)):
                draw_node(vao_gun,guns[i],MVP,MVP_loc_obj,color_loc_obj)

            for i in range(0,len(hearts)):
                draw_node(vao_heart,hearts[i],MVP,MVP_loc_obj,color_loc_obj)
        else:
            glUseProgram(shader_for_frame)
            # draw triangle w.r.t. the current frame
            glBindVertexArray(vao_triangle)
            glDrawArrays(GL_TRIANGLES, 0, 36)

        # swap front and back buffers
        glfwSwapBuffers(window)

        g_cam_xoffset = 0
        g_cam_yoffset = 0

        # poll events
        glfwPollEvents()


    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
