from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

g_cam_azimuth = 0.
g_cam_elevation = 0.

g_cam_before_scail = 1.
g_cam_scail = 1.

g_cam_xoffset = 0.
g_cam_yoffset = 0.
g_cam_before_xpos = 0.
g_cam_before_ypos = 0.

g_cam_viewport_mode = 1

g_cam_distance = np.sqrt(192/100)

g_cam_camera_point = glm.vec3(0.8,0.8,0.8)
g_cam_looking_point = glm.vec3(0.0,0.0,0.0)
g_cam_up_vector = glm.vec3(0,1,0)

CW = glm.vec4(glm.normalize(g_cam_camera_point - g_cam_looking_point),1)
CU = glm.vec4(glm.normalize(glm.cross(g_cam_up_vector,CW.xyz)),1)
CV = glm.vec4(glm.normalize(glm.cross(CW.xyz,CU.xyz)),1)
P = glm.perspective(45, 1, .1, 3)

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
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
    global P, g_cam_viewport_mode
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    if key==GLFW_KEY_V and action==GLFW_PRESS:
        if g_cam_viewport_mode==1:
            P = glm.ortho(-1.5,1.5,-1.5,1.5,-1.5,1.5)
            g_cam_viewport_mode = 0
        else:
            P = glm.perspective(45, 1, .1, 3)
            g_cam_viewport_mode = 1


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
        g_cam_xoffset = x_offset/100
        g_cam_yoffset = -y_offset/100
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
    if ypos < 0 and g_cam_distance > np.radians(1):
        g_cam_scail -= np.radians(1)
    if ypos > 0:
        g_cam_scail += np.radians(1)
    g_cam_camera_point += (g_cam_scail - g_cam_before_scail)*CW.xyz
    g_cam_distance = np.sqrt(glm.length2(g_cam_camera_point - g_cam_looking_point))
    calculate_uvw()

def prepare_vao_triangle():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         0.3, 0.3, -0.3, 0.0, 1.0, 1.0,
         0.3, -0.3, -0.3, 0.0, 1.0, 1.0,
         -0.3, 0.3, -0.3, 0.0, 1.0, 1.0,
         -0.3, 0.3, -0.3, 0.0, 1.0, 1.0,
         0.3, -0.3, -0.3, 0.0, 1.0, 1.0,
         -0.3, -0.3, -0.3, 0.0, 1.0, 1.0,

         0.3, 0.3, -0.3, 0.0, 1.0, 0.0,
         0.3, 0.3, 0.3, 0.0, 1.0, 0.0,
         0.3, -0.3, -0.3, 0.0, 1.0, 0.0,
         0.3, 0.3, 0.3, 0.0, 1.0, 0.0,
         0.3, -0.3, 0.3, 0.0, 1.0, 0.0,
         0.3, -0.3, -0.3, 0.0, 1.0, 0.0,

         0.3, 0.3, 0.3, 0.0, 0.0, 1.0,
         0.3, 0.3, -0.3, 0.0, 0.0, 1.0,
         -0.3, 0.3, -0.3, 0.0, 0.0, 1.0,
         -0.3, 0.3, -0.3, 0.0, 0.0, 1.0,
         0.3, 0.3, 0.3, 0.0, 0.0, 1.0,
         -0.3, 0.3, 0.3, 0.0, 0.0, 1.0,

         -0.3, -0.3, 0.3, 1.0, 0.0, 0.0,
         -0.3, 0.3, 0.3, 1.0, 0.0, 0.0,
         0.3, -0.3, 0.3, 1.0, 0.0, 0.0,
         0.3, -0.3, 0.3, 1.0, 0.0, 0.0,
         -0.3, 0.3, 0.3, 1.0, 0.0, 0.0,
         0.3, 0.3, 0.3, 1.0, 0.0, 0.0,

         -0.3, -0.3, 0.3, 1.0, 0.0, 1.0,
         -0.3, -0.3, -0.3, 1.0, 0.0, 1.0,
         -0.3, 0.3, 0.3, 1.0, 0.0, 1.0,
         -0.3, -0.3, -0.3, 1.0, 0.0, 1.0,
         -0.3, 0.3, -0.3, 1.0, 0.0, 1.0,
         -0.3, 0.3, 0.3, 1.0, 0.0, 1.0,

         -0.3, -0.3, -0.3, 1.0, 1.0, 0.0,
         -0.3, -0.3, 0.3, 1.0, 1.0, 0.0,
         0.3, -0.3, 0.3, 1.0, 1.0, 0.0,
         0.3, -0.3, 0.3, 1.0, 1.0, 0.0,
         -0.3, -0.3, -0.3, 1.0, 1.0, 0.0,
         0.3, -0.3, -0.3, 1.0, 1.0, 0.0,
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
    vertices = glm.array(glm.float32,
        # position        # color
         0.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis start
         1.5, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end 
         0.0, 0.0, 0.0,  0.0, 1.0, 0.0, # y-axis start
         0.0, 1.5, 0.0,  0.0, 1.0, 0.0, # y-axis end 
         0.0, 0.0, 0.0,  0.0, 0.0, 1.0, # z-axis start
         0.0, 0.0, 1.5,  0.0, 0.0, 1.0, # z-axis end 

         0.0, 0.0, 0.0,  1.0, 1.0, 1.0,
         0.0, 0.0, -1.5,  1.0, 1.0, 1.0,

         0.0, 0.0, 0.0,  1.0, 1.0, 1.0,
         -1.5, 0.0, 0.0,  1.0, 1.0, 1.0,

         0.4, 0.0, 1.5,  1.0, 1.0, 1.0,
         0.4, 0.0, -1.5,  1.0, 1.0, 1.0,
         0.8, 0.0, 1.5,  1.0, 1.0, 1.0,
         0.8, 0.0, -1.5,  1.0, 1.0, 1.0,
         1.2, 0.0, 1.5,  1.0, 1.0, 1.0,
         1.2, 0.0, -1.5,  1.0, 1.0, 1.0,

         -0.4, 0.0, 1.5,  1.0, 1.0, 1.0,
         -0.4, 0.0, -1.5,  1.0, 1.0, 1.0,
         -0.8, 0.0, 1.5,  1.0, 1.0, 1.0,
         -0.8, 0.0, -1.5,  1.0, 1.0, 1.0,
         -1.2, 0.0, 1.5,  1.0, 1.0, 1.0,
         -1.2, 0.0, -1.5,  1.0, 1.0, 1.0,

         1.5, 0.0, 0.4,  1.0, 1.0, 1.0,
         -1.5, 0.0, 0.4,  1.0, 1.0, 1.0,
         1.5, 0.0, 0.8,  1.0, 1.0, 1.0,
         -1.5, 0.0, 0.8,  1.0, 1.0, 1.0,
         1.5, 0.0, 1.2,  1.0, 1.0, 1.0,
         -1.5, 0.0, 1.2,  1.0, 1.0, 1.0,

         1.5, 0.0, -0.4,  1.0, 1.0, 1.0,
         -1.5, 0.0, -0.4,  1.0, 1.0, 1.0,
         1.5, 0.0, -0.8,  1.0, 1.0, 1.0,
         -1.5, 0.0, -0.8,  1.0, 1.0, 1.0,
         1.5, 0.0, -1.2,  1.0, 1.0, 1.0,
         -1.5, 0.0, -1.2,  1.0, 1.0, 1.0,
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
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, mouse_cursor_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    
    # prepare vaos
    vao_triangle = prepare_vao_triangle()
    vao_frame = prepare_vao_frame()

    global g_cam_camera_point, g_cam_looking_point, g_cam_up_vector, g_cam_azimuth, g_cam_elevation, g_cam_xoffset, g_cam_yoffset, CU, CV, CW, P
    g_cam_before_scail = g_cam_scail

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        # view matrix
        # rotate camera position with g_cam_ang / move camera up & down with g_cam_height
        g_cam_camera_point += (g_cam_xoffset)*CU.xyz + (g_cam_yoffset)*CV.xyz
        g_cam_looking_point += (g_cam_xoffset)*CU.xyz + (g_cam_yoffset)*CV.xyz
        calculate_uvw()

        V = glm.lookAt(g_cam_camera_point, g_cam_looking_point, g_cam_up_vector)

        I = glm.mat4()

        MVP = P*V*I
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

        # draw current frame
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 34)

        t = glfwGetTime()
        T = glm.translate(glm.vec3(.5*np.sin(t), .2, 0.))

        # M = R
        M = T
        # M = S
        # M = R @ T
        # M = S

        # current frame: P*V*M
        MVP = P*V*M
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))

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
