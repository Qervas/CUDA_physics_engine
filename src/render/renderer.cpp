#include "renderer.h"
#include <iostream>
#include <stdexcept>
#include <chrono>

Renderer::Renderer() {
    initialize();
}

Renderer::~Renderer() {
    cleanUp();
}

std::string Renderer::loadShaderSource(const std::string& filePath) {
    std::ifstream shaderFile;
    shaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    
    try {
        // Open file
        shaderFile.open(filePath);
        std::stringstream shaderStream;
        
        // Read file's buffer contents into stream
        shaderStream << shaderFile.rdbuf();
        
        // Close file
        shaderFile.close();
        
        // Convert stream into string
        return shaderStream.str();
    } catch (std::ifstream::failure& e) {
        throw std::runtime_error("ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " + std::string(e.what()));
    }
}

GLuint Renderer::compileShader(GLenum shaderType, const std::string& source) {
    GLuint shader = glCreateShader(shaderType);
    const char* src = source.c_str();
    
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    
    // Check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    
    if (!success) {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::string shaderTypeName;
        switch (shaderType) {
            case GL_VERTEX_SHADER: shaderTypeName = "VERTEX"; break;
            case GL_FRAGMENT_SHADER: shaderTypeName = "FRAGMENT"; break;
            case GL_COMPUTE_SHADER: shaderTypeName = "COMPUTE"; break;
            default: shaderTypeName = "UNKNOWN"; break;
        }
        throw std::runtime_error("ERROR::SHADER::" + shaderTypeName + "::COMPILATION_FAILED\n" + infoLog);
    }
    
    return shader;
}

GLuint Renderer::createShaderProgram(const std::string& vertexPath, const std::string& fragmentPath) {
    // Compile shaders
    std::string vertexSource = loadShaderSource(vertexPath);
    std::string fragmentSource = loadShaderSource(fragmentPath);
    
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);
    
    // Create shader program
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    // Check for linking errors
    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        throw std::runtime_error("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + std::string(infoLog));
    }
    
    // Delete shaders after linking
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return shaderProgram;
}

GLuint Renderer::createComputeShaderProgram(const std::string& computePath) {
    // Compile compute shader
    std::string computeSource = loadShaderSource(computePath);
    GLuint computeShader = compileShader(GL_COMPUTE_SHADER, computeSource);
    
    // Create shader program
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, computeShader);
    glLinkProgram(shaderProgram);
    
    // Check for linking errors
    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        throw std::runtime_error("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + std::string(infoLog));
    }
    
    // Delete shader after linking
    glDeleteShader(computeShader);
    
    return shaderProgram;
}

void Renderer::initialize() {
    // Initialize GLFW
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    
    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // OpenGL 4.3 for compute shaders
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    // Create window
    m_window = glfwCreateWindow(m_windowWidth, m_windowHeight, "Physics Engine - Compute Shader", nullptr, nullptr);
    if (!m_window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    
    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1); // Enable vsync
    
    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        throw std::runtime_error("Failed to initialize GLEW");
    }
    
    // Check for compute shader support
    if (!GLEW_ARB_compute_shader) {
        throw std::runtime_error("Compute shaders not supported on this hardware");
    }
    
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    
    // Create needed resources
    createShaders();
    createComputeResources();
    setupScreenQuad();
}

void Renderer::createShaders() {
    try {
        // Create compute shader program
        m_computeShaderProgram = createComputeShaderProgram(m_computeShaderPath);
        
        // Create quad shader program for display
        m_quadShaderProgram = createShaderProgram(m_quadVertexShaderPath, m_quadFragmentShaderPath);
        
        std::cout << "Shaders compiled and linked successfully" << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Shader creation failed: " + std::string(e.what()));
    }
}

void Renderer::createComputeResources() {
    // Generate texture that will hold the compute shader output
    glGenTextures(1, &m_outputTexture);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_outputTexture);
    
    // Allocate storage for the texture
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, m_windowWidth, m_windowHeight);
    
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // Bind the output texture to the 0 binding point of the compute shader
    glBindImageTexture(0, m_outputTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
}

void Renderer::setupScreenQuad() {
    // Set up a full-screen quad to display the compute shader output
    float quadVertices[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    
    // Create VAO & VBO for screen quad
    glGenVertexArrays(1, &m_quadVAO);
    glGenBuffers(1, &m_quadVBO);
    
    glBindVertexArray(m_quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
}

void Renderer::dispatchCompute() {
    // Calculate delta time for smooth animation
    float currentFrame = glfwGetTime();
    m_deltaTime = currentFrame - m_lastFrame;
    m_lastFrame = currentFrame;
    m_time += m_deltaTime;
    
    // Use compute shader
    glUseProgram(m_computeShaderProgram);
    
    // Set time uniform (for animation)
    glUniform1f(glGetUniformLocation(m_computeShaderProgram, "u_time"), m_time);
    
    // Dispatch compute shader with workgroups
    // Each workgroup is 16x16, so we need (width + 15) / 16 workgroups in x direction
    // and (height + 15) / 16 workgroups in y direction
    int workGroupsX = (m_windowWidth + 15) / 16;
    int workGroupsY = (m_windowHeight + 15) / 16;
    glDispatchCompute(workGroupsX, workGroupsY, 1);
    
    // Wait for compute shader to finish
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void Renderer::render() {
    // Run the compute shader to generate the image
    dispatchCompute();
    
    // Clear the screen
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Use the quad shader program to draw the texture
    glUseProgram(m_quadShaderProgram);
    glBindVertexArray(m_quadVAO);
    
    // Bind the texture output from compute shader
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_outputTexture);
    glUniform1i(glGetUniformLocation(m_quadShaderProgram, "screenTexture"), 0);
    
    // Draw the quad
    glDrawArrays(GL_TRIANGLES, 0, 6);
    
    // Swap buffers and poll events
    glfwSwapBuffers(m_window);
    glfwPollEvents();
}

void Renderer::processInput() {
    // Process keyboard input
    if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(m_window, true);
    }
}

bool Renderer::windowShouldClose() const {
    return glfwWindowShouldClose(m_window);
}

void Renderer::cleanUp() {
    // Clean up OpenGL objects
    if (m_quadVAO != 0) {
        glDeleteVertexArrays(1, &m_quadVAO);
        m_quadVAO = 0;
    }
    
    if (m_quadVBO != 0) {
        glDeleteBuffers(1, &m_quadVBO);
        m_quadVBO = 0;
    }
    
    if (m_outputTexture != 0) {
        glDeleteTextures(1, &m_outputTexture);
        m_outputTexture = 0;
    }
    
    if (m_computeShaderProgram != 0) {
        glDeleteProgram(m_computeShaderProgram);
        m_computeShaderProgram = 0;
    }
    
    if (m_quadShaderProgram != 0) {
        glDeleteProgram(m_quadShaderProgram);
        m_quadShaderProgram = 0;
    }
    
    // Clean up GLFW
    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    
    glfwTerminate();
}