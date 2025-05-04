#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>

class Renderer {
public:
    Renderer();
    ~Renderer();
    
    void initialize();
    void render();
    void processInput();
    bool windowShouldClose() const;
    
private:
    std::string loadShaderSource(const std::string& filePath);
    GLuint compileShader(GLenum shaderType, const std::string& source);
    GLuint createShaderProgram(const std::string& vertexPath, const std::string& fragmentPath);
    GLuint createComputeShaderProgram(const std::string& computePath);
    
    void createShaders();
    void createComputeResources();
    void setupScreenQuad();
    void dispatchCompute();
    void cleanUp();
    
    GLFWwindow* m_window = nullptr;
    
    // Shader programs
    GLuint m_computeShaderProgram = 0;
    GLuint m_quadShaderProgram = 0;
    
    // Compute resources
    GLuint m_ssbo = 0;           // Shader Storage Buffer Object for compute data
    GLuint m_outputTexture = 0;  // Texture for compute shader output
    
    // Screen quad for displaying results
    GLuint m_quadVAO = 0;
    GLuint m_quadVBO = 0;
    
    // Window dimensions
    int m_windowWidth = 1280;
    int m_windowHeight = 720;
    
    // Time tracking for animations
    float m_time = 0.0f;
    float m_deltaTime = 0.0f;
    float m_lastFrame = 0.0f;
    
    // Shader paths
    const std::string m_shaderDir = "shaders/";
    const std::string m_computeShaderPath = m_shaderDir + "compute.glsl";
    const std::string m_quadVertexShaderPath = m_shaderDir + "quad.vert";
    const std::string m_quadFragmentShaderPath = m_shaderDir + "quad.frag";
};