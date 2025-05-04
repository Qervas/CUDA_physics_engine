#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <unordered_map>

// Forward declare RenderMode to avoid circular dependencies
enum class RenderMode;

class Renderer {
public:
    Renderer();
    ~Renderer();
    
    void initialize();
    void render();
    void processInput();
    bool windowShouldClose() const;
    
    // Window accessor
    GLFWwindow* getWindow() const { return m_window; }
    
    // UI interaction methods
    void setRenderMode(RenderMode mode);
    void setWireframeMode(bool enabled);
    void setAnimationSpeed(float speed);
    
private:
    std::string loadShaderSource(const std::string& filePath);
    GLuint compileShader(GLenum shaderType, const std::string& source);
    GLuint createShaderProgram(const std::string& vertexPath, const std::string& fragmentPath);
    GLuint createComputeShaderProgram(const std::string& computePath);
    
    // Initialization
    void createShaders();
    void createComputeResources();
    void setupScreenQuad();
    void setup3DScene();
    
    // Rendering
    void dispatchCompute();
    void render2D();
    void render3D();
    
    // Cleanup
    void cleanUp();
    
    GLFWwindow* m_window = nullptr;
    
    // Shader programs
    GLuint m_computeShaderProgram = 0;
    GLuint m_quadShaderProgram = 0;
    GLuint m_3dShaderProgram = 0;
    
    // Compute resources
    GLuint m_ssbo = 0;           // Shader Storage Buffer Object for compute data
    GLuint m_outputTexture = 0;  // Texture for compute shader output
    
    // Screen quad for displaying results
    GLuint m_quadVAO = 0;
    GLuint m_quadVBO = 0;
    
    // 3D scene objects
    struct {
        GLuint vao = 0;
        GLuint vbo = 0;
        GLuint ebo = 0;
        int vertexCount = 0;
    } m_cube;
    
    // Window dimensions
    int m_windowWidth = 1280;
    int m_windowHeight = 720;
    
    // Rendering state
    RenderMode m_renderMode;
    bool m_wireframeMode = false;
    float m_animationSpeed = 1.0f;
    
    // Camera state
    struct {
        float position[3] = {0.0f, 0.0f, 5.0f};
        float target[3] = {0.0f, 0.0f, 0.0f};
        float up[3] = {0.0f, 1.0f, 0.0f};
        float fov = 45.0f;
        float nearPlane = 0.1f;
        float farPlane = 100.0f;
    } m_camera;
    
    // Time tracking for animations
    float m_time = 0.0f;
    float m_deltaTime = 0.0f;
    float m_lastFrame = 0.0f;
    
    // Shader paths
    const std::string m_shaderDir = "shaders/";
    const std::string m_computeShaderPath = m_shaderDir + "compute.glsl";
    const std::string m_quadVertexShaderPath = m_shaderDir + "quad.vert";
    const std::string m_quadFragmentShaderPath = m_shaderDir + "quad.frag";
    const std::string m_3dVertexShaderPath = m_shaderDir + "basic3d.vert";
    const std::string m_3dFragmentShaderPath = m_shaderDir + "basic3d.frag";
};