#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <fstream>

enum class RenderMode {
    MODE_2D,
    MODE_3D
};

class UIManager {
public:
    UIManager(GLFWwindow* window);
    ~UIManager();

    void initialize();
    void newFrame();
    void render();
    void update();

    // UI State getters
    RenderMode getRenderMode() const { return m_renderMode; }
    bool isWireframeMode() const { return m_wireframeMode; }
    float getAnimationSpeed() const { return m_animationSpeed; }

    // Callbacks
    using ModeChangeCallback = std::function<void(RenderMode)>;
    void setModeChangeCallback(ModeChangeCallback callback) { m_modeChangeCallback = callback; }

private:
    // UI Components
    void showMainMenuBar();
    void showStatsWindow();
    void showControlPanel();
    void showSceneSettings();
    void showRenderSettings();
    void showPhysicsSettings();
    void showUISettings();

    // Settings management
    void loadSettings();
    void saveSettings();
    void applyUIScale();

    // Helper functions
    void applyStyle();

    // Styling
    void setupUnrealTheme();
    ImFont* loadFont(const std::string& name, float size);

private:
    GLFWwindow* m_window;

    // UI state
    bool m_showDemoWindow = false;
    bool m_showStatsWindow = true;
    bool m_showControlPanel = true;
    bool m_showSceneSettings = false;
    bool m_showRenderSettings = false;
    bool m_showPhysicsSettings = false;
    bool m_showUISettings = false;

    // Application state
    RenderMode m_renderMode = RenderMode::MODE_2D;
    bool m_wireframeMode = false;
    float m_animationSpeed = 1.0f;

    // UI settings
    float m_uiScale = 1.0f;
    float m_pendingScale = 1.5f;
    bool m_settingsChanged = false;
    bool m_scaleNeedsRestart = false;

    // Performance metrics
    float m_frameTimeHistory[100] = {};
    int m_frameTimeIndex = 0;

    // Fonts
    ImFont* m_titleFont = nullptr;
    ImFont* m_regularFont = nullptr;
    ImFont* m_smallFont = nullptr;

    // Callbacks
    ModeChangeCallback m_modeChangeCallback;

    // Settings file path
    const std::string m_settingsFile = "ui_settings.cfg";
};
