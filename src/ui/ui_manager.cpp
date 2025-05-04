#include "ui_manager.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

UIManager::UIManager(GLFWwindow* window) : m_window(window) {
    loadSettings();
    m_pendingScale = m_uiScale;
    initialize();
}

UIManager::~UIManager() {
    saveSettings();

    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void UIManager::loadSettings() {
    // Try to load settings from file
    std::ifstream file(m_settingsFile);
    if (file.is_open()) {
        // Read UI scale
        std::string line;
        while (std::getline(file, line)) {
            if (line.find("ui_scale=") == 0) {
                try {
                    m_uiScale = std::stof(line.substr(9));
                    // Clamp to reasonable values
                    m_uiScale = std::max(0.5f, std::min(3.0f, m_uiScale));
                } catch (...) {
                    m_uiScale = 1.0f; // Default if parsing fails
                }
            }
        }
        file.close();
        std::cout << "UI settings loaded: scale=" << m_uiScale << std::endl;
    } else {
        // Use defaults if file doesn't exist
        m_uiScale = 1.0f;
        std::cout << "Using default UI settings" << std::endl;
    }
}

void UIManager::saveSettings() {
    // Only save if settings were changed
    if (m_settingsChanged) {
        // Save the pending scale (the value that will be applied on restart)
        std::ofstream file(m_settingsFile);
        if (file.is_open()) {
            file << "ui_scale=" << m_pendingScale << std::endl;
            file.close();
            std::cout << "UI settings saved" << std::endl;
        } else {
            std::cerr << "Failed to save UI settings" << std::endl;
        }
        m_settingsChanged = false;
    }
}

void UIManager::applyUIScale() {
    // Apply the scale to ImGui style and fonts
    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = m_uiScale;

    // Scale all ImGui sizes
    ImGuiStyle& style = ImGui::GetStyle();

    // First, get default style
    ImGuiStyle defaultStyle;

    // Copy values from default style, then scale everything
    style = defaultStyle;
    style.ScaleAllSizes(m_uiScale);

    // Apply our custom theme on top of the scaled style
    setupUnrealTheme();
}

void UIManager::initialize() {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 430");

    // Load fonts - this must be done BEFORE first call to NewFrame()
    io.Fonts->AddFontDefault();

    // Initialize frame time history
    for (int i = 0; i < IM_ARRAYSIZE(m_frameTimeHistory); i++) {
        m_frameTimeHistory[i] = 0.0f;
    }

    // Apply UI scale from settings
    applyUIScale();

    std::cout << "UI manager initialized with scale: " << m_uiScale << std::endl;
}


ImFont* UIManager::loadFont(const std::string& name, float size) {
    ImGuiIO& io = ImGui::GetIO();
    return io.Fonts->AddFontFromFileTTF(name.c_str(), size * m_uiScale);
}

void UIManager::setupUnrealTheme() {
    // Setup Unreal Engine-like dark theme
    ImGuiStyle& style = ImGui::GetStyle();

    // Colors
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_Text]                   = ImVec4(0.90f, 0.90f, 0.90f, 1.00f);
    colors[ImGuiCol_TextDisabled]           = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_WindowBg]               = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
    colors[ImGuiCol_ChildBg]                = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
    colors[ImGuiCol_Border]                 = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_FrameBg]                = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(0.67f, 0.67f, 0.67f, 0.39f);
    colors[ImGuiCol_TitleBg]                = ImVec4(0.08f, 0.08f, 0.09f, 1.00f);
    colors[ImGuiCol_TitleBgActive]          = ImVec4(0.08f, 0.08f, 0.09f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(0.00f, 0.00f, 0.00f, 0.51f);
    colors[ImGuiCol_MenuBarBg]              = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    colors[ImGuiCol_ScrollbarBg]            = ImVec4(0.02f, 0.02f, 0.02f, 0.53f);
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.31f, 0.31f, 0.31f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.41f, 0.41f, 0.41f, 1.00f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.51f, 0.51f, 0.51f, 1.00f);
    colors[ImGuiCol_CheckMark]              = ImVec4(0.11f, 0.64f, 0.92f, 1.00f);
    colors[ImGuiCol_SliderGrab]             = ImVec4(0.11f, 0.64f, 0.92f, 1.00f);
    colors[ImGuiCol_SliderGrabActive]       = ImVec4(0.08f, 0.50f, 0.72f, 1.00f);
    colors[ImGuiCol_Button]                 = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_ButtonHovered]          = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
    colors[ImGuiCol_ButtonActive]           = ImVec4(0.67f, 0.67f, 0.67f, 0.39f);
    colors[ImGuiCol_Header]                 = ImVec4(0.22f, 0.22f, 0.22f, 1.00f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(0.67f, 0.67f, 0.67f, 0.39f);
    colors[ImGuiCol_Separator]              = ImVec4(0.43f, 0.43f, 0.50f, 0.50f);
    colors[ImGuiCol_SeparatorHovered]       = ImVec4(0.41f, 0.42f, 0.44f, 1.00f);
    colors[ImGuiCol_SeparatorActive]        = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
    colors[ImGuiCol_ResizeGrip]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(0.29f, 0.30f, 0.31f, 0.67f);
    colors[ImGuiCol_ResizeGripActive]       = ImVec4(0.26f, 0.59f, 0.98f, 0.95f);
    colors[ImGuiCol_Tab]                    = ImVec4(0.08f, 0.08f, 0.09f, 0.83f);
    colors[ImGuiCol_TabHovered]             = ImVec4(0.33f, 0.34f, 0.36f, 0.83f);
    colors[ImGuiCol_TabActive]              = ImVec4(0.23f, 0.23f, 0.24f, 1.00f);
    colors[ImGuiCol_TabUnfocused]           = ImVec4(0.08f, 0.08f, 0.09f, 1.00f);
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(0.13f, 0.14f, 0.15f, 1.00f);
    colors[ImGuiCol_PlotLines]              = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered]       = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
    colors[ImGuiCol_PlotHistogram]          = ImVec4(0.11f, 0.64f, 0.92f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(0.08f, 0.50f, 0.72f, 1.00f);
    colors[ImGuiCol_TableHeaderBg]          = ImVec4(0.19f, 0.19f, 0.20f, 1.00f);
    colors[ImGuiCol_TableBorderStrong]      = ImVec4(0.31f, 0.31f, 0.35f, 1.00f);
    colors[ImGuiCol_TableBorderLight]       = ImVec4(0.23f, 0.23f, 0.25f, 1.00f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.00f, 1.00f, 1.00f, 0.07f);
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(0.26f, 0.59f, 0.98f, 0.35f);
    colors[ImGuiCol_DragDropTarget]         = ImVec4(0.11f, 0.64f, 0.92f, 1.00f);
    colors[ImGuiCol_NavHighlight]           = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);

    // // Style
    // style.WindowPadding            = ImVec2(10, 10);
    // style.FramePadding             = ImVec2(8, 4);
    // style.CellPadding              = ImVec2(4, 2);
    // style.ItemSpacing              = ImVec2(8, 6);
    // style.ItemInnerSpacing         = ImVec2(4, 4);
    // style.TouchExtraPadding        = ImVec2(0, 0);
    // style.IndentSpacing            = 21;
    // style.ScrollbarSize            = 14;
    // style.GrabMinSize              = 10;

    // // Borders
    // style.WindowBorderSize         = 1;
    // style.ChildBorderSize          = 1;
    // style.PopupBorderSize          = 1;
    // style.FrameBorderSize          = 1;
    // style.TabBorderSize            = 1;

    // // Rounding
    // style.WindowRounding           = 4;
    // style.ChildRounding            = 4;
    // style.FrameRounding            = 2;
    // style.PopupRounding            = 2;
    // style.ScrollbarRounding        = 12;
    // style.GrabRounding             = 2;
    // style.TabRounding              = 4;

    // // Alignment
    // style.WindowTitleAlign         = ImVec2(0.0f, 0.5f);
    // style.WindowMenuButtonPosition = ImGuiDir_Right;
    // style.ColorButtonPosition      = ImGuiDir_Right;
    // style.ButtonTextAlign          = ImVec2(0.5f, 0.5f);
    // style.SelectableTextAlign      = ImVec2(0.0f, 0.0f);
}

void UIManager::newFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Update frame time history for statistics
    static float lastFrameTime = 0.0f;
    float currentTime = ImGui::GetTime();
    float deltaTime = currentTime - lastFrameTime;
    lastFrameTime = currentTime;

    m_frameTimeHistory[m_frameTimeIndex] = deltaTime * 1000.0f; // Convert to milliseconds
    m_frameTimeIndex = (m_frameTimeIndex + 1) % IM_ARRAYSIZE(m_frameTimeHistory);
}

void UIManager::update() {
    showMainMenuBar();

    if (m_showStatsWindow) {
        showStatsWindow();
    }

    if (m_showControlPanel) {
        showControlPanel();
    }

    if (m_showSceneSettings) {
        showSceneSettings();
    }

    if (m_showRenderSettings) {
        showRenderSettings();
    }

    if (m_showPhysicsSettings) {
        showPhysicsSettings();
    }

    if (m_showUISettings) {
        showUISettings();
    }
}

void UIManager::render() {
    // Render ImGui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void UIManager::showMainMenuBar() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("New Scene")) {}
            if (ImGui::MenuItem("Open Scene")) {}
            if (ImGui::MenuItem("Save")) {}
            if (ImGui::MenuItem("Save As...")) {}
            ImGui::Separator();
            if (ImGui::MenuItem("Exit")) {
                glfwSetWindowShouldClose(m_window, GLFW_TRUE);
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Statistics", nullptr, &m_showStatsWindow);
            ImGui::MenuItem("Control Panel", nullptr, &m_showControlPanel);
            ImGui::MenuItem("Scene Settings", nullptr, &m_showSceneSettings);
            ImGui::MenuItem("Render Settings", nullptr, &m_showRenderSettings);
            ImGui::MenuItem("Physics Settings", nullptr, &m_showPhysicsSettings);
            ImGui::MenuItem("UI Settings", nullptr, &m_showUISettings);
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Render Mode")) {
            bool is2D = m_renderMode == RenderMode::MODE_2D;
            bool is3D = m_renderMode == RenderMode::MODE_3D;

            if (ImGui::MenuItem("2D Mode", nullptr, &is2D)) {
                if (is2D && m_renderMode != RenderMode::MODE_2D) {
                    m_renderMode = RenderMode::MODE_2D;
                    if (m_modeChangeCallback) {
                        m_modeChangeCallback(m_renderMode);
                    }
                }
            }

            if (ImGui::MenuItem("3D Mode", nullptr, &is3D)) {
                if (is3D && m_renderMode != RenderMode::MODE_3D) {
                    m_renderMode = RenderMode::MODE_3D;
                    if (m_modeChangeCallback) {
                        m_modeChangeCallback(m_renderMode);
                    }
                }
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("Documentation")) {}
            if (ImGui::MenuItem("About")) {}
            ImGui::EndMenu();
        }

        // Right-aligned items
        ImGui::SetCursorPosX(ImGui::GetWindowWidth() - 150);
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);

        ImGui::EndMainMenuBar();
    }
}

void UIManager::showStatsWindow() {
    ImGui::SetNextWindowSize(ImVec2(320, 240), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Statistics", &m_showStatsWindow)) {
        ImGui::Text("Application Performance");
        ImGui::Separator();

        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Text("Frame Time: %.3f ms", 1000.0f / ImGui::GetIO().Framerate);

        ImGui::Separator();
        ImGui::Text("Frame Time History");
        ImGui::PlotLines("##frametime", m_frameTimeHistory, IM_ARRAYSIZE(m_frameTimeHistory),
                         m_frameTimeIndex, "Frame Time (ms)", 0.0f, 33.3f, ImVec2(0, 80));
    }
    ImGui::End();
}

void UIManager::showControlPanel() {
    ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - 330, 30), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320, 400), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Control Panel", &m_showControlPanel)) {
        // Render Mode selection
        ImGui::Text("Render Mode");
        ImGui::Separator();

        if (ImGui::RadioButton("2D Mode", m_renderMode == RenderMode::MODE_2D)) {
            m_renderMode = RenderMode::MODE_2D;
            if (m_modeChangeCallback) {
                m_modeChangeCallback(m_renderMode);
            }
        }

        if (ImGui::RadioButton("3D Mode", m_renderMode == RenderMode::MODE_3D)) {
            m_renderMode = RenderMode::MODE_3D;
            if (m_modeChangeCallback) {
                m_modeChangeCallback(m_renderMode);
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Text("Visual Settings");
        ImGui::Separator();

        // Wireframe mode
        ImGui::Checkbox("Wireframe Mode", &m_wireframeMode);

        // Animation speed
        ImGui::SliderFloat("Animation Speed", &m_animationSpeed, 0.0f, 2.0f);

        // Button to reset settings
        ImGui::Spacing();
        if (ImGui::Button("Reset Settings", ImVec2(120, 30))) {
            m_wireframeMode = false;
            m_animationSpeed = 1.0f;
        }

        ImGui::Spacing();
        ImGui::Separator();

        // Scene controls - vary based on 2D/3D mode
        if (m_renderMode == RenderMode::MODE_2D) {
            ImGui::Text("2D Scene Controls");
            ImGui::Separator();

            static float scale = 1.0f;
            ImGui::SliderFloat("Scale", &scale, 0.1f, 3.0f);

            static float rotation = 0.0f;
            ImGui::SliderFloat("Rotation", &rotation, 0.0f, 360.0f);

            static float position[2] = {0.0f, 0.0f};
            ImGui::SliderFloat2("Position", position, -1.0f, 1.0f);
        }
        else { // 3D Mode
            ImGui::Text("3D Scene Controls");
            ImGui::Separator();

            static float cameraPos[3] = {0.0f, 2.0f, 5.0f};
            ImGui::SliderFloat3("Camera Position", cameraPos, -10.0f, 10.0f);

            static float modelRotation[3] = {0.0f, 0.0f, 0.0f};
            ImGui::SliderFloat3("Model Rotation", modelRotation, 0.0f, 360.0f);

            static float lightPosition[3] = {5.0f, 5.0f, 5.0f};
            ImGui::SliderFloat3("Light Position", lightPosition, -10.0f, 10.0f);
        }
    }
    ImGui::End();
}

void UIManager::showSceneSettings() {
    ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Scene Settings", &m_showSceneSettings)) {
        ImGui::Text("Scene Configuration");
        ImGui::Separator();

        // Object list
        static int selectedObject = 0;
        const char* objects[] = { "Triangle", "Cube", "Sphere", "Plane", "Custom Model" };
        ImGui::ListBox("Objects", &selectedObject, objects, IM_ARRAYSIZE(objects), 4);

        ImGui::Spacing();
        ImGui::Separator();

        // Object properties
        ImGui::Text("Object Properties");

        static float position[3] = {0.0f, 0.0f, 0.0f};
        ImGui::InputFloat3("Position", position);

        static float rotation[3] = {0.0f, 0.0f, 0.0f};
        ImGui::InputFloat3("Rotation", rotation);

        static float scale[3] = {1.0f, 1.0f, 1.0f};
        ImGui::InputFloat3("Scale", scale);

        ImGui::Spacing();

        // Material settings
        ImGui::Text("Material");
        static float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        ImGui::ColorEdit4("Color", color);

        static float roughness = 0.5f;
        ImGui::SliderFloat("Roughness", &roughness, 0.0f, 1.0f);

        static float metallic = 0.0f;
        ImGui::SliderFloat("Metallic", &metallic, 0.0f, 1.0f);
    }
    ImGui::End();
}

void UIManager::showRenderSettings() {
    ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Render Settings", &m_showRenderSettings)) {
        ImGui::Text("Rendering Configuration");
        ImGui::Separator();

        // Resolution
        static int resolution[2] = {1280, 720};
        ImGui::InputInt2("Resolution", resolution);

        // Fullscreen mode
        static bool fullscreen = false;
        if (ImGui::Checkbox("Fullscreen", &fullscreen)) {
            // Handle fullscreen toggle
        }

        // VSync
        static bool vsync = true;
        if (ImGui::Checkbox("VSync", &vsync)) {
            glfwSwapInterval(vsync ? 1 : 0);
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Text("Graphics Settings");

        // Anti-aliasing
        static int msaaSamples = 4;
        const char* msaaOptions[] = { "Off", "2x", "4x", "8x" };
        ImGui::Combo("Anti-aliasing", &msaaSamples, msaaOptions, IM_ARRAYSIZE(msaaOptions));

        // Shadow quality
        static int shadowQuality = 2;
        const char* shadowOptions[] = { "Off", "Low", "Medium", "High", "Ultra" };
        ImGui::Combo("Shadow Quality", &shadowQuality, shadowOptions, IM_ARRAYSIZE(shadowOptions));

        // Texture quality
        static int textureQuality = 2;
        const char* textureOptions[] = { "Low", "Medium", "High", "Ultra" };
        ImGui::Combo("Texture Quality", &textureQuality, textureOptions, IM_ARRAYSIZE(textureOptions));
    }
    ImGui::End();
}

void UIManager::showPhysicsSettings() {
    ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Physics Settings", &m_showPhysicsSettings)) {
        ImGui::Text("Physics Engine Configuration");
        ImGui::Separator();

        // Global settings
        static bool physicsEnabled = true;
        ImGui::Checkbox("Enable Physics", &physicsEnabled);

        static float gravity[3] = {0.0f, -9.81f, 0.0f};
        ImGui::InputFloat3("Gravity", gravity);

        // Simulation settings
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Text("Simulation");

        static float timeStep = 1.0f / 60.0f;
        ImGui::InputFloat("Time Step", &timeStep, 0.001f, 0.01f, "%.3f");

        static int subSteps = 2;
        ImGui::InputInt("Sub Steps", &subSteps);
        subSteps = std::max(1, subSteps);

        // Material presets
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Text("Material Presets");

        static int currentMaterial = 0;
        const char* materials[] = { "Default", "Ice", "Rubber", "Metal", "Wood", "Custom" };
        ImGui::Combo("Material", &currentMaterial, materials, IM_ARRAYSIZE(materials));

        // Material properties
        ImGui::Spacing();
        ImGui::Text("Material Properties");

        static float density = 1.0f;
        ImGui::SliderFloat("Density", &density, 0.1f, 10.0f);

        static float restitution = 0.5f;
        ImGui::SliderFloat("Restitution", &restitution, 0.0f, 1.0f);

        static float friction = 0.5f;
        ImGui::SliderFloat("Friction", &friction, 0.0f, 1.0f);

        // Collision settings
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Text("Collision Settings");

        static bool continuousCollisionDetection = true;
        ImGui::Checkbox("Continuous Collision Detection", &continuousCollisionDetection);

        static int collisionPairs = 32;
        ImGui::InputInt("Max Collision Pairs", &collisionPairs);
    }
    ImGui::End();
}

// New UI Settings window for adjusting UI scale
void UIManager::showUISettings() {
    ImGui::SetNextWindowSize(ImVec2(400, 250), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("UI Settings", &m_showUISettings)) {
        ImGui::Text("UI Configuration");
        ImGui::Separator();

        // Show current UI scale
        ImGui::Text("Current UI Scale: %.1f (%.0f%%)", m_uiScale, m_uiScale * 100.0f);

        ImGui::Spacing();
        ImGui::Separator();

        // UI scale slider for pending scale (to be applied on restart)
        ImGui::Text("New UI Scale (applied after restart):");
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.7f);

        if (ImGui::SliderFloat("##UIScale", &m_pendingScale, 0.5f, 3.0f, "%.1f")) {
            m_settingsChanged = true;
            m_scaleNeedsRestart = m_pendingScale != m_uiScale;
        }

        // Format as percentage
        ImGui::SameLine();
        ImGui::Text("%.0f%%", m_pendingScale * 100.0f);

        // Reset button
        if (ImGui::Button("Reset to Default")) {
            m_pendingScale = 1.0f;
            m_settingsChanged = true;
            m_scaleNeedsRestart = m_pendingScale != m_uiScale;
        }

        ImGui::Spacing();

        // Save button
        if (m_settingsChanged) {
            if (ImGui::Button("Save Settings", ImVec2(150, 30))) {
                saveSettings();

                // Show a message about restart being needed if scale changed
                if (m_scaleNeedsRestart) {
                    ImGui::OpenPopup("RestartNeeded");
                } else {
                    ImGui::OpenPopup("SettingsSaved");
                }
            }

            // Success popup
            if (ImGui::BeginPopupModal("SettingsSaved", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::Text("UI settings saved successfully!");
                ImGui::Separator();

                if (ImGui::Button("OK", ImVec2(120, 0))) {
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }

            // Restart needed popup
            if (ImGui::BeginPopupModal("RestartNeeded", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::Text("UI settings saved successfully!");
                ImGui::Separator();
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "The new UI scale will be applied when you restart the application.");
                ImGui::Separator();

                if (ImGui::Button("Restart Now", ImVec2(120, 0))) {
                    // Request application to close, allowing a restart
                    glfwSetWindowShouldClose(m_window, GLFW_TRUE);
                    ImGui::CloseCurrentPopup();
                }

                ImGui::SameLine();

                if (ImGui::Button("Later", ImVec2(120, 0))) {
                    ImGui::CloseCurrentPopup();
                }

                ImGui::EndPopup();
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextWrapped("Note: UI scaling requires an application restart to take full effect.");
    }
    ImGui::End();
}
