#include "application.h"
#include <stdexcept>
#include <iostream>

Application::Application() {
    initialize();
}

Application::~Application() {
    shutdown();
}

void Application::initialize() {
    try {
        // Initialize renderer first
        m_renderer = std::make_unique<Renderer>();

        // Initialize UI manager
        m_uiManager = std::make_unique<UIManager>(m_renderer->getWindow());

        // Set up mode change callback
        m_uiManager->setModeChangeCallback([this](RenderMode mode) {
            this->handleModeChange(mode);
        });

        m_running = true;
        std::cout << "Application initialized successfully" << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize application: " + std::string(e.what()));
    }
}

void Application::handleModeChange(RenderMode mode) {
    std::cout << "Switching to " << (mode == RenderMode::MODE_2D ? "2D" : "3D") << " mode" << std::endl;
    m_renderer->setRenderMode(mode);
}

void Application::run() {
    while (m_running) {
        update();
        render();

        // Check if window should close
        if (m_renderer->windowShouldClose()) {
            m_running = false;
        }
    }
}

void Application::update() {
    // Process input events
    m_renderer->processInput();

    // Update renderer with UI settings
    m_renderer->setWireframeMode(m_uiManager->isWireframeMode());
    m_renderer->setAnimationSpeed(m_uiManager->getAnimationSpeed());

    // Create a new ImGui frame
    m_uiManager->newFrame();

    // Update UI components
    m_uiManager->update();

    // Future physics update will go here
}

void Application::render() {
    // Render the scene
    m_renderer->render();

    // Render the UI on top
    m_uiManager->render();

    glfwSwapBuffers(m_renderer->getWindow());
    glfwPollEvents();
}

void Application::shutdown() {
    // Destroy UI manager first
    m_uiManager.reset();

    // Then destroy renderer
    m_renderer.reset();

    std::cout << "Application shut down successfully" << std::endl;
}
