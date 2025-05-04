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
        m_renderer = std::make_unique<Renderer>();
        m_running = true;
        std::cout << "Application initialized successfully" << std::endl;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize application: " + std::string(e.what()));
    }
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
    m_renderer->processInput();
    // Future physics update will go here
}

void Application::render() {
    m_renderer->render();
}

void Application::shutdown() {
    m_renderer.reset();
    std::cout << "Application shut down successfully" << std::endl;
}