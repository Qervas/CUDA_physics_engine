#pragma once

#include "../render/renderer.h"
#include <memory>

class Application {
public:
    Application();
    ~Application();
    
    void run();
    
private:
    void initialize();
    void update();
    void render();
    void shutdown();
    
    std::unique_ptr<Renderer> m_renderer;
    bool m_running = false;
};