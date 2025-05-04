#pragma once

#include "../render/renderer.h"
#include "../ui/ui_manager.h"
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
    
    // Mode change handler
    void handleModeChange(RenderMode mode);
    
    std::unique_ptr<Renderer> m_renderer;
    std::unique_ptr<UIManager> m_uiManager;
    bool m_running = false;
};