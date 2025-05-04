#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

class PhysicsEngine {
public:
    PhysicsEngine();
    ~PhysicsEngine();
    
    void initialize();
    void update(float deltaTime);
    
private:
    // CUDA resources
    cublasHandle_t m_cublasHandle;
    bool m_initialized = false;
    
    // Physics state
    // These will be expanded as we develop the physics engine
};