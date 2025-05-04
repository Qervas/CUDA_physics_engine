cmake_minimum_required(VERSION 3.18)
project(PhysicsEngine LANGUAGES CXX CUDA)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17) # CUDA 12.9 supports C++17 features
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(CUDA 12.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture based on your GPU
set(CMAKE_CUDA_ARCHITECTURES 75) # Adjust this for your specific GPU architecture

# Include directories
include_directories(
    ${OPENGL_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    src
)

# Source files
set(SOURCES
    src/main.cpp
    src/render/renderer.cpp
    src/core/application.cpp
)

set(CUDA_SOURCES
    src/physics/physics.cu
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES}
    glfw
    CUDA::cublas
    CUDA::cudart
)

# Set installation directory
install(TARGETS ${PROJECT_NAME} DESTINATION bin)

# Enable CUDA separable compilation
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Create shader directory in binary directory and copy shaders
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)
file(COPY ${CMAKE_SOURCE_DIR}/shaders/ DESTINATION ${CMAKE_BINARY_DIR}/shaders/)