# CUDA Physics Engine with OpenGL & Compute Shader Rendering

A high-performance physics engine that leverages CUDA for physics calculations and modern OpenGL (4.3+) with compute shaders for rendering. This project aims to demonstrate efficient GPU utilization for both computation and visualization.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![OpenGL](https://img.shields.io/badge/OpenGL-4.3+-yellow.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.9-green.svg)
![C++](https://img.shields.io/badge/C++-20-orange.svg)

## Features

### Current
- Modern OpenGL (4.3+) rendering pipeline with compute shaders
- GPU-accelerated rendering using compute shaders
- Real-time animation and visualization
- C++20 features for modern, efficient code
- CMake-based build system

### Planned
- CUDA-accelerated physics calculations using cuBLAS
- Rigid body dynamics simulation
- Particle systems
- Collision detection and response
- Interactive demo scenes
- Performance profiling and optimization tools

## Requirements

### System Requirements
- Graphics card with OpenGL 4.3+ support
- NVIDIA GPU with CUDA compute capability 6.0 or higher
- 2GB+ GPU memory recommended
- Modern multi-core CPU
- 4GB+ RAM

### Software Dependencies
- Ubuntu 20.04 or newer (other Linux distros should work too)
- GCC 13 or higher (for C++20 support)
- CUDA Toolkit 12.9
- CMake 3.18+
- GLFW3
- GLEW
- GLM (OpenGL Mathematics)

## Building from Source

1. **Install Dependencies**
   ```bash
   # Update package list
   sudo apt update
   
   # Install build tools and libraries
   sudo apt install build-essential cmake git
   sudo apt install libglfw3-dev libglew-dev libgl1-mesa-dev
   sudo apt install nvidia-cuda-toolkit
   ```

2. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/physics-engine.git
   cd physics-engine
   ```

3. **Configure and Build**
   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ```

4. **Run the Demo**
   ```bash
   ./PhysicsEngine
   ```

## Project Structure

```
physics-engine/
├── CMakeLists.txt           # Main CMake configuration
├── src/
│   ├── core/               # Core application framework
│   │   ├── application.h
│   │   └── application.cpp
│   ├── render/            # Graphics rendering system
│   │   ├── renderer.h
│   │   └── renderer.cpp
│   └── physics/           # CUDA physics implementation
│       ├── physics.h
│       └── physics.cu
├── shaders/               # GLSL shader files
│   ├── compute.glsl      # Main compute shader
│   ├── quad.vert        # Vertex shader for display
│   └── quad.frag        # Fragment shader for display
└── tests/                # Test files (planned)
```

## Development

### Code Style
- Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Use modern C++ features (C++20)
- Document all public APIs
- Keep functions focused and small
- Use meaningful variable names

### Git Workflow
1. Create a new branch for each feature/bugfix
2. Write clear, concise commit messages
3. Keep commits focused and logical
4. Create pull requests for review
5. Squash commits before merging

### Debug Tools
- Use `nvidia-smi` to monitor GPU usage
- Enable OpenGL debug output for graphics debugging
- CUDA debugging available through `cuda-gdb`
- Performance profiling with NVIDIA Nsight

## Current Implementation

The application demonstrates compute shader rendering:

1. **Compute Shader Pipeline**
   - Compute shader generates triangle visualization
   - Output stored in texture
   - Displayed via full-screen quad

2. **Features**
   - Real-time animation of rotating triangle
   - GPU-accelerated pixel calculations
   - Efficient texture-based display

3. **Technical Details**
   - Uses OpenGL 4.3 compute shaders
   - 16x16 workgroup size
   - RGBA32F texture format
   - Double-buffered display

## Controls

- `ESC` - Exit the application
- More controls coming soon...

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and suggestions:
- Create an issue in the GitHub repository
- Submit a pull request with improvements
- Contact the maintainers directly

## Acknowledgments

- OpenGL Compute Shader Documentation
- NVIDIA CUDA Documentation
- CUDA and OpenGL integration examples
- Open source physics engine implementations