#version 430 core
layout (local_size_x = 16, local_size_y = 16) in;
layout (rgba32f, binding = 0) uniform image2D outputImage;

// Time uniform
uniform float u_time;

void main() {
    // Get the global invocation ID
    ivec2 texCoord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(outputImage);
    
    if(texCoord.x >= size.x || texCoord.y >= size.y) {
        return;
    }
    
    // Normalized position (0.0 to 1.0)
    vec2 uv = vec2(texCoord) / vec2(size);
    
    // Create a simple color pattern (a triangle)
    vec4 color = vec4(0.1, 0.1, 0.1, 1.0);
    
    // Create triangle bounds
    vec2 center = vec2(0.5, 0.5);
    float radius = 0.3;
    
    // Animate triangle vertices
    float angle = u_time * 0.5;
    vec2 p1 = center + radius * vec2(cos(angle), sin(angle));
    vec2 p2 = center + radius * vec2(cos(angle + 2.0 * 3.14159 / 3.0), sin(angle + 2.0 * 3.14159 / 3.0));
    vec2 p3 = center + radius * vec2(cos(angle + 4.0 * 3.14159 / 3.0), sin(angle + 4.0 * 3.14159 / 3.0));
    
    // Test if point is inside triangle
    vec2 v0 = p2 - p1;
    vec2 v1 = p3 - p1;
    vec2 v2 = uv - p1;
    
    float dot00 = dot(v0, v0);
    float dot01 = dot(v0, v1);
    float dot02 = dot(v0, v2);
    float dot11 = dot(v1, v1);
    float dot12 = dot(v1, v2);
    
    float invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    
    if ((u >= 0.0) && (v >= 0.0) && (u + v < 1.0)) {
        // Inside triangle, create a nice gradient
        color = vec4(u, v, 1.0 - u - v, 1.0);
    }
    
    // Store the computed color in the output image
    imageStore(outputImage, texCoord, color);
}