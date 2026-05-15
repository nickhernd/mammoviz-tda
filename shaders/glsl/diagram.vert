#version 460 core

// Renders the persistence diagram as a scatter plot
// Each point = (birth, death) of a topological feature
// Points above the diagonal y=x are real features; at diagonal = noise

layout(location = 0) in vec2  a_birth_death;  // (birth_ε, death_ε)
layout(location = 1) in int   a_dimension;    // 0=β₀, 1=β₁, 2=β₂
layout(location = 2) in float a_persistence;  // death - birth

uniform mat4  u_ortho;           // orthographic projection for 2D panel
uniform float u_min_persistence; // points below this are drawn smaller
uniform int   u_selected_dim;    // -1 = show all

out float v_persistence;
out vec3  v_color;
out float v_alpha;

vec3 dimensionColor(int dim) {
    if (dim == 0) return vec3(0.2, 0.5, 1.0);  // β₀ = blue
    if (dim == 1) return vec3(0.2, 0.9, 0.3);  // β₁ = green
               return vec3(1.0, 0.3, 0.2);      // β₂ = red
}

void main() {
    v_persistence = a_persistence;
    v_color       = dimensionColor(a_dimension);
    v_alpha       = (a_persistence >= u_min_persistence) ? 1.0 : 0.2;

    if (u_selected_dim != -1 && a_dimension != u_selected_dim)
        v_alpha *= 0.1;

    // Point size proportional to persistence (more persistent = more important)
    gl_PointSize = 4.0 + clamp(a_persistence * 20.0, 0.0, 16.0);
    gl_Position  = u_ortho * vec4(a_birth_death, 0.0, 1.0);
}
