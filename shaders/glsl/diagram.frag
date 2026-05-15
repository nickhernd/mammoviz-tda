#version 460 core

in  float v_persistence;
in  vec3  v_color;
in  float v_alpha;
out vec4  frag_color;

void main() {
    // Circular point (discard corners of the gl_PointSize quad)
    vec2  coord = gl_PointCoord - vec2(0.5);
    float r     = dot(coord, coord);
    if (r > 0.25) discard;

    // Soft edge
    float edge  = 1.0 - smoothstep(0.20, 0.25, r);
    frag_color  = vec4(v_color, v_alpha * edge);
}
