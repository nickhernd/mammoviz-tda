#version 460 core

in  float v_confidence;
in  vec3  v_color;
in  float v_selected;
out vec4  frag_color;

void main() {
    vec2  coord = gl_PointCoord - vec2(0.5);
    float r     = dot(coord, coord);
    if (r > 0.25) discard;

    float edge = 1.0 - smoothstep(0.18, 0.25, r);

    vec3 color = v_color;
    if (v_selected > 0.5) {
        // Selected sample: white ring outline
        float ring = smoothstep(0.18, 0.20, r) * (1.0 - smoothstep(0.22, 0.25, r));
        color = mix(color, vec3(1.0), ring * 3.0);
    }

    frag_color = vec4(color, edge);
}
