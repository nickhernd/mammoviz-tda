#version 460 core

// Renders the 3D UMAP latent space manifold
// Each point = one patient sample projected from high-dim CNN activations

layout(location = 0) in vec3  a_pos;         // UMAP 3D coordinates
layout(location = 1) in float a_confidence;  // model prediction confidence [0,1]
layout(location = 2) in int   a_label;       // ground truth: 0=benign, 1=malignant
layout(location = 3) in float a_selected;    // 1.0 if this sample is selected

uniform mat4 u_mvp;

out float v_confidence;
out vec3  v_color;
out float v_selected;

vec3 labelColor(int label, float conf) {
    vec3 benign    = vec3(0.2, 0.7, 0.4);   // green for benign
    vec3 malignant = vec3(0.9, 0.2, 0.2);   // red for malignant
    vec3 base      = (label == 0) ? benign : malignant;
    // Desaturate low-confidence predictions
    return mix(vec3(0.5), base, conf);
}

void main() {
    v_confidence = a_confidence;
    v_color      = labelColor(a_label, a_confidence);
    v_selected   = a_selected;

    gl_PointSize = (a_selected > 0.5) ? 14.0 : 6.0;
    gl_Position  = u_mvp * vec4(a_pos, 1.0);
}
