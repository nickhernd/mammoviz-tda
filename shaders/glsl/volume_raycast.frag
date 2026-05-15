#version 460 core

// ── Volume Ray Casting Fragment Shader ──────────────────────────────────────
// Implements front-to-back compositing DVR with optional GradCAM overlay
// Ray marching through 3D texture (the medical volume)

in  vec3 v_ray_dir;
in  vec3 v_ray_origin;
out vec4 frag_color;

uniform sampler3D u_volume;     // normalized intensity [0,1]
uniform sampler3D u_saliency;   // GradCAM map [0,1]
uniform sampler1D u_transfer;   // transfer function LUT (256 entries)
uniform bool      u_show_gradcam;
uniform float     u_step_size;  // ray march step (smaller = higher quality)
uniform vec3      u_vol_size;   // volume dimensions for normalization

// AABB intersection for the unit cube [0,1]³
bool intersectBox(vec3 ro, vec3 rd, out float t_near, out float t_far) {
    vec3 inv = 1.0 / rd;
    vec3 t0  = (vec3(0.0) - ro) * inv;
    vec3 t1  = (vec3(1.0) - ro) * inv;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    t_near = max(max(tmin.x, tmin.y), tmin.z);
    t_far  = min(min(tmax.x, tmax.y), tmax.z);
    return t_near < t_far && t_far > 0.0;
}

vec4 transferFunction(float intensity) {
    return texture(u_transfer, intensity);
}

void main() {
    vec3 rd = normalize(v_ray_dir);
    vec3 ro = v_ray_origin;

    float t_near, t_far;
    if (!intersectBox(ro, rd, t_near, t_far)) {
        frag_color = vec4(0.0);
        return;
    }

    // Front-to-back compositing
    vec4  color      = vec4(0.0);
    float t          = max(t_near, 0.0);
    float step       = u_step_size;

    while (t < t_far && color.a < 0.99) {
        vec3  pos       = ro + t * rd;                    // [0,1]³ position
        float intensity = texture(u_volume, pos).r;
        vec4  sample    = transferFunction(intensity);

        if (u_show_gradcam) {
            float sal = texture(u_saliency, pos).r;
            // Hot colormap overlay: blue→yellow→red for saliency
            vec3 hot = mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), sal);
            sample.rgb = mix(sample.rgb, hot, sal * 0.7);
        }

        // Pre-multiplied alpha front-to-back compositing
        sample.rgb *= sample.a;
        color += (1.0 - color.a) * sample;
        t     += step;
    }

    frag_color = color;
}
