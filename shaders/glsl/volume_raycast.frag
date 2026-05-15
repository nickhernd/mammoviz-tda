#version 460 core

in  vec3 v_ray_dir;
in  vec3 v_ray_origin;
out vec4 frag_color;

uniform sampler3D u_volume;
uniform sampler3D u_saliency;
uniform sampler1D u_transfer;
uniform int       u_show_gradcam;
uniform float     u_step_size;
uniform vec3      u_vol_size;

bool intersectBox(vec3 ro, vec3 rd, out float t_near, out float t_far) {
    vec3 inv_rd = vec3(
        (abs(rd.x) < 1e-6) ? 1e6 : 1.0 / rd.x,
        (abs(rd.y) < 1e-6) ? 1e6 : 1.0 / rd.y,
        (abs(rd.z) < 1e-6) ? 1e6 : 1.0 / rd.z
    );
    vec3 t0   = (vec3(0.0) - ro) * inv_rd;
    vec3 t1   = (vec3(1.0) - ro) * inv_rd;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    t_near = max(max(tmin.x, tmin.y), tmin.z);
    t_far  = min(min(tmax.x, tmax.y), tmax.z);
    return t_near < t_far && t_far > 0.0;
}

void main() {
    vec3 rd = normalize(v_ray_dir);
    vec3 ro = v_ray_origin;

    float t_near, t_far;
    if (!intersectBox(ro, rd, t_near, t_far)) {
        discard;
        return;
    }

    vec4  color = vec4(0.0);
    float t     = max(t_near, 0.0);
    float dt    = u_step_size;

    for (int i = 0; i < 1000 && t < t_far && color.a < 0.99; ++i) {
        vec3  pos       = ro + t * rd;
        float intensity = texture(u_volume, pos).r;

        // Transfer function: opacity mapped strongly so even low-intensity tissue shows
        float alpha = 0.0;
        vec3  rgb   = vec3(0.0);

        if (intensity > 0.01) {
            // fat: dim grey
            alpha = mix(0.0, 0.15, smoothstep(0.01, 0.15, intensity));
            rgb   = vec3(0.4, 0.35, 0.3);
        }
        if (intensity > 0.30) {
            // glandular: lighter grey-white
            alpha = mix(0.15, 0.35, smoothstep(0.30, 0.55, intensity));
            rgb   = vec3(0.75, 0.70, 0.65);
        }
        if (intensity > 0.55) {
            // skin: bright white
            alpha = mix(0.35, 0.80, smoothstep(0.55, 0.70, intensity));
            rgb   = vec3(0.95, 0.92, 0.88);
        }
        if (intensity > 0.85) {
            // calcifications: pure white / bright yellow
            alpha = 0.95;
            rgb   = vec3(1.0, 0.95, 0.6);
        }

        if (u_show_gradcam != 0) {
            float sal = texture(u_saliency, pos).r;
            if (sal > 0.05) {
                vec3 hot = mix(vec3(0.0, 0.3, 1.0), vec3(1.0, 0.1, 0.0), sal);
                rgb = mix(rgb, hot, sal * 0.6);
                alpha = max(alpha, sal * 0.5);
            }
        }

        // Front-to-back accumulation (pre-multiplied)
        vec4 rgba = vec4(rgb * alpha, alpha);
        color += (1.0 - color.a) * rgba;
        t += dt;
    }

    frag_color = color;
}
