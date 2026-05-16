#version 460 core

in  vec3 v_ray_dir;
in  vec3 v_ray_origin;
out vec4 frag_color;

uniform sampler3D u_volume;
uniform sampler3D u_saliency;
uniform int       u_show_gradcam;
uniform float     u_step_size;
uniform vec3      u_vol_size;

// ── Per-layer controls ────────────────────────────────────────────────────────
// Fat layer      (T1 bright: ~0.05–0.38)
uniform vec3  u_fat_color;
uniform float u_fat_opacity;
uniform int   u_fat_visible;

// Fibroglandular (medium: ~0.38–0.60)
uniform vec3  u_gland_color;
uniform float u_gland_opacity;
uniform int   u_gland_visible;

// Dense / high-signal tissue (~0.60–0.82)
uniform vec3  u_dense_color;
uniform float u_dense_opacity;
uniform int   u_dense_visible;

// Calcifications / hyper-intense spots (>0.82)
uniform vec3  u_calc_color;
uniform float u_calc_opacity;
uniform int   u_calc_visible;

// GradCAM saliency overlay
uniform float u_lesion_sal_thresh;  // saliency value above which we paint as lesion
uniform float u_lesion_opacity;
uniform int   u_lesion_visible;

// Clipping planes (axis-aligned; voxel discarded when pos > threshold)
uniform float u_clip_x;   // 0..1  (1.0 = no clip)
uniform float u_clip_y;   // 0..1
uniform float u_clip_z;   // 0..1

// ── Ray–box intersection ──────────────────────────────────────────────────────
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
    if (!intersectBox(ro, rd, t_near, t_far)) { discard; return; }

    vec4  color = vec4(0.0);
    float t     = max(t_near, 0.0);
    float dt    = u_step_size;

    for (int i = 0; i < 1200 && t < t_far && color.a < 0.99; ++i) {
        vec3  pos = ro + t * rd;

        // Apply clipping planes — skip voxels outside the visible region
        if (pos.x > u_clip_x || pos.y > u_clip_y || pos.z > u_clip_z) {
            t += dt;
            continue;
        }

        float intensity = texture(u_volume, pos).r;
        float sal       = (u_show_gradcam != 0) ? texture(u_saliency, pos).r : 0.0;

        float alpha = 0.0;
        vec3  rgb   = vec3(0.0);
        bool  hit   = false;

        // ── Fat layer ─────────────────────────────────────────────────────────
        if (u_fat_visible != 0 && intensity > 0.05 && intensity <= 0.38) {
            float f = smoothstep(0.05, 0.20, intensity) *
                      (1.0 - smoothstep(0.30, 0.38, intensity));
            alpha = f * 0.18 * u_fat_opacity;
            rgb   = u_fat_color;
            hit   = true;
        }

        // ── Fibroglandular layer ──────────────────────────────────────────────
        if (u_gland_visible != 0 && intensity > 0.38 && intensity <= 0.60) {
            float f = smoothstep(0.38, 0.48, intensity) *
                      (1.0 - smoothstep(0.55, 0.60, intensity));
            alpha = f * 0.28 * u_gland_opacity;
            rgb   = u_gland_color;
            hit   = true;
        }

        // ── Dense tissue layer ────────────────────────────────────────────────
        if (u_dense_visible != 0 && intensity > 0.60 && intensity <= 0.82) {
            float f = smoothstep(0.60, 0.68, intensity) *
                      (1.0 - smoothstep(0.76, 0.82, intensity));
            alpha = f * 0.55 * u_dense_opacity;
            rgb   = u_dense_color;
            hit   = true;
        }

        // ── Calcification / hyper-intense ─────────────────────────────────────
        if (u_calc_visible != 0 && intensity > 0.82) {
            float f = smoothstep(0.82, 0.90, intensity);
            alpha = f * 0.90 * u_calc_opacity;
            rgb   = u_calc_color;
            hit   = true;
        }

        // ── GradCAM / lesion overlay ──────────────────────────────────────────
        // Shown independently of tissue layers — volumetric "heat glow"
        if (u_lesion_visible != 0 && sal > u_lesion_sal_thresh) {
            float t_sal = (sal - u_lesion_sal_thresh) / max(1.0 - u_lesion_sal_thresh, 0.01);
            t_sal = clamp(t_sal, 0.0, 1.0);
            vec3 hot = mix(vec3(1.0, 0.55, 0.0), vec3(1.0, 0.02, 0.0), t_sal);
            if (hit) {
                // Overlay on existing tissue: tint towards red
                rgb   = mix(rgb, hot, t_sal * 0.75 * u_lesion_opacity);
                alpha = max(alpha, t_sal * 0.50 * u_lesion_opacity);
            } else {
                // No tissue here: render as transparent hot glow
                rgb   = hot;
                alpha = t_sal * t_sal * 0.30 * u_lesion_opacity;
            }
        }

        // Front-to-back accumulation (pre-multiplied alpha)
        vec4 rgba = vec4(rgb * alpha, alpha);
        color += (1.0 - color.a) * rgba;
        t += dt;
    }

    frag_color = color;
}
