#version 460 core

layout(location = 0) in vec3 a_pos;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec3 u_camera_pos;

out vec3 v_ray_dir;
out vec3 v_ray_origin;

void main() {
    vec4 world_pos = u_model * vec4(a_pos, 1.0);
    // Ray origin is camera; ray direction is from camera to vertex
    v_ray_origin = u_camera_pos;
    v_ray_dir    = world_pos.xyz - u_camera_pos;
    gl_Position  = u_proj * u_view * world_pos;
}
