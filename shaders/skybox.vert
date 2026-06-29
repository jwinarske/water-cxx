// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#version 450
// Skybox background: a full-screen triangle whose per-pixel world-space view ray is
// reconstructed from the inverse view-projection, used to sample the sky cubemap. The
// scene pass's negative-height viewport is consistent because the same glm-convention
// NDC is both emitted here and inverted by invVP. Validates the cubemap load + orientation
// that 's water reflections depend on.
layout(location = 0) out vec3 vDir;
layout(push_constant) uniform Push {
  mat4 inv_vp;
  vec4 eye;
} pc;

void main() {
  vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
  vec2 ndc = p * 2.0 - 1.0;
  vec4 world_far = pc.inv_vp * vec4(ndc, 1.0, 1.0);
  vDir = world_far.xyz / world_far.w - pc.eye.xyz;
  gl_Position = vec4(ndc, 1.0, 1.0);  // depth = far (skybox sits behind all geometry)
}
