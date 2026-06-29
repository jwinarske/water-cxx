// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// Shared surface push-constant block. Included by every scene
// shader stage so the layout is identical across vert/frag. 112 B <= 128.
layout(push_constant) uniform Push {
  mat4 vp;          // matrix_viewProjection
  vec4 eyePos;      // xyz = camera world position
  vec4 lightDir;    // xyz = normalized light direction
  vec4 sphereInfo;  // xyz = sphere center, w = radius
} pc;
