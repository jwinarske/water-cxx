// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#version 450
// Pool vertex shader (surface.glsl cubeVertex). The unit cube's Y is
// remapped so the floor sits at -poolHeight and the rim at 2/12; geometry is world space
// so the only transform is matrix_viewProjection. No model matrix.
#include "surface_push.glsl"

layout(location = 0) in vec3 aPosition;
layout(location = 0) out vec3 vPosition;

const float poolHeight = 1.0;

void main() {
  vec3 p = aPosition;
  p.y = ((1.0 - p.y) * (7.0 / 12.0) - 1.0) * poolHeight;
  vPosition = p;
  gl_Position = pc.vp * vec4(p, 1.0);
}
