// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#version 450
// Sphere vertex shader (surface.glsl sphereVertex). Scales the unit
// sphere by radius and offsets to the center, in the vertex stage. World space; only
// matrix_viewProjection. No model matrix.
#include "surface_push.glsl"

layout(location = 0) in vec3 aPosition;
layout(location = 0) out vec3 vPosition;

void main() {
  vec3 p = pc.sphereInfo.xyz + aPosition * pc.sphereInfo.w;
  vPosition = p;
  gl_Position = pc.vp * vec4(p, 1.0);
}
