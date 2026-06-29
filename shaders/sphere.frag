// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#version 450
// Sphere fragment (surface.glsl sphereFragment): getSphereColor
// (caustic-lit), tinted underwater where the surface is above the point.
#include "surface_common.glsl"

layout(location = 0) in vec3 vPosition;
layout(location = 0) out vec4 outColor;

void main() {
  vec4 color = vec4(getSphereColor(vPosition), 1.0);
  vec4 info = texture(water, vPosition.xz * 0.5 + 0.5);
  if (vPosition.y < info.r) color.rgb *= underwaterColor * 1.2;
  outColor = color;
}
