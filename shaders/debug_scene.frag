// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#version 450
// / debug fragment: flat-shade by screen-space derivatives of world position so the
// pool faces and sphere read as 3D geometry and depth occlusion is visible. The real
// ray-traced shading is pool.frag / sphere.frag.
#include "surface_push.glsl"

layout(location = 0) in vec3 vPosition;
layout(location = 0) out vec4 outColor;

void main() {
  vec3 n = normalize(cross(dFdx(vPosition), dFdy(vPosition)));
  float lambert = 0.3 + 0.7 * abs(dot(n, normalize(pc.lightDir.xyz)));
  outColor = vec4(vec3(0.70, 0.75, 0.85) * lambert, 1.0);
}
