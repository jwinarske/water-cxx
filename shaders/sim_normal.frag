// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#version 450
// S4 normal (water.js normalShader): reconstruct the surface normal
// from height slopes and store its xz in B/A. normal.y is recovered in the surface
// shader as sqrt(max(0, 1 - dot(ba,ba))) (the NAN guard lives there). Run 1x/frame
// after the two update steps.
layout(set = 0, binding = 0) uniform sampler2D uSource;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push { vec2 delta; } pc;

void main() {
  ivec2 sz = textureSize(uSource, 0);
  vec2 coord = gl_FragCoord.xy / vec2(sz);
  vec4 info = texture(uSource, coord);

  vec3 ddx = vec3(pc.delta.x,
                  texture(uSource, vec2(coord.x + pc.delta.x, coord.y)).r - info.r, 0.0);
  vec3 ddy = vec3(0.0,
                  texture(uSource, vec2(coord.x, coord.y + pc.delta.y)).r - info.r, pc.delta.y);
  info.ba = normalize(cross(ddy, ddx)).xz;   // B = normal.x, A = normal.z
  outColor = info;
}
