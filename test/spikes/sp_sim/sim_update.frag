// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#version 450
// Fragment ping-pong update pass (water.js updateFragment):
// 4-tap neighbor gather, velocity integrate + damp, height step. R=height, G=velocity.
layout(set = 0, binding = 0) uniform sampler2D uSource;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push { vec2 delta; } pc;  // 1/textureSize

void main() {
  ivec2 sz = textureSize(uSource, 0);
  vec2 uv = gl_FragCoord.xy / vec2(sz);
  vec4 info = texture(uSource, uv);
  vec2 dx = vec2(pc.delta.x, 0.0);
  vec2 dy = vec2(0.0, pc.delta.y);
  float avg = (texture(uSource, uv - dx).r + texture(uSource, uv - dy).r +
               texture(uSource, uv + dx).r + texture(uSource, uv + dy).r) * 0.25;
  info.g += (avg - info.r) * 2.0;
  info.g *= 0.995;
  info.r += info.g;
  outColor = info;
}
