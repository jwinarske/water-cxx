// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#version 450
// S3 update (water.js updateShader): one explicit-Euler step of the
// height-field wave equation. 4-tap Laplacian drives velocity (G), damped *0.995, then
// integrated into height (R). Run 2x per frame.
layout(set = 0, binding = 0) uniform sampler2D uSource;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push { vec2 delta; } pc;  // 1 / textureSize

// NaN guard: hard clamp so a 16F overflow (RGBA16F path on Mali/V3DV/Adreno) can't
// produce an Inf/NaN that poisons the ping-pong feedback field permanently. Normal
// water values are < 1; this bound only catches a runaway, far below 16F's 65504 max.
const float LIMIT = 1000.0;

void main() {
  ivec2 sz = textureSize(uSource, 0);
  vec2 coord = gl_FragCoord.xy / vec2(sz);
  vec4 info = texture(uSource, coord);

  vec2 dx = vec2(pc.delta.x, 0.0);
  vec2 dy = vec2(0.0, pc.delta.y);
  float average = (texture(uSource, coord - dx).r + texture(uSource, coord - dy).r +
                   texture(uSource, coord + dx).r + texture(uSource, coord + dy).r) * 0.25;

  info.g += (average - info.r) * 2.0;   // acceleration toward neighbor mean
  info.g *= 0.995;                       // viscous damping
  info.r += info.g;                      // integrate height

  info.r = clamp(info.r, -LIMIT, LIMIT);
  info.g = clamp(info.g, -LIMIT, LIMIT);
  outColor = info;
}
