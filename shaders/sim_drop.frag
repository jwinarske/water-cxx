// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#version 450
// S1 drop / line (water.js dropShader generalized to a segment so
// one pass serves both addDrop (center==center2) and addLine). Raises height in a
// cosine-shaped disc/capsule of `radius` around the segment by `strength`.
layout(set = 0, binding = 0) uniform sampler2D uSource;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push {
  vec2 center;    // [-1,1] clip-ish space, as upstream
  vec2 center2;   // == center for a point drop
  float radius;
  float strength;
} pc;

const float PI = 3.141592653589793;

void main() {
  ivec2 sz = textureSize(uSource, 0);
  vec2 coord = gl_FragCoord.xy / vec2(sz);
  vec4 info = texture(uSource, coord);

  // Distance from this texel to the segment [a,b] in [0,1] space.
  vec2 a = pc.center * 0.5 + 0.5;
  vec2 b = pc.center2 * 0.5 + 0.5;
  vec2 ab = b - a;
  vec2 ap = coord - a;
  float t = clamp(dot(ap, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
  float dist = length(ap - ab * t);

  float drop = max(0.0, 1.0 - dist / pc.radius);
  drop = 0.5 - cos(drop * PI) * 0.5;          // smooth cosine falloff
  info.r += drop * pc.strength;
  outColor = info;
}
