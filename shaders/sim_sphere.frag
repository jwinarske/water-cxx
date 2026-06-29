// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#version 450
// S2 sphere (water.js sphereShader): displace water by the volume
// the sphere swept as it moved from oldCenter to newCenter. Adds the old footprint,
// subtracts the new — conserving the displaced volume as the ball pushes through.
layout(set = 0, binding = 0) uniform sampler2D uSource;
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push {
  vec3 oldCenter;   // world space (x, y, z); pool spans [-1,1] in x/z
  vec3 newCenter;
  float radius;
} pc;

float volumeInSphere(vec2 coord, vec3 center) {
  vec3 toCenter = vec3(coord.x * 2.0 - 1.0, 0.0, coord.y * 2.0 - 1.0) - center;
  float t = length(toCenter) / pc.radius;
  float dy = exp(-pow(t * 1.5, 2.0));
  float ymin = min(0.0, center.y - dy);
  float ymax = min(max(0.0, center.y + dy), ymin + 2.0 * dy);
  return (ymax - ymin) * 0.1;
}

void main() {
  ivec2 sz = textureSize(uSource, 0);
  vec2 coord = gl_FragCoord.xy / vec2(sz);
  vec4 info = texture(uSource, coord);
  info.r += volumeInSphere(coord, pc.oldCenter);
  info.r -= volumeInSphere(coord, pc.newCenter);
  outColor = info;
}
