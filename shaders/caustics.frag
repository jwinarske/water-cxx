// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#version 450
// Caustics fragment (surface.glsl causticsFragment). Differential-area:
// the projected cell that shrank (newArea < oldArea) focused light -> brighter. Plus the
// sphere blob shadow (G) and the pool-rim shadow term (modulates R). Opaque last-write,
// NO additive blend (— the area ratio already encodes focusing).
#include "surface_push.glsl"
#include "intersect.glsl"

layout(location = 0) in vec3 oldPos;
layout(location = 1) in vec3 newPos;
layout(location = 0) out vec4 outColor;

void main() {
  // if the triangle gets smaller, it gets brighter, and vice versa
  float oldArea = length(dFdx(oldPos)) * length(dFdy(oldPos));
  float newArea = length(dFdx(newPos)) * length(dFdy(newPos));
  // min(3) caps the focus gain; max(1e-9) guards the divide on degenerate facets.
  float caustic = min(oldArea / max(newArea, 1.0e-9), 3.0) * 0.2;
  vec4 color = vec4(caustic, 1.0, 0.0, 0.0);

  vec3 refractedLight = refract(-pc.lightDir.xyz, vec3(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);

  // sphere blob shadow (only where the ball blocks the light)
  vec3 dir = (pc.sphereInfo.xyz - newPos) / pc.sphereInfo.w;
  vec3 area = cross(dir, refractedLight);
  float shadow = dot(area, area);
  float dist = dot(dir, -refractedLight);
  shadow = 1.0 + (shadow - 1.0) / (0.05 + dist * 0.025);
  shadow = clamp(1.0 / (1.0 + exp(-shadow)), 0.0, 1.0);
  shadow = mix(1.0, shadow, clamp(dist * 2.0, 0.0, 1.0));
  color.g = shadow;

  // shadow for the rim of the pool
  vec2 t = intersectCube(newPos, -refractedLight, vec3(-1.0, -poolHeight, -1.0), vec3(1.0, 2.0, 1.0));
  color.r *= 1.0 / (1.0 + exp(-200.0 / (1.0 + 10.0 * (t.y - t.x)) *
                              (newPos.y - refractedLight.y * t.y - 2.0 / 12.0)));

  outColor = color;
}
