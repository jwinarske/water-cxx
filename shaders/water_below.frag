// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#version 450
// Water surface seen from below (surface.glsl waterBelowFragment):
// inverted normal, water->air IOR (total internal reflection via refractedRay length),
// underwater tint.
#include "water_common.glsl"

layout(location = 0) out vec4 outColor;

void main() {
  vec3 normal, incomingRay;
  computeNormalAndRay(normal, incomingRay);
  normal = -normal;

  vec3 reflectedRay = reflect(incomingRay, normal);
  vec3 refractedRay = refract(incomingRay, normal, IOR_WATER / IOR_AIR);
  float fresnel = mix(0.5, 1.0, pow(1.0 - dot(normal, -incomingRay), 3.0));

  vec3 reflectedColor = getSurfaceRayColor(vPosition, reflectedRay, underwaterColor);
  vec3 refractedColor = getSurfaceRayColor(vPosition, refractedRay, vec3(1.0)) * vec3(0.8, 1.0, 1.1);

  outColor = vec4(mix(reflectedColor, refractedColor, (1.0 - fresnel) * length(refractedRay)), 1.0);
}
