// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#version 450
// Water surface seen from above (surface.glsl waterAboveFragment):
// Fresnel mix of reflection and refraction, both ray-traced into the scene.
#include "water_common.glsl"

layout(location = 0) out vec4 outColor;

void main() {
  vec3 normal, incomingRay;
  computeNormalAndRay(normal, incomingRay);

  vec3 reflectedRay = reflect(incomingRay, normal);
  vec3 refractedRay = refract(incomingRay, normal, IOR_AIR / IOR_WATER);
  float fresnel = mix(0.25, 1.0, pow(1.0 - dot(normal, -incomingRay), 3.0));

  vec3 reflectedColor = getSurfaceRayColor(vPosition, reflectedRay, abovewaterColor);
  vec3 refractedColor = getSurfaceRayColor(vPosition, refractedRay, abovewaterColor);

  outColor = vec4(mix(refractedColor, reflectedColor, fresnel), 1.0);
}
