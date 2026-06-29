// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#version 450
// Caustics vertex shader (surface.glsl causticsVertex). Projects each
// water-surface vertex along its refracted light ray onto the pool floor, off the flat
// surface (oldPos) and off the displaced surface (newPos). The fragment takes dpdx/dpdy of
// these to measure how the projected area changed (the differential-area method).
//
// NDC is built DIRECTLY here (bypassing matrix_viewProjection). : the caustics pass uses
// its own positive-height viewport (NOT the scene's negative-height flip), and generation
// NDC (x = newPos.x, y = newPos.z) matches the scene's caustic sampling UV
// (0.75*(point.xz - ...)) — so no Y-negate, and no double-negate.
#include "surface_push.glsl"
#include "intersect.glsl"

layout(set = 0, binding = 0) uniform sampler2D water;
layout(location = 0) in vec3 aPosition;
layout(location = 0) out vec3 oldPos;
layout(location = 1) out vec3 newPos;

// Refract along `ray`, intersect the pool box, then drop to the floor plane (y = -1).
vec3 project(vec3 origin, vec3 ray, vec3 refractedLight) {
  vec2 tcube = intersectCube(origin, ray, vec3(-1.0, -poolHeight, -1.0), vec3(1.0, 2.0, 1.0));
  origin += ray * tcube.y;
  float tplane = (-origin.y - 1.0) / refractedLight.y;
  return origin + refractedLight * tplane;
}

void main() {
  vec4 info = textureLod(water, aPosition.xy * 0.5 + 0.5, 0.0);
  info.ba *= 0.5;
  vec3 normal = vec3(info.b, sqrt(max(0.0, 1.0 - dot(info.ba, info.ba))), info.a);  // NaN guard

  vec3 refractedLight = refract(-pc.lightDir.xyz, vec3(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
  vec3 ray = refract(-pc.lightDir.xyz, normal, IOR_AIR / IOR_WATER);
  oldPos = project(aPosition.xzy, refractedLight, refractedLight);
  newPos = project(aPosition.xzy + vec3(0.0, info.r, 0.0), ray, refractedLight);

  gl_Position = vec4(0.75 * (newPos.xz + refractedLight.xz / refractedLight.y), 0.0, 1.0);
}
