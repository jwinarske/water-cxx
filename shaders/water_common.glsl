// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// Shared water-surface fragment helpers (surface.glsl surfaceRayColor +
// computeNormalAndRay). Adds the sky cubemap (set 0 binding 3) on top of surface_common's
// water/tiles/causticTex + getWallColor/getSphereColor. Prepended to the above/below frags.
#include "surface_common.glsl"

layout(set = 0, binding = 3) uniform samplerCube sky;
layout(location = 0) in vec3 vPosition;  // world-space surface point ("position" upstream)

// Trace a ray from the water surface into the scene: sphere, pool walls, or sky.
vec3 getSurfaceRayColor(vec3 origin, vec3 ray, vec3 waterColor) {
  vec3 color;
  float q = intersectSphere(origin, ray, pc.sphereInfo.xyz, pc.sphereInfo.w);
  if (q < 1.0e6) {
    color = getSphereColor(origin + ray * q);
  } else if (ray.y < 0.0) {
    vec2 t = intersectCube(origin, ray, vec3(-1.0, -poolHeight, -1.0), vec3(1.0, 2.0, 1.0));
    color = getWallColor(origin + ray * t.y);
  } else {
    vec2 t = intersectCube(origin, ray, vec3(-1.0, -poolHeight, -1.0), vec3(1.0, 2.0, 1.0));
    vec3 hit = origin + ray * t.y;
    if (hit.y < 2.0 / 12.0) {
      color = getWallColor(hit);
    } else {
      color = textureLod(sky, ray, 0.0).rgb;
      color += vec3(pow(max(0.0, dot(pc.lightDir.xyz, ray)), 5000.0)) * vec3(10.0, 8.0, 6.0);
    }
  }
  if (ray.y < 0.0) color *= waterColor;
  return color;
}

// Surface normal (peaked-water refinement) + incoming view ray.
void computeNormalAndRay(out vec3 normal, out vec3 incomingRay) {
  vec2 coord = vPosition.xz * 0.5 + 0.5;
  vec4 info = textureLod(water, coord, 0.0);

  // make water look more "peaked": walk along the stored slope a few steps
  for (int i = 0; i < 5; i++) {
    coord += info.ba * 0.005;
    info = textureLod(water, coord, 0.0);
  }

  // / NaN guard: clamp the radicand so the 16F path can't yield a NaN normal.
  normal = vec3(info.b, sqrt(max(0.0, 1.0 - dot(info.ba, info.ba))), info.a);
  incomingRay = normalize(vPosition - pc.eyePos.xyz);
}
