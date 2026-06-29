// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// Shared scene-shader helpers, ported verbatim from Evan Wallace's WebGL Water
// (renderer.js helperFunctions / common.glsl). Prepended to FRAGMENT shaders only
// (getWallColor/getSphereColor use textureLod). The GL uniforms become: light/
// sphereCenter/sphereRadius -> push constants (surface_push.glsl); water/tiles/
// causticTex -> combined image samplers in set 0. texture2DLod -> textureLod.
#include "surface_push.glsl"
#include "intersect.glsl"

layout(set = 0, binding = 0) uniform sampler2D water;       // R height, G vel, BA normal.xz
layout(set = 0, binding = 1) uniform sampler2D tiles;       // pool tiles (repeat, mip)
layout(set = 0, binding = 2) uniform sampler2D causticTex;  // R intensity, G shadow
// binding 3 (samplerCube sky) is declared by the water-surface fragments.

vec3 getSphereColor(vec3 point) {
  vec3 color = vec3(0.5);

  // ambient occlusion with walls
  color *= 1.0 - 0.9 / pow((1.0 + pc.sphereInfo.w - abs(point.x)) / pc.sphereInfo.w, 3.0);
  color *= 1.0 - 0.9 / pow((1.0 + pc.sphereInfo.w - abs(point.z)) / pc.sphereInfo.w, 3.0);
  color *= 1.0 - 0.9 / pow((point.y + 1.0 + pc.sphereInfo.w) / pc.sphereInfo.w, 3.0);

  // caustics
  vec3 sphereNormal = (point - pc.sphereInfo.xyz) / pc.sphereInfo.w;
  vec3 refractedLight = refract(-pc.lightDir.xyz, vec3(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
  float diffuse = max(0.0, dot(-refractedLight, sphereNormal)) * 0.5;
  vec4 info = textureLod(water, point.xz * 0.5 + 0.5, 0.0);
  if (point.y < info.r) {
    vec4 caustic = textureLod(
        causticTex, 0.75 * (point.xz - point.y * refractedLight.xz / refractedLight.y) * 0.5 + 0.5,
        0.0);
    diffuse *= caustic.r * 4.0;
  }
  color += diffuse;
  return color;
}

vec3 getWallColor(vec3 point) {
  float scale = 0.5;

  vec3 wallColor;
  vec3 normal;
  if (abs(point.x) > 0.999) {
    wallColor = textureLod(tiles, point.yz * 0.5 + vec2(1.0, 0.5), 0.0).rgb;
    normal = vec3(-point.x, 0.0, 0.0);
  } else if (abs(point.z) > 0.999) {
    wallColor = textureLod(tiles, point.yx * 0.5 + vec2(1.0, 0.5), 0.0).rgb;
    normal = vec3(0.0, 0.0, -point.z);
  } else {
    wallColor = textureLod(tiles, point.xz * 0.5 + 0.5, 0.0).rgb;
    normal = vec3(0.0, 1.0, 0.0);
  }

  scale /= length(point);                                                        // pool AO
  scale *= 1.0 - 0.9 / pow(length(point - pc.sphereInfo.xyz) / pc.sphereInfo.w, 4.0);  // sphere AO

  // caustics
  vec3 refractedLight = -refract(-pc.lightDir.xyz, vec3(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
  float diffuse = max(0.0, dot(refractedLight, normal));
  vec4 info = textureLod(water, point.xz * 0.5 + 0.5, 0.0);
  if (point.y < info.r) {
    vec4 caustic = textureLod(
        causticTex, 0.75 * (point.xz - point.y * refractedLight.xz / refractedLight.y) * 0.5 + 0.5,
        0.0);
    scale += diffuse * caustic.r * 2.0 * caustic.g;
  } else {
    // shadow for the rim of the pool
    vec2 t = intersectCube(point, refractedLight, vec3(-1.0, -poolHeight, -1.0), vec3(1.0, 2.0, 1.0));
    diffuse *= 1.0 / (1.0 + exp(-200.0 / (1.0 + 10.0 * (t.y - t.x)) *
                                (point.y + refractedLight.y * t.y - 2.0 / 12.0)));
    scale += diffuse * 0.5;
  }
  return wallColor * scale;
}
