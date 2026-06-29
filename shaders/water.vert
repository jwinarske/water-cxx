// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#version 450
// Water surface vertex shader (surface.glsl waterVertex). Displaces the
// flat XY grid into an XZ water plane (aPosition.xzy) by the simulated height — a
// vertex-stage texture fetch of the heightfield (explicit LOD 0; no mips). World space, so
// the only transform is matrix_viewProjection.
#include "surface_push.glsl"

layout(set = 0, binding = 0) uniform sampler2D water;
layout(location = 0) in vec3 aPosition;
layout(location = 0) out vec3 vPosition;

void main() {
  vec4 info = textureLod(water, aPosition.xy * 0.5 + 0.5, 0.0);
  vec3 position = aPosition.xzy;
  position.y += info.r;
  vPosition = position;
  gl_Position = pc.vp * vec4(position, 1.0);
}
