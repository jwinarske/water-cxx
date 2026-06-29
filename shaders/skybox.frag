// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#version 450
// Sample the sky cubemap along the interpolated world-space view ray.
layout(location = 0) in vec3 vDir;
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform samplerCube sky;

void main() {
  outColor = vec4(texture(sky, normalize(vDir)).rgb, 1.0);
}
