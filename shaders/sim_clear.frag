// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#version 450
// S0 clear: reset the heightfield to a flat, still surface.
// R=height, G=velocity, B=normal.x, A=normal.z -> flat = (0,0,0,1) reconstructed
// normal points up. Stored normal.xz = (0,0); normal.y is reconstructed in shading.
layout(location = 0) out vec4 outColor;
void main() { outColor = vec4(0.0, 0.0, 0.0, 0.0); }
