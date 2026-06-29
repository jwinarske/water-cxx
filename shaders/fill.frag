// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
#version 450
//  smoke shader: a deterministic UV gradient, so the golden readback verifies the
// device + format + FS-quad pass + offscreen target + readback chain end to end.
// outColor = (u, v, 0.25, 1.0). Replaced by the real surface shaders from .
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push { vec2 extent; } pc;

void main() {
  vec2 uv = gl_FragCoord.xy / pc.extent;  // pixel centers -> ((x+0.5)/W, (y+0.5)/H)
  outColor = vec4(uv, 0.25, 1.0);
}
