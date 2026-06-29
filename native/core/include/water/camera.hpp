// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// water/camera.hpp — orbit camera (main.js applyCamera). Produces the
// matrix_viewProjection pushed to every scene shader. Authored in GL/glm
// convention (Y-up, right-handed); the Vulkan Y-down framebuffer is handled
// once by the scene pass's negative-height viewport, and Z is mapped to [0,1]
// by GLM_FORCE_DEPTH_ZERO_TO_ONE (set on water_core). No model matrix —
// geometry is world space already.
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace water {

struct OrbitCamera {
  glm::vec3 target{0.0f, -0.5f, 0.0f};
  float distance = 4.0f;
  float angle_x_deg = -25.0f;   // pitch
  float angle_y_deg = -200.5f;  // yaw
  float fov_deg = 45.0f;        // vertical
  float near_clip = 0.01f;
  float far_clip = 100.0f;

  [[nodiscard]] glm::vec3 position() const {
    // PlayCanvas: q = qy * qx; offset = q * (0,0,distance); pos = target +
    // offset.
    const glm::quat qx =
        glm::angleAxis(glm::radians(angle_x_deg), glm::vec3(1, 0, 0));
    const glm::quat qy =
        glm::angleAxis(glm::radians(angle_y_deg), glm::vec3(0, 1, 0));
    return target + (qy * qx) * glm::vec3(0.0f, 0.0f, distance);
  }

  [[nodiscard]] glm::mat4 view_projection(float aspect) const {
    const glm::mat4 view = glm::lookAt(position(), target, glm::vec3(0, 1, 0));
    const glm::mat4 proj =
        glm::perspective(glm::radians(fov_deg), aspect, near_clip, far_clip);
    return proj * view;
  }

  // Clamp like the demo (main.js): pitch ±89.999, distance [1.5, 6].
  void clamp() {
    angle_x_deg = glm::clamp(angle_x_deg, -89.999f, 89.999f);
    distance = glm::clamp(distance, 1.5f, 6.0f);
  }
};

}  // namespace water
