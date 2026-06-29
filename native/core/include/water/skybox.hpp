// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// water/skybox.hpp — renders the sky cubemap as the scene background. A
// full-screen triangle into ScenePass's render pass, drawn before the geometry
// (depth = far). Not part of the final demo (which clears to black) — it
// validates the cubemap load + orientation that 's water reflections rely on,
// and is a useful debug background.
#pragma once

#include "water/device.hpp"
#include "water/result.hpp"
#include "water/texture.hpp"

#include <glm/glm.hpp>

namespace water {

class Skybox {
 public:
  // Borrows `sky` + `sampler` (must outlive the Skybox); builds a pipeline
  // against `render_pass` (e.g. ScenePass::render_pass()).
  [[nodiscard]] static Result<Skybox> create(const Device& dev,
                                             VkRenderPass render_pass,
                                             const Texture& sky,
                                             VkSampler sampler);
  ~Skybox();
  Skybox(Skybox&&) noexcept;
  Skybox& operator=(Skybox&&) noexcept;
  Skybox(const Skybox&) = delete;
  Skybox& operator=(const Skybox&) = delete;

  // Record the background draw inside an active render pass (viewport already
  // set).
  void record(VkCommandBuffer cmd,
              const glm::mat4& inv_vp,
              const glm::vec3& eye) const;

 private:
  Skybox() = default;
  void destroy() noexcept;

  const Device* dev_ = nullptr;
  VkDescriptorSetLayout dsl_ = VK_NULL_HANDLE;
  VkDescriptorPool pool_ = VK_NULL_HANDLE;
  VkDescriptorSet set_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
};

}  // namespace water
