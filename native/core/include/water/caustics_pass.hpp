// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// water/caustics_pass.hpp — the caustics pass: rasterize the projected
// water mesh from the light's POV into a 1024² RGBA8 target, double-buffered so
// the scene samples the previous frame (one-frame-stale; avoids the same-frame
// read/write hazard). Depth off, blend off, cull none, opaque last-write (the
// differential-area ratio encodes focusing — no additive accumulation).
#pragma once

#include "water/device.hpp"
#include "water/mesh.hpp"
#include "water/result.hpp"
#include "water/scene_pass.hpp"  // ScenePush (light + sphere; vp/eye unused here)

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace water {

class CausticsPass {
 public:
  [[nodiscard]] static Result<CausticsPass> create(const Device& dev,
                                                   uint32_t size = 1024,
                                                   uint32_t buffers = 2);
  ~CausticsPass();
  CausticsPass(CausticsPass&&) noexcept;
  CausticsPass& operator=(CausticsPass&&) noexcept;
  CausticsPass(const CausticsPass&) = delete;
  CausticsPass& operator=(const CausticsPass&) = delete;

  // A descriptor set binding the heightfield as the caustics VS's `water`
  // input.
  [[nodiscard]] Result<VkDescriptorSet> make_water_set(VkImageView water,
                                                       VkSampler sampler);

  // Render caustics into buffer `index` (the projected water mesh). Leaves it
  // SHADER_READ.
  void render(VkCommandBuffer cmd,
              uint32_t index,
              const ScenePush& push,
              const Mesh& plane,
              VkDescriptorSet water_set);

  [[nodiscard]] const Image& image(uint32_t index) const noexcept {
    return images_[index];
  }
  [[nodiscard]] uint32_t buffers() const noexcept {
    return uint32_t(images_.size());
  }
  [[nodiscard]] uint32_t size() const noexcept { return size_; }

 private:
  CausticsPass() = default;
  void destroy() noexcept;

  const Device* dev_ = nullptr;
  uint32_t size_ = 0;
  std::vector<Image> images_;
  std::vector<VkFramebuffer> fb_;
  VkRenderPass render_pass_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsl_ = VK_NULL_HANDLE;
  VkDescriptorPool pool_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
};

}  // namespace water
