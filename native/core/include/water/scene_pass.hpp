// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// water/scene_pass.hpp — the scene render pass: RGBA8 color + depth,
// a world-space mesh pipeline (vec3 aPosition, no model matrix), depth
// test+write, and the Vulkan Y-flip applied once via a negative-height
// viewport. Pool/sphere and water above/below are pipelines created against
// this pass.
#pragma once

#include "water/device.hpp"
#include "water/mesh.hpp"
#include "water/result.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace water {

// Surface push set (112 B <= 128). One layout for every scene shader.
struct ScenePush {
  glm::mat4 vp;      // matrix_viewProjection
  glm::vec4 eye;     // xyz
  glm::vec4 light;   // xyz (normalized)
  glm::vec4 sphere;  // xyz = center, w = radius
};
static_assert(sizeof(ScenePush) == 112);

class ScenePass {
 public:
  [[nodiscard]] static Result<ScenePass> create(
      const Device& dev,
      uint32_t width,
      uint32_t height,
      VkFormat color = VK_FORMAT_R8G8B8A8_UNORM);
  ~ScenePass();
  ScenePass(ScenePass&&) noexcept;
  ScenePass& operator=(ScenePass&&) noexcept;
  ScenePass(const ScenePass&) = delete;
  ScenePass& operator=(const ScenePass&) = delete;

  struct PipelineDesc {
    std::span<const uint32_t> vert;
    std::span<const uint32_t> frag;
    VkCullModeFlags cull = VK_CULL_MODE_NONE;
    bool depth_test = true;
    bool depth_write = true;
  };
  // Pipeline is owned by the pass (destroyed with it); the returned handle is
  // for draw().
  [[nodiscard]] Result<VkPipeline> make_pipeline(const PipelineDesc& desc);

  // The surface descriptor set: set 0, bindings 0=water 1=tiles 2=causticTex
  // 3=sky, all combined image samplers. Every scene pipeline's layout includes
  // it.
  struct Binding {
    VkImageView view;
    VkSampler sampler;
  };
  // Allocate + write one surface set (owned by the pass). Order: water, tiles,
  // caustic, sky.
  [[nodiscard]] Result<VkDescriptorSet> make_surface_set(
      const std::array<Binding, 4>& bindings);

  void begin(VkCommandBuffer cmd, glm::vec4 clear_color);
  // `set` (optional) binds the surface descriptors; pass VK_NULL_HANDLE for
  // shaders that use no textures (the / debug fragment).
  void draw(VkCommandBuffer cmd,
            VkPipeline pipeline,
            const ScenePush& push,
            const Mesh& mesh,
            VkDescriptorSet set = VK_NULL_HANDLE);
  void end(VkCommandBuffer cmd);

  [[nodiscard]] const Image& color() const noexcept { return color_; }
  [[nodiscard]] VkRenderPass render_pass() const noexcept {
    return render_pass_;
  }
  [[nodiscard]] uint32_t width() const noexcept { return width_; }
  [[nodiscard]] uint32_t height() const noexcept { return height_; }
  [[nodiscard]] float aspect() const noexcept {
    return float(width_) / float(height_);
  }

  // Copy the color target to host memory as tightly-packed RGBA8 (row 0 = top).
  [[nodiscard]] Result<std::vector<uint8_t>> readback_color(
      const Device& dev) const;

 private:
  ScenePass() = default;
  void destroy() noexcept;

  const Device* dev_ = nullptr;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  Image color_{};
  Image depth_{};
  VkRenderPass render_pass_ = VK_NULL_HANDLE;
  VkFramebuffer fb_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout surface_dsl_ = VK_NULL_HANDLE;
  VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  std::vector<VkPipeline> pipelines_;
};

}  // namespace water
