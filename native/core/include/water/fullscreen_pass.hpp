// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// water/fullscreen_pass.hpp — the full-screen-quad pass helper.
//
// One render pass + one graphics pipeline that draws the 3-vertex clip-space
// triangle (no vertex input) with a supplied fragment shader. Dynamic viewport/
// scissor so a single pass serves any target size (256² sim RTs, WxH scene).
// This is the reusable primitive the 5 sim passes + caustics are built on in /.
#pragma once

#include "water/device.hpp"
#include "water/result.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace water {

struct FullscreenPassDesc {
  VkFormat color_format = VK_FORMAT_UNDEFINED;
  VkAttachmentLoadOp load_op = VK_ATTACHMENT_LOAD_OP_CLEAR;
  VkImageLayout final_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  std::span<const uint32_t> vert_spirv;
  std::span<const uint32_t> frag_spirv;
  uint32_t push_constant_size = 0;  // 0 = none
  VkDescriptorSetLayout set_layout =
      VK_NULL_HANDLE;  // VK_NULL_HANDLE = no descriptors
  VkShaderStageFlags push_stages = VK_SHADER_STAGE_FRAGMENT_BIT;
};

class FullscreenPass {
 public:
  // Null pass (so it can be a member that is later move-assigned, e.g. in
  // HeightfieldSim).
  FullscreenPass() = default;
  [[nodiscard]] static Result<FullscreenPass> create(
      const Device& dev,
      const FullscreenPassDesc& desc);
  ~FullscreenPass();
  FullscreenPass(FullscreenPass&&) noexcept;
  FullscreenPass& operator=(FullscreenPass&&) noexcept;
  FullscreenPass(const FullscreenPass&) = delete;
  FullscreenPass& operator=(const FullscreenPass&) = delete;

  [[nodiscard]] VkRenderPass render_pass() const noexcept {
    return render_pass_;
  }

  // A framebuffer binding `view` (caller owns it; destroy with
  // destroy_framebuffer).
  [[nodiscard]] Result<VkFramebuffer> make_framebuffer(VkImageView view,
                                                       uint32_t w,
                                                       uint32_t h) const;
  void destroy_framebuffer(VkFramebuffer fb) const noexcept;

  // Record one full-screen draw into `fb` at `extent`. `clear` is used iff the
  // pass was created with LOAD_OP_CLEAR. `push`/`set` are optional (match the
  // desc).
  void record(VkCommandBuffer cmd,
              VkFramebuffer fb,
              VkExtent2D extent,
              std::array<float, 4> clear = {0, 0, 0, 1},
              std::span<const std::byte> push = {},
              VkDescriptorSet set = VK_NULL_HANDLE) const;

 private:
  void destroy() noexcept;

  VkDevice device_ = VK_NULL_HANDLE;
  VkRenderPass render_pass_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  VkShaderStageFlags push_stages_ = 0;
  uint32_t push_size_ = 0;
};

}  // namespace water
