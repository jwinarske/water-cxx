// SPDX-FileCopyrightText: (c) 2026 Joel Winarske
// SPDX-License-Identifier: MIT
// water/heightfield_sim.hpp — the ping-pong heightfield water simulation (ports
// water.js to HeightfieldSim, passes S0-S4).
//
// Texel layout: R=height, G=velocity, B=normal.x, A=normal.z. Two RGBA{16,32}F
// images ping-pong; each pass reads the source and writes the destination, then
// the indices swap. All sim state lives here and is driven from a single thread
// (single-threaded) — the renderer owns it, input is drained at frame start.
//
// Methods record into a caller-supplied command buffer so many steps batch into
// one submit (the invariant test runs 10^5 steps in a handful of submits).
#pragma once

#include "water/device.hpp"
#include "water/fullscreen_pass.hpp"
#include "water/result.hpp"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace water {

class HeightfieldSim {
 public:
  // `size` is the grid resolution (256 upstream; 128 mobile per ). Heightfield
  // format comes from the device caps.
  [[nodiscard]] static Result<HeightfieldSim> create(const Device& dev,
                                                     uint32_t size = 256);
  ~HeightfieldSim();
  HeightfieldSim(HeightfieldSim&&) noexcept;
  HeightfieldSim& operator=(HeightfieldSim&&) noexcept;
  HeightfieldSim(const HeightfieldSim&) = delete;
  HeightfieldSim& operator=(const HeightfieldSim&) = delete;

  // --- recorded operations ---
  void clear(
      VkCommandBuffer cmd);  // S0: reset BOTH ping-pong buffers to flat/still
  void add_drop(VkCommandBuffer cmd,
                glm::vec2 center,
                float radius,
                float strength);  // S1 point
  void add_line(VkCommandBuffer cmd,
                glm::vec2 a,
                glm::vec2 b,
                float radius,
                float strength);  // S1
  void move_sphere(VkCommandBuffer cmd,
                   glm::vec3 old_c,
                   glm::vec3 new_c,
                   float radius);            // S2
  void update(VkCommandBuffer cmd);          // S3 (call 2x/frame)
  void update_normals(VkCommandBuffer cmd);  // S4 (1x/frame after updates)

  // Per-frame core: 2x update + 1x normal (the main.js update loop minus
  // input/physics).
  void step(VkCommandBuffer cmd);

  [[nodiscard]] uint32_t size() const noexcept { return size_; }
  [[nodiscard]] VkFormat format() const noexcept { return format_; }
  // The current (latest-written) heightfield, for surface sampling (/) or
  // readback.
  [[nodiscard]] const Image& current() const noexcept {
    return img_[read_idx_];
  }
  // The two ping-pong images (0,1). Lets a caller bake one descriptor set per
  // image and pick the one matching current() each frame, freeing the per-frame
  // swap count from any parity constraint.
  [[nodiscard]] const Image& image(uint32_t i) const noexcept {
    return img_[i];
  }

  // Copy the current heightfield to host memory as RGBA float32 (converts the
  // 16F path). Leaves the image back in SHADER_READ_ONLY_OPTIMAL. Test/debug
  // helper (NaN trap).
  [[nodiscard]] Result<std::vector<glm::vec4>> readback(
      const Device& dev) const;

 private:
  HeightfieldSim() = default;
  void destroy() noexcept;
  void run_pass(VkCommandBuffer cmd,
                const FullscreenPass& pass,
                std::span<const std::byte> push,
                bool use_source);

  const Device* dev_ = nullptr;
  uint32_t size_ = 0;
  VkFormat format_ = VK_FORMAT_UNDEFINED;
  int read_idx_ = 0;

  std::array<Image, 2> img_{};
  std::array<VkFramebuffer, 2> fb_{VK_NULL_HANDLE, VK_NULL_HANDLE};
  VkSampler sampler_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsl_ = VK_NULL_HANDLE;
  VkDescriptorPool pool_ = VK_NULL_HANDLE;
  std::array<VkDescriptorSet, 2> set_{VK_NULL_HANDLE, VK_NULL_HANDLE};

  FullscreenPass clear_, drop_, sphere_, update_, normal_;
};

}  // namespace water
